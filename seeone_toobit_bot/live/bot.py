"""See One -> Toobit live perpetual bot.

Run:  python3 -m live.bot            (live loop)
      python3 -m live.bot --once --dry-run   (print one cycle of decisions, place nothing)
Pause new entries: touch logs/PAUSE   (open trades keep being managed)
"""
import argparse
import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from . import strategy as S
from .learner import Learner
from .shadow import Shadow
from .toobit import Toobit, ToobitError

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs" / "trades"
TRADE_LOG = LOG_DIR / "live_trades.jsonl"
STATE_FILE = ROOT / "logs" / "live_state.json"
PAUSE_FILE = ROOT / "logs" / "PAUSE"
LEARN_FILE = ROOT / "logs" / "learned_params.json"
INSIGHT_LOG = LOG_DIR / "insights.jsonl"
SHADOW_FILE = ROOT / "logs" / "shadow_archive.json"
LOOP_SECONDS = 30
DAILY_LOSS_PCT = 0.03
FEE_BUFFER = 0.0012


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log_event(event: str, **data):
    rec = {"ts": now_iso(), "event": event, **data}
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with TRADE_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    print(json.dumps(rec, ensure_ascii=False, default=str), flush=True)


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"trades": {}, "closed": [], "traded_ids": [], "skips": {}, "day": None, "day_start_equity": None}


def save_state(st: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, ensure_ascii=False, indent=1, default=str))
    tmp.replace(STATE_FILE)


class Bot:
    def __init__(self, dry_run: bool):
        global TRADE_LOG
        load_dotenv(ROOT / ".env")
        self.dry = dry_run
        if dry_run:
            TRADE_LOG = LOG_DIR / "dry_run.jsonl"
        self.tb = Toobit(os.environ["TOOBIT_API_KEY"], os.environ["TOOBIT_API_SECRET"])
        self.so = S.SeeOne(os.environ["SEEONE_RT_SESSION"])
        self.st = load_state()
        self.learner = Learner(LEARN_FILE, INSIGHT_LOG, log_event)
        self.shadow = Shadow(SHADOW_FILE, self.learner, log_event)

    # ------------------------------------------------------------------ helpers
    def equity(self) -> tuple[float, float]:
        b = self.tb.balance()
        eq = float(b["balance"]) + float(b.get("crossUnRealizedPnl") or 0) + float(b.get("isolatedUnRealizedPnl") or 0)
        return eq, float(b["availableBalance"])

    def cancel_quiet(self, order_id, kind="STOP"):
        try:
            self.tb.cancel_order(order_id, kind)
        except ToobitError:
            pass

    def replace_stop(self, tr: dict, new_stop: float, reason: str):
        sym, side = tr["tsym"], tr["side"]
        px = self.tb.fmt_price(sym, new_stop)
        self.tb.set_position_stop(sym, side, px)
        stale = []
        for o in self.tb.open_orders(sym, "STOP_PROFIT_LOSS"):
            if o.get("orderType") == f"STOP_{side}_LOSS" and float(o["stopPrice"]) != float(px):
                self.cancel_quiet(o["orderId"], "STOP_PROFIT_LOSS")
                stale.append(o["orderId"])
        log_event("STOP_MOVE", trade_id=tr["id"], symbol=sym, side=side, old_stop=tr["stop"], new_stop=px,
                  reason=reason, cancelled_old_sl=stale)
        tr["stop"] = float(px)

    # ------------------------------------------------------------------ management
    def manage(self, positions: dict, sigmap: dict):
        for tid, tr in list(self.st["trades"].items()):
            sym, side = tr["tsym"], tr["side"]
            pos = positions.get((sym, side))
            if pos is None:
                self.finish(tid, tr)
                continue

            open_ids = {str(o["orderId"]) for o in self.tb.open_orders(sym, "STOP")}
            newly = [tp for tp in tr["tps"] if not tp["filled"] and tp.get("order_id") and str(tp["order_id"]) not in open_ids]
            for tp in newly:
                tp["filled"] = True
                tp["filled_at"] = now_iso()
                log_event("TP_HIT", trade_id=tid, symbol=sym, side=side, label=tp["label"], price=tp["price"],
                          qty=tp["qty"], reason=f"{tp['label']} ({tp['r']}R) 觸發，部分平倉 {tp['qty']} 張")
            if newly:
                filled = [i for i, tp in enumerate(tr["tps"]) if tp["filled"]]
                k = max(filled)
                if k == 0:
                    target = tr["entry"] * (1 + FEE_BUFFER if side == "LONG" else 1 - FEE_BUFFER)
                    why = "TP1 達成，止損移至保本(含手續費)"
                else:
                    target = tr["tps"][k - 1]["price"]
                    why = f"{tr['tps'][k]['label']} 達成，止損移至 {tr['tps'][k - 1]['label']}"
                better = target > tr["stop"] if side == "LONG" else target < tr["stop"]
                if better and not all(tp["filled"] for tp in tr["tps"]):
                    try:
                        self.replace_stop(tr, target, why)
                    except ToobitError as e:
                        log_event("ERROR", trade_id=tid, where="replace_stop", error=str(e))

            sig = sigmap.get(tr["signal_id"])
            if sig and sig.get("status") == "CLOSED" and not tr.get("exit_reason"):
                tr["exit_reason"] = f"See One 訊號結束: {sig.get('exit_reason') or 'CLOSED'}"
                log_event("SIGNAL_CLOSED", trade_id=tid, symbol=sym, side=side, seeone_exit=sig.get("exit_reason"),
                          seeone_realized_r=sig.get("realized_r"), action="市價平倉剩餘部位")
                try:
                    self.tb.flash_close(sym, side)
                except ToobitError as e:
                    log_event("ERROR", trade_id=tid, where="flash_close", error=str(e))
            elif not sig and tr.get("source") == "官方訊號" and time.time() - tr["opened_ts"] > 48 * 3600:
                tr["exit_reason"] = "超過 48 小時且已不在 See One 訊號列表，時間止損"
                try:
                    self.tb.flash_close(sym, side)
                except ToobitError as e:
                    log_event("ERROR", trade_id=tid, where="flash_close", error=str(e))

    def finish(self, tid: str, tr: dict):
        sym, side = tr["tsym"], tr["side"]
        for tp in tr["tps"]:
            if not tp["filled"] and tp.get("order_id"):
                self.cancel_quiet(tp["order_id"])
        pnl, close_px = None, None
        try:
            for h in self.tb.history_positions(sym, 10):
                if h.get("side") == side and int(h.get("closeTime") or 0) >= tr["opened_ts"] * 1000 - 60000:
                    pnl = (pnl or 0) + float(h.get("realizedPnL") or h.get("realizedPnl") or 0)
                    close_px = h.get("closeAvgPrice")
        except ToobitError as e:
            log_event("ERROR", trade_id=tid, where="history_positions", error=str(e))
        filled = [tp["label"] for tp in tr["tps"] if tp["filled"]]
        if tr.get("exit_reason"):
            reason = tr["exit_reason"]
        elif len(filled) == len(tr["tps"]):
            reason = f"全部止盈完成 ({', '.join(filled)})"
        elif filled:
            reason = f"已達 {filled[-1]} 後，移動止損 {tr['stop']} 觸發"
        else:
            reason = f"初始止損 {tr['stop']} 觸發"
        r_result = round(pnl / tr["risk_usdt"], 3) if pnl is not None and tr["risk_usdt"] else None
        log_event("EXIT", trade_id=tid, symbol=sym, side=side, reason=reason, tps_filled=filled,
                  entry=tr["entry"], close_avg=close_px, realized_pnl_usdt=pnl, r_result=r_result,
                  path=tr.get("path"), held_min=round((time.time() - tr["opened_ts"]) / 60, 1))
        self.st["closed"].append({"id": tid, "symbol": sym, "side": side, "path": tr.get("path"),
                                  "pnl": pnl, "r_result": r_result, "closed_at": now_iso(), "reason": reason})
        del self.st["trades"][tid]
        save_state(self.st)
        self.learner.on_close(tr, r_result, pnl, reason, filled, round((time.time() - tr["opened_ts"]) / 60, 1))

    # ------------------------------------------------------------------ entries
    def open_trade(self, sig: dict, ev: dict, source: str, regime: tuple, equity: float) -> bool:
        sym, side = S.to_toobit(sig["symbol"]), sig["side"]
        spec = self.tb.specs()[sym]
        price = self.tb.price(sym)
        stop = float(sig["stop_price"])
        if (side == "LONG" and price <= stop * 1.001) or (side == "SHORT" and price >= stop * 0.999):
            self.note_skip(sig, f"現價 {price} 已貼近/穿越止損 {stop}")
            return False
        sz = S.size_position(equity, price, stop, spec, self.learner.p["risk_pct"])
        if not sz:
            self.note_skip(sig, "以 1% 風險無法達到交易所最小下單量 (或保證金過大)")
            return False
        used = sum(t["margin"] for t in self.st["trades"].values())
        if used + sz["margin"] > equity * S.MAX_TOTAL_MARGIN_PCT:
            self.note_skip(sig, f"總保證金將超過 {S.MAX_TOTAL_MARGIN_PCT:.0%} 權益")
            return False
        plan = S.tp_plan(sig, sz["contracts"], spec["min_contracts"], self.learner.p.get("tp_split"))
        entry_info = dict(signal_id=sig["id"], source=source, symbol=sym, side=side, price=price,
                          stop=stop, sizing=sz, tp_plan=plan, score=ev["score"], regime=regime,
                          path=sig.get("signal_path"), subtype=sig.get("path_subtype"),
                          timeframe=sig.get("timeframe"), reasons=ev["reasons"], factors=ev["factors"],
                          learned_min_score=self.learner.p["min_score"], risk_pct=self.learner.p["risk_pct"])
        if self.dry:
            log_event("DRY_ENTRY", **entry_info)
            return True

        self.tb.set_isolated(sym)
        self.tb.set_leverage(sym, sz["leverage"])
        order = self.tb.market_open(sym, side, sz["contracts"], self.tb.fmt_price(sym, stop))
        time.sleep(1.5)
        pos = next((p for p in self.tb.positions(sym) if p.get("side") == side), None)
        if not pos:
            log_event("ERROR", where="open", symbol=sym, order=order, error="下單後查無持倉")
            self.st["traded_ids"].append(sig["id"])
            return False
        entry = float(pos.get("avgPrice") or price)
        tps = []
        for tp in plan:
            beyond = tp["price"] <= entry if side == "LONG" else tp["price"] >= entry
            if beyond:
                continue
            try:
                o = self.tb.close_trigger(sym, side, tp["qty"], self.tb.fmt_price(sym, tp["price"]), tp["label"].lower())
                tps.append({**tp, "order_id": str(o.get("orderId")), "filled": False})
            except ToobitError as e:
                log_event("ERROR", where="tp_order", symbol=sym, label=tp["label"], error=str(e))
                tps.append({**tp, "order_id": None, "filled": False})
        tid = f"{sym}:{side}:{int(time.time())}"
        mult = float(spec["multiplier"])
        self.st["trades"][tid] = {
            "id": tid, "signal_id": sig["id"], "source": source, "tsym": sym, "side": side,
            "contracts": sz["contracts"], "entry": entry, "stop": stop, "tps": tps,
            "risk_usdt": round(sz["contracts"] * mult * abs(entry - stop), 4), "margin": sz["margin"],
            "leverage": sz["leverage"], "path": sig.get("signal_path"), "opened_ts": time.time(),
            "factors": ev["factors"], "entry_cur_r": ev["cur_r"], "score": ev["score"], "regime": list(regime),
        }
        self.st["traded_ids"].append(sig["id"])
        log_event("ENTRY", trade_id=tid, entry_avg=entry, order_id=order.get("orderId"),
                  tp_orders=[(t["label"], t["price"], t["qty"], t["order_id"]) for t in tps], **entry_info)
        return True

    def note_skip(self, sig: dict, why: str):
        key = sig["id"]
        if self.st["skips"].get(key) != why:
            self.st["skips"][key] = why
            log_event("SKIP", signal_id=key, symbol=sig["symbol"], side=sig["side"], reason=why)

    # ------------------------------------------------------------------ cycle
    def cycle(self):
        signals = self.so.signals()
        sigmap = {s["id"]: s for s in signals}
        self.shadow.ingest(signals)
        self.shadow.optimize()
        positions = {(p["symbol"], p.get("side")): p for p in self.tb.positions()}

        if not self.dry:
            self.manage(positions, sigmap)
            positions = {(p["symbol"], p.get("side")): p for p in self.tb.positions()}

        equity, avail = self.equity()
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.st["day"] != day:
            self.st["day"], self.st["day_start_equity"] = day, equity
            self.st.pop("halted", None)
        if equity < self.st["day_start_equity"] * (1 - DAILY_LOSS_PCT):
            if not self.st.get("halted"):
                self.st["halted"] = True
                log_event("HALT", reason=f"當日虧損達 {DAILY_LOSS_PCT:.0%}，今日停止開新倉",
                          day_start_equity=self.st["day_start_equity"], equity=equity)
            return
        if PAUSE_FILE.exists():
            return
        if len(self.st["trades"]) >= S.MAX_POSITIONS:
            return

        dirs = self.so.directions()
        reg = S.regime(dirs)
        board_rows = self.so.strength_board()
        board = {b["symbol"]: b for b in board_rows}
        rhits = S.recipe_hits(self.so.recipes())
        pexp = S.path_expectancy(signals, self.st["closed"])
        specs = self.tb.specs()
        held_syms = {s for s, _ in positions} | {t["tsym"] for t in self.st["trades"].values()}

        cands = [(s, "官方訊號") for s in signals if s.get("status") == "ACTIVE"]
        cands += [(b, "強勢榜") for b in board_rows if b.get("id") and b["id"] not in sigmap]
        scored = []
        for sig, src in cands:
            if sig["id"] in self.st["traded_ids"]:
                continue
            tsym = S.to_toobit(sig["symbol"])
            if tsym not in specs:
                self.note_skip(sig, f"Toobit 無 {tsym} 合約")
                continue
            if tsym in held_syms:
                self.note_skip(sig, "該幣已有持倉")
                continue
            ev = S.evaluate(sig, src, reg[0], reg[1], board, rhits, pexp, self.learner.p)
            if src == "官方訊號":
                self.shadow.snapshot(sig, ev, S._age_min(sig["entry_at"]))
            if not ev["ok"]:
                self.note_skip(sig, ev["skip"])
                continue
            scored.append((ev["score"], sig, src, ev))

        if self.dry or os.environ.get("BOT_VERBOSE"):
            print(json.dumps({"ts": now_iso(), "regime": reg, "directions": dirs, "equity": round(equity, 2),
                              "open_trades": len(self.st["trades"]), "candidates": len(cands),
                              "qualified": [(round(s, 1), x["symbol"], x["side"]) for s, x, _, _ in scored],
                              "recipe_hits": rhits, "path_expectancy": pexp["seeone"],
                              "learned": self.learner.summary()}, ensure_ascii=False), flush=True)

        for score, sig, src, ev in sorted(scored, key=lambda x: -x[0]):
            if len(self.st["trades"]) >= S.MAX_POSITIONS:
                break
            try:
                self.open_trade(sig, ev, src, reg, equity)
            except ToobitError as e:
                self.st["traded_ids"].append(sig["id"])
                log_event("ERROR", where="open_trade", symbol=sig["symbol"], error=str(e))

    def run(self, once: bool):
        me = self.so.me()
        self.shadow.ingest(self.so.signals())
        self.shadow.optimize(force=True)
        log_event("START", dry_run=self.dry, seeone_tier=me.get("tier") or me.get("user", {}).get("tier"),
                  max_positions=S.MAX_POSITIONS, max_leverage=S.MAX_LEVERAGE,
                  daily_loss_pct=DAILY_LOSS_PCT, learned=self.learner.summary())
        while True:
            try:
                self.cycle()
            except Exception as e:  # keep the loop alive; open positions have exchange-side SL/TP
                log_event("ERROR", where="cycle", error=repr(e), tb=traceback.format_exc()[-800:])
            finally:
                self.st["traded_ids"] = self.st["traded_ids"][-500:]
                if len(self.st["skips"]) > 500:
                    self.st["skips"] = dict(list(self.st["skips"].items())[-300:])
                if not self.dry:
                    save_state(self.st)
                    self.shadow.save()
            if once:
                return
            time.sleep(LOOP_SECONDS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    load_dotenv(ROOT / ".env")
    env_dry = str(os.environ.get("DRY_RUN", "true")).lower() in ("1", "true", "yes")
    Bot(dry_run=a.dry_run or env_dry).run(a.once)


if __name__ == "__main__":
    main()
