"""See One -> Toobit live perpetual bot.

Run:  python3 -m live.bot            (live loop)
      python3 -m live.bot --once --dry-run   (print one cycle of decisions, place nothing)
Pause new entries: touch logs/PAUSE   (open trades keep being managed)
"""
import argparse
import json
import re
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from . import strategy as S
from .learner import Learner
from .shadow import Shadow
from .selfsignal import SelfScanner
from .toobit import Toobit, ToobitError

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs" / "trades"
TRADE_LOG = LOG_DIR / "live_trades.jsonl"
STATE_FILE = ROOT / "logs" / "live_state.json"
PAUSE_FILE = ROOT / "logs" / "PAUSE"
LEARN_FILE = ROOT / "logs" / "learned_params.json"
INSIGHT_LOG = LOG_DIR / "insights.jsonl"
SHADOW_FILE = ROOT / "logs" / "shadow_archive.json"
CALIB_FILE = ROOT / "logs" / "self_calib.json"
SELF_SOURCE = "自選驗算"
LOOP_SECONDS = 30
DAILY_LOSS_PCT = 0.12
SLIPPAGE_BUFFER = 0.0003
MAKER_TRIES = 6
MAKER_WAIT_S = 30
MARKET_ENTRY_FALLBACK = False
MAKER_CLOSE_TRIES = 6
MAKER_CLOSE_WAIT_S = 30
FINAL_STATUS = {"FILLED", "REJECTED", "CANCELED", "PARTIALLY_CANCELED"}


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
        self.selfscan = SelfScanner(self.so, self.tb, self.learner, CALIB_FILE, log_event)
        self.last_history = 0.0

    # ------------------------------------------------------------------ helpers
    def equity(self) -> tuple[float, float]:
        b = self.tb.balance()
        self.realized = float(b["balance"])
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

            if not tr.get("tp_migrated") and not tr.get("exit_reason"):
                self.migrate_tps(tr)
            open_ids = set()
            for kind in {tp.get("kind", "STOP") for tp in tr["tps"] if not tp["filled"]}:
                open_ids |= {str(o["orderId"]) for o in self.tb.open_orders(sym, kind)}
            newly = [tp for tp in tr["tps"] if not tp["filled"] and not tp.get("cancelled") and tp.get("order_id")
                     and str(tp["order_id"]) not in open_ids]
            for tp in newly:
                tp["filled"] = True
                tp["filled_at"] = now_iso()
                log_event("TP_HIT", trade_id=tid, symbol=sym, side=side, label=tp["label"], price=tp["price"],
                          qty=tp["qty"], reason=f"{tp['label']} ({tp['r']}R) 觸發，部分平倉 {tp['qty']} 張")
            if newly:
                filled = [i for i, tp in enumerate(tr["tps"]) if tp["filled"]]
                k = max(filled)
                if k == 0:
                    buf = tr.get("be_buffer", 0.0012)
                    target = tr["entry"] * (1 + buf if side == "LONG" else 1 - buf)
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

            ref_eq = self.st.get("day_start_equity") or 0
            if (not tr.get("exit_reason") and not any(tp["filled"] for tp in tr["tps"])
                    and tr["risk_usdt"] < ref_eq * S.MIN_RISK_PCT * 0.5):
                tr["exit_reason"] = (f"小倉位清理: 風險僅 {tr['risk_usdt']:.2f}u "
                                     f"(< {S.MIN_RISK_PCT:.0%} 權益的一半)，釋出倉位給正常尺寸的單")
                log_event("SMALL_CLOSE", trade_id=tid, symbol=sym, side=side, reason=tr["exit_reason"])
                self.maker_close(tr)
                continue

            if not tr.get("exit_reason"):
                try:
                    self.protect(tr)
                except ToobitError as e:
                    log_event("ERROR", trade_id=tid, where="protect", error=str(e))
                if tr.get("exit_reason"):
                    continue

            if tr.get("source") == SELF_SOURCE and not tr.get("exit_reason"):
                close, why = self.selfscan.recheck(tr)
                if close:
                    tr["exit_reason"] = why
                    log_event("SELF_EXIT_DECISION", trade_id=tid, symbol=sym, side=side, reason=why,
                              rechecks=tr.get("rechecks", [])[-3:])
                    self.maker_close(tr)
                continue

            sig = sigmap.get(tr["signal_id"])
            if sig and sig.get("status") == "CLOSED" and not tr.get("exit_reason"):
                tr["exit_reason"] = f"See One 訊號結束: {sig.get('exit_reason') or 'CLOSED'}"
                log_event("SIGNAL_CLOSED", trade_id=tid, symbol=sym, side=side, seeone_exit=sig.get("exit_reason"),
                          seeone_realized_r=sig.get("realized_r"), action="Maker 掛單平倉剩餘部位")
                self.maker_close(tr)
            elif not sig and tr.get("source") == "官方訊號" and time.time() - tr["opened_ts"] > 48 * 3600:
                tr["exit_reason"] = "超過 48 小時且已不在 See One 訊號列表，時間止損"
                self.maker_close(tr)

    def protect(self, tr: dict):
        """Ratchet the exchange stop from the best price reached: break-even first, then lock part of the open profit."""
        sym, side = tr["tsym"], tr["side"]
        sgn = 1 if side == "LONG" else -1
        if not tr.get("init_dist"):
            mult = float(self.tb.specs()[sym]["multiplier"])
            tr["init_dist"] = tr["risk_usdt"] / (tr["contracts"] * mult) if tr["contracts"] else abs(tr["entry"] - tr["stop"])
        dist, entry = tr["init_dist"], tr["entry"]
        if not dist:
            return
        px = self.tb.price(sym)
        cur_r = (px - entry) * sgn / dist
        if cur_r > tr.get("mfe_r", 0.0):
            tr["mfe_r"] = round(cur_r, 3)
        lk = S.lock_r(tr.get("mfe_r", 0.0), tr.get("tier") == "HIGH", bool(self.st.get("day_lock_hit")))
        if lk is None or lk <= tr.get("lock_r", -1):
            return
        buf = tr.get("be_buffer", 0.0012)
        target = entry * (1 + sgn * buf) + sgn * lk * dist
        if (target - tr["stop"]) * sgn <= 0:
            tr["lock_r"] = lk
            return
        label = "保本(含手續費)" if lk == 0 else f"鎖利 +{lk}R"
        why = f"最高浮盈 {tr['mfe_r']:+.2f}R，止損上移至{label}"
        if (px - target) * sgn <= px * 0.0015:
            tr["exit_reason"] = f"{why}，但現價 {px} 已回落至鎖利線附近，直接 Maker 平倉鎖住利潤"
            log_event("LOCK_EXIT", trade_id=tr["id"], symbol=sym, side=side, price=px, lock_r=lk,
                      mfe_r=tr["mfe_r"], reason=tr["exit_reason"])
            self.maker_close(tr)
            return
        self.replace_stop(tr, target, why)
        tr["lock_r"] = lk
        log_event("PROFIT_LOCK", trade_id=tr["id"], symbol=sym, side=side, mfe_r=tr["mfe_r"], lock_r=lk,
                  stop=tr["stop"], reason=why)

    def finish(self, tid: str, tr: dict):
        sym, side = tr["tsym"], tr["side"]
        for tp in tr["tps"]:
            if not tp["filled"] and tp.get("order_id"):
                self.cancel_quiet(tp["order_id"], tp.get("kind", "STOP"))
        pnl, close_px = None, None
        try:
            for h in self.tb.history_positions(sym, 10):
                if h.get("side") == side and int(h.get("closeTime") or 0) >= tr["opened_ts"] * 1000 - 60000:
                    pnl = (pnl or 0) + float(h.get("realizedPnL") or h.get("realizedPnl") or 0)
                    close_px = h.get("closeAvgPrice")
        except ToobitError as e:
            log_event("ERROR", trade_id=tid, where="history_positions", error=str(e))
        fees = self.trade_fees(sym, tr["opened_ts"])
        filled = [tp["label"] for tp in tr["tps"] if tp["filled"]]
        if tr.get("exit_reason"):
            reason = tr["exit_reason"]
        elif len(filled) == len(tr["tps"]):
            reason = f"全部止盈完成 ({', '.join(filled)})"
        elif tr.get("lock_r") is not None:
            reason = (f"最高浮盈 {tr.get('mfe_r', 0):+.2f}R 後鎖利止損 {tr['stop']} 觸發 (鎖 +{tr['lock_r']}R)"
                      + (f"，已達 {filled[-1]}" if filled else ""))
        elif filled:
            reason = f"已達 {filled[-1]} 後，移動止損 {tr['stop']} 觸發"
        else:
            reason = f"初始止損 {tr['stop']} 觸發"
        r_result = round(pnl / tr["risk_usdt"], 3) if pnl is not None and tr["risk_usdt"] else None
        log_event("EXIT", trade_id=tid, symbol=sym, side=side, reason=reason, tps_filled=filled,
                  entry=tr["entry"], close_avg=close_px, realized_pnl_usdt=pnl, r_result=r_result,
                  path=tr.get("path"), held_min=round((time.time() - tr["opened_ts"]) / 60, 1),
                  fees=fees, fee_r=round(fees["fee_usdt"] / tr["risk_usdt"], 3) if fees and tr["risk_usdt"] else None)
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
        risk_pct = max(S.MIN_RISK_PCT, self.learner.p["risk_pct"])
        margin_cap = S.NORMAL_MARGIN_CAP_PCT
        high_open = sum(1 for t in self.st["trades"].values() if t.get("tier") == "HIGH")
        tier = ("HIGH" if ev.get("tier") == "HIGH" and high_open < S.MAX_HIGH_POSITIONS
                and not self.st.get("day_lock_armed") else "NORMAL")
        if tier == "HIGH":
            risk_pct = S.HIGH_RISK_PCT
            margin_cap = S.HIGH_MARGIN_CAP_PCT
        sz = S.size_position(equity, price, stop, spec, risk_pct, margin_cap)
        if not sz:
            self.note_skip(sig, f"無法以至少 {S.MIN_RISK_PCT:.0%} 風險建立部位 (槓桿/保證金上限或最小下單量)，不開小倉")
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
                          learned_min_score=self.learner.p["min_score"], risk_pct=risk_pct, tier=tier,
                          projected_profit_usdt=self.projected(sig, plan, price, spec))
        if self.dry:
            log_event("DRY_ENTRY", **entry_info)
            return True

        self.tb.set_isolated(sym)
        self.tb.set_leverage(sym, sz["leverage"])
        self.st["traded_ids"].append(sig["id"])
        fill = self.maker_entry(sym, side, sz["contracts"], stop, sig, spec)
        pos = next((p for p in self.tb.positions(sym) if p.get("side") == side), None)
        if not pos:
            log_event("SKIP" if not fill["filled"] else "ERROR", signal_id=sig["id"], symbol=sym, side=side,
                      reason=fill.get("note") or "下單後查無持倉", fill=fill)
            return False
        entry = float(pos.get("avgPrice") or price)
        contracts = fill["filled"] or sz["contracts"]
        if contracts != sz["contracts"]:
            plan = S.tp_plan(sig, contracts, spec["min_contracts"], self.learner.p.get("tp_split"))
        tps = self.place_tps(sym, side, entry, plan)
        sz = {**sz, "contracts": contracts}
        entry_fee = S.MAKER_FEE if fill["maker"] >= fill["taker"] else S.TAKER_FEE
        order = {"orderId": fill.get("order_ids")}
        tid = f"{sym}:{side}:{int(time.time())}"
        mult = float(spec["multiplier"])
        self.st["trades"][tid] = {
            "id": tid, "signal_id": sig["id"], "source": source, "tsym": sym, "side": side,
            "contracts": sz["contracts"], "entry": entry, "stop": stop, "tps": tps,
            "risk_usdt": round(sz["contracts"] * mult * abs(entry - stop), 4), "margin": sz["margin"],
            "leverage": sz["leverage"], "path": sig.get("signal_path"), "opened_ts": time.time(),
            "be_buffer": entry_fee + S.TAKER_FEE + SLIPPAGE_BUFFER, "entry_fill": fill,
            "analysis": ev.get("analysis"), "analysis_bias": (ev.get("analysis") or {}).get("bias", 0.0),
            "tier": tier, "m15_at_entry": getattr(self, "m15", None),
            "same_side_at_entry": sum(1 for t in self.st["trades"].values() if t["side"] == side),
            "factors": ev["factors"], "entry_cur_r": ev["cur_r"], "score": ev["score"], "regime": list(regime),
        }
        log_event("ENTRY", trade_id=tid, entry_avg=entry, order_id=order.get("orderId"),
                  tp_orders=[(t["label"], t["price"], t["qty"], t["order_id"]) for t in tps], **entry_info)
        return True

    def _entry_ok(self, px: float, sig: dict, stop: float) -> bool:
        side, entry = sig["side"], float(sig["entry_price"])
        risk = abs(entry - stop)
        cur_r = (px - entry) / risk * (1 if side == "LONG" else -1) if risk else 0
        away = px > stop * 1.001 if side == "LONG" else px < stop * 0.999
        return away and cur_r <= self.learner.p["entry_r_window"][1]

    def maker_entry(self, sym: str, side: str, contracts: int, stop: float, sig: dict, spec: dict) -> dict:
        """Work the entry as post-only (maker) at the touch; fall back to market only while still in the entry band."""
        open_side = "BUY_OPEN" if side == "LONG" else "SELL_OPEN"
        res = {"filled": 0, "maker": 0, "taker": 0, "order_ids": [], "note": None}
        stop_px = self.tb.fmt_price(sym, stop)
        stop_set = False
        for _ in range(MAKER_TRIES):
            left = contracts - res["filled"]
            if left < spec["min_contracts"]:
                break
            bid, ask = self.tb.book(sym)
            tick = float(spec["tick"])
            if ask - bid > tick * 1.5:
                px = bid + tick if side == "LONG" else ask - tick
            else:
                px = bid if side == "LONG" else ask
            if not self._entry_ok(px, sig, stop):
                res["note"] = f"掛單期間價格 {px} 已離開進場區間，不追價"
                break
            try:
                o = self.tb.post_only(sym, open_side, left, self.tb.fmt_price(sym, px), "mo")
            except ToobitError as e:
                log_event("ERROR", where="post_only_entry", symbol=sym, error=str(e))
                continue
            oid = str(o["orderId"])
            res["order_ids"].append(oid)
            q, deadline = o, time.time() + MAKER_WAIT_S
            while time.time() < deadline:
                time.sleep(3)
                q = self.tb.order(oid)
                if q.get("status") in FINAL_STATUS:
                    break
            if q.get("status") not in FINAL_STATUS:
                self.cancel_quiet(oid, "LIMIT")
                time.sleep(0.5)
                q = self.tb.order(oid)
            ex = int(float(q.get("executedQty") or 0))
            if ex:
                res["filled"] += ex
                res["maker"] += ex
                if not stop_set:
                    self.tb.set_position_stop(sym, side, stop_px)
                    stop_set = True
            if res["filled"] >= contracts:
                break
        left = contracts - res["filled"]
        if left >= spec["min_contracts"] and res["note"] is None and not MARKET_ENTRY_FALLBACK:
            res["note"] = (f"Maker 掛單 {MAKER_TRIES} 次未完全成交，剩餘 {left} 張不以市價補單"
                           + ("，保留已成交部位" if res["filled"] else "，放棄此訊號"))
        elif left >= spec["min_contracts"] and res["note"] is None:
            px = self.tb.price(sym)
            if self._entry_ok(px, sig, stop):
                if stop_set:
                    o = self.tb.market(sym, open_side, left)
                else:
                    o = self.tb.market_open(sym, side, left, stop_px)
                    stop_set = True
                res["order_ids"].append(str(o.get("orderId")))
                res["filled"] += left
                res["taker"] += left
                time.sleep(1.5)
            else:
                res["note"] = f"Maker 未成交且價格 {px} 已離開進場區間，放棄剩餘 {left} 張"
        if res["filled"] and not stop_set:
            self.tb.set_position_stop(sym, side, stop_px)
        log_event("ENTRY_FILL", symbol=sym, side=side, **res,
                  maker_ratio=round(res["maker"] / res["filled"], 3) if res["filled"] else None)
        return res

    def _maker_px(self, sym: str, buy: bool) -> str:
        bid, ask = self.tb.book(sym)
        tick = float(self.tb.specs()[sym]["tick"])
        wide = ask - bid > tick * 1.5
        px = (bid + tick if wide else bid) if buy else (ask - tick if wide else ask)
        return self.tb.fmt_price(sym, px)

    def maker_close(self, tr: dict):
        """Exit a whole position with post-only orders at the touch; market only for whatever is left after the retries.
        The exchange-side stop stays in place the whole time."""
        sym, side = tr["tsym"], tr["side"]
        for tp in tr["tps"]:
            if not tp["filled"] and tp.get("order_id"):
                self.cancel_quiet(tp["order_id"], tp.get("kind", "STOP"))
                tp["cancelled"] = True
        close_side = "SELL_CLOSE" if side == "LONG" else "BUY_CLOSE"
        maker = taker = 0
        for _ in range(MAKER_CLOSE_TRIES):
            pos = next((p for p in self.tb.positions(sym) if p.get("side") == side), None)
            if not pos:
                break
            qty = int(float(pos["position"]))
            try:
                o = self.tb.post_only(sym, close_side, qty, self._maker_px(sym, buy=side == "SHORT"), "mc")
            except ToobitError as e:
                log_event("ERROR", where="maker_close", symbol=sym, error=str(e))
                continue
            q, deadline = o, time.time() + MAKER_CLOSE_WAIT_S
            while time.time() < deadline:
                time.sleep(2)
                q = self.tb.order(o["orderId"])
                if q.get("status") in FINAL_STATUS:
                    break
            if q.get("status") not in FINAL_STATUS:
                self.cancel_quiet(o["orderId"], "LIMIT")
                time.sleep(0.5)
                q = self.tb.order(o["orderId"])
            maker += int(float(q.get("executedQty") or 0))
        pos = next((p for p in self.tb.positions(sym) if p.get("side") == side), None)
        if pos:
            taker = int(float(pos["position"]))
            try:
                self.tb.flash_close(sym, side)
            except ToobitError as e:
                log_event("ERROR", trade_id=tr["id"], where="flash_close", error=str(e))
        log_event("EXIT_FILL", trade_id=tr["id"], symbol=sym, side=side, maker_contracts=maker, taker_contracts=taker,
                  reason=tr.get("exit_reason"))

    def migrate_tps(self, tr: dict):
        """Swap trigger-market TPs (taker) for post-only limit TPs (maker) on trades opened before maker TPs existed."""
        tr["tp_migrated"] = True
        sym, side = tr["tsym"], tr["side"]
        close_side = "SELL_CLOSE" if side == "LONG" else "BUY_CLOSE"
        moved = []
        for tp in tr["tps"]:
            if tp["filled"] or tp.get("kind", "STOP") != "STOP" or not tp.get("order_id"):
                continue
            try:
                self.tb.cancel_order(tp["order_id"], "STOP")
            except ToobitError:
                continue
            px = self.tb.fmt_price(sym, tp["price"])
            try:
                o = self.tb.post_only(sym, close_side, tp["qty"], px, tp["label"].lower())
                time.sleep(0.6)
                if self.tb.order(o["orderId"]).get("status") == "REJECTED":
                    raise ToobitError("post-only rejected")
                tp.update(order_id=str(o["orderId"]), kind="LIMIT")
                moved.append(tp["label"])
            except ToobitError:
                o = self.tb.close_trigger(sym, side, tp["qty"], px, tp["label"].lower())
                tp.update(order_id=str(o.get("orderId")), kind="STOP")
        if moved:
            log_event("TP_MIGRATE", trade_id=tr["id"], symbol=sym, moved_to_maker=moved,
                      reason="觸發市價止盈改為 Maker 限價止盈，降低手續費")

    def place_tps(self, sym: str, side: str, entry: float, plan: list) -> list:
        """TPs as post-only limit closes (maker fee); a TP that would cross falls back to a trigger order."""
        close_side = "SELL_CLOSE" if side == "LONG" else "BUY_CLOSE"
        tps = []
        for tp in plan:
            if (tp["price"] <= entry) if side == "LONG" else (tp["price"] >= entry):
                continue
            px = self.tb.fmt_price(sym, tp["price"])
            rec = {**tp, "order_id": None, "kind": "LIMIT", "filled": False}
            try:
                o = self.tb.post_only(sym, close_side, tp["qty"], px, tp["label"].lower())
                time.sleep(0.6)
                if self.tb.order(o["orderId"]).get("status") == "REJECTED":
                    raise ToobitError("post-only rejected")
                rec["order_id"] = str(o["orderId"])
            except ToobitError:
                try:
                    o = self.tb.close_trigger(sym, side, tp["qty"], px, tp["label"].lower())
                    rec.update(order_id=str(o.get("orderId")), kind="STOP")
                except ToobitError as e:
                    log_event("ERROR", where="tp_order", symbol=sym, label=tp["label"], error=str(e))
            tps.append(rec)
        return tps

    def trade_fees(self, sym: str, opened_ts: float) -> dict | None:
        try:
            rows = [t for t in self.tb.user_trades(sym, 200) if int(t["time"]) >= opened_ts * 1000 - 120000]
        except ToobitError:
            return None
        fee = sum(float(t.get("commission") or 0) for t in rows)
        maker = sum(1 for t in rows if t.get("isMaker"))
        return {"fee_usdt": round(fee, 5), "fills": len(rows), "maker_fills": maker}

    @staticmethod
    def projected(sig: dict, plan: list, price: float, spec: dict) -> dict:
        """Cumulative USDT gained (before fees) if price reaches each TP in turn."""
        mult, out, total = float(spec["multiplier"]), {}, 0.0
        for tp in plan:
            total += tp["qty"] * mult * abs(tp["price"] - price)
            out[tp["label"]] = round(total, 2)
        out["SL"] = round(-sum(tp["qty"] for tp in plan) * mult * abs(price - float(sig["stop_price"])), 2)
        return out

    def note_skip(self, sig: dict, why: str):
        key = sig["id"]
        kind = re.sub(r"[-+]?\d+(\.\d+)?", "#", why)
        if self.st["skips"].get(key) != kind:
            self.st["skips"][key] = kind
            log_event("SKIP", signal_id=key, symbol=sig["symbol"], side=sig["side"], reason=why)

    # ------------------------------------------------------------------ cycle
    def cycle(self):
        signals = self.so.signals()
        sigmap = {s["id"]: s for s in signals}
        self.shadow.ingest(signals)
        if time.time() - self.last_history > 3600:
            self.last_history = time.time()
            try:
                self.shadow.ingest(self.so.history())
            except Exception as e:
                log_event("ERROR", where="history", error=repr(e)[:200])
        self.shadow.optimize()
        positions = {(p["symbol"], p.get("side")): p for p in self.tb.positions()}

        if not self.dry:
            self.manage(positions, sigmap)
            positions = {(p["symbol"], p.get("side")): p for p in self.tb.positions()}

        equity, avail = self.equity()
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        bal = self.realized
        if self.st["day"] != day or "day_start_balance" not in self.st:
            today = 0.0
            if self.st["day"] != day:
                self.st["day"], self.st["day_start_equity"] = day, equity
            else:
                today = sum(c["pnl"] or 0 for c in self.st["closed"] if c["closed_at"].startswith(day))
            self.st["day_start_balance"] = bal - today
            self.st["day_peak_balance"] = bal
            for k in ("halted", "day_lock_armed", "day_lock_hit", "day_peak_equity"):
                self.st.pop(k, None)
        self.st.setdefault("principal", equity)
        # Profit lock tracks realized balance only, so a floating profit that is later given back can't trip it.
        start = self.st["day_start_balance"]
        peak = self.st["day_peak_balance"] = max(self.st["day_peak_balance"], bal)
        if peak - start >= start * S.DAY_LOCK_TRIGGER_PCT and not self.st.get("day_lock_armed"):
            self.st["day_lock_armed"] = True
            log_event("DAY_LOCK_ARMED", day_start_balance=round(start, 2), balance=round(bal, 2),
                      reason=f"當日已實現獲利達 {S.DAY_LOCK_TRIGGER_PCT:.0%}，今日停用高信心放大單；"
                             f"若回吐 {S.DAY_LOCK_GIVEBACK:.0%} 當日最高已實現獲利則停止開新倉")
        floor = start + (peak - start) * (1 - S.DAY_LOCK_GIVEBACK)
        if self.st.get("day_lock_armed") and bal < floor:
            if not self.st.get("day_lock_hit"):
                self.st["day_lock_hit"] = True
                log_event("DAY_LOCK_HIT", day_start_balance=round(start, 2), day_peak_balance=round(peak, 2),
                          balance=round(bal, 2), floor=round(floor, 2),
                          reason="當日已實現獲利回吐過半，今日停止開新倉，持倉浮盈 +0.3R 即移至保本")
            return
        if equity < self.st["day_start_equity"] * (1 - DAILY_LOSS_PCT):
            if not self.st.get("halted"):
                self.st["halted"] = True
                log_event("HALT", reason=f"當日虧損達 {DAILY_LOSS_PCT:.0%}，今日停止開新倉",
                          day_start_equity=self.st["day_start_equity"], equity=equity)
            return
        if PAUSE_FILE.exists() or time.time() < self.learner.p.get("pause_until", 0):
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

        try:
            for f in self.selfscan.scan(reg, rhits, held_syms, specs, dirs.get("15m")):
                if f["sig"]["id"] not in self.st["traded_ids"]:
                    scored.append((f["ev"]["score"], f["sig"], SELF_SOURCE, f["ev"]))
        except Exception as e:
            log_event("ERROR", where="self_scan", error=repr(e)[:300], tb=traceback.format_exc()[-600:])

        if self.dry or os.environ.get("BOT_VERBOSE"):
            print(json.dumps({"ts": now_iso(), "regime": reg, "directions": dirs, "equity": round(equity, 2),
                              "open_trades": len(self.st["trades"]), "candidates": len(cands),
                              "qualified": [(round(s, 1), x["symbol"], x["side"]) for s, x, _, _ in scored],
                              "recipe_hits": rhits, "path_expectancy": pexp["seeone"],
                              "learned": self.learner.summary()}, ensure_ascii=False), flush=True)

        principal = self.st["principal"]
        size_eq = equity - S.PROFIT_RESERVE * max(0.0, equity - principal)
        m15 = self.m15 = dirs.get("15m") or 0.0
        m15_max = self.learner.p.get("m15_max", S.M15_MAX_DEFAULT)
        for score, sig, src, ev in sorted(scored, key=lambda x: -x[0]):
            if len(self.st["trades"]) >= S.MAX_POSITIONS:
                break
            sgn = 1 if sig["side"] == "LONG" else -1
            cd = self.learner.p.get("symbol_cooldown", {}).get(sig["symbol"], 0)
            if cd > time.time():
                self.note_skip(sig, f"{sig['symbol']} 近期虧損冷卻中 (至台北 "
                                    f"{datetime.fromtimestamp(cd + 8 * 3600, timezone.utc):%H:%M})")
                continue
            if m15 * sgn >= m15_max:
                self.note_skip(sig, f"See One 15m 方向 {m15:+.1f}，順勢{'做多' if sgn > 0 else '做空'}動能已達 "
                                    f"{m15 * sgn:+.1f} (上限 {m15_max:+.1f})，追在走勢中段/末端，等回檔或盤整")
                continue
            same = [t for t in self.st["trades"].values() if t["side"] == sig["side"]]
            if len(same) >= S.MAX_SAME_SIDE:
                self.note_skip(sig, f"同方向已有 {len(same)} 筆持倉，避免同一方向集中押注")
                continue
            last = max((t["opened_ts"] for t in same), default=0)
            if time.time() - last < S.SAME_SIDE_GAP_MIN * 60:
                self.note_skip(sig, f"同方向 {S.SAME_SIDE_GAP_MIN} 分鐘內已開倉，分散進場時間")
                continue
            try:
                self.open_trade(sig, ev, src, reg, size_eq)
            except ToobitError as e:
                self.st["traded_ids"].append(sig["id"])
                log_event("ERROR", where="open_trade", symbol=sig["symbol"], error=str(e))

    def run(self, once: bool):
        me = self.so.me()
        self.shadow.ingest(self.so.signals())
        self.shadow.ingest(self.so.history())
        self.last_history = time.time()
        self.shadow.optimize(force=True)
        log_event("START", dry_run=self.dry, seeone_tier=me.get("tier") or me.get("user", {}).get("tier"),
                  max_positions=S.MAX_POSITIONS, max_leverage=S.MAX_LEVERAGE,
                  daily_loss_pct=DAILY_LOSS_PCT, lock_ladder=S.LOCK_LADDER, trail_gap_r=S.TRAIL_GAP_R,
                  day_lock=[S.DAY_LOCK_TRIGGER_PCT, S.DAY_LOCK_GIVEBACK], profit_reserve=S.PROFIT_RESERVE,
                  principal=self.st.get("principal"), learned=self.learner.summary())
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
