"""See One live perpetual bot. EXCHANGE=toobit by default; EXCHANGE=gate uses Gate.io.

Run:  python3 -m live.bot            (live loop)
      python3 -m live.bot --once --dry-run   (print one cycle of decisions, place nothing)
Pause new entries: touch logs/PAUSE   (open trades keep being managed)
"""
import argparse
import json
import re
import os
import signal
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from . import strategy as S
from .exchange import connect, exchange_name
from .learner import Learner
from .market import klines, listing
from .shadow import Shadow
from .selfsignal import MAJORS, SelfScanner, btc_ret1h, in_side, move_2h, move_before, rsi_side
from .toobit import ToobitError

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs" / "trades"
TRADE_LOG = LOG_DIR / "live_trades.jsonl"
STATE_FILE = ROOT / "logs" / "live_state.json"
PAUSE_FILE = ROOT / "logs" / "PAUSE"
# Holds a UTC date: on that day the daily profit lock and its give-back stop are off (daily loss stop still applies).
LOCK_OFF_FILE = ROOT / "logs" / "DAY_LOCK_OFF"
# Present: this venue ignores the daily loss ceiling and sizes at the normal risk. Toobit only unless the same file is created under logs/gate.
RISK_OPEN_FILE = ROOT / "logs" / "RISK_OPEN"
LEARN_FILE = ROOT / "logs" / "learned_params.json"
INSIGHT_LOG = LOG_DIR / "insights.jsonl"
SHADOW_FILE = ROOT / "logs" / "shadow_archive.json"
CALIB_FILE = ROOT / "logs" / "self_calib.json"


def use_venue(venue: str):
    """Point this process at one book. Gate keeps the same rules and a separate state file."""
    global LOG_DIR, TRADE_LOG, STATE_FILE, PAUSE_FILE, LOCK_OFF_FILE, RISK_OPEN_FILE, LEARN_FILE, INSIGHT_LOG, SHADOW_FILE, CALIB_FILE
    base = ROOT / "logs" / "gate" if venue == "gate" else ROOT / "logs"
    LOG_DIR = base / "trades"
    TRADE_LOG = LOG_DIR / "live_trades.jsonl"
    STATE_FILE = base / "live_state.json"
    PAUSE_FILE = base / "PAUSE"
    LOCK_OFF_FILE = base / "DAY_LOCK_OFF"
    RISK_OPEN_FILE = base / "RISK_OPEN"
    LEARN_FILE = base / "learned_params.json"
    INSIGHT_LOG = LOG_DIR / "insights.jsonl"
    SHADOW_FILE = base / "shadow_archive.json"
    CALIB_FILE = base / "self_calib.json"


use_venue(exchange_name())
SELF_SOURCE = "自選驗算"
ADOPT_SOURCE = "接管持倉"
RECONCILE_S = 60
ADOPT_TP_R = [1.0, 1.5, 2.0, 3.0, 4.0]
ADOPT_MAX_STOP_PCT = 0.06
LOOP_SECONDS = 30
SCAN_RESULT_TTL_S = 120
# While the main loop is inside a See One scan, this thread still ratchets exchange stops.
FAST_GUARD_S = 1               # with a position open, watch price every second
SPIKE_DROP_R = 0.45             # one-second adverse jump treated as a wick
DEEP_ADVERSE_R = -0.85          # exit before a wick can slip through the full stop
# After TP1 the exchange stop moves here, and the runner is also cut here.
# With TP1 at 10% of the size, the 552-review replay averaged +0.80R at -0.5R,
# +0.76R at -0.25R and lower at the entry, ahead in both halves of Nov 4-10.
AFTER_TP_DEEP_R = -0.5
# +0.25R marks a trade that has worked, for the higher-timeframe exit only.
# Closing the whole position at the entry used to sit beside the +2R lock and the
# -0.25R stop after TP1. HOT's long on Nov 8 peaked at +0.70R and was closed at
# the entry for +2.22 USDT. HMSTR peaked at +0.37R and was closed for +0.43 USDT.
GIVEBACK_ARM_R = 0.25
# A legal entry is not flattened by a later higher-timeframe reading while it is
# still near the entry. ARX long at 20:11 Taipei on Nov 8 was allowed at -0.4.
# Eleven minutes later the score was -1.25 and the guard closed -0.11R. WLD, XMR,
# and AKE were closed by this rule only after they had already reached +0.25R.
HTF_FLAT_AFTER_R = -0.50
# A normal cycle with several maker entries/exits stays well under this.
WATCHDOG_S = 1200
SLIPPAGE_BUFFER = 0.0003
# A maker fill inside this fraction of the reviewed stop distance is a different trade.
# The touch walked into the stop: ZK short on Nov 11 05:42 Taipei was reviewed at 0.80%
# and filled at 0.57% (72% of the plan). It stopped out in 14 minutes for -1.10R and
# price then ran to about -2.7R. The other two live fills under 75% (CLANKER, BR) also
# lost, and no winner sat in that bucket. HAEDAL's worse fill (away from the stop) is
# handled by the size cap below, not by this floor.
STOP_KEEP_FRAC = 0.75
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


_STATE_LOCK = threading.Lock()


def save_state(st: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(st, ensure_ascii=False, indent=1, default=str)
    with _STATE_LOCK:
        tmp = STATE_FILE.with_suffix(".tmp")
        tmp.write_text(payload)
        tmp.replace(STATE_FILE)


class Bot:
    def __init__(self, dry_run: bool):
        global TRADE_LOG
        load_dotenv(ROOT / ".env")
        self.dry = dry_run
        if dry_run:
            TRADE_LOG = LOG_DIR / "dry_run.jsonl"
        self.tb = connect()
        self.paper = os.environ.get("GATE_PAPER") == "1" and getattr(self.tb, "name", "") == "gate"
        self.so = S.SeeOne(os.environ["SEEONE_RT_SESSION"])
        self.so.log = log_event
        self.st = load_state()
        self.learner = Learner(LEARN_FILE, INSIGHT_LOG, log_event)
        self.shadow = Shadow(SHADOW_FILE, self.learner, log_event)
        self.selfscan = SelfScanner(self.so, self.tb, self.learner, CALIB_FILE, log_event)
        self._scan_th: threading.Thread | None = None
        self._scan_lock = threading.Lock()
        self._scan_out: list = []
        self.last_history = 0.0
        self.pre_moves: dict[str, float | None] = {}
        self._protect_lock = threading.Lock()

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
        if tr.get("paper"):
            px = self.tb.fmt_price(sym, new_stop)
            log_event("STOP_MOVE", trade_id=tr["id"], symbol=sym, side=side, old_stop=tr["stop"], new_stop=px,
                      reason=reason, paper=True)
            tr["stop"] = float(px)
            return
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

            if tr.get("paper"):
                self._paper_mark(tr)
            elif not tr.get("tp_migrated") and not tr.get("exit_reason"):
                self.migrate_tps(tr)
            open_ids = set()
            if not tr.get("paper"):
                for kind in {tp.get("kind", "STOP") for tp in tr["tps"] if not tp["filled"]}:
                    open_ids |= {str(o["orderId"]) for o in self.tb.open_orders(sym, kind)}
            newly = [tp for tp in tr["tps"] if not tp["filled"] and not tp.get("cancelled") and tp.get("order_id")
                     and str(tp["order_id"]) not in open_ids]
            for tp in newly:
                tp["filled"] = True
                tp["filled_at"] = now_iso()
                log_event("TP_HIT", trade_id=tid, symbol=sym, side=side, label=tp["label"], price=tp["price"],
                          qty=tp["qty"], reason=f"{tp['label']} ({tp['r']}R) 觸發，部分平倉 {tp['qty']} 張")
            if any(tp.get("filled") and tp.get("label") == "TP1" for tp in tr["tps"]) and not tr.get("exit_reason"):
                if not tr.get("init_dist"):
                    mult = float(self.tb.specs()[sym]["multiplier"])
                    tr["init_dist"] = (tr["risk_usdt"] / (tr["contracts"] * mult) if tr["contracts"]
                                       else abs(tr["entry"] - tr["stop"]))
                sgn = 1 if side == "LONG" else -1
                target = float(self.tb.fmt_price(sym, tr["entry"] + sgn * AFTER_TP_DEEP_R * tr["init_dist"]))
                if tr["init_dist"] and (target - tr["stop"]) * sgn > 0 and not all(tp.get("filled") for tp in tr["tps"]):
                    try:
                        self.replace_stop(tr, target, f"TP1 達成，止損前移到 {AFTER_TP_DEEP_R:+.2f}R")
                    except ToobitError as e:
                        log_event("ERROR", trade_id=tid, where="tp1_stop", error=str(e))

            # Dust is half a minimum trade on the equity the book is trading now.
            # Day-start equity stays the loss-halt baseline. On Nov 11 it was still 317
            # while the account was near 160, so half of 5% of 317 was 7.9u and every
            # full-size entry (risk about 7.4-7.9u) was flattened within a minute.
            # Eighteen of those closes cost 12.15u, mostly fees.
            ref_eq = getattr(self, "_mark_equity", None) or self.st.get("day_start_equity") or 0
            # A fresh signal that only partly filled is not dust. ALGO long at 19:07
            # Taipei on Nov 8 was a qualified review; the maker kept 32,791 of 353,265
            # contracts and this cleanup flattened it at +0.06R. Price later reached +1R.
            partial_keep = "保留已成交" in str((tr.get("entry_fill") or {}).get("note") or "")
            if (not tr.get("exit_reason") and not partial_keep
                    and not any(tp["filled"] for tp in tr["tps"])
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

            if tr.get("source") in (SELF_SOURCE, ADOPT_SOURCE) and not tr.get("exit_reason"):
                close, why = self.selfscan.recheck(tr)
                if close:
                    with self._protect_lock:
                        if not tr.get("exit_reason"):
                            tr["exit_reason"] = why
                            log_event("SELF_EXIT_DECISION", trade_id=tid, symbol=sym, side=side, reason=why,
                                      rechecks=tr.get("rechecks", [])[-3:])
                            self.maker_close(tr)
                continue

            if sigmap is None:
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

    def _spike_through_lock(self, tr: dict, cur_r: float) -> bool:
        """A one-second drop still above a profit-lock stop is left to that stop.

        Market-closing it filled through the stop on XMR (Nov 10 10:57 Taipei): the print
        was +3.06R, the lock was about +3R, and the fill realized +2.43R.
        """
        lk = S.lock_r(tr.get("mfe_r", 0.0), tr.get("tier") == "HIGH", bool(self.st.get("day_lock_hit")))
        return not (lk is not None and lk > 0 and cur_r > lk)

    def protect(self, tr: dict):
        """Ratchet the exchange stop from the best price reached: break-even first, then lock part of the open profit."""
        with self._protect_lock:
            self._protect(tr)

    def _protect(self, tr: dict):
        if tr.get("exit_reason"):
            return
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
        prev_r = tr.get("_prev_r")
        tr["_prev_r"] = round(cur_r, 3)
        htf = getattr(self, "htf", None)
        htf_entry = tr.get("htf_at_entry")
        opened_against = htf_entry is not None and htf_entry * sgn < S.HTF_MIN_DEFAULT
        worked = tr.get("mfe_r", 0.0) >= GIVEBACK_ARM_R and time.time() - tr["opened_ts"] >= 900
        # Bounce longs open only when 1h/4h is already below this line, so the check would close them at once
        # (BR on Nov 10 was closed one second after entry).
        htf_flat = (htf is not None and htf * sgn < S.HTF_MIN_DEFAULT and tr.get("path") != S.BOUNCE_PATH
                    and not tr.get("after_red") and not tr.get("recipe_spike")
                    and tr.get("path") != S.BOUNCE_PATH
                    and (opened_against or worked or cur_r <= HTF_FLAT_AFTER_R))
        if htf_flat:
            tr["exit_reason"] = (f"大週期順方向 {htf * sgn:+.2f} 已低於 {S.HTF_MIN_DEFAULT}，趨勢消失，平倉")
        # A one-second wick that is still on the profitable side of the entry is not a stop-out.
        # MOCA short (Gate, Nov 11 12:35 Taipei) fell from +0.75R to +0.15R and was market-closed
        # at +0.05R. Price then reached about +3.3R. MELANIA (Toobit, Nov 10 06:06) was the same
        # kind of wick and was still about +1.5R an hour later. A wick that crosses to a loss
        # still flattens, so a spike through the stop is not left on.
        elif (prev_r is not None and prev_r - cur_r >= SPIKE_DROP_R and cur_r <= 0
              and self._spike_through_lock(tr, cur_r)):
            tr["exit_reason"] = (f"一秒內逆向 {prev_r - cur_r:.2f}R（從 {prev_r:+.2f}R 到 {cur_r:+.2f}R），視為插針，平倉")
        elif cur_r <= (AFTER_TP_DEEP_R if any(tp.get("filled") for tp in tr.get("tps") or []) else DEEP_ADVERSE_R):
            line = AFTER_TP_DEEP_R if any(tp.get("filled") for tp in tr.get("tps") or []) else DEEP_ADVERSE_R
            tr["exit_reason"] = (f"浮虧 {cur_r:+.2f}R 已深於 {line}R，先平倉，避免插針滑過止損")
        if tr.get("exit_reason") and tr["exit_reason"].startswith(("大週期", "一秒", "浮虧")):
            log_event("ADVERSE_EXIT", trade_id=tr["id"], symbol=sym, side=side, price=px, cur_r=round(cur_r, 3),
                      reason=tr["exit_reason"])
            self.maker_close(tr, urgent=not tr["exit_reason"].startswith("大週期"))
            return
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

    def _paper_mark(self, tr: dict):
        """Stop and targets for a simulated position, using the live Gate price and no orders."""
        sym, side = tr["tsym"], tr["side"]
        px = self.tb.price(sym)
        sgn = 1 if side == "LONG" else -1
        tr["paper_px"] = px
        if (px - float(tr["stop"])) * sgn <= 0 and not tr.get("exit_reason"):
            tr["paper_exit_px"] = float(tr["stop"])
            tr["exit_reason"] = f"模擬止損 {tr['stop']}"
            log_event("LOCK_EXIT" if tr.get("lock_r") is not None else "EXIT", trade_id=tr["id"], symbol=sym,
                      side=side, price=px, paper=True, reason=tr["exit_reason"])
            return
        for tp in tr["tps"]:
            if tp.get("filled"):
                continue
            if (px - float(tp["price"])) * sgn < 0:
                continue
            tp["filled"] = True
            tp["filled_at"] = now_iso()
            log_event("TP_HIT", trade_id=tr["id"], symbol=sym, side=side, label=tp["label"], price=tp["price"],
                      qty=tp["qty"], paper=True, reason=f"模擬盤觸及 {tp['label']}")

    def _paper_positions(self) -> dict:
        out = {}
        for tr in self.st["trades"].values():
            if tr.get("exit_reason"):
                continue
            out[(tr["tsym"], tr["side"])] = {
                "symbol": tr["tsym"], "side": tr["side"], "position": tr["contracts"],
                "avgPrice": tr["entry"], "leverage": tr.get("leverage"), "margin": tr.get("margin"),
                "unrealizedPnL": 0, "markPrice": tr.get("paper_px") or tr["entry"],
            }
        return out

    def _paper_finish(self, tid: str, tr: dict):
        sym, side = tr["tsym"], tr["side"]
        sgn = 1 if side == "LONG" else -1
        close_px = tr.get("paper_exit_px")
        if close_px is None:
            try:
                close_px = self.tb.price(sym)
            except Exception:
                close_px = tr["entry"]
        mult = float(self.tb.specs()[sym]["multiplier"])
        pnl = (float(close_px) - float(tr["entry"])) * sgn * float(tr["contracts"]) * mult
        r_result = round(pnl / tr["risk_usdt"], 3) if tr.get("risk_usdt") else None
        reason = tr.get("exit_reason") or "模擬盤結束"
        log_event("EXIT", trade_id=tid, symbol=sym, side=side, reason=reason, paper=True,
                  entry=tr["entry"], close_avg=close_px, realized_pnl_usdt=round(pnl, 4), r_result=r_result,
                  held_min=round((time.time() - tr["opened_ts"]) / 60, 1))
        self.st["closed"].append({"id": tid, "symbol": sym, "side": side, "path": tr.get("path"),
                                  "pnl": round(pnl, 4), "r_result": r_result, "closed_at": now_iso(),
                                  "reason": reason, "paper": True, "listing": tr.get("listing")})
        self.st["trades"].pop(tid, None)

    def _end_paper(self):
        for tid, tr in list(self.st["trades"].items()):
            if not tr.get("paper"):
                continue
            if not tr.get("exit_reason"):
                tr["exit_reason"] = "模擬一天結束，按現價結算，未送真單"
                try:
                    tr["paper_exit_px"] = self.tb.price(tr["tsym"])
                except Exception:
                    tr["paper_exit_px"] = tr["entry"]
            self._paper_finish(tid, tr)
        go_live = os.environ.get("GATE_LIVE_AFTER_SIM") == "1"
        log_event("SIM_DONE", exchange="gate", live_next=go_live,
                  note=("Gate 模擬結束，模擬倉已按現價結算；使用者指示轉實盤，重啟後送真單" if go_live
                        else "Gate 模擬一天結束，程式停止，不會自動改送真單"))
        save_state(self.st)
        if not go_live:
            (ROOT / "logs" / "gate" / "SIM_ENDED").write_text(now_iso() + "\n")
        os._exit(0)

    def finish(self, tid: str, tr: dict):
        if tr.get("paper"):
            self._paper_finish(tid, tr)
            return
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
        elif pnl is not None and pnl > 0:
            reason = f"交易所止盈掛單成交後平倉 (平均出場價 {close_px}，未即時追蹤到各 TP 成交)"
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
        if tr.get("source") != ADOPT_SOURCE:
            self.learner.on_close(tr, r_result, pnl, reason, filled, round((time.time() - tr["opened_ts"]) / 60, 1))

    # ------------------------------------------------------------------ entries
    def bounce_blocked(self, sig: dict, htf: float | None) -> str | None:
        sl_pct = abs(float(sig["entry_price"]) - float(sig["stop_price"])) / float(sig["entry_price"])
        if not S.bounce_ok(htf, sl_pct):
            return (f"反彈多單條件已不成立：大盤 1h/4h {htf if htf is not None else '未知'}（須 < {S.BOUNCE_HTF_MAX}），"
                    f"止損 {sl_pct:.2%}（須 ≥ {S.BOUNCE_MIN_SL_PCT:.1%}）")
        held = sum(1 for t in self.st["trades"].values() if t.get("path") == S.BOUNCE_PATH)
        if held >= S.BOUNCE_MAX_OPEN:
            return f"反彈多單同時最多 {S.BOUNCE_MAX_OPEN} 筆，已有 {held} 筆"
        # Dust flats are not a read on the setup. On Nov 11 00:16-07:05 Taipei the
        # day-start equity line closed full-size bounces within a minute. Those scratches
        # sat in this window and erased PIEVERSE (-0.90R), so the path kept firing.
        decisive = [c for c in self.st["closed"] if c.get("path") == S.BOUNCE_PATH
                    and "小倉" not in (c.get("reason") or "")]
        last = decisive[-S.BOUNCE_STOP_AFTER_LOSSES:]
        if len(last) == S.BOUNCE_STOP_AFTER_LOSSES and all((c.get("r_result") or 0) <= S.BOUNCE_LOSS_R for c in last):
            return f"反彈多單實盤已連續 {S.BOUNCE_STOP_AFTER_LOSSES} 筆虧損，停用"
        return None

    def open_trade(self, sig: dict, ev: dict, source: str, regime: tuple, equity: float) -> bool:
        sym, side = S.to_toobit(sig["symbol"]), sig["side"]
        if S.SKIP_GATE_ONLY and getattr(self.tb, "name", "") == "gate" and listing(sym) == "only":
            self.note_skip(sig, "只在 Gate 上市的幣先不開：11-09 COOL、FATCOIN、FLORK、ZZZ 4 筆全虧")
            return False
        spec = self.tb.specs()[sym]
        price = self.tb.price(sym)
        stop = float(sig["stop_price"])
        if (side == "LONG" and price <= stop * 1.001) or (side == "SHORT" and price >= stop * 0.999):
            self.note_skip(sig, f"現價 {price} 已貼近/穿越止損 {stop}")
            return False
        risk_pct = max(S.MIN_RISK_PCT, self.learner.p["risk_pct"])
        margin_cap = S.NORMAL_MARGIN_CAP_PCT
        high_open = sum(1 for t in self.st["trades"].values() if t.get("tier") == "HIGH")
        tier = "HIGH" if ev.get("tier") == "HIGH" and high_open < S.MAX_HIGH_POSITIONS else "NORMAL"
        if tier == "HIGH" and not self.learner.p.get("high_size_ok"):
            ev["reasons"] = ev["reasons"] + ["高機率評估的實測先到 TP1 比例未優於一般評估，高信心單以一般風險開倉"]
        elif tier == "HIGH":
            risk_pct = S.HIGH_RISK_PCT
            margin_cap = S.HIGH_MARGIN_CAP_PCT
        room_pct = getattr(self, "day_room", equity * risk_pct) / equity
        if room_pct < S.MIN_RISK_PCT:
            self.note_skip(sig, f"今日剩餘虧損額度僅 {room_pct:.1%} 權益 (每日止損 {S.DAILY_STOP_PCT:.0%}，"
                                f"已計入持倉到止損的風險)，不足 {S.MIN_RISK_PCT:.0%} 最低風險")
            return False
        if room_pct < risk_pct:
            ev["reasons"] = ev["reasons"] + [f"風險 {risk_pct:.0%} 受每日止損剩餘額度限制為 {room_pct:.1%}"]
            risk_pct = room_pct
        sz = S.size_position(equity, price, stop, spec, risk_pct, margin_cap)
        if not sz:
            self.note_skip(sig, f"無法以至少 {S.MIN_RISK_PCT:.0%} 風險建立部位 (槓桿/保證金上限或最小下單量)，不開小倉")
            return False
        used = sum(t.get("margin") or 0 for t in self.st["trades"].values())
        room = equity * S.MAX_TOTAL_MARGIN_PCT - used
        if sz["margin"] > room:
            self.note_skip(sig, f"這筆保證金 {sz['margin']:.0f}u、{sz['leverage']}x，"
                                f"剩餘保證金額度 {max(0, room):.0f}u，總上限 {S.MAX_TOTAL_MARGIN_PCT:.0%}，再開不下")
            return False
        plan = S.tp_plan(sig, sz["contracts"], spec["min_contracts"])
        entry_info = dict(signal_id=sig["id"], source=source, symbol=sym, side=side, price=price,
                          stop=stop, sizing=sz, tp_plan=plan, score=ev["score"], regime=regime,
                          path=sig.get("signal_path"), subtype=sig.get("path_subtype"),
                          timeframe=sig.get("timeframe"), reasons=ev["reasons"], factors=ev["factors"],
                          learned_min_score=self.learner.p["min_score"], risk_pct=risk_pct, tier=tier,
                          projected_profit_usdt=self.projected(sig, plan, price, spec))
        if self.dry:
            log_event("DRY_ENTRY", **entry_info)
            return True
        if self.paper:
            self.st["traded_ids"].append(sig["id"])
            fill = {"filled": sz["contracts"], "maker": sz["contracts"], "taker": 0, "order_ids": [],
                    "note": "模擬盤，未送交易所"}
            pos = {"avgPrice": price, "position": sz["contracts"], "leverage": sz["leverage"],
                   "margin": sz["margin"]}
            self.record_entry({"sig": sig, "ev": ev, "source": source, "regime": list(regime), "tier": tier,
                               "sz": sz, "stop": stop, "price": price,
                               "entry_info": entry_info,
                               "ctx": {"m15": getattr(self, "m15", None), "htf": getattr(self, "htf", None),
                                       "ext": getattr(self, "ext", None), "pre": getattr(self, "pre", None),
                                       "rpos": getattr(self, "rpos", None), "dirs": getattr(self, "dirs", None),
                                       "rsi": getattr(self, "rsi", None)}}, fill, pos)
            return True

        self.tb.set_isolated(sym)
        self.tb.set_leverage(sym, sz["leverage"])
        self.st["traded_ids"].append(sig["id"])
        # Journal the entry before any order goes out: if the process dies while the maker order rests, the
        # restart adopts whatever filled (stop + TPs) instead of leaving a naked position.
        self.st["pending_entry"] = {
            "sig": sig, "ev": {k: ev.get(k) for k in ("analysis", "factors", "cur_r", "score")},
            "source": source, "regime": list(regime), "tier": tier, "sz": sz, "stop": stop, "price": price,
            "entry_info": entry_info, "ts": time.time(),
            "ctx": {"m15": getattr(self, "m15", None), "htf": getattr(self, "htf", None), "ext": getattr(self, "ext", None),
                    "pre": getattr(self, "pre", None), "rpos": getattr(self, "rpos", None), "dirs": getattr(self, "dirs", None),
                    "rsi": getattr(self, "rsi", None)},
        }
        save_state(self.st)
        # An exception below leaves pending_entry in place; the next cycle's recover_pending_entry() adopts it.
        fill = self.maker_entry(sym, side, sz["contracts"], stop, sig, spec, price)
        pos = next((p for p in self.tb.positions(sym) if p.get("side") == side), None)
        if not pos:
            log_event("SKIP" if not fill["filled"] else "ERROR", signal_id=sig["id"], symbol=sym, side=side,
                      reason=fill.get("note") or "下單後查無持倉", fill=fill)
        else:
            self.record_entry(self.st["pending_entry"], fill, pos)
        self.st.pop("pending_entry", None)
        save_state(self.st)
        return bool(pos)

    def record_entry(self, p: dict, fill: dict, pos: dict):
        sig, ev, sz, stop = p["sig"], p["ev"], p["sz"], p["stop"]
        sym, side = S.to_toobit(sig["symbol"]), sig["side"]
        spec = self.tb.specs()[sym]
        entry = float(pos.get("avgPrice") or p["price"])
        contracts = fill["filled"] or sz["contracts"]
        plan = S.tp_plan(sig, contracts, spec["min_contracts"])
        tps = self.place_tps(sym, side, p.get("tp_ref", entry), plan)
        sz = {**sz, "contracts": contracts}
        entry_fee = S.MAKER_FEE if fill["maker"] >= fill["taker"] else S.TAKER_FEE
        tid = f"{sym}:{side}:{int(time.time())}"
        mult = float(spec["multiplier"])
        ctx = p["ctx"]
        self.st["trades"][tid] = {
            "id": tid, "signal_id": sig["id"], "source": p["source"], "tsym": sym, "side": side,
            "contracts": sz["contracts"], "entry": entry, "stop": stop, "tps": tps,
            "risk_usdt": round(sz["contracts"] * mult * abs(entry - stop), 4), "margin": sz["margin"],
            "leverage": sz["leverage"], "path": sig.get("signal_path"), "opened_ts": time.time(),
            "be_buffer": entry_fee + S.TAKER_FEE + SLIPPAGE_BUFFER, "entry_fill": fill,
            "after_red": bool(sig.get("after_red")),
            "recipe_spike": bool(sig.get("recipe_spike")),
            "analysis": ev.get("analysis"), "analysis_bias": (ev.get("analysis") or {}).get("bias", 0.0),
            "tier": p["tier"], "m15_at_entry": ctx["m15"], "htf_at_entry": ctx["htf"], "ext_r_at_entry": ctx["ext"],
            "pre_signal_r": ctx["pre"], "range_pos_at_entry": ctx["rpos"], "dirs_at_entry": ctx["dirs"],
            "rsi15_at_entry": ctx.get("rsi"),
            "break_level": sig.get("broken"),
            "same_side_at_entry": sum(1 for t in self.st["trades"].values() if t["side"] == side),
            "factors": ev["factors"], "entry_cur_r": ev["cur_r"], "score": ev["score"], "regime": p["regime"],
            "listing": listing(sym),
            "paper": self.paper,
        }
        log_event("ENTRY", trade_id=tid, entry_avg=entry, order_id=fill.get("order_ids"),
                  listing=self.st["trades"][tid]["listing"], paper=bool(self.paper),
                  tp_orders=[(t["label"], t["price"], t["qty"], t["order_id"]) for t in tps], **p["entry_info"])
        if hasattr(self, "day_room"):
            self.day_room -= self.st["trades"][tid]["risk_usdt"]

    def reconcile(self, positions: dict):
        """Every RECONCILE_S: every exchange position must be managed by the bot and carry an exchange stop."""
        if time.time() - getattr(self, "last_reconcile", 0) < RECONCILE_S:
            return
        self.last_reconcile = time.time()
        tracked = {(t["tsym"], t["side"]): t for t in self.st["trades"].values()}
        for (sym, side), pos in positions.items():
            tr = tracked.get((sym, side))
            try:
                if tr is None:
                    self.adopt_external(sym, side, pos)
                elif not tr.get("exit_reason") and not any(
                        str(o.get("clientOrderId", "")).startswith(f"STOP_{side}_LOSS")
                        for o in self.tb.open_orders(sym, "STOP_PROFIT_LOSS")):
                    self.tb.set_position_stop(sym, side, self.tb.fmt_price(sym, tr["stop"]))
                    log_event("STOP_RESTORED", trade_id=tr["id"], symbol=sym, side=side, stop=tr["stop"],
                              reason="每分鐘巡檢發現交易所上沒有止損，已補掛")
            except ToobitError as e:
                log_event("ERROR", where="reconcile", symbol=sym, side=side, error=str(e))
                if tr is not None and "stop" in str(e).lower():
                    tr["exit_reason"] = "巡檢補掛止損失敗 (價格可能已穿越止損)，平倉"
                    self.maker_close(tr)

    def adopt_external(self, sym: str, side: str, pos: dict):
        """Take over a position the bot did not open: See One-style ATR stop and TP ladder from the current price."""
        qty = int(float(pos["position"]))
        px = self.tb.price(sym)
        pl = self.selfscan.plan(sym, side)
        dist = pl["dist"] if pl else ADOPT_MAX_STOP_PCT * px
        sgn = 1 if side == "LONG" else -1
        stop = px - sgn * dist
        for kind in ("LIMIT", "STOP"):
            for o in self.tb.open_orders(sym, kind):
                if str(o.get("side", "")).endswith("CLOSE"):
                    self.cancel_quiet(str(o["orderId"]), kind)
        try:
            self.tb.set_position_stop(sym, side, self.tb.fmt_price(sym, stop))
        except ToobitError as e:
            log_event("ADOPT_CLOSE", symbol=sym, side=side, contracts=qty, error=str(e), reason="接管時無法設定止損，直接平倉")
            self.tb.flash_close(sym, side)
            return
        so_sym = sym.replace("-SWAP-USDT", "USDT")
        sig = {"id": f"adopt:{so_sym}:{side}:{int(time.time())}", "symbol": so_sym, "side": side, "entry_price": px,
               "stop_price": stop, "signal_path": "ADOPTED", "levels": [
                   {"label": f"TP{i + 1}", "kind": "TP", "sequence": i + 1, "r_multiple": r, "price": px + sgn * dist * r}
                   for i, r in enumerate(ADOPT_TP_R)]}
        reason = (f"接管非機器人開立的持倉 {qty} 張 (均價 {pos.get('avgPrice')}、{pos.get('leverage')}x)：止損設於現價 "
                  f"{px} 外 {dist / px:.2%} (1.5×ATR15m)，止盈 {ADOPT_TP_R[0]}R～{ADOPT_TP_R[-1]}R，之後依 See One 分析驗算重評")
        p = {"sig": sig, "ev": {"analysis": None, "factors": {}, "cur_r": 0.0, "score": None},
             "source": ADOPT_SOURCE, "regime": [0.0, "N/A"], "tier": "NORMAL", "stop": stop, "price": px, "tp_ref": px,
             "sz": {"contracts": qty, "margin": float(pos.get("margin") or 0), "leverage": int(float(pos.get("leverage") or 1))},
             "entry_info": {"signal_id": sig["id"], "source": ADOPT_SOURCE, "symbol": sym, "side": side, "price": px,
                            "stop": stop, "reasons": [reason]},
             "ctx": {"m15": None, "htf": None, "ext": None, "pre": None, "rpos": None, "dirs": None}}
        log_event("ADOPT", symbol=sym, side=side, contracts=qty, entry=pos.get("avgPrice"), stop=stop, reason=reason)
        self.record_entry(p, {"filled": qty, "maker": qty, "taker": 0, "order_ids": [], "note": "接管"}, pos)
        save_state(self.st)

    def recover_pending_entry(self):
        """After a crash/kill mid-entry: cancel the resting maker entry and adopt any filled position."""
        p = self.st.get("pending_entry")
        if not p:
            return
        sym, side = S.to_toobit(p["sig"]["symbol"]), p["sig"]["side"]
        for o in self.tb.open_orders(sym, "LIMIT"):
            if str(o.get("clientOrderId", "")).startswith("mo"):
                self.cancel_quiet(str(o["orderId"]), "LIMIT")
        time.sleep(1)
        pos = next((x for x in self.tb.positions(sym) if x.get("side") == side), None)
        tracked = any(t["tsym"] == sym and t["side"] == side for t in self.st["trades"].values())
        if pos and not tracked:
            qty = int(float(pos["position"]))
            fill = {"filled": qty, "maker": qty, "taker": 0, "order_ids": [],
                    "note": "程式中斷時進場掛單已成交，重啟後接管並補掛止損/止盈"}
            try:
                self.tb.set_position_stop(sym, side, self.tb.fmt_price(sym, p["stop"]))
            except ToobitError as e:
                log_event("ADOPT_CLOSE", symbol=sym, side=side, contracts=qty, error=str(e),
                          reason="接管時無法設定止損 (價格可能已穿越止損)，直接平倉")
                self.tb.flash_close(sym, side)
            else:
                log_event("ADOPT", symbol=sym, side=side, contracts=qty, entry=pos.get("avgPrice"), stop=p["stop"],
                          reason=fill["note"])
                self.record_entry(p, fill, pos)
        self.st.pop("pending_entry", None)
        save_state(self.st)

    def _entry_ok(self, px: float, sig: dict, stop: float) -> bool:
        side, entry = sig["side"], float(sig["entry_price"])
        risk = abs(entry - stop)
        cur_r = (px - entry) / risk * (1 if side == "LONG" else -1) if risk else 0
        away = px > stop * 1.001 if side == "LONG" else px < stop * 0.999
        if sig.get("broken") is not None and S.break_half_retrace(side, float(sig["broken"]), entry, px):
            return False
        hi = self.learner.p["entry_r_window"][1]
        if str(sig.get("id", "")).startswith("self:"):
            hi = min(hi, S.SELF_ENTRY_CHASE_MAX_R)
        return away and cur_r <= hi

    def maker_entry(self, sym: str, side: str, contracts: int, stop: float, sig: dict, spec: dict,
                    ref_px: float | None = None) -> dict:
        """Work the entry as post-only (maker) at the touch; fall back to market only while still in the entry band."""
        open_side = "BUY_OPEN" if side == "LONG" else "SELL_OPEN"
        res = {"filled": 0, "maker": 0, "taker": 0, "order_ids": [], "note": None}
        stop_px = self.tb.fmt_price(sym, stop)
        stop_set = False
        # Size was set from ref_px. A maker order that follows the touch away from the stop
        # must buy fewer contracts, or the loss at the fixed stop grows past the planned risk
        # (HAEDAL: sized at 0.02729, filled at 0.02763, 1.59× the planned loss).
        budget = contracts * abs(ref_px - stop) if ref_px else None
        spent = 0.0
        plan_px = float(sig.get("entry_price") or ref_px or 0)
        planned = abs(plan_px - stop)

        def cap(left: int, px: float) -> int:
            if budget is None or abs(px - stop) <= 0:
                return left
            return min(left, int((budget - spent) / abs(px - stop)))

        def into_stop(px: float) -> bool:
            return planned > 0 and abs(px - stop) < planned * STOP_KEEP_FRAC

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
            if into_stop(px):
                res["note"] = (f"掛單價 {px} 距止損只剩規劃的 {abs(px - stop) / planned:.0%}，"
                               f"低於 {STOP_KEEP_FRAC:.0%}，不再往止損裡追")
                break
            if not self._entry_ok(px, sig, stop):
                res["note"] = f"掛單期間價格 {px} 已離開進場區間，不追價"
                break
            left = cap(left, px)
            if left < spec["min_contracts"]:
                res["note"] = f"價格 {px} 離止損較遠，規劃風險已用完，不再加倉"
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
                spent += ex * abs((float(q.get("avgPrice") or 0) or px) - stop)
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
            left = cap(left, px)
            if into_stop(px):
                res["note"] = (f"市價 {px} 距止損只剩規劃的 {abs(px - stop) / planned:.0%}，"
                               f"低於 {STOP_KEEP_FRAC:.0%}，不補單")
            elif self._entry_ok(px, sig, stop) and left >= spec["min_contracts"]:
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

    def maker_close(self, tr: dict, urgent: bool = False):
        """Exit a whole position with post-only orders at the touch; market only for whatever is left after the retries.
        The exchange-side stop stays in place the whole time.

        urgent skips the post-only retries: a quote resting at the touch while price runs
        against it does not fill. Toobit -0.85R exits waited 34-336 s that way and closed at
        -1.0R to -1.21R (ENA, BNT, IREN, IRYS, USELESS, REZ), worse than the stop itself."""
        if tr.get("paper"):
            try:
                tr["paper_exit_px"] = self.tb.price(tr["tsym"])
            except Exception:
                tr["paper_exit_px"] = tr.get("paper_exit_px") or tr["entry"]
            log_event("EXIT_FILL", trade_id=tr["id"], symbol=tr["tsym"], side=tr["side"], paper=True,
                      price=tr["paper_exit_px"], reason=tr.get("exit_reason"))
            return
        sym, side = tr["tsym"], tr["side"]
        for tp in tr["tps"]:
            if not tp["filled"] and tp.get("order_id"):
                self.cancel_quiet(tp["order_id"], tp.get("kind", "STOP"))
                tp["cancelled"] = True
        close_side = "SELL_CLOSE" if side == "LONG" else "BUY_CLOSE"
        maker = taker = 0
        for _ in range(0 if urgent else MAKER_CLOSE_TRIES):
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
        if self.paper:
            tps = []
            for tp in plan:
                if (tp["price"] <= entry) if side == "LONG" else (tp["price"] >= entry):
                    continue
                tps.append({**tp, "price": float(self.tb.fmt_price(sym, tp["price"])), "order_id": None,
                            "kind": "PAPER", "filled": False})
            return tps
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

    def _margin_full(self, equity: float) -> bool:
        used = sum(t.get("margin") or 0 for t in self.st["trades"].values())
        return used >= equity * S.MAX_TOTAL_MARGIN_PCT - 1

    def open_risk(self) -> float:
        specs = self.tb.specs()
        total = 0.0
        for t in self.st["trades"].values():
            sgn = 1 if t["side"] == "LONG" else -1
            mult = float(specs.get(t["tsym"], {}).get("multiplier", 0))
            total += t["contracts"] * mult * max(0.0, (t["entry"] - t["stop"]) * sgn)
        return total

    def pre_move(self, sig: dict) -> float | None:
        if sig["id"] not in self.pre_moves:
            try:
                end = int(datetime.fromisoformat(sig["entry_at"].replace("Z", "+00:00")).timestamp() * 1000)
                risk = abs(float(sig["entry_price"]) - float(sig["stop_price"]))
                self.pre_moves[sig["id"]] = move_before(S.to_toobit(sig["symbol"]), end, sig["side"], risk)
            except Exception as e:
                log_event("ERROR", where="pre_move", symbol=sig["symbol"], error=repr(e)[:200])
                return None
        return self.pre_moves[sig["id"]]

    def record_signal_snapshots(self, signals: list):
        """Score every fresh official signal for the threshold calibration, including cycles where no new entry is
        allowed (book full, daily lock, halt, pause). snapshot() keeps only signals seen within 60 minutes of entry."""
        try:
            dirs = self.so.directions()
            reg = S.regime(dirs)
            board = {b["symbol"]: b for b in self.so.strength_board()}
            rhits = S.recipe_hits(self.so.recipes())
            pexp = S.path_expectancy(signals, self.st["closed"])
        except Exception as e:
            log_event("ERROR", where="snapshot", error=repr(e)[:200])
            return
        for sig in signals:
            if sig.get("status") == "ACTIVE":
                ev = S.evaluate(sig, "官方訊號", reg[0], reg[1], board, rhits, pexp, self.learner.p)
                self.shadow.snapshot(sig, ev, S._age_min(sig["entry_at"]))

    def note_skip(self, sig: dict, why: str):
        key = sig["id"]
        kind = re.sub(r"[-+]?\d+(\.\d+)?", "#", why)
        if self.st["skips"].get(key) != kind:
            self.st["skips"][key] = kind
            log_event("SKIP", signal_id=key, symbol=sig["symbol"], side=sig["side"], reason=why)

    # ------------------------------------------------------------------ cycle
    def cycle(self):
        if not self.dry:
            self.recover_pending_entry()
        try:
            self.so.ensure_session()
        except Exception as e:
            log_event("ERROR", where="seeone_session", error=repr(e)[:200])
        try:
            signals = self.so.signals()
        except Exception as e:
            signals = None
            if not isinstance(e, S.SeeOneDown):
                log_event("ERROR", where="signals", error=repr(e)[:200], note="See One 暫時無法取得訊號，本輪只管理持倉、不開新倉")
        sigmap = {s["id"]: s for s in signals} if signals is not None else None
        if signals is not None:
            self.shadow.ingest(signals)
        if signals is not None and time.time() - self.last_history > 3600:
            self.last_history = time.time()
            try:
                self.shadow.ingest(self.so.history())
            except Exception as e:
                log_event("ERROR", where="history", error=repr(e)[:200])
        self.shadow.optimize()
        if self.paper:
            until = os.environ.get("GATE_SIM_UNTIL") or ""
            try:
                deadline = datetime.fromisoformat(until)
                if deadline.tzinfo is None:
                    deadline = deadline.replace(tzinfo=timezone.utc)
            except ValueError:
                deadline = None
            if deadline and datetime.now(timezone.utc) >= deadline:
                self._end_paper()
            positions = self._paper_positions()
            self.manage(positions, sigmap)
        else:
            positions = {(p["symbol"], p.get("side")): p for p in self.tb.positions()}

        if not self.dry and not self.paper:
            try:
                self._mark_equity, _ = self.equity()
            except Exception:
                self._mark_equity = self.st.get("day_start_equity") or 0
            self.manage(positions, sigmap)
            positions = {(p["symbol"], p.get("side")): p for p in self.tb.positions()}
            self.reconcile(positions)

        equity, _ = self.equity()
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
            for k in ("halted", "day_lock_armed", "day_lock_hit", "day_peak_equity", "day_lock_ts", "day_target_done",
                      "day_target_hit", "day_peak_gain"):
                self.st.pop(k, None)
        self.st.setdefault("principal", equity)
        # A deposit is not trading profit. Gate's 151u deposit on Nov 10 stayed out of principal,
        # so half of it was kept out of position size and each new trade risked about 4.4u
        # instead of 5% of the account. Principal only moves up.
        cutoff = os.environ.get("GATE_SIM_UNTIL") or "" if getattr(self.tb, "name", "") == "gate" else ""
        booked = sum((c.get("pnl") or 0) for c in self.st["closed"]
                     if not cutoff or (c.get("closed_at") or "") >= cutoff)
        capital = self.realized - booked
        if capital > self.st["principal"] + 1:
            added = capital - self.st["principal"]
            self.st["principal"] = round(capital, 2)
            log_event("PRINCIPAL_DEPOSIT", added=round(added, 2), principal=self.st["principal"],
                      balance=round(self.realized, 2),
                      reason="餘額高於本金加已平倉損益，差額視為入金並計入本金；之後開倉用全部權益")
        # The profit target and lock count today's closed-trade PnL, not the balance change: a 151u Gate deposit
        # on Nov 10 read as 151u profit and stopped new entries for the day.
        self.st.pop("day_target_done", None)
        start = self.st["day_start_balance"]
        gained = sum(c.get("pnl") or 0 for c in self.st["closed"] if (c.get("closed_at") or "").startswith(day))
        peak_gain = self.st["day_peak_gain"] = max(self.st.get("day_peak_gain", 0.0), gained)
        peak = start + peak_gain
        override = LOCK_OFF_FILE.exists() and LOCK_OFF_FILE.read_text().strip() == day
        if override and self.st.get("day_lock_off_logged") != day:
            for k in ("day_lock_armed", "day_lock_hit", "day_lock_ts"):
                self.st.pop(k, None)
            self.st["day_lock_off_logged"] = day
            log_event("DAY_LOCK_OVERRIDE", day=day,
                      reason="使用者指示今日照常開一般倉：解除當日鎖利與回吐停止 (UTC 日界前有效)；當日虧損上限仍有效")
        if not override and gained >= S.DAY_TARGET_CAP_USDT and not self.st.get("day_target_hit"):
            self.st["day_target_hit"] = True
            log_event("DAY_TARGET_DONE", day_start_balance=round(start, 2), balance=round(bal, 2),
                      gained=round(gained, 2),
                      reason=f"當日已實現獲利 {gained:.0f}u 達到目標上限 {S.DAY_TARGET_CAP_USDT:.0f}u，今日停止開新倉")
        if self.st.get("day_target_hit"):
            if signals is not None:
                self.record_signal_snapshots(signals)
            return
        if not override and peak - start >= S.DAY_TARGET_ARM_USDT and not self.st.get("day_lock_armed"):
            self.st["day_lock_armed"] = True
            self.st["day_lock_ts"] = time.time()
            log_event("DAY_LOCK_ARMED", day_start_balance=round(start, 2), balance=round(bal, 2),
                      reason=f"當日已實現獲利達 {S.DAY_TARGET_ARM_USDT:.0f}u。進場仍用同一套規則，"
                             f"一路放到 {S.DAY_TARGET_CAP_USDT:.0f}u 才停止開新倉")
        loss_floor = self.st["day_start_equity"] * (1 - S.DAILY_STOP_PCT)
        if RISK_OPEN_FILE.exists():
            if self.st.get("risk_open_logged") != day:
                self.st.pop("halted", None)
                self.st["risk_open_logged"] = day
                log_event("RISK_OPEN", day=day, equity=round(equity, 2),
                          day_start_equity=self.st.get("day_start_equity"),
                          reason="使用者指示額度全開：不因當日虧損上限停手，單筆仍照原風險、原進場規則")
            self.day_room = equity
        elif equity < loss_floor:
            if not self.st.get("halted"):
                self.st["halted"] = True
                log_event("HALT", reason=f"當日虧損達 {S.DAILY_STOP_PCT:.0%}，今日停止開新倉",
                          day_start_equity=self.st["day_start_equity"], equity=equity)
            if signals is not None:
                self.record_signal_snapshots(signals)
            return
        else:
            self.day_room = equity - loss_floor - self.open_risk()
        if sigmap is None:
            return
        if PAUSE_FILE.exists() or time.time() < self.learner.p.get("pause_until", 0):
            self.record_signal_snapshots(signals)
            return
        if self._margin_full(equity):
            self.record_signal_snapshots(signals)
            return

        dirs = self.dirs = self.so.directions()
        reg = S.regime(dirs)
        self.st["market"] = {"ts": now_iso(), "dirs": dirs, "regime": list(reg), "htf": S.htf_score(dirs)}
        board_rows = self.so.strength_board()
        board = {b["symbol"]: b for b in board_rows}
        rhits = S.recipe_hits(self.so.recipes())
        pexp = S.path_expectancy(signals, self.st["closed"])
        specs = self.tb.specs()
        # Each venue publishes its own contract list. A coin listed only on the other exchange is not traded here.
        self.st["tradable_contracts"] = len(specs)
        held_syms = {s for s, _ in positions} | {t["tsym"] for t in self.st["trades"].values()}

        cands = [(s, "官方訊號") for s in signals if s.get("status") == "ACTIVE"]
        cands += [(b, "強勢榜") for b in board_rows if b.get("id") and b["id"] not in sigmap]
        scored = []
        for sig, src in cands:
            if sig["id"] in self.st["traded_ids"]:
                continue
            tsym = S.to_toobit(sig["symbol"])
            if tsym not in specs:
                venue = "Gate" if getattr(self.tb, "name", "") == "gate" else "Toobit"
                self.note_skip(sig, f"{venue} 無 {tsym} 合約")
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

        if self._scan_th is None or not self._scan_th.is_alive():
            self._scan_th = threading.Thread(target=self._scan_worker, name="self-scan", daemon=True,
                                             args=(reg, rhits, set(held_syms), specs, dirs.get("15m"), S.htf_score(dirs)))
            self._scan_th.start()
        with self._scan_lock:
            found, self._scan_out = self._scan_out, []
        for at, f, src in found:
            if time.time() - at > SCAN_RESULT_TTL_S:
                continue
            if f["sig"]["id"] in self.st["traded_ids"] or S.to_toobit(f["sig"]["symbol"]) in held_syms:
                continue
            scored.append((f["ev"]["score"], f["sig"], f.get("source", src), f["ev"]))

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
        htf = self.htf = S.htf_score(dirs)
        htf_min = self.learner.p.get("htf_min", S.HTF_MIN_DEFAULT)
        htf_max = self.learner.p.get("htf_max", S.HTF_MAX_DEFAULT)
        try:
            btc1h = btc_ret1h()
        except Exception:
            btc1h = None
        moves = {}
        for score, sig, src, ev in sorted(scored, key=lambda x: -x[0]):
            if self._margin_full(size_eq):
                break
            sgn = 1 if sig["side"] == "LONG" else -1
            cd = self.learner.p.get("symbol_cooldown", {}).get(sig["symbol"], 0)
            if cd > time.time():
                self.note_skip(sig, f"{sig['symbol']} 近期虧損冷卻中 (至台北 "
                                    f"{datetime.fromtimestamp(cd + 8 * 3600, timezone.utc):%H:%M})")
                continue
            self.pre = None
            if S.to_toobit(sig["symbol"]) == "ONE-SWAP-USDT" and not sig.get("recipe_spike"):
                self.note_skip(sig, "ONE 這筆提前平倉後不接回，改開其他幣")
                continue
            if not sig.get("breakout") and sig["symbol"] in MAJORS:
                self.note_skip(sig, "主流幣不靠自選單進場。今天 SOL、XRP 都在驗算轉弱時虧損出場，WLD、XMR 這種榜上的空單有打到 TP1")
                continue
            if sig.get("recipe_spike"):
                same = [t for t in self.st["trades"].values() if t["side"] == sig["side"]]
                if len(same) >= S.MAX_SAME_SIDE:
                    self.note_skip(sig, f"同方向已有 {len(same)} 筆持倉，避免同一方向集中押注")
                    continue
                try:
                    self.open_trade(sig, ev, src, reg, size_eq)
                except ToobitError as e:
                    self.st["traded_ids"].append(sig["id"])
                    log_event("ERROR", where="open_trade", symbol=sig["symbol"], error=str(e))
                continue
            if sig.get("breakout"):
                same = [t for t in self.st["trades"].values() if t["side"] == sig["side"]]
                if len(same) >= S.MAX_SAME_SIDE:
                    self.note_skip(sig, f"同方向已有 {len(same)} 筆持倉，避免同一方向集中押注")
                    continue
                if sig["side"] == "LONG" and not sig.get("after_red"):
                    self.note_skip(sig, "突破做多先停。BMT 平在 -0.18R、ERA 平在 -0.55R，之後價格還在跌")
                    continue
                if not sig.get("after_red") and htf is not None and htf * sgn < htf_min:
                    self.note_skip(sig, f"1h/4h 大方向 {htf:+.1f} 與{'做多' if sgn > 0 else '做空'}相反 "
                                        f"(順方向 {htf * sgn:+.1f} < 下限 {htf_min:+.1f})，不做逆大勢的單")
                    continue
                brk_max = (max(htf_max, S.BREAK_SHORT_HTF_MAX_PAPER)
                           if self.paper and sig["side"] == "SHORT" else htf_max)
                if not sig.get("after_red") and htf is not None and htf * sgn >= brk_max:
                    self.note_skip(sig, f"1h/4h 大方向 {htf:+.1f} 已強烈偏{'多' if sgn > 0 else '空'} "
                                        f"(順方向 {htf * sgn:+.1f} ≥ 上限 {brk_max:+.1f})，大週期走勢已走一段，進場偏晚")
                    continue
                if not sig.get("after_red") and m15 * sgn >= m15_max:
                    self.note_skip(sig, f"See One 15m 方向 {m15:+.1f}，順勢{'做多' if sgn > 0 else '做空'}動能已達 "
                                        f"{m15 * sgn:+.1f} (上限 {m15_max:+.1f})，追在走勢中段/末端，等回檔或盤整")
                    continue
                if not sig.get("after_red"):
                    aligned = None if btc1h is None else btc1h * sgn
                    if not S.btc_dir_ok(aligned, sig["side"]):
                        self.note_skip(sig, f"BTC 近 1 小時 {btc1h:+.2f}%，與{'做多' if sgn > 0 else '做空'}方向不同步 "
                                            f"(順方向 {aligned:+.2f}% < {S.BTC1H_MIN_DEFAULT:+.2f}%)，等大盤同向再進")
                        continue
                    if aligned is not None and aligned >= S.BTC1H_CHASE_MAX:
                        self.note_skip(sig, f"BTC 近 1 小時已順{'做多' if sgn > 0 else '做空'}方向走 {aligned:+.2f}% "
                                            f"(≥ {S.BTC1H_CHASE_MAX:+.1f}%)，突破追在大盤急走之後：實單 6 筆只贏 1 筆，-1.88R")
                        continue
                # 1h/4h can still sit inside the entry band while 15m has already pushed the blend
                # into a bear regime. Both BEAR breakout shorts (ENA, IREN) were stopped out.
                if sig["side"] == "SHORT" and reg[0] <= S.REGIME_BEAR:
                    self.note_skip(sig, f"綜合方向 {reg[0]:+.1f} 已是空頭區間 (≤ {S.REGIME_BEAR:+.1f})，"
                                        f"15 分鐘跌破發生在大勢已空之後。ENA、IREN 這種單直接打到止損")
                    continue
                try:
                    if sig.get("broken") is not None and S.break_half_retrace(
                            sig["side"], float(sig["broken"]), float(sig["entry_price"]),
                            self.tb.price(S.to_toobit(sig["symbol"]))):
                        self.note_skip(sig, "突破後價格已收回區間一半以上，不追這根回檔")
                        continue
                    self.open_trade(sig, ev, src, reg, size_eq)
                except ToobitError as e:
                    self.st["traded_ids"].append(sig["id"])
                    log_event("ERROR", where="open_trade", symbol=sig["symbol"], error=str(e))
                continue
            # Self-scan shorts once the blend is already BEAR: 7 closes, net about -81 USDT.
            # The two winners (AKE, XRP on Nov 4) do not cover JCT, UAI, Q, HUMA and BNT.
            # Softer-bear shorts such as XMR stay eligible. Quiet shorts and the crash recipe do too.
            bounce = src == SELF_SOURCE and sig.get("signal_path") == S.BOUNCE_PATH
            if bounce:
                why = self.bounce_blocked(sig, htf)
                if why:
                    self.note_skip(sig, why)
                    continue
            elif src == SELF_SOURCE and sig["side"] not in self.selfscan.sides(htf, btc1h):
                self.note_skip(sig, "自選驗算做多未開放：" + self.selfscan.long_gate(htf, btc1h)[1])
                continue
            if src == SELF_SOURCE and not bounce:
                sl_pct = abs(float(sig["entry_price"]) - float(sig["stop_price"])) / float(sig["entry_price"])
                if sl_pct >= S.SELF_MAX_SL_PCT:
                    self.note_skip(sig, f"自選驗算止損 {sl_pct:.2%} ≥ {S.SELF_MAX_SL_PCT:.0%}：止損 1-1.8% 的評估只有 40-56% "
                                        f"先到 TP1，實盤 9 筆只贏 1 筆；1% 以下的空單 73-75%")
                    continue
            if src == SELF_SOURCE and sig["side"] == "SHORT" and reg[0] <= S.REGIME_BEAR:
                self.note_skip(sig, f"自選驗算做空時綜合方向 {reg[0]:+.1f} 已是空頭區間 (≤ {S.REGIME_BEAR:+.1f})，"
                                    f"這段實測淨虧，BNT 開倉 7 分鐘就打到止損")
                continue
            if src != SELF_SOURCE:
                self.pre = self.pre_move(sig)
                if self.pre is not None and self.pre >= S.FRESH_MAX_R:
                    self.note_skip(sig, f"See One 訊號發出前 2 小時本幣已順方向走 {self.pre:+.2f}R "
                                        f"(≥ {S.FRESH_MAX_R}R)，不是走勢前端，不追")
                    continue
            if m15 * sgn >= m15_max:
                self.note_skip(sig, f"See One 15m 方向 {m15:+.1f}，順勢{'做多' if sgn > 0 else '做空'}動能已達 "
                                    f"{m15 * sgn:+.1f} (上限 {m15_max:+.1f})，追在走勢中段/末端，等回檔或盤整")
                continue
            if htf is not None and htf * sgn < htf_min and not bounce:
                self.note_skip(sig, f"1h/4h 大方向 {htf:+.1f} 與{'做多' if sgn > 0 else '做空'}相反 "
                                    f"(順方向 {htf * sgn:+.1f} < 下限 {htf_min:+.1f})，不做逆大勢的單")
                continue
            if htf is not None and htf * sgn >= htf_max:
                self.note_skip(sig, f"1h/4h 大方向 {htf:+.1f} 已強烈偏{'多' if sgn > 0 else '空'} "
                                    f"(順方向 {htf * sgn:+.1f} ≥ 上限 {htf_max:+.1f})，大週期走勢已走一段，進場偏晚")
                continue
            aligned = None if btc1h is None else btc1h * sgn
            btc_min = S.SELF_BTC1H_MIN if src == SELF_SOURCE else S.BTC1H_MIN_DEFAULT
            # Bounce longs held 56-76% TP1-first in every BTC 1h bucket, so neither BTC bound applies to them.
            if aligned is not None and aligned < btc_min and not bounce:
                self.note_skip(sig, f"BTC 近 1 小時 {btc1h:+.2f}%，與{'做多' if sgn > 0 else '做空'}方向不同步 "
                                    f"(順方向 {aligned:+.2f}% < {btc_min:+.2f}%)，等大盤同向再進")
                continue
            if src == SELF_SOURCE and aligned is not None and aligned >= S.BTC1H_CHASE_MAX and not bounce:
                self.note_skip(sig, f"BTC 近 1 小時已順{'做多' if sgn > 0 else '做空'}方向走 {aligned:+.2f}% "
                                    f"(≥ {S.BTC1H_CHASE_MAX:+.1f}%)，自選單追在大盤急走之後，驗算只有 48% 先到 TP1")
                continue
            tsym = S.to_toobit(sig["symbol"])
            if tsym not in moves:
                try:
                    moves[tsym] = move_2h(tsym)
                except Exception:
                    moves[tsym] = None
            mv = moves[tsym]
            if mv and mv["r"] * sgn >= S.MAX_EXT_R:
                self.note_skip(sig, f"{sig['symbol']} 近 2 小時已{'漲' if mv['pct'] > 0 else '跌'} "
                                    f"{abs(mv['pct']):.1f}% ({mv['r'] * sgn:+.1f}R ≥ {S.MAX_EXT_R}R)，"
                                    f"{'做多' if sgn > 0 else '做空'}會追在走勢末端，等回檔再進")
                continue
            self.ext = mv["r"] * sgn if mv else None
            self.rpos = in_side(mv["rpos"], sig["side"]) if mv and mv.get("rpos") is not None else None
            if self.rpos is not None and self.rpos > S.MAX_RANGE_POS:
                self.note_skip(sig, f"{sig['symbol']} 現價已超出近 3 小時{'高點' if sgn > 0 else '低點'} "
                                    f"(區間位置 {self.rpos:.2f} > {S.MAX_RANGE_POS:.2f})，追在突破延伸段，等回到區間內再進")
                continue
            self.rsi = rsi = rsi_side(mv.get("rsi"), sig["side"]) if mv else None
            rsi_min = self.learner.p.get("rsi_min", S.RSI_MIN_DEFAULT)
            if rsi is not None and rsi < rsi_min:
                self.note_skip(sig, f"{sig['symbol']} 15m RSI 在{'做多' if sgn > 0 else '做空'}方向只有 {rsi:.0f} "
                                    f"(< 下限 {rsi_min:.0f})，短線動能與方向相反，等轉強再進")
                continue
            same = [t for t in self.st["trades"].values() if t["side"] == sig["side"]]
            same_cap = S.QUIET_MAX_SAME_SIDE if sig.get("quiet_strong_short") or sig.get("quiet_strong_long") else S.MAX_SAME_SIDE
            if len(same) >= same_cap:
                self.note_skip(sig, f"同方向已有 {len(same)} 筆持倉，避免同一方向集中押注")
                continue
            try:
                self.open_trade(sig, ev, src, reg, size_eq)
            except ToobitError as e:
                self.st["traded_ids"].append(sig["id"])
                log_event("ERROR", where="open_trade", symbol=sig["symbol"], error=str(e))

    def _scan_worker(self, reg, rhits, held, specs, m15, htf):
        """Self-scan, order-book spikes and 15m breakouts. The self-scan spaces its See One reviews
        15 s apart and took 4-8 minutes, during which the main loop read no signals and booked no closes."""
        steps = (("self_scan", SELF_SOURCE, lambda: self.selfscan.scan(reg, rhits, held, specs, m15, htf)),
                 ("orderbook_spike", "訂單簿爆漲", lambda: self.selfscan.orderbook_spikes(held, specs)),
                 ("breakout", "突破", lambda: self.selfscan.fresh_breaks(held, specs, htf)))
        for where, src, run in steps:
            if self.stopping:
                return
            try:
                out = run()
            except Exception as e:
                log_event("ERROR", where=where, error=repr(e)[:300], tb=traceback.format_exc()[-600:])
                continue
            if out:
                with self._scan_lock:
                    self._scan_out.extend((time.time(), f, src) for f in out)

    def fast_guard(self):
        """Move exchange stops within a few seconds of a new high, without waiting for the See One scan to finish."""
        while not self.stopping:
            time.sleep(FAST_GUARD_S)
            if self.dry or self.stopping:
                continue
            moved = False
            for tr in list(self.st["trades"].values()):
                if tr.get("exit_reason"):
                    continue
                try:
                    before = (tr.get("stop"), tr.get("lock_r"), tr.get("mfe_r"))
                    self.protect(tr)
                    moved = moved or (tr.get("stop"), tr.get("lock_r"), tr.get("mfe_r")) != before
                except Exception as e:
                    # A read timeout is not a ToobitError. If it escapes, this thread dies and
                    # stops only move again on the next full scan.
                    log_event("ERROR", trade_id=tr.get("id"), where="fast_guard", error=repr(e)[:200])
            if moved:
                save_state(self.st)
            if time.time() - getattr(self, "_rev_at", 0) >= 15:
                self._rev_at = time.time()
                try:
                    self.watch_reversals()
                except Exception as e:
                    log_event("ERROR", where="watch_reversals", error=repr(e)[:200])

    def watch_reversals(self):
        """Record a 15m reversal. Exit only if the next bar also finishes against the trade.

        A bar that comes back through the entry is a second reversal: record it and keep the position.
        """
        now_ms = time.time() * 1000
        changed = False
        for tr in list(self.st["trades"].values()):
            if tr.get("reversal_closing"):
                continue
            # The first version marked these for exit and then failed before the order went out.
            # Price has since reversed again on AKE and WLD, so management has to resume.
            if str(tr.get("exit_reason") or "").startswith("訊號反轉"):
                tr.pop("exit_reason", None)
                tr.pop("reversal_exit", None)
                changed = True
                log_event("SIGNAL_REVERSAL", decision="resume", trade_id=tr["id"], symbol=tr["tsym"],
                          side=tr["side"], note="反轉平倉沒有送出。下一根若轉回進場價這一側就留倉，兩根都在反向才平")
            sym, side = tr["tsym"], tr["side"]
            rows = klines(sym, "15m", 4)
            if not isinstance(rows, list) or not rows:
                continue
            parsed = []
            for bar in rows:
                try:
                    parsed.append((int(bar[0]), float(bar[1]), float(bar[2]), float(bar[3]), float(bar[4])))
                except (TypeError, ValueError, IndexError):
                    continue
            events, pending = S.reversal_plan(side, tr["entry"], tr.get("break_level"), tr.get("mfe_r") or 0,
                                              parsed, now_ms, tr["opened_ts"] * 1000)
            tr["reversal_bar"] = pending
            for ev in events:
                base = {"trade_id": tr["id"], "symbol": sym, "side": side, "path": tr.get("path"),
                        "entry": tr["entry"], "mfe_r": tr.get("mfe_r"), "bar_open": ev["opened"],
                        "bar_closed": ev["closed"], "o": ev["o"], "h": ev["h"], "l": ev["l"]}
                if ev["decision"] == "record" and tr.get("reversal_logged") != ev["opened"]:
                    log_event("SIGNAL_REVERSAL", decision="record",
                              note="訊號反轉先記下來。下一根 15 分鐘也收在進場價反向才平；轉回來就留倉",
                              **base, **{k: v for k, v in ev.items() if k not in base and k not in ("decision", "opened", "closed")})
                    tr["reversal_logged"] = ev["opened"]
                    changed = True
                elif ev["decision"] == "back" and tr.get("reversal_back") != ev["opened"]:
                    log_event("SIGNAL_REVERSAL", decision="back",
                              note="訊號再度反轉，價格回到進場價這一側，原方向的倉留著",
                              **base, **{k: v for k, v in ev.items() if k not in base and k not in ("decision", "opened", "closed")})
                    tr["reversal_back"] = ev["opened"]
                    changed = True
                elif ev["decision"] == "exit" and tr.get("reversal_exit") != ev["opened"]:
                    tr["reversal_exit"] = ev["opened"]
                    if not S.REVERSAL_EXIT:
                        log_event("SIGNAL_REVERSAL", decision="would_exit",
                                  note="15 分鐘反轉平倉已關閉，只記錄；倉位交給止損、-0.85R 與插針保護", **base)
                        changed = True
                        continue
                    tr["reversal_closing"] = True
                    why = (f"訊號反轉成立：前一根已到進場價另一側，這根 15 分鐘收盤 {ev['c']:.6g} 仍在那一側，"
                           f"而且後面沒有 K 線轉回進場價這一側，平掉剩餘部位")
                    with self._protect_lock:
                        if tr.get("exit_reason"):
                            tr["reversal_closing"] = False
                            continue
                        tr["exit_reason"] = why
                    log_event("SIGNAL_REVERSAL", decision="exit", reason=why,
                              note="連續兩根 15 分鐘都在進場價反向，後面也沒有轉回，剩餘部位改掛單出場", **base)
                    changed = True
                    threading.Thread(target=self._close_reversal, args=(tr,), daemon=True).start()
        if changed:
            save_state(self.st)

    def _close_reversal(self, tr: dict):
        try:
            self.maker_close(tr)
        except Exception as e:
            log_event("ERROR", where="reversal_close", trade_id=tr.get("id"), error=repr(e)[:200])

    def watchdog(self):
        tick = time.time()
        while True:
            time.sleep(30)
            if time.time() - tick > 120:
                # This thread itself overslept: the host was paused, not the main loop. Restart the stall clock.
                log_event("HOST_RESUMED", paused_s=int(time.time() - tick - 30),
                          note="主機暫停後恢復；重設看門狗計時，不把暫停誤判為主迴圈卡住")
                self.beat = time.time()
            tick = time.time()
            stalled = time.time() - self.beat
            if stalled > WATCHDOG_S:
                frame = sys._current_frames().get(threading.main_thread().ident)
                log_event("WATCHDOG_RESTART", stalled_s=int(stalled),
                          stack="".join(traceback.format_stack(frame)[-6:]) if frame else None,
                          note=f"主迴圈超過 {WATCHDOG_S // 60} 分鐘沒有完成一輪，強制結束由外層自動重啟；交易所止損/止盈掛單不受影響")
                os._exit(3)

    def request_stop(self, signum, frame):
        self.stopping = True
        log_event("STOP_REQUESTED", signal=signum, note="收到停止訊號，完成這一輪 (含進行中的下單) 後結束")

    def run(self, once: bool):
        self.beat = time.time()
        self.stopping = False
        signal.signal(signal.SIGTERM, self.request_stop)
        if not once:
            threading.Thread(target=self.watchdog, daemon=True).start()
            threading.Thread(target=self.fast_guard, daemon=True, name="fast-guard").start()
        try:
            me = self.so.me()
        except Exception as e:
            me = {}
            log_event("ERROR", where="seeone_me", error=repr(e)[:200],
                      note="See One 登入失效或無法連線；既有持倉照常管理，恢復登入前不開新倉")
        try:
            self.shadow.ingest(self.so.signals())
            self.shadow.ingest(self.so.history())
        except Exception as e:
            log_event("ERROR", where="seeone_startup", error=repr(e)[:200])
        self.last_history = time.time()
        self.shadow.optimize(force=True)
        try:
            tradable = len(self.tb.specs())
        except Exception:
            tradable = None
        self.st["tradable_contracts"] = tradable
        log_event("START", dry_run=self.dry, paper=self.paper, exchange=self.tb.name, tradable_contracts=tradable,
                  sim_until=os.environ.get("GATE_SIM_UNTIL"),
                  seeone_tier=me.get("tier") or me.get("user", {}).get("tier"),
                  max_positions=S.MAX_POSITIONS, max_leverage=S.MAX_LEVERAGE,
                  daily_stop_pct=S.DAILY_STOP_PCT, lock_ladder=S.LOCK_LADDER, trail_gap_r=S.TRAIL_GAP_R,
                  day_lock=[S.DAY_LOCK_TRIGGER_PCT, S.DAY_LOCK_GIVEBACK], profit_reserve=S.PROFIT_RESERVE,
                  principal=self.st.get("principal"), learned=self.learner.summary())
        while True:
            self.beat = time.time()
            try:
                self.cycle()
            except Exception as e:  # keep the loop alive; open positions have exchange-side SL/TP
                log_event("ERROR", where="cycle", error=repr(e), tb=traceback.format_exc()[-800:],
                          frames=[f"{f.filename.rsplit('/', 1)[-1]}:{f.lineno} {f.name}"
                                  for f in traceback.extract_tb(e.__traceback__) if "/live/" in f.filename])
            finally:
                self.st["traded_ids"] = self.st["traded_ids"][-500:]
                if len(self.st["skips"]) > 500:
                    self.st["skips"] = dict(list(self.st["skips"].items())[-300:])
                if not self.dry:
                    save_state(self.st)
                    self.shadow.save()
            if once or self.stopping:
                return
            for _ in range(LOOP_SECONDS):
                if self.stopping:
                    return
                time.sleep(1)


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
