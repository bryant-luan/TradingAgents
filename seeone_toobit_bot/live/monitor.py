"""Independent once-a-minute safety monitor for the live bot (runs as its own process).

Every minute:
  1. the bot process is alive and its state file is still being written (else restart / kill a hung bot);
  2. every position the bot owns has a stop on the exchange (else re-place it, or close if price is past it);
  3. a position opened by a bot entry order but not tracked (orphan) gets a 1.5×ATR stop.
Positions not opened by the bot (manual trades) are only reported, never touched.

Run:  ./live/run_monitor.sh   (tmux session seeone-monitor)
"""
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from .selfsignal import ATR_MULT, MAX_STOP_PCT, MIN_STOP_PCT, atr
from .exchange import connect
from .toobit import ToobitError

ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = ROOT / "logs" / "live_state.json"
MON_LOG = ROOT / "logs" / "monitor.jsonl"
CHECK_S = 60
# The bot saves state at the end of every cycle; its own watchdog fires at 20 minutes, this is the backstop.
STATE_STALE_S = 25 * 60
NO_PROCESS_S = 120
PENDING_GRACE_S = 60
BOT_ENTRY_PREFIX = "mo"


def log(event: str, **data):
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "event": event, **data}
    with MON_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    print(json.dumps(rec, ensure_ascii=False, default=str), flush=True)


def pids(pattern: str) -> list[int]:
    out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True).stdout.split()
    return [int(p) for p in out if int(p) != os.getpid()]


class Monitor:
    def __init__(self):
        load_dotenv(ROOT / ".env")
        self.tb = connect()
        self.no_proc_since = None
        self.last_tick = time.time()
        self.resumed_at = 0.0
        self.origin: dict[tuple, str] = {}   # (sym, side, avgPrice) -> "bot" | "manual"
        self.reported: set = set()

    # ------------------------------------------------------------------ bot process health
    def check_process(self) -> dict:
        bot, loop = pids("python3 -u -m live.bot"), pids("run_forever.sh")
        now = time.time()
        if now - self.last_tick > 3 * CHECK_S:
            # The monitor itself overslept, so the whole host was paused; the bot was frozen too, not hung.
            self.resumed_at = now
            self.no_proc_since = None
            log("MON_HOST_RESUMED", paused_s=int(now - self.last_tick),
                note="主機暫停後恢復；狀態檔停更期間不算卡住，從恢復時重新計時")
        self.last_tick = now
        age = now - max(STATE_FILE.stat().st_mtime, self.resumed_at) if STATE_FILE.exists() else None
        if not bot:
            self.no_proc_since = self.no_proc_since or time.time()
            if time.time() - self.no_proc_since > NO_PROCESS_S:
                if not loop:
                    subprocess.run(["tmux", "-f", "/exec-daemon/tmux.portal.conf", "kill-session", "-t", "seeone-live-bot"],
                                   capture_output=True)
                    subprocess.run(["tmux", "-f", "/exec-daemon/tmux.portal.conf", "new-session", "-d", "-s", "seeone-live-bot",
                                    "-c", str(ROOT), "./live/run_forever.sh"], capture_output=True)
                    log("MON_BOT_RESTART", reason=f"機器人行程消失超過 {NO_PROCESS_S}s 且無自動重啟迴圈，已重新啟動")
                else:
                    log("MON_BOT_MISSING", reason="機器人行程消失，自動重啟迴圈仍在，等待其重啟")
                self.no_proc_since = time.time()
        else:
            self.no_proc_since = None
            if age is not None and age > STATE_STALE_S:
                for p in bot:
                    os.kill(p, signal.SIGKILL)
                log("MON_BOT_KILL", state_age_s=int(age),
                    reason=f"機器人狀態檔 {int(age // 60)} 分鐘未更新 (卡住)，強制結束由自動重啟迴圈重開")
        return {"bot": bool(bot), "loop": bool(loop), "state_age_s": None if age is None else int(age)}

    # ------------------------------------------------------------------ stops
    def stop_of(self, sym: str, side: str) -> float | None:
        for o in self.tb.open_orders(sym, "STOP_PROFIT_LOSS"):
            if f"_{side}_LOSS" in str(o.get("clientOrderId", "")) or (
                    o.get("side") == ("SELL_CLOSE" if side == "LONG" else "BUY_CLOSE") and "LOSS" in str(o.get("clientOrderId", ""))):
                return float(o.get("stopPrice") or 0) or None
        return None

    def ensure_stop(self, sym: str, side: str, stop: float, why: str):
        px = self.tb.price(sym)
        past = px <= stop if side == "LONG" else px >= stop
        if past:
            self.tb.flash_close(sym, side)
            log("MON_CLOSE", symbol=sym, side=side, price=px, stop=stop, reason=f"{why}；價格已穿越止損，直接平倉")
            return
        self.tb.set_position_stop(sym, side, self.tb.fmt_price(sym, stop))
        log("MON_STOP_FIX", symbol=sym, side=side, price=px, stop=stop, reason=f"{why}；已補掛止損")

    def opened_by_bot(self, sym: str, side: str, avg: str) -> bool:
        key = (sym, side, avg)
        if key not in self.origin:
            open_side = "BUY_OPEN" if side == "LONG" else "SELL_OPEN"
            fills = [t for t in self.tb.user_trades(sym, 50) if t.get("side") == open_side]
            fills.sort(key=lambda t: int(t["time"]), reverse=True)
            oid = fills[0]["orderId"] if fills else None
            cid = str(self.tb.order(oid).get("clientOrderId", "")) if oid else ""
            self.origin[key] = "bot" if cid.startswith(BOT_ENTRY_PREFIX) else "manual"
        return self.origin[key] == "bot"

    def check_positions(self) -> dict:
        st = json.loads(STATE_FILE.read_text())
        tracked = {(t["tsym"], t["side"]): t for t in st.get("trades", {}).values()}
        pend = st.get("pending_entry")
        pend_key = None
        if pend:
            s = pend["sig"]["symbol"]
            pend_key = (s[:-4] + "-SWAP-USDT" if s.endswith("USDT") else s, pend["sig"]["side"])
        summary = {"bot_positions": 0, "protected": 0, "fixed": 0, "manual": 0}
        for p in self.tb.positions():
            sym, side = p["symbol"], p.get("side")
            key = (sym, side)
            stop = self.stop_of(sym, side)
            if key in tracked:
                summary["bot_positions"] += 1
                if stop:
                    summary["protected"] += 1
                    continue
                self.ensure_stop(sym, side, float(tracked[key]["stop"]), "機器人持倉在交易所上沒有止損")
                summary["fixed"] += 1
            elif key == pend_key:
                summary["bot_positions"] += 1
                if stop or time.time() - pend["ts"] < PENDING_GRACE_S:
                    summary["protected"] += bool(stop)
                    continue
                self.ensure_stop(sym, side, float(pend["stop"]), "進場中的倉位超過 1 分鐘仍無止損")
                summary["fixed"] += 1
            elif self.opened_by_bot(sym, side, str(p.get("avgPrice"))):
                summary["bot_positions"] += 1
                if stop:
                    summary["protected"] += 1
                    if key not in self.reported:
                        self.reported.add(key)
                        log("MON_ORPHAN", symbol=sym, side=side, contracts=p.get("position"), stop=stop,
                            reason="機器人下單成交但未被追蹤的倉位，已有止損")
                    continue
                a = atr(sym)
                avg = float(p["avgPrice"])
                dist = min(max(a[0] * ATR_MULT if a else avg * MIN_STOP_PCT, avg * MIN_STOP_PCT), avg * MAX_STOP_PCT)
                self.ensure_stop(sym, side, avg - dist if side == "LONG" else avg + dist,
                                 "機器人下單成交但未被追蹤的倉位 (無止損)，以 1.5×ATR 設止損")
                summary["fixed"] += 1
            else:
                summary["manual"] += 1
                if key not in self.reported:
                    self.reported.add(key)
                    log("MON_MANUAL", symbol=sym, side=side, contracts=p.get("position"), avg=p.get("avgPrice"), stop=stop,
                        reason="非機器人開的倉位 (手動單)，只回報不處理")
        return summary

    def run(self):
        log("MON_START", check_s=CHECK_S)
        n = 0
        while True:
            t0 = time.time()
            try:
                health = self.check_process()
                pos = self.check_positions()
                n += 1
                line = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **health, **pos}
                print(json.dumps(line, ensure_ascii=False), flush=True)
                if n % 60 == 1 or pos["fixed"] or not health["bot"]:
                    log("MON_OK" if not pos["fixed"] and health["bot"] else "MON_STATUS", **health, **pos)
            except (ToobitError, OSError, ValueError, KeyError) as e:
                log("MON_ERROR", error=repr(e)[:300])
            except Exception as e:  # never let the monitor die
                log("MON_ERROR", error=repr(e)[:300])
            time.sleep(max(1.0, CHECK_S - (time.time() - t0)))


if __name__ == "__main__":
    Monitor().run()
