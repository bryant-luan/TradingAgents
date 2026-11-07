"""Status report for a human or agent check-in: health, account, open positions, entries/exits and market trend
since the previous report.

    python3 -m live.report            # since the last report (first run: last 6 hours)
    python3 -m live.report --hours 24 # fixed window, does not move the marker

Read-only: Toobit is queried for balance/positions/prices; See One is not called (its session belongs to the bot).
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

from .rulescore import line as rule_line
from .toobit import Toobit

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
MARK = LOGS / "report_mark.json"
TPE = timezone(timedelta(hours=8))
WATCH = {"ENTRY", "EXIT", "ADOPT", "ADOPT_CLOSE", "SELF_EXIT_DECISION", "PROFIT_LOCK", "STOP_MOVE", "TP_HIT",
         "DAY_LOCK_ARMED", "DAY_LOCK_HIT", "HALT", "LOCK_EXIT", "SMALL_CLOSE", "STOP_RESTORED"}
HEALTH = {"WATCHDOG_RESTART", "HOST_RESUMED", "ERROR", "START"}


def tpe(ts: str | float) -> str:
    t = datetime.fromtimestamp(ts, timezone.utc) if isinstance(ts, (int, float)) else datetime.fromisoformat(ts)
    return t.astimezone(TPE).strftime("%m-%d %H:%M")


def events_since(path: Path, since: str) -> list:
    out = []
    if not path.exists():
        return out
    for line in path.open(encoding="utf-8", errors="replace"):
        if not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("ts", "") > since:
            out.append(d)
    return out


def btc_trend() -> str:
    rows = requests.get("https://api.toobit.com/quote/v1/klines",
                        params={"symbol": "BTC-SWAP-USDT", "interval": "1h", "limit": 25}, timeout=10).json()
    c = [float(r[4]) for r in rows] if isinstance(rows, list) else []
    if len(c) < 2:
        return "BTC 價格暫時取不到"
    def ch(n):
        return None if len(c) <= n or not c[-1 - n] else (c[-1] - c[-1 - n]) / c[-1 - n] * 100
    parts = [f"BTC {c[-1]:.0f}"]
    for n, label in ((1, "1h"), (4, "4h"), (24, "24h")):
        v = ch(n)
        parts.append(f"{label} {v:+.2f}%" if v is not None else f"{label} —")
    return "  ".join(parts)


def main():
    hours = float(sys.argv[sys.argv.index("--hours") + 1]) if "--hours" in sys.argv else None
    mark = json.loads(MARK.read_text()) if MARK.exists() else {}
    now = datetime.now(timezone.utc)
    since_dt = now - timedelta(hours=hours) if hours else (
        datetime.fromisoformat(mark["ts"]) if mark.get("ts") else now - timedelta(hours=6))
    since = since_dt.isoformat(timespec="seconds")
    st = json.loads((LOGS / "live_state.json").read_text())
    evs = events_since(LOGS / "live_bot_console.log", since)

    print(f"== 報告 {tpe(now.isoformat())} (台北)，涵蓋 {tpe(since)} 起")
    bot = subprocess.run(["pgrep", "-f", "python3 -u -m live.bot"], capture_output=True).returncode == 0
    mon = subprocess.run(["pgrep", "-f", "python3 -u -m live.monitor"], capture_output=True).returncode == 0
    age = time.time() - (LOGS / "live_state.json").stat().st_mtime
    hc = {k: sum(1 for e in evs if e["event"] == k) for k in HEALTH}
    resumed = [e for e in evs if e["event"] == "HOST_RESUMED"]
    print(f"健康: 機器人 {'運行' if bot else '停止'} / 監控 {'運行' if mon else '停止'} / 狀態檔 {age:.0f}s 前更新 / "
          f"重啟 {hc['START']} 次 / 看門狗 {hc['WATCHDOG_RESTART']} 次 / 錯誤 {hc['ERROR']} 次")
    for e in resumed:
        print(f"  主機暫停 {e.get('paused_s', 0) // 60} 分鐘，{tpe(e['ts'])} 恢復")

    load_dotenv(ROOT / ".env")
    tb = Toobit(os.environ["TOOBIT_API_KEY"], os.environ["TOOBIT_API_SECRET"])
    bal = tb.balance()
    pos = {(p["symbol"], p["side"]): p for p in tb.positions()}
    upnl = sum(float(p.get("unrealizedPnL") or 0) for p in pos.values())
    equity = float(bal["balance"]) + upnl
    day0 = st.get("day_start_equity")
    day = f"，今日 {equity - day0:+.2f}u ({(equity - day0) / day0:+.1%})" if day0 else ""
    print(f"帳戶: 權益 {equity:.2f}u (餘額 {float(bal['balance']):.2f} + 未實現 {upnl:+.2f}){day}"
          f"{'，已鎖利' if st.get('day_lock_armed') else ''}")

    print(f"持倉 ({len(pos)}):")
    for t in st["trades"].values():
        p = pos.pop((t["tsym"], t["side"]), None)
        if not p:
            print(f"  {t['tsym']} {t['side']} 狀態檔有但交易所無持倉 (待機器人對帳)")
            continue
        sgn = 1 if t["side"] == "LONG" else -1
        last = float(p.get("lastPrice") or p.get("markPrice"))
        risk = abs(t["entry"] - t["stop"]) if t.get("lock_r") is None else abs(t["entry"] - t.get("init_stop", t["stop"]))
        r_now = (last - t["entry"]) * sgn / risk if risk else 0
        filled = [x["label"] for x in t.get("tps", []) if x.get("filled")]
        print(f"  {t['tsym'].replace('-SWAP-USDT', '')} {'多' if sgn > 0 else '空'} 進 {t['entry']:g} 現 {last:g} "
              f"止損 {t['stop']:.6g} | {r_now:+.2f}R 未實現 {float(p.get('unrealizedPnL') or 0):+.2f}u | "
              f"{'已達 ' + ','.join(filled) if filled else '未達 TP'} | 持有 {(time.time() - t['opened_ts']) / 60:.0f} 分 "
              f"| {t.get('source')}")
    for (sym, side), p in pos.items():
        print(f"  {sym} {side} 不在機器人狀態中 (監控/對帳會接管) 未實現 {float(p.get('unrealizedPnL') or 0):+.2f}u")

    acts = [e for e in evs if e["event"] in WATCH]
    print(f"期間動作 ({len(acts)}):")
    for e in acts:
        sym = str(e.get("symbol") or e.get("trade_id") or "").replace("-SWAP-USDT", "").split(":")[0]
        if e["event"] == "ENTRY":
            sz = e.get("sizing") or {}
            msg = (f"開倉 {sym} {e.get('side')} 均價 {e.get('entry_avg')} 止損 {e.get('stop', 0):.6g} "
                   f"槓桿 {sz.get('leverage')}x 風險 {sz.get('risk_usdt')}u ({e.get('source')})")
        elif e["event"] == "EXIT":
            msg = (f"平倉 {sym} {e.get('side')} {e.get('r_result')}R {e.get('realized_pnl_usdt')}u "
                   f"持有 {e.get('held_min')} 分 — {e.get('reason')}")
        else:
            msg = f"{e['event']} {sym} {str(e.get('reason') or e.get('label') or e.get('note') or '')[:80]}"
        print(f"  {tpe(e['ts'])} {msg}")
    closed = [c for c in st["closed"] if c.get("closed_at", "") > since and c.get("pnl") is not None]
    if closed:
        print(f"  期間已平倉 {len(closed)} 筆，合計 {sum(c['pnl'] for c in closed):+.2f}u，"
              f"平均 {sum(c['r_result'] or 0 for c in closed) / len(closed):+.2f}R")

    m = st.get("market") or {}
    trend = btc_trend()
    if m:
        d = m.get("dirs", {})
        print(f"趨勢: {trend} | See One 15m {d.get('15m')} 1h {d.get('1h')} 4h {d.get('4h')} "
              f"大週期 {m.get('htf')} 盤勢 {m.get('regime')} ({tpe(m['ts'])})")
    else:
        print(f"趨勢: {trend}")
    skips = [e for e in evs if e["event"] == "SKIP"]
    if skips:
        why = {}
        for e in skips:
            k = str(e.get("reason", ""))
            for key in ("RSI", "大方向", "同方向", "已有持倉", "訊號已", "區間", "近 2 小時", "冷卻"):
                if key in k:
                    k = key
                    break
            else:
                k = k[:20]
            why[k] = why.get(k, 0) + 1
        print("略過原因: " + "、".join(f"{k} {v}" for k, v in sorted(why.items(), key=lambda x: -x[1])[:6]))
    print(rule_line())
    if not hours:
        MARK.write_text(json.dumps({"ts": now.isoformat(timespec="seconds")}))


if __name__ == "__main__":
    main()
