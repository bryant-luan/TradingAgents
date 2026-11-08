"""Separate status reports for the Toobit bot and the Gate bot.

    python3 -m live.report            # both, since each venue's last report
    python3 -m live.report --hours 24 # both, fixed window, markers stay put
    python3 -m live.report toobit     # one venue
    python3 -m live.report gate

Read-only. See One is not called. Each venue has its own log, book, and time marker.
"""
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

from .exchange import connect
from .market import klines
from .rulescore import line as rule_line

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
TPE = timezone(timedelta(hours=8))
WATCH = {"ENTRY", "EXIT", "ADOPT", "ADOPT_CLOSE", "SELF_EXIT_DECISION", "PROFIT_LOCK", "STOP_MOVE", "TP_HIT",
         "DAY_LOCK_ARMED", "DAY_LOCK_HIT", "HALT", "LOCK_EXIT", "SMALL_CLOSE", "STOP_RESTORED"}
HEALTH = {"WATCHDOG_RESTART", "HOST_RESUMED", "ERROR", "START"}
VENUES = {
    "toobit": {
        "title": "Toobit",
        "state": LOGS / "live_state.json",
        "log": LOGS / "live_bot_console.log",
        "mark": LOGS / "report_mark.json",
        "trades": LOGS / "trades" / "live_trades.jsonl",
        "bot_cmd": "live.bot",
        "gate": False,
    },
    "gate": {
        "title": "Gate",
        "state": LOGS / "gate" / "live_state.json",
        "log": LOGS / "gate" / "live_bot_console.log",
        "mark": LOGS / "gate" / "report_mark.json",
        "trades": LOGS / "gate" / "trades" / "live_trades.jsonl",
        "bot_cmd": "live.gate_bot",
        "gate": True,
    },
}


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


def _cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return ""


def _environ(pid: int) -> dict:
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    except OSError:
        return {}
    out = {}
    for item in raw:
        if b"=" not in item:
            continue
        k, v = item.split(b"=", 1)
        out[k.decode(errors="replace")] = v.decode(errors="replace")
    return out


def _pids() -> list[int]:
    out = []
    for name in Path("/proc").iterdir():
        if name.name.isdigit() and int(name.name) != os.getpid():
            out.append(int(name.name))
    return out


def running(bot_cmd: str, gate: bool) -> tuple[bool, bool]:
    """Whether this venue's bot and its monitor are alive. The two monitors share a module name."""
    bot = mon = False
    for pid in _pids():
        cmd = _cmdline(pid)
        if bot_cmd in cmd and "live.report" not in cmd:
            bot = True
        if "live.monitor" in cmd:
            is_gate = _environ(pid).get("EXCHANGE", "").lower() in ("gate", "gateio", "gate.io")
            if is_gate == gate:
                mon = True
    return bot, mon


def btc_trend(venue: str) -> str:
    os.environ["EXCHANGE"] = venue
    rows = klines("BTC-SWAP-USDT", "1h", 25)
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


def contract_book(name: str) -> set[str]:
    """Symbols this venue can actually trade. The other venue's list is not a substitute."""
    os.environ["EXCHANGE"] = name
    try:
        return set(connect().specs())
    except Exception:
        return set()


def as_tsym(raw: str) -> str:
    s = str(raw or "").split(":")[0]
    if s.endswith("-SWAP-USDT"):
        return s
    if s.endswith("USDT"):
        return s[:-4] + "-SWAP-USDT"
    return s


def place_label(tsym: str, book: set[str], other: set[str]) -> str:
    if tsym in book and tsym in other:
        return "兩邊都有"
    if tsym in book:
        return "只在本所"
    if tsym in other:
        return "只在另一所"
    return ""


def report_venue(name: str, hours: float | None, now: datetime, book: set[str], other: set[str]):
    cfg = VENUES[name]
    mark = json.loads(cfg["mark"].read_text()) if cfg["mark"].exists() else {}
    since_dt = now - timedelta(hours=hours) if hours else (
        datetime.fromisoformat(mark["ts"]) if mark.get("ts") else now - timedelta(hours=6))
    since = since_dt.isoformat(timespec="seconds")
    st = json.loads(cfg["state"].read_text()) if cfg["state"].exists() else {
        "trades": {}, "closed": [], "day_start_equity": None}
    evs = events_since(cfg["log"], since)

    print(f"== {cfg['title']} 報告 {tpe(now.isoformat())} (台北)，涵蓋 {tpe(since)} 起")
    bot, mon = running(cfg["bot_cmd"], cfg["gate"])
    age = None if not cfg["state"].exists() else time.time() - cfg["state"].stat().st_mtime
    hc = {k: sum(1 for e in evs if e["event"] == k) for k in HEALTH}
    resumed = [e for e in evs if e["event"] == "HOST_RESUMED"]
    age_s = "尚無" if age is None else f"{age:.0f}s 前更新"
    print(f"健康: 機器人 {'運行' if bot else '停止'} / 監控 {'運行' if mon else '停止'} / 狀態檔 {age_s} / "
          f"重啟 {hc['START']} 次 / 看門狗 {hc['WATCHDOG_RESTART']} 次 / 錯誤 {hc['ERROR']} 次")
    for e in resumed:
        print(f"  主機暫停 {e.get('paused_s', 0) // 60} 分鐘，{tpe(e['ts'])} 恢復")

    os.environ["EXCHANGE"] = name
    try:
        tb = connect()
        bal = tb.balance()
        pos = {(p["symbol"], p["side"]): p for p in tb.positions()}
    except Exception as e:
        print(f"帳戶: 連線失敗 ({type(e).__name__})")
        pos, bal = {}, None
    if bal is not None:
        upnl = sum(float(p.get("unrealizedPnL") or 0) for p in pos.values())
        equity = float(bal["balance"]) + upnl
        day0 = st.get("day_start_equity")
        day = f"，今日 {equity - day0:+.2f}u ({(equity - day0) / day0:+.1%})" if day0 else ""
        print(f"帳戶: 權益 {equity:.2f}u (餘額 {float(bal['balance']):.2f} + 未實現 {upnl:+.2f}){day}"
              f"{'，已鎖利' if st.get('day_lock_armed') else ''}")
    if book:
        print(f"可交易合約: {len(book)}。只交易本所有的幣，另一所多出來的不下單。")

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
        where = place_label(t["tsym"], book, other)
        print(f"  {t['tsym'].replace('-SWAP-USDT', '')} {'多' if sgn > 0 else '空'} 進 {t['entry']:g} 現 {last:g} "
              f"止損 {t['stop']:.6g} | {r_now:+.2f}R 未實現 {float(p.get('unrealizedPnL') or 0):+.2f}u | "
              f"{'已達 ' + ','.join(filled) if filled else '未達 TP'} | 持有 {(time.time() - t['opened_ts']) / 60:.0f} 分 "
              f"| {t.get('source')}" + (f" | {where}" if where else ""))
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
        where = e.get("listing")
        if where == "only":
            where = "只在本所"
        elif where == "both":
            where = "兩邊都有"
        else:
            where = place_label(as_tsym(e.get("symbol") or e.get("trade_id")), book, other)
        prefix = "模擬 " if e.get("paper") else ""
        print(f"  {tpe(e['ts'])} {prefix}{msg}" + (f" | {where}" if where else ""))
    closed = [c for c in st["closed"] if c.get("closed_at", "") > since and c.get("pnl") is not None]
    if closed:
        print(f"  期間已平倉 {len(closed)} 筆，合計 {sum(c['pnl'] for c in closed):+.2f}u，"
              f"平均 {sum(c['r_result'] or 0 for c in closed) / len(closed):+.2f}R")
        buckets: dict[str, list] = {}
        for c in closed:
            buckets.setdefault(place_label(c.get("symbol") or "", book, other) or "未分類", []).append(c)
        for label, rows in buckets.items():
            rs = [x.get("r_result") or 0 for x in rows]
            print(f"  其中{label} {len(rows)} 筆，合計 {sum(x.get('pnl') or 0 for x in rows):+.2f}u，"
                  f"平均 {sum(rs) / len(rows):+.2f}R")

    m = st.get("market") or {}
    trend = btc_trend(name)
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
    only = [e for e in evs if e.get("event") == "BREAKOUT" and e.get("listing") == "only"]
    if only:
        print("只在本所的突破: " + "、".join(f"{e.get('symbol')} {e.get('side')}" for e in only[:8]))
    elif name == "gate":
        print("只在本所的成交還沒有。同一套規則用在 Gate 獨有的幣上，平倉之後再決定要不要改。")
    print(rule_line(cfg["trades"]))
    if not hours:
        cfg["mark"].parent.mkdir(parents=True, exist_ok=True)
        cfg["mark"].write_text(json.dumps({"ts": now.isoformat(timespec="seconds")}))
    print()


def main():
    load_dotenv(ROOT / ".env")
    load_dotenv(ROOT / ".env.gate", override=True)
    hours = float(sys.argv[sys.argv.index("--hours") + 1]) if "--hours" in sys.argv else None
    names = [a for a in sys.argv[1:] if a in VENUES]
    if not names:
        names = list(VENUES)
    now = datetime.now(timezone.utc)
    books = {name: contract_book(name) for name in VENUES}
    if "toobit" in names and "gate" in names and books["toobit"] and books["gate"]:
        a, b = books["toobit"], books["gate"]
        print(f"合約清單不同：Toobit {len(a)}、Gate {len(b)}，兩邊都有 {len(a & b)}，"
              f"只在 Toobit {len(a - b)}，只在 Gate {len(b - a)}。")
        print()
    for name in names:
        report_venue(name, hours, now, books.get(name) or set(),
                     books.get("gate" if name == "toobit" else "toobit") or set())


if __name__ == "__main__":
    main()
