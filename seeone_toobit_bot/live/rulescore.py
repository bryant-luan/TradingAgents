"""Score live trades opened under the current entry rules against the replay baseline.

The baseline is the ELSA/AIO entry set loaded at 2026-11-08 08:40 Taipei:
1h/4h direction in [-0.5, 1.5), BTC's last hour not against the trade,
15m score under 2.5, and 3h range position at or under 1.05.
On the resolved reviews that set is 704 rows, win rate 57.7%, mean +0.31R.
The review still requires a 25-point edge, quality 60, and +0.20R after fees.

Keep the rules while a live sample is still small, or while both win rate and mean R
are at least that baseline. If five or more trades have closed and either number is
worse, the check says to optimize.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRADES = ROOT / "logs" / "trades" / "live_trades.jsonl"

RULE_SINCE = "2026-11-08T00:40:00+00:00"
BASE_WIN = 0.577
BASE_R = 0.313
MIN_CLOSED = 5


def _rows(path: Path | None = None) -> list[dict]:
    src = path or TRADES
    if not src.exists():
        return []
    out = []
    for line in src.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def score(path: Path | None = None) -> dict:
    opened = {}
    closed = {}
    for ev in _rows(path):
        tid = ev.get("trade_id")
        if not tid or ev.get("ts", "") < RULE_SINCE:
            continue
        if ev.get("event") == "ENTRY":
            opened[tid] = ev
        elif ev.get("event") == "EXIT" and ev.get("r_result") is not None:
            closed[tid] = ev
    done = [closed[tid] for tid in opened if tid in closed]
    rs = [float(e["r_result"]) for e in done]
    n = len(rs)
    wins = sum(1 for r in rs if r > 0)
    win = None if n == 0 else wins / n
    mean_r = None if n == 0 else sum(rs) / n
    if n < MIN_CLOSED:
        verdict = "hold"
    elif win >= BASE_WIN and mean_r >= BASE_R:
        verdict = "keep"
    else:
        verdict = "optimize"
    return {
        "since": RULE_SINCE,
        "n": n,
        "wins": wins,
        "win_rate": None if win is None else round(win, 3),
        "mean_r": None if mean_r is None else round(mean_r, 3),
        "base_win": BASE_WIN,
        "base_r": BASE_R,
        "min_closed": MIN_CLOSED,
        "open": len(opened) - n,
        "verdict": verdict,
    }


def line(path: Path | None = None) -> str:
    s = score(path)
    live = "尚無平倉" if s["win_rate"] is None else f"勝率 {s['win_rate']:.0%}、平均 {s['mean_r']:+.3f}R"
    verdict = {"hold": "樣本未滿，繼續使用", "keep": "不低於基準，繼續使用", "optimize": "低於基準，應優化規則"}[s["verdict"]]
    return (f"規則成績 (自 11-07 16:36 台北): 已平 {s['n']} 筆、持倉 {s['open']}，{live}。"
            f"基準 勝率 {s['base_win']:.0%}、平均 {s['base_r']:+.3f}R（159 筆回放）。{verdict}")
