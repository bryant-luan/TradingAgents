"""Shadow stand-in for See One 分析驗算.

Estimates the chance TP1 is reached before the stop from similar resolved reviews
(same side, similar 3h range position, BTC hour, and higher-timeframe band).
Live orders do not use this. It only writes a simulation line so the estimate
can be checked against what actually happens, and against See One when that
call is available.
"""
import json
from pathlib import Path

MIN_BUCKET = 40
RPOS_EDGES = (0.25, 0.45, 0.70, 1.05)


def _rpos_bin(value: float | None) -> int | None:
    if value is None:
        return None
    for i, edge in enumerate(RPOS_EDGES):
        if value < edge:
            return i
    return len(RPOS_EDGES)


def _htf_band(value: float | None) -> str | None:
    if value is None:
        return None
    if value < -0.5:
        return "against"
    if value >= 1.5:
        return "late"
    return "ok"


def _btc_band(value: float | None) -> str | None:
    if value is None:
        return None
    return "with" if value >= 0 else "against"


class LocalReview:
    def __init__(self, path: Path):
        self.path = path
        self.mtime = 0.0
        self.rows: list[dict] = []

    def reload(self):
        if not self.path.exists():
            return
        mtime = self.path.stat().st_mtime
        if mtime == self.mtime:
            return
        data = json.loads(self.path.read_text())
        self.rows = [r for r in data.get("resolved", []) if r.get("hit") is not None and r.get("side")]
        self.mtime = mtime

    def _match(self, side: str, rpos: int | None, btc: str | None, htf: str | None, level: int) -> list[dict]:
        out = []
        for row in self.rows:
            if row["side"] != side:
                continue
            if level <= 2 and _rpos_bin(row.get("rpos")) != rpos:
                continue
            if level <= 1 and _btc_band(row.get("btc1h")) != btc:
                continue
            if level == 0 and _htf_band(row.get("htf")) != htf:
                continue
            out.append(row)
        return out

    def estimate(self, side: str, rpos: float | None, btc1h: float | None, htf_dir: float | None) -> dict:
        """Probability, in percent, that TP1 prints before the stop. `level` says how coarse the match was."""
        self.reload()
        rbin, bband, hband = _rpos_bin(rpos), _btc_band(btc1h), _htf_band(htf_dir)
        chosen, level = [], 3
        for level in range(4):
            chosen = self._match(side, rbin, bband, hband, level)
            if len(chosen) >= MIN_BUCKET:
                break
        if not chosen:
            return {"prob": None, "n": 0, "level": None}
        prob = 100.0 * sum(r["hit"] for r in chosen) / len(chosen)
        return {"prob": round(prob, 1), "n": len(chosen), "level": level}
