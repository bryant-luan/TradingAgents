"""Shadow stand-in for See One 分析驗算.

Estimates the chance TP1 is reached before the stop from similar resolved reviews.
Live orders do not use this. Alignment is measured only where a row also has See
One's own probability: the table is fit on earlier reviews and scored on later
ones, plus any fresh pairs logged while both systems run.
"""
import json
from collections import defaultdict
from pathlib import Path

MIN_BUCKET = 40
# Out-of-sample pairs against See One before the shadow estimate is called aligned.
ALIGN_MIN_N = 200
ALIGN_MAX_MAE = 8.0          # percentage points
ALIGN_MIN_AGREE = 0.75       # share of rows where both are on the same side of 60
PASS_AT = 60.0
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


class _Index:
    def __init__(self, rows: list[dict]):
        self.exact = defaultdict(list)
        self.rpos_btc = defaultdict(list)
        self.rpos = defaultdict(list)
        self.side = defaultdict(list)
        for row in rows:
            hit = row["hit"]
            side = row["side"]
            rbin, bband, hband = _rpos_bin(row.get("rpos")), _btc_band(row.get("btc1h")), _htf_band(row.get("htf"))
            self.exact[(side, rbin, bband, hband)].append(hit)
            self.rpos_btc[(side, rbin, bband)].append(hit)
            self.rpos[(side, rbin)].append(hit)
            self.side[side].append(hit)

    def hits(self, side: str, rpos: float | None, btc1h: float | None, htf_dir: float | None) -> tuple[list, int]:
        rbin, bband, hband = _rpos_bin(rpos), _btc_band(btc1h), _htf_band(htf_dir)
        for level, bucket in (
            (0, self.exact[(side, rbin, bband, hband)]),
            (1, self.rpos_btc[(side, rbin, bband)]),
            (2, self.rpos[(side, rbin)]),
            (3, self.side[side]),
        ):
            if len(bucket) >= MIN_BUCKET:
                return bucket, level
        return self.side[side], 3


class LocalReview:
    def __init__(self, path: Path):
        self.path = path
        self.mtime = 0.0
        self.rows: list[dict] = []
        self.index = _Index([])
        self._align_mtime = -1.0
        self.align: dict = {}

    def reload(self):
        if not self.path.exists():
            return
        mtime = self.path.stat().st_mtime
        if mtime == self.mtime:
            return
        data = json.loads(self.path.read_text())
        self.rows = [r for r in data.get("resolved", []) if r.get("hit") is not None and r.get("side")]
        self.index = _Index(self.rows)
        self.mtime = mtime

    def estimate(self, side: str, rpos: float | None, btc1h: float | None, htf_dir: float | None) -> dict:
        """Probability, in percent, that TP1 prints before the stop."""
        self.reload()
        bucket, level = self.index.hits(side, rpos, btc1h, htf_dir)
        if not bucket:
            return {"prob": None, "n": 0, "level": None}
        return {"prob": round(100.0 * sum(bucket) / len(bucket), 1), "n": len(bucket), "level": level}

    def holdout_vs_seeone(self) -> dict:
        """Fit on earlier reviews, score later ones that carry See One's probability."""
        self.reload()
        paired = [r for r in self.rows if r.get("prob") is not None]
        paired.sort(key=lambda r: r.get("ts") or 0)
        cut = int(len(paired) * 0.7)
        train, test = paired[:cut], paired[cut:]
        index = _Index(train)
        err, agree, hit_gap = [], 0, []
        used = 0
        for row in test:
            bucket, _ = index.hits(row["side"], row.get("rpos"), row.get("btc1h"), row.get("htf"))
            if len(bucket) < MIN_BUCKET:
                continue
            local = 100.0 * sum(bucket) / len(bucket)
            seeone = float(row["prob"])
            err.append(local - seeone)
            used += 1
            if (local >= PASS_AT) == (seeone >= PASS_AT):
                agree += 1
            hit_gap.append((local, row["hit"]))
        n = len(err)
        mae = sum(abs(e) for e in err) / n if n else None
        bias = sum(err) / n if n else None
        agree_rate = agree / n if n else None
        # Does "we said 60" actually hit TP1 about 60% of the time on the same holdout?
        near = [h for p, h in hit_gap if 50 <= p < 70]
        hit_near = sum(near) / len(near) if near else None
        aligned = bool(n >= ALIGN_MIN_N and mae is not None and mae <= ALIGN_MAX_MAE
                       and agree_rate is not None and agree_rate >= ALIGN_MIN_AGREE)
        return {"n": n, "train": len(train), "test": len(test), "used": used,
                "mae": None if mae is None else round(mae, 2),
                "bias": None if bias is None else round(bias, 2),
                "agree": None if agree_rate is None else round(agree_rate, 3),
                "hit_when_local_50_70": None if hit_near is None else round(hit_near, 3),
                "hit_n": len(near), "aligned": aligned}

    def report(self, fresh_pairs: list[dict] | None = None) -> dict:
        """Historical holdout plus pairs collected live. Cached until the review file changes."""
        self.reload()
        if self.mtime != self._align_mtime:
            self.align = self.holdout_vs_seeone()
            self._align_mtime = self.mtime
        fresh = [p for p in (fresh_pairs or []) if p.get("prob") is not None and p.get("seeone_prob") is not None]
        err = [float(p["prob"]) - float(p["seeone_prob"]) for p in fresh]
        agree = sum((float(p["prob"]) >= PASS_AT) == (float(p["seeone_prob"]) >= PASS_AT) for p in fresh)
        live_n = len(err)
        out = {**self.align, "live_pairs": live_n,
               "live_mae": None if not err else round(sum(abs(e) for e in err) / live_n, 2),
               "live_agree": None if not err else round(agree / live_n, 3)}
        # Live orders wait until the historical holdout is aligned and fresh pairs agree too.
        out["ready"] = bool(out["aligned"] and live_n >= 50 and out["live_mae"] is not None
                            and out["live_mae"] <= ALIGN_MAX_MAE and out["live_agree"] >= ALIGN_MIN_AGREE)
        return out
