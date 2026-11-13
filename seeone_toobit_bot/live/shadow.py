"""Shadow optimizer: learns from every See One signal outcome, traded or not.

- Archives each CLOSED See One signal with its intrinsic traits (side, subtype, scanner/readiness bucket, rank).
- Fresh signals are snapshotted at first sight with the bot's full factor set and score, so once they close
  the entry threshold can be calibrated against what actually happened.
- Re-derives: per-trait score adjustments, a calibrated min score, and the TP1-TP5 size split.
"""
import json
import time
from pathlib import Path

from . import strategy as S

ARCHIVE_CAP = 3000
SNAPSHOT_MAX_AGE_MIN = 60
REOPT_EVERY = 5
TAG_MIN_N = 8
TAG_SHRINK = 10
TAG_GAIN, TAG_ADJ_MIN, TAG_ADJ_MAX = 20.0, -15.0, 10.0
CAL_MIN_N, CAL_MIN_ACCEPTED = 20, 5
SPLIT_MIN_N = 30
SPLIT_BLEND = 0.4
SPLIT_FLOOR = 0.05
TP_R = [1, 2, 4, 7, 10]


def highest_tp(sig: dict) -> int:
    hit = [int(l["label"][2:]) for l in sig.get("levels", [])
           if str(l.get("label", "")).startswith("TP") and l.get("hit")]
    if hit:
        return max(hit)
    lbl = sig.get("highest_hit_tp_label")
    return int(lbl[2:]) if lbl and lbl.startswith("TP") else 0


class Shadow:
    def __init__(self, path: Path, learner, log_event):
        self.path, self.learner, self.log = path, learner, log_event
        self.d = {"archive": {}, "pending": {}, "since_opt": 0}
        if path.exists():
            self.d.update(json.loads(path.read_text()))

    def save(self):
        if len(self.d["archive"]) > ARCHIVE_CAP:
            items = sorted(self.d["archive"].items(), key=lambda kv: kv[1].get("closed_at") or "")
            self.d["archive"] = dict(items[-ARCHIVE_CAP:])
        cutoff = time.time() - 3 * 86400
        self.d["pending"] = {k: v for k, v in self.d["pending"].items() if v["ts"] > cutoff}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.d, ensure_ascii=False))
        tmp.replace(self.path)

    def snapshot(self, sig: dict, ev: dict, age_min: float):
        if sig["id"] in self.d["pending"] or sig["id"] in self.d["archive"] or age_min > SNAPSHOT_MAX_AGE_MIN:
            return
        self.d["pending"][sig["id"]] = {"ts": time.time(), "score": ev["score"],
                                       "factors": ev["factors"], "ok": ev["ok"], "skip": ev["skip"]}

    def ingest(self, signals: list) -> int:
        new = 0
        for s in signals:
            if s.get("status") != "CLOSED" or s.get("realized_r") is None or s["id"] in self.d["archive"]:
                continue
            snap = self.d["pending"].pop(s["id"], None)
            self.d["archive"][s["id"]] = {
                "symbol": s["symbol"], "r": float(s["realized_r"]), "tags": S.intrinsic_tags(s),
                "highest_tp": highest_tp(s), "closed_at": s.get("closed_at"), "exit": s.get("exit_reason"),
                "cvd_ratio": S.cvd_ratio_of(s), "snap": snap,
            }
            new += 1
        self.d["since_opt"] += new
        return new

    # ---------------------------------------------------------------- optimizers
    def _tag_adjustments(self, rows: list) -> tuple[dict, list]:
        overall = sum(r["r"] for r in rows) / len(rows)
        acc: dict[str, list] = {}
        for r in rows:
            for t in r["tags"]:
                acc.setdefault(t, []).append(r["r"])
        adj, evidence = {}, []
        for t, v in acc.items():
            if len(v) < TAG_MIN_N:
                continue
            edge = (sum(v) / len(v) - overall) * len(v) / (len(v) + TAG_SHRINK)
            a = round(max(TAG_ADJ_MIN, min(TAG_ADJ_MAX, edge * TAG_GAIN)), 1)
            if abs(a) >= 1:
                adj[t] = a
                evidence.append(f"{t}: n={len(v)} 平均 {sum(v) / len(v):+.2f}R (全體 {overall:+.2f}R) → {a:+.1f}")
        return adj, evidence

    def _calibrate_threshold(self, rows: list) -> tuple[int | None, str]:
        snaps = [(r["snap"]["score"], r["r"]) for r in rows if r.get("snap")]
        if len(snaps) < CAL_MIN_N:
            return None, f"門檻校準樣本 {len(snaps)}/{CAL_MIN_N}，尚不足"
        best_t, best = None, None
        for t in range(40, 76, 2):
            taken = [r for s, r in snaps if s >= t]
            if len(taken) < CAL_MIN_ACCEPTED:
                continue
            total = sum(taken)
            if best is None or total > best:
                best_t, best = t, total
        if best_t is None:
            return None, "門檻校準: 無足夠通過樣本"
        n = sum(1 for s, _ in snaps if s >= best_t)
        return best_t, f"門檻 {best_t}: 通過 {n}/{len(snaps)} 筆，合計 {best:+.2f}R"

    def _tp_split(self, rows: list) -> tuple[list | None, str]:
        if len(rows) < SPLIT_MIN_N:
            return None, f"TP 分配樣本 {len(rows)}/{SPLIT_MIN_N}，尚不足"
        ev = []
        for i in range(1, 6):
            pay = []
            for r in rows:
                h = r["highest_tp"]
                if h >= i:
                    pay.append(TP_R[i - 1])
                elif h == 0:
                    pay.append(-1.0)
                elif h == 1:
                    pay.append(0.0)
                else:
                    pay.append(TP_R[h - 2])
            ev.append(sum(pay) / len(pay))
        lo = min(ev)
        raw = [e - lo for e in ev]
        tot = sum(raw)
        learned = [x / tot for x in raw] if tot > 0 else list(S.TP_SPLIT)
        split = [(1 - SPLIT_BLEND) * d + SPLIT_BLEND * l for d, l in zip(S.TP_SPLIT, learned)]
        split = [max(SPLIT_FLOOR, x) for x in split]
        s = sum(split)
        split = [round(x / s, 3) for x in split]
        reach = [sum(1 for r in rows if r["highest_tp"] >= i) / len(rows) for i in range(1, 6)]
        return split, ("各 TP 到達率 " + " ".join(f"TP{i + 1}={p:.0%}" for i, p in enumerate(reach))
                       + " | 每單位期望 " + " ".join(f"{e:+.2f}R" for e in ev))

    def _calibrate_cvd(self, rows: list) -> tuple[float, str]:
        """Largest buy/sell ratio cut whose skipped side lost in both halves of the archive."""
        rows = sorted((r for r in rows if r.get("cvd_ratio") is not None), key=lambda r: r.get("closed_at") or "")
        if len(rows) < 100:
            return S.CVD_RATIO_MAX_DEFAULT, f"CVD 門檻樣本 {len(rows)}/100，用預設 {S.CVD_RATIO_MAX_DEFAULT}"
        mid = len(rows) // 2
        best = None
        for cut in (1.0, 1.25, 1.5, 2.0):
            skip = [r for r in rows if r["cvd_ratio"] >= cut]
            keep = [r for r in rows if r["cvd_ratio"] < cut]
            if len(skip) < 30 or len(keep) < 20:
                continue
            mean = lambda xs: sum(r["r"] for r in xs) / len(xs)
            h1 = [r for r in rows[:mid] if r["cvd_ratio"] >= cut]
            h2 = [r for r in rows[mid:] if r["cvd_ratio"] >= cut]
            if not h1 or not h2 or mean(h1) >= 0 or mean(h2) >= 0 or mean(skip) >= mean(keep):
                continue
            total = sum(r["r"] for r in keep)
            if best is None or total > best[0]:
                best = (total, cut, len(keep), mean(keep), len(skip), mean(skip))
        if best is None:
            return S.CVD_RATIO_MAX_DEFAULT, "CVD 門檻: 沒有切點通過 (前後兩段被擋的都要是虧的)，用預設"
        _, cut, nk, mk, ns, ms = best
        return cut, (f"CVD 買賣量比 < {cut}: 放行 {nk} 筆平均 {mk:+.2f}R、擋下 {ns} 筆平均 {ms:+.2f}R")

    def optimize(self, force: bool = False):
        if not force and self.d["since_opt"] < REOPT_EVERY:
            return
        rows = list(self.d["archive"].values())
        if len(rows) < TAG_MIN_N:
            return
        p = self.learner.p
        changes = []

        adj, evidence = self._tag_adjustments(rows)
        old_adj = p.get("shadow_adj", {})
        for t in sorted(set(adj) | set(old_adj)):
            if abs(adj.get(t, 0) - old_adj.get(t, 0)) >= 0.5:
                changes.append(f"特徵 [{t}] {old_adj.get(t, 0):+.1f} → {adj.get(t, 0):+.1f}")
        p["shadow_adj"] = adj

        cal, cal_note = self._calibrate_threshold(rows)
        if cal is not None and cal != p.get("calibrated_min_score"):
            changes.append(f"校準門檻 {p.get('calibrated_min_score')} → {cal}")
        p["calibrated_min_score"] = cal

        cvd_cut, cvd_note = self._calibrate_cvd(rows)
        if cvd_cut != p.get("cvd_ratio_max", S.CVD_RATIO_MAX_DEFAULT):
            changes.append(f"CVD 買賣量比上限 {p.get('cvd_ratio_max', S.CVD_RATIO_MAX_DEFAULT)} → {cvd_cut}")
        p["cvd_ratio_max"] = cvd_cut

        # Suggestion only: its payoff model ignores the stop moves after TP1; orders use S.TP_SPLIT.
        split, split_note = self._tp_split(rows)
        p.pop("tp_split", None)
        if split:
            split_note = f"建議 {split}（未套用，使用 {S.TP_SPLIT}）；{split_note}"

        self.learner.refresh_effective()
        self.d["since_opt"] = 0
        wins = sum(1 for r in rows if r["r"] > 0)
        self.log("SHADOW_OPTIMIZE", samples=len(rows), win_rate=round(wins / len(rows), 3),
                 avg_r=round(sum(r["r"] for r in rows) / len(rows), 3), changes=changes or ["無變動"],
                 evidence=evidence, threshold=cal_note, cvd_ratio=cvd_note, tp_split=split_note,
                 effective_min_score=p["effective_min_score"])
        self.learner.save()
