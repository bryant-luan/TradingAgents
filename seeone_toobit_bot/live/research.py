"""Indicator research on the bot's own See One analysis-review records.

Each resolved review (both sides of every symbol the self-scanner reviewed) is replayed on Toobit 5m bars with the
live exit policy (TP ladder, break-even at 2R, lock 1.6R at 3R, trail), fees included, giving its outcome in R.
For every candidate entry-time indicator and every quartile cut, the script asks: if trades on one side of the cut
were skipped, would expectancy rise, and does that hold in both halves of the period, on enough distinct hours,
and on longs and shorts separately? Results go to logs/research_latest.txt and one line per run to
logs/research_history.jsonl, so a finding has to persist across runs before it is worth trading on.
It also tracks the live entry filters out of sample (records after each filter went live).

    python3 -m live.research
"""
import bisect
import concurrent.futures as cf
import json
import math
import statistics as st
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from . import strategy as S

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
OUT_TXT = LOGS / "research_latest.txt"
HISTORY = LOGS / "research_history.jsonl"
TPE = timezone(timedelta(hours=8))

MIN_PROB = 50.0
DEDUPE_S = 1800
HORIZON_BARS = 288
FEE_RT, BE_BUF = 0.0011, 0.0011
TPS, SPLIT = [1.0, 1.5, 2.0, 3.0, 4.0], [0.375, 0.218, 0.227, 0.09, 0.09]
# A cut is "robust" only if the skipped side loses in both halves, spans enough hours, and is not one-sided.
MIN_SKIP_N, MIN_SKIP_HOURS, MIN_SIDE_N = 30, 8, 15
LIVE_FILTERS = {  # name -> (went live UTC, predicate on feature row meaning "the live filter would skip it")
    "大週期方向 ≥ 1.5": ("2026-11-05T07:24:00+00:00", lambda f: f["htf"] is not None and f["htf"] >= 1.5),
    "BTC 1h 方向 < 0": ("2026-11-05T08:00:00+00:00",
                        lambda f: f["btc1h_dir_pct"] is not None and f["btc1h_dir_pct"] < 0),
    "近 1h/24h 波動比 < 0.8": ("2026-11-06T01:10:00+00:00",
                              lambda f: f["atr_ratio"] is not None and f["atr_ratio"] < S.ATR_RATIO_MIN_DEFAULT),
    "1h 相對 BTC ≥ 0.924": ("2026-11-06T01:10:00+00:00",
                            lambda f: f["rs1h_vs_btc"] is not None and f["rs1h_vs_btc"] >= S.RS1H_MAX_DEFAULT),
}
# Features already gated live (BTC 4h measured redundant with BTC 1h inside the live gates).
LIVE_FEATURES = {"htf", "btc1h_dir_pct", "m15", "rpos", "prob", "rsi15_dir", "atr_ratio", "rs1h_vs_btc"}
REDUNDANT = {"btc4h_dir_pct": "與 BTC 1h 重複，現行條件內無額外效果 (11-05 驗證)"}
NAMES = {
    "prob": "分析驗算機率", "m15": "See One 15m 方向", "htf": "1h/4h 大方向", "rpos": "3h 區間位置",
    "stop_pct": "止損距離 %", "ret1h_R": "本幣 1h 走勢 (R)", "ret4h_R": "本幣 4h 走勢 (R)",
    "ret24h_pct": "本幣 24h 漲跌 %", "vol_ratio_1h": "1h 成交量 / 均量", "rsi15_dir": "15m RSI (方向)",
    "ema50_dist_atr": "距 15m EMA50 (ATR)", "atr_pct": "15m ATR %", "atr_ratio": "近 1h / 24h 波動比",
    "btc1h_dir_pct": "BTC 1h (方向) %", "btc4h_dir_pct": "BTC 4h (方向) %", "rs1h_vs_btc": "1h 相對 BTC 強弱 %",
    "dist24h_hi_atr": "距 24h 極值 (ATR)", "wick_dir": "近 3 根逆向影線比", "streak_dir": "15m 連續順向根數",
            "quote_vol_24h": "24h 成交額", "side_long": "做多=1",
    "px15_dir_pct": "15m 漲跌 (方向) %", "oi15": "15m 持倉量變化 %",
}


def klines(sym: str) -> tuple[str, list]:
    out, end = [], None
    for _ in range(2):
        p = {"symbol": sym, "interval": "5m", "limit": 1000}
        if end:
            p["endTime"] = end
        try:
            d = requests.get("https://api.toobit.com/quote/v1/klines", params=p, timeout=15).json()
        except (requests.RequestException, ValueError):
            break
        if not d:
            break
        out = [(int(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5])) for x in d] + out
        end = int(d[0][0]) - 1
    return sym, sorted(set(out))


def records() -> list:
    learned = json.loads((LOGS / "learned_params.json").read_text()) if (LOGS / "learned_params.json").exists() else {}
    m15_max = learned.get("m15_max", S.M15_MAX_DEFAULT)
    htf_min = learned.get("htf_min", S.HTF_MIN_DEFAULT)
    rpos_max = learned.get("range_pos_max", S.MAX_RANGE_POS)
    res = json.loads((LOGS / "self_calib.json").read_text())["resolved"]
    # Only gates that are not under study here; the htf cap and RSI floor are left open so both sides are measured.
    g = sorted((r for r in res if r["prob"] >= MIN_PROB
                and (r.get("m15") is None or r["m15"] <= m15_max)
                and (r.get("htf") is None or r["htf"] >= htf_min)
                and (r.get("rpos") is None or r["rpos"] <= rpos_max)), key=lambda r: r["ts"])
    last, out = {}, []
    for r in g:
        k = (r["symbol"], r["side"])
        if r["ts"] - last.get(k, 0) >= DEDUPE_S:
            last[k] = r["ts"]
            out.append(r)
    return out


class Lab:
    def __init__(self, recs: list):
        syms = sorted({r["tsym"] for r in recs} | {"BTC-SWAP-USDT"})
        with cf.ThreadPoolExecutor(8) as ex:
            self.K = dict(ex.map(klines, syms))
        self.idx = {s: [x[0] for x in rows] for s, rows in self.K.items()}

    def at(self, s: str, ts: float) -> int:
        return bisect.bisect_left(self.idx[s], int(ts * 1000))

    def outcome(self, r: dict) -> float | None:
        s = r["tsym"]
        rows = self.K[s][self.at(s, r["ts"]):self.at(s, r["ts"]) + HORIZON_BARS]
        if len(rows) < 12:
            return None
        sgn = 1 if r["side"] == "LONG" else -1
        e, d = r["entry"], abs(r["entry"] - r["stop"])
        if d <= 0:
            return None
        be = BE_BUF * e / d
        left, got, stop_r, mfe, hit = 1.0, 0.0, -1.0, 0.0, [False] * 5
        for _, o, h, l, c, v in rows:
            fav = ((h if sgn > 0 else l) - e) * sgn / d
            adv = ((l if sgn > 0 else h) - e) * sgn / d
            if adv <= stop_r:  # same-bar ambiguity resolved against us
                got += left * stop_r
                left = 0
                break
            for j, t in enumerate(TPS):
                if not hit[j] and fav >= t:
                    hit[j] = True
                    q = min(SPLIT[j], left)
                    got += q * t
                    left -= q
            if left <= 1e-9:
                break
            mfe = max(mfe, fav)
            for trig, lk in S.LOCK_LADDER:
                if mfe >= trig:
                    stop_r = max(stop_r, be if lk == 0 else lk)
            if mfe >= S.LOCK_LADDER[-1][0]:
                stop_r = max(stop_r, math.floor((mfe - S.TRAIL_GAP_R) / S.TRAIL_STEP_R) * S.TRAIL_STEP_R)
        if left > 0:
            got += left * ((rows[-1][4] - e) * sgn / d)
        return got - FEE_RT * e / d

    def features(self, r: dict) -> dict | None:
        s = r["tsym"]
        i = self.at(s, r["ts"])
        if i < 300:
            return None
        past = self.K[s][i - 300:i]
        sgn = 1 if r["side"] == "LONG" else -1
        e, d = r["entry"], abs(r["entry"] - r["stop"])
        c = [x[4] for x in past]
        qv = [x[5] * x[4] for x in past]
        c15 = c[2::3]
        diffs = [(b - a) * sgn for a, b in zip(c15[-15:], c15[-14:])]
        up, dn = sum(max(0, x) for x in diffs), sum(max(0, -x) for x in diffs)
        ema = c15[0]
        for x in c15:
            ema += (x - ema) * 2 / 51
        tr15 = [max(x[2] for x in past[k:k + 3]) - min(x[3] for x in past[k:k + 3]) for k in range(len(past) - 42, len(past), 3)]
        atr = sum(tr15) / len(tr15)
        rng = lambda bars: sum(x[2] - x[3] for x in bars) / len(bars)
        bi = self.at("BTC-SWAP-USDT", r["ts"])
        b = [x[4] for x in self.K["BTC-SWAP-USDT"][max(0, bi - 48):bi]]
        btc1h = (b[-1] - b[-12]) / b[-12] * 100 if len(b) >= 12 else None
        hi24, lo24 = max(x[2] for x in past[-288:]), min(x[3] for x in past[-288:])
        ext = (hi24 - e) if sgn > 0 else (e - lo24)
        last3 = past[-3:]
        wick = sum((x[2] - max(x[1], x[4])) if sgn > 0 else (min(x[1], x[4]) - x[3]) for x in last3)
        streak = 0
        for a, bb in zip(reversed(c15[:-1]), reversed(c15[1:])):
            if (bb - a) * sgn > 0:
                streak += 1
            else:
                break
        ret1h_pct = (e - c[-12]) / c[-12] * 100
        px15 = (c[-1] - c[-4]) / c[-4] * 100 * sgn if len(c) >= 4 and c[-4] else None
        return {
            "prob": r["prob"], "m15": r.get("m15"), "htf": r.get("htf"), "rpos": r.get("rpos"),
            "stop_pct": d / e * 100,
            "ret1h_R": (e - c[-12]) * sgn / d, "ret4h_R": (e - c[-48]) * sgn / d,
            "ret24h_pct": (e - c[0]) * sgn / c[0] * 100,
            "vol_ratio_1h": (sum(qv[-12:]) / 12) / ((sum(qv[:-12]) / len(qv[:-12])) or 1),
            "rsi15_dir": 50.0 if up + dn == 0 else 100 * up / (up + dn),
            "ema50_dist_atr": (e - ema) * sgn / atr if atr else None,
            "atr_pct": atr / e * 100,
            "atr_ratio": rng(past[-12:]) / (rng(past) or 1e-12),
            "btc1h_dir_pct": None if btc1h is None else btc1h * sgn,
            "btc4h_dir_pct": (b[-1] - b[0]) / b[0] * 100 * sgn if len(b) >= 48 else None,
            "rs1h_vs_btc": None if btc1h is None else (ret1h_pct - btc1h) * sgn,
            "dist24h_hi_atr": ext / atr if atr else None,
            "wick_dir": wick / (sum(x[2] - x[3] for x in last3) or 1e-12),
            "streak_dir": streak,
            "quote_vol_24h": sum(qv[-288:]),
            "side_long": 1 if sgn > 0 else 0,
            "px15_dir_pct": None if px15 is None else round(px15, 3),
            "oi15": r.get("oi15"),
        }


def mean(xs):
    return st.mean(xs) if xs else None


def scan_cuts(rows: list, name: str, mid: float) -> list:
    vals = sorted(f[name] for _, _, f in rows if f[name] is not None)
    if len(vals) < 60 or vals[0] == vals[-1]:
        return []
    out = []
    for q in (0.25, 0.5, 0.75):
        cut = vals[int(len(vals) * q)]
        for skip_low in (True, False):
            skip = [(t, o, f) for t, o, f in rows if f[name] is not None and ((f[name] < cut) if skip_low else (f[name] >= cut))]
            keep = [(t, o, f) for t, o, f in rows if f[name] is None or not ((f[name] < cut) if skip_low else (f[name] >= cut))]
            if len(skip) < MIN_SKIP_N or len(keep) < MIN_SKIP_N:
                continue
            sk, kp = mean([o for _, o, _ in skip]), mean([o for _, o, _ in keep])
            halves = [mean([o for t, o, _ in skip if (t < mid) == h]) for h in (True, False)]
            khalves = [mean([o for t, o, _ in keep if (t < mid) == h]) for h in (True, False)]
            sides = []
            for side in (1, 0):
                s_ = [o for _, o, f in skip if f["side_long"] == side]
                k_ = [o for _, o, f in keep if f["side_long"] == side]
                if len(s_) >= MIN_SIDE_N and len(k_) >= MIN_SIDE_N:
                    sides.append(mean(s_) < mean(k_))
            hours = len({int(t // 3600) for t, _, _ in skip})
            robust = (sk < 0 and sk < kp and all(h is not None and h < 0 for h in halves)
                      and all(k is not None and h < k for h, k in zip(halves, khalves))
                      and hours >= MIN_SKIP_HOURS and len(sides) == 2 and all(sides))
            out.append({"feature": name, "rule": f"{'<' if skip_low else '≥'} {cut:.3g} 不進場",
                        "skip_n": len(skip), "skip_hours": hours, "skip_r": round(sk, 3), "keep_r": round(kp, 3),
                        "skip_halves": [None if h is None else round(h, 2) for h in halves],
                        "keep_halves": [None if h is None else round(h, 2) for h in khalves],
                        "sides_agree": f"{sum(sides)}/{len(sides)}", "robust": robust,
                        "gain_total_r": round(-sum(o for _, o, _ in skip), 1)})
    return out


def main():
    t0 = time.time()
    recs = records()
    lab = Lab(recs)
    rows = []
    for r in recs:
        o, f = lab.outcome(r), lab.features(r)
        if o is not None and f is not None:
            rows.append((r["ts"], o, f))
    if len(rows) < 100:
        print("樣本不足")
        return
    (LOGS / "research_rows.json").write_text(json.dumps([{"ts": t, "r": o, **f} for t, o, f in rows]))
    mid = sorted(t for t, _, _ in rows)[len(rows) // 2]
    base = mean([o for _, o, _ in rows])
    hours = len({int(t // 3600) for t, _, _ in rows})
    span = (datetime.fromtimestamp(rows[0][0], TPE).strftime("%m-%d %H:%M"),
            datetime.fromtimestamp(rows[-1][0], TPE).strftime("%m-%d %H:%M"))
    cuts = [c for name in NAMES for c in scan_cuts(rows, name, mid)]
    robust = sorted((c for c in cuts if c["robust"]), key=lambda c: -c["gain_total_r"])
    near = sorted((c for c in cuts if not c["robust"] and c["skip_r"] < 0 and c["skip_r"] < c["keep_r"]),
                  key=lambda c: -c["gain_total_r"])[:6]

    live = {}
    for name, (since, skip) in LIVE_FILTERS.items():
        ts = datetime.fromisoformat(since).timestamp()
        post = [(o, f) for t, o, f in rows if t >= ts]
        sk = [o for o, f in post if skip(f)]
        kp = [o for o, f in post if not skip(f)]
        live[name] = {"since": since, "skipped_n": len(sk), "skipped_r": mean(sk) and round(mean(sk), 3),
                      "kept_n": len(kp), "kept_r": mean(kp) and round(mean(kp), 3)}

    lines = [f"== 指標研究 {datetime.now(TPE):%m-%d %H:%M} (台北)，樣本 {len(rows)} 筆，{hours} 小時 "
             f"({span[0]} ~ {span[1]})，基準 {base:+.3f}R/筆",
             "已上線條件的樣本外表現 (上線後的紀錄；被擋的平均應為負):"]
    for name, v in live.items():
        lines.append(f"  {name}: 被擋 {v['skipped_n']} 筆 平均 {v['skipped_r']}R / 放行 {v['kept_n']} 筆 平均 {v['kept_r']}R")
    lines.append(f"穩健候選 ({len(robust)})：被擋的一側前後兩段都虧、涵蓋 ≥{MIN_SKIP_HOURS} 小時、多空一致")
    for c in robust[:8]:
        lines.append(f"  {NAMES[c['feature']]} {c['rule']}: 被擋 {c['skip_n']} 筆/{c['skip_hours']} 小時 平均 {c['skip_r']:+.2f}R "
                     f"(前後 {c['skip_halves']}) → 放行平均 {c['keep_r']:+.2f}R，多空 {c['sides_agree']}，總省 {c['gain_total_r']}R")
    prev = [json.loads(x) for x in HISTORY.read_text(encoding="utf-8").splitlines()[-2:]] if HISTORY.exists() else []
    key = lambda c: (c["feature"], c["rule"].split()[0])  # feature + direction; the cut value drifts between runs
    runs = [{key(c) for c in h.get("robust", [])} for h in prev] + [{key(c) for c in robust}]
    persistent = set.intersection(*runs) if len(runs) == 3 else set()
    lines.append("連續 3 次穩健 (指標+方向): " + ("、".join(
        f"{NAMES[f]} {d}" + (" [已上線]" if f in LIVE_FEATURES else f" [{REDUNDANT[f]}]" if f in REDUNDANT else "") for f, d in sorted(persistent)) or "無"))
    lines.append("接近但未達穩健:")
    for c in near:
        lines.append(f"  {NAMES[c['feature']]} {c['rule']}: 被擋 {c['skip_n']} 筆/{c['skip_hours']} 小時 平均 {c['skip_r']:+.2f}R "
                     f"(前後 {c['skip_halves']})，多空 {c['sides_agree']}")
    lines.append(f"(耗時 {time.time() - t0:.0f}s)")
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with HISTORY.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "n": len(rows),
                            "hours": hours, "base_r": round(base, 3), "live": live,
                            "robust": [{k: c[k] for k in ("feature", "rule", "skip_n", "skip_hours", "skip_r", "keep_r")}
                                       for c in robust]}, ensure_ascii=False) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
