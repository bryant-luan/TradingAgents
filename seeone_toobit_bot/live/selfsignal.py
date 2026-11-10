"""Self-sourced signals: pick coins from See One boards, plan the trade, verify it with See One 分析驗算.

Flow per scan:
  1. Universe = potential board (scanner) + strength board + customized-recipe hits + majors.
  2. Direction = board / recipe / scanner hint, else the market regime.
  3. Geometry from Toobit 15m candles: stop = max(1.5 x ATR14, fee floor), capped; TPs at fixed R multiples.
  4. POST /api/analysis-review (15M) -> per-target hit probabilities, trade quality, factor breakdown.
     Survivors are confirmed on 1H.
  5. Expected value in R from the per-target probabilities and the bot's TP split; trade if above threshold.
Every review is also stored and later resolved against Toobit candles (TP1 or SL first) to calibrate the
probability threshold, so the filter tightens or loosens based on how well 分析驗算 actually predicts.
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from . import strategy as S
from .market import klines, listing, quote_volumes
from .localreview import LocalReview

SCAN_EVERY_S = 900          # one analysis-review pass every 15 minutes
MAJORS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT"]
ATR_MULT = 1.5
MIN_STOP_PCT = 0.008
MAX_STOP_PCT = 0.06
TP_R = [1.0, 1.5, 2.0, 3.0, 4.0]
DEFAULT_MIN_PROB = 60.0          # de-biased probability of TP1 before SL
# On the ELSA set, reviews with an edge of 25 or more won 49% at +0.20R (37 rows).
# Reviews under 25 won 72% at +0.73R (174 rows). The 15–25 band was the best:
# 64 rows, 84% wins, +0.96R. A floor of 25 keeps the weak tail and blocks that band.
# The chosen side still has to be ahead (edge >= 0).
MIN_EDGE = 0.0
# The review finds several names. Trying only the top three dropped XRP and RUNE on
# 11-08 06:10 Taipei after the higher-scoring names could not be sized. One same-side
# gap still allows a single new position.
MAX_PER_SCAN = 6
# High-conviction tier = setups like the CLANKER trade: strong two-sided edge, high de-biased probability,
# high trade quality, 1H agreement, and the coin is on the strength board or potential board top 10.
HIGH_MIN_EDGE, HIGH_MIN_PROB, HIGH_MIN_QUALITY, HIGH_MIN_1H = 35.0, 68.0, 75.0, 60.0
# The enlarged size is only used while gated reviews at HIGH_MIN_PROB+ actually beat the rest.
HIGH_SIZE_PASS = 0.55
HIGH_SIZE_MARGIN = 0.05
EXT_BARS = 8                     # 2h of 15m bars
RANGE_BARS = 12                  # 3h of closed 15m bars for the entry range position
# Range position in the trade direction (1 = at the 3h extreme the trade is moving toward); the entry cap is
# the top of the highest contiguous bucket that still passes, same rules as the 15m cap.
BTC_COUNTER_ATR = 2.0            # BTC off its 2h extreme against the trade by this many ATRs = market turning
EXT_TAG_R = 2.0
MIN_QUALITY = 60.0
MIN_EV_R = 0.20
CONFIRM_MIN_PROB = 50.0
REVIEW_COOLDOWN_S = 900
# See One answered ~200 reviews/hour for two days before the Nov 6 11:52 burst; six per scan per bot
# left both books near 48/hour. Ten per scan keeps the 15s gap and stays well under the old pace.
REVIEWS_PER_SCAN = 10
# One side plus the saved bias replaces a two-sided review; every BIAS_EVERY-th name still reviews both.
BIAS_EVERY = 6
# Both bots review the same See One symbols on one account. A fresh review with a similar stop width is reused.
SHARED_REVIEW_FILE = S.ENV_FILE.parent / "logs" / "review_cache.json"
SHARED_REVIEW_TTL_S = 600
SHARED_REVIEW_SL_TOL = 0.15
# Self-scan longs open only in an up-trending market whose recent long reviews are working.
# Over Nov 4-10 long reviews lost on the replay in every bucket, including 1h/4h >= +1 (-0.5R to -0.65R),
# but almost none of that week was an up-trend, so the gate learns from fresh resolved reviews instead.
LONG_TREND_HTF = 0.5
LONG_GATE_HOURS = 48
LONG_GATE_MIN_N = 20
LONG_GATE_HIT = 0.60
# The user chose to open longs on the market trend alone; the recent long record is logged, not enforced.
LONG_GATE_REQUIRE_RECORD = False
RECHECK_OPEN_S = 900
EXIT_PROB_BELOW = 40.0
EXIT_PROB_DROP = 25.0        # close when the next-target probability falls this many points from entry
# See One vs Toobit price ratio: cached per symbol, re-aligned when the levels sent are off by more than BASIS_TOL_R.
BASIS_TTL_S = 600
BASIS_TOL_R = 0.05
BASIS_LOG_PCT = 0.003
MAX_HOLD_H = 24
CAL_BUCKETS = [(50, 60), (60, 70), (70, 80), (80, 101)]
CAL_MIN_N = 15
CAL_MIN_HIT = 0.55
CAL_RESOLVE_MAX_H = 24
# See One 15m direction score signed to the trade side, bucketed; the entry cap is the top of the highest
# bucket whose reviews still reach TP1 first at least M15_PASS of the time.
M15_BUCKETS = [(-99.0, -1.0), (-1.0, 1.0), (1.0, 2.5), (2.5, 99.0)]
M15_MIN_N = 20
M15_PASS = 0.50
# Average of the 1h and 4h See One direction scores, signed to the trade side; the entry floor is the
# bottom of the lowest bucket that passes, with the same sample and pass-rate rules.
HTF_BUCKETS = [(-99.0, -0.5), (-0.5, 0.5), (0.5, 99.0)]
# Cap side of the same score: the cap is the top of the highest bucket reached while every bucket below passes.
HTF_CAP_BUCKETS = [(-99.0, 0.5), (0.5, 1.5), (1.5, 2.5), (2.5, 99.0)]
# Reviews taken in the same scan all ride the same market move, so sample counts overstate the evidence.
# The gates keep their defaults until the records span this many distinct hours.
GATE_MIN_HOURS = 24
# ~190 resolved reviews per active hour: keep several days so the gates can ever reach GATE_MIN_HOURS,
# and calibrate on a time window instead of a fixed record count.
RESOLVED_KEEP = 20000
CAL_WINDOW_H = 72


def atr(symbol: str, n: int = 14) -> tuple[float, list] | None:
    rows = klines(symbol, "15m", n + 1)
    if len(rows) < n + 1:
        return None
    trs = []
    for prev, cur in zip(rows, rows[1:]):
        h, l, pc = float(cur[2]), float(cur[3]), float(prev[4])
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return sum(trs) / len(trs), rows


def move_before(symbol: str, end_ms: int, side: str, risk: float) -> float | None:
    """Move over the 2h before end_ms in the trade direction, in units of the given stop distance."""
    rows = klines(symbol, "15m", EXT_BARS + 1, end_ms - 1)
    if len(rows) < EXT_BARS or not risk:
        return None
    sgn = 1 if side == "LONG" else -1
    return round((float(rows[-1][4]) - float(rows[-EXT_BARS][1])) * sgn / risk, 2)


def range_pos(rows: list, price: float) -> float | None:
    """Where price sits in the high-low range of the closed bars before the current one: 0 = low, 1 = high."""
    prior = rows[-RANGE_BARS - 1:-1]
    if len(prior) < RANGE_BARS:
        return None
    hi, lo = max(float(r[2]) for r in prior), min(float(r[3]) for r in prior)
    return None if hi <= lo else round((price - lo) / (hi - lo), 3)


def in_side(rpos_long: float, side: str) -> float:
    return rpos_long if side == "LONG" else 1 - rpos_long


def rsi_long(rows: list) -> float | None:
    """RSI over the 15m closes in rows (long perspective; a short's RSI in its own direction is 100 - this)."""
    c = [float(r[4]) for r in rows]
    if len(c) < 15:
        return None
    up = sum(max(0.0, b - a) for a, b in zip(c, c[1:]))
    dn = sum(max(0.0, a - b) for a, b in zip(c, c[1:]))
    return 50.0 if up + dn == 0 else round(100 * up / (up + dn), 1)


def btc_ret1h() -> float | None:
    """BTC's move over the last hour (4 closed-or-current 15m closes back), percent, long perspective."""
    a = atr("BTC-SWAP-USDT")
    if not a:
        return None
    rows = a[1]
    ref = float(rows[-5][4])
    return round((float(rows[-1][4]) - ref) / ref * 100, 3)


def micro_1h(symbol: str) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    """(range ratio, 1h return %, volume ratio, 24h quote volume, 15m return %) on closed 5m bars.

    Same definitions research.py scores. Range ratio is the mean high-low of the last 12 bars over the
    last 300 (~25h). The 1h return is the last close versus the close 12 bars earlier, long perspective.
    The 15m return is the last close versus the close 3 bars (15 minutes) earlier, also long perspective.
    Volume ratio is the last hour's mean quote volume over the mean of the 288 bars before it.
    Quote volume is base volume times close, summed over those 288 bars (24h).
    """
    empty = (None, None, None, None, None)
    rows = klines(symbol, "5m", 301)
    if not isinstance(rows, list) or len(rows) < 301:
        return empty
    past = rows[:-1][-300:]

    def mean_range(bars: list) -> float:
        return sum(float(x[2]) - float(x[3]) for x in bars) / len(bars)

    base = mean_range(past)
    ratio = None if base <= 0 else mean_range(past[-12:]) / base
    c0, c1 = float(past[-12][4]), float(past[-1][4])
    c15 = float(past[-4][4])
    ret = None if c0 <= 0 else (c1 - c0) / c0 * 100
    px15 = None if c15 <= 0 else (c1 - c15) / c15 * 100
    qv = [float(x[5]) * float(x[4]) for x in past]
    earlier = qv[:-12]
    vol_ratio = (sum(qv[-12:]) / 12) / ((sum(earlier) / len(earlier)) or 1)
    quote_vol = sum(qv[-288:])
    return ratio, ret, vol_ratio, quote_vol, px15


def contract_quote_vol() -> dict[str, float]:
    """24h quote volume in USDT for every USDT-M contract. One public call."""
    return quote_volumes()


def rsi_side(rsi_l: float | None, side: str) -> float | None:
    return None if rsi_l is None else (rsi_l if side == "LONG" else round(100 - rsi_l, 1))


def move_2h(symbol: str) -> dict | None:
    """Price move over the last 2h, in percent and in self-signal stop distances (1.5×ATR15m, floor 0.8%),
    plus the price's position in the 3h range."""
    a = atr(symbol)
    if not a:
        return None
    atr_v, rows = a
    price, start = float(rows[-1][4]), float(rows[-EXT_BARS][1])
    dist = max(ATR_MULT * atr_v, MIN_STOP_PCT * price)
    return {"pct": round((price - start) / start * 100, 2), "r": round((price - start) / dist, 2),
            "rpos": range_pos(rows, price), "rsi": rsi_long(rows)}


class SelfScanner:
    def __init__(self, so, tb, learner, calib_path: Path, log_event):
        self.so, self.tb, self.learner, self.log = so, tb, learner, log_event
        self.path = calib_path
        self.d = {"pending": [], "resolved": [], "last_review": {}}
        if calib_path.exists():
            self.d.update(json.loads(calib_path.read_text()))
        self.last_scan = 0.0
        self.last_break = 0.0
        self.last_spike = 0.0
        self.basis: dict[str, tuple[float, float]] = {}
        self.local = LocalReview(calib_path)
        self.sim_log = calib_path.parent / "local_review_sim.jsonl"

    def save(self):
        self.d["resolved"] = self.d["resolved"][-RESOLVED_KEEP:]
        cutoff = time.time() - 86400
        self.d["last_review"] = {k: v for k, v in self.d["last_review"].items() if v > cutoff}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.d, ensure_ascii=False))
        tmp.replace(self.path)

    # ------------------------------------------------------------------ See One 分析驗算
    def _review(self, symbol: str, side: str, stop: float, tps: list, tf: str) -> dict:
        body = {"symbol": symbol, "side": side, "timeframe": tf, "entry_mode": "MARKET", "entry_price": None,
                "take_profit_prices": [float(f"{p:.8g}") for p in tps], "stop_price": float(f"{stop:.8g}"),
                "language": "zh-TW"}
        return self.so.request("POST", "/api/analysis-review", deadline=S.REVIEW_DEADLINE_S, json=body)

    def _shared(self) -> dict:
        try:
            data = json.loads(SHARED_REVIEW_FILE.read_text())
        except (OSError, ValueError):
            return {}
        now = time.time()
        return {k: v for k, v in data.items() if now - v.get("ts", 0) < SHARED_REVIEW_TTL_S}

    def _share(self, key: str, sl_pct: float, rv: dict):
        data = self._shared()
        data[key] = {"ts": time.time(), "sl_pct": sl_pct, "rv": rv}
        tmp = SHARED_REVIEW_FILE.with_suffix(f".{id(self)}.tmp")
        try:
            tmp.write_text(json.dumps(data, ensure_ascii=False))
            tmp.replace(SHARED_REVIEW_FILE)
        except OSError:
            pass

    def review(self, symbol: str, side: str, stop: float, tps: list, tf: str = "15M", share: bool = False) -> dict:
        """share=True only for fresh plans, whose TPs are fixed R multiples from the current price."""
        px = self.tb.price(S.to_toobit(symbol))
        if not share:
            return self._review_aligned(symbol, side, stop, tps, tf, px)
        sl_pct = abs(px - stop) / px if px else 0.0
        key = f"{symbol}|{side}|{tf}"
        hit = self._shared().get(key)
        if hit and sl_pct and abs(hit["sl_pct"] - sl_pct) <= SHARED_REVIEW_SL_TOL * sl_pct:
            self.shared_hits = getattr(self, "shared_hits", 0) + 1
            return json.loads(json.dumps(hit["rv"]))
        rv = self._review_aligned(symbol, side, stop, tps, tf, px)
        if sl_pct:
            self._share(key, sl_pct, rv)
        return rv

    def _review_aligned(self, symbol: str, side: str, stop: float, tps: list, tf: str, px: float) -> dict:
        """分析驗算 on See One's own price basis: See One's market price can sit apart from Toobit's, so the Toobit
        stop/TPs are scaled by the See One/Toobit price ratio to keep the same R geometry."""
        risk = max(abs(px - stop), px * MIN_STOP_PCT)
        cached = self.basis.get(symbol)
        ratio = cached[0] if cached and time.time() - cached[1] < BASIS_TTL_S else 1.0
        rv = None
        for _ in range(3):
            try:
                rv = self._review(symbol, side, stop * ratio, [p * ratio for p in tps], tf)
            except requests.HTTPError as e:
                if e.response is None or e.response.status_code != 400 or "進場價" not in e.response.text:
                    raise
                wide = (px * 1e-3, [px * 1e3]) if side == "LONG" else (px * 1e3, [px * 1e-3])
                ratio = self._set_basis(symbol, self._review(symbol, side, wide[0], wide[1], tf), px) or ratio
                continue
            seeone_px = float((rv.get("geometry") or {}).get("currentPrice") or 0)
            if not seeone_px or abs(seeone_px - px * ratio) <= BASIS_TOL_R * risk * ratio:
                if seeone_px:
                    self._set_basis(symbol, rv, px)
                return rv
            ratio = self._set_basis(symbol, rv, px)
        if rv is None:
            raise requests.HTTPError(f"{symbol} 分析驗算無法對齊 See One 價格基準")
        return rv

    def _set_basis(self, symbol: str, rv: dict, px: float) -> float | None:
        sp = float((rv.get("geometry") or {}).get("currentPrice") or 0)
        if not sp or not px:
            return None
        ratio = sp / px
        prev = self.basis.get(symbol)
        self.basis[symbol] = (ratio, time.time())
        if abs(ratio - 1) >= BASIS_LOG_PCT and (not prev or abs(prev[0] - ratio) >= BASIS_LOG_PCT):
            self.log("SEEONE_BASIS", symbol=symbol, seeone_price=sp, toobit_price=px, gap_pct=round((ratio - 1) * 100, 2),
                     note="See One 與 Toobit 價格有落差；分析驗算的止盈/止損已按比例換算到 See One 價格，維持相同 R 距離")
        return ratio

    # ------------------------------------------------------------------ calibration of 分析驗算 probabilities
    def min_prob(self) -> float:
        return self.learner.p.get("self_min_prob", DEFAULT_MIN_PROB)

    def record(self, symbol: str, tsym: str, side: str, entry: float, stop: float, tp1: float, prob: float,
               m15: float | None = None, htf: float | None = None, rpos_long: float | None = None,
               rsi_l: float | None = None, btc1h_long: float | None = None,
               px15_long: float | None = None, oi15: float | None = None):
        sgn = 1 if side == "LONG" else -1
        self.d["pending"].append({"symbol": symbol, "tsym": tsym, "side": side, "entry": entry, "stop": stop,
                                  "tp1": tp1, "prob": prob, "ts": time.time(),
                                  "m15": None if m15 is None else round(m15 * sgn, 2),
                                  "htf": None if htf is None else round(htf * sgn, 2),
                                  "rpos": None if rpos_long is None else round(in_side(rpos_long, side), 2),
                                  "rsi": rsi_side(rsi_l, side),
                                  "btc1h": None if btc1h_long is None else round(btc1h_long * sgn, 3),
                                  "px15": None if px15_long is None else round(px15_long * sgn, 3),
                                  "oi15": None if oi15 is None else round(oi15, 3)})

    def resolve(self):
        still, new = [], 0
        for p in self.d["pending"]:
            age_h = (time.time() - p["ts"]) / 3600
            if age_h < 0.25:
                still.append(p)
                continue
            rows = klines(p["tsym"], "15m", 100)
            if not rows:
                still.append(p)
                continue
            outcome = None
            for k in rows:
                if int(k[0]) < p["ts"] * 1000 - 900_000:
                    continue
                hi, lo = float(k[2]), float(k[3])
                hit_tp = hi >= p["tp1"] if p["side"] == "LONG" else lo <= p["tp1"]
                hit_sl = lo <= p["stop"] if p["side"] == "LONG" else hi >= p["stop"]
                if hit_sl:
                    outcome = 0
                    break
                if hit_tp:
                    outcome = 1
                    break
            if outcome is None and age_h < CAL_RESOLVE_MAX_H:
                still.append(p)
                continue
            if outcome is not None:
                self.d["resolved"].append({**p, "hit": outcome, "resolved_ts": time.time()})
                new += 1
        self.d["pending"] = still
        if new:
            self.calibrate()

    def calibrate(self):
        cutoff = time.time() - CAL_WINDOW_H * 3600
        rows = [r for r in self.d["resolved"] if r["ts"] >= cutoff]
        table, chosen = [], None
        for lo, hi in CAL_BUCKETS:
            b = [r["hit"] for r in rows if lo <= r["prob"] < hi]
            rate = sum(b) / len(b) if b else None
            table.append((lo, hi, len(b), round(rate, 3) if rate is not None else None))
        for lo, hi, n, rate in table:
            if n >= CAL_MIN_N and rate is not None and rate >= CAL_MIN_HIT:
                chosen = float(lo)
                break
        if chosen is None and all(n >= CAL_MIN_N for _, _, n, _ in table):
            chosen = 80.0
        old = self.min_prob()
        if chosen is not None:
            self.learner.p["self_min_prob"] = chosen
            self.learner.p["self_calibrated"] = chosen < 80.0
            self.learner.save()
        self.log("SELF_CALIBRATE", buckets=[{"prob": f"{lo}-{hi}", "n": n, "tp1_first_rate": r} for lo, hi, n, r in table],
                 min_prob_old=old, min_prob_new=self.min_prob(),
                 note="分析驗算預估機率 vs 實際先到 TP1 的比例；樣本足夠時自動調整最低機率門檻")
        self.calibrate_m15(rows)
        self.calibrate_htf(rows)
        self.calibrate_htf_max(rows)
        self.calibrate_rpos(rows)
        self.calibrate_rsi(rows)
        self.calibrate_btc1h(rows)
        self.calibrate_atr_ratio(rows)
        self.calibrate_rs1h(rows)
        self.calibrate_vol_ratio(rows)
        self.calibrate_px15(rows)
        self.calibrate_quote_vol(rows)
        self.calibrate_high(rows)

    def calibrate_high(self, rows: list):
        p = self.learner.p
        g = [r for r in rows
             if (r.get("m15") is None or r["m15"] <= p.get("m15_max", S.M15_MAX_DEFAULT))
             and (r.get("htf") is None or p.get("htf_min", S.HTF_MIN_DEFAULT) <= r["htf"]
                  < p.get("htf_max", S.HTF_MAX_DEFAULT))
             and (r.get("rpos") is None or r["rpos"] <= p.get("range_pos_max", S.MAX_RANGE_POS))
             and (r.get("rsi") is None or r["rsi"] >= p.get("rsi_min", S.RSI_MIN_DEFAULT))
             and (r.get("btc1h") is None or r["btc1h"] >= p.get("btc1h_min", S.BTC1H_MIN_DEFAULT))]
        hi = [r for r in g if r["prob"] >= HIGH_MIN_PROB]
        lo = [r for r in g if self.min_prob() <= r["prob"] < HIGH_MIN_PROB]
        hours = len({int(r["ts"] // 3600) for r in hi})
        rate_hi = sum(r["hit"] for r in hi) / len(hi) if hi else None
        rate_lo = sum(r["hit"] for r in lo) / len(lo) if lo else None
        ok = (len(hi) >= M15_MIN_N and hours >= GATE_MIN_HOURS and len(lo) >= M15_MIN_N
              and rate_hi >= HIGH_SIZE_PASS and rate_hi >= rate_lo + HIGH_SIZE_MARGIN)
        # Overconfidence cap: once the high-probability bucket has enough independent hours and still reaches TP1
        # first less often than the gate pass rate, those reviews are not traded at all.
        cap = (HIGH_MIN_PROB if len(hi) >= M15_MIN_N and hours >= GATE_MIN_HOURS and rate_hi < M15_PASS else None)
        if cap != p.get("self_max_prob"):
            self.log("SELF_MAXPROB_CALIBRATE", self_max_prob_old=p.get("self_max_prob"), self_max_prob_new=cap,
                     hours=hours, n=len(hi), tp1_first_rate=rate_hi and round(rate_hi, 3),
                     note=f"分析驗算機率 ≥{HIGH_MIN_PROB}% 的評估實測先到 TP1 比例 < {M15_PASS:.0%}，視為模型過度自信，不進場"
                     if cap else "高機率評估實測已不低於門檻 (或樣本不足)，取消過度自信上限")
        p["self_max_prob"] = cap
        old = p.get("high_size_ok")
        p["high_size_ok"] = ok
        self.learner.save()
        if ok != old:
            self.log("SELF_HIGH_CALIBRATE", high_size_ok=ok, hours=hours,
                     high={"prob": f">={HIGH_MIN_PROB}", "n": len(hi), "tp1_first_rate": rate_hi and round(rate_hi, 3)},
                     rest={"prob": f"{self.min_prob()}-{HIGH_MIN_PROB}", "n": len(lo), "tp1_first_rate": rate_lo and round(rate_lo, 3)},
                     note=f"高機率評估須比一般評估先到 TP1 比例高 {HIGH_SIZE_MARGIN:.0%} 且 ≥{HIGH_SIZE_PASS:.0%} 才放大部位；否則高信心單以一般風險開倉")

    def _gate_ready(self, rows: list, key: str, param: str, default: float) -> bool:
        hours = len({int(r["ts"] // 3600) for r in rows if r.get(key) is not None})
        if hours >= GATE_MIN_HOURS:
            return True
        if self.learner.p.get(param) != default:
            self.learner.p[param] = default
            self.learner.save()
            self.log("SELF_GATE_DEFAULT", param=param, value=default, hours=hours,
                     note=f"樣本只涵蓋 {hours} 個小時 (< {GATE_MIN_HOURS})，同一時段的評估高度相關，先用預設值")
        return False

    def calibrate_htf(self, rows: list):
        if not self._gate_ready(rows, "htf", "htf_min", S.HTF_MIN_DEFAULT):
            return
        table = []
        for lo, hi in HTF_BUCKETS:
            b = [r["hit"] for r in rows if r.get("htf") is not None and lo <= r["htf"] < hi]
            table.append((lo, hi, len(b), round(sum(b) / len(b), 3) if b else None))
        if any(n < M15_MIN_N for _, _, n, _ in table[:2]):
            return
        floor = HTF_BUCKETS[-1][0]
        for lo, _, n, rate in reversed(table):
            if n < M15_MIN_N or rate < M15_PASS:
                break
            floor = lo
        # Don't raise the floor above the approved default: that plus a tight cap blocked every short
        # through the overnight drop. A floor at or above the cap is an empty band.
        if floor > S.HTF_MIN_DEFAULT or floor >= self.learner.p.get("htf_max", S.HTF_MAX_DEFAULT):
            floor = S.HTF_MIN_DEFAULT
        old = self.learner.p.get("htf_min", S.HTF_MIN_DEFAULT)
        self.learner.p["htf_min"] = floor
        self.learner.save()
        if floor != old or not self.learner.p.get("htf_logged"):
            self.learner.p["htf_logged"] = True
            self.log("SELF_HTF_CALIBRATE", htf_min_old=old, htf_min_new=floor,
                     buckets=[{"htf_in_side": f"{lo}~{hi}", "n": n, "tp1_first_rate": r} for lo, hi, n, r in table],
                     note="1h/4h 大方向與交易方向相反時先到 TP1 比例偏低；自動設定可進場的大週期方向下限")

    def calibrate_htf_max(self, rows: list):
        if not self._gate_ready(rows, "htf", "htf_max", S.HTF_MAX_DEFAULT):
            return
        table = []
        for lo, hi in HTF_CAP_BUCKETS:
            b = [r["hit"] for r in rows if r.get("htf") is not None and lo <= r["htf"] < hi]
            table.append((lo, hi, len(b), round(sum(b) / len(b), 3) if b else None))
        if any(n < M15_MIN_N for _, _, n, _ in table[:2]):
            return
        # Keep the default unless a bucket actually passes. The old start value was the first boundary,
        # so a failing first bucket still installed that boundary and could meet htf_min, blocking every entry.
        cap = S.HTF_MAX_DEFAULT
        for _, hi, n, rate in table:
            if n < M15_MIN_N or rate < M15_PASS:
                break
            # Do not auto-raise past the replay cap. The 1.5–2.5 bucket still loses in R.
            if hi > S.HTF_MAX_DEFAULT:
                break
            if hi >= S.HTF_MAX_DEFAULT:
                cap = hi
        floor = self.learner.p.get("htf_min", S.HTF_MIN_DEFAULT)
        if cap <= floor or cap < S.HTF_MAX_DEFAULT:
            cap = S.HTF_MAX_DEFAULT
        old = self.learner.p.get("htf_max", S.HTF_MAX_DEFAULT)
        self.learner.p["htf_max"] = cap
        self.learner.save()
        if cap != old or not self.learner.p.get("htf_max_logged"):
            self.learner.p["htf_max_logged"] = True
            self.log("SELF_HTF_MAX_CALIBRATE", htf_max_old=old, htf_max_new=cap,
                     buckets=[{"htf_in_side": f"{lo}~{hi}", "n": n, "tp1_first_rate": r} for lo, hi, n, r in table],
                     note="1h/4h 大方向已強烈順著交易方向時先到 TP1 比例偏低 (進場偏晚)；自動設定可進場的大週期方向上限")

    def calibrate_m15(self, rows: list):
        if not self._gate_ready(rows, "m15", "m15_max", S.M15_MAX_DEFAULT):
            return
        table = []
        for lo, hi in M15_BUCKETS:
            b = [r["hit"] for r in rows if r.get("m15") is not None and lo <= r["m15"] < hi]
            table.append((lo, hi, len(b), round(sum(b) / len(b), 3) if b else None))
        if any(n < M15_MIN_N for _, _, n, _ in table[:2]):
            return
        cap = S.M15_MAX_DEFAULT
        for _, hi, n, rate in table:
            if n < M15_MIN_N or rate < M15_PASS:
                break
            # A cap below the default, including a negative one, blocks ordinary and waterfall shorts.
            if hi >= S.M15_MAX_DEFAULT:
                cap = hi
        old = self.learner.p.get("m15_max", S.M15_MAX_DEFAULT)
        self.learner.p["m15_max"] = cap
        self.learner.save()
        if cap != old or not self.learner.p.get("m15_logged"):
            self.learner.p["m15_logged"] = True
            self.log("SELF_M15_CALIBRATE", m15_max_old=old, m15_max_new=cap,
                     buckets=[{"m15_in_side": f"{lo}~{hi}", "n": n, "tp1_first_rate": r} for lo, hi, n, r in table],
                     note="順方向 15m 動能越強進場，先到 TP1 比例越低；自動設定可進場的 15m 動能上限")

    def calibrate_rpos(self, rows: list):
        # Longs belong near the 3h low and shorts near the 3h high. Don't let the hit-rate
        # buckets above 0.85 reopen the old 1.05 chase cap.
        if self.learner.p.get("range_pos_max", S.MAX_RANGE_POS) != S.MAX_RANGE_POS:
            self.learner.p["range_pos_max"] = S.MAX_RANGE_POS
            self.learner.save()

    def calibrate_rsi(self, rows: list):
        # The measured best set leaves this off. Don't let a short sample turn it back on.
        if self.learner.p.get("rsi_min", S.RSI_MIN_DEFAULT) != S.RSI_MIN_DEFAULT:
            self.learner.p["rsi_min"] = S.RSI_MIN_DEFAULT
            self.learner.save()

    def calibrate_btc1h(self, rows: list):
        # Longs use +0.055 and shorts use -0.0425. A single learned floor would collapse that.
        return

    def calibrate_atr_ratio(self, rows: list):
        # Quiet-hour filter. Three robust passes fixed the cut at 0.8; do not let a short sample move it.
        if self.learner.p.get("atr_ratio_min", S.ATR_RATIO_MIN_DEFAULT) != S.ATR_RATIO_MIN_DEFAULT:
            self.learner.p["atr_ratio_min"] = S.ATR_RATIO_MIN_DEFAULT
            self.learner.save()

    def calibrate_rs1h(self, rows: list):
        # Don't chase a coin that already outran BTC. Only the ≥ 0.924 cut held for three runs.
        if self.learner.p.get("rs1h_max", S.RS1H_MAX_DEFAULT) != S.RS1H_MAX_DEFAULT:
            self.learner.p["rs1h_max"] = S.RS1H_MAX_DEFAULT
            self.learner.save()

    def calibrate_vol_ratio(self, rows: list):
        # Thin last-hour volume. Three robust passes fixed the cut at 0.54; do not let a short sample move it.
        if self.learner.p.get("vol_ratio_min", S.VOL_RATIO_MIN_DEFAULT) != S.VOL_RATIO_MIN_DEFAULT:
            self.learner.p["vol_ratio_min"] = S.VOL_RATIO_MIN_DEFAULT
            self.learner.save()

    def calibrate_px15(self, rows: list):
        # Last 15m already against the trade. Three robust passes fixed the cut; do not move it.
        if self.learner.p.get("px15_dir_min", S.PX15_DIR_MIN_DEFAULT) != S.PX15_DIR_MIN_DEFAULT:
            self.learner.p["px15_dir_min"] = S.PX15_DIR_MIN_DEFAULT
            self.learner.save()

    def calibrate_quote_vol(self, rows: list):
        # Illiquid names. The top-quartile cut held for three runs; lock the loosest of them.
        if self.learner.p.get("quote_vol_min", S.QUOTE_VOL_MIN_DEFAULT) != S.QUOTE_VOL_MIN_DEFAULT:
            self.learner.p["quote_vol_min"] = S.QUOTE_VOL_MIN_DEFAULT
            self.learner.save()

    # ------------------------------------------------------------------ scanning
    def universe(self, regime_label: str, rhits: dict) -> list:
        boards = self.so.boards()
        cands: dict[str, dict] = {}
        for t in boards.get("turtleActive", []):
            cands.setdefault(t["symbol"], {"side": t["side"], "tags": []})["tags"].append(f"強勢榜#{t.get('rank')}")
        for x in boards.get("scanner", []):
            hint = (x.get("raw_features") or {}).get("directionHint")
            c = cands.setdefault(x["symbol"], {"side": hint, "tags": []})
            c["side"] = c["side"] or hint
            c["tags"].append(f"潛力榜#{x['rank']}({x.get('state')})")
        for base, hits in rhits.items():
            sym = base + "USDT"
            if sym not in cands:
                continue
            for name, side, passed, total in hits:
                if passed * 2 < total or side != cands[sym].get("side"):
                    continue
                cands[sym]["tags"].append(f"客製化:{name} {passed}/{total}")
        # Majors are not forced onto the list. On Nov 8 the only self-analysis trades
        # that came from this injection were SOL and XRP, and the recheck closed both
        # at a loss before any target. The potential-board shorts, WLD and XMR, both
        # reached TP1. A major still enters when a board or a breakout names it.
        # Every USDT-margined contract, including TradFi (stocks, indices, metals, forex).
        # Boards keep the side hint they already have; names See One does not list are skipped later.
        for tsym, spec in self.tb.specs().items():
            if spec.get("quote") not in ("", "USDT"):
                continue
            sym = spec.get("seeone")
            if not sym:
                continue
            c = cands.setdefault(sym, {"side": None, "tags": []})
            if spec.get("rwa") and "TradFi" not in c["tags"]:
                c["tags"].append("TradFi")
        out = []
        for sym, c in cands.items():
            out.append({"symbol": sym, "side": c.get("side"), "tags": c["tags"]})
        return out

    def plan(self, tsym: str, side: str) -> dict | None:
        a = atr(tsym)
        if not a:
            return None
        atr_v, rows = a
        price = self.tb.price(tsym)
        dist = max(ATR_MULT * atr_v, MIN_STOP_PCT * price)
        if dist / price > MAX_STOP_PCT:
            return None
        sgn = 1 if side == "LONG" else -1
        move = (price - float(rows[-EXT_BARS][1])) / dist
        prev = float(rows[-2][4]) if len(rows) >= 2 else 0.0
        px15 = (price - prev) / prev * 100 if prev else None
        return {"price": price, "atr": atr_v, "stop": price - sgn * dist,
                "tps": [price + sgn * dist * r for r in TP_R], "dist": dist, "move_r": round(move, 2),
                "rpos": range_pos(rows, price), "rsi": rsi_long(rows), "px15": None if px15 is None else round(px15, 3)}

    def btc_counter(self) -> dict:
        """How far BTC has moved off its recent low and high, in BTC 15m ATRs."""
        a = atr("BTC-SWAP-USDT")
        if not a:
            return {"up": 0.0, "down": 0.0}
        atr_v, rows = a
        recent = rows[-EXT_BARS:]
        last = float(recent[-1][4])
        lo, hi = min(float(r[3]) for r in recent), max(float(r[2]) for r in recent)
        ref = float(rows[-5][4])
        return {"up": round((last - lo) / atr_v, 2), "down": round((hi - last) / atr_v, 2),
                "ret1h": round((last - ref) / ref * 100, 3)}

    def expected_r(self, rv: dict, split: list, sl_pct: float, bias: float = 0.0) -> float:
        probs = [max(0.0, t["estimatedHitProbability"] - bias) / 100 for t in rv["geometry"]["targets"]]
        p1 = probs[0]
        ev = sum(w * p * r for w, p, r in zip(split, probs, TP_R)) - (1 - p1)
        fee_r = (S.MAKER_FEE * (1 + p1) + S.TAKER_FEE * (1 - p1)) / sl_pct
        return round(ev - fee_r, 3)

    def _pairs(self) -> list:
        if getattr(self, "_pair_cache", None) is None:
            rows = []
            if self.sim_log.exists():
                for line in self.sim_log.read_text(encoding="utf-8").splitlines():
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    if rec.get("prob") is not None and rec.get("seeone_prob") is not None:
                        rows.append(rec)
            self._pair_cache = rows
        return self._pair_cache

    def _write_align(self):
        rep = self.local.report(self._pairs())
        lines = [
            "自建驗算對齊 See One（實盤不下單）",
            f"歷史留出樣本 {rep['n']} 筆（訓練 {rep['train']} / 測試 {rep['test']}）",
            f"平均絕對誤差 {rep['mae']} 個百分點，偏差 {rep['bias']}（正=我們高於 See One）",
            f"60 分以上是否同一側 {rep['agree']}",
            f"我們估 50–70 時，實際先碰到 TP1 的比例 {rep['hit_when_local_50_70']}（{rep['hit_n']} 筆）",
            f"歷史對齊 {'是' if rep['aligned'] else '否'}（需要 ≥{200} 筆、誤差 ≤8、同一側 ≥0.75）",
            f"上線後同時有兩邊機率的新對照 {rep['live_pairs']} 筆，誤差 {rep['live_mae']}，同一側 {rep['live_agree']}",
            f"可以考慮接上實盤 {'是' if rep['ready'] else '否'}",
        ]
        path = self.sim_log.with_name("local_review_align.txt")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.log("LOCAL_ALIGN", n=rep["n"], mae=rep["mae"], bias=rep["bias"], agree=rep["agree"],
                 aligned=rep["aligned"], live_pairs=rep["live_pairs"], live_mae=rep["live_mae"],
                 live_agree=rep["live_agree"], ready=rep["ready"])

    def _emit_shadow(self, sym: str, pl_long: dict, htf: float | None, seeone_prob: float | None, seeone_side: str | None):
        """Log what our own TP1-before-stop table would say. Never opens a trade."""
        btc = (self.btc or {}).get("ret1h")
        rpos_long = pl_long.get("rpos")
        est = {}
        for side in ("LONG", "SHORT"):
            rpos = None if rpos_long is None else (rpos_long if side == "LONG" else round(1 - rpos_long, 3))
            htf_dir = None if htf is None else htf * (1 if side == "LONG" else -1)
            btc_dir = None if btc is None else btc * (1 if side == "LONG" else -1)
            est[side] = {**self.local.estimate(side, rpos, btc_dir, htf_dir), "rpos": rpos, "htf_dir": htf_dir}
        side = max(est, key=lambda s: est[s]["prob"] if est[s]["prob"] is not None else -1)
        other = "SHORT" if side == "LONG" else "LONG"
        prob, other_prob = est[side]["prob"], est[other]["prob"]
        edge = None if prob is None or other_prob is None else round(prob - other_prob, 1)
        htf_dir, rpos = est[side]["htf_dir"], est[side]["rpos"]
        would = (prob is not None and prob >= DEFAULT_MIN_PROB and edge is not None and edge >= MIN_EDGE
                 and rpos is not None and rpos <= S.MAX_RANGE_POS
                 and S.direc_window(htf_dir, side) is not None)
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "symbol": sym, "side": side,
               "prob": prob, "other_prob": other_prob, "edge": edge, "n": est[side]["n"],
               "level": est[side]["level"], "rpos": rpos, "htf_dir": None if htf_dir is None else round(htf_dir, 2),
               "would_pass": would, "seeone_prob": seeone_prob, "seeone_side": seeone_side}
        with self.sim_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if seeone_prob is not None and prob is not None:
            self._pairs().append(rec)
        self.log("LOCAL_SIM", **rec, note="自建驗算模擬，不下單；有 See One 機率的才計入對齊")

    def long_gate(self, htf: float | None, btc1h: float | None) -> tuple[bool, str]:
        if htf is None or htf < LONG_TREND_HTF:
            return False, f"大盤 1h/4h 方向 {htf if htf is not None else '未知'} 未達多頭 +{LONG_TREND_HTF}"
        if btc1h is not None and btc1h < S.SELF_BTC1H_MIN:
            return False, f"BTC 近 1 小時 {btc1h:+.2f}%，跌幅超過 {S.SELF_BTC1H_MIN:+.1f}%"
        cutoff = time.time() - LONG_GATE_HOURS * 3600
        rows = [r for r in self.d["resolved"] if r["side"] == "LONG" and r["ts"] >= cutoff
                and r.get("htf") is not None and r["htf"] >= LONG_TREND_HTF and r.get("prob", 0) >= 50]
        hit = sum(r["hit"] for r in rows) / len(rows) if rows else None
        note = (f"大盤 1h/4h {htf:+.2f}，近 {LONG_GATE_HOURS} 小時同條件多單驗算先到 TP1 "
                f"{'-' if hit is None else f'{hit:.0%}'}（{len(rows)} 筆）")
        if LONG_GATE_REQUIRE_RECORD and (len(rows) < LONG_GATE_MIN_N or hit < LONG_GATE_HIT):
            return False, f"{note}，未達 {LONG_GATE_MIN_N} 筆且 {LONG_GATE_HIT:.0%}"
        return True, note

    def sides(self, htf: float | None, btc1h: float | None) -> tuple:
        if "LONG" in S.SELF_SIDES or not self.long_gate(htf, btc1h)[0]:
            return S.SELF_SIDES
        return S.SELF_SIDES + ("LONG",)

    def _near_extreme(self, ordered: list, htf: float | None, specs: dict, held: set,
                      open_sides: tuple = S.SELF_SIDES) -> list:
        """Names whose 3h price is still near the reversal extreme on an open side.

        The review budget is REVIEWS_PER_SCAN See One calls. Names already through the range
        cannot be opened, so they no longer take a call. Setups self-scan can open (side in
        SELF_SIDES, stop under SELF_MAX_SL_PCT) go first; the rest of the budget keeps the
        calibration records coming when no such setup exists.
        """
        if htf is None:
            return []
        sides = []
        for side in ("LONG", "SHORT"):
            direc = htf if side == "LONG" else -htf
            band = S.direc_window(direc, side)
            if band:
                sides.append((side, band))
        if not sides:
            return []
        btc = (self.btc or {}).get("ret1h")
        ranked = []
        looked = 0
        for c in ordered:
            if looked >= 160 or sum(1 for t, _, _ in ranked if t) >= REVIEWS_PER_SCAN:
                break
            tsym = S.to_toobit(c["symbol"])
            if tsym not in specs or tsym in held:
                continue
            looked += 1
            try:
                pl = self.plan(tsym, "LONG")
            except Exception:
                continue
            if not pl or pl.get("rpos") is None:
                continue
            narrow = pl["dist"] / pl["price"] < S.SELF_MAX_SL_PCT
            best, tradable = None, False
            for side, _band in sides:
                sgn = 1 if side == "LONG" else -1
                rp = pl["rpos"] if side == "LONG" else 1 - pl["rpos"]
                if rp > S.MAX_RANGE_POS:
                    continue
                htf_dir = None if htf is None else htf * sgn
                btc_dir = None if btc is None else btc * sgn
                est = self.local.estimate(side, rp, btc_dir, htf_dir)
                prob = est.get("prob") if est else None
                if best is None or (prob or -1) > best:
                    best = prob or -1
                tradable = tradable or (narrow and side in open_sides)
            if best is None:
                continue
            ranked.append((tradable, best, c))
        ranked.sort(key=lambda x: (not x[0], -x[1]))
        return [c for _, _, c in ranked[:REVIEWS_PER_SCAN]]

    def orderbook_spikes(self, held: set, specs: dict) -> list:
        """One maker long when the explosion recipe fully matches.

        The short-crash recipe is off. XPL, ERA and both IRYS shorts all
        finished near -0.9R within about 16 minutes. The Nov 8 cards
        (PONS, LYN, CT, ONE) are not traded.
        A name that has already run 2.5R is not chased.
        """
        if time.time() - self.last_spike < SCAN_EVERY_S:
            return []
        self.last_spike = time.time()
        sig = self._spike_side("LONG", "爆漲", held, specs)
        return [sig] if sig else []

    def _spike_side(self, side: str, token: str, held: set, specs: dict) -> dict | None:
        best = None
        matched = 0
        for r in self.so.recipes():
            rec = r.get("recipe") or {}
            rules = rec.get("rules") or []
            name = rec.get("name") or ""
            metrics = {x.get("metric") for x in rules}
            if not rec.get("enabled", True) or token not in name:
                continue
            if side == "LONG" and not {"depth-delta", "order-wall"} <= metrics:
                continue
            if S.recipe_direction(name, rules) != side:
                continue
            rows = list((r.get("evaluation") or {}).get("results") or [])
            seen = set()
            for row in rows:
                base = str(row.get("asset") or "").upper()
                if base and row.get("match"):
                    seen.add(base)
                    pot = ((row.get("readings") or {}).get("potential") or {}).get("score") or 0
                    best, matched = self._spike_pick(best, matched, base, name, pot, held, specs, side)
            for base, hit in (r.get("members") or {}).items():
                base = str(base).upper()
                if hit and base not in seen:
                    best, matched = self._spike_pick(best, matched, base, name, 0, held, specs, side)
        self.log("ORDERBOOK_SPIKE", side=side, matched=matched, found=None if best is None else best[0])
        if best is None:
            return None
        sym, name, pl, _pot = best
        price = pl["price"]
        levels = [{"label": f"TP{i + 1}", "kind": "TP", "sequence": i + 1, "r_multiple": TP_R[i],
                   "price": pl["tps"][i], "hit": False} for i in range(len(TP_R))]
        day = time.strftime("%Y-%m-%d", time.gmtime())
        verb = "漲" if side == "LONG" else "跌"
        label = "訂單簿爆漲" if side == "LONG" else "空頭爆擊"
        sig = {"id": f"rcp:{sym}:{side}:{day}", "symbol": sym, "side": side, "status": "ACTIVE",
               "timeframe": "15m", "entry_price": price, "stop_price": pl["stop"], "current_price": price,
               "entry_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "levels": levels,
               "signal_path": "ORDERBOOK", "path_subtype": name, "recipe_spike": True}
        reasons = [
            f"{label}配方「{name}」全中",
            "只掛單，價格離開進場區間不追",
            f"近 2 小時已{verb}超過 2.5R 的不追",
        ]
        ev = {"ok": True, "score": 96.0, "tier": "NORMAL", "reasons": reasons, "factors": {f"source:{label}": 0.0},
              "skip": None, "cur_r": 0.0, "analysis": {"prob": None, "edge": None, "quality": None, "ev_r": None}}
        self.log("ORDERBOOK_SPIKE", symbol=sym, side=side, recipe=name, price=price, stop=pl["stop"],
                 move_r=pl["move_r"], note="配方全中，掛不到就算")
        return {"sig": sig, "ev": ev, "source": label}

    def _spike_pick(self, best, matched, base, name, pot, held, specs, side):
        sym = base + "USDT"
        tsym = S.to_toobit(sym)
        if tsym in held or tsym not in specs:
            return best, matched
        pl = self.plan(tsym, side)
        signed = pl["move_r"] if pl and side == "LONG" else (None if not pl else -pl["move_r"])
        if not pl or signed >= S.MAX_EXT_R:
            return best, matched
        matched += 1
        if best is None or pot > best[3]:
            best = (sym, name, pl, pot)
        return best, matched

    def fresh_breaks(self, held: set, specs: dict, htf: float | None = None) -> list:
        """One backup signal: the largest fresh 15m close through the prior 3h range.

        The breaking bar has to be the first close outside that range, and its
        high-low at least 1%. A later bar in the same move is already gone.
        """
        if time.time() - self.last_break < SCAN_EVERY_S:
            return []
        self.last_break = time.time()
        now_ms = time.time() * 1000
        names = [(tsym, (spec or {}).get("seeone")) for tsym, spec in (specs or {}).items()
                 if tsym not in held and (spec or {}).get("seeone")]

        def one(item):
            tsym, seeone = item
            rows = klines(tsym, "15m", 32)
            if not isinstance(rows, list) or len(rows) < 16:
                return None
            closed = rows[:-1] if int(rows[-1][0]) >= now_ms - 15 * 60 * 1000 else rows
            if len(closed) < 15:
                return None
            bar, prev = closed[-1], closed[-2]
            prior = closed[-14:-2]
            o, h, l, c = (float(bar[1]), float(bar[2]), float(bar[3]), float(bar[4]))
            if o <= 0 or (h - l) / o < S.BREAK_MIN_RANGE_PCT:
                return None
            hi, lo = max(float(x[2]) for x in prior), min(float(x[3]) for x in prior)
            prev_c = float(prev[4])
            side = broken = None
            if c < lo and prev_c >= lo:
                side, broken = "SHORT", lo
            elif c > hi and prev_c <= hi:
                side, broken = "LONG", hi
            if side is None:
                return None
            if side == "LONG":
                if len(closed) < 17:
                    return None
                if not S.long_after_red((h - l) / o, float(closed[-17][4]), float(closed[-2][4])):
                    return None
            atr_v = sum(max(float(x[2]) - float(x[3]), abs(float(x[2]) - float(p[4])), abs(float(x[3]) - float(p[4])))
                        for p, x in zip(closed[-15:-1], closed[-14:])) / 14
            sgn = 1 if side == "LONG" else -1
            dist = max(abs(c - broken), ATR_MULT * atr_v, MIN_STOP_PCT * c)
            if dist / c > MAX_STOP_PCT:
                return None
            start = float(closed[-9][1])
            if start <= 0 or (c - start) * sgn / dist >= S.MAX_EXT_R:
                return None
            return ((h - l) / o, seeone, side, c, dist, bar, broken)

        best = None
        htf_skipped = long_skipped = 0
        with ThreadPoolExecutor(16) as pool:
            for hit in pool.map(one, names):
                if not hit:
                    continue
                side = hit[2]
                sgn = 1 if side == "LONG" else -1
                if side != "LONG" and htf is not None and htf * sgn < S.HTF_MIN_DEFAULT:
                    htf_skipped += 1
                    continue
                if best is None or hit[0] > best[0]:
                    best = hit
        self.log("BREAKOUT_SCAN", checked=len(names), htf_skipped=htf_skipped,
                 long_skipped=long_skipped, found=None if best is None else best[1])
        if best is None:
            self.log("BREAKOUT", found=None, note="這一輪沒有剛收在 3 小時區間外的 15 分鐘 K 線")
            return []
        span, sym, side, price, dist, bar, broken = best
        sgn = 1 if side == "LONG" else -1
        stop = price - sgn * dist
        tps = [price + sgn * dist * r for r in TP_R]
        levels = [{"label": f"TP{i + 1}", "kind": "TP", "sequence": i + 1, "r_multiple": TP_R[i],
                   "price": tps[i], "hit": False} for i in range(len(TP_R))]
        sig = {"id": f"brk:{sym}:{side}:{int(bar[0])}", "symbol": sym, "side": side, "status": "ACTIVE",
               "timeframe": "15m", "entry_price": price, "stop_price": stop, "current_price": price,
               "entry_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "levels": levels,
               "signal_path": "BREAKOUT", "path_subtype": None, "breakout": True, "broken": broken,
               "after_red": side == "LONG"}
        reasons = [
            f"突破備案：15 分鐘收盤{'跌破' if side == 'SHORT' else '漲破'}前 3 小時{'低點' if side == 'SHORT' else '高點'} {broken:.6g}",
            f"這根高低振幅 {span:.1%}，至少 {S.BREAK_MIN_RANGE_PCT:.0%}"
            + ("；先連跌再一根大綠 K" if side == "LONG" else ""),
            "只做剛收出去的那一根；下一根還在外面就當沒了",
            "只掛單，價格離開進場區間不追",
        ]
        ev = {"ok": True, "score": 95.0, "tier": "NORMAL", "reasons": reasons, "factors": {"source:突破": 0.0},
              "skip": None, "cur_r": 0.0, "analysis": {"prob": None, "edge": None, "quality": None, "ev_r": None}}
        tsym = S.to_toobit(sym)
        where = listing(tsym)
        self.log("BREAKOUT", symbol=sym, side=side, range_pct=round(span * 100, 2),
                 broken=broken, price=price, stop=stop, listing=where,
                 note="備案，掛不到就算" + ("。這個幣只在本所" if where == "only" else ""))
        return [{"sig": sig, "ev": ev, "source": "突破"}]

    def scan(self, regime: tuple, rhits: dict, held: set, specs: dict, m15: float | None = None,
             htf: float | None = None) -> list:
        """Review every candidate both LONG and SHORT on mirrored geometry. 分析驗算 is direction-biased
        (mirrored 1R probabilities sum to well over 100%), so each side is de-biased by half the excess and
        a trade needs a clear edge over the opposite side, not just a high raw probability."""
        if time.time() - self.last_scan < SCAN_EVERY_S:
            return []
        seeone_open = self.so.review_not_before <= time.time()
        self.last_scan = time.time()
        self.shared_hits = 0
        self.resolve()
        split = S.TP_SPLIT
        min_prob = self.min_prob()
        try:
            self.btc = self.btc_counter()
        except Exception:
            self.btc = {"up": 0.0, "down": 0.0}
        open_sides = self.sides(htf, self.btc.get("ret1h"))
        long_note = self.long_gate(htf, self.btc.get("ret1h"))[1]
        found, watched, reviewed, by_symbol = [], [], 0, {}
        for c in self.universe(regime[1], rhits):
            by_symbol.setdefault(c["symbol"], c)
        now = time.time()
        unsupported = self.d.setdefault("unsupported", {})
        for sym in [s for s, until in unsupported.items() if until <= now]:
            unsupported.pop(sym, None)

        def due(sym: str) -> bool:
            if unsupported.get(sym, 0) > now:
                return False
            return now - self.d["last_review"].get(sym, 0) >= REVIEW_COOLDOWN_S

        def priority(c: dict) -> bool:
            return any(t.startswith(("強勢榜", "潛力榜", "客製化", "主流幣")) for t in c["tags"])

        boards = sorted((c for c in by_symbol.values() if priority(c) and due(c["symbol"])),
                        key=lambda c: self.d["last_review"].get(c["symbol"], 0))
        rest = sorted((c for c in by_symbol.values() if not priority(c) and due(c["symbol"])),
                      key=lambda c: self.d["last_review"].get(c["symbol"], 0))
        # The analysis-review budget is a few calls per cooldown. Spend three of every four
        # on names already on a See One board; the fourth keeps TradFi and the rest of the book moving.
        ordered = []
        while boards or rest:
            for _ in range(3):
                if boards:
                    ordered.append(boards.pop(0))
            if rest:
                ordered.append(rest.pop(0))
        # A full scan used to finish about twenty names. After the 11:52 burst, capping
        # each round keeps us under the pace See One was answering before the penalty.
        shadowed = 0
        qv_map: dict[str, float] = {}
        btc_5m = None
        if not seeone_open:
            fresh = self.d.setdefault("last_shadow", {})
            ordered = [c for c in ordered if now - fresh.get(c["symbol"], 0) >= REVIEW_COOLDOWN_S]
            # Dust names pass the higher-timeframe and range checks and used to fill both
            # slots before a liquid contract was looked at. Quote volume is known up front.
            qv_map = contract_quote_vol()
            qv_min = self.learner.p.get("quote_vol_min", S.QUOTE_VOL_MIN_DEFAULT)
            ordered.sort(key=lambda c: 0 if qv_map.get(S.to_toobit(c["symbol"]), 0) >= qv_min else 1)
            btc_5m = micro_1h("BTC-SWAP-USDT")[1]
        if seeone_open:
            # Local plans are free. Spend the See One calls on names already sitting
            # near the 3h extreme on a side the window allows, including the liquid-short band.
            ordered = self._near_extreme(ordered, htf, specs, held, open_sides)
        limit = 40 if not seeone_open else REVIEWS_PER_SCAN
        pl_long = None
        for c in ordered[:limit]:
            sym = c["symbol"]
            tsym = S.to_toobit(sym)
            if tsym not in specs or tsym in held:
                continue
            if not seeone_open:
                if now - self.d.setdefault("last_shadow", {}).get(sym, 0) < REVIEW_COOLDOWN_S:
                    continue
                pl_long = self.plan(tsym, "LONG")
                self.d["last_shadow"][sym] = time.time()
                if not pl_long:
                    continue
                self._emit_shadow(sym, pl_long, htf, None, None)
                shadowed += 1
                if not S.QUIET_ENTRIES:
                    continue
                short_band = htf is not None and 1.5 <= -htf < 2.5
                long_band = htf is not None and 1.5 <= htf < 2.5
                if short_band and sum(1 for x in found if x.get("source") == "安靜期強勢空") < S.QUIET_MAX_SAME_SIDE:
                    item = self._quiet_short(c, pl_long, regime, htf)
                    if item:
                        why = self._quiet_blocked(tsym, btc_5m, qv_map.get(tsym), "SHORT")
                        if why:
                            self.log("QUIET_GATE", symbol=sym, side="SHORT", reason=why)
                        else:
                            found.append(item)
                if long_band and sum(1 for x in found if x.get("source") == "安靜期強勢多") < S.QUIET_MAX_SAME_SIDE:
                    item = self._quiet_long(c, pl_long, regime, htf)
                    if item:
                        why = self._quiet_blocked(tsym, btc_5m, qv_map.get(tsym), "LONG")
                        if why:
                            self.log("QUIET_GATE", symbol=sym, side="LONG", reason=why)
                        else:
                            found.append(item)
                short_done = short_band and sum(1 for x in found if x.get("source") == "安靜期強勢空") >= S.QUIET_MAX_SAME_SIDE
                long_done = long_band and sum(1 for x in found if x.get("source") == "安靜期強勢多") >= S.QUIET_MAX_SAME_SIDE
                if short_done or long_done:
                    break
                continue
            try:
                pl_long = None
                pl_long = self.plan(tsym, "LONG")
                if not pl_long:
                    continue
                d, px = pl_long["dist"], pl_long["price"]
                pl_short = {**pl_long, "stop": px + d, "tps": [px - d * r for r in TP_R]}
                plans = {"LONG": pl_long, "SHORT": pl_short}
                # Review a side in the core window, or a short in the liquid 2.0–2.5 band.
                # Near zero, both sides of the core window are eligible.
                def m15_ok(sd: str) -> bool:
                    if m15 is None:
                        return True
                    sgn = 1 if sd == "LONG" else -1
                    return m15 * sgn < S.M15_MAX_DEFAULT

                eligible = [sd for sd, direc in (("LONG", htf), ("SHORT", None if htf is None else -htf))
                            if S.direc_window(direc, sd) is not None and m15_ok(sd)]
                tradable = [sd for sd in eligible if sd in open_sides] if d / px < S.SELF_MAX_SL_PCT else []
                pick = tradable or eligible
                hint = c.get("side") if c.get("side") in pick else pick[0]
                # Both sides are only needed to measure the model's long/short bias. Once that bias is known,
                # one call plus the saved bias matches the old de-bias math and cuts the request in half.
                both = (reviewed + 1) % BIAS_EVERY == 0 and len(eligible) == 2
                rv = {hint: self.review(sym, hint, plans[hint]["stop"], plans[hint]["tps"], share=True)}
                if both:
                    other_side = "SHORT" if hint == "LONG" else "LONG"
                    rv[other_side] = self.review(sym, other_side, plans[other_side]["stop"], plans[other_side]["tps"],
                                                 share=True)
            except Exception as e:
                self.log("ERROR", where="self_review", symbol=sym, error=repr(e)[:300])
                # SeeOneLimited is raised only after a real 429 (or when the shared cooldown is
                # still open). The clock lives on the client, so this scan just stops.
                if isinstance(e, S.SeeOneLimited):
                    self.d["last_review"][sym] = time.time()
                    if pl_long:
                        self._emit_shadow(sym, pl_long, htf, None, None)
                    break
                if isinstance(e, S.SeeOneDown):
                    break
                if any(code in repr(e) for code in ("400 ", "404 ", "422 ")):
                    self.d.setdefault("unsupported", {})[sym] = time.time() + 6 * 3600
                continue
            reviewed += 1
            self.d["last_review"][sym] = time.time()
            time.sleep(1.5)
            if any(len((r.get("geometry") or {}).get("targets") or []) != len(TP_R) for r in rv.values()):
                continue
            raw = {sd: float(r.get("probability") or 0) for sd, r in rv.items()}
            samples = self.d.setdefault("bias_samples", [])
            if "LONG" in raw and "SHORT" in raw:
                bias = (raw["LONG"] + raw["SHORT"] - 100) / 2
                samples.append(round(bias, 2))
                del samples[:-40]
                side = max((sd for sd in raw if sd in open_sides), key=raw.get, default=max(raw, key=raw.get))
            else:
                bias = sum(samples[-20:]) / len(samples[-20:]) if len(samples) >= 4 else 13.0
                side = next(iter(raw))
                raw["SHORT" if side == "LONG" else "LONG"] = 100 + 2 * bias - raw[side]
            other = "SHORT" if side == "LONG" else "LONG"
            pl = pl_long if side == "LONG" else pl_short
            prob = round(raw[side] - bias, 1)
            edge = round(raw[side] - raw[other], 1)
            self._emit_shadow(sym, pl_long, htf, prob, side)
            try:
                snap = self.so.flow().get(sym) or {}
            except Exception:
                snap = {}
            px15 = snap.get("px15")
            if px15 is None:
                px15 = pl_long.get("px15")
            oi15 = snap.get("oi15")
            for sd, plx in plans.items():
                if sd not in rv:
                    continue
                self.record(sym, tsym, sd, px, plx["stop"], plx["tps"][0], round(raw[sd] - bias, 1), m15, htf,
                            pl_long.get("rpos"), pl_long.get("rsi"), self.btc.get("ret1h"), px15, oi15)
            r = rv[side]
            quality = float(r.get("tradeQuality") or 0)
            ev = self.expected_r(r, split, d / px, bias)
            max_prob = self.learner.p.get("self_max_prob")
            if max_prob is not None and prob >= max_prob:
                continue
            if prob < min_prob or edge < MIN_EDGE or quality < MIN_QUALITY or ev < MIN_EV_R:
                self.log("SELF_GATE", symbol=sym, side=side, prob=prob, edge=edge,
                         quality=quality, ev=ev, min_prob=min_prob, min_edge=MIN_EDGE,
                         min_quality=MIN_QUALITY, min_ev=MIN_EV_R)
                continue
            try:
                p1h = float(self.review(sym, side, pl["stop"], pl["tps"], "1H", share=True).get("probability") or 0) - bias
            except Exception:
                p1h = 0.0
            if p1h < CONFIRM_MIN_PROB:
                self.log("SELF_REJECT", symbol=sym, side=side, prob_15m=prob, prob_1h=round(p1h, 1),
                         reason=f"1H 驗算 (去偏後) {p1h:.1f}% < {CONFIRM_MIN_PROB}%，週期不一致")
                continue
            band = S.direc_window(htf if side == "LONG" else None if htf is None else -htf, side)
            found.append(self.to_signal({**c, "side": side,
                                        "strong_liquid_short": band == "strong_short",
                                        "fade_entry": band in ("fade_long", "fade_short")},
                                        r, pl, prob, round(p1h, 1), quality, ev, regime, raw, bias, edge))
            time.sleep(0.3)
        found.sort(key=lambda f: -f["ev"]["score"])
        self.save()
        self._write_align()
        self.log("SELF_SCAN", reviewed_symbols=reviewed, shadow=shadowed, seeone_open=seeone_open,
                 shared_reviews=getattr(self, "shared_hits", 0), open_sides=list(open_sides), long_gate=long_note,
                 universe=len(by_symbol), tradable_contracts=len(specs or {}),
                 tradefi=sum(1 for c in by_symbol.values() if "TradFi" in c["tags"]),
                 watched=[(w["symbol"], w["side"], w["prob"], w["edge"]) for w in watched],
                 qualified=[(f["sig"]["symbol"], f["sig"]["side"], f["ev"]["score"], f["ev"]["analysis"]["edge"]) for f in found],
                 min_prob=min_prob, min_edge=MIN_EDGE, pending_calibration=len(self.d["pending"]),
                 resolved=len(self.d["resolved"]))
        return found[:MAX_PER_SCAN]

    def _quiet_blocked(self, tsym: str, btc_5m: float | None, quote_known: float | None,
                       side: str = "SHORT") -> str | None:
        """Live gates for a quiet entry. None means this name can take a quiet slot.

        Matches bot.py: 1h range ratio, 1h volume ratio, 24h quote volume, not already
        extended versus BTC, not already extended over 2h, and still near the 3h extreme.
        Missing measurements are a skip. An unmeasured name is not a quiet trade.
        """
        sgn = 1 if side == "LONG" else -1
        atr_ratio, ret1h, vol_ratio, quote_vol, px15 = micro_1h(tsym)
        qv = quote_known if quote_known is not None else quote_vol
        qv_min = self.learner.p.get("quote_vol_min", S.QUOTE_VOL_MIN_DEFAULT)
        if qv is None or qv < qv_min:
            shown = 0.0 if qv is None else qv / 1e6
            return f"24h 成交額 {shown:.1f} 百萬 < {qv_min / 1e6:.0f} 百萬"
        atr_min = self.learner.p.get("atr_ratio_min", S.ATR_RATIO_MIN_DEFAULT)
        if atr_ratio is None:
            return "近 1 小時波幅比未知"
        if sgn > 0 and atr_ratio < atr_min:
            return f"近 1 小時波幅比 {atr_ratio:.2f} < {atr_min:.2f}"
        if sgn < 0 and atr_ratio >= S.ATR_RATIO_MAX_SHORT:
            return f"近 1 小時波幅比 {atr_ratio:.2f} ≥ {S.ATR_RATIO_MAX_SHORT:.2f}"
        vol_min = self.learner.p.get("vol_ratio_min", S.VOL_RATIO_MIN_DEFAULT)
        if vol_ratio is None:
            return "近 1 小時量比未知"
        if sgn > 0 and vol_ratio < vol_min:
            return f"近 1 小時量比 {vol_ratio:.2f} < {vol_min:.2f}"
        if sgn < 0 and vol_ratio >= S.VOL_RATIO_MAX_SHORT:
            return f"近 1 小時量比 {vol_ratio:.2f} ≥ {S.VOL_RATIO_MAX_SHORT:.2f}"
        px15_min = self.learner.p.get("px15_dir_min", S.PX15_DIR_MIN_DEFAULT)
        if px15 is not None and px15 * sgn < px15_min:
            return f"近 15 分鐘順方向 {px15 * sgn:+.2f}% < {px15_min:.3f}%"
        rs_max = self.learner.p.get("rs1h_max", S.RS1H_MAX_DEFAULT)
        if ret1h is not None and btc_5m is not None and (ret1h - btc_5m) * sgn >= rs_max:
            return f"近 1 小時已領先 BTC {(ret1h - btc_5m) * sgn:+.2f}% ≥ {rs_max:.3f}%"
        mv = move_2h(tsym)
        if mv and mv["r"] * sgn >= S.MAX_EXT_R:
            return f"近 2 小時已走 {abs(mv['pct']):.1f}% ({mv['r'] * sgn:+.1f}R ≥ {S.MAX_EXT_R}R)"
        rpos = None if not mv or mv.get("rpos") is None else (mv["rpos"] if side == "LONG" else 1 - mv["rpos"])
        if rpos is not None and rpos > S.MAX_RANGE_POS:
            return f"3 小時區間位置 {rpos:.2f} > {S.MAX_RANGE_POS:.2f}"
        return None

    def _quiet_short(self, c: dict, pl_long: dict, regime: tuple, htf: float | None):
        """While 分析驗算 is paused, a liquid short with the higher timeframe already
        aligned, but not extremely so, still had positive R in both halves:
        directional HTF in [1.5, 2.5) and 24h quote volume at or above the live floor,
        37 reviews, +0.76R. The long mirror is separate and was about -1R.
        Quote volume, the 1h range and volume ratios, relative strength, and the 2h
        extension are applied here as well, so a thin name does not take a slot
        before a liquid name is seen.
        """
        if regime[0] > S.REGIME_VETO:
            return None
        if htf is None:
            return None
        htf_dir = -htf
        if not (1.5 <= htf_dir < 2.5):
            return None
        rpos_long = pl_long.get("rpos")
        rpos = None if rpos_long is None else round(1 - rpos_long, 3)
        if rpos is None or rpos > S.MAX_RANGE_POS:
            return None
        px, dist = pl_long["price"], pl_long["dist"]
        pl = dict(pl_long)
        pl["stop"] = px + dist
        pl["tps"] = [px - dist * r for r in TP_R]
        sym = c["symbol"]
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        levels = [{"label": f"TP{i + 1}", "kind": "TP", "sequence": i + 1, "r_multiple": TP_R[i],
                   "price": pl["tps"][i], "hit": False} for i in range(len(TP_R))]
        sig = {"id": f"quiet:{sym}:SHORT:{int(time.time() // 900)}", "symbol": sym, "side": "SHORT",
               "status": "ACTIVE", "timeframe": "15m", "entry_price": px, "stop_price": pl["stop"],
               "current_price": px, "entry_at": now, "levels": levels, "signal_path": "QUIET SHORT",
               "path_subtype": None, "quiet_strong_short": True}
        factors = {"source:安靜期強勢空": 0.0, "side:SHORT": 0.0, f"regime:{regime[1].lower()}": 0.0}
        for t in c["tags"]:
            factors[f"origin:{t.split('#')[0].split('(')[0].split(':')[0]}"] = 0.0
        learned = sum(self.learner.p.get("factor_adj", {}).get(t, 0.0) for t in factors)
        score = round(self.learner.p.get("effective_min_score", S.MIN_SCORE) + learned, 1)
        reasons = [
            "安靜期強勢空：分析驗算暫停，沿用已測過的強勢空單條件",
            f"大週期順方向 {htf_dir:+.2f}，允許 1.5 到 2.5（2.5 以上歷史為負）",
            f"區間位置 {rpos:.2f}，須靠近高點（上限 {S.MAX_RANGE_POS}）",
            f"24h 成交額仍須達 {S.QUOTE_VOL_MIN_DEFAULT/1e6:.0f} 百萬，這一段 +0.76R，流動性不足的是 -0.23R",
            f"止損 {dist / px:.2%} (1.5×ATR15m)，大盤 {regime[1]} {regime[0]:+.2f}",
        ]
        ev = {"ok": True, "score": score, "tier": "NORMAL", "reasons": reasons, "factors": factors,
              "skip": None, "cur_r": 0.0,
              "analysis": {"prob": None, "bias": 0.0, "edge": None, "prob_1h": None,
                           "quality": None, "ev_r": None, "targets": [], "quiet_strong_short": True}}
        return {"sig": sig, "ev": ev, "source": "安靜期強勢空"}

    def _quiet_long(self, c: dict, pl_long: dict, regime: tuple, htf: float | None):
        """Quiet-period long. Same geometry as the short, on the other side.

        Directional HTF in [1.5, 2.5) with the live volume floor was 14 longs,
        about -0.99R in both halves. The path is open because that side was
        requested anyway. It does not fire unless the higher timeframe is up.
        """
        if regime[0] < -S.REGIME_VETO:
            return None
        if htf is None:
            return None
        htf_dir = htf
        if not (1.5 <= htf_dir < 2.5):
            return None
        rpos = pl_long.get("rpos")
        if rpos is None or rpos > S.MAX_RANGE_POS:
            return None
        rpos = round(rpos, 3)
        px, dist = pl_long["price"], pl_long["dist"]
        sym = c["symbol"]
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        levels = [{"label": f"TP{i + 1}", "kind": "TP", "sequence": i + 1, "r_multiple": TP_R[i],
                   "price": pl_long["tps"][i], "hit": False} for i in range(len(TP_R))]
        sig = {"id": f"quiet:{sym}:LONG:{int(time.time() // 900)}", "symbol": sym, "side": "LONG",
               "status": "ACTIVE", "timeframe": "15m", "entry_price": px, "stop_price": pl_long["stop"],
               "current_price": px, "entry_at": now, "levels": levels, "signal_path": "QUIET LONG",
               "path_subtype": None, "quiet_strong_long": True}
        factors = {"source:安靜期強勢多": 0.0, "side:LONG": 0.0, f"regime:{regime[1].lower()}": 0.0}
        for t in c["tags"]:
            factors[f"origin:{t.split('#')[0].split('(')[0].split(':')[0]}"] = 0.0
        learned = sum(self.learner.p.get("factor_adj", {}).get(t, 0.0) for t in factors)
        score = round(self.learner.p.get("effective_min_score", S.MIN_SCORE) + learned, 1)
        reasons = [
            "安靜期強勢多：分析驗算暫停，沿用強勢空的鏡像條件",
            f"大週期順方向 {htf_dir:+.2f}，允許 1.5 到 2.5",
            "這一側 14 筆流動性夠的樣本約 -0.99R，前後半都是負的",
            f"區間位置 {rpos:.2f}，須靠近低點（上限 {S.MAX_RANGE_POS}）",
            f"24h 成交額仍須達 {S.QUOTE_VOL_MIN_DEFAULT/1e6:.0f} 百萬",
            f"止損 {dist / px:.2%} (1.5×ATR15m)，大盤 {regime[1]} {regime[0]:+.2f}",
        ]
        ev = {"ok": True, "score": score, "tier": "NORMAL", "reasons": reasons, "factors": factors,
              "skip": None, "cur_r": 0.0,
              "analysis": {"prob": None, "bias": 0.0, "edge": None, "prob_1h": None,
                           "quality": None, "ev_r": None, "targets": [], "quiet_strong_long": True}}
        return {"sig": sig, "ev": ev, "source": "安靜期強勢多"}

    def to_signal(self, c, rv, pl, prob, p1h, quality, ev_r, regime, raw, bias, edge) -> dict:
        sym, side = c["symbol"], c["side"]
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        targets = rv["geometry"]["targets"]
        levels = [{"label": f"TP{i + 1}", "kind": "TP", "sequence": i + 1, "r_multiple": TP_R[i],
                   "price": pl["tps"][i], "hit": False} for i in range(len(TP_R))]
        sig = {"id": f"self:{sym}:{side}:{int(time.time() // 900)}", "symbol": sym, "side": side, "status": "ACTIVE",
               "timeframe": "15m", "entry_price": pl["price"], "stop_price": pl["stop"], "current_price": pl["price"],
               "entry_at": now, "levels": levels, "signal_path": "SELF ANALYSIS", "path_subtype": None,
               "strong_liquid_short": bool(c.get("strong_liquid_short")),
               "fade_entry": bool(c.get("fade_entry"))}
        factors = {"source:自選驗算": 0.0, f"side:{side}": 0.0, f"regime:{regime[1].lower()}": 0.0,
                   f"selfprob:{int(prob // 10 * 10)}": 0.0, f"selfedge:{min(40, int(edge // 10 * 10))}": 0.0}
        for t in c["tags"]:
            factors[f"origin:{t.split('#')[0].split('(')[0].split(':')[0]}"] = 0.0
        strong = []
        for f in rv.get("factors", []):
            if not f.get("available"):
                continue
            if f["score"] >= 65:
                factors[f"rf:{f['key']}:hi"] = 0.0
                strong.append(f"{f['label']} {f['score']:.0f}")
            elif f["score"] <= 35:
                factors[f"rf:{f['key']}:lo"] = 0.0
        on_board = any(t.startswith("強勢榜") or (t.startswith("潛力榜#") and int(t[4:].split("(")[0]) <= 10)
                       for t in c["tags"])
        ext = pl["move_r"] * (1 if side == "LONG" else -1)
        btc = getattr(self, "btc", {"up": 0.0, "down": 0.0})
        btc_against = btc["down"] if side == "LONG" else btc["up"]
        if ext >= EXT_TAG_R:
            factors["ext:hi"] = 0.0
        if btc_against >= BTC_COUNTER_ATR:
            factors["btc:counter"] = 0.0
        high = (edge >= HIGH_MIN_EDGE and prob >= HIGH_MIN_PROB and quality >= HIGH_MIN_QUALITY
                and p1h >= HIGH_MIN_1H and on_board and self.learner.high_tier_on()
                and ext < S.MAX_EXT_R and btc_against < BTC_COUNTER_ATR)
        if high:
            factors["tier:high"] = 0.0
        adj = self.learner.p.get("factor_adj", {})
        learned = sum(adj.get(t, 0.0) for t in factors)
        score = round(50 + ev_r * 40 + edge * 0.2 + learned, 1)
        aux = 0
        px15 = pl.get("px15")
        if side == "SHORT" and px15 is not None:
            # Inside the ELSA rules, shorts that had not already fallen 0.46% in 15 minutes
            # were the only customized-style split that stayed positive in both halves.
            aux = 4 if -px15 < 0.46 else -6
            score += aux
        reasons = [
            f"自選訊號來源: {', '.join(c['tags'])}",
            f"分析驗算 15M 雙向: 做多 {raw['LONG']}% / 做空 {raw['SHORT']}% (模型偏差 {bias:+.1f})",
            f"去偏後先到 TP1 機率 {prob}%，方向優勢 {edge} 點，交易品質 {quality}，證據分數 {rv.get('evidenceScore')}",
            f"分析驗算 1H 確認 (去偏後) {p1h}%",
            "各 TP 預估命中率: " + " ".join(f"TP{i + 1}({TP_R[i]}R)={t['estimatedHitProbability']}%" for i, t in enumerate(targets)),
            f"期望值 {ev_r:+.2f}R (已扣手續費)，止損 {pl['dist'] / pl['price']:.2%} (1.5×ATR15m)",
            f"大盤 {regime[1]} {regime[0]:+.2f}",
            f"本幣近 2 小時順方向已走 {ext:+.2f}R；BTC 逆方向反彈/回落 {btc_against:.1f} ATR",
            "強項因子: " + ("、".join(strong) if strong else "無"),
            f"See One 結論: {rv.get('conclusion')}",
        ]
        if learned:
            reasons.append(f"實單優化權重合計 {learned:+.1f}")
        if aux:
            reasons.append(f"空頭預備輔助（不決定進場）: 近15分鐘漲跌 {px15:+.2f}%，分數 {aux:+d}")
        if c.get("strong_liquid_short"):
            reasons.append("流動強勢空：這條路徑目前不開新倉")
        if c.get("fade_entry"):
            reasons.append("逆勢帶：1h/4h 略微相反，其餘條件與核心相同。這一段回放勝率與平均 R 都在核心之上")
        if high:
            reasons.append(f"高信心單 (CLANKER 型): 優勢 ≥{HIGH_MIN_EDGE}、機率 ≥{HIGH_MIN_PROB}%、品質 ≥{HIGH_MIN_QUALITY}、"
                           f"1H ≥{HIGH_MIN_1H}%、在強勢榜或潛力榜前 10 → 放大部位")
        ev = {"ok": True, "score": score + (10 if high else 0), "tier": "HIGH" if high else "NORMAL", "reasons": reasons, "factors": factors, "skip": None, "cur_r": 0.0,
              "analysis": {"prob": prob, "raw": raw, "bias": round(bias, 1), "edge": edge, "prob_1h": p1h, "quality": quality, "ev_r": ev_r,
                           "targets": [t["estimatedHitProbability"] for t in targets]}}
        return {"sig": sig, "ev": ev}

    # ------------------------------------------------------------------ open-trade re-verification
    def recheck(self, tr: dict) -> tuple[bool, str | None]:
        """Re-run 分析驗算 on an open self-sourced trade; returns (close?, reason)."""
        if time.time() - tr.get("last_recheck", tr["opened_ts"]) < RECHECK_OPEN_S:
            return False, None
        tr["last_recheck"] = time.time()
        if time.time() - tr["opened_ts"] > MAX_HOLD_H * 3600:
            return True, f"自選訊號持倉超過 {MAX_HOLD_H} 小時，時間止損"
        left = [tp["price"] for tp in tr["tps"] if not tp["filled"]]
        if not left:
            return False, None
        try:
            rv = self.review(tr["tsym"].replace("-SWAP-USDT", "USDT"), tr["side"], tr["stop"], left)
        except Exception:
            return False, None
        p = float(rv.get("probability") or 0) - tr.get("analysis_bias", 0.0)
        tr.setdefault("rechecks", []).append({"ts": int(time.time()), "prob": p})
        if p < EXIT_PROB_BELOW:
            return True, f"分析驗算重評：下一個止盈先到機率降至 {p}% (< {EXIT_PROB_BELOW}%)，提前平倉"
        entry_prob = (tr.get("analysis") or {}).get("prob")
        if entry_prob is not None and entry_prob - p >= EXIT_PROB_DROP:
            return True, (f"分析驗算變弱：進場 {entry_prob:.0f}% 降到 {p:.0f}%"
                          f"（少了 {entry_prob - p:.0f} 點，≥ {EXIT_PROB_DROP:.0f}），提前平倉")
        return False, None
