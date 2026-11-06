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
import math
import time
from pathlib import Path

import requests

from . import strategy as S

SCAN_EVERY_S = 300
MAJORS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT"]
ATR_MULT = 1.5
MIN_STOP_PCT = 0.008
MAX_STOP_PCT = 0.06
TP_R = [1.0, 1.5, 2.0, 3.0, 4.0]
DEFAULT_MIN_PROB = 60.0          # de-biased probability of TP1 before SL
MIN_EDGE = 25.0                  # raw prob(side) - prob(opposite side), percentage points
MAX_PER_SCAN = 3
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
RECHECK_OPEN_S = 900
EXIT_PROB_BELOW = 40.0
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
# 15m RSI in the trade direction; the floor is the bottom of the lowest bucket that still passes, scanning down.
RSI_BUCKETS = [(0.0, 40.0), (40.0, 50.0), (50.0, 60.0), (60.0, 101.0)]
# BTC 1h move in the trade direction, percent.
BTC1H_BUCKETS = [(-99.0, -0.3), (-0.3, 0.0), (0.0, 0.3), (0.3, 99.0)]
# Reviews taken in the same scan all ride the same market move, so sample counts overstate the evidence.
# The gates keep their defaults until the records span this many distinct hours.
GATE_MIN_HOURS = 24
# ~190 resolved reviews per active hour: keep several days so the gates can ever reach GATE_MIN_HOURS,
# and calibrate on a time window instead of a fixed record count.
RESOLVED_KEEP = 20000
CAL_WINDOW_H = 72


def atr(symbol: str, n: int = 14) -> tuple[float, list] | None:
    rows = requests.get("https://api.toobit.com/quote/v1/klines",
                        params={"symbol": symbol, "interval": "15m", "limit": n + 1}, timeout=10).json()
    if len(rows) < n + 1:
        return None
    trs = []
    for prev, cur in zip(rows, rows[1:]):
        h, l, pc = float(cur[2]), float(cur[3]), float(prev[4])
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return sum(trs) / len(trs), rows


def move_before(symbol: str, end_ms: int, side: str, risk: float) -> float | None:
    """Move over the 2h before end_ms in the trade direction, in units of the given stop distance."""
    rows = requests.get("https://api.toobit.com/quote/v1/klines",
                        params={"symbol": symbol, "interval": "15m", "limit": EXT_BARS + 1, "endTime": end_ms - 1},
                        timeout=10).json()
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


def micro_1h(symbol: str) -> tuple[float | None, float | None, float | None, float | None]:
    """(range ratio, 1h return %, volume ratio, 24h quote volume) on closed 5m bars.

    Same definitions research.py scores. Range ratio is the mean high-low of the last 12 bars over the
    last 300 (~25h). The return is the last close versus the close 12 bars earlier, long perspective.
    Volume ratio is the last hour's mean quote volume over the mean of the 288 bars before it.
    Quote volume is base volume times close, summed over those 288 bars (24h).
    """
    empty = (None, None, None, None)
    try:
        rows = requests.get("https://api.toobit.com/quote/v1/klines",
                            params={"symbol": symbol, "interval": "5m", "limit": 301}, timeout=10).json()
    except (requests.RequestException, ValueError):
        return empty
    if not isinstance(rows, list) or len(rows) < 301:
        return empty
    past = rows[:-1][-300:]

    def mean_range(bars: list) -> float:
        return sum(float(x[2]) - float(x[3]) for x in bars) / len(bars)

    base = mean_range(past)
    ratio = None if base <= 0 else mean_range(past[-12:]) / base
    c0, c1 = float(past[-12][4]), float(past[-1][4])
    ret = None if c0 <= 0 else (c1 - c0) / c0 * 100
    qv = [float(x[5]) * float(x[4]) for x in past]
    earlier = qv[:-12]
    vol_ratio = (sum(qv[-12:]) / 12) / ((sum(earlier) / len(earlier)) or 1)
    quote_vol = sum(qv[-288:])
    return ratio, ret, vol_ratio, quote_vol


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
        self.basis: dict[str, tuple[float, float]] = {}

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

    def review(self, symbol: str, side: str, stop: float, tps: list, tf: str = "15M") -> dict:
        """分析驗算 on See One's own price basis: See One's market price can sit apart from Toobit's, so the Toobit
        stop/TPs are scaled by the See One/Toobit price ratio to keep the same R geometry."""
        px = self.tb.price(S.to_toobit(symbol))
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
    def calibrated(self) -> bool:
        return bool(self.learner.p.get("self_calibrated"))

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
            try:
                rows = requests.get("https://api.toobit.com/quote/v1/klines",
                                    params={"symbol": p["tsym"], "interval": "15m", "limit": 100}, timeout=10).json()
            except requests.RequestException:
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
        # The measured best set is "BTC 1h must not be against the trade" (floor 0). Keep that floor.
        if self.learner.p.get("btc1h_min", S.BTC1H_MIN_DEFAULT) != S.BTC1H_MIN_DEFAULT:
            self.learner.p["btc1h_min"] = S.BTC1H_MIN_DEFAULT
            self.learner.save()

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

    def calibrate_quote_vol(self, rows: list):
        # Illiquid names. The top-quartile cut held for three runs; lock the loosest of them.
        if self.learner.p.get("quote_vol_min", S.QUOTE_VOL_MIN_DEFAULT) != S.QUOTE_VOL_MIN_DEFAULT:
            self.learner.p["quote_vol_min"] = S.QUOTE_VOL_MIN_DEFAULT
            self.learner.save()

    def _calibrate_floor(self, rows: list, key: str, buckets: list, param: str, default: float, event: str, note: str):
        """Floor = bottom of the lowest bucket reached scanning down from the top while every bucket passes;
        the lowest bucket's bottom (no floor) when all pass."""
        if not self._gate_ready(rows, key, param, default):
            return
        table = []
        for lo, hi in buckets:
            b = [r["hit"] for r in rows if r.get(key) is not None and lo <= r[key] < hi]
            table.append((lo, hi, len(b), round(sum(b) / len(b), 3) if b else None))
        if any(n < M15_MIN_N for _, _, n, _ in table[-2:]):
            return
        floor = buckets[-1][0]
        for lo, _, n, rate in reversed(table):
            if n < M15_MIN_N or rate < M15_PASS:
                break
            floor = lo
        old = self.learner.p.get(param, default)
        self.learner.p[param] = floor
        self.learner.save()
        if floor != old or not self.learner.p.get(f"{param}_logged"):
            self.learner.p[f"{param}_logged"] = True
            self.log(event, **{f"{param}_old": old, f"{param}_new": floor},
                     buckets=[{f"{key}_in_side": f"{lo}~{hi}", "n": n, "tp1_first_rate": r} for lo, hi, n, r in table],
                     note=note)

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
        default = "LONG" if regime_label == "BULL" else "SHORT" if regime_label == "BEAR" else None
        for m in MAJORS:
            cands.setdefault(m, {"side": default, "tags": ["主流幣"]})
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

    def scan(self, regime: tuple, rhits: dict, held: set, specs: dict, m15: float | None = None,
             htf: float | None = None) -> list:
        """Review every candidate both LONG and SHORT on mirrored geometry. 分析驗算 is direction-biased
        (mirrored 1R probabilities sum to well over 100%), so each side is de-biased by half the excess and
        a trade needs a clear edge over the opposite side, not just a high raw probability."""
        if time.time() - self.last_scan < SCAN_EVERY_S:
            return []
        if self.so.review_not_before > time.time():
            return []
        self.last_scan = time.time()
        self.resolve()
        split = self.learner.p.get("tp_split") or S.TP_SPLIT
        min_prob = self.min_prob()
        try:
            self.btc = self.btc_counter()
        except Exception:
            self.btc = {"up": 0.0, "down": 0.0}
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
        ordered = []
        while boards or rest:
            if boards:
                ordered.append(boards.pop(0))
            if rest:
                ordered.append(rest.pop(0))
        for c in ordered:
            sym = c["symbol"]
            tsym = S.to_toobit(sym)
            if tsym not in specs or tsym in held:
                continue
            try:
                pl_long = self.plan(tsym, "LONG")
                if not pl_long:
                    continue
                d, px = pl_long["dist"], pl_long["price"]
                pl_short = {**pl_long, "stop": px + d, "tps": [px - d * r for r in TP_R]}
                rv = {"LONG": self.review(sym, "LONG", pl_long["stop"], pl_long["tps"])}
                time.sleep(1.2)
                rv["SHORT"] = self.review(sym, "SHORT", pl_short["stop"], pl_short["tps"])
            except Exception as e:
                self.log("ERROR", where="self_review", symbol=sym, error=repr(e)[:300])
                # SeeOneLimited is raised only after a real 429 (or when the shared cooldown is
                # still open). The clock lives on the client, so this scan just stops.
                if isinstance(e, S.SeeOneLimited):
                    self.d["last_review"][sym] = time.time()
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
            bias = (raw["LONG"] + raw["SHORT"] - 100) / 2
            side = max(raw, key=raw.get)
            other = "SHORT" if side == "LONG" else "LONG"
            pl = pl_long if side == "LONG" else pl_short
            prob = round(raw[side] - bias, 1)
            edge = round(raw[side] - raw[other], 1)
            try:
                snap = self.so.flow().get(sym) or {}
            except Exception:
                snap = {}
            px15 = snap.get("px15")
            if px15 is None:
                px15 = pl_long.get("px15")
            oi15 = snap.get("oi15")
            for sd, plx in (("LONG", pl_long), ("SHORT", pl_short)):
                self.record(sym, tsym, sd, px, plx["stop"], plx["tps"][0], round(raw[sd] - bias, 1), m15, htf,
                            pl_long.get("rpos"), pl_long.get("rsi"), self.btc.get("ret1h"), px15, oi15)
            r = rv[side]
            quality = float(r.get("tradeQuality") or 0)
            ev = self.expected_r(r, split, d / px, bias)
            max_prob = self.learner.p.get("self_max_prob")
            if max_prob is not None and prob >= max_prob:
                continue
            if prob < min_prob or edge < MIN_EDGE or quality < MIN_QUALITY or ev < MIN_EV_R:
                continue
            # Longs in a bear tape and shorts in a bull tape are still scored and written down. They are not
            # opened: countertrend longs at the 3h low hit TP1 only about 43%, and a 15m turn-up against the
            # higher timeframe hit TP1 about 17%.
            if (side == "LONG" and regime[0] < -S.REGIME_VETO) or (side == "SHORT" and regime[0] > S.REGIME_VETO):
                rec = {"symbol": sym, "side": side, "prob": prob, "edge": edge, "ev_r": ev,
                       "rpos": None if pl_long.get("rpos") is None else round(in_side(pl_long["rpos"], side), 2),
                       "htf": None if htf is None else round(htf * (1 if side == "LONG" else -1), 2)}
                watched.append(rec)
                self.log("SIDE_WATCH", **rec, regime=round(regime[0], 2),
                         note="與大盤相反，已記錄，不進場")
                continue
            try:
                p1h = float(self.review(sym, side, pl["stop"], pl["tps"], "1H").get("probability") or 0) - bias
            except Exception:
                p1h = 0.0
            if p1h < CONFIRM_MIN_PROB:
                self.log("SELF_REJECT", symbol=sym, side=side, prob_15m=prob, prob_1h=round(p1h, 1),
                         reason=f"1H 驗算 (去偏後) {p1h:.1f}% < {CONFIRM_MIN_PROB}%，週期不一致")
                continue
            found.append(self.to_signal({**c, "side": side}, r, pl, prob, round(p1h, 1), quality, ev, regime,
                                        raw, bias, edge))
            time.sleep(0.3)
        found.sort(key=lambda f: -f["ev"]["score"])
        self.save()
        self.log("SELF_SCAN", reviewed_symbols=reviewed,
                 universe=len(by_symbol),
                 tradefi=sum(1 for c in by_symbol.values() if "TradFi" in c["tags"]),
                 watched=[(w["symbol"], w["side"], w["prob"], w["edge"]) for w in watched],
                 qualified=[(f["sig"]["symbol"], f["sig"]["side"], f["ev"]["score"], f["ev"]["analysis"]["edge"]) for f in found],
                 min_prob=min_prob, min_edge=MIN_EDGE, pending_calibration=len(self.d["pending"]),
                 resolved=len(self.d["resolved"]))
        return found[:MAX_PER_SCAN]

    def to_signal(self, c, rv, pl, prob, p1h, quality, ev_r, regime, raw, bias, edge) -> dict:
        sym, side = c["symbol"], c["side"]
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        targets = rv["geometry"]["targets"]
        levels = [{"label": f"TP{i + 1}", "kind": "TP", "sequence": i + 1, "r_multiple": TP_R[i],
                   "price": pl["tps"][i], "hit": False} for i in range(len(TP_R))]
        sig = {"id": f"self:{sym}:{side}:{int(time.time() // 900)}", "symbol": sym, "side": side, "status": "ACTIVE",
               "timeframe": "15m", "entry_price": pl["price"], "stop_price": pl["stop"], "current_price": pl["price"],
               "entry_at": now, "levels": levels, "signal_path": "SELF ANALYSIS", "path_subtype": None}
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
        return False, None
