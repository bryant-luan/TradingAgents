"""See One data + decision logic: market regime, candidate scoring, position sizing."""
import base64
import json
import math
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

SEEONE = "https://seeone.club"
# See One normally answers in well under a second (analysis-review included).
SEEONE_CONNECT_S = 5
HTTP_DEADLINE_S = 20
REVIEW_DEADLINE_S = 40
# After this many failed requests in a row, stop calling See One for the cooldown instead of waiting on every call.
BREAKER_FAILS = 3
BREAKER_COOLDOWN_S = 60


class SeeOneDown(requests.ConnectionError):
    pass


class SeeOneLimited(Exception):
    """analysis-review is in a local cooldown after HTTP 429. No request was sent."""

    def __init__(self, wait_s: float):
        self.wait_s = wait_s
        super().__init__(f"分析驗算限流中，{wait_s:.0f} 秒內不再打")


# Minimum space between analysis-review posts, and how long to stay silent after a 429
# when See One does not send Retry-After. A burst at 11:52 Taipei on Nov 6 started the
# penalty; probing every 5 or 15 minutes kept it alive. One quiet hour matches the gap
# that used to let a normal scan through.
REVIEW_GAP_S = 15
REVIEW_PENALTY_S = 3600
REVIEW_PENALTY_LONG_S = 6 * 3600
REVIEW_PAUSE_FILE = ENV_FILE.parent / "logs" / "seeone_review_pause.json"

# --- tunables -------------------------------------------------------------
REGIME_WEIGHTS = {"15m": 0.25, "1h": 0.35, "4h": 0.40}
REGIME_BULL, REGIME_BEAR = 1.5, -1.5
REGIME_VETO = 1.0               # no longs below -1.0, no shorts above +1.0
MAX_SIGNAL_AGE_MIN = 240
# Official signals whose 15m CVD buy/sell volume ratio is at or above this are skipped: across 168 closed See One
# signals (159 distinct hours) those below 1.5 averaged +0.72R in both halves of the period, those above averaged
# -0.31R. Recalibrated by the shadow optimizer.
CVD_RATIO_MAX_DEFAULT = 1.5
ENTRY_R_WINDOW = (-0.30, 0.50)  # current price must sit within this R band of the signal entry
MIN_SCORE = 50
RISK_PCT = 0.05                 # equity risked per trade at the stop
MIN_RISK_PCT = 0.05             # never open below this risk: a trade that can't be sized this big is skipped
MAX_MIN_SIZE_RISK_PCT = 0.07    # allow exchange-minimum size only up to this risk
MAX_LEVERAGE = 25
MAX_POSITIONS = 3
MAX_TOTAL_MARGIN_PCT = 0.70
MARGIN_PER_TRADE_PCT = 0.07
NORMAL_MARGIN_CAP_PCT = 0.22       # lets a 1% stop at 25x still reach the 5% risk
HIGH_RISK_PCT = 0.10            # high-conviction tier (CLANKER-type setups): equity at risk at the stop
HIGH_MARGIN_CAP_PCT = 0.30
MAX_HIGH_POSITIONS = 1
TP_SPLIT = [0.30, 0.25, 0.20, 0.15, 0.10]
M15_MAX_DEFAULT = 2.5           # cap on the 15m See One direction score in the trade's direction; recalibrated live
HTF_MIN_DEFAULT = -0.5          # floor on the 1h/4h average direction in the trade's direction; recalibrated live
HTF_MAX_DEFAULT = 1.5           # cap on it too: entries with the 1h/4h already strongly aligned arrive late; recalibrated live
FRESH_MAX_R = 0.5               # See One signals: max run-up in the signal's own R over the 2h before it fired
# 3h range position in the trade's direction: 0 = at the reversal extreme (long at the low,
# short at the high), 1 = at the other extreme. Replays through 11-06 12:00 Taipei: entries at
# or above 0.45 averaged about -0.23R in both halves and on both sides; entries below 0.45
# averaged about +0.06R, and the shorts in that zone (shorting the high) about +0.31R.
# Locked here. The old 1.05 cap only blocked prices already outside the range.
MAX_RANGE_POS = 0.45
RSI_MIN_DEFAULT = 0.0           # floor on the 15m RSI in the trade's direction (short: 100 - RSI); off until calibrated
                                # (its edge disappeared once the BTC direction below was accounted for)
# Floor on BTC's last-hour move in the trade's direction, percent. The two sides do not share a cut.
# Longs below +0.055 averaged -0.38R over 20 hours, both halves negative; longs at or above it
# averaged +0.70R (698-row replay, 11-07). Shorts only lose once BTC is meaningfully against them:
# below -0.0425 the short book is flat-to-negative, while the sliver just under 0 was +0.64R on 17 shorts.
BTC1H_MIN_LONG = 0.055
BTC1H_MIN_SHORT = -0.0425
BTC1H_MIN_DEFAULT = BTC1H_MIN_SHORT
# Last-hour average bar range vs the ~25h average, on 5m bars.
# Longs under 0.8 averaged -0.46R, both halves negative. The same low-range shorts averaged +0.26R,
# both halves positive, so the floor is long-only. Shorts lose when the range is already expanded:
# a ratio at or above 1.29 averaged -0.37R, both halves negative.
ATR_RATIO_MIN_DEFAULT = 0.8          # long floor
ATR_RATIO_MAX_SHORT = 1.29           # short cap
# Coin's last-hour return minus BTC's, in the trade's direction, percent. The ≥ 0.924 cut
# (top quartile) was robust on the same three runs: skipped side about -0.09R to -0.16R.
# Looser cuts of this feature were not stable across the three runs, so only this one is live.
RS1H_MAX_DEFAULT = 0.924
# Last-hour quote volume vs the prior ~24h average, on 5m bars.
# Thin volume is a long problem: longs under 0.54 averaged -0.33R. The same thin shorts averaged
# +0.21R, both halves positive, so that floor is long-only. Shorts lose when volume is already hot:
# a ratio at or above 1.19 averaged -0.24R, both halves negative.
VOL_RATIO_MIN_DEFAULT = 0.54          # long floor
VOL_RATIO_MAX_SHORT = 1.19            # short cap
# 24h quote volume (base volume × close, summed over 288 closed 5m bars). The top-quartile cut
# held on the same three runs (~1.14e8 to 1.18e8 USDT): names below it averaged about -0.08R,
# names above stayed positive, including after the range-ratio gate. Lock the loosest of the three.
QUOTE_VOL_MIN_DEFAULT = 1.14e8
# 15-minute price change on closed 5m bars, in the trade's direction, percent.
# Three consecutive robust passes (11-06 17:10–18:10 Taipei) all cut at -0.484:
# a coin that already moved against the trade by more than that averaged about -0.10R
# over 35 hours and both sides; the rest stayed about +0.20R. Locked.
PX15_DIR_MIN_DEFAULT = -0.484
MAX_EXT_R = 2.5                 # no entry once the coin has run this many stop distances in the trade direction over 2h
MAX_SAME_SIDE = 2               # open positions on one side
# Quiet-period shorts may fill the whole book. The live sample was still a handful of
# trades after XAG and XAU; three concurrent names is the cap the account already allows.
QUIET_MAX_SAME_SIDE = 3
# ELSA window (HTF -0.5 to 1.5, BTC 1h aligned, 15m under 2.5, after a See One review)
# is +0.64R on 211 resolved rows, both halves positive. Quiet shorts in the 1.5–2.5
# band are +0.38R on 33 rows. Quiet longs in that band are -0.99R on 14 rows.
# New entries stay on the higher-expectancy ELSA path.
QUIET_ENTRIES = False
SAME_SIDE_GAP_MIN = 30          # minutes between two new entries on the same side
MAX_STOP_PCT = 0.15             # skip signals whose stop is further than this from price
MAKER_FEE, TAKER_FEE = 0.0002, 0.0006
MAX_FEE_R = 0.10                # skip if maker entry + taker stop costs more than this fraction of 1R

# Break-even / profit lock, driven by the best price reached (MFE) rather than by TP fills.
# (MFE in R reached, R locked by the stop); 0.0 = break-even plus round-trip fees.
# Replaying gated reviews on 5m bars, tightening before +2R cut the average result roughly in half
# (+0.10R vs +0.20R): early break-even stops were wicked out ahead of the bigger moves.
# Once MFE is +3R the stop used to sit 1.2R behind the high, so a +3.02R push (XRP, 11-07)
# only locked +1.75R and the runner gave the rest back. After +3R the stop now stays 0.5R
# behind the high, stepped by 0.25R: +3.02R locks +2.50R, +3.66R locks +3.00R.
LOCK_LADDER = ((2.0, 0.0), (3.0, 2.5))
HIGH_BE_TRIGGER_R = 2.0
TP_STOP_MOVES = False           # moving the stop to break-even / the previous TP on each TP fill (same finding)
DAY_LOCK_BE_TRIGGER_R = 0.3     # once the day's profit lock fires, open trades go risk-free at +0.3R
TRAIL_GAP_R = 0.5               # beyond +3R, trail the stop this far behind the best price
TRAIL_STEP_R = 0.25
DAY_LOCK_TRIGGER_PCT = 0.10     # day realized gain at which only high-conviction signals may open
DAILY_STOP_PCT = 0.12           # day loss (realized + floating, incl. open risk to stops) that ends the day
DAY_LOCK_GIVEBACK = 0.5         # stop opening trades after giving back this share of the day's peak gain
POST_LOCK_MAX_OPEN = 1          # after the lock arms: positions opened since then that may be open at once
# After the lock arms, only the 1h/4h band with the best measured TP1-first rate may open (neutral, not already
# strongly aligned: aligned entries arrive late). High-probability reviews are not required: they measured no better.
POST_LOCK_HTF_MIN = -0.5
POST_LOCK_HTF_MAX = 0.5
PROFIT_RESERVE = 0.5            # share of gains above principal kept out of position sizing


def htf_score(dirs: dict) -> float | None:
    vals = [dirs.get(k) for k in ("1h", "4h") if dirs.get(k) is not None]
    return round(sum(vals) / len(vals), 2) if vals else None


def lock_r(mfe_r: float, high: bool = False, day_locked: bool = False) -> float | None:
    ladder = list(LOCK_LADDER)
    if high:
        ladder[0] = (HIGH_BE_TRIGGER_R, 0.0)
    if day_locked:
        ladder[0] = (DAY_LOCK_BE_TRIGGER_R, 0.0)
    best = None
    for trig, lk in ladder:
        if mfe_r >= trig:
            best = lk
    if mfe_r >= ladder[-1][0]:
        trail = math.floor((mfe_r - TRAIL_GAP_R) / TRAIL_STEP_R) * TRAIL_STEP_R
        best = max(best, round(trail, 2))
    return best


class SeeOne:
    def __init__(self, rt_session: str):
        self.rt_session = rt_session
        self.log = lambda event, **kw: None
        self.fails, self.open_until, self.down_since, self.last_error = 0, 0.0, None, None
        self.review_not_before, self._review_streak = self._load_review_pause()
        self.last_review_post = 0.0
        self._cache: dict = {}
        self.http = None
        self._new_session()

    def _new_session(self):
        old = self.http
        self.http = requests.Session()
        self.http.cookies.set("rt_session", self.rt_session, domain="seeone.club")
        self.http.headers["User-Agent"] = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")
        if old is not None:
            old.close()

    def _call(self, method: str, path: str, deadline: float, **kw) -> requests.Response:
        # The socket timeout alone once left a read hanging for 11 hours, so the request runs in its own daemon
        # thread under a wall-clock deadline; a thread that never returns is abandoned instead of blocking the bot.
        out = {}

        def work():
            try:
                out["r"] = self.http.request(method, SEEONE + path, timeout=(min(SEEONE_CONNECT_S, deadline / 2), deadline), **kw)
            except Exception as e:
                out["e"] = e

        th = threading.Thread(target=work, daemon=True)
        th.start()
        th.join(deadline)
        if th.is_alive():
            raise requests.Timeout(f"{path} 超過 {deadline}s 未回應")
        if "e" in out:
            raise out["e"]
        return out["r"]

    def _cached(self, key: str, ttl: float, path: str, **params):
        now = time.time()
        hit = self._cache.get(key)
        if hit and now - hit[0] < ttl:
            return hit[1]
        data = self._get(path, **params)
        self._cache[key] = (now, data)
        return data

    def _load_review_pause(self) -> tuple[float, int]:
        try:
            data = json.loads(REVIEW_PAUSE_FILE.read_text())
            return float(data.get("until") or 0), int(data.get("streak") or 0)
        except (OSError, ValueError):
            return 0.0, 0

    def _save_review_pause(self, until: float, streak: int):
        REVIEW_PAUSE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = REVIEW_PAUSE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps({"until": until, "streak": streak}))
        tmp.replace(REVIEW_PAUSE_FILE)

    def _review_penalty(self, response) -> int:
        raw = response.headers.get("Retry-After") if response is not None else None
        if raw and str(raw).isdigit() and 30 <= int(raw) <= 1800:
            return int(raw)
        return REVIEW_PENALTY_S

    def request(self, method: str, path: str, deadline: float = HTTP_DEADLINE_S, tries: int = 2, **kw):
        if time.time() < self.open_until:
            raise SeeOneDown(f"See One 連線異常 ({self.last_error})，暫停呼叫 {self.open_until - time.time():.0f}s")
        is_review = path.endswith("analysis-review")
        if is_review:
            wait = self.review_not_before - time.time()
            if wait > 0:
                raise SeeOneLimited(wait)
            gap = REVIEW_GAP_S - (time.time() - self.last_review_post)
            if gap > 0:
                time.sleep(gap)
        for attempt in range(tries):
            try:
                r = self._call(method, path, deadline, **kw)
                if r.status_code == 429 and is_review:
                    self._review_streak = getattr(self, "_review_streak", 0) + 1
                    pause = self._review_penalty(r)
                    if pause == REVIEW_PENALTY_S and self._review_streak >= 2:
                        pause = REVIEW_PENALTY_LONG_S
                    self.review_not_before = time.time() + pause
                    self._save_review_pause(self.review_not_before, self._review_streak)
                    self.log("SEEONE_RATE_LIMIT", pause_s=pause, streak=self._review_streak, path=path,
                             note=f"分析驗算 429，{pause // 60} 分鐘內這支完全不打，避免處罰被延續")
                    raise SeeOneLimited(pause)
                if r.status_code == 429:
                    raise requests.HTTPError(f"429 {path}", response=r)
                if r.status_code >= 500:
                    raise requests.HTTPError(f"{r.status_code} {path}")
                r.raise_for_status()
                data = r.json()
            except requests.HTTPError as e:
                status = e.response.status_code if e.response is not None else 0
                # 429 is a rate limit, not an outage: the caller pauses the scan. Other 4xx are the
                # caller's problem and must not open the breaker.
                if e.response is not None and status < 500:
                    raise
                err = e
            except (requests.ConnectionError, requests.Timeout, requests.JSONDecodeError) as e:
                self._new_session()
                err = e
            else:
                if is_review:
                    self.last_review_post = time.time()
                    self._review_streak = 0
                    if REVIEW_PAUSE_FILE.exists():
                        REVIEW_PAUSE_FILE.unlink()
                self._ok()
                return data
            if attempt == tries - 1:
                self._fail(path, err)
                raise err
            time.sleep(2)

    def _ok(self):
        if self.down_since is not None:
            self.log("SEEONE_UP", down_s=int(time.time() - self.down_since), last_error=self.last_error,
                     note="See One 連線恢復，恢復開新倉")
        self.fails, self.open_until, self.down_since = 0, 0.0, None

    def _fail(self, path: str, err: Exception):
        self.fails += 1
        self.last_error = f"{path}: {repr(err)[:160]}"
        if self.fails >= BREAKER_FAILS:
            self.open_until = time.time() + BREAKER_COOLDOWN_S
            if self.down_since is None:
                self.down_since = time.time()
                self.log("SEEONE_DOWN", fails=self.fails, last_error=self.last_error,
                         note=f"See One 連續 {self.fails} 次請求失敗，每 {BREAKER_COOLDOWN_S}s 試一次；"
                              "期間持倉照常管理 (交易所止損/止盈 + 移動止損)，不開新倉")

    def _get(self, path: str, **params):
        return self.request("GET", path, params=params or None)

    def me(self):
        return self._get("/api/account/me")

    def token_exp(self) -> float | None:
        try:
            payload = self.rt_session.split(".")[0]
            payload += "=" * ((4 - len(payload) % 4) % 4)
            return float(json.loads(base64.urlsafe_b64decode(payload)).get("exp"))
        except Exception:
            return None

    def _save_env(self, updates: dict):
        text = ENV_FILE.read_text() if ENV_FILE.exists() else ""
        lines, seen = [], set()
        for line in text.splitlines(True):
            key = line.split("=", 1)[0]
            if key in updates:
                lines.append(key + "=" + updates[key] + ("\n" if line.endswith("\n") else ""))
                seen.add(key)
            else:
                lines.append(line)
        for key, value in updates.items():
            if key not in seen:
                lines.append(key + "=" + value + "\n")
        ENV_FILE.write_text("".join(lines))
        ENV_FILE.chmod(0o600)

    def _adopt(self, session: str, refresh: str | None, note: str):
        self.rt_session = session
        self._new_session()
        saved = {"SEEONE_RT_SESSION": session}
        if refresh:
            saved["SEEONE_RT_REFRESH"] = refresh
        self._save_env(saved)
        self.log("SEEONE_LOGIN", note=note)

    def _refresh_cookie(self) -> bool:
        """A still-valid rt_refresh cookie (about 30 days) mints a new rt_session."""
        from dotenv import load_dotenv
        load_dotenv(ENV_FILE, override=True)
        refresh = os.environ.get("SEEONE_RT_REFRESH") or ""
        if not refresh:
            return False
        try:
            r = requests.get(f"{SEEONE}/api/account/me",
                             cookies={"rt_refresh": refresh},
                             headers={"User-Agent": self.http.headers.get("User-Agent", ""), "Accept": "application/json"},
                             timeout=20)
        except requests.RequestException as e:
            self.log("SEEONE_LOGIN_FAIL", error=repr(e)[:160])
            return False
        session = r.cookies.get("rt_session")
        if r.status_code >= 400 or not session:
            self.log("SEEONE_LOGIN_FAIL", status=r.status_code, note="用長期登入換發失敗，需要在瀏覽器重新登入一次")
            return False
        self._adopt(session, r.cookies.get("rt_refresh") or refresh, "已用長期登入換發 rt_session")
        return True

    def ensure_session(self):
        """Renew rt_session before it expires. The refresh cookie is tried before the email password."""
        exp = self.token_exp()
        if exp and exp - time.time() > 600:
            return
        now = time.time()
        if now < getattr(self, "_login_next", 0):
            return
        self._login_next = now + 300
        if self._refresh_cookie():
            return
        from dotenv import load_dotenv
        load_dotenv(ENV_FILE, override=True)
        email = os.environ.get("SEEONE_EMAIL") or os.environ.get("SEEONE_USERNAME") or ""
        password = os.environ.get("SEEONE_PASSWORD") or ""
        if not email or not password:
            self.log("SEEONE_LOGIN_NEED", note="rt_session 即將到期，沒有可用的長期登入或密碼")
            return
        try:
            r = requests.post(f"{SEEONE}/api/account/login", json={"email": email, "password": password},
                              headers={"User-Agent": self.http.headers.get("User-Agent", ""), "Accept": "application/json"},
                              timeout=20)
            token = r.cookies.get("rt_session")
            if r.status_code >= 400 or not token:
                self.log("SEEONE_LOGIN_FAIL", status=r.status_code, note="帳號密碼登入失敗")
                return
        except (requests.RequestException, ValueError) as e:
            self.log("SEEONE_LOGIN_FAIL", error=repr(e)[:160])
            return
        self._adopt(token, r.cookies.get("rt_refresh"), "已用帳號密碼換發 rt_session")

    def flow(self) -> dict:
        """Current 15m price change and open-interest change by See One symbol, from the visual screener."""
        if time.time() - getattr(self, "_flow_at", 0) < 60 and getattr(self, "_flow", None) is not None:
            return self._flow
        data = self._get("/api/intelligence")
        out = {}
        for row in data.get("visualScreener") or []:
            sym = row.get("symbol")
            if sym:
                out[sym] = {"px15": row.get("priceChange15mPct"), "oi15": row.get("oiChange15mPct")}
        self._flow, self._flow_at = out, time.time()
        return out

    def directions(self) -> dict:
        d = self._cached("directions", 90, "/app/api/market-pulse", part="direction")
        return {x["horizon"]: float(x["score"]) for x in d.get("directions", []) if x.get("score") is not None}

    def boards(self) -> dict:
        return self._cached("boards", 90, "/api/dashboard/boards")

    def signals(self) -> list:
        return self._get("/api/signals")

    def history(self, limit: int = 500) -> list:
        return self._get("/api/history/signals", limit=limit)

    def strength_board(self) -> list:
        return self._cached("turtle", 90, "/api/turtle-active")

    def recipes(self) -> list:
        return self._cached("recipes", 180, "/app/api/customized-signals").get("recipes", [])

    def core_trend_metrics(self) -> dict:
        return self._get("/api/metrics", path="CORE TREND")


def to_toobit(symbol: str) -> str:
    return symbol[:-4] + "-SWAP-USDT" if symbol.endswith("USDT") else symbol


def regime(directions: dict) -> tuple[float, str]:
    score = sum(directions.get(h, 0.0) * w for h, w in REGIME_WEIGHTS.items())
    label = "BULL" if score >= REGIME_BULL else "BEAR" if score <= REGIME_BEAR else "NEUTRAL"
    return round(score, 2), label


def recipe_direction(name: str, rules: list | None = None) -> str | None:
    n = name.lower()
    if "short" in n or "空" in name or "暴跌" in name:
        return "SHORT"
    if "long" in n or "squeeze" in n or "爆漲" in name or "多" in name:
        return "LONG"
    if rules:
        bear = sum(1 for r in rules if r.get("operator") in ("lte", "lt") and (r.get("score") or 0) < 0)
        bull = sum(1 for r in rules if r.get("operator") in ("gte", "gt") and (r.get("score") or 0) > 0)
        if bear > bull:
            return "SHORT"
        if bull > bear:
            return "LONG"
    return None


def _rule_pass(rule: dict, reading: dict | None) -> bool:
    if not isinstance(reading, dict) or reading.get("score") is None or rule.get("score") is None:
        return False
    score, thr, op = reading["score"], rule["score"], rule.get("operator")
    return {"lte": score <= thr, "gte": score >= thr, "lt": score < thr, "gt": score > thr, "eq": score == thr}.get(op, False)


def recipe_hits(recipes: list) -> dict:
    """base asset -> (recipe name, side, rules passed, rule count). Includes partial matches, not only full hits."""
    out: dict[str, list] = {}
    for r in recipes:
        rec = r.get("recipe") or {}
        if not rec.get("enabled", True):
            continue
        rules = rec.get("rules") or []
        name = rec.get("name") or "?"
        side = recipe_direction(name, rules)
        if not side or not rules:
            continue
        seen = set()
        for row in (r.get("evaluation") or {}).get("results") or []:
            base = str(row.get("asset") or "").upper()
            if not base:
                continue
            readings = row.get("readings") or {}
            passed = sum(_rule_pass(rule, readings.get(rule.get("metric"))) for rule in rules)
            if row.get("match"):
                passed = len(rules)
            if passed <= 0:
                continue
            seen.add(base)
            out.setdefault(base, []).append((name, side, passed, len(rules)))
        for base, hit in (r.get("members") or {}).items():
            if hit and str(base).upper() not in seen:
                out.setdefault(str(base).upper(), []).append((name, side, len(rules), len(rules)))
    return out


def recipe_for(hits: dict, base: str, side: str) -> tuple | None:
    """Best same-direction recipe progress for this asset, or None."""
    rows = [h for h in hits.get(base, []) if h[1] == side]
    return max(rows, key=lambda h: h[2] / h[3]) if rows else None


def path_expectancy(signals: list, own_trades: list) -> dict:
    """Mean realized R per signal_path, from See One's closed signals and our own closed trades."""
    stats: dict[str, list] = {}
    for s in signals:
        if s.get("status") == "CLOSED" and s.get("realized_r") is not None:
            stats.setdefault(s.get("signal_path") or "?", []).append(float(s["realized_r"]))
    own: dict[str, list] = {}
    for t in own_trades[-30:]:
        if t.get("r_result") is not None:
            own.setdefault(t.get("path") or "?", []).append(float(t["r_result"]))
    return {
        "seeone": {k: (round(sum(v) / len(v), 3), len(v)) for k, v in stats.items()},
        "own": {k: (round(sum(v) / len(v), 3), len(v)) for k, v in own.items()},
    }


def cvd_ratio_of(sig: dict) -> float | None:
    """Buy/sell volume ratio from the signal's own CVD snapshot (None when See One didn't record one)."""
    ratio = ((sig.get("indicator_snapshot") or {}).get("cvd") or {}).get("ratio")
    return None if ratio is None else float(ratio)


def intrinsic_tags(sig: dict) -> list:
    """Traits known at signal time, independent of market context; used by the shadow optimizer."""
    snap = sig.get("scanner_snapshot") or {}
    sc = float(sig.get("scanner_score") or 0)
    rd, rk = snap.get("readiness"), snap.get("rank")
    tags = [f"side:{sig['side']}", f"tf:{sig.get('timeframe')}", f"subtype:{sig.get('path_subtype') or 'NONE'}",
            f"scanner:{int(sc // 5 * 5)}", "rank:top5" if rk and rk <= 5 else "rank:top10" if rk and rk <= 10 else "rank:other"]
    if rd is not None:
        tags.append(f"ready:{int(float(rd) // 10 * 10)}")
    return tags


def _age_min(ts: str) -> float:
    t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return (datetime.now(timezone.utc) - t).total_seconds() / 60


def evaluate(sig: dict, source: str, regime_score: float, regime_label: str,
             board: dict, rhits: dict, pexp: dict, learned: dict | None = None) -> dict:
    """Score one signal. Returns ok/score/skip, human reasons, and machine factor tags (tag -> points)."""
    learned = learned or {}
    side = sig["side"]
    base = sig["symbol"][:-4]
    reasons, factors, skip = [], {}, None
    entry, stop, cur = float(sig["entry_price"]), float(sig["stop_price"]), float(sig["current_price"])
    risk = abs(entry - stop)
    cur_r = (cur - entry) / risk * (1 if side == "LONG" else -1) if risk else 0.0
    age = _age_min(sig["entry_at"])
    tp_hit = any(l.get("hit") for l in sig.get("levels", []) if l.get("kind") == "TP" or str(l.get("label", "")).startswith("TP"))
    r_lo, r_hi = learned.get("entry_r_window", ENTRY_R_WINDOW)
    min_score = learned.get("effective_min_score", learned.get("min_score", MIN_SCORE))
    cooldown_until = learned.get("symbol_cooldown", {}).get(sig["symbol"], 0)

    if cooldown_until > datetime.now(timezone.utc).timestamp():
        skip = f"{sig['symbol']} 近期虧損冷卻中 (至 {datetime.fromtimestamp(cooldown_until, timezone.utc):%m-%d %H:%M} UTC)"
    elif age > MAX_SIGNAL_AGE_MIN:
        skip = f"訊號已 {age:.0f} 分鐘 (> {MAX_SIGNAL_AGE_MIN})"
    elif tp_hit or sig.get("highest_hit_tp_label"):
        skip = "已打到 TP，不追價"
    elif not (r_lo <= cur_r <= r_hi):
        skip = f"現價偏離進場 {cur_r:+.2f}R (允許 {r_lo}~{r_hi}R)"
    elif (MAKER_FEE + TAKER_FEE) / (risk / cur) > MAX_FEE_R:
        skip = f"止損僅 {risk / cur:.2%}，手續費約佔 {(MAKER_FEE + TAKER_FEE) / (risk / cur):.2f}R (> {MAX_FEE_R}R)"
    elif risk / cur > MAX_STOP_PCT:
        skip = f"止損距離 {risk / cur:.0%} 過寬 (> {MAX_STOP_PCT:.0%})，TP 難以到達"
    else:
        ratio = cvd_ratio_of(sig)
        cvd_max = learned.get("cvd_ratio_max", CVD_RATIO_MAX_DEFAULT)
        if ratio is not None and ratio >= cvd_max:
            skip = (f"CVD 買賣量比 {ratio:.2f} ≥ {cvd_max:.2f}，買盤已遠大於賣盤 (擁擠)，"
                    f"這類訊號歷史上平均虧損，等買賣量接近均衡再進")

    def add(tag: str, pts: float, text: str):
        factors[tag] = round(pts, 1)
        reasons.append(f"{text} {pts:+.1f}")

    scanner = float(sig.get("scanner_score") or (sig.get("strength_score") or 0))
    add(f"source:{source}", scanner * 0.4, f"來源={source} 掃描分數={scanner:.1f}×0.4")

    aligned = (regime_label == "BULL" and side == "LONG") or (regime_label == "BEAR" and side == "SHORT")
    if aligned:
        add("regime:aligned", 20, f"大盤 {regime_label} {regime_score:+.2f} 與方向一致")
    elif regime_label == "NEUTRAL":
        add("regime:neutral", 10, f"大盤中性 {regime_score:+.2f}")
    else:
        add("regime:counter", 0, f"大盤 {regime_label} {regime_score:+.2f} 逆勢")

    b = board.get(sig["symbol"])
    if b and b["side"] == side:
        add("board:same", 15, f"強勢榜 #{b.get('rank')} 同向 (strength {b.get('strength_score')})")
    elif b:
        add("board:opposite", -10, f"強勢榜 #{b.get('rank')} 反向")

    snap = sig.get("scanner_snapshot") or {}
    rank = snap.get("rank")
    if rank is not None and rank <= 10:
        add("potential:top10", 5, f"潛力榜排名 #{rank}")
    hint = (snap.get("raw_features") or {}).get("directionHint")
    if hint and hint != side:
        add("hint:opposite", -10, f"掃描方向提示 {hint} 與訊號相反")

    for name, d, passed, total in rhits.get(base, []):
        if d == side and passed == total:
            add(f"recipe:{name}", 15, f"客製化訊號「{name}」{passed}/{total} 條全中")
        elif d == side and passed * 2 >= total:
            add(f"recipe:{name}", 8, f"客製化訊號「{name}」過半 {passed}/{total}")
        elif d and d != side and passed * 2 >= total:
            add(f"recipe_opp:{name}", -15, f"客製化訊號「{name}」反向已過半 {passed}/{total}")

    path = sig.get("signal_path") or "?"
    exp, n = pexp["seeone"].get(path, (0.0, 0))
    if n >= 10:
        add(f"path:{path}", max(-20.0, min(15.0, exp * 30)), f"路徑 {path} See One 近期期望值 {exp:+.3f}R (n={n})")
    else:
        factors[f"path:{path}"] = 0.0
    for tag in intrinsic_tags(sig):
        factors.setdefault(tag, 0.0)

    own_adj, sh_adj = learned.get("factor_adj", {}), learned.get("shadow_adj", {})
    for tag in list(factors):
        if own_adj.get(tag):
            reasons.append(f"實單優化權重 [{tag}] {own_adj[tag]:+.1f}")
        if sh_adj.get(tag):
            reasons.append(f"See One 全訊號優化權重 [{tag}] {sh_adj[tag]:+.1f}")
    score = round(sum(factors.values()) + sum(own_adj.get(t, 0.0) + sh_adj.get(t, 0.0) for t in factors), 1)

    reasons.append(f"訊號年齡 {age:.0f}m，現價位於 {cur_r:+.2f}R")
    if not skip and score < min_score:
        skip = f"綜合分數 {score} < {min_score}"
    return {"ok": skip is None, "score": score, "reasons": reasons, "factors": factors,
            "skip": skip, "cur_r": round(cur_r, 3)}


def size_position(equity: float, price: float, stop: float, spec: dict, risk_pct: float = RISK_PCT,
                  margin_cap_pct: float = MARGIN_PER_TRADE_PCT * 1.5) -> dict | None:
    """Risk-based sizing in contracts plus leverage. Returns None if the trade can't be sized safely."""
    mult = float(spec["multiplier"])
    dist = abs(price - stop)
    if dist <= 0:
        return None
    sl_pct = dist / price
    risk_usdt = equity * risk_pct
    contracts = math.floor(risk_usdt / (dist * mult))
    if contracts * dist * mult < equity * MIN_RISK_PCT:
        up = math.ceil(equity * MIN_RISK_PCT / (dist * mult))
        if up * dist * mult <= equity * MAX_MIN_SIZE_RISK_PCT:
            contracts = up
    if contracts < spec["min_contracts"]:
        min_risk = spec["min_contracts"] * dist * mult
        if min_risk > equity * MAX_MIN_SIZE_RISK_PCT:
            return None
        contracts = spec["min_contracts"]
    notional = contracts * mult * price
    # 0.60 keeps the stop inside the liquidation distance at higher leverage.
    # A 2.3% stop can use 25x; the old 0.40 cap held that stop near 17x.
    lev_cap = max(1, min(spec["max_leverage"], MAX_LEVERAGE, math.floor(0.60 / sl_pct)))
    leverage = max(1, min(lev_cap, math.ceil(notional / (MARGIN_PER_TRADE_PCT * equity))))
    margin = notional / leverage
    if margin > margin_cap_pct * equity:
        contracts = math.floor(margin_cap_pct * equity * leverage / (mult * price))
        if contracts < spec["min_contracts"]:
            return None
        notional = contracts * mult * price
        margin = notional / leverage
    if contracts * mult * dist < equity * MIN_RISK_PCT * 0.95:
        return None
    return {
        "contracts": contracts,
        "leverage": leverage,
        "notional": round(notional, 2),
        "margin": round(margin, 2),
        "risk_usdt": round(contracts * mult * dist, 2),
        "sl_pct": round(sl_pct * 100, 2),
    }


def tp_plan(sig: dict, contracts: int, min_contracts: int, split: list | None = None) -> list:
    """Split contracts over the signal's TP1..TP5; fold undersized slices into the previous one."""
    levels = sorted((l for l in sig.get("levels", []) if str(l.get("label", "")).startswith("TP")),
                    key=lambda l: l.get("sequence", int(l["label"][2:])))
    split = split or TP_SPLIT
    plan, left = [], contracts
    for i, lvl in enumerate(levels[:5]):
        qty = left if i == len(levels[:5]) - 1 else int(contracts * split[i])
        qty = min(qty, left)
        if qty < min_contracts:
            if plan:
                continue
            qty = min(left, min_contracts)
        plan.append({"label": lvl["label"], "price": float(lvl["price"]), "r": lvl.get("r_multiple"), "qty": qty})
        left -= qty
        if left <= 0:
            break
    if left > 0 and plan:
        plan[-1]["qty"] += left
    return plan
