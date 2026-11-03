"""See One data + decision logic: market regime, candidate scoring, position sizing."""
import math
from datetime import datetime, timezone

import requests

SEEONE = "https://seeone.club"

# --- tunables -------------------------------------------------------------
REGIME_WEIGHTS = {"15m": 0.25, "1h": 0.35, "4h": 0.40}
REGIME_BULL, REGIME_BEAR = 1.5, -1.5
REGIME_VETO = 1.0               # no longs below -1.0, no shorts above +1.0
MAX_SIGNAL_AGE_MIN = 240
ENTRY_R_WINDOW = (-0.30, 0.50)  # current price must sit within this R band of the signal entry
MIN_SCORE = 50
RISK_PCT = 0.01                 # equity risked per trade at the stop
MAX_MIN_SIZE_RISK_PCT = 0.015   # allow exchange-minimum size only up to this risk
MAX_LEVERAGE = 10
MAX_POSITIONS = 3
MAX_TOTAL_MARGIN_PCT = 0.60
MARGIN_PER_TRADE_PCT = 0.20
TP_SPLIT = [0.30, 0.25, 0.20, 0.15, 0.10]


class SeeOne:
    def __init__(self, rt_session: str):
        self.http = requests.Session()
        self.http.cookies.set("rt_session", rt_session, domain="seeone.club")
        self.http.headers["User-Agent"] = "Mozilla/5.0 seeone-toobit-bot"

    def _get(self, path: str, **params):
        r = self.http.get(SEEONE + path, params=params or None, timeout=20)
        r.raise_for_status()
        return r.json()

    def me(self):
        return self._get("/api/account/me")

    def directions(self) -> dict:
        d = self._get("/app/api/market-pulse", part="direction")
        return {x["horizon"]: float(x["score"]) for x in d.get("directions", [])}

    def signals(self) -> list:
        return self._get("/api/signals")

    def strength_board(self) -> list:
        return self._get("/api/turtle-active")

    def recipes(self) -> list:
        return self._get("/app/api/customized-signals").get("recipes", [])

    def core_trend_metrics(self) -> dict:
        return self._get("/api/metrics", path="CORE TREND")


def to_toobit(symbol: str) -> str:
    return symbol[:-4] + "-SWAP-USDT" if symbol.endswith("USDT") else symbol


def regime(directions: dict) -> tuple[float, str]:
    score = sum(directions.get(h, 0.0) * w for h, w in REGIME_WEIGHTS.items())
    label = "BULL" if score >= REGIME_BULL else "BEAR" if score <= REGIME_BEAR else "NEUTRAL"
    return round(score, 2), label


def recipe_direction(name: str) -> str | None:
    n = name.lower()
    if "short" in n or "空" in name or "暴跌" in name:
        return "SHORT"
    if "long" in n or "squeeze" in n or "爆漲" in name or "多" in name:
        return "LONG"
    return None


def recipe_hits(recipes: list) -> dict:
    """base asset -> list of (recipe name, implied direction) currently matching."""
    out: dict[str, list] = {}
    for r in recipes:
        if not r.get("recipe", {}).get("enabled", True):
            continue
        name = r["recipe"]["name"]
        for base, hit in (r.get("members") or {}).items():
            if hit:
                out.setdefault(base, []).append((name, recipe_direction(name)))
    return out


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


def _age_min(ts: str) -> float:
    t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return (datetime.now(timezone.utc) - t).total_seconds() / 60


def evaluate(sig: dict, source: str, regime_score: float, regime_label: str,
             board: dict, rhits: dict, pexp: dict) -> dict:
    """Return {'ok': bool, 'score': float, 'reasons': [...], 'skip': str|None} for one signal."""
    side = sig["side"]
    base = sig["symbol"][:-4]
    reasons, skip = [], None
    entry, stop, cur = float(sig["entry_price"]), float(sig["stop_price"]), float(sig["current_price"])
    risk = abs(entry - stop)
    cur_r = (cur - entry) / risk * (1 if side == "LONG" else -1) if risk else 0.0
    age = _age_min(sig["entry_at"])
    tp_hit = any(l.get("hit") for l in sig.get("levels", []) if l.get("kind") == "TP" or str(l.get("label", "")).startswith("TP"))

    if age > MAX_SIGNAL_AGE_MIN:
        skip = f"訊號已 {age:.0f} 分鐘 (> {MAX_SIGNAL_AGE_MIN})"
    elif tp_hit or sig.get("highest_hit_tp_label"):
        skip = "已打到 TP，不追價"
    elif not (ENTRY_R_WINDOW[0] <= cur_r <= ENTRY_R_WINDOW[1]):
        skip = f"現價偏離進場 {cur_r:+.2f}R (允許 {ENTRY_R_WINDOW[0]}~{ENTRY_R_WINDOW[1]}R)"
    elif side == "LONG" and regime_score < -REGIME_VETO:
        skip = f"大盤偏空 {regime_score:+.2f}，否決做多"
    elif side == "SHORT" and regime_score > REGIME_VETO:
        skip = f"大盤偏多 {regime_score:+.2f}，否決做空"

    scanner = float(sig.get("scanner_score") or (sig.get("strength_score") or 0))
    score = scanner * 0.4
    reasons.append(f"來源={source} 掃描分數={scanner:.1f} (×0.4={scanner * 0.4:.1f})")

    aligned = (regime_label == "BULL" and side == "LONG") or (regime_label == "BEAR" and side == "SHORT")
    if aligned:
        score += 20
        reasons.append(f"大盤 {regime_label} {regime_score:+.2f} 與方向一致 +20")
    elif regime_label == "NEUTRAL":
        score += 10
        reasons.append(f"大盤中性 {regime_score:+.2f} +10")
    else:
        reasons.append(f"大盤 {regime_label} {regime_score:+.2f} 逆勢 +0")

    b = board.get(sig["symbol"])
    if b and b["side"] == side:
        score += 15
        reasons.append(f"強勢榜 #{b.get('rank')} 同向 (strength {b.get('strength_score')}) +15")
    elif b:
        score -= 10
        reasons.append(f"強勢榜 #{b.get('rank')} 反向 -10")

    snap = sig.get("scanner_snapshot") or {}
    rank = snap.get("rank")
    if rank is not None and rank <= 10:
        score += 5
        reasons.append(f"潛力榜排名 #{rank} +5")
    hint = (snap.get("raw_features") or {}).get("directionHint")
    if hint and hint != side:
        score -= 10
        reasons.append(f"掃描方向提示 {hint} 與訊號相反 -10")

    for name, d in rhits.get(base, []):
        if d == side:
            score += 15
            reasons.append(f"客製化訊號「{name}」同向命中 +15")
        elif d:
            score -= 15
            reasons.append(f"客製化訊號「{name}」反向命中 -15")

    path = sig.get("signal_path") or "?"
    exp, n = pexp["seeone"].get(path, (0.0, 0))
    if n >= 10:
        adj = round(max(-20.0, min(15.0, exp * 30)), 1)
        score += adj
        reasons.append(f"路徑 {path} 近期期望值 {exp:+.3f}R (n={n}) {adj:+.1f}")
    oexp, on = pexp["own"].get(path, (0.0, 0))
    if on >= 5 and oexp < 0:
        score -= 15
        reasons.append(f"本機器人在 {path} 實績 {oexp:+.3f}R (n={on}) -15")

    reasons.append(f"訊號年齡 {age:.0f}m，現價位於 {cur_r:+.2f}R")
    score = round(score, 1)
    if not skip and score < MIN_SCORE:
        skip = f"綜合分數 {score} < {MIN_SCORE}"
    return {"ok": skip is None, "score": score, "reasons": reasons, "skip": skip, "cur_r": cur_r}


def size_position(equity: float, price: float, stop: float, spec: dict) -> dict | None:
    """Risk-based sizing in contracts plus leverage. Returns None if the trade can't be sized safely."""
    mult = float(spec["multiplier"])
    dist = abs(price - stop)
    if dist <= 0:
        return None
    sl_pct = dist / price
    risk_usdt = equity * RISK_PCT
    contracts = math.floor(risk_usdt / (dist * mult))
    if contracts < spec["min_contracts"]:
        min_risk = spec["min_contracts"] * dist * mult
        if min_risk > equity * MAX_MIN_SIZE_RISK_PCT:
            return None
        contracts = spec["min_contracts"]
    notional = contracts * mult * price
    lev_cap = max(1, min(spec["max_leverage"], MAX_LEVERAGE, math.floor(0.4 / sl_pct)))
    leverage = max(1, min(lev_cap, math.ceil(notional / (MARGIN_PER_TRADE_PCT * equity))))
    margin = notional / leverage
    if margin > MARGIN_PER_TRADE_PCT * equity * 1.5:
        contracts = math.floor(MARGIN_PER_TRADE_PCT * equity * leverage / (mult * price))
        if contracts < spec["min_contracts"]:
            return None
        notional = contracts * mult * price
        margin = notional / leverage
    return {
        "contracts": contracts,
        "leverage": leverage,
        "notional": round(notional, 2),
        "margin": round(margin, 2),
        "risk_usdt": round(contracts * mult * dist, 2),
        "sl_pct": round(sl_pct * 100, 2),
    }


def tp_plan(sig: dict, contracts: int, min_contracts: int) -> list:
    """Split contracts over the signal's TP1..TP5; fold undersized slices into the previous one."""
    levels = sorted((l for l in sig.get("levels", []) if str(l.get("label", "")).startswith("TP")),
                    key=lambda l: l.get("sequence", int(l["label"][2:])))
    plan, left = [], contracts
    for i, lvl in enumerate(levels[:5]):
        qty = left if i == len(levels[:5]) - 1 else int(contracts * TP_SPLIT[i])
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
