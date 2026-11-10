"""Order-book and large-order readings stored next to each self-scan review, for later study only.

Nothing here gates an entry. See One's large-orders / order-wall / depth-delta / recent-taker-flow
readings are snapshots with no history, and only appear for names a recipe evaluated that cycle,
so each review also takes its own Gate order-book snapshot. Taker flow is not stored: Gate keeps
historical trades, so it can be rebuilt for any review afterwards.
"""
import calendar
import time

import requests

GATE = "https://api.gateio.ws/api/v4/futures/usdt"
WALL_USD = 100_000          # same threshold as See One large-orders
BAND = 0.01                 # same ±1% band as See One depth-delta
SEEONE_METRICS = ("large-orders", "order-wall", "depth-delta", "recent-taker-flow")
SEEONE_MAX_AGE_S = 900

_mult: dict = {}
_mult_at = 0.0


def _multiplier(contract: str) -> float | None:
    global _mult, _mult_at
    if time.time() - _mult_at > 3600:
        rows = requests.get(f"{GATE}/contracts", timeout=10).json()
        _mult = {c["name"]: float(c["quanto_multiplier"]) for c in rows}
        _mult_at = time.time()
    return _mult.get(contract)


def _imbalance(bid: float, ask: float) -> float | None:
    return round((bid - ask) / (bid + ask), 4) if bid + ask else None


def book(symbol: str) -> dict:
    """symbol like ENAUSDT. Imbalances are (bid - ask) / total, so +1 is all bids."""
    contract = symbol[:-4] + "_USDT"
    m = _multiplier(contract)
    if not m:
        return {"book": "not on Gate"}
    ob = requests.get(f"{GATE}/order_book", params={"contract": contract, "limit": 300}, timeout=5).json()
    bids = [(float(x["p"]), float(x["s"]) * m * float(x["p"])) for x in ob.get("bids") or []]
    asks = [(float(x["p"]), float(x["s"]) * m * float(x["p"])) for x in ob.get("asks") or []]
    if not bids or not asks:
        return {"book": "empty"}
    mid = (bids[0][0] + asks[0][0]) / 2
    b1 = sum(v for p, v in bids if p >= mid * (1 - BAND))
    a1 = sum(v for p, v in asks if p <= mid * (1 + BAND))
    bw = [(p, v) for p, v in bids if v >= WALL_USD]
    aw = [(p, v) for p, v in asks if v >= WALL_USD]
    return {
        "depth1_bid": round(b1), "depth1_ask": round(a1), "depth1_delta": _imbalance(b1, a1),
        "wall_bid": round(sum(v for _, v in bw)), "wall_ask": round(sum(v for _, v in aw)),
        "wall_delta": _imbalance(sum(v for _, v in bw), sum(v for _, v in aw)),
        "wall_bid_px": bw[0][0] if bw else None, "wall_ask_px": aw[0][0] if aw else None,
        "spread_pct": round((asks[0][0] - bids[0][0]) / mid * 100, 4),
    }


def seeone(recipes: list, symbol: str) -> dict:
    base = symbol[:-4].upper()
    out = {}
    for r in recipes:
        for row in (r.get("evaluation") or {}).get("results") or []:
            if str(row.get("asset") or "").upper() != base:
                continue
            for k, v in (row.get("readings") or {}).items():
                if k not in SEEONE_METRICS or k in out or v.get("score") is None:
                    continue
                try:
                    age = time.time() - calendar.timegm(time.strptime(v["at"][:19], "%Y-%m-%dT%H:%M:%S"))
                except (KeyError, ValueError):
                    age = None
                if age is not None and age > SEEONE_MAX_AGE_S:
                    continue
                out[k] = {"raw": v.get("raw"), "score": v.get("score"), "state": v.get("state")}
    return out
