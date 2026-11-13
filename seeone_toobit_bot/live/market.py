"""Public candles and 24h quote volume for the active venue.

Callers keep Toobit symbols (BTC-SWAP-USDT) and Toobit bar rows
[open_ms, open, high, low, close, base_volume].
"""
import time

import requests

from .exchange import exchange_name
from .toobit import PUB

_LISTED: dict[str, tuple[float, set[str]]] = {}

TOOBIT = "https://api.toobit.com"


def listed_names(venue: str) -> set[str]:
    """Tradable symbols on one venue. Cached so a scan does not refetch the whole list."""
    hit = _LISTED.get(venue)
    if hit and time.time() - hit[0] < 1800:
        return hit[1]
    if venue == "gate":
        from .gate import public_names
    else:
        from .toobit import public_names
    names = public_names()
    _LISTED[venue] = (time.time(), names)
    return names


def listing(tsym: str, venue: str | None = None) -> str:
    """both: the other exchange lists it too. only: this exchange alone. The rules stay the same either way."""
    venue = venue or exchange_name()
    other = "toobit" if venue == "gate" else "gate"
    theirs = listed_names(other)
    if not theirs:
        return "unknown"
    return "both" if tsym in theirs else "only"


def klines(symbol: str, interval: str, limit: int = 100, end_ms: int | None = None) -> list:
    if exchange_name() == "gate":
        from .gate import public_klines
        return public_klines(symbol, interval, limit, end_ms)
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if end_ms:
        params["endTime"] = int(end_ms)
    try:
        rows = PUB.get(f"{TOOBIT}/quote/v1/klines", params=params, timeout=15).json()
    except (requests.RequestException, ValueError):
        return []
    return rows if isinstance(rows, list) else []


def quote_volumes() -> dict[str, float]:
    """24h quote volume in USDT, keyed by the Toobit-style symbol the scanners already use."""
    if exchange_name() == "gate":
        from .gate import public_quote_volumes
        return public_quote_volumes()
    try:
        rows = PUB.get(f"{TOOBIT}/quote/v1/contract/ticker/24hr", timeout=15).json()
    except (requests.RequestException, ValueError):
        return {}
    out = {}
    if not isinstance(rows, list):
        return out
    for row in rows:
        sym = row.get("s")
        if not sym:
            continue
        try:
            out[sym] = float(row.get("qv") or 0)
        except (TypeError, ValueError):
            continue
    return out
