"""Public candles and 24h quote volume for the active venue.

Callers keep Toobit symbols (BTC-SWAP-USDT) and Toobit bar rows
[open_ms, open, high, low, close, base_volume].
"""
import requests

from .exchange import exchange_name

TOOBIT = "https://api.toobit.com"


def klines(symbol: str, interval: str, limit: int = 100, end_ms: int | None = None) -> list:
    if exchange_name() == "gate":
        from .gate import public_klines
        return public_klines(symbol, interval, limit, end_ms)
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if end_ms:
        params["endTime"] = int(end_ms)
    try:
        rows = requests.get(f"{TOOBIT}/quote/v1/klines", params=params, timeout=15).json()
    except (requests.RequestException, ValueError):
        return []
    return rows if isinstance(rows, list) else []


def quote_volumes() -> dict[str, float]:
    """24h quote volume in USDT, keyed by the Toobit-style symbol the scanners already use."""
    if exchange_name() == "gate":
        from .gate import public_quote_volumes
        return public_quote_volumes()
    try:
        rows = requests.get(f"{TOOBIT}/quote/v1/contract/ticker/24hr", timeout=15).json()
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
