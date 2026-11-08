import hashlib
import hmac
import threading
import time
import uuid
from decimal import Decimal, ROUND_DOWN
from urllib.parse import urlencode

import requests

BASE = "https://api.toobit.com"


class ToobitError(Exception):
    pass


def public_names() -> set[str]:
    """Tradable USDT-M symbols, for comparing the Gate list without signing in."""
    try:
        info = requests.get(f"{BASE}/api/v1/exchangeInfo", timeout=20).json()
    except (requests.RequestException, ValueError):
        return set()
    return {c["symbol"] for c in info.get("contracts", []) if c.get("status") == "TRADING"}


class Toobit:
    """Toobit USDT-M perpetual client (v1 REST, hedge mode). Quantities are in contracts."""

    name = "toobit"

    def __init__(self, key: str, secret: str):
        self.secret = secret.encode()
        self.http = requests.Session()
        self.http.headers["X-BB-APIKEY"] = key
        self._specs = None
        self._lock = threading.Lock()

    def _request(self, method: str, path: str, params: dict | None = None):
        with self._lock:
            return self._request_locked(method, path, params)

    def _request_locked(self, method: str, path: str, params: dict | None = None):
        params = {k: v for k, v in (params or {}).items() if v is not None}
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = 10000
        qs = urlencode(params)
        sig = hmac.new(self.secret, qs.encode(), hashlib.sha256).hexdigest()
        r = self.http.request(method, f"{BASE}{path}?{qs}&signature={sig}", timeout=15)
        try:
            data = r.json()
        except ValueError:
            raise ToobitError(f"{method} {path} -> HTTP {r.status_code}: {r.text[:300]}")
        if r.status_code != 200 or (isinstance(data, dict) and data.get("code") not in (None, 200)):
            raise ToobitError(f"{method} {path} -> {r.status_code} {data}")
        return data

    # ---- market data ----
    def specs(self) -> dict:
        if self._specs is None:
            info = requests.get(f"{BASE}/api/v1/exchangeInfo", timeout=20).json()
            out = {}
            for c in info.get("contracts", []):
                if c.get("status") != "TRADING":
                    continue
                f = {x["filterType"]: x for x in c["filters"]}
                mult = Decimal(c["contractMultiplier"])
                sym = c["symbol"]
                seeone = sym[: -len("-SWAP-USDT")] + "USDT" if sym.endswith("-SWAP-USDT") else sym
                cats = list(c.get("categories") or [])
                out[sym] = {
                    "multiplier": mult,
                    "min_contracts": int(Decimal(f["LOT_SIZE"]["minQty"]) / mult),
                    "tick": Decimal(f["PRICE_FILTER"]["tickSize"]),
                    "max_leverage": int(float(c["riskLimits"][0]["maxLeverage"])) if c.get("riskLimits") else 20,
                    "seeone": seeone,
                    "rwa": bool(c.get("isRwa")) or "TradFi" in cats,
                    "categories": cats,
                    "quote": c.get("quoteAsset") or "",
                }
            self._specs = out
        return self._specs

    def price(self, symbol: str) -> float:
        r = requests.get(f"{BASE}/quote/v1/ticker/price", params={"symbol": symbol}, timeout=10).json()
        return float(r[0]["p"])

    def book(self, symbol: str) -> tuple[float, float]:
        r = requests.get(f"{BASE}/quote/v1/ticker/bookTicker", params={"symbol": symbol}, timeout=10).json()
        return float(r[0]["b"]), float(r[0]["a"])

    def fee_rates(self, symbol: str) -> dict:
        r = self._request("GET", "/api/v1/futures/commissionRate", {"symbol": symbol})
        return {"maker": float(r["openMakerFee"]), "taker": float(r["openTakerFee"])}

    def user_trades(self, symbol: str, limit: int = 100) -> list:
        return self._request("GET", "/api/v1/futures/userTrades", {"symbol": symbol, "limit": limit})

    def order(self, order_id: str) -> dict:
        return self._request("GET", "/api/v1/futures/order", {"orderId": order_id})

    def fmt_price(self, symbol: str, price: float) -> str:
        tick = self.specs()[symbol]["tick"]
        return str((Decimal(str(price)) / tick).quantize(Decimal(1), rounding=ROUND_DOWN) * tick)

    # ---- account ----
    def balance(self) -> dict:
        return next(r for r in self._request("GET", "/api/v1/futures/balance") if r["asset"] == "USDT")

    def positions(self, symbol: str | None = None) -> list:
        rows = self._request("GET", "/api/v1/futures/positions", {"symbol": symbol})
        return [p for p in rows if float(p.get("position") or 0) > 0]

    def history_positions(self, symbol: str, limit: int = 5) -> list:
        return self._request("GET", "/api/v1/futures/historyPositions", {"symbol": symbol, "limit": limit})

    def set_isolated(self, symbol: str):
        try:
            self._request("POST", "/api/v1/futures/marginType", {"symbol": symbol, "marginType": "ISOLATED"})
        except ToobitError:
            pass  # already isolated, or a position exists

    def set_leverage(self, symbol: str, leverage: int):
        return self._request("POST", "/api/v1/futures/leverage", {"symbol": symbol, "leverage": leverage})

    # ---- orders ----
    def market_open(self, symbol: str, direction: str, contracts: int, stop_loss: str) -> dict:
        return self._request("POST", "/api/v1/futures/order", {
            "symbol": symbol,
            "side": "BUY_OPEN" if direction == "LONG" else "SELL_OPEN",
            "type": "LIMIT",
            "priceType": "MARKET",
            "quantity": contracts,
            "newClientOrderId": f"so{uuid.uuid4().hex[:18]}",
            "stopLoss": stop_loss,
            "slTriggerBy": "CONTRACT_PRICE",
            "slOrderType": "MARKET",
        })

    def post_only(self, symbol: str, side: str, contracts: int, price: str, tag: str) -> dict:
        """Maker-only limit order (rejected instead of filling as taker). side: BUY_OPEN/SELL_OPEN/BUY_CLOSE/SELL_CLOSE."""
        return self._request("POST", "/api/v1/futures/order", {
            "symbol": symbol,
            "side": side,
            "type": "LIMIT",
            "price": price,
            "quantity": contracts,
            "timeInForce": "LIMIT_MAKER",
            "newClientOrderId": f"{tag}{uuid.uuid4().hex[:14]}",
        })

    def market(self, symbol: str, side: str, contracts: int) -> dict:
        return self._request("POST", "/api/v1/futures/order", {
            "symbol": symbol, "side": side, "type": "LIMIT", "priceType": "MARKET",
            "quantity": contracts, "newClientOrderId": f"mk{uuid.uuid4().hex[:16]}",
        })

    def close_trigger(self, symbol: str, direction: str, contracts: int, trigger: str, tag: str) -> dict:
        """Conditional market close of part of a position (used for TP1-TP5)."""
        return self._request("POST", "/api/v1/futures/order", {
            "symbol": symbol,
            "side": "SELL_CLOSE" if direction == "LONG" else "BUY_CLOSE",
            "type": "STOP",
            "priceType": "MARKET",
            "stopPrice": trigger,
            "quantity": contracts,
            "newClientOrderId": f"{tag}{uuid.uuid4().hex[:14]}",
        })

    def set_position_stop(self, symbol: str, direction: str, stop_loss: str):
        return self._request("POST", "/api/v1/futures/position/trading-stop",
                             {"symbol": symbol, "side": direction, "stopLoss": stop_loss,
                              "slTriggerBy": "CONTRACT_PRICE"})

    def open_orders(self, symbol: str, kind: str = "STOP") -> list:
        return self._request("GET", "/api/v1/futures/openOrders", {"symbol": symbol, "type": kind, "limit": 200})

    def cancel_order(self, order_id: str, kind: str = "STOP"):
        return self._request("DELETE", "/api/v1/futures/order", {"orderId": order_id, "type": kind})

    def flash_close(self, symbol: str, direction: str):
        return self._request("POST", "/api/v1/futures/flashClose", {"symbol": symbol, "side": direction})
