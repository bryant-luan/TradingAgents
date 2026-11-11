"""Gate.io USDT-M perpetual client (APIv4).

https://www.gate.com/docs/developers/apiv4/en/

The rest of the bot speaks Toobit symbols (BTC-SWAP-USDT) and order fields.
This client translates those to Gate contracts (BTC_USDT) and back.
"""
import hashlib
import hmac
import threading
import time
import uuid
from decimal import Decimal, ROUND_DOWN

import requests

from .toobit import PUB, ToobitError

HOST = "https://api.gateio.ws"
PREFIX = "/api/v4"
# Gate rejects a custom id that is not t- plus at most 28 of [0-9A-Za-z_.-].
_EMPTY_HASH = hashlib.sha512(b"").hexdigest()


class GateError(ToobitError):
    pass


def to_gate(symbol: str) -> str:
    if symbol.endswith("-SWAP-USDT"):
        return symbol[: -len("-SWAP-USDT")] + "_USDT"
    if symbol.endswith("USDT") and "_" not in symbol and "-" not in symbol:
        return symbol[:-4] + "_USDT"
    return symbol


def from_gate(contract: str) -> str:
    if contract.endswith("_USDT") and contract.count("_") == 1:
        return contract[:-5] + "-SWAP-USDT"
    return contract


def sign(secret: str, method: str, url_path: str, query: str, payload: str, timestamp: str) -> str:
    """APIv4 signature: HMAC-SHA512 of method, path, query, SHA512(body), timestamp."""
    hashed = hashlib.sha512(payload.encode()).hexdigest() if payload else _EMPTY_HASH
    msg = f"{method}\n{url_path}\n{query}\n{hashed}\n{timestamp}"
    return hmac.new(secret.encode(), msg.encode(), hashlib.sha512).hexdigest()


def _oid(row: dict) -> str:
    if row.get("id_string"):
        return str(row["id_string"])
    return str(row.get("id") or row.get("order_id") or "")


def _client_id(text: str | None) -> str:
    text = str(text or "")
    return text[2:] if text.startswith("t-") else text


def _order_status(row: dict) -> str:
    if row.get("status") == "open":
        return "NEW"
    size = abs(int(float(row.get("size") or 0)))
    left = abs(int(float(row.get("left") or 0)))
    filled = size - left
    if filled > 0 and left == 0:
        return "FILLED"
    if filled > 0:
        return "PARTIALLY_CANCELED"
    return "REJECTED"


def _order_side(row: dict) -> str:
    size = int(float(row.get("size") or 0))
    closing = bool(row.get("is_reduce_only") or row.get("reduce_only") or row.get("is_close") or row.get("close"))
    if size >= 0:
        return "BUY_CLOSE" if closing else "BUY_OPEN"
    return "SELL_CLOSE" if closing else "SELL_OPEN"


def _norm_order(row: dict) -> dict:
    size = abs(int(float(row.get("size") or 0)))
    left = abs(int(float(row.get("left") or 0)))
    return {
        "orderId": _oid(row),
        "clientOrderId": _client_id(row.get("text")),
        "status": _order_status(row),
        "executedQty": size - left,
        "side": _order_side(row),
        "price": row.get("price"),
    }


def _norm_position(row: dict) -> dict | None:
    mode = str(row.get("mode") or "")
    size = int(float(row.get("size") or 0))
    if mode == "dual_long":
        side, qty = "LONG", abs(size)
    elif mode == "dual_short":
        side, qty = "SHORT", abs(size)
    elif size > 0:
        side, qty = "LONG", size
    elif size < 0:
        side, qty = "SHORT", abs(size)
    else:
        return None
    if qty <= 0:
        return None
    lev = row.get("leverage")
    if lev in (None, "", "0", 0):
        lev = row.get("cross_leverage_limit") or 1
    return {
        "symbol": from_gate(str(row.get("contract") or "")),
        "side": side,
        "position": str(qty),
        "avgPrice": row.get("entry_price"),
        "leverage": lev,
        "margin": row.get("margin") or 0,
        "unrealizedPnL": row.get("unrealised_pnl") or 0,
        "markPrice": row.get("mark_price"),
        "lastPrice": row.get("mark_price"),
    }


def _bars(rows: list, multiplier: Decimal) -> list:
    out = []
    for row in rows:
        try:
            close = Decimal(str(row["c"]))
            volume = Decimal(str(row.get("v") or 0)) * multiplier
            if close > 0 and row.get("sum") not in (None, ""):
                volume = Decimal(str(row["sum"])) / close
            opened = int(float(row["t"]))
            if opened < 10_000_000_000:
                opened *= 1000
            out.append([opened, row["o"], row["h"], row["l"], row["c"], format(volume, "f")])
        except (KeyError, TypeError, ValueError, ArithmeticError):
            continue
    return out


_MULT: dict[str, Decimal] = {}


def _multiplier(contract: str) -> Decimal:
    if contract not in _MULT:
        row = PUB.get(f"{HOST}{PREFIX}/futures/usdt/contracts/{contract}", timeout=15).json()
        _MULT[contract] = Decimal(str(row.get("quanto_multiplier") or "1"))
    return _MULT[contract]


def public_klines(symbol: str, interval: str, limit: int = 100, end_ms: int | None = None) -> list:
    contract = to_gate(symbol)
    params = {"contract": contract, "interval": interval, "limit": min(int(limit), 2000)}
    if end_ms:
        params["to"] = int(end_ms) // 1000
    try:
        rows = PUB.get(f"{HOST}{PREFIX}/futures/usdt/candlesticks", params=params, timeout=15).json()
    except (requests.RequestException, ValueError):
        return []
    if not isinstance(rows, list):
        return []
    try:
        mult = _multiplier(contract)
    except (requests.RequestException, ValueError, KeyError):
        mult = Decimal(1)
    return _bars(rows, mult)


def public_names() -> set[str]:
    """Tradable USDT-M symbols in the bot's BTC-SWAP-USDT form."""
    try:
        rows = PUB.get(f"{HOST}{PREFIX}/futures/usdt/contracts", timeout=20).json()
    except (requests.RequestException, ValueError):
        return set()
    out = set()
    if not isinstance(rows, list):
        return out
    for contract in rows:
        name = contract.get("name") or ""
        if contract.get("status") != "trading" or contract.get("in_delisting"):
            continue
        if name.endswith("_USDT") and name.count("_") == 1:
            out.add(from_gate(name))
    return out


def public_quote_volumes() -> dict[str, float]:
    try:
        rows = PUB.get(f"{HOST}{PREFIX}/futures/usdt/tickers", timeout=20).json()
    except (requests.RequestException, ValueError):
        return {}
    out = {}
    if not isinstance(rows, list):
        return out
    for row in rows:
        contract = row.get("contract")
        if not contract:
            continue
        try:
            out[from_gate(contract)] = float(row.get("volume_24h_quote") or row.get("volume_24h_settle") or 0)
        except (TypeError, ValueError):
            continue
    return out


class Gate:
    """Same methods as Toobit, signed against Gate APIv4 USDT futures."""

    name = "gate"

    def __init__(self, key: str, secret: str):
        self.key = key
        self.secret = secret
        self.http = requests.Session()
        self._specs = None
        self._hedge = None
        self._lock = threading.Lock()

    def _request(self, method: str, path: str, params: dict | None = None, body: dict | None = None):
        # Only reads are retried; a resent order could fill twice.
        tries = 3 if method == "GET" else 1
        for attempt in range(tries):
            try:
                with self._lock:
                    return self._request_locked(method, path, params, body)
            except (requests.Timeout, requests.ConnectionError):
                if attempt == tries - 1:
                    raise
                time.sleep(1 + attempt)

    def _request_locked(self, method: str, path: str, params: dict | None, body: dict | None):
        import json
        query = "&".join(f"{k}={v}" for k, v in (params or {}).items() if v is not None)
        payload = "" if body is None else json.dumps(body, separators=(",", ":"))
        url_path = PREFIX + path
        timestamp = str(int(time.time()))
        headers = {
            "KEY": self.key,
            "Timestamp": timestamp,
            "SIGN": sign(self.secret, method, url_path, query, payload, timestamp),
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        url = f"{HOST}{url_path}"
        if query:
            url = f"{url}?{query}"
        response = self.http.request(method, url, data=payload or None, headers=headers, timeout=15)
        if response.status_code == 204 or not response.text:
            if response.status_code not in (200, 201, 204):
                raise GateError(f"{method} {path} -> HTTP {response.status_code}")
            return {}
        try:
            data = response.json()
        except ValueError:
            raise GateError(f"{method} {path} -> HTTP {response.status_code}: {response.text[:300]}")
        if response.status_code not in (200, 201):
            raise GateError(f"{method} {path} -> {response.status_code} {data}")
        return data

    def _hedged(self) -> bool:
        if self._hedge is None:
            account = self._request("GET", "/futures/usdt/accounts")
            mode = str(account.get("position_mode") or "")
            if mode:
                self._hedge = mode != "single"
            else:
                self._hedge = bool(account.get("in_dual_mode"))
        return self._hedge

    def _text(self, tag: str) -> str:
        raw = "".join(ch for ch in f"{tag}{uuid.uuid4().hex}" if ch.isalnum() or ch in "_.-")
        return "t-" + raw[:28]

    # ---- market data ----
    def specs(self) -> dict:
        if self._specs is None:
            rows = PUB.get(f"{HOST}{PREFIX}/futures/usdt/contracts", timeout=20).json()
            out = {}
            for contract in rows:
                name = contract.get("name") or ""
                if contract.get("status") != "trading" or contract.get("in_delisting"):
                    continue
                if not name.endswith("_USDT") or name.count("_") != 1:
                    continue
                mult = Decimal(str(contract.get("quanto_multiplier") or "1"))
                _MULT[name] = mult
                out[from_gate(name)] = {
                    "multiplier": mult,
                    "min_contracts": int(contract.get("order_size_min") or 1),
                    "tick": Decimal(str(contract.get("order_price_round") or "0.00000001")),
                    "max_leverage": int(float(contract.get("leverage_max") or 20)),
                    "mmr_tiers": ([(float("inf"), float(contract["maintenance_rate"]),
                                    int(float(contract.get("leverage_max") or 20)))]
                                  if contract.get("maintenance_rate") else []),
                    "seeone": name[:-5] + "USDT",
                    "rwa": False,
                    "categories": [],
                    "quote": "USDT",
                }
            self._specs = out
        return self._specs

    def price(self, symbol: str) -> float:
        rows = PUB.get(f"{HOST}{PREFIX}/futures/usdt/tickers",
                            params={"contract": to_gate(symbol)}, timeout=10).json()
        return float(rows[0]["last"])

    def book(self, symbol: str) -> tuple[float, float]:
        row = PUB.get(f"{HOST}{PREFIX}/futures/usdt/order_book",
                           params={"contract": to_gate(symbol), "limit": 1}, timeout=10).json()
        return float(row["bids"][0]["p"]), float(row["asks"][0]["p"])

    def fmt_price(self, symbol: str, price: float) -> str:
        tick = self.specs()[symbol]["tick"]
        return str((Decimal(str(price)) / tick).quantize(Decimal(1), rounding=ROUND_DOWN) * tick)

    # ---- account ----
    def balance(self) -> dict:
        account = self._request("GET", "/futures/usdt/accounts")
        # `total` is the wallet and excludes unrealised PnL, same split the bot already uses.
        return {
            "asset": "USDT",
            "balance": account.get("total") or 0,
            "availableBalance": account.get("available") or 0,
            "crossUnRealizedPnl": account.get("unrealised_pnl") or 0,
            "isolatedUnRealizedPnl": 0,
        }

    def positions(self, symbol: str | None = None) -> list:
        if symbol:
            contract = to_gate(symbol)
            if self._hedged():
                rows = self._request("GET", f"/futures/usdt/dual_comp/positions/{contract}")
            else:
                row = self._request("GET", f"/futures/usdt/positions/{contract}")
                rows = row if isinstance(row, list) else [row]
        else:
            rows = self._request("GET", "/futures/usdt/positions")
        out = []
        for row in rows:
            pos = _norm_position(row)
            if pos and (symbol is None or pos["symbol"] == symbol):
                out.append(pos)
        return out

    def history_positions(self, symbol: str, limit: int = 5) -> list:
        rows = self._request("GET", "/futures/usdt/position_close",
                             {"contract": to_gate(symbol), "limit": min(limit, 100)})
        out = []
        for row in rows:
            side = "LONG" if row.get("side") == "long" else "SHORT"
            when = float(row.get("time") or 0)
            if when < 10_000_000_000:
                when *= 1000
            close_px = row.get("short_price") if side == "LONG" else row.get("long_price")
            out.append({"side": side, "closeTime": int(when), "realizedPnL": row.get("pnl") or 0,
                        "closeAvgPrice": close_px})
        return out

    def set_isolated(self, symbol: str):
        path = "/futures/usdt/dual_comp/positions/cross_mode" if self._hedged() else "/futures/usdt/positions/cross_mode"
        try:
            self._request("POST", path, body={"mode": "ISOLATED", "contract": to_gate(symbol)})
        except GateError:
            pass

    def set_leverage(self, symbol: str, leverage: int):
        contract = to_gate(symbol)
        path = (f"/futures/usdt/dual_comp/positions/{contract}/leverage" if self._hedged()
                else f"/futures/usdt/positions/{contract}/leverage")
        return self._request("POST", path, {"leverage": int(leverage)})

    # ---- orders ----
    def _place(self, symbol: str, side: str, contracts: int, price: str, tif: str, tag: str) -> dict:
        closing = side.endswith("_CLOSE") or side.endswith("CLOSE")
        size = int(contracts) if side.startswith("BUY") else -int(contracts)
        body = {"contract": to_gate(symbol), "size": size, "price": price, "tif": tif, "text": self._text(tag)}
        if closing:
            body["reduce_only"] = True
        return _norm_order(self._request("POST", "/futures/usdt/orders", body=body))

    def market_open(self, symbol: str, direction: str, contracts: int, stop_loss: str) -> dict:
        side = "BUY_OPEN" if direction == "LONG" else "SELL_OPEN"
        order = self.market(symbol, side, contracts)
        self.set_position_stop(symbol, direction, stop_loss)
        return order

    def post_only(self, symbol: str, side: str, contracts: int, price: str, tag: str) -> dict:
        return self._place(symbol, side, contracts, price, "poc", tag)

    def market(self, symbol: str, side: str, contracts: int) -> dict:
        return self._place(symbol, side, contracts, "0", "ioc", "mk")

    def close_trigger(self, symbol: str, direction: str, contracts: int, trigger: str, tag: str) -> dict:
        """Partial take-profit. Long closes by selling, short closes by buying."""
        if direction == "LONG":
            size, order_type, rule = -int(contracts), "plan-close-long-position", 1
        else:
            size, order_type, rule = int(contracts), "plan-close-short-position", 2
        body = {
            "initial": {
                "contract": to_gate(symbol),
                "size": size,
                "price": "0",
                "tif": "ioc",
                "close": False,
                "reduce_only": True,
                "text": self._text(tag),
            },
            "trigger": {
                "strategy_type": 0,
                "price_type": 0,
                "price": str(trigger),
                "rule": rule,
                "expiration": 30 * 86400,
            },
            "order_type": order_type,
        }
        row = self._request("POST", "/futures/usdt/price_orders", body=body)
        return {"orderId": _oid(row)}

    def set_position_stop(self, symbol: str, direction: str, stop_loss: str):
        for order in self.open_orders(symbol, "STOP_PROFIT_LOSS"):
            if order.get("orderType") == f"STOP_{direction}_LOSS":
                self.cancel_order(order["orderId"], "STOP_PROFIT_LOSS")
        hedge = self._hedged()
        if direction == "LONG":
            order_type, rule, auto_size = "close-long-position", 2, "close_long"
        else:
            order_type, rule, auto_size = "close-short-position", 1, "close_short"
        initial = {
            "contract": to_gate(symbol),
            "size": 0,
            "price": "0",
            "tif": "ioc",
            "close": not hedge,
            "reduce_only": True,
            "text": f"t-STOP_{direction}_LOSS",
        }
        if hedge:
            initial["auto_size"] = auto_size
        body = {
            "initial": initial,
            "trigger": {
                "strategy_type": 0,
                "price_type": 0,
                "price": str(stop_loss),
                "rule": rule,
                "expiration": 30 * 86400,
            },
            "order_type": order_type,
        }
        return self._request("POST", "/futures/usdt/price_orders", body=body)

    def open_orders(self, symbol: str, kind: str = "STOP") -> list:
        contract = to_gate(symbol)
        if kind == "LIMIT":
            rows = self._request("GET", "/futures/usdt/orders", {"contract": contract, "status": "open"})
            return [_norm_order(row) for row in rows]
        rows = self._request("GET", "/futures/usdt/price_orders", {"contract": contract, "status": "open"})
        out = []
        for row in rows:
            order_type = str(row.get("order_type") or "")
            initial = row.get("initial") or {}
            text = _client_id(initial.get("text"))
            full_stop = order_type in ("close-long-position", "close-short-position") or "STOP_" in text
            if kind == "STOP_PROFIT_LOSS" and not full_stop:
                continue
            if kind == "STOP" and full_stop:
                continue
            if "long" in order_type:
                direction, close_side = "LONG", "SELL_CLOSE"
            elif "short" in order_type:
                direction, close_side = "SHORT", "BUY_CLOSE"
            else:
                direction = "LONG" if "LONG" in text else "SHORT"
                close_side = "SELL_CLOSE" if direction == "LONG" else "BUY_CLOSE"
            trigger = (row.get("trigger") or {}).get("price")
            out.append({
                "orderId": _oid(row),
                # In hedge mode Gate stores the full-position stop's text as "api", so the label is derived.
                "clientOrderId": f"STOP_{direction}_LOSS" if full_stop else (text or f"STOP_{direction}_LOSS"),
                "orderType": f"STOP_{direction}_LOSS" if full_stop else order_type,
                "stopPrice": trigger,
                "side": close_side,
                "status": "NEW",
            })
        return out

    def cancel_order(self, order_id: str, kind: str = "STOP"):
        path = "/futures/usdt/orders" if kind == "LIMIT" else "/futures/usdt/price_orders"
        return self._request("DELETE", f"{path}/{order_id}")

    def order(self, order_id: str) -> dict:
        return _norm_order(self._request("GET", f"/futures/usdt/orders/{order_id}"))

    def user_trades(self, symbol: str, limit: int = 100) -> list:
        rows = self._request("GET", "/futures/usdt/my_trades",
                             {"contract": to_gate(symbol), "limit": min(limit, 1000)})
        out = []
        for row in rows:
            size = int(float(row.get("size") or 0))
            closed = int(float(row.get("close_size") or 0)) != 0
            if size >= 0:
                side = "BUY_CLOSE" if closed else "BUY_OPEN"
            else:
                side = "SELL_CLOSE" if closed else "SELL_OPEN"
            when = float(row.get("create_time") or 0)
            if when < 10_000_000_000:
                when *= 1000
            out.append({
                "time": int(when),
                "commission": row.get("fee") or 0,
                "isMaker": str(row.get("role") or "") == "maker",
                "side": side,
                "orderId": str(row.get("order_id") or ""),
            })
        return out

    def flash_close(self, symbol: str, direction: str):
        held = next((p for p in self.positions(symbol) if p["side"] == direction), None)
        if not held or int(float(held["position"])) <= 0:
            return {}
        side = "SELL_CLOSE" if direction == "LONG" else "BUY_CLOSE"
        return self.market(symbol, side, int(float(held["position"])))
