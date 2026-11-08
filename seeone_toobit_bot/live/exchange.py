"""Pick the futures venue. The bot speaks Toobit-shaped symbols and order fields either way.

EXCHANGE=toobit (default) keeps the live account on Toobit.
EXCHANGE=gate uses Gate.io USDT-M perpetuals and needs GATE_API_KEY plus GATE_API_SECRET.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

from .toobit import Toobit, ToobitError

ROOT = Path(__file__).resolve().parent.parent
_ENV_LOADED = False


def exchange_name() -> str:
    global _ENV_LOADED
    if not _ENV_LOADED:
        load_dotenv(ROOT / ".env")
        _ENV_LOADED = True
    raw = os.environ.get("EXCHANGE", "toobit").strip().lower()
    if raw in ("gate", "gateio", "gate.io"):
        return "gate"
    return "toobit"


def connect():
    if exchange_name() == "gate":
        key = os.environ.get("GATE_API_KEY") or ""
        secret = os.environ.get("GATE_API_SECRET") or ""
        if not key or not secret:
            raise ToobitError("EXCHANGE=gate 但沒有 GATE_API_KEY / GATE_API_SECRET")
        from .gate import Gate
        return Gate(key, secret)
    return Toobit(os.environ["TOOBIT_API_KEY"], os.environ["TOOBIT_API_SECRET"])
