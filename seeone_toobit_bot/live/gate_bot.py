"""Second bot, Gate.io only. Entry and exit rules are the Toobit bot's rules.

Gate reads seeone_toobit_bot/.env.gate. See One login still comes from .env.
Its book, pause file, and log live under logs/gate, so it does not manage Toobit positions.
The learned thresholds and review calibration are copied once from the Toobit bot, then kept separately.
"""
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
GATE_ENV = ROOT / ".env.gate"


def load_gate_env():
    """Load the Gate file over the shared See One login. Do not write key values."""
    if not GATE_ENV.exists():
        GATE_ENV.write_text("EXCHANGE=gate\n")
    load_dotenv(ROOT / ".env")
    load_dotenv(GATE_ENV, override=True)
    os.environ["EXCHANGE"] = "gate"


load_gate_env()


def seed_rules():
    """Start from the same learned thresholds. Later closes update only the Gate copy."""
    base = ROOT / "logs" / "gate"
    base.mkdir(parents=True, exist_ok=True)
    for name in ("learned_params.json", "self_calib.json"):
        dest = base / name
        src = ROOT / "logs" / name
        if not dest.exists() and src.exists():
            shutil.copy(src, dest)


def main():
    seed_rules()
    from .bot import main as run
    run()


if __name__ == "__main__":
    main()
