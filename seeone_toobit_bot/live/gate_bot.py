"""Second bot, Gate.io only. Entry and exit rules are the Toobit bot's rules.

Gate reads seeone_toobit_bot/.env.gate. See One login still comes from .env.
Its book, pause file, and log live under logs/gate, so it does not manage Toobit positions.
The learned thresholds and review calibration are copied once from the Toobit bot.
Later, each book keeps its own cooldowns and score floor, and both recompute factor
weights from the two books' closed trades together.
"""
import json
import os
import shutil
from datetime import datetime, timezone
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
    raw = os.environ.get("GATE_SIM_UNTIL") or ""
    until = None
    if raw:
        try:
            until = datetime.fromisoformat(raw)
            if until.tzinfo is None:
                until = until.replace(tzinfo=timezone.utc)
        except ValueError:
            until = None
    ended = ROOT / "logs" / "gate" / "SIM_ENDED"
    if until and datetime.now(timezone.utc) < until:
        os.environ["GATE_PAPER"] = "1"
    else:
        os.environ.pop("GATE_PAPER", None)
        if until and ended.exists():
            raise SystemExit(0)


load_gate_env()


def seed_rules():
    """Start from the same learned thresholds. Loss timers on the other account stay there."""
    base = ROOT / "logs" / "gate"
    base.mkdir(parents=True, exist_ok=True)
    for name in ("learned_params.json", "self_calib.json"):
        dest = base / name
        src = ROOT / "logs" / name
        if not dest.exists() and src.exists():
            shutil.copy(src, dest)
            if name == "learned_params.json":
                # Same score gates. The other account's loss timers do not apply here.
                data = json.loads(dest.read_text())
                data["symbol_cooldown"] = {}
                data["pause_until"] = 0
                data["consec_losses"] = 0
                dest.write_text(json.dumps(data))


def main():
    seed_rules()
    from .bot import main as run
    run()


if __name__ == "__main__":
    main()
