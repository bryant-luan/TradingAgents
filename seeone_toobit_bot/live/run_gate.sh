#!/usr/bin/env bash
# Gate bot. Same rules as run_forever.sh, separate process and logs/gate book.
# Create logs/gate/STOP to stop for good. Does not touch the Toobit bot.
cd "$(dirname "$0")/.."
mkdir -p logs/gate
rm -f logs/gate/STOP
if ! grep -Eq '^GATE_API_KEY=.+$' .env.gate && [ -z "$GATE_API_KEY" ]; then
  echo "$(date -u +%FT%TZ) Gate bot not started: GATE_API_KEY is empty in .env.gate" | tee -a logs/gate/live_bot_console.log
  exit 1
fi
if ! grep -Eq '^GATE_API_SECRET=.+$' .env.gate && [ -z "$GATE_API_SECRET" ]; then
  echo "$(date -u +%FT%TZ) Gate bot not started: GATE_API_SECRET is empty in .env.gate" | tee -a logs/gate/live_bot_console.log
  exit 1
fi
pid=""
on_term() {
  touch logs/gate/STOP
  [ -n "$pid" ] && kill -TERM "$pid" 2>/dev/null
}
trap on_term TERM INT
while [ ! -f logs/gate/STOP ]; do
  python3 -u -m live.gate_bot >> logs/gate/live_bot_console.log 2>&1 &
  pid=$!
  while true; do
    wait "$pid"; code=$?
    kill -0 "$pid" 2>/dev/null || break
  done
  pid=""
  echo "$(date -u +%FT%TZ) gate bot exited (${code}), restarting in 10s" | tee -a logs/gate/live_bot_console.log
  [ -f logs/gate/STOP ] && break
  sleep 10
done
