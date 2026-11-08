#!/usr/bin/env bash
# Safety monitor for the Gate bot only. Create logs/gate/STOP_MONITOR to stop it.
cd "$(dirname "$0")/.."
mkdir -p logs/gate
rm -f logs/gate/STOP_MONITOR
export EXCHANGE=gate
while [ ! -f logs/gate/STOP_MONITOR ]; do
  python3 -u -m live.monitor >> logs/gate/monitor_console.log 2>&1
  echo "$(date -u +%FT%TZ) gate monitor exited, restarting in 10s" >> logs/gate/monitor_console.log
  sleep 10
done
