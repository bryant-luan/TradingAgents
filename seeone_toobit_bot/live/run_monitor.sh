#!/usr/bin/env bash
# Keep the once-a-minute safety monitor running. Create logs/STOP_MONITOR to stop it.
cd "$(dirname "$0")/.."
rm -f logs/STOP_MONITOR
while [ ! -f logs/STOP_MONITOR ]; do
  python3 -u -m live.monitor >> logs/monitor_console.log 2>&1
  echo "$(date -u +%FT%TZ) monitor exited, restarting in 10s" >> logs/monitor_console.log
  sleep 10
done
