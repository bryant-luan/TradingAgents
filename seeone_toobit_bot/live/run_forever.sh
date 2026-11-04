#!/usr/bin/env bash
# Keep the live bot running: restart it after any exit (watchdog, crash). Create logs/STOP to stop for good.
cd "$(dirname "$0")/.."
rm -f logs/STOP
while [ ! -f logs/STOP ]; do
  python3 -u -m live.bot 2>&1 | tee -a logs/live_bot_console.log
  echo "$(date -u +%FT%TZ) bot exited (${PIPESTATUS[0]}), restarting in 10s" | tee -a logs/live_bot_console.log
  sleep 10
done
