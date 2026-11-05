#!/usr/bin/env bash
# Keep the live bot running: restart it after any exit (watchdog, crash). Create logs/STOP to stop for good.
# SIGTERM to this script (systemctl stop) is passed to the bot, which finishes its cycle before exiting.
cd "$(dirname "$0")/.."
rm -f logs/STOP
pid=""
on_term() {
  touch logs/STOP
  [ -n "$pid" ] && kill -TERM "$pid" 2>/dev/null
}
trap on_term TERM INT
while [ ! -f logs/STOP ]; do
  python3 -u -m live.bot > >(tee -a logs/live_bot_console.log) 2>&1 &
  pid=$!
  while true; do
    wait "$pid"; code=$?
    kill -0 "$pid" 2>/dev/null || break   # wait returns early when a signal arrives; keep waiting
  done
  pid=""
  echo "$(date -u +%FT%TZ) bot exited (${code}), restarting in 10s" | tee -a logs/live_bot_console.log
  [ -f logs/STOP ] && break
  sleep 10
done
