#!/usr/bin/env bash
# Re-run the indicator research every 30 minutes (results: logs/research_latest.txt, logs/research_history.jsonl).
cd "$(dirname "$0")/.."
while [ ! -f logs/STOP_RESEARCH ]; do
  nice -n 10 python3 -m live.research > /dev/null 2>> logs/research_errors.log
  sleep 1800
done
