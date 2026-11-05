#!/usr/bin/env bash
# Move the live bot from this machine to an always-on Linux host (Ubuntu/Debian with systemd).
#
#   ./deploy/handover.sh user@host [path/to/ssh_key]
#
# Steps: install code, .env (chmod 600) and Python deps on the host; register systemd services;
# stop the local bot and monitor gracefully (never mid-order); copy state/learned params/calibration;
# start the host's services; show its first log lines. Only one bot may ever trade the account,
# so the host starts only after the local bot has exited.
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET="${1:?usage: deploy/handover.sh user@host [ssh_key]}"
SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=15)
[ -n "${2:-}" ] && SSH_OPTS+=(-i "$2")
REMOTE_DIR=seeone_toobit_bot
ssh_() { ssh "${SSH_OPTS[@]}" "$TARGET" "$@"; }
rsync_() { rsync -az -e "ssh ${SSH_OPTS[*]}" "$@"; }

echo "== 1/6 checking host"
ssh_ 'command -v systemctl >/dev/null && command -v python3 >/dev/null' \
  || { echo "host needs systemd and python3"; exit 1; }
ssh_ "curl -s -o /dev/null -w 'toobit %{http_code}\n' https://api.toobit.com/api/v1/time; \
      curl -s -o /dev/null -w 'seeone %{http_code}\n' https://seeone.club/"

echo "== 2/6 installing code and dependencies"
ssh_ "mkdir -p $REMOTE_DIR/logs"
rsync_ --exclude logs --exclude .env --exclude .git --exclude __pycache__ --exclude .venv ./ "$TARGET:$REMOTE_DIR/"
rsync_ --chmod=F600 .env "$TARGET:$REMOTE_DIR/.env"
ssh_ "cd $REMOTE_DIR && (python3 -m venv .venv 2>/dev/null || (sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv && python3 -m venv .venv)) \
      && .venv/bin/pip install -q -r requirements.txt && command -v tmux >/dev/null || sudo apt-get install -y -qq tmux"

echo "== 3/6 registering systemd services"
ssh_ "set -e; U=\$(id -un); D=\$HOME/$REMOTE_DIR
for s in bot monitor; do
  sed -e \"s#@USER@#\$U#g\" -e \"s#@DIR@#\$D#g\" \$D/deploy/seeone-\$s.service | sudo tee /etc/systemd/system/seeone-\$s.service >/dev/null
done
sudo systemctl daemon-reload"

echo "== 4/6 stopping the local bot and monitor (graceful)"
touch logs/STOP logs/STOP_MONITOR
pkill -TERM -f "python3 -u -m live.bot" || true
for _ in $(seq 1 60); do pgrep -f "python3 -u -m live.bot" >/dev/null || break; sleep 5; done
if pgrep -f "python3 -u -m live.bot" >/dev/null; then
  echo "local bot still running after 5 minutes; aborting before the host starts"; exit 1
fi
pkill -TERM -f "python3 -u -m live.monitor" || true

echo "== 5/6 copying state, learned parameters and calibration"
rsync_ logs/ "$TARGET:$REMOTE_DIR/logs/"

echo "== 6/6 starting on the host"
ssh_ "sudo systemctl enable --now seeone-bot seeone-monitor && sleep 40 && \
      systemctl is-active seeone-bot seeone-monitor && tail -n 5 $REMOTE_DIR/logs/live_bot_console.log | cut -c1-200"
echo "handover done: the bot now runs on $TARGET (local copies stay stopped while logs/STOP exists)"
