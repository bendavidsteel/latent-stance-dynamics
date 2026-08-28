#!/bin/bash
# Kill sweep trials that exceed a wall-clock budget.
#
# Some configurations drive the SDE solver into batches that take minutes each,
# which stalls the agent indefinitely; killing the trial lets it move on.
#
# Trials already running when the watchdog starts are left alone: they predate
# the budget, and one of them may be a long run that is still making progress.
#
#   bash watchdog.sh [limit_seconds]
LIMIT=${1:-6000}
PATTERN="flows/nn_potential.py"

GRANDFATHERED=" $(pgrep -f "$PATTERN" | tr '\n' ' ')"
echo "$(date '+%F %T') watchdog up, limit ${LIMIT}s, ignoring pre-existing pids:${GRANDFATHERED:-' none'}"

while true; do
  for p in $(pgrep -f "$PATTERN"); do
    case "$GRANDFATHERED" in *" $p "*) continue ;; esac
    e=$(ps -o etimes= -p "$p" 2>/dev/null | tr -d ' ')
    if [ -n "$e" ] && [ "$e" -gt "$LIMIT" ]; then
      echo "$(date '+%F %T') killing pid $p after ${e}s (limit ${LIMIT}s)"
      kill -9 "$p"
    fi
  done
  sleep 120
done
