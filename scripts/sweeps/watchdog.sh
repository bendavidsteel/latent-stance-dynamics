#!/bin/bash
# Kill sweep trials that exceed a wall-clock budget. Some configurations drive
# the SDE solver into batches that take minutes each, which stalls the agent
# indefinitely; killing the trial lets the agent move to the next one.
LIMIT=${1:-6000}
while true; do
  for p in $(pgrep -f "flows/nn_potential.py"); do
    e=$(ps -o etimes= -p "$p" 2>/dev/null | tr -d ' ')
    if [ -n "$e" ] && [ "$e" -gt "$LIMIT" ]; then
      echo "$(date '+%F %T') killing pid $p after ${e}s (limit ${LIMIT}s)"
      kill -9 "$p"
    fi
  done
  sleep 120
done
