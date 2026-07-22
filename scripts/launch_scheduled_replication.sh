#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

first_ar="out/tinystories-106m-ar/summary.json"
first_mdlm="out/tinystories-106m-mdlm/summary.json"

echo "scheduled replication entered tmux at $(date --iso-8601=seconds)"
while [[ ! -f "$first_ar" || ! -f "$first_mdlm" ]]; do
    echo "$(date --iso-8601=seconds): waiting for the seed-1337 AR/MDLM pair to finish"
    sleep 60
done

echo "first pair is complete; starting seed-2027 replication"
exec env PYTHONUNBUFFERED=1 SEED=2027 \
    bash scripts/run_tinystories_replication.sh
