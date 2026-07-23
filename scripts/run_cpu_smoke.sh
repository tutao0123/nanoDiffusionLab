#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=""

python_bin="${PYTHON_BIN:-python}"
smoke_root="$(mktemp -d "${TMPDIR:-/tmp}/nanodiffusion-smoke.XXXXXX")"
trap 'rm -rf "$smoke_root"' EXIT

run_objective() {
    local objective="$1"
    local output="$smoke_root/$objective"
    local sample="$output/sample.txt"
    local sample_args=()

    if [[ "$objective" == "autoregressive" ]]; then
        sample_args=(--prompt "Once")
    else
        sample_args=(--steps 4)
    fi

    "$python_bin" train.py \
        --config configs/smoke_char.py \
        --objective "$objective" \
        --out-dir "$output"

    "$python_bin" sample.py \
        --checkpoint "$output/best.pt" \
        --length 32 \
        --temperature 1.0 \
        --top-k 5 \
        --seed 7 \
        --output "$sample" \
        "${sample_args[@]}"

    "$python_bin" - "$output" "$objective" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
objective = sys.argv[2]
required = {
    "best.pt",
    "last.pt",
    "meta.json",
    "metrics.jsonl",
    "run.json",
    "sample.txt",
    "summary.json",
}
missing = sorted(name for name in required if not (output / name).is_file())
if missing:
    raise SystemExit(f"{objective}: missing smoke artifacts: {missing}")
summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
if summary["objective"] != objective or summary["step"] != 2:
    raise SystemExit(f"{objective}: unexpected summary: {summary}")
if not (output / "sample.txt").read_text(encoding="utf-8").strip():
    raise SystemExit(f"{objective}: sample is empty")
print(f"{objective}: CPU smoke passed")
PY
}

run_objective autoregressive
run_objective masked_diffusion
