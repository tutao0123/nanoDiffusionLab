#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python_bin="${PYTHON_BIN:-/root/miniconda3/bin/python}"
seed="${SEED:-2027}"
config="configs/tinystories_106m.py"
manifest="data/tinystories-gpt2/manifest.json"
ar_dir="out/tinystories-106m-ar-seed${seed}"
mdlm_dir="out/tinystories-106m-mdlm-seed${seed}"

if [[ ! -f "$manifest" ]]; then
    echo "missing $manifest; run python scripts/prepare_tinystories.py first" >&2
    exit 1
fi

run_objective() {
    local objective="$1"
    local output="$2"
    local resume=()
    mkdir -p "$output"
    if [[ -f "$output/last.pt" ]]; then
        resume=(--resume)
    fi
    "$python_bin" -m torch.distributed.run --standalone --nproc_per_node=4 train.py \
        --config "$config" \
        --objective "$objective" \
        --out-dir "$output" \
        --seed "$seed" \
        "${resume[@]}" 2>&1 | tee -a "$output/console.log"

    "$python_bin" sample.py \
        --checkpoint "$output/best.pt" \
        --prompt "Once upon a time" \
        --length 256 \
        --steps 32 \
        --temperature 1.0 \
        --top-k 40 \
        --seed "$seed" \
        --num-samples 4 \
        --output "$output/samples.txt"
}

run_objective autoregressive "$ar_dir"
run_objective masked_diffusion "$mdlm_dir"
"$python_bin" compare_runs.py "$ar_dir" "$mdlm_dir" \
    --output "reports/tinystories_106m_seed${seed}.md"
