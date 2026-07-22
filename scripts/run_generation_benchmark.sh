#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python_bin="${PYTHON_BIN:-/root/miniconda3/bin/python}"
output_dir="${EVAL_OUTPUT_DIR:-out/tinystories-106m-generation-eval}"

"$python_bin" scripts/benchmark_generation.py --output-dir "$output_dir" prepare \
    --num-prompts 1000 --prompt-length 32 --output-length 128 --seed 4242

"$python_bin" -m torch.distributed.run --standalone --nproc_per_node=4 \
    scripts/benchmark_generation.py --output-dir "$output_dir" generate \
    --batch-size 8 --steps 8 16 32 64 --temperature 1.0 --top-k 40

"$python_bin" scripts/benchmark_generation.py --output-dir "$output_dir" performance \
    --batch-sizes 1 8 32 --steps 8 16 32 64 --warmup 10 --repeats 30 --device cuda:0

judge_env="${JUDGE_ENV_FILE:-/root/.config/nanodiffusionlab/judge.env}"
if [[ -f "$judge_env" ]]; then
    # shellcheck disable=SC1090
    source "$judge_env"
fi

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
    echo "generation and performance are complete; set DEEPSEEK_API_KEY to continue judging" >&2
    exit 2
fi

"$python_bin" scripts/benchmark_generation.py --output-dir "$output_dir" judge \
    --flash-model deepseek-v4-flash --pro-model deepseek-v4-pro \
    --pro-samples 100 --concurrency 32 --max-retries 5

"$python_bin" scripts/benchmark_generation.py --output-dir "$output_dir" report
