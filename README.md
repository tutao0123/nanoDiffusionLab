# nanoDiffusionLab

**English** | [简体中文](README.zh-CN.md)

A minimal PyTorch laboratory for learning, training, and comparing autoregressive and diffusion
language models from scratch.

The project borrows nanoGPT's small-and-readable philosophy, while keeping its implementation
independent. One Transformer supports two runnable objectives today:

- **autoregressive** — causal attention and next-token prediction;
- **masked diffusion** — bidirectional attention, random mask corruption, and iterative parallel
  decoding.

Block diffusion is the next architecture milestone. It is documented but not presented as
implemented.

## What works

- compact character-level smoke tests and memory-mapped token-shard training;
- masked-only output projection to reduce diffusion logits memory;
- time/noise-level conditioning;
- confidence-based parallel unmasking and prompt-preserving infill;
- an autoregressive baseline using the same Transformer blocks;
- single-GPU, CPU, and DDP training with gradient accumulation;
- BF16/FP16 autocast, gradient checkpointing, SDPA, optional `torch.compile`;
- tests for corruption, causal versus bidirectional attention, and sampling.

## Quick start

Requires Python 3.10+ and PyTorch 2.3+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

mkdir -p data/tinyshakespeare
curl -L https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt \
  -o data/tinyshakespeare/input.txt

python train.py --config configs/shakespeare_char.py
python sample.py --checkpoint out/shakespeare-mdlm/best.pt --show-steps
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` and download the data with:

```powershell
New-Item -ItemType Directory -Force data/tinyshakespeare
Invoke-WebRequest https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt `
  -OutFile data/tinyshakespeare/input.txt
```

For a short plumbing check, add `--max-iters 10`. Good samples require a real training run.

## Switch to the AR baseline

Copy a config and change the model fields:

```python
objective="autoregressive"
mask_token_id=None
time_conditioning=False
```

Then train and sample with the same commands. Keeping model size, data, and training tokens fixed
makes the comparison meaningful.

## TinyStories 106M experiment on 4× NVIDIA A40

Prepare the pinned TinyStories revision with GPT-2 BPE:

```bash
pip install -e ".[data,dev]"
python scripts/prepare_tinystories.py
```

The paired model has 12 layers, width 576, 9 attention heads, a 1024-token context, and roughly
106M parameters. Train each objective for the same 2B input-token budget:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 train.py \
  --config configs/tinystories_106m.py --objective autoregressive \
  --out-dir out/tinystories-106m-ar

python -m torch.distributed.run --standalone --nproc_per_node=4 train.py \
  --config configs/tinystories_106m.py --objective masked_diffusion \
  --out-dir out/tinystories-106m-mdlm
```

Each run writes a resolved configuration, environment metadata, JSONL metrics, atomic resumable
checkpoints, milestone checkpoints, and a final summary. Use `--resume` to continue from that run's
`last.pt`.

To run the complete pair sequentially, including fixed-seed samples and a Markdown report:

```bash
bash scripts/run_tinystories_pair.sh
```

## Reports

- [106M implementation and initial AR run report](reports/initial_run_report.html) — a frozen
  step-250 snapshot; the paired training is still in progress.

## 4× NVIDIA A40 over PCIe

The included 350M target is designed for four 48 GB A40s using DDP. Start it with:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 train.py \
  --config configs/fineweb_350m.py
```

The TinyStories loader is tokenized and memory-mapped. `fineweb_350m.py` remains a model target,
not yet a production FineWeb recipe; its data acquisition and curation still need implementation.

For a PCIe/PHB machine without active NVLink:

- prefer DDP over tensor parallelism or full parameter sharding at this scale;
- keep per-GPU micro-batches large enough to amortize all-reduce;
- use gradient accumulation (`no_sync()` avoids redundant reductions);
- benchmark NCCL on the actual server before claiming scaling numbers.

## Project map

```text
model.py                    shared Transformer and AR sampler
diffusion.py                corruption, denoising loss, parallel sampler
train.py                    token-budget training, evaluation, checkpointing, DDP
data.py                     character and memory-mapped token-shard loaders
experiment.py               atomic artifacts and local experiment metadata
compare_runs.py             Markdown report generation from completed runs
sample.py                   checkpoint loading and text generation
config.py                   typed model/training configuration
configs/shakespeare_char.py runnable ~10M character model
configs/tinystories_106m.py runnable paired TinyStories configuration
configs/fineweb_350m.py     4×A40 target configuration
tests/                      behavioral tests
docs/architecture.md        design choices and roadmap
```

## Scope and limitations

This release optimizes for clarity and a correct experimental skeleton. It does not yet include
exact MDLM likelihood weighting, remasking, block diffusion, KV caching, or pretrained
checkpoints. Objective-specific validation losses are recorded but must not be compared as though
they were the same likelihood metric.

See [docs/architecture.md](docs/architecture.md) for the objective split and planned milestones.

## License

MIT
