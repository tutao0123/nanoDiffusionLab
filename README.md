# nanoDiffusionLab

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

- compact character-level training on any UTF-8 text file;
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
python sample.py --checkpoint out/shakespeare-mdlm/ckpt.pt --show-steps
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

## 4× NVIDIA A40 over PCIe

The included 350M target is designed for four 48 GB A40s using DDP. Start it with:

```bash
torchrun --standalone --nproc_per_node=4 train.py --config configs/fineweb_350m.py
```

The current data loader is character-level and in-memory, so `fineweb_350m.py` describes the model
and distributed training target but is **not yet a production FineWeb recipe**. A sharded tokenizer
data pipeline is the next required step.

For a PCIe/PHB machine without active NVLink:

- prefer DDP over tensor parallelism or full parameter sharding at this scale;
- keep per-GPU micro-batches large enough to amortize all-reduce;
- use gradient accumulation (`no_sync()` avoids redundant reductions);
- benchmark NCCL on the actual server before claiming scaling numbers.

## Project map

```text
model.py                    shared Transformer and AR sampler
diffusion.py                corruption, denoising loss, parallel sampler
train.py                    character data, optimization, checkpointing, DDP
sample.py                   checkpoint loading and text generation
config.py                   typed model/training configuration
configs/shakespeare_char.py runnable ~10M character model
configs/tinystories_50m.py  next-stage target configuration
configs/fineweb_350m.py     4×A40 target configuration
tests/                      behavioral tests
docs/architecture.md        design choices and roadmap
```

## Scope and limitations

This first release optimizes for clarity and a correct experimental skeleton. It does not yet
include a subword tokenizer, memory-mapped/sharded datasets, exact MDLM likelihood weighting,
remasking, block diffusion, KV caching, benchmark harnesses, or pretrained checkpoints. The
TinyStories and FineWeb configs are capacity targets awaiting the tokenized data pipeline.

See [docs/architecture.md](docs/architecture.md) for the objective split and planned milestones.

## License

MIT
