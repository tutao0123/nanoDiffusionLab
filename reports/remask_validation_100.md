# TinyStories 106M Remask Candidate Gate

## Decision

Only **10% remask with EOT protection and final-quarter decay** passed the preregistered screen.
It reduced repeated 4-grams by 26.9% relative to the irreversible baseline while limiting the EOT
completion drop to 0.5 percentage points and improving distinct-3. It therefore advanced to the
1,000-prompt evaluation; this gate did not establish a quality improvement.

## Protocol

- Checkpoints: completed TinyStories 106M MDLM runs for seeds 1337 and 2027.
- Prompts: 100 deterministic validation prompts per checkpoint, selected with seed 4242.
- Prompt and continuation lengths: 32 and 128 GPT-2 tokens.
- Sampling: 64 denoising steps, temperature 1.0, top-k 40, BF16.
- Pairing: every variant reused the same prompt-level random seed.
- Hardware: four NVIDIA A40 GPUs, one evaluation rank per GPU.
- Total: 1,200 continuations, 200 per variant.

The gate required all three conditions:

1. repeated-4gram at least 20% lower than the baseline;
2. EOT completion no more than 2 percentage points below the baseline;
3. distinct-3 no more than 0.01 below the baseline.

## Aggregate results

| Variant | Samples | EOT completion | Unique tokens | Distinct-3 | Repeated 4-grams |
|---|---:|---:|---:|---:|---:|
| Irreversible baseline | 200 | 0.230 | 0.412 | 0.832 | 0.0963 |
| Fixed 5% | 200 | 0.170 | 0.422 | 0.863 | 0.0749 |
| Fixed 10% | 200 | 0.180 | 0.419 | 0.888 | 0.0544 |
| Fixed 10% + EOT protection | 200 | 0.195 | 0.421 | 0.888 | 0.0545 |
| 5% + EOT protection + final-quarter decay | 200 | 0.220 | 0.417 | 0.848 | 0.0842 |
| 10% + EOT protection + final-quarter decay | 200 | 0.225 | 0.414 | 0.862 | 0.0704 |

The baseline repeated-4gram ratio closely reproduces the 0.095 value from the earlier 1,000-prompt
benchmark, so the screen is operating in the expected regime.

## Gate result

In the table below, a positive EOT drop means worse completion. A negative distinct-3 drop means
the candidate improved distinct-3.

| Variant | Pass | Repetition reduction | EOT drop | Distinct-3 drop |
|---|---:|---:|---:|---:|
| Fixed 5% | No | 22.3% | +0.060 | -0.031 |
| Fixed 10% | No | 43.5% | +0.050 | -0.055 |
| Fixed 10% + EOT protection | No | 43.4% | +0.035 | -0.056 |
| 5% + EOT protection + final-quarter decay | No | 12.6% | +0.010 | -0.016 |
| 10% + EOT protection + final-quarter decay | **Yes** | 26.9% | +0.005 | -0.030 |

Protecting EOT alone was insufficient because a fixed 10% policy could still alter nearby content
late in sampling and reduce completion. Combining structural-token protection with decay met all
three constraints.

## Paired repetition changes

| Variant | Improved prompts | Unchanged | Worsened | Mean change |
|---|---:|---:|---:|---:|
| Fixed 5% | 117 | 12 | 71 | -0.0214 |
| Fixed 10% | 154 | 7 | 39 | -0.0419 |
| Fixed 10% + EOT protection | 155 | 7 | 38 | -0.0418 |
| 5% + EOT protection + final-quarter decay | 110 | 5 | 85 | -0.0122 |
| 10% + EOT protection + final-quarter decay | 120 | 10 | 70 | -0.0259 |

The generated records, resolved configuration, environment metadata, and machine-readable summary
remain local under `out/remask-gate-100/`. Observed generation throughput was intentionally excluded
from the decision because the sequential variant order makes it unsuitable as a controlled latency
measurement. The sanitized, bounded result is committed as
[`reports/data/remask_validation_100.json`](data/remask_validation_100.json).

## Reproduction

```bash
python scripts/validate_remask.py \
  --output-dir out/remask-gate-100 prepare \
  --num-prompts 100 --prompt-length 32 --output-length 128 \
  --steps 64 --seed 4242

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/validate_remask.py \
  --output-dir out/remask-gate-100 generate \
  --batch-size 8 --temperature 1.0 --top-k 40

python scripts/validate_remask.py \
  --output-dir out/remask-gate-100 report
```

The selected candidate's larger blinded evaluation is reported separately in
[`tinystories_106m_remask_generation.md`](tinystories_106m_remask_generation.md).
