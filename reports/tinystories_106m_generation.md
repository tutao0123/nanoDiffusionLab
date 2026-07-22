# TinyStories 106M Generation Quality and Speed

This benchmark compares the completed 106M autoregressive (AR) and masked-diffusion (MDLM)
checkpoints under the same prompts, output length, sampling settings, and inference precision. It
answers a different question from the training report: not how the objective-specific validation
losses compare, but what quality and latency trade-off the two decoders produce.

## Main result

On this implementation and training budget, MDLM decoding is substantially faster, but AR wins the
large majority of blinded quality comparisons. Increasing MDLM from 8 to 64 denoising steps moves
quality in the right direction, yet does not close the gap.

![TinyStories 106M quality-latency frontier](../docs/assets/quality_latency_frontier.png)

| Decoder | Batch-1 latency | Speed relative to cached AR | MDLM pairwise utility | 95% CI |
|---|---:|---:|---:|---:|
| Cached AR, 128 updates | 0.614 s | 1.00× | 0.500 reference | — |
| MDLM, 8 steps | 0.045 s | 13.63× | 0.0028 | [0.0008, 0.0053] |
| MDLM, 16 steps | 0.093 s | 6.56× | 0.0138 | [0.0092, 0.0187] |
| MDLM, 32 steps | 0.175 s | 3.51× | 0.0370 | [0.0295, 0.0450] |
| MDLM, 64 steps | 0.348 s | 1.76× | 0.0745 | [0.0632, 0.0860] |

Pairwise utility assigns 1 to an MDLM win, 0.5 to a tie, and 0 to an AR win. The AR reference is
shown at 0.5 only to mark equal preference; it is not an AR self-comparison measurement.

## Quality results

DeepSeek V4 Flash judged 8,000 blinded, position-balanced pairs. Each MDLM continuation was compared
with the AR continuation from the same validation prompt and training seed.

| MDLM steps | Pairs | MDLM wins | Ties | AR wins | MDLM utility |
|---:|---:|---:|---:|---:|---:|
| 8 | 2,000 | 5 | 1 | 1,994 | 0.0028 |
| 16 | 2,000 | 21 | 13 | 1,966 | 0.0138 |
| 32 | 2,000 | 65 | 18 | 1,917 | 0.0370 |
| 64 | 2,000 | 140 | 18 | 1,842 | 0.0745 |

A stratified audit of 100 pairs with DeepSeek V4 Pro produced **98.0% exact agreement** with the
Flash judge and **Cohen's kappa 0.960**. The two training seeds tell the same overall story. At 64
steps, MDLM utility was 0.086 for seed 1337 and 0.064 for seed 2027.

Transparent local metrics support the judge result. Across seeds, AR continuations have an average
unique-token ratio of about 0.565 and repeated-4-gram ratio of about 0.019. MDLM at 64 steps reaches
about 0.410 unique-token ratio and about 0.095 repeated-4-gram ratio. The current MDLM sampler is
therefore faster, but produces markedly more repetition and less coherent stories.

## Performance results

All timings use one NVIDIA A40, BF16 autocast, explicit CUDA synchronization, 10 warmup iterations,
and 30 measured iterations. Each value below is the mean across the two independently trained
checkpoints. Peak memory is the maximum PyTorch allocation observed across the two seeds.

| Decoder | Batch | Latency | Sequences/s | Peak memory |
|---|---:|---:|---:|---:|
| AR cached | 1 | 0.614 s | 1.63 | 571 MiB |
| AR cached | 8 | 0.736 s | 10.87 | 634 MiB |
| AR cached | 32 | 1.029 s | 31.09 | 844 MiB |
| AR uncached | 32 | 1.790 s | 17.88 | 1,075 MiB |
| MDLM, 8 steps | 32 | 0.325 s | 98.62 | 3,070 MiB |
| MDLM, 16 steps | 32 | 0.627 s | 51.07 | 3,191 MiB |
| MDLM, 32 steps | 32 | 1.235 s | 25.91 | 3,252 MiB |
| MDLM, 64 steps | 32 | 2.451 s | 13.05 | 3,283 MiB |

The AR K/V cache matters most at larger batches: at batch 32 it improves throughput by 74% over the
uncached implementation. MDLM exposes a different trade-off. Its small number of parallel network
updates gives excellent low-step throughput, while holding logits for many unresolved positions
raises batch memory relative to cached AR.

## Protocol

- Dataset: pinned `roneneldan/TinyStories` revision `f54c09f` with GPT-2 BPE.
- Models: two AR and two MDLM checkpoints, trained with seeds 1337 and 2027.
- Budget: 2.000B input tokens per checkpoint; approximately 106M parameters.
- Prompts: 1,000 deterministic validation prompts, 32 tokens each.
- Continuations: 128 tokens, temperature 1.0, top-k 40, deterministic per-example RNG streams.
- Inference: BF16; cached AR and confidence-based MDLM unmasking at 8, 16, 32, and 64 steps.
- Judging: identity-hidden and position-balanced pairwise comparisons; Flash on every pair and Pro on
  a deterministic stratified audit.
- Uncertainty: prompt-level bootstrap 95% confidence intervals.

The benchmark generated 10,000 continuations in total. The exact settings, local metrics,
per-seed judge summaries, timing rows, environment metadata, and frontier points are retained in
[`reports/data/tinystories_106m_generation_eval.json`](data/tinystories_106m_generation_eval.json).
Raw generations and judge responses remain outside Git because they are large and may contain
provider-specific response data.

## Interpretation and next experiments

The result is evidence about this compact baseline, not a general verdict on diffusion language
models. The current model uses a simple random-mask training objective and confidence-based
unmasking. A useful next stage is to test likelihood-aware weighting, remasking or correction,
alternative reveal schedules, and a block-diffusion objective while preserving this same evaluation
protocol. Larger backbones and longer training budgets can then show whether the quality-speed
frontier changes with scale.
