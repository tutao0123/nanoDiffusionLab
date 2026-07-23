# Architecture

nanoDiffusionLab is first an end-to-end training implementation of a masked-diffusion language
model: corruption, noise conditioning, denoising loss, validation, distributed training,
checkpointing, and iterative sampling are all included. It also implements a complete GPT-style
autoregressive training and inference path rather than relying on an external baseline.

The two paths share one Transformer implementation and change only the parts that define the
language-modeling objective. This keeps each implementation useful on its own while enabling a
controlled comparison.

| Concern | Autoregressive | Masked diffusion | Block diffusion |
|---|---|---|---|
| Attention | causal | bidirectional | planned: causal across blocks, bidirectional within a block |
| Corruption | shift targets by one | replace a random fraction with `[MASK]` | planned |
| Loss | next-token cross entropy | cross entropy at masked positions | planned |
| Sampling | one token at a time | confidence-based parallel reveal | planned |
| Time input | none | scalar mask ratio through a small MLP | planned |

## Shared Transformer

`model.py` contains pre-normalized Transformer blocks using PyTorch scaled dot-product attention.
`ModelConfig.objective` selects causal or bidirectional attention. The masked-diffusion input
embedding has one extra row for `[MASK]`, while the output head predicts only real vocabulary
items.

The model accepts an optional output mask. During diffusion training the output projection is
applied only to corrupted positions, avoiding a full `batch × sequence × vocabulary` logits
tensor. This matters for the 32k-vocabulary, 350M target configuration.

## Masked diffusion objective

For each sequence, `diffusion.py` samples a noise level `t` uniformly from `(0, 1]`, masks each
token independently with probability `t`, and guarantees at least one training target. The
initial objective is intentionally the readable denoising cross entropy:

```text
x0 → sample t → mask tokens → Transformer(xt, t) → CE on masked positions
```

This is a practical teaching baseline, not yet the exact continuous-time MDLM likelihood
objective. Future experiments can add schedule weighting without replacing the model.

## Sampling

Generation starts from an all-mask sequence, or a prompt followed by masks. Each forward pass
samples a candidate token for every masked position, ranks those candidates by confidence, and
reveals enough positions to finish in the requested number of steps. Prompt tokens are never
changed. Remasking and exact ancestral MDLM sampling are intentionally deferred.

Autoregressive inference uses an optional per-layer K/V cache. Prefill runs causal attention over
the prompt once; each later forward pass receives one new token and appends its keys and values.
The cache contains no learned parameters, so existing checkpoints remain compatible. Cached
generation is limited to the configured context length rather than silently changing positional
semantics.

## Generation evaluation

The generation benchmark selects validation prompts at document boundaries and compares cached AR
with multiple MDLM step counts. Quality generation can be sharded across GPUs with per-example
random generators; latency measurements run on one GPU with explicit synchronization. DeepSeek
pairwise judgments are blinded and position-balanced, while distinct-n and repetition metrics act
as model-independent guardrails. Raw samples and API responses remain local; reports record model
IDs, usage, confidence intervals, and cross-judge agreement.

## Distributed training

`train.py` discovers `RANK`, `LOCAL_RANK`, and `WORLD_SIZE` from `torchrun`, wraps the model in
DDP, uses `no_sync()` during gradient accumulation, and saves checkpoints only on rank zero.
The 4×A40 target uses data parallelism because its PCIe/PHB topology favors fewer, larger gradient
reductions over communication-heavy parameter or tensor sharding.

## Intended milestones

1. Character-level masked diffusion and AR baseline (implemented).
2. Tokenized/sharded TinyStories loader and a reproducible 106M paired experiment (implemented).
3. Cached AR inference and a quality-latency AR/MDLM evaluation (implemented and completed at
   106M scale).
4. Block-diffusion attention, objective, and sampler.
5. FineWeb-Edu ingestion, likelihood-aware evaluation, and instruction tuning.
