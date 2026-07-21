# Architecture

nanoDiffusionLab keeps one Transformer implementation and changes only the parts that define
the language-modeling objective.

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

## Distributed training

`train.py` discovers `RANK`, `LOCAL_RANK`, and `WORLD_SIZE` from `torchrun`, wraps the model in
DDP, uses `no_sync()` during gradient accumulation, and saves checkpoints only on rank zero.
The 4×A40 target uses data parallelism because its PCIe/PHB topology favors fewer, larger gradient
reductions over communication-heavy parameter or tensor sharding.

## Intended milestones

1. Character-level masked diffusion and AR baseline (implemented).
2. Tokenized/sharded TinyStories and FineWeb-Edu loaders.
3. Fair AR/MDLM experiments with likelihood-aware evaluation.
4. Block-diffusion attention, objective, and sampler.
5. Interactive denoising visualization and instruction tuning.

