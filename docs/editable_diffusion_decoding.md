# Revisable Diffusion Decoding

This note records a deliberately small experiment inspired by the
[LLaDA2.2 technical report](https://github.com/inclusionAI/LLaDA2.X/blob/main/LLaDA2_2_tech_report.pdf).
It does **not** reproduce LLaDA2.2. The repository's main question remains a controlled comparison
between autoregressive (AR) and masked-diffusion (MDLM) objectives using the same Transformer,
dataset, tokenizer, training budget, prompts, and evaluation protocol.

## The transferable idea

The most transferable idea in LLaDA2.2 is not its 100B MoE scale, 128K context, or agentic
reinforcement learning. It is that diffusion decoding should not turn an early local prediction
into an irreversible hard constraint too soon.

LLaDA2.2 implements that principle with Levenshtein editing. Its model predicts KEEP, SUBSTITUTE,
DELETE, and INSERT operations; supervised labels are recomputed from longest-common-subsequence
(LCS) alignments, and L-EBPO later optimizes block-level editing decisions with environment
rewards. Our experiment asks a narrower question: can a fixed-length MDLM sampler benefit from
occasionally masking its least-confident revealed tokens again?

## What was implemented

`sample_masked` now accepts three optional controls:

- `remask_fraction`: the largest fraction of editable revealed positions revisited at a step;
- `remask_decay_start`: the progress point after which that fraction decays linearly to zero;
- `protected_token_ids`: structural tokens, such as GPT-2 EOT, that cannot be masked again.

Prompt positions are always fixed. Ranking uses the probability recorded when a position was
revealed, not a new score computed after the fact. The last step never adds masks. A value of
`remask_fraction=0` preserves the original sampler token for token and remains the default.

For example:

```bash
python sample.py \
  --checkpoint out/tinystories-106m-mdlm-seed1337/best.pt \
  --length 160 --steps 64 \
  --remask-fraction 0.10 \
  --remask-decay-start 0.75 \
  --protect-eot
```

## Decoder comparison

| Property | Cached AR | Original MDLM | Remask MDLM | LLaDA2.2 Levenshtein editing |
|---|---|---|---|---|
| Attention and conditioning | Causal; next token depends on the fixed prefix | Bidirectional over prompt, masks, and revealed tokens | Same MDLM backbone and conditioning | Block-diffusion conditioning with edit actions |
| Can revealed tokens change? | No | No | Yes, by masking low-confidence tokens again | Yes, through KEEP/SUBSTITUTE/DELETE/INSERT |
| Insert/delete or dynamic length | No | No; fixed output canvas | No; fixed output canvas | Yes; explicit insertion and deletion |
| Network updates in this experiment | 128 for a 128-token continuation | 64 | 64 | Architecture- and trajectory-dependent |
| Parallelism | One new token per cached update | Multiple positions revealed per update | Multiple positions revealed; some may be revisited | Parallel block denoising plus edit decisions |
| Characteristic risk | Left-to-right error accumulation | Early mistakes become arbitrary-order hard constraints | Repeated work and unstable late structure | Harder edit credit assignment; INSERT is more difficult than DELETE |

Remasking is therefore closer to a sampler-side rollback than to a learned editor. It changes
neither the training objective nor the checkpoint.

## Stage 1: 100-prompt screen

The screen used the same fixed prompts and both 106M MDLM training seeds. Each row contains 200
continuations.

| Variant | EOT completion | Distinct-3 | Repeated 4-grams | Gate |
|---|---:|---:|---:|---|
| Original MDLM | 23.0% | 0.832 | 0.0963 | Reference |
| Fixed 5% | 17.0% | 0.863 | 0.0749 | Fail: EOT |
| Fixed 10% | 18.0% | 0.888 | 0.0544 | Fail: EOT |
| 10% + EOT protection | 19.5% | 0.888 | 0.0545 | Fail: EOT |
| 5% + EOT protection + final-quarter decay | 22.0% | 0.848 | 0.0842 | Fail: repetition |
| 10% + EOT protection + final-quarter decay | 22.5% | 0.862 | 0.0704 | **Pass** |

The gate required at least a 20% relative reduction in repeated 4-grams, no more than a two-point
drop in EOT completion, and no more than a 0.01 drop in distinct-3. Only the final variant passed,
with a 26.9% repetition reduction and a 0.5-point EOT drop.

## Stage 2: 1,000-prompt evaluation

The selected variant generated 2,000 continuations: 1,000 prompts for each of two training seeds.
The existing 2,000 cached-AR and 2,000 irreversible 64-step MDLM continuations were reused. DeepSeek
V4 Flash judged all 4,000 new blinded, position-balanced pairs; DeepSeek V4 Pro audited 100
deterministically sampled pairs from each comparison.

![TinyStories 106M revisable decoding quality-latency frontier](assets/remask_quality_latency_frontier.png)

| Decoder | EOT completion | Distinct-3 | Repeated 4-grams |
|---|---:|---:|---:|
| Cached AR | 25.20% | 0.950 | 0.0195 |
| Original MDLM | 26.50% | 0.832 | 0.0951 |
| Remask MDLM | 28.25% | 0.867 | 0.0672 |

The transparent metrics improve: remasking reduces repeated 4-grams by 29.4% relative to the
original MDLM, raises distinct-3 by 0.034, and does not hurt EOT completion. The blinded comparison,
however, rejects the tempting proxy-metric conclusion.

| Judge and comparison | Pairs | Remask wins | Ties | Remask losses | Remask utility |
|---|---:|---:|---:|---:|---:|
| Flash: remask vs AR | 2,000 | 64 | 17 | 1,919 | 0.036 |
| Flash: remask vs original MDLM | 2,000 | 356 | 714 | 930 | 0.357 |
| Pro audit: remask vs AR | 100 | 6 | 0 | 94 | 0.060 |
| Pro audit: remask vs original MDLM | 100 | 27 | 6 | 67 | 0.300 |

Utility assigns 1 to a remask win, 0.5 to a tie, and 0 to a loss. Both judges prefer the original
MDLM to remasking in the direct comparison, even though the local diversity and repetition metrics
move in the desired direction. The likely interpretation is that low reveal-time confidence is not
a reliable edit target after the surrounding context changes; resampling also spends the same
64-update budget revisiting tokens rather than resolving only new positions.

This is a useful negative result. The experimental sampler stays available for research, but the
public default remains the original irreversible MDLM.

The same-run timing comparison reaches the same engineering conclusion. At batch 1, remask MDLM
takes 0.410 seconds versus 0.386 seconds for the original MDLM and 0.678 seconds for cached AR. At
batch 32, remasking takes 3.492 seconds versus 2.570 seconds for the original MDLM. Revisability
keeps the 64 network updates but projects and samples extra positions, so it does not come for free.

## Scope and next experiments

This implementation does not include Block Diffusion, learned KEEP/SUBSTITUTE/DELETE/INSERT
actions, LCS-derived labels, MoE routing, or L-EBPO. Those features would change the model,
training data construction, objective, or optimization procedure and should be evaluated as
separate experiments.

The next revisability experiment should use a context-aware current confidence or disagreement
signal, explicitly account for the update budget, and test learned editing or block diffusion
separately. Any candidate should retain the same AR comparison and blinded audit; repeated n-grams
alone are not a sufficient selection target.

See the complete measured tables and provenance in
[`reports/tinystories_106m_remask_generation.md`](../reports/tinystories_106m_remask_generation.md)
and the screening record in
[`reports/remask_validation_100.md`](../reports/remask_validation_100.md).
