# Changelog

All notable changes to nanoDiffusionLab are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-23

### Added

- End-to-end masked-diffusion language-model training and iterative parallel sampling.
- Complete GPT-style autoregressive training and K/V-cached generation.
- A shared Transformer backbone with causal and bidirectional SDPA attention.
- Character and memory-mapped tokenized data paths, DDP, mixed precision, exact resume, and
  reproducible experiment artifacts.
- Two-seed, fixed-token-budget 106M TinyStories training results.
- A 10,000-sample quality and latency benchmark across four MDLM sampling-step settings.
- A detailed Chinese training tutorial and animated AR/MDLM explanations.
- Offline CPU smoke tests and continuous integration for Python 3.10 and 3.12.

### Changed

- Separated runnable configurations from unvalidated architecture-only targets.
- Organized command-line workflows under `scripts/` and added contribution guidance.

### Known limitations

- Block diffusion is planned but not implemented.
- The repository does not include pretrained checkpoints or a production FineWeb data recipe.
- AR and MDLM training losses are objective-specific and are not directly comparable likelihoods.
- Current MDLM samples trail the AR baseline in the reported TinyStories quality evaluation.

[0.1.0]: https://github.com/tutao0123/nanoDiffusionLab/releases/tag/v0.1.0
