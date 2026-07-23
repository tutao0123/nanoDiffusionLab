# Contributing to nanoDiffusionLab

Thank you for helping make diffusion language-model experiments easier to understand and reproduce.

## Development setup

Use Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[data,dev]"
```

Run the fast checks before opening a pull request:

```bash
ruff check .
pytest
bash -n scripts/*.sh
bash scripts/run_cpu_smoke.sh
```

The smoke script trains and samples both objectives on an included character fixture. It forces
CPU execution and does not download data.

## Pull requests

Keep each pull request focused and explain:

- what behavior changes and why;
- how the change was tested;
- whether existing experiment results remain comparable;
- any new data, compute, or API requirements.

Do not commit API keys, local environment files, raw datasets, generated output directories, or
large checkpoints. Prefer small fixtures that make tests fully offline.

## Contributing an experiment

A comparable AR/MDLM experiment should hold the backbone, data split, context length, input-token
budget, and evaluation prompts fixed. Report:

- the exact configuration and git commit;
- GPU model, topology, world size, precision, and software versions;
- parameter count, batch size, gradient accumulation, and random seeds;
- input-token budget, supervised-target count, throughput, and elapsed time;
- objective-specific validation metrics;
- sampling steps, latency methodology, and quality methodology.

Include a concise Markdown report and the commands needed to reproduce it. Small summary data and
figures are welcome; checkpoints and large generated corpora should be hosted separately and linked.

## Configuration policy

Files directly under `configs/` must be runnable with data preparation included in the repository.
Architecture-only targets belong in `configs/experimental/` and must state what is missing.

## Code style

Favor compact, explicit PyTorch code over deep abstraction. New behavior should have a focused test,
and changes to public commands or experiment semantics should update both English and Chinese
documentation where applicable.
