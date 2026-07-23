"""Create a compact Markdown comparison from two completed run directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs=2, type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/tinystories_106m.md"))
    return parser.parse_args()


def load(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    run = json.loads((path / "run.json").read_text(encoding="utf-8"))
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    return run, summary


def value(summary: dict[str, Any], key: str) -> str:
    item = summary["final_metrics"].get(key)
    return "—" if item is None else f"{item:.4f}"


def clean_sample(text: str) -> str:
    """Keep generated text verbatim except for Markdown-hostile line-end whitespace."""
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def main() -> None:
    args = parse_args()
    records = [load(path) for path in args.runs]
    lines = [
        "# TinyStories 106M AR vs. Masked Diffusion",
        "",
        "> AR perplexity and masked-diffusion denoising loss are objective-specific; "
        "do not compare them as the same metric.",
        "",
        "| Metric | " + " | ".join(summary["objective"] for _, summary in records) + " |",
        "|---|" + "---|" * len(records),
    ]
    rows = (
        ("Parameters", lambda run, summary: f"{summary['parameter_count']:,}"),
        ("Input tokens", lambda run, summary: f"{summary['tokens_seen']:,}"),
        ("Supervised targets", lambda run, summary: f"{summary['target_tokens_seen']:,}"),
        ("Elapsed hours", lambda run, summary: f"{summary['elapsed_seconds'] / 3600:.2f}"),
        (
            "Mean training tokens/s",
            lambda run, summary: f"{summary['mean_training_tokens_per_second']:,.0f}",
        ),
        ("Peak memory MiB/GPU", lambda run, summary: f"{summary['peak_memory_mb']:,.0f}"),
        ("Best validation loss", lambda run, summary: f"{summary['best_val_loss']:.4f}"),
        ("Validation loss", lambda run, summary: value(summary, "val_loss")),
        ("AR perplexity", lambda run, summary: value(summary, "val_perplexity")),
        ("MDLM masked accuracy", lambda run, summary: value(summary, "val_masked_accuracy")),
        ("Git commit", lambda run, summary: str(run["environment"]["git_commit"])),
    )
    for label, render in rows:
        lines.append(f"| {label} | " + " | ".join(render(*record) for record in records) + " |")
    lines.extend(("", "## Fixed-seed samples", ""))
    for path, (_run, summary) in zip(args.runs, records, strict=True):
        sample_path = path / "samples.txt"
        lines.append(f"### {summary['objective']}")
        lines.append("")
        lines.append("```text")
        sample = (
            clean_sample(sample_path.read_text(encoding="utf-8"))
            if sample_path.exists()
            else "Not generated"
        )
        lines.append(sample)
        lines.append("```")
        lines.append("")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
