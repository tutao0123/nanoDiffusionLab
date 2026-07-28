"""Run a reproducible multi-GPU remask sweep on the trained TinyStories MDLMs."""

# ruff: noqa: E402 -- make repository modules importable when executed as a script

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from config import ModelConfig
from diffusion import sample_masked
from evaluation import (
    append_jsonl,
    load_validation_prompts,
    local_text_metrics,
    read_jsonl,
    stable_seed,
)
from experiment import atomic_json, environment_metadata
from model import Transformer

DEFAULT_OUTPUT = ROOT / "out/remask-validation"
DEFAULT_MANIFEST = ROOT / "data/tinystories-gpt2/manifest.json"
CHECKPOINTS = (
    ("mdlm-seed1337", 1337, ROOT / "out/tinystories-106m-mdlm/best.pt"),
    ("mdlm-seed2027", 2027, ROOT / "out/tinystories-106m-mdlm-seed2027/best.pt"),
)
VARIANT_CATALOG: dict[str, dict[str, Any]] = {
    "baseline": {
        "label": "Irreversible baseline",
        "remask_fraction": 0.0,
        "remask_decay_start": 1.0,
        "protect_eot": False,
    },
    "fixed-05": {
        "label": "Fixed 5% remask",
        "remask_fraction": 0.05,
        "remask_decay_start": 1.0,
        "protect_eot": False,
    },
    "fixed-10": {
        "label": "Fixed 10% remask",
        "remask_fraction": 0.1,
        "remask_decay_start": 1.0,
        "protect_eot": False,
    },
    "protect-10": {
        "label": "Fixed 10% remask, EOT protected",
        "remask_fraction": 0.1,
        "remask_decay_start": 1.0,
        "protect_eot": True,
    },
    "decay-protect-05": {
        "label": "5% remask, EOT protected, final-quarter decay",
        "remask_fraction": 0.05,
        "remask_decay_start": 0.75,
        "protect_eot": True,
    },
    "decay-protect-10": {
        "label": "10% remask, EOT protected, final-quarter decay",
        "remask_fraction": 0.1,
        "remask_decay_start": 0.75,
        "protect_eot": True,
    },
}
DEFAULT_VARIANTS = tuple(VARIANT_CATALOG)
METRICS = (
    "length",
    "ended",
    "unique_token_ratio",
    "distinct_1",
    "distinct_2",
    "distinct_3",
    "repeated_4gram_ratio",
)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    root.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    subparsers = root.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--num-prompts", type=int, default=100)
    prepare.add_argument("--prompt-length", type=int, default=32)
    prepare.add_argument("--output-length", type=int, default=128)
    prepare.add_argument("--steps", type=int, default=64)
    prepare.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(VARIANT_CATALOG),
        default=list(DEFAULT_VARIANTS),
    )
    prepare.add_argument("--seed", type=int, default=4242)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--batch-size", type=int, default=8)
    generate.add_argument("--temperature", type=float, default=1.0)
    generate.add_argument("--top-k", type=int, default=40)
    generate.add_argument("--limit", type=int)

    subparsers.add_parser("report")
    return root


def tokenizer(manifest_path: Path):
    try:
        import tiktoken
    except ImportError as error:
        raise SystemExit('validation requires: pip install -e ".[data]"') from error
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return tiktoken.get_encoding(manifest["tokenizer"]["name"]), manifest


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[Transformer, ModelConfig]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ModelConfig(**checkpoint["config"]["model"])
    model = Transformer(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, config


def variant_identity(variant: dict[str, Any]) -> str:
    """Return a stable identity that changes whenever sampler behavior changes."""
    return (
        f"{variant['name']}:fraction={float(variant['remask_fraction']):.6g}:"
        f"decay={float(variant['remask_decay_start']):.6g}:"
        f"protect_eot={int(bool(variant['protect_eot']))}"
    )


def variant_key(label: str, variant: dict[str, Any], prompt_id: int) -> str:
    return f"{label}:{variant_identity(variant)}:p{prompt_id}"


def prepare(args: argparse.Namespace) -> None:
    if len(set(args.variants)) != len(args.variants):
        raise SystemExit("variants must be unique")
    encoding, manifest = tokenizer(args.manifest)
    prompts = load_validation_prompts(
        args.manifest, args.num_prompts, args.prompt_length, args.seed
    )
    for prompt in prompts:
        prompt["text"] = encoding.decode(prompt["token_ids"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = args.output_dir / "prompts.jsonl"
    prompt_path.write_text(
        "".join(json.dumps(prompt, ensure_ascii=False) + "\n" for prompt in prompts),
        encoding="utf-8",
    )
    atomic_json(
        args.output_dir / "validation.json",
        {
            "format": "nanodiffusion-remask-validation/2",
            "dataset": manifest["dataset"],
            "tokenizer": manifest["tokenizer"],
            "num_prompts": args.num_prompts,
            "prompt_length": args.prompt_length,
            "output_length": args.output_length,
            "steps": args.steps,
            "variants": [
                {"name": name, **VARIANT_CATALOG[name]} for name in args.variants
            ],
            "prompt_seed": args.seed,
            "inference_dtype": "bfloat16",
            "checkpoints": [
                {"label": label, "train_seed": seed, "path": str(path)}
                for label, seed, path in CHECKPOINTS
            ],
            "selection_gate": {
                "minimum_relative_repetition_reduction": 0.2,
                "maximum_eot_completion_drop": 0.02,
                "maximum_distinct_3_drop": 0.01,
            },
            "environment": environment_metadata(),
        },
    )
    print(f"wrote {len(prompts)} prompts to {prompt_path}")


def generate(args: argparse.Namespace) -> None:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise SystemExit("remask validation requires CUDA")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    settings = json.loads((args.output_dir / "validation.json").read_text())
    encoding, _manifest = tokenizer(args.manifest)
    prompts = [
        prompt
        for prompt in read_jsonl(args.output_dir / "prompts.jsonl")
        if prompt["prompt_id"] % world_size == rank
    ]
    if args.limit is not None:
        prompts = prompts[: args.limit]
    output_path = args.output_dir / "generations" / f"rank{rank:02d}.jsonl"
    completed = {
        variant_key(record["label"], record["variant_config"], int(record["prompt_id"]))
        for record in read_jsonl(output_path)
    }
    total_length = int(settings["prompt_length"]) + int(settings["output_length"])
    eot_token_id = int(settings["tokenizer"]["eot_token_id"])
    performance_path = args.output_dir / f"performance-rank{rank:02d}.json"
    variants_by_name = {variant["name"]: variant for variant in settings["variants"]}
    performance_lookup = {}
    for row in json.loads(performance_path.read_text()) if performance_path.exists() else []:
        config = row.get("variant_config") or variants_by_name[row["variant"]]
        performance_lookup[(row["label"], variant_identity(config))] = row

    for label, train_seed, checkpoint_path in CHECKPOINTS:
        model, model_config = load_model(checkpoint_path, device)
        if model_config.mask_token_id is None:
            raise ValueError("MDLM checkpoint has no mask token")
        for variant in settings["variants"]:
            variant_name = variant["name"]
            pending = [
                prompt
                for prompt in prompts
                if variant_key(label, variant, prompt["prompt_id"]) not in completed
            ]
            torch.cuda.reset_peak_memory_stats(device)
            variant_start = time.perf_counter()
            generated_count = 0
            for offset in range(0, len(pending), args.batch_size):
                batch = pending[offset : offset + args.batch_size]
                prompt_tokens = torch.tensor(
                    [prompt["token_ids"] for prompt in batch], dtype=torch.long, device=device
                )
                initial = torch.full(
                    (len(batch), total_length),
                    model_config.mask_token_id,
                    dtype=torch.long,
                    device=device,
                )
                initial[:, : settings["prompt_length"]] = prompt_tokens
                generators = [
                    torch.Generator(device=device).manual_seed(
                        stable_seed("remask-validation", label, prompt["prompt_id"])
                    )
                    for prompt in batch
                ]
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    result = sample_masked(
                        model=model,
                        shape=tuple(initial.shape),
                        mask_token_id=model_config.mask_token_id,
                        steps=int(settings["steps"]),
                        temperature=args.temperature,
                        top_k=args.top_k,
                        initial_tokens=initial,
                        generator=generators,
                        remask_fraction=float(variant["remask_fraction"]),
                        remask_decay_start=float(variant["remask_decay_start"]),
                        protected_token_ids=(eot_token_id,) if variant["protect_eot"] else None,
                    )
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                for row, prompt in enumerate(batch):
                    continuation_ids = result[row, settings["prompt_length"] :].tolist()
                    append_jsonl(
                        output_path,
                        {
                            "key": variant_key(label, variant, prompt["prompt_id"]),
                            "label": label,
                            "train_seed": train_seed,
                            "prompt_id": prompt["prompt_id"],
                            "prompt": prompt["text"],
                            "steps": settings["steps"],
                            "variant": variant_name,
                            "variant_config": variant,
                            "sample_seed": stable_seed(
                                "remask-validation", label, prompt["prompt_id"]
                            ),
                            "output_token_ids": continuation_ids,
                            "continuation": encoding.decode(continuation_ids),
                            "local_metrics": local_text_metrics(continuation_ids, eot_token_id),
                            "batch_seconds": elapsed,
                            "batch_size": len(batch),
                        },
                    )
                generated_count += len(batch)
            variant_seconds = time.perf_counter() - variant_start
            if generated_count:
                performance_lookup[(label, variant_identity(variant))] = {
                    "rank": rank,
                    "label": label,
                    "train_seed": train_seed,
                    "variant": variant_name,
                    "variant_config": variant,
                    "sequences": generated_count,
                    "elapsed_seconds": variant_seconds,
                    "sequences_per_second": generated_count / max(variant_seconds, 1e-9),
                    "peak_memory_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
                }
            print(
                f"rank {rank}: {label} variant={variant_name} "
                f"complete ({generated_count} new)"
            )
        del model
        torch.cuda.empty_cache()
    atomic_json(
        performance_path,
        [
            performance_lookup[key]
            for key in sorted(performance_lookup, key=lambda item: (item[0], item[1]))
        ],
    )


def generation_records(output_dir: Path) -> list[dict[str, Any]]:
    return [
        record
        for path in sorted((output_dir / "generations").glob("rank*.jsonl"))
        for record in read_jsonl(path)
    ]


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for record in records:
        grouped[record["variant"]].append(record["local_metrics"])
    return {
        variant: {
            "samples": float(len(rows)),
            **{metric: statistics.mean(row[metric] for row in rows) for metric in METRICS},
        }
        for variant, rows in sorted(grouped.items())
    }


def select_candidate(
    aggregate: dict[str, dict[str, float]],
    gate: dict[str, float],
) -> tuple[str | None, dict[str, dict[str, float | bool]]]:
    baseline = aggregate["baseline"]
    decisions = {}
    for variant, row in aggregate.items():
        if variant == "baseline":
            continue
        repetition_reduction = (
            baseline["repeated_4gram_ratio"] - row["repeated_4gram_ratio"]
        ) / max(baseline["repeated_4gram_ratio"], 1e-12)
        eot_drop = baseline["ended"] - row["ended"]
        distinct_3_drop = baseline["distinct_3"] - row["distinct_3"]
        passed = (
            repetition_reduction >= gate["minimum_relative_repetition_reduction"]
            and eot_drop <= gate["maximum_eot_completion_drop"] + 1e-12
            and distinct_3_drop <= gate["maximum_distinct_3_drop"] + 1e-12
        )
        decisions[variant] = {
            "passed": passed,
            "relative_repetition_reduction": repetition_reduction,
            "eot_completion_drop": eot_drop,
            "distinct_3_drop": distinct_3_drop,
        }
    passing = [variant for variant, row in decisions.items() if row["passed"]]
    selected = (
        min(passing, key=lambda variant: (aggregate[variant]["repeated_4gram_ratio"], variant))
        if passing
        else None
    )
    return selected, decisions


def report(args: argparse.Namespace) -> None:
    settings = json.loads((args.output_dir / "validation.json").read_text())
    records = generation_records(args.output_dir)
    expected = int(settings["num_prompts"]) * len(settings["checkpoints"]) * len(
        settings["variants"]
    )
    if len(records) != expected:
        raise SystemExit(f"expected {expected} generation records, found {len(records)}")
    aggregate = aggregate_records(records)
    selected, gate_decisions = select_candidate(aggregate, settings["selection_gate"])
    seed_groups: dict[tuple[int, str], list[dict[str, float]]] = defaultdict(list)
    for record in records:
        seed_groups[(int(record["train_seed"]), record["variant"])].append(
            record["local_metrics"]
        )
    by_seed = {
        f"seed{seed}:{variant}": {
            "samples": len(rows),
            **{metric: statistics.mean(row[metric] for row in rows) for metric in METRICS},
        }
        for (seed, variant), rows in sorted(seed_groups.items())
    }
    performance_rows = [
        row
        for path in sorted(args.output_dir.glob("performance-rank*.json"))
        for row in json.loads(path.read_text())
    ]
    performance_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in performance_rows:
        performance_groups[row["variant"]].append(row)
    performance = {
        variant: {
            "mean_sequences_per_second_per_gpu": statistics.mean(
                row["sequences_per_second"] for row in rows if row["sequences"]
            ),
            "max_peak_memory_mib": max(row["peak_memory_mib"] for row in rows),
        }
        for variant, rows in sorted(performance_groups.items())
        if any(row["sequences"] for row in rows)
    }
    record_lookup = {
        (int(record["train_seed"]), int(record["prompt_id"]), record["variant"]): record
        for record in records
    }
    paired_repetition = {}
    for variant in aggregate:
        if variant == "baseline":
            continue
        deltas = []
        for record in records:
            if record["variant"] != "baseline":
                continue
            comparison = record_lookup[
                (int(record["train_seed"]), int(record["prompt_id"]), variant)
            ]
            deltas.append(
                comparison["local_metrics"]["repeated_4gram_ratio"]
                - record["local_metrics"]["repeated_4gram_ratio"]
            )
        paired_repetition[variant] = {
            "improved": sum(delta < 0 for delta in deltas),
            "unchanged": sum(delta == 0 for delta in deltas),
            "worsened": sum(delta > 0 for delta in deltas),
            "mean_delta": statistics.mean(deltas),
        }
    summary = {
        "settings": settings,
        "generation_records": len(records),
        "local_metrics": aggregate,
        "local_metrics_by_seed": by_seed,
        "performance": performance,
        "selection": {
            "selected_variant": selected,
            "gate_passed": selected is not None,
            "decisions": gate_decisions,
        },
        "paired_repetition": paired_repetition,
    }
    atomic_json(args.output_dir / "summary.json", summary)
    public_summary = {
        **summary,
        "settings": {
            "format": settings["format"],
            "dataset": settings["dataset"],
            "tokenizer": settings["tokenizer"],
            "num_prompts": settings["num_prompts"],
            "prompt_length": settings["prompt_length"],
            "output_length": settings["output_length"],
            "steps": settings["steps"],
            "variants": settings["variants"],
            "prompt_seed": settings["prompt_seed"],
            "inference_dtype": settings["inference_dtype"],
            "training_seeds": sorted(
                {int(checkpoint["train_seed"]) for checkpoint in settings["checkpoints"]}
            ),
            "selection_gate": settings["selection_gate"],
        },
    }
    atomic_json(args.output_dir / "public-summary.json", public_summary)
    variant_labels = {row["name"]: row["label"] for row in settings["variants"]}
    lines = [
        "# TinyStories 106M Remask Candidate Gate",
        "",
        f"- Prompts: {settings['num_prompts']} per seed",
        f"- Checkpoints: {len(settings['checkpoints'])}",
        f"- Denoising steps: {settings['steps']}",
        f"- Generated continuations: {len(records)}",
        f"- Selected variant: {selected or 'none'}",
        "",
        "| Variant | Samples | Ended | Unique | Distinct-3 | Repeated 4-grams | "
        "Observed seq/s/GPU | Peak MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, row in aggregate.items():
        perf = performance.get(variant, {})
        lines.append(
            f"| {variant_labels[variant]} | {int(row['samples'])} | {row['ended']:.3f} | "
            f"{row['unique_token_ratio']:.3f} | {row['distinct_3']:.3f} | "
            f"{row['repeated_4gram_ratio']:.4f} | "
            f"{perf.get('mean_sequences_per_second_per_gpu', float('nan')):.2f} | "
            f"{perf.get('max_peak_memory_mib', float('nan')):.0f} |"
        )
    lines.extend(
        (
            "",
            "Observed throughput is order-sensitive and is not a controlled latency benchmark.",
            "",
            "## Candidate gate",
            "",
            "| Variant | Pass | Repetition reduction | EOT drop | Distinct-3 drop |",
            "|---|---:|---:|---:|---:|",
        )
    )
    for variant, row in gate_decisions.items():
        lines.append(
            f"| {variant_labels[variant]} | {'yes' if row['passed'] else 'no'} | "
            f"{row['relative_repetition_reduction']:.1%} | "
            f"{row['eot_completion_drop']:+.3f} | {row['distinct_3_drop']:+.3f} |"
        )
    lines.extend(
        (
            "",
            "## Paired repetition changes versus baseline",
            "",
            "| Variant | Improved | Unchanged | Worsened | Mean delta |",
            "|---|---:|---:|---:|---:|",
        )
    )
    for variant, row in paired_repetition.items():
        lines.append(
            f"| {variant_labels[variant]} | {row['improved']} | {row['unchanged']} | "
            f"{row['worsened']} | {row['mean_delta']:+.4f} |"
        )
    lines.extend(
        (
            "",
            "Local metrics are diagnostic guardrails, not a substitute for blinded quality "
            "judgments.",
            "",
        )
    )
    report_path = args.output_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(report_path)


def main() -> None:
    args = parser().parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "generate":
        generate(args)
    elif args.command == "report":
        report(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
