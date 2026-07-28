"""Benchmark the selected revisable MDLM sampler against AR and irreversible MDLM."""

# ruff: noqa: E402 -- make repository modules importable when executed as a script

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from diffusion import sample_masked
from evaluation import (
    JUDGE_DIMENSIONS,
    append_jsonl,
    bootstrap_mean_interval,
    cohens_kappa,
    judge_messages,
    local_text_metrics,
    read_jsonl,
    stable_seed,
    validate_judgment,
)
from experiment import atomic_json, environment_metadata
from scripts.benchmark_generation import CHECKPOINTS, generation_seed, load_model, timed_call

DEFAULT_OUTPUT = ROOT / "out/tinystories-106m-remask-eval"
DEFAULT_BASE_EVAL = ROOT / "out/tinystories-106m-generation-eval"
DEFAULT_MANIFEST = ROOT / "data/tinystories-gpt2/manifest.json"
SELECTED_VARIANT = {
    "name": "decay-protect-10",
    "label": "MDLM 64-step revisable",
    "remask_fraction": 0.1,
    "remask_decay_start": 0.75,
    "protect_eot": True,
}
AR_CHECKPOINTS = [row for row in CHECKPOINTS if row[1] == "autoregressive"]
MDLM_CHECKPOINTS = [row for row in CHECKPOINTS if row[1] == "masked_diffusion"]


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    root.add_argument("--base-eval-dir", type=Path, default=DEFAULT_BASE_EVAL)
    root.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    subparsers = root.add_subparsers(dest="command", required=True)

    subparsers.add_parser("prepare")

    generate = subparsers.add_parser("generate")
    generate.add_argument("--batch-size", type=int, default=8)
    generate.add_argument("--temperature", type=float, default=1.0)
    generate.add_argument("--top-k", type=int, default=40)
    generate.add_argument("--limit", type=int)

    performance = subparsers.add_parser("performance")
    performance.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    performance.add_argument("--warmup", type=int, default=10)
    performance.add_argument("--repeats", type=int, default=30)
    performance.add_argument("--device", default="cuda:0")

    judge = subparsers.add_parser("judge")
    judge.add_argument("--flash-model", default="deepseek-v4-flash")
    judge.add_argument("--pro-model", default="deepseek-v4-pro")
    judge.add_argument("--pro-samples-per-comparison", type=int, default=100)
    judge.add_argument("--concurrency", type=int, default=32)
    judge.add_argument("--max-retries", type=int, default=5)
    judge.add_argument("--limit", type=int)

    subparsers.add_parser("report")
    return root


def tokenizer(manifest_path: Path):
    try:
        import tiktoken
    except ImportError as error:
        raise SystemExit('benchmark requires: pip install -e ".[data,eval,viz]"') from error
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return tiktoken.get_encoding(manifest["tokenizer"]["name"]), manifest


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(args: argparse.Namespace) -> None:
    base_settings_path = args.base_eval_dir / "evaluation.json"
    base_prompts_path = args.base_eval_dir / "prompts.jsonl"
    if not base_settings_path.exists() or not base_prompts_path.exists():
        raise SystemExit("base generation evaluation is incomplete")
    base_settings = json.loads(base_settings_path.read_text())
    prompts = read_jsonl(base_prompts_path)
    if len(prompts) != 1000:
        raise SystemExit(f"expected 1000 base prompts, found {len(prompts)}")
    base_records = base_generation_records(args.base_eval_dir)
    required = {
        ("autoregressive", seed, 0): 0 for seed in (1337, 2027)
    } | {("masked_diffusion", seed, 64): 0 for seed in (1337, 2027)}
    for record in base_records:
        key = (
            record["objective"],
            int(record["train_seed"]),
            int(record.get("steps") or 0),
        )
        if key in required:
            required[key] += 1
    if any(count != 1000 for count in required.values()):
        raise SystemExit(f"base generation counts are incomplete: {required}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(base_prompts_path, args.output_dir / "prompts.jsonl")
    encoding, manifest = tokenizer(args.manifest)
    del encoding
    atomic_json(
        args.output_dir / "evaluation.json",
        {
            "format": "nanodiffusion-remask-generation-eval/1",
            "dataset": manifest["dataset"],
            "tokenizer": manifest["tokenizer"],
            "num_prompts": 1000,
            "prompt_length": int(base_settings["prompt_length"]),
            "output_length": int(base_settings["output_length"]),
            "steps": 64,
            "prompt_seed": int(base_settings["prompt_seed"]),
            "prompt_sha256": file_sha256(base_prompts_path),
            "base_evaluation_dir": str(args.base_eval_dir),
            "base_generation_counts": {
                f"{objective}:seed{seed}:steps{steps}": count
                for (objective, seed, steps), count in required.items()
            },
            "selected_variant": SELECTED_VARIANT,
            "inference_dtype": "bfloat16",
            "checkpoints": [
                {
                    "label": label,
                    "objective": objective,
                    "train_seed": seed,
                    "path": str(path),
                }
                for label, objective, seed, path in CHECKPOINTS
            ],
            "environment": environment_metadata(),
        },
    )
    print(args.output_dir / "evaluation.json")


def candidate_key(train_seed: int, prompt_id: int) -> str:
    return f"remask:seed{train_seed}:p{prompt_id}"


def generate(args: argparse.Namespace) -> None:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise SystemExit("remask benchmark generation requires CUDA")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    settings = json.loads((args.output_dir / "evaluation.json").read_text())
    encoding, _manifest = tokenizer(args.manifest)
    prompts = [
        prompt
        for prompt in read_jsonl(args.output_dir / "prompts.jsonl")
        if prompt["prompt_id"] % world_size == rank
    ]
    if args.limit is not None:
        prompts = prompts[: args.limit]
    output_path = args.output_dir / "generations" / f"rank{rank:02d}.jsonl"
    completed = {record["key"] for record in read_jsonl(output_path)}
    total_length = int(settings["prompt_length"]) + int(settings["output_length"])
    eot_token_id = int(settings["tokenizer"]["eot_token_id"])
    variant = settings["selected_variant"]

    for label, _objective, train_seed, checkpoint_path in MDLM_CHECKPOINTS:
        model, model_config = load_model(checkpoint_path, device)
        if model_config.mask_token_id is None:
            raise ValueError("MDLM checkpoint has no mask token")
        pending = [
            prompt
            for prompt in prompts
            if candidate_key(train_seed, prompt["prompt_id"]) not in completed
        ]
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
                    generation_seed(label, 64, prompt["prompt_id"])
                )
                for prompt in batch
            ]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                result = sample_masked(
                    model=model,
                    shape=tuple(initial.shape),
                    mask_token_id=model_config.mask_token_id,
                    steps=64,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    initial_tokens=initial,
                    generator=generators,
                    remask_fraction=float(variant["remask_fraction"]),
                    remask_decay_start=float(variant["remask_decay_start"]),
                    protected_token_ids=(eot_token_id,),
                )
            continuations = result[:, settings["prompt_length"] :].cpu().tolist()
            for prompt, token_ids in zip(batch, continuations, strict=True):
                append_jsonl(
                    output_path,
                    {
                        "key": candidate_key(train_seed, prompt["prompt_id"]),
                        "prompt_id": prompt["prompt_id"],
                        "label": f"mdlm-remask-seed{train_seed}",
                        "kind": "remask",
                        "objective": "masked_diffusion",
                        "train_seed": train_seed,
                        "steps": 64,
                        "variant": variant,
                        "sample_seed": generation_seed(label, 64, prompt["prompt_id"]),
                        "network_evaluations": 64,
                        "prompt": prompt["text"],
                        "continuation": encoding.decode(token_ids),
                        "output_token_ids": token_ids,
                    },
                )
        print(f"rank {rank}: seed={train_seed} complete ({len(pending)} new)")
        del model
        torch.cuda.empty_cache()


def base_generation_records(base_eval_dir: Path) -> list[dict[str, Any]]:
    return [
        record
        for path in sorted((base_eval_dir / "generations").glob("rank*.jsonl"))
        for record in read_jsonl(path)
    ]


def candidate_records(output_dir: Path) -> list[dict[str, Any]]:
    return [
        record
        for path in sorted((output_dir / "generations").glob("rank*.jsonl"))
        for record in read_jsonl(path)
    ]


def selected_base_records(base_eval_dir: Path) -> list[dict[str, Any]]:
    return [
        record
        for record in base_generation_records(base_eval_dir)
        if record["objective"] == "autoregressive"
        or (record["objective"] == "masked_diffusion" and record["steps"] == 64)
    ]


def record_kind(record: dict[str, Any]) -> str:
    if record.get("kind") == "remask":
        return "remask"
    return "ar" if record["objective"] == "autoregressive" else "baseline"


def build_pairs(
    base_records: list[dict[str, Any]],
    remask_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    lookup: dict[tuple[str, int, int], dict[str, Any]] = {}
    for record in [*base_records, *remask_records]:
        lookup[(record_kind(record), int(record["train_seed"]), int(record["prompt_id"]))] = record
    pairs = []
    for remask in remask_records:
        seed = int(remask["train_seed"])
        prompt_id = int(remask["prompt_id"])
        for comparison, opponent_kind in (
            ("remask-vs-ar", "ar"),
            ("remask-vs-baseline", "baseline"),
        ):
            opponent = lookup.get((opponent_kind, seed, prompt_id))
            if opponent is None:
                continue
            pair_id = f"{comparison}:seed{seed}:p{prompt_id}"
            swapped = stable_seed("remask-side", pair_id) % 2 == 1
            first, second = (remask, opponent) if swapped else (opponent, remask)
            pairs.append(
                {
                    "pair_id": pair_id,
                    "comparison": comparison,
                    "prompt_id": prompt_id,
                    "train_seed": seed,
                    "steps": 64,
                    "prompt": remask["prompt"],
                    "A_label": record_kind(first),
                    "A_text": first["continuation"],
                    "B_label": record_kind(second),
                    "B_text": second["continuation"],
                }
            )
    return sorted(pairs, key=lambda pair: pair["pair_id"])


def request_judgment(
    client,
    pair: dict[str, Any],
    model: str,
    max_retries: int,
) -> dict[str, Any]:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=judge_messages(pair["prompt"], pair["A_text"], pair["B_text"]),
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=512,
                stream=False,
                extra_body={
                    "thinking": {"type": "disabled"},
                    "user_id": "nanodiffusionlab-remask-eval",
                },
            )
            raw = response.choices[0].message.content or ""
            return {
                "pair_id": pair["pair_id"],
                "comparison": pair["comparison"],
                "train_seed": pair["train_seed"],
                "steps": pair["steps"],
                "A_label": pair["A_label"],
                "B_label": pair["B_label"],
                "judge_model_requested": model,
                "judge_model_returned": response.model,
                "judgment": validate_judgment(json.loads(raw)),
                "usage": response.usage.model_dump() if response.usage is not None else None,
                "raw_content": raw,
            }
        except Exception as error:
            if attempt + 1 == max_retries:
                raise RuntimeError(f"judge failed for {pair['pair_id']}: {error}") from error
            time.sleep(min(30.0, 2**attempt))
    raise AssertionError("unreachable")


def run_judge_model(
    pairs: list[dict[str, Any]],
    output_path: Path,
    model: str,
    concurrency: int,
    max_retries: int,
) -> None:
    try:
        from openai import OpenAI
    except ImportError as error:
        raise SystemExit('judge requires: pip install -e ".[eval]"') from error
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("set DEEPSEEK_API_KEY in the process environment")
    completed = {record["pair_id"] for record in read_jsonl(output_path)}
    pending = [pair for pair in pairs if pair["pair_id"] not in completed]
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(request_judgment, client, pair, model, max_retries): pair
            for pair in pending
        }
        for index, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            with lock:
                append_jsonl(output_path, record)
            if index % 100 == 0 or index == len(pending):
                print(f"{model}: {index}/{len(pending)} new judgments")


def pro_subset(pairs: list[dict[str, Any]], count_per_comparison: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        groups[pair["comparison"]].append(pair)
    selected = []
    for comparison, rows in sorted(groups.items()):
        ranked = sorted(
            rows,
            key=lambda pair: stable_seed("remask-pro-audit", comparison, pair["pair_id"]),
        )
        selected.extend(ranked[: min(count_per_comparison, len(ranked))])
    return selected


def judge(args: argparse.Namespace) -> None:
    pairs = build_pairs(
        selected_base_records(args.base_eval_dir),
        candidate_records(args.output_dir),
    )
    if args.limit is not None:
        pairs = pairs[: args.limit]
    expected = 4000
    if args.limit is None and len(pairs) != expected:
        raise SystemExit(f"expected {expected} complete pairs, found {len(pairs)}")
    run_judge_model(
        pairs,
        args.output_dir / "judgments-flash.jsonl",
        args.flash_model,
        args.concurrency,
        args.max_retries,
    )
    audit = pro_subset(pairs, args.pro_samples_per_comparison)
    run_judge_model(
        audit,
        args.output_dir / "judgments-pro.jsonl",
        args.pro_model,
        min(args.concurrency, 16),
        args.max_retries,
    )


def performance(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("performance benchmark requires a CUDA device")
    settings = json.loads((args.output_dir / "evaluation.json").read_text())
    prompts = read_jsonl(args.output_dir / "prompts.jsonl")
    eot_token_id = int(settings["tokenizer"]["eot_token_id"])
    records = []
    checkpoints_by_seed = {
        (objective, seed): (label, path)
        for label, objective, seed, path in CHECKPOINTS
    }
    for seed in (1337, 2027):
        ar_label, ar_path = checkpoints_by_seed[("autoregressive", seed)]
        mdlm_label, mdlm_path = checkpoints_by_seed[("masked_diffusion", seed)]
        for objective, label, checkpoint_path in (
            ("ar-cached", ar_label, ar_path),
            ("mdlm-baseline", mdlm_label, mdlm_path),
            ("mdlm-remask", mdlm_label, mdlm_path),
        ):
            model, model_config = load_model(checkpoint_path, device)
            for batch_size in args.batch_sizes:
                prompt_rows = [
                    prompts[index % len(prompts)]["token_ids"] for index in range(batch_size)
                ]
                prompt_tokens = torch.tensor(prompt_rows, dtype=torch.long, device=device)
                generators = [
                    torch.Generator(device=device).manual_seed(
                        generation_seed(label, 64 if objective != "ar-cached" else None, index)
                    )
                    for index in range(batch_size)
                ]
                if objective == "ar-cached":

                    def run(
                        model=model,
                        prompt_tokens=prompt_tokens,
                        generators=generators,
                    ):
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            return model.generate(
                                prompt_tokens,
                                settings["output_length"],
                                top_k=1,
                                use_cache=True,
                                generator=generators,
                            )

                    def first_update(model=model, prompt_tokens=prompt_tokens):
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            return model(prompt_tokens)

                    network_evaluations = int(settings["output_length"])
                else:
                    if model_config.mask_token_id is None:
                        raise ValueError("MDLM checkpoint has no mask token")
                    total_length = settings["prompt_length"] + settings["output_length"]
                    initial = torch.full(
                        (batch_size, total_length),
                        model_config.mask_token_id,
                        dtype=torch.long,
                        device=device,
                    )
                    initial[:, : settings["prompt_length"]] = prompt_tokens

                    def run(
                        model=model,
                        initial=initial,
                        generators=generators,
                        mask_token_id=model_config.mask_token_id,
                        objective=objective,
                    ):
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            return sample_masked(
                                model=model,
                                shape=tuple(initial.shape),
                                mask_token_id=mask_token_id,
                                steps=64,
                                top_k=1,
                                initial_tokens=initial,
                                generator=generators,
                                remask_fraction=(
                                    SELECTED_VARIANT["remask_fraction"]
                                    if objective == "mdlm-remask"
                                    else 0.0
                                ),
                                remask_decay_start=(
                                    SELECTED_VARIANT["remask_decay_start"]
                                    if objective == "mdlm-remask"
                                    else 1.0
                                ),
                                protected_token_ids=(
                                    (eot_token_id,) if objective == "mdlm-remask" else None
                                ),
                            )

                    def first_update(
                        model=model,
                        initial=initial,
                        mask_token_id=model_config.mask_token_id,
                    ):
                        t = initial.eq(mask_token_id).float().mean(dim=1)
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            return model(initial, t, initial.eq(mask_token_id))

                    network_evaluations = 64
                measured = timed_call(run, args.warmup, args.repeats, device)
                first = timed_call(first_update, args.warmup, args.repeats, device)
                measured.update(
                    {
                        "label": objective,
                        "train_seed": seed,
                        "batch_size": batch_size,
                        "network_evaluations": network_evaluations,
                        "output_tokens_per_second": (
                            batch_size * settings["output_length"] / measured["mean_seconds"]
                        ),
                        "sequences_per_second": batch_size / measured["mean_seconds"],
                        "first_update_ms": first["mean_seconds"] * 1000,
                    }
                )
                records.append(measured)
                print(seed, objective, f"batch={batch_size}", measured["mean_seconds"])
            del model
            torch.cuda.empty_cache()
    atomic_json(args.output_dir / "performance.json", records)


def judgment_outcome(record: dict[str, Any], target: str = "remask") -> str:
    winner = record["judgment"]["winner"]
    if winner == "tie":
        return "tie"
    return "win" if record[f"{winner}_label"] == target else "loss"


def summarize_judgments(rows: list[dict[str, Any]]) -> dict[str, Any]:
    outcomes = [judgment_outcome(row) for row in rows]
    utilities = [1.0 if value == "win" else 0.5 if value == "tie" else 0.0 for value in outcomes]
    mean, low, high = bootstrap_mean_interval(utilities)
    dimensions = {}
    for dimension in JUDGE_DIMENSIONS:
        deltas = []
        for row in rows:
            remask_side = "A" if row["A_label"] == "remask" else "B"
            other_side = "B" if remask_side == "A" else "A"
            deltas.append(
                row["judgment"]["scores"][remask_side][dimension]
                - row["judgment"]["scores"][other_side][dimension]
            )
        dimensions[dimension] = bootstrap_mean_interval(deltas)
    return {
        "count": len(rows),
        "remask_wins": outcomes.count("win"),
        "ties": outcomes.count("tie"),
        "remask_losses": outcomes.count("loss"),
        "remask_utility": mean,
        "utility_ci95": [low, high],
        "dimension_deltas": dimensions,
    }


def historical_baseline_judgments(base_eval_dir: Path) -> list[dict[str, Any]]:
    return [
        row
        for row in read_jsonl(base_eval_dir / "judgments-flash.jsonl")
        if int(row["steps"]) == 64
    ]


def render_frontier(
    remask_vs_ar: dict[str, Any],
    baseline_vs_ar: dict[str, Any],
    performance_rows: list[dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit('report figures require: pip install -e ".[viz]"') from error
    latency_groups: dict[str, list[float]] = defaultdict(list)
    for row in performance_rows:
        if row["batch_size"] == 1:
            latency_groups[row["label"]].append(float(row["mean_seconds"]))
    points = [
        {
            "label": "AR cached",
            "latency_seconds": statistics.mean(latency_groups["ar-cached"]),
            "utility": 0.5,
            "ci_low": 0.5,
            "ci_high": 0.5,
        },
        {
            "label": "MDLM baseline",
            "latency_seconds": statistics.mean(latency_groups["mdlm-baseline"]),
            "utility": baseline_vs_ar["remask_utility"],
            "ci_low": baseline_vs_ar["utility_ci95"][0],
            "ci_high": baseline_vs_ar["utility_ci95"][1],
        },
        {
            "label": "MDLM revisable",
            "latency_seconds": statistics.mean(latency_groups["mdlm-remask"]),
            "utility": remask_vs_ar["remask_utility"],
            "ci_low": remask_vs_ar["utility_ci95"][0],
            "ci_high": remask_vs_ar["utility_ci95"][1],
        },
    ]
    figure, axis = plt.subplots(figsize=(8.4, 5.2))
    colors = {"AR cached": "#2563eb", "MDLM baseline": "#ea580c", "MDLM revisable": "#16a34a"}
    for point in points:
        utility = float(point["utility"])
        axis.errorbar(
            float(point["latency_seconds"]),
            utility,
            yerr=[
                [utility - float(point["ci_low"])],
                [float(point["ci_high"]) - utility],
            ],
            marker="s" if point["label"] == "AR cached" else "o",
            color=colors[point["label"]],
            capsize=4,
            markersize=8,
        )
        axis.annotate(
            point["label"],
            (float(point["latency_seconds"]), utility),
            xytext=(7, 7),
            textcoords="offset points",
        )
    axis.axhline(0.5, color="#64748b", linestyle="--", linewidth=1)
    axis.set_xscale("log")
    axis.set_xlabel("Batch-1 latency per 128-token continuation (seconds, log scale)")
    axis.set_ylabel("Pairwise utility versus same-seed AR")
    axis.set_title("TinyStories 106M revisable decoding frontier", loc="left", weight="bold")
    axis.grid(axis="both", color="#e2e8f0", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(output_dir / "remask_quality_latency_frontier.png", dpi=180, facecolor="white")
    plt.close(figure)
    return points


def report(args: argparse.Namespace) -> None:
    settings = json.loads((args.output_dir / "evaluation.json").read_text())
    prompts_path = args.output_dir / "prompts.jsonl"
    if file_sha256(prompts_path) != settings["prompt_sha256"]:
        raise SystemExit("prompt hash no longer matches evaluation.json")
    remask_records = candidate_records(args.output_dir)
    if len(remask_records) != 2000:
        raise SystemExit(f"expected 2000 remask records, found {len(remask_records)}")
    expected_keys = {
        candidate_key(seed, prompt_id)
        for seed in (1337, 2027)
        for prompt_id in range(int(settings["num_prompts"]))
    }
    actual_keys = {record["key"] for record in remask_records}
    if actual_keys != expected_keys:
        raise SystemExit("remask generation keys do not match the expected seed/prompt grid")
    base_records = selected_base_records(args.base_eval_dir)
    eot_token_id = int(settings["tokenizer"]["eot_token_id"])
    local_groups: dict[str, list[dict[str, float]]] = defaultdict(list)
    for record in [*base_records, *remask_records]:
        local_groups[record_kind(record)].append(
            local_text_metrics(record["output_token_ids"], eot_token_id)
        )
    local_summary = {
        group: {metric: statistics.mean(row[metric] for row in rows) for metric in rows[0]}
        for group, rows in sorted(local_groups.items())
    }
    flash = read_jsonl(args.output_dir / "judgments-flash.jsonl")
    pro = read_jsonl(args.output_dir / "judgments-pro.jsonl")
    if len(flash) != 4000 or len(pro) != 200:
        raise SystemExit(
            f"expected 4000 Flash and 200 Pro judgments, found {len(flash)} and {len(pro)}"
        )
    flash_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in flash:
        flash_groups[row["comparison"]].append(row)
    judge_summary = {
        comparison: summarize_judgments(rows)
        for comparison, rows in sorted(flash_groups.items())
    }
    pro_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pro:
        pro_groups[row["comparison"]].append(row)
    pro_summary = {
        comparison: summarize_judgments(rows)
        for comparison, rows in sorted(pro_groups.items())
    }
    per_seed_summary = {
        comparison: {
            str(seed): summarize_judgments(
                [row for row in rows if int(row["train_seed"]) == seed]
            )
            for seed in (1337, 2027)
        }
        for comparison, rows in sorted(flash_groups.items())
    }
    agreement = {}
    flash_lookup = {row["pair_id"]: row for row in flash}
    for comparison in ("remask-vs-ar", "remask-vs-baseline"):
        matched = [
            (flash_lookup[row["pair_id"]], row)
            for row in pro
            if row["comparison"] == comparison and row["pair_id"] in flash_lookup
        ]
        if matched:
            flash_labels = [left["judgment"]["winner"] for left, _right in matched]
            pro_labels = [right["judgment"]["winner"] for _left, right in matched]
            agreement[comparison] = {
                "count": len(matched),
                "exact": sum(
                    left == right for left, right in zip(flash_labels, pro_labels, strict=True)
                )
                / len(matched),
                "cohens_kappa": cohens_kappa(flash_labels, pro_labels),
            }
    historical = historical_baseline_judgments(args.base_eval_dir)
    baseline_outcomes = []
    for row in historical:
        winner = row["judgment"]["winner"]
        if winner == "tie":
            baseline_outcomes.append("tie")
        else:
            baseline_outcomes.append(
                "win" if row[f"{winner}_objective"] == "masked_diffusion" else "loss"
            )
    baseline_values = [
        1.0 if value == "win" else 0.5 if value == "tie" else 0.0
        for value in baseline_outcomes
    ]
    baseline_mean, baseline_low, baseline_high = bootstrap_mean_interval(baseline_values)
    baseline_vs_ar = {
        "count": len(historical),
        "remask_wins": baseline_outcomes.count("win"),
        "ties": baseline_outcomes.count("tie"),
        "remask_losses": baseline_outcomes.count("loss"),
        "remask_utility": baseline_mean,
        "utility_ci95": [baseline_low, baseline_high],
        "source": "historical 64-step baseline judgments",
    }
    performance_rows = json.loads((args.output_dir / "performance.json").read_text())
    expected_performance = {
        (label, seed, batch)
        for label in ("ar-cached", "mdlm-baseline", "mdlm-remask")
        for seed in (1337, 2027)
        for batch in (1, 8, 32)
    }
    actual_performance = {
        (row["label"], int(row["train_seed"]), int(row["batch_size"]))
        for row in performance_rows
    }
    if actual_performance != expected_performance:
        raise SystemExit("performance rows do not match the expected decoder/seed/batch grid")
    frontier = render_frontier(
        judge_summary["remask-vs-ar"],
        baseline_vs_ar,
        performance_rows,
        args.output_dir,
    )
    summary = {
        "settings": settings,
        "generation_records": len(remask_records),
        "flash_judgments": len(flash),
        "pro_judgments": len(pro),
        "local_metrics": local_summary,
        "judge": judge_summary,
        "judge_by_seed": per_seed_summary,
        "pro_audit": pro_summary,
        "historical_baseline_vs_ar": baseline_vs_ar,
        "judge_agreement": agreement,
        "performance": performance_rows,
        "quality_latency_frontier": frontier,
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
            "prompt_seed": settings["prompt_seed"],
            "prompt_sha256": settings["prompt_sha256"],
            "selected_variant": settings["selected_variant"],
            "inference_dtype": settings["inference_dtype"],
            "training_seeds": [1337, 2027],
        },
    }
    atomic_json(args.output_dir / "public-summary.json", public_summary)
    lines = [
        "# TinyStories 106M Revisable Diffusion Decoding",
        "",
        f"- Remask generations: {len(remask_records):,}",
        f"- Flash judgments: {len(flash):,}",
        f"- Pro audit judgments: {len(pro):,}",
        "",
        "![Quality-latency frontier](remask_quality_latency_frontier.png)",
        "",
        "## Local metrics",
        "",
        "| Decoder | Ended | Unique tokens | Distinct-3 | Repeated 4-grams |",
        "|---|---:|---:|---:|---:|",
    ]
    for kind in ("ar", "baseline", "remask"):
        row = local_summary[kind]
        lines.append(
            f"| {kind} | {row['ended']:.3f} | {row['unique_token_ratio']:.3f} | "
            f"{row['distinct_3']:.3f} | {row['repeated_4gram_ratio']:.4f} |"
        )
    lines.extend(
        (
            "",
            "## Blinded quality",
            "",
            "| Comparison | N | Remask wins | Ties | Remask losses | Utility (95% CI) |",
            "|---|---:|---:|---:|---:|---:|",
        )
    )
    for comparison, row in judge_summary.items():
        lines.append(
            f"| {comparison} | {row['count']} | {row['remask_wins']} | {row['ties']} | "
            f"{row['remask_losses']} | {row['remask_utility']:.3f} "
            f"[{row['utility_ci95'][0]:.3f}, {row['utility_ci95'][1]:.3f}] |"
        )
    lines.append(
        f"| historical baseline-vs-ar | {baseline_vs_ar['count']} | "
        f"{baseline_vs_ar['remask_wins']} | {baseline_vs_ar['ties']} | "
        f"{baseline_vs_ar['remask_losses']} | {baseline_vs_ar['remask_utility']:.3f} "
        f"[{baseline_low:.3f}, {baseline_high:.3f}] |"
    )
    lines.extend(
        (
            "",
            "The baseline-vs-AR row reuses the earlier 64-step judgments and is marked historical "
            "because external judge behavior can change over time.",
            "",
            "## Pro audit",
            "",
            "| Comparison | N | Remask wins | Ties | Remask losses | Utility (95% CI) |",
            "|---|---:|---:|---:|---:|---:|",
        )
    )
    for comparison, row in pro_summary.items():
        lines.append(
            f"| {comparison} | {row['count']} | {row['remask_wins']} | {row['ties']} | "
            f"{row['remask_losses']} | {row['remask_utility']:.3f} "
            f"[{row['utility_ci95'][0]:.3f}, {row['utility_ci95'][1]:.3f}] |"
        )
    lines.extend(
        (
            "",
            "## Performance",
            "",
            "| Decoder | Seed | Batch | Latency | Sequences/s | First update | Peak memory |",
            "|---|---:|---:|---:|---:|---:|---:|",
        )
    )
    for row in sorted(
        performance_rows,
        key=lambda item: (item["label"], item["train_seed"], item["batch_size"]),
    ):
        lines.append(
            f"| {row['label']} | {row['train_seed']} | {row['batch_size']} | "
            f"{row['mean_seconds']:.3f} s | {row['sequences_per_second']:.2f} | "
            f"{row['first_update_ms']:.2f} ms | "
            f"{row['peak_memory_mb']:.0f} MiB |"
        )
    lines.append("")
    report_path = args.output_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(report_path)


def main() -> None:
    args = parser().parse_args()
    commands = {
        "prepare": prepare,
        "generate": generate,
        "performance": performance,
        "judge": judge,
        "report": report,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
