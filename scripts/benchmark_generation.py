"""Generate, time, judge, and report the TinyStories AR/MDLM comparison."""

# ruff: noqa: E402 -- make repository modules importable when executed as a script

from __future__ import annotations

import argparse
import json
import os
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

from config import ModelConfig
from diffusion import sample_masked
from evaluation import (
    JUDGE_DIMENSIONS,
    append_jsonl,
    bootstrap_mean_interval,
    cohens_kappa,
    judge_messages,
    load_validation_prompts,
    local_text_metrics,
    read_jsonl,
    sample_key,
    stable_seed,
    validate_judgment,
)
from experiment import atomic_json, environment_metadata
from model import Transformer

DEFAULT_OUTPUT = ROOT / "out/tinystories-106m-generation-eval"
DEFAULT_MANIFEST = ROOT / "data/tinystories-gpt2/manifest.json"
CHECKPOINTS = (
    ("ar-seed1337", "autoregressive", 1337, ROOT / "out/tinystories-106m-ar/best.pt"),
    ("mdlm-seed1337", "masked_diffusion", 1337, ROOT / "out/tinystories-106m-mdlm/best.pt"),
    (
        "ar-seed2027",
        "autoregressive",
        2027,
        ROOT / "out/tinystories-106m-ar-seed2027/best.pt",
    ),
    (
        "mdlm-seed2027",
        "masked_diffusion",
        2027,
        ROOT / "out/tinystories-106m-mdlm-seed2027/best.pt",
    ),
)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    root.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    subparsers = root.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--num-prompts", type=int, default=1000)
    prepare.add_argument("--prompt-length", type=int, default=32)
    prepare.add_argument("--output-length", type=int, default=128)
    prepare.add_argument("--seed", type=int, default=4242)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--batch-size", type=int, default=8)
    generate.add_argument("--steps", type=int, nargs="+", default=[8, 16, 32, 64])
    generate.add_argument("--temperature", type=float, default=1.0)
    generate.add_argument("--top-k", type=int, default=40)
    generate.add_argument("--limit", type=int)

    performance = subparsers.add_parser("performance")
    performance.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    performance.add_argument("--steps", type=int, nargs="+", default=[8, 16, 32, 64])
    performance.add_argument("--warmup", type=int, default=10)
    performance.add_argument("--repeats", type=int, default=30)
    performance.add_argument("--device", default="cuda:0")

    judge = subparsers.add_parser("judge")
    judge.add_argument("--flash-model", default="deepseek-v4-flash")
    judge.add_argument("--pro-model", default="deepseek-v4-pro")
    judge.add_argument("--pro-samples", type=int, default=100)
    judge.add_argument("--concurrency", type=int, default=32)
    judge.add_argument("--max-retries", type=int, default=5)
    judge.add_argument("--limit", type=int)

    subparsers.add_parser("report")
    return root


def tokenizer(manifest_path: Path):
    try:
        import tiktoken
    except ImportError as error:
        raise SystemExit('evaluation requires: pip install -e ".[data,eval,viz]"') from error
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return tiktoken.get_encoding(manifest["tokenizer"]["name"]), manifest


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[Transformer, ModelConfig]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ModelConfig(**checkpoint["config"]["model"])
    model = Transformer(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, config


def prepare(args: argparse.Namespace) -> None:
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
        args.output_dir / "evaluation.json",
        {
            "format": "nanodiffusion-generation-eval/1",
            "dataset": manifest["dataset"],
            "tokenizer": manifest["tokenizer"],
            "num_prompts": args.num_prompts,
            "prompt_length": args.prompt_length,
            "output_length": args.output_length,
            "prompt_seed": args.seed,
            "inference_dtype": "bfloat16",
            "mdlm_steps": [8, 16, 32, 64],
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
    print(f"wrote {len(prompts)} prompts to {prompt_path}")


def generation_seed(label: str, steps: int | None, prompt_id: int) -> int:
    return stable_seed("generation", label, steps or 0, prompt_id)


def generate(args: argparse.Namespace) -> None:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise SystemExit("generation benchmark requires CUDA")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    encoding, _manifest = tokenizer(args.manifest)
    settings = json.loads((args.output_dir / "evaluation.json").read_text())
    prompts = read_jsonl(args.output_dir / "prompts.jsonl")
    prompts = [prompt for prompt in prompts if prompt["prompt_id"] % world_size == rank]
    if args.limit is not None:
        prompts = prompts[: args.limit]
    output_path = args.output_dir / "generations" / f"rank{rank:02d}.jsonl"
    completed = {sample_key(record) for record in read_jsonl(output_path)}
    total_length = int(settings["prompt_length"]) + int(settings["output_length"])

    for label, objective, train_seed, checkpoint_path in CHECKPOINTS:
        model, model_config = load_model(checkpoint_path, device)
        step_values: list[int | None] = [None] if objective == "autoregressive" else args.steps
        for steps in step_values:
            pending = [
                prompt
                for prompt in prompts
                if f"{objective}:seed{train_seed}:steps{steps or 0}:p{prompt['prompt_id']}"
                not in completed
            ]
            for offset in range(0, len(pending), args.batch_size):
                batch = pending[offset : offset + args.batch_size]
                prompt_tokens = torch.tensor(
                    [prompt["token_ids"] for prompt in batch], dtype=torch.long, device=device
                )
                generators = [
                    torch.Generator(device=device).manual_seed(
                        generation_seed(label, steps, prompt["prompt_id"])
                    )
                    for prompt in batch
                ]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    if objective == "autoregressive":
                        result = model.generate(
                            prompt_tokens,
                            settings["output_length"],
                            args.temperature,
                            args.top_k,
                            use_cache=True,
                            generator=generators,
                        )
                        network_evaluations = settings["output_length"]
                    else:
                        if model_config.mask_token_id is None:
                            raise ValueError("MDLM checkpoint has no mask token")
                        initial = torch.full(
                            (len(batch), total_length),
                            model_config.mask_token_id,
                            dtype=torch.long,
                            device=device,
                        )
                        initial[:, : settings["prompt_length"]] = prompt_tokens
                        result = sample_masked(
                            model,
                            tuple(initial.shape),
                            model_config.mask_token_id,
                            steps=steps or 1,
                            temperature=args.temperature,
                            top_k=args.top_k,
                            initial_tokens=initial,
                            generator=generators,
                        )
                        network_evaluations = steps
                continuations = result[:, settings["prompt_length"] :].cpu().tolist()
                for prompt, token_ids in zip(batch, continuations, strict=True):
                    record = {
                        "prompt_id": prompt["prompt_id"],
                        "label": label,
                        "objective": objective,
                        "train_seed": train_seed,
                        "steps": steps,
                        "sample_seed": generation_seed(label, steps, prompt["prompt_id"]),
                        "network_evaluations": network_evaluations,
                        "prompt": prompt["text"],
                        "continuation": encoding.decode(token_ids),
                        "output_token_ids": token_ids,
                    }
                    append_jsonl(output_path, record)
                    completed.add(sample_key(record))
            print(f"rank {rank}: {label} steps={steps} complete ({len(pending)} new)")
        del model
        torch.cuda.empty_cache()


def timed_call(function, warmup: int, repeats: int, device: torch.device) -> dict[str, float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    timings = []
    for _ in range(repeats):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        function()
        torch.cuda.synchronize(device)
        timings.append(time.perf_counter() - start)
    return {
        "mean_seconds": statistics.mean(timings),
        "median_seconds": statistics.median(timings),
        "stdev_seconds": statistics.stdev(timings) if len(timings) > 1 else 0.0,
        "peak_memory_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
    }


def performance(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("performance benchmark requires a CUDA device")
    settings = json.loads((args.output_dir / "evaluation.json").read_text())
    prompts = read_jsonl(args.output_dir / "prompts.jsonl")
    records = []
    for label, objective, train_seed, checkpoint_path in CHECKPOINTS:
        model, model_config = load_model(checkpoint_path, device)
        variants: list[tuple[int | None, bool | None]] = (
            [(None, True), (None, False)]
            if objective == "autoregressive"
            else [(steps, None) for steps in args.steps]
        )
        for batch_size in args.batch_sizes:
            prompt_rows = [
                prompts[index % len(prompts)]["token_ids"] for index in range(batch_size)
            ]
            prompt_tokens = torch.tensor(prompt_rows, dtype=torch.long, device=device)
            for steps, use_cache in variants:
                generators = [
                    torch.Generator(device=device).manual_seed(generation_seed(label, steps, index))
                    for index in range(batch_size)
                ]
                if objective == "autoregressive":

                    def run(
                        model: Transformer = model,
                        prompt_tokens: torch.Tensor = prompt_tokens,
                        use_cache: bool | None = use_cache,
                        generators=generators,
                    ) -> torch.Tensor:
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            return model.generate(
                                prompt_tokens,
                                settings["output_length"],
                                top_k=1,
                                use_cache=bool(use_cache),
                                generator=generators,
                            )

                    network_evaluations = settings["output_length"]
                    variant = "cached" if use_cache else "uncached"

                    def first_update(
                        model: Transformer = model,
                        prompt_tokens: torch.Tensor = prompt_tokens,
                    ) -> torch.Tensor:
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            return model(prompt_tokens)

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
                        model: Transformer = model,
                        initial: torch.Tensor = initial,
                        mask_token_id: int = model_config.mask_token_id,
                        steps: int | None = steps,
                        generators=generators,
                    ) -> torch.Tensor:
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            return sample_masked(
                                model,
                                tuple(initial.shape),
                                mask_token_id,
                                steps=steps or 1,
                                top_k=1,
                                initial_tokens=initial,
                                generator=generators,
                            )

                    network_evaluations = steps
                    variant = f"steps-{steps}"

                    def first_update(
                        model: Transformer = model,
                        initial: torch.Tensor = initial,
                        mask_token_id: int = model_config.mask_token_id,
                    ) -> torch.Tensor:
                        t = initial.eq(mask_token_id).float().mean(dim=1)
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            return model(initial, t, initial.eq(mask_token_id))

                measured = timed_call(run, args.warmup, args.repeats, device)
                first_update = timed_call(
                    first_update,
                    args.warmup,
                    args.repeats,
                    device,
                )
                measured.update(
                    {
                        "label": label,
                        "objective": objective,
                        "train_seed": train_seed,
                        "variant": variant,
                        "steps": steps,
                        "batch_size": batch_size,
                        "network_evaluations": network_evaluations,
                        "output_tokens_per_second": (
                            batch_size * settings["output_length"] / measured["mean_seconds"]
                        ),
                        "sequences_per_second": batch_size / measured["mean_seconds"],
                        "first_update_ms": first_update["mean_seconds"] * 1000,
                    }
                )
                records.append(measured)
                print(label, variant, f"batch={batch_size}", measured["mean_seconds"])
        del model
        torch.cuda.empty_cache()
    atomic_json(args.output_dir / "performance.json", records)


def generation_records(output_dir: Path) -> list[dict[str, Any]]:
    return [
        record
        for path in sorted((output_dir / "generations").glob("rank*.jsonl"))
        for record in read_jsonl(path)
    ]


def build_pairs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {sample_key(record): record for record in records}
    pairs = []
    for record in records:
        if record["objective"] != "masked_diffusion":
            continue
        ar_key = f"autoregressive:seed{record['train_seed']}:steps0:p{record['prompt_id']}"
        ar = lookup.get(ar_key)
        if ar is None:
            continue
        pair_id = f"seed{record['train_seed']}:steps{record['steps']}:p{record['prompt_id']}"
        swapped = stable_seed("side", pair_id) % 2 == 1
        first, second = (record, ar) if swapped else (ar, record)
        pairs.append(
            {
                "pair_id": pair_id,
                "prompt_id": record["prompt_id"],
                "train_seed": record["train_seed"],
                "steps": record["steps"],
                "prompt": record["prompt"],
                "A_label": first["label"],
                "A_objective": first["objective"],
                "A_text": first["continuation"],
                "B_label": second["label"],
                "B_objective": second["objective"],
                "B_text": second["continuation"],
            }
        )
    return sorted(pairs, key=lambda pair: pair["pair_id"])


def pro_subset(pairs: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    by_steps: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        by_steps[int(pair["steps"])].append(pair)
    selected = []
    base, remainder = divmod(count, len(by_steps))
    for group_index, steps in enumerate(sorted(by_steps)):
        amount = base + int(group_index < remainder)
        ranked = sorted(by_steps[steps], key=lambda pair: stable_seed("pro-audit", pair["pair_id"]))
        selected.extend(ranked[:amount])
    return selected


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
                    "user_id": "nanodiffusionlab-eval",
                },
            )
            raw = response.choices[0].message.content or ""
            judgment = validate_judgment(json.loads(raw))
            usage = response.usage.model_dump() if response.usage is not None else None
            return {
                "pair_id": pair["pair_id"],
                "train_seed": pair["train_seed"],
                "steps": pair["steps"],
                "A_label": pair["A_label"],
                "A_objective": pair["A_objective"],
                "B_label": pair["B_label"],
                "B_objective": pair["B_objective"],
                "judge_model_requested": model,
                "judge_model_returned": response.model,
                "judgment": judgment,
                "usage": usage,
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


def judge(args: argparse.Namespace) -> None:
    pairs = build_pairs(generation_records(args.output_dir))
    if args.limit is not None:
        pairs = pairs[: args.limit]
    expected = len(read_jsonl(args.output_dir / "prompts.jsonl")) * 2 * 4
    if args.limit is None and len(pairs) != expected:
        raise SystemExit(f"expected {expected} complete pairs, found {len(pairs)}")
    run_judge_model(
        pairs,
        args.output_dir / "judgments-flash.jsonl",
        args.flash_model,
        args.concurrency,
        args.max_retries,
    )
    audit = pro_subset(pairs, min(args.pro_samples, len(pairs)))
    run_judge_model(
        audit,
        args.output_dir / "judgments-pro.jsonl",
        args.pro_model,
        min(args.concurrency, 16),
        args.max_retries,
    )


def winner_for_objective(record: dict[str, Any], objective: str) -> str:
    winner = record["judgment"]["winner"]
    if winner == "tie":
        return "tie"
    return "win" if record[f"{winner}_objective"] == objective else "loss"


def render_frontier(
    flash: list[dict[str, Any]], performance_rows: list[dict[str, Any]], output_dir: Path
) -> list[dict[str, float | int | str]]:
    """Plot batch-one latency against pairwise quality for the report."""
    if not flash or not performance_rows:
        return []
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit('report figures require: pip install -e ".[viz]"') from error
    latency: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in performance_rows:
        if row["batch_size"] != 1 or row["variant"] == "uncached":
            continue
        steps = int(row.get("steps") or 0)
        latency[(row["objective"], steps)].append(float(row["mean_seconds"]))
    utilities: dict[int, list[float]] = defaultdict(list)
    for row in flash:
        outcome = winner_for_objective(row, "masked_diffusion")
        utilities[int(row["steps"])].append(
            1.0 if outcome == "win" else 0.5 if outcome == "tie" else 0.0
        )
    points: list[dict[str, float | int | str]] = []
    if latency.get(("autoregressive", 0)):
        points.append(
            {
                "label": "AR cached",
                "steps": 0,
                "latency_seconds": statistics.mean(latency[("autoregressive", 0)]),
                "utility": 0.5,
                "ci_low": 0.5,
                "ci_high": 0.5,
            }
        )
    for steps in sorted(utilities):
        if not latency.get(("masked_diffusion", steps)):
            continue
        mean, low, high = bootstrap_mean_interval(utilities[steps])
        points.append(
            {
                "label": f"MDLM {steps} steps",
                "steps": steps,
                "latency_seconds": statistics.mean(latency[("masked_diffusion", steps)]),
                "utility": mean,
                "ci_low": low,
                "ci_high": high,
            }
        )
    if not points:
        return []
    figure, axis = plt.subplots(figsize=(8.4, 5.2))
    for point in points:
        color = "#2563eb" if point["steps"] == 0 else "#ea580c"
        marker = "s" if point["steps"] == 0 else "o"
        utility = float(point["utility"])
        axis.errorbar(
            float(point["latency_seconds"]),
            utility,
            yerr=[
                [utility - float(point["ci_low"])],
                [float(point["ci_high"]) - utility],
            ],
            marker=marker,
            color=color,
            capsize=4,
            markersize=8,
        )
        axis.annotate(
            str(point["label"]),
            (float(point["latency_seconds"]), utility),
            xytext=(7, 7),
            textcoords="offset points",
        )
    axis.axhline(0.5, color="#64748b", linestyle="--", linewidth=1)
    axis.set_xscale("log")
    axis.set_xlabel("Batch-1 latency per 128-token continuation (seconds, log scale)")
    axis.set_ylabel("MDLM pairwise utility versus same-seed AR")
    axis.set_title("TinyStories 106M quality–latency frontier", loc="left", weight="bold")
    axis.grid(axis="both", color="#e2e8f0", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(output_dir / "quality_latency_frontier.png", dpi=180, facecolor="white")
    plt.close(figure)
    return points


def report(args: argparse.Namespace) -> None:
    records = generation_records(args.output_dir)
    flash = read_jsonl(args.output_dir / "judgments-flash.jsonl")
    pro = read_jsonl(args.output_dir / "judgments-pro.jsonl")
    settings = json.loads((args.output_dir / "evaluation.json").read_text())
    eot = int(settings["tokenizer"]["eot_token_id"])
    local_groups: dict[str, list[dict[str, float]]] = defaultdict(list)
    for record in records:
        local_groups[record["label"] + f":steps{record.get('steps') or 0}"].append(
            local_text_metrics(record["output_token_ids"], eot)
        )
    local_summary = {
        group: {metric: statistics.mean(row[metric] for row in rows) for metric in rows[0]}
        for group, rows in local_groups.items()
    }
    judge_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in flash:
        judge_groups[f"seed{record['train_seed']}:steps{record['steps']}"].append(record)
    judge_summary = {}
    for group, rows in judge_groups.items():
        outcomes = [winner_for_objective(row, "masked_diffusion") for row in rows]
        utility = [1.0 if value == "win" else 0.5 if value == "tie" else 0.0 for value in outcomes]
        mean, low, high = bootstrap_mean_interval(utility)
        dimensions = {}
        for dimension in JUDGE_DIMENSIONS:
            deltas = []
            for row in rows:
                md_side = "A" if row["A_objective"] == "masked_diffusion" else "B"
                ar_side = "B" if md_side == "A" else "A"
                deltas.append(
                    row["judgment"]["scores"][md_side][dimension]
                    - row["judgment"]["scores"][ar_side][dimension]
                )
            dimensions[dimension] = bootstrap_mean_interval(deltas)
        judge_summary[group] = {
            "count": len(rows),
            "mdlm_wins": outcomes.count("win"),
            "ties": outcomes.count("tie"),
            "ar_wins": outcomes.count("loss"),
            "mdlm_pairwise_utility": mean,
            "utility_ci95": [low, high],
            "dimension_deltas_mdlm_minus_ar": dimensions,
        }
    flash_lookup = {record["pair_id"]: record for record in flash}
    matched = [(flash_lookup[row["pair_id"]], row) for row in pro if row["pair_id"] in flash_lookup]
    agreement = None
    if matched:
        flash_labels = [row[0]["judgment"]["winner"] for row in matched]
        pro_labels = [row[1]["judgment"]["winner"] for row in matched]
        agreement = {
            "count": len(matched),
            "exact": sum(a == b for a, b in zip(flash_labels, pro_labels, strict=True))
            / len(matched),
            "cohens_kappa": cohens_kappa(flash_labels, pro_labels),
        }
    performance_rows = (
        json.loads((args.output_dir / "performance.json").read_text())
        if (args.output_dir / "performance.json").exists()
        else []
    )
    frontier = render_frontier(flash, performance_rows, args.output_dir)
    summary = {
        "settings": settings,
        "generation_records": len(records),
        "flash_judgments": len(flash),
        "pro_judgments": len(pro),
        "local_metrics": local_summary,
        "judge": judge_summary,
        "judge_agreement": agreement,
        "performance": performance_rows,
        "quality_latency_frontier": frontier,
    }
    atomic_json(args.output_dir / "summary.json", summary)
    lines = [
        "# TinyStories 106M Generation Benchmark",
        "",
        f"- Generated samples: {len(records):,}",
        f"- DeepSeek V4 Flash judgments: {len(flash):,}",
        f"- DeepSeek V4 Pro audit judgments: {len(pro):,}",
    ]
    if frontier:
        lines.extend(("", "![Quality-latency frontier](quality_latency_frontier.png)"))
    lines.extend(
        (
            "",
            "## Pairwise quality: MDLM versus same-seed AR",
            "",
            "| Configuration | N | MDLM wins | Ties | AR wins | MDLM utility (95% CI) |",
            "|---|---:|---:|---:|---:|---:|",
        )
    )
    for group, row in sorted(judge_summary.items()):
        low, high = row["utility_ci95"]
        lines.append(
            f"| {group} | {row['count']} | {row['mdlm_wins']} | {row['ties']} | "
            f"{row['ar_wins']} | {row['mdlm_pairwise_utility']:.3f} "
            f"[{low:.3f}, {high:.3f}] |"
        )
    if agreement:
        lines.extend(
            (
                "",
                "## Judge agreement",
                "",
                f"Flash and Pro exact agreement: {agreement['exact']:.1%}; "
                f"Cohen's kappa: {agreement['cohens_kappa']:.3f} (N={agreement['count']}).",
            )
        )
    lines.extend(
        (
            "",
            "> Judge scores are external, version-sensitive measurements. Local diversity and "
            "repetition metrics are retained in summary.json as independent guardrails.",
            "",
        )
    )
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(args.output_dir / "report.md")


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
