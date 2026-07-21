"""Train an autoregressive or masked-diffusion Transformer."""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from config import TrainConfig, load_config
from data import CharacterData, TokenizedData
from diffusion import corrupt_tokens, masked_diffusion_loss
from experiment import (
    JsonlLogger,
    atomic_json,
    atomic_torch_save,
    environment_metadata,
    snapshot_sources,
)
from model import Transformer


def setup_distributed(config: TrainConfig) -> tuple[bool, int, int, int, torch.device]:
    distributed = int(os.environ.get("RANK", -1)) >= 0
    if distributed:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        backend = config.backend if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
        dist.init_process_group(backend=backend)
    else:
        rank, local_rank, world_size = 0, 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return distributed, rank, local_rank, world_size, device


def learning_rate(step: int, total_steps: int, config: TrainConfig) -> float:
    if step < config.warmup_iters:
        return config.learning_rate * (step + 1) / max(config.warmup_iters, 1)
    if step >= total_steps:
        return config.min_lr
    ratio = (step - config.warmup_iters) / max(total_steps - config.warmup_iters, 1)
    coefficient = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return config.min_lr + coefficient * (config.learning_rate - config.min_lr)


def objective_loss(
    model: torch.nn.Module, clean: torch.Tensor, objective: str, mask_token_id: int | None
) -> tuple[torch.Tensor, dict[str, float]]:
    if objective == "masked_diffusion":
        if mask_token_id is None:
            raise ValueError("masked diffusion requires mask_token_id")
        return masked_diffusion_loss(model, clean, mask_token_id)
    inputs, targets = clean[:, :-1], clean[:, 1:]
    logits = model(inputs)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    accuracy = (logits.argmax(dim=-1) == targets).float().mean()
    return loss, {
        "token_accuracy": float(accuracy.detach()),
        "target_tokens": float(targets.numel()),
        "correct_tokens": float((logits.argmax(dim=-1) == targets).sum()),
    }


def reduce_sums(values: torch.Tensor, distributed: bool) -> torch.Tensor:
    if distributed:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return values


def distributed_barrier(distributed: bool, device: torch.device) -> None:
    if distributed:
        if device.type == "cuda":
            dist.barrier(device_ids=[device.index])
        else:
            dist.barrier()


@torch.no_grad()
def estimate_metrics(
    model: torch.nn.Module,
    data: CharacterData | TokenizedData,
    config: TrainConfig,
    autocast: Callable[[], Any],
    device: torch.device,
    rank: int,
    distributed: bool,
) -> dict[str, float]:
    model.eval()
    data_generator = torch.Generator().manual_seed(config.eval_seed + rank)
    noise_generator = torch.Generator(device=device).manual_seed(config.eval_seed + 10_000 + rank)
    objective = config.model.objective
    if objective == "autoregressive":
        totals = torch.zeros(3, dtype=torch.float64, device=device)
        for _ in range(config.eval_iters):
            split = "validation" if config.data_format == "tokenized" else "val"
            clean = data.batch(split, config.batch_size, generator=data_generator)
            inputs, targets = clean[:, :-1], clean[:, 1:]
            with autocast():
                logits = model(inputs)
                loss_sum = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="sum"
                )
            totals += torch.stack(
                (
                    loss_sum.double(),
                    (logits.argmax(dim=-1) == targets).sum().double(),
                    torch.tensor(targets.numel(), dtype=torch.float64, device=device),
                )
            )
        totals = reduce_sums(totals, distributed)
        loss = float(totals[0] / totals[2])
        result = {
            "val_loss": loss,
            "val_perplexity": math.exp(min(loss, 20.0)),
            "val_bits_per_token": loss / math.log(2),
            "val_token_accuracy": float(totals[1] / totals[2]),
        }
    else:
        if config.model.mask_token_id is None:
            raise ValueError("masked diffusion requires mask_token_id")
        result = {}
        losses = []
        accuracies = []
        for ratio in config.eval_mask_ratios:
            totals = torch.zeros(3, dtype=torch.float64, device=device)
            for _ in range(config.eval_iters):
                split = "validation" if config.data_format == "tokenized" else "val"
                clean = data.batch(split, config.batch_size, generator=data_generator)
                t = torch.full((clean.size(0),), ratio, device=device)
                noisy, mask, sampled_t = corrupt_tokens(
                    clean, config.model.mask_token_id, t, generator=noise_generator
                )
                with autocast():
                    logits = model(noisy, sampled_t, mask)
                    targets = clean[mask]
                    loss_sum = F.cross_entropy(logits, targets, reduction="sum")
                totals += torch.stack(
                    (
                        loss_sum.double(),
                        (logits.argmax(dim=-1) == targets).sum().double(),
                        torch.tensor(targets.numel(), dtype=torch.float64, device=device),
                    )
                )
            totals = reduce_sums(totals, distributed)
            ratio_loss = float(totals[0] / totals[2])
            ratio_accuracy = float(totals[1] / totals[2])
            key = f"{ratio:.2f}"
            result[f"val_loss_mask_{key}"] = ratio_loss
            result[f"val_accuracy_mask_{key}"] = ratio_accuracy
            losses.append(ratio_loss)
            accuracies.append(ratio_accuracy)
        result["val_loss"] = sum(losses) / len(losses)
        result["val_masked_accuracy"] = sum(accuracies) / len(accuracies)
    model.train()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/shakespeare_char.py")
    parser.add_argument("--data", help="override the text file or token manifest")
    parser.add_argument("--out-dir")
    parser.add_argument("--run-name")
    parser.add_argument("--objective", choices=("autoregressive", "masked_diffusion"))
    parser.add_argument("--max-iters", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--eval-iters", type=int)
    parser.add_argument("--resume", nargs="?", const="auto")
    parser.add_argument("--no-compile", action="store_true")
    return parser.parse_args()


def apply_overrides(config: TrainConfig, args: argparse.Namespace) -> None:
    for argument, attribute in (
        ("data", "data_path"),
        ("out_dir", "out_dir"),
        ("run_name", "run_name"),
        ("max_iters", "max_iters"),
        ("max_tokens", "max_tokens"),
        ("batch_size", "batch_size"),
        ("gradient_accumulation_steps", "gradient_accumulation_steps"),
        ("eval_interval", "eval_interval"),
        ("eval_iters", "eval_iters"),
    ):
        value = getattr(args, argument)
        if value is not None:
            setattr(config, attribute, value)
    if args.no_compile:
        config.compile = False
    if args.objective is not None:
        config.model.objective = args.objective
        if args.objective == "autoregressive":
            config.model.mask_token_id = None
            config.model.time_conditioning = False
            if args.out_dir is None:
                config.out_dir = config.out_dir.replace("mdlm", "ar")
        else:
            config.model.mask_token_id = config.model.vocab_size
            config.model.time_conditioning = True


def capture_rank_state(device: torch.device, train_generator: torch.Generator) -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "train_generator": train_generator.get_state(),
    }
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def restore_rank_state(
    state: dict[str, Any], device: torch.device, train_generator: torch.Generator
) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    train_generator.set_state(state["train_generator"].cpu())
    if device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)


def gather_rank_states(
    local_state: dict[str, Any], distributed: bool, world_size: int
) -> list[dict[str, Any]]:
    if not distributed:
        return [local_state]
    states: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(states, local_state)
    return [state for state in states if state is not None]


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    apply_overrides(config, args)
    distributed, rank, _, world_size, device = setup_distributed(config)
    master = rank == 0
    seed = config.seed + rank
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    out_dir = Path(config.out_dir)
    if master:
        out_dir.mkdir(parents=True, exist_ok=True)
    distributed_barrier(distributed, device)

    if config.data_format == "char":
        data: CharacterData | TokenizedData = CharacterData(
            config.data_path, config.model.block_size, device
        )
        if config.model.vocab_size != data.vocab_size:
            if master:
                print(f"setting vocab_size={data.vocab_size} from dataset")
            config.model.vocab_size = data.vocab_size
            if config.model.objective == "masked_diffusion":
                config.model.mask_token_id = data.vocab_size
    elif config.data_format == "tokenized":
        data = TokenizedData(config.data_path, config.model.block_size, device)
        if config.model.vocab_size != data.vocab_size:
            raise ValueError(
                f"model vocab_size={config.model.vocab_size} != "
                f"tokenizer vocab_size={data.vocab_size}"
            )
    else:
        raise ValueError(f"unknown data format: {config.data_format}")
    if master:
        data.save_metadata(out_dir)

    raw_model = Transformer(config.model, config.gradient_checkpointing).to(device)
    fused_available = "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
        fused=fused_available and device.type == "cuda",
    )
    amp_dtype = (
        {"float16": torch.float16, "bfloat16": torch.bfloat16}[config.dtype]
        if config.dtype != "float32"
        else None
    )

    def autocast() -> Any:
        if amp_dtype is not None and device.type in {"cuda", "cpu"}:
            return torch.autocast(device_type=device.type, dtype=amp_dtype)
        return nullcontext()

    scaler = torch.amp.GradScaler(
        "cuda", enabled=config.dtype == "float16" and device.type == "cuda"
    )
    train_generator = torch.Generator().manual_seed(seed + 1_000)
    step = 0
    tokens_seen = 0
    target_tokens_seen = 0
    best_val = float("inf")
    training_seconds = 0.0
    peak_memory_mb = 0.0
    resume_path: Path | None = None
    if args.resume is not None:
        resume_path = out_dir / "last.pt" if args.resume == "auto" else Path(args.resume)
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        if checkpoint["config"]["model"] != config.to_dict()["model"]:
            raise ValueError("checkpoint model configuration does not match requested model")
        if checkpoint["world_size"] != world_size:
            raise ValueError("exact resume requires the same world_size")
        raw_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        step = int(checkpoint["step"])
        tokens_seen = int(checkpoint["tokens_seen"])
        target_tokens_seen = int(checkpoint["target_tokens_seen"])
        best_val = float(checkpoint["best_val"])
        training_seconds = float(checkpoint.get("training_seconds", 0.0))
        peak_memory_mb = float(checkpoint.get("peak_memory_mb", 0.0))
        restore_rank_state(checkpoint["rank_states"][rank], device, train_generator)

    model: torch.nn.Module = raw_model
    if config.compile:
        model = torch.compile(model)
    if distributed:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)

    input_tokens_per_step = (
        config.batch_size
        * config.model.block_size
        * config.gradient_accumulation_steps
        * world_size
    )
    token_steps = (
        math.ceil(config.max_tokens / input_tokens_per_step)
        if config.max_tokens is not None
        else config.max_iters
    )
    total_steps = min(config.max_iters, token_steps)
    logger = JsonlLogger(out_dir / "metrics.jsonl") if master else None
    started = time.time()
    if master:
        source_files = snapshot_sources(out_dir / "source")
        run_record = {
            "run_name": config.run_name,
            "config": config.to_dict(),
            "data": data.description(),
            "environment": environment_metadata(),
            "parameter_count": raw_model.parameter_count(),
            "input_tokens_per_step": input_tokens_per_step,
            "planned_steps": total_steps,
            "resumed_from": str(resume_path) if resume_path else None,
            "source_snapshot": source_files,
        }
        atomic_json(out_dir / "run.json", run_record)
        print(
            f"{config.model.objective} | {raw_model.parameter_count() / 1e6:.2f}M params | "
            f"device={device} | world_size={world_size} | tokens/step={input_tokens_per_step:,} | "
            f"planned_steps={total_steps:,}"
        )

    def save_checkpoint(path: Path) -> None:
        rank_states = gather_rank_states(
            capture_rank_state(device, train_generator), distributed, world_size
        )
        if master:
            atomic_torch_save(
                path,
                {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "config": config.to_dict(),
                    "step": step,
                    "tokens_seen": tokens_seen,
                    "target_tokens_seen": target_tokens_seen,
                    "best_val": best_val,
                    "training_seconds": training_seconds,
                    "peak_memory_mb": peak_memory_mb,
                    "world_size": world_size,
                    "rank_states": rank_states,
                },
            )
        distributed_barrier(distributed, device)

    last_milestone = max(
        (value for value in config.milestone_tokens if value <= tokens_seen), default=0
    )
    while step < total_steps:
        if step % config.eval_interval == 0:
            metrics = estimate_metrics(model, data, config, autocast, device, rank, distributed)
            if master:
                event = {
                    "event": "eval",
                    "time": time.time(),
                    "step": step,
                    "tokens_seen": tokens_seen,
                    **metrics,
                }
                logger.write(event)  # type: ignore[union-attr]
                print(f"step {step}: tokens={tokens_seen:,} val_loss={metrics['val_loss']:.4f}")
                improved = metrics["val_loss"] < best_val
                best_val = min(best_val, metrics["val_loss"])
            else:
                improved = False
            if distributed:
                improved_tensor = torch.tensor(int(improved), device=device)
                dist.broadcast(improved_tensor, src=0)
                improved = bool(improved_tensor.item())
            save_checkpoint(out_dir / "last.pt")
            if improved or config.always_save_checkpoint:
                save_checkpoint(out_dir / "best.pt")

        lr = learning_rate(step, total_steps, config)
        for group in optimizer.param_groups:
            group["lr"] = lr
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        local_target_tokens = 0
        local_correct_tokens = 0
        local_loss = 0.0
        for micro_step in range(config.gradient_accumulation_steps):
            clean = data.batch("train", config.batch_size, generator=train_generator)
            sync = not distributed or micro_step == config.gradient_accumulation_steps - 1
            context = nullcontext() if sync else model.no_sync()  # type: ignore[attr-defined]
            with context, autocast():
                loss, batch_metrics = objective_loss(
                    model, clean, config.model.objective, config.model.mask_token_id
                )
                scaled_loss = loss / config.gradient_accumulation_steps
            local_loss += float(scaled_loss.detach())
            local_target_tokens += int(batch_metrics["target_tokens"])
            local_correct_tokens += int(batch_metrics["correct_tokens"])
            scaler.scale(scaled_loss).backward()
        if config.grad_clip:
            scaler.unscale_(optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip))
        else:
            grad_norm = float("nan")
        scaler.step(optimizer)
        scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - step_started
        training_seconds += elapsed
        if device.type == "cuda":
            peak_memory_mb = max(peak_memory_mb, torch.cuda.max_memory_allocated(device) / 1024**2)
        aggregate = torch.tensor(
            [local_loss, local_target_tokens, local_correct_tokens],
            dtype=torch.float64,
            device=device,
        )
        aggregate = reduce_sums(aggregate, distributed)
        mean_loss = float(aggregate[0] / world_size)
        global_target_tokens = int(aggregate[1])
        global_correct_tokens = int(aggregate[2])
        previous_tokens = tokens_seen
        step += 1
        tokens_seen += input_tokens_per_step
        target_tokens_seen += global_target_tokens

        if master and (step == 1 or step % config.log_interval == 0):
            event = {
                "event": "train",
                "time": time.time(),
                "step": step,
                "tokens_seen": tokens_seen,
                "target_tokens_seen": target_tokens_seen,
                "loss": mean_loss,
                "learning_rate": lr,
                "grad_norm": grad_norm,
                "step_seconds": elapsed,
                "tokens_per_second": input_tokens_per_step / elapsed,
                "peak_memory_mb": peak_memory_mb,
            }
            accuracy_key = (
                "masked_accuracy"
                if config.model.objective == "masked_diffusion"
                else "token_accuracy"
            )
            event[accuracy_key] = global_correct_tokens / global_target_tokens
            if config.model.objective == "masked_diffusion":
                event["mask_ratio"] = global_target_tokens / input_tokens_per_step
            logger.write(event)  # type: ignore[union-attr]
            print(
                f"iter {step}: loss={mean_loss:.4f} lr={lr:.2e} "
                f"tokens={tokens_seen:,} tok/s={event['tokens_per_second']:,.0f}"
            )

        crossed = [
            value
            for value in config.milestone_tokens
            if previous_tokens < value <= tokens_seen and value > last_milestone
        ]
        for milestone in crossed:
            save_checkpoint(out_dir / f"tokens-{milestone:012d}.pt")
            last_milestone = milestone

    metrics = estimate_metrics(model, data, config, autocast, device, rank, distributed)
    if master:
        logger.write(  # type: ignore[union-attr]
            {
                "event": "eval",
                "time": time.time(),
                "step": step,
                "tokens_seen": tokens_seen,
                **metrics,
            }
        )
        final_improved = metrics["val_loss"] < best_val
        best_val = min(best_val, metrics["val_loss"])
    else:
        final_improved = False
    if distributed:
        improved_tensor = torch.tensor(int(final_improved), device=device)
        dist.broadcast(improved_tensor, src=0)
        final_improved = bool(improved_tensor.item())
    save_checkpoint(out_dir / "last.pt")
    if final_improved:
        save_checkpoint(out_dir / "best.pt")
    if master:
        atomic_json(
            out_dir / "summary.json",
            {
                "objective": config.model.objective,
                "parameter_count": raw_model.parameter_count(),
                "step": step,
                "tokens_seen": tokens_seen,
                "target_tokens_seen": target_tokens_seen,
                "best_val_loss": best_val,
                "final_metrics": metrics,
                "elapsed_seconds": time.time() - started,
                "training_seconds": training_seconds,
                "mean_training_tokens_per_second": tokens_seen / max(training_seconds, 1e-9),
                "peak_memory_mb": peak_memory_mb,
            },
        )
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
