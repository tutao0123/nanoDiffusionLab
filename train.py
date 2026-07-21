"""Train a character-level autoregressive or masked-diffusion Transformer."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from config import TrainConfig, load_config
from diffusion import masked_diffusion_loss
from model import Transformer


class CharacterData:
    """Small in-memory character dataset; intentionally simple for teaching."""

    def __init__(self, path: str | Path, block_size: int, device: torch.device) -> None:
        text = Path(path).read_text(encoding="utf-8")
        chars = sorted(set(text))
        if len(text) < block_size + 2:
            raise ValueError(f"dataset needs at least {block_size + 2} characters")
        self.stoi = {char: index for index, char in enumerate(chars)}
        self.itos = chars
        self.tokens = torch.tensor([self.stoi[char] for char in text], dtype=torch.long)
        self.split = int(0.9 * len(self.tokens))
        self.block_size = block_size
        self.device = device

    def batch(self, split: str, batch_size: int) -> torch.Tensor:
        data = self.tokens[: self.split] if split == "train" else self.tokens[self.split :]
        if len(data) < self.block_size + 1:
            data = self.tokens
        starts = torch.randint(len(data) - self.block_size, (batch_size,))
        batch = torch.stack([data[i : i + self.block_size] for i in starts])
        return batch.to(self.device, non_blocking=True)

    def save_metadata(self, out_dir: Path) -> None:
        payload = {"stoi": self.stoi, "itos": self.itos}
        (out_dir / "meta.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )


def setup_distributed(config: TrainConfig) -> tuple[bool, int, int, int, torch.device]:
    distributed = int(os.environ.get("RANK", -1)) >= 0
    if distributed:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        backend = config.backend if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
    else:
        rank, local_rank, world_size = 0, 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return distributed, rank, local_rank, world_size, device


def learning_rate(iteration: int, config: TrainConfig) -> float:
    if iteration < config.warmup_iters:
        return config.learning_rate * (iteration + 1) / max(config.warmup_iters, 1)
    if iteration >= config.max_iters:
        return config.min_lr
    ratio = (iteration - config.warmup_iters) / max(config.max_iters - config.warmup_iters, 1)
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
    return loss, {}


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module, data: CharacterData, config: TrainConfig, autocast: Any
) -> dict[str, float]:
    model.eval()
    result: dict[str, float] = {}
    for split in ("train", "val"):
        losses = []
        for _ in range(config.eval_iters):
            clean = data.batch(split, config.batch_size)
            with autocast():
                loss, _ = objective_loss(
                    model, clean, config.model.objective, config.model.mask_token_id
                )
            losses.append(float(loss))
        result[split] = sum(losses) / len(losses)
    model.train()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/shakespeare_char.py")
    parser.add_argument("--data", help="override the character text file")
    parser.add_argument("--out-dir")
    parser.add_argument("--max-iters", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.data:
        config.data_path = args.data
    if args.out_dir:
        config.out_dir = args.out_dir
    if args.max_iters is not None:
        config.max_iters = args.max_iters

    distributed, rank, _, world_size, device = setup_distributed(config)
    master = rank == 0
    seed = config.seed + rank
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    out_dir = Path(config.out_dir)
    if master:
        out_dir.mkdir(parents=True, exist_ok=True)
    if config.data_format != "char":
        raise NotImplementedError(
            "tokenized/sharded data is a planned milestone; use data_format='char' for now"
        )
    data = CharacterData(config.data_path, config.model.block_size, device)
    if config.model.vocab_size != len(data.itos):
        if master:
            print(f"setting vocab_size={len(data.itos)} from dataset")
        config.model.vocab_size = len(data.itos)
        if config.model.objective == "masked_diffusion":
            config.model.mask_token_id = len(data.itos)
    if master:
        data.save_metadata(out_dir)

    raw_model = Transformer(config.model, config.gradient_checkpointing).to(device)
    model = raw_model
    if config.compile:
        model = torch.compile(model)
    if distributed:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)

    fused_available = "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
    optimizer = torch.optim.AdamW(
        model.parameters(),
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

    def autocast():
        if amp_dtype is not None and device.type in {"cuda", "cpu"}:
            return torch.autocast(device_type=device.type, dtype=amp_dtype)
        return nullcontext()

    scaler = torch.amp.GradScaler(
        "cuda", enabled=config.dtype == "float16" and device.type == "cuda"
    )

    if master:
        effective = config.batch_size * config.gradient_accumulation_steps * world_size
        print(
            f"{config.model.objective} | {raw_model.parameter_count() / 1e6:.2f}M params | "
            f"device={device} | world_size={world_size} | effective_batch={effective}"
        )

    best_val = float("inf")
    started = time.time()
    for iteration in range(config.max_iters + 1):
        lr = learning_rate(iteration, config)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if iteration % config.eval_interval == 0:
            metrics = estimate_loss(model, data, config, autocast)
            if master:
                print(f"step {iteration}: train {metrics['train']:.4f}, val {metrics['val']:.4f}")
                should_save = metrics["val"] < best_val or config.always_save_checkpoint
                if should_save:
                    best_val = min(best_val, metrics["val"])
                    torch.save(
                        {
                            "model": raw_model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "config": config.to_dict(),
                            "iteration": iteration,
                            "best_val": best_val,
                        },
                        out_dir / "ckpt.pt",
                    )
            if distributed:
                dist.barrier()
        if iteration == config.max_iters:
            break

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_metrics: dict[str, float] = {}
        for micro_step in range(config.gradient_accumulation_steps):
            clean = data.batch("train", config.batch_size)
            sync = not distributed or micro_step == config.gradient_accumulation_steps - 1
            context = nullcontext() if sync else model.no_sync()
            with context, autocast():
                loss, step_metrics = objective_loss(
                    model, clean, config.model.objective, config.model.mask_token_id
                )
                loss = loss / config.gradient_accumulation_steps
            step_loss += float(loss.detach())
            scaler.scale(loss).backward()
        if config.grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if master and iteration % config.log_interval == 0:
            elapsed = time.time() - started
            extras = " ".join(f"{key}={value:.3f}" for key, value in step_metrics.items())
            print(
                f"iter {iteration}: loss={step_loss:.4f} lr={lr:.2e} time={elapsed:.1f}s {extras}"
            )
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
