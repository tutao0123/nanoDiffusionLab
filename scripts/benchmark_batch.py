"""Measure single-GPU batch capacity for a configured model."""

from __future__ import annotations

import argparse
import json
import time

import torch

from config import load_config
from model import Transformer
from train import objective_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/tinystories_106m.py")
    parser.add_argument(
        "--objective", choices=("autoregressive", "masked_diffusion"), required=True
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batches", type=int, nargs="+", default=(32, 16, 8, 4))
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config.model.objective = args.objective
    config.model.mask_token_id = (
        None if args.objective == "autoregressive" else config.model.vocab_size
    )
    config.model.time_conditioning = args.objective == "masked_diffusion"
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    results = []
    for batch_size in args.batches:
        torch.cuda.empty_cache()
        try:
            model = Transformer(config.model, gradient_checkpointing=True).to(device).train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
            tokens = torch.randint(
                config.model.vocab_size,
                (batch_size, config.model.block_size),
                device=device,
            )
            torch.cuda.reset_peak_memory_stats(device)
            timings = []
            for step in range(args.steps + 1):
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, _ = objective_loss(
                        model, tokens, config.model.objective, config.model.mask_token_id
                    )
                loss.backward()
                optimizer.step()
                torch.cuda.synchronize(device)
                if step:
                    timings.append(time.perf_counter() - started)
            result = {
                "batch_size": batch_size,
                "status": "ok",
                "step_seconds": sum(timings) / len(timings),
                "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
            }
            del model, optimizer, tokens, loss
        except torch.OutOfMemoryError:
            result = {"batch_size": batch_size, "status": "oom"}
        results.append(result)
        print(json.dumps(result), flush=True)
    payload = {
        "objective": args.objective,
        "device": args.device,
        "model": config.model.__dict__,
        "results": results,
    }
    if args.output:
        with open(args.output, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)


if __name__ == "__main__":
    main()
