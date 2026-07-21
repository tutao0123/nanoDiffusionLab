"""Sample text from a nanoDiffusionLab checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from config import ModelConfig
from diffusion import sample_masked
from model import Transformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="out/shakespeare-mdlm/ckpt.pt")
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--show-steps", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = ModelConfig(**checkpoint["config"]["model"])
    model = Transformer(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    metadata = json.loads((checkpoint_path.parent / "meta.json").read_text(encoding="utf-8"))
    stoi, itos = metadata["stoi"], metadata["itos"]

    if model_config.objective == "autoregressive":
        encoded = [stoi[char] for char in args.prompt] or [0]
        tokens = torch.tensor([encoded], dtype=torch.long, device=device)
        result = model.generate(tokens, args.length, args.temperature, args.top_k)[0]
    else:
        if model_config.mask_token_id is None:
            raise ValueError("checkpoint has no mask token")
        length = min(args.length, model_config.block_size)
        initial = torch.full((1, length), model_config.mask_token_id, dtype=torch.long)
        prompt = [stoi[char] for char in args.prompt[:length]]
        if prompt:
            initial[0, : len(prompt)] = torch.tensor(prompt)

        def show(step: int, value: torch.Tensor) -> None:
            if args.show_steps:
                text = "".join(
                    "·" if token == model_config.mask_token_id else itos[token]
                    for token in value[0].tolist()
                )
                print(f"step {step:02d}: {text}")

        result = sample_masked(
            model,
            (1, length),
            model_config.mask_token_id,
            args.steps,
            args.temperature,
            args.top_k,
            initial,
            show,
        )[0]
    print("".join(itos[token] for token in result.tolist()))


if __name__ == "__main__":
    main()
