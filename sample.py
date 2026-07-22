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
    parser.add_argument("--checkpoint", default="out/shakespeare-mdlm/best.pt")
    parser.add_argument("--length", type=int, default=256, help="total output length in tokens")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--show-steps", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    checkpoint_path = Path(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = ModelConfig(**checkpoint["config"]["model"])
    model = Transformer(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    metadata = json.loads((checkpoint_path.parent / "meta.json").read_text(encoding="utf-8"))
    if metadata.get("format") == "tokenized":
        try:
            import tiktoken
        except ImportError as error:
            raise SystemExit('tokenized sampling requires: pip install -e ".[data]"') from error
        encoding = tiktoken.get_encoding(metadata["tokenizer"]["name"])

        def encode(text: str) -> list[int]:
            return encoding.encode_ordinary(text)

        def decode(tokens: list[int]) -> str:
            return encoding.decode(tokens)

        start_token = int(metadata["tokenizer"]["eot_token_id"])
    else:
        stoi, itos = metadata["stoi"], metadata["itos"]

        def encode(text: str) -> list[int]:
            return [stoi[char] for char in text]

        def decode(tokens: list[int]) -> str:
            return "".join(itos[token] for token in tokens)

        start_token = 0

    samples = []
    for sample_index in range(args.num_samples):
        generator = torch.Generator(device=device).manual_seed(args.seed + sample_index)
        if model_config.objective == "autoregressive":
            encoded = encode(args.prompt) or [start_token]
            tokens = torch.tensor([encoded], dtype=torch.long, device=device)
            total_length = min(args.length, model_config.block_size)
            new_tokens = max(0, total_length - len(encoded))
            result = model.generate(
                tokens,
                new_tokens,
                args.temperature,
                args.top_k,
                generator=generator,
            )[0]
        else:
            if model_config.mask_token_id is None:
                raise ValueError("checkpoint has no mask token")
            length = min(args.length, model_config.block_size)
            initial = torch.full((1, length), model_config.mask_token_id, dtype=torch.long)
            prompt = encode(args.prompt)[:length]
            if prompt:
                initial[0, : len(prompt)] = torch.tensor(prompt)

            def show(step: int, value: torch.Tensor) -> None:
                if args.show_steps:
                    visible = [
                        start_token if token == model_config.mask_token_id else token
                        for token in value[0].tolist()
                    ]
                    print(f"step {step:02d}: {decode(visible)}")

            result = sample_masked(
                model,
                (1, length),
                model_config.mask_token_id,
                args.steps,
                args.temperature,
                args.top_k,
                initial,
                show,
                generator,
            )[0]
        samples.append(decode(result.tolist()))
    output = "\n\n--- sample ---\n\n".join(samples)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
