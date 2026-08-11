"""Export compact, inference-only Hugging Face artifacts from training checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

VARIANTS = {
    "ar": "out/tinystories-106m-ar/best.pt",
    "mdlm": "out/tinystories-106m-mdlm/best.pt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    return parser.parse_args()


def export_variant(label: str, checkpoint_path: Path, output_dir: Path, dtype: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    model_config = checkpoint["config"]["model"]
    expected_objective = "autoregressive" if label == "ar" else "masked_diffusion"
    if model_config["objective"] != expected_objective:
        raise ValueError(
            f"{checkpoint_path} has objective {model_config['objective']!r}, "
            f"expected {expected_objective!r}"
        )
    tensor_dtype = torch.float32 if dtype == "float32" else torch.bfloat16
    state_dict = {
        name: tensor.detach().to(dtype=tensor_dtype).contiguous()
        for name, tensor in checkpoint["model"].items()
    }
    save_file(
        state_dict,
        output_dir / f"{label}.safetensors",
        metadata={
            "format": "pt",
            "objective": expected_objective,
            "source_checkpoint": checkpoint_path.as_posix(),
        },
    )
    metadata = json.loads((checkpoint_path.parent / "meta.json").read_text(encoding="utf-8"))
    export_config = {
        "model": model_config,
        "tokenizer": metadata["tokenizer"],
        "dataset": metadata["dataset"],
        "training": {
            "tokens_seen": checkpoint["tokens_seen"],
            "step": checkpoint["step"],
            "best_val": checkpoint["best_val"],
        },
        "weights": {
            "dtype": dtype,
            "file": f"{label}.safetensors",
        },
    }
    (output_dir / f"{label}_config.json").write_text(
        json.dumps(export_config, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for label, checkpoint in VARIANTS.items():
        export_variant(label, Path(checkpoint), args.output_dir, args.dtype)


if __name__ == "__main__":
    main()
