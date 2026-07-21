"""Configuration objects shared by training and sampling."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from runpy import run_path
from typing import Any, Literal

Objective = Literal["autoregressive", "masked_diffusion"]


@dataclass
class ModelConfig:
    vocab_size: int = 65
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = False
    objective: Objective = "masked_diffusion"
    mask_token_id: int | None = 65
    time_conditioning: bool = True

    @property
    def causal(self) -> bool:
        return self.objective == "autoregressive"

    @property
    def input_vocab_size(self) -> int:
        if self.mask_token_id is None:
            return self.vocab_size
        return max(self.vocab_size, self.mask_token_id + 1)


@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data_format: Literal["char", "tokenized"] = "char"
    data_path: str = "data/tinyshakespeare/input.txt"
    out_dir: str = "out/shakespeare-mdlm"
    seed: int = 1337
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    max_iters: int = 5_000
    max_tokens: int | None = None
    milestone_tokens: tuple[int, ...] = ()
    eval_interval: int = 250
    eval_iters: int = 50
    eval_seed: int = 4242
    eval_mask_ratios: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)
    log_interval: int = 10
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"
    compile: bool = False
    gradient_checkpointing: bool = False
    backend: str = "nccl"
    always_save_checkpoint: bool = False
    run_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> TrainConfig:
    """Load a Python config containing a top-level ``config`` object."""
    namespace = run_path(str(path))
    config = namespace.get("config")
    if not isinstance(config, TrainConfig):
        raise TypeError(f"{path} must define `config = TrainConfig(...)`")
    return config
