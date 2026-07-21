import torch

from config import ModelConfig
from model import Transformer


def tiny_config(objective: str) -> ModelConfig:
    return ModelConfig(
        vocab_size=8,
        block_size=8,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        objective=objective,
        mask_token_id=8 if objective == "masked_diffusion" else None,
        time_conditioning=False,
    )


def test_causal_attention_cannot_see_future_tokens() -> None:
    torch.manual_seed(1)
    model = Transformer(tiny_config("autoregressive")).eval()
    first = torch.tensor([[1, 2, 3, 4]])
    changed_future = torch.tensor([[1, 2, 3, 7]])

    logits_a = model(first)
    logits_b = model(changed_future)
    torch.testing.assert_close(logits_a[:, :3], logits_b[:, :3])


def test_diffusion_attention_uses_future_tokens() -> None:
    torch.manual_seed(1)
    model = Transformer(tiny_config("masked_diffusion")).eval()
    first = torch.tensor([[1, 2, 3, 4]])
    changed_future = torch.tensor([[1, 2, 3, 7]])

    difference = (model(first)[:, 0] - model(changed_future)[:, 0]).abs().max()
    assert difference > 1e-7
