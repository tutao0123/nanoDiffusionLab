import pytest
import torch
import torch.nn as nn

from config import ModelConfig
from diffusion import sample_masked
from model import Transformer


class PredictByPosition(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        tokens: torch.Tensor,
        t: torch.Tensor | None = None,
        output_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, length = tokens.shape
        logits = torch.full((batch, length, self.vocab_size), -20.0, device=tokens.device)
        for position in range(length):
            logits[:, position, position % self.vocab_size] = 20.0 + self.anchor
        return logits if output_mask is None else logits[output_mask]


def test_sampler_reveals_every_position_and_preserves_prompt() -> None:
    model = PredictByPosition(vocab_size=5)
    initial = torch.full((1, 7), 5, dtype=torch.long)
    initial[0, 0] = 4
    result = sample_masked(
        model,
        shape=(1, 7),
        mask_token_id=5,
        steps=3,
        temperature=0.1,
        initial_tokens=initial,
    )

    assert not result.eq(5).any()
    assert result[0, 0].item() == 4
    assert result.shape == initial.shape


def autoregressive_model() -> Transformer:
    torch.manual_seed(11)
    model = Transformer(
        ModelConfig(
            vocab_size=17,
            block_size=12,
            n_layer=2,
            n_head=2,
            n_embd=8,
            objective="autoregressive",
            mask_token_id=None,
            time_conditioning=False,
        )
    )
    return model.eval()


def test_cached_logits_match_full_causal_forward() -> None:
    model = autoregressive_model()
    tokens = torch.tensor([[1, 4, 2, 8]])

    full_logits = model(tokens)
    _, caches = model.forward_cached(tokens[:, :3])
    cached_logits, caches = model.forward_cached(tokens[:, 3:], caches)

    assert torch.allclose(cached_logits[:, -1], full_logits[:, -1], atol=1e-6)
    assert len(caches) == model.config.n_layer
    assert caches[0][0].shape[2] == tokens.size(1)


def test_cached_and_uncached_greedy_generation_match() -> None:
    model = autoregressive_model()
    prompt = torch.tensor([[1, 4, 2]])

    cached = model.generate(prompt, 5, top_k=1, use_cache=True)
    uncached = model.generate(prompt, 5, top_k=1, use_cache=False)

    assert torch.equal(cached, uncached)


def test_cached_generation_rejects_context_overflow() -> None:
    model = autoregressive_model()
    prompt = torch.ones((1, model.config.block_size), dtype=torch.long)

    with pytest.raises(ValueError, match="exceed block_size"):
        model.generate(prompt, 1, use_cache=True)
