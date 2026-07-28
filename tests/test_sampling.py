import pytest
import torch
import torch.nn as nn

from config import ModelConfig
from diffusion import remask_rate, sample_masked
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


class RecordingPredictByPosition(PredictByPosition):
    def __init__(self, vocab_size: int) -> None:
        super().__init__(vocab_size)
        self.projected_positions: list[int] = []

    def forward(
        self,
        tokens: torch.Tensor,
        t: torch.Tensor | None = None,
        output_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output_mask is not None
        self.projected_positions.append(int(output_mask.sum()))
        return super().forward(tokens, t, output_mask)


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


def test_sampler_can_remask_weak_predictions_without_changing_prompt() -> None:
    model = RecordingPredictByPosition(vocab_size=5)
    initial = torch.full((1, 9), 5, dtype=torch.long)
    initial[0, :2] = torch.tensor([4, 3])

    result = sample_masked(
        model,
        shape=(1, 9),
        mask_token_id=5,
        steps=4,
        temperature=0.1,
        initial_tokens=initial,
        remask_fraction=0.5,
    )

    assert not result.eq(5).any()
    assert torch.equal(result[0, :2], initial[0, :2])
    assert sum(model.projected_positions) > 7


def test_sampler_rejects_invalid_remask_fraction() -> None:
    model = PredictByPosition(vocab_size=5)
    with pytest.raises(ValueError, match="remask_fraction"):
        sample_masked(model, (1, 4), mask_token_id=5, remask_fraction=1.1)


def test_remask_rate_decays_over_the_final_quarter() -> None:
    assert remask_rate(0.1, 0.75, step=0, steps=5) == pytest.approx(0.1)
    assert remask_rate(0.1, 0.75, step=3, steps=5) == pytest.approx(0.1)
    assert remask_rate(0.1, 0.75, step=4, steps=5) == pytest.approx(0.0)


def test_sampler_rejects_invalid_remask_decay_start() -> None:
    model = PredictByPosition(vocab_size=5)
    with pytest.raises(ValueError, match="remask_decay_start"):
        sample_masked(model, (1, 4), mask_token_id=5, remask_decay_start=-0.1)


def test_sampler_does_not_remask_protected_tokens() -> None:
    model = RecordingPredictByPosition(vocab_size=2)
    initial = torch.full((1, 10), 2, dtype=torch.long)
    result = sample_masked(
        model,
        shape=(1, 10),
        mask_token_id=2,
        steps=5,
        temperature=0.1,
        initial_tokens=initial,
        remask_fraction=1.0,
        protected_token_ids=(0, 1),
    )

    assert not result.eq(2).any()
    assert sum(model.projected_positions) == 30


def test_zero_remask_keeps_legacy_generator_stream() -> None:
    model = PredictByPosition(vocab_size=5)
    initial = torch.full((2, 7), 5, dtype=torch.long)
    generators_a = [torch.Generator().manual_seed(seed) for seed in (11, 12)]
    generators_b = [torch.Generator().manual_seed(seed) for seed in (11, 12)]

    legacy = sample_masked(
        model,
        shape=(2, 7),
        mask_token_id=5,
        steps=3,
        initial_tokens=initial,
        generator=generators_a,
    )
    explicit_zero = sample_masked(
        model,
        shape=(2, 7),
        mask_token_id=5,
        steps=3,
        initial_tokens=initial,
        generator=generators_b,
        remask_fraction=0.0,
        remask_decay_start=0.75,
        protected_token_ids=(0,),
    )

    assert torch.equal(legacy, explicit_zero)


def test_remask_per_example_rng_is_reproducible_across_batching() -> None:
    model = PredictByPosition(vocab_size=5)
    initial = torch.full((2, 9), 5, dtype=torch.long)
    batched = sample_masked(
        model,
        shape=(2, 9),
        mask_token_id=5,
        steps=4,
        initial_tokens=initial,
        generator=[torch.Generator().manual_seed(seed) for seed in (21, 22)],
        remask_fraction=0.5,
        remask_decay_start=0.75,
        protected_token_ids=(0,),
    )
    separate = torch.cat(
        [
            sample_masked(
                model,
                shape=(1, 9),
                mask_token_id=5,
                steps=4,
                initial_tokens=initial[row : row + 1],
                generator=[torch.Generator().manual_seed(seed)],
                remask_fraction=0.5,
                remask_decay_start=0.75,
                protected_token_ids=(0,),
            )
            for row, seed in enumerate((21, 22))
        ]
    )

    assert torch.equal(batched, separate)


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
