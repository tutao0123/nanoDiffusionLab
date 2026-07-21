import torch

from diffusion import corrupt_tokens


def test_corruption_only_changes_masked_positions() -> None:
    tokens = torch.arange(24).view(3, 8)
    noisy, mask, t = corrupt_tokens(tokens, mask_token_id=99, t=torch.tensor([0.0, 0.5, 1.0]))

    assert mask.any(dim=1).all()
    assert torch.equal(noisy[~mask], tokens[~mask])
    assert noisy[mask].eq(99).all()
    assert t.shape == (3,)
    assert mask[2].all()


def test_corruption_is_reproducible_with_generator() -> None:
    tokens = torch.zeros((2, 16), dtype=torch.long)
    first = torch.Generator().manual_seed(7)
    second = torch.Generator().manual_seed(7)
    noisy_a, mask_a, t_a = corrupt_tokens(tokens, 1, generator=first)
    noisy_b, mask_b, t_b = corrupt_tokens(tokens, 1, generator=second)

    assert torch.equal(noisy_a, noisy_b)
    assert torch.equal(mask_a, mask_b)
    assert torch.equal(t_a, t_b)
