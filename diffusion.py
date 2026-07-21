"""Mask corruption, denoising loss, and a readable confidence sampler."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F


def corrupt_tokens(
    tokens: torch.Tensor,
    mask_token_id: int,
    t: torch.Tensor | None = None,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replace tokens with MASK at a independently sampled rate per sequence."""
    batch, length = tokens.shape
    if t is None:
        t = torch.rand(batch, device=tokens.device, generator=generator)
    t = t.clamp(1e-3, 1.0)
    mask = torch.rand(tokens.shape, device=tokens.device, generator=generator) < t[:, None]

    # Every training example should contribute at least one target token.
    empty = ~mask.any(dim=1)
    if empty.any():
        rows = torch.where(empty)[0]
        columns = torch.randint(length, (rows.numel(),), device=tokens.device, generator=generator)
        mask[rows, columns] = True

    noisy = tokens.clone()
    noisy[mask] = mask_token_id
    return noisy, mask, t


def masked_diffusion_loss(
    model: torch.nn.Module,
    clean_tokens: torch.Tensor,
    mask_token_id: int,
    t: torch.Tensor | None = None,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    noisy, loss_mask, t = corrupt_tokens(clean_tokens, mask_token_id, t, generator=generator)
    logits = model(noisy, t, loss_mask)
    targets = clean_tokens[loss_mask]
    loss = F.cross_entropy(logits, targets)
    accuracy = (logits.argmax(dim=-1) == targets).float().mean()
    metrics = {
        "masked_accuracy": float(accuracy.detach()),
        "mask_ratio": float(loss_mask.float().mean()),
        "target_tokens": float(loss_mask.sum()),
        "correct_tokens": float((logits.argmax(dim=-1) == targets).sum()),
    }
    return loss, metrics


@torch.no_grad()
def sample_masked(
    model: torch.nn.Module,
    shape: tuple[int, int],
    mask_token_id: int,
    steps: int = 32,
    temperature: float = 1.0,
    top_k: int | None = None,
    initial_tokens: torch.Tensor | None = None,
    callback: Callable[[int, torch.Tensor], None] | None = None,
) -> torch.Tensor:
    """Generate by repeatedly revealing the most confident masked positions."""
    if steps < 1:
        raise ValueError("steps must be positive")
    device = next(model.parameters()).device
    x = (
        initial_tokens.to(device).clone()
        if initial_tokens is not None
        else torch.full(shape, mask_token_id, dtype=torch.long, device=device)
    )
    if tuple(x.shape) != shape:
        raise ValueError("initial_tokens must match shape")

    for step in range(steps):
        masked = x.eq(mask_token_id)
        if not masked.any():
            break
        t = masked.float().mean(dim=1)
        logits = model(x, t) / max(temperature, 1e-5)
        if top_k is not None:
            values = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values
            logits = logits.masked_fill(logits < values[..., [-1]], float("-inf"))
        probabilities = F.softmax(logits, dim=-1)
        predictions = torch.multinomial(probabilities.view(-1, probabilities.size(-1)), 1).view_as(
            x
        )
        confidence = probabilities.gather(-1, predictions[..., None]).squeeze(-1)
        remaining_steps = steps - step

        for row in range(x.size(0)):
            positions = torch.where(masked[row])[0]
            reveal = math.ceil(positions.numel() / remaining_steps)
            chosen = positions[confidence[row, positions].topk(reveal).indices]
            x[row, chosen] = predictions[row, chosen]
        if callback is not None:
            callback(step + 1, x.clone())
    return x
