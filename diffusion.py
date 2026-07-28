"""Mask corruption, denoising loss, and a readable confidence sampler."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F


def corrupt_tokens(
    tokens: torch.Tensor,
    mask_token_id: int,
    t: torch.Tensor | None = None,
    *,
    generator: torch.Generator | Sequence[torch.Generator] | None = None,
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


def remask_rate(
    remask_fraction: float,
    remask_decay_start: float,
    step: int,
    steps: int,
) -> float:
    """Return the active remask fraction, optionally decayed near the end."""
    if not 0.0 <= remask_fraction <= 1.0:
        raise ValueError("remask_fraction must be between 0 and 1")
    if not 0.0 <= remask_decay_start <= 1.0:
        raise ValueError("remask_decay_start must be between 0 and 1")
    if steps < 1:
        raise ValueError("steps must be positive")
    progress = step / max(steps - 1, 1)
    if remask_decay_start >= 1.0 or progress <= remask_decay_start:
        return remask_fraction
    return remask_fraction * max(0.0, (1.0 - progress) / (1.0 - remask_decay_start))


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
    generator: torch.Generator | Sequence[torch.Generator] | None = None,
    remask_fraction: float = 0.0,
    remask_decay_start: float = 1.0,
    protected_token_ids: Sequence[int] | None = None,
) -> torch.Tensor:
    """Generate by revealing confident positions, optionally revisiting weak predictions."""
    if steps < 1:
        raise ValueError("steps must be positive")
    remask_rate(remask_fraction, remask_decay_start, step=0, steps=steps)
    device = next(model.parameters()).device
    x = (
        initial_tokens.to(device).clone()
        if initial_tokens is not None
        else torch.full(shape, mask_token_id, dtype=torch.long, device=device)
    )
    if tuple(x.shape) != shape:
        raise ValueError("initial_tokens must match shape")
    fixed = (
        initial_tokens.to(device).ne(mask_token_id)
        if initial_tokens is not None
        else torch.zeros(shape, dtype=torch.bool, device=device)
    )
    revealed_confidence = torch.full(x.shape, float("-inf"), device=device)

    for step in range(steps):
        masked = x.eq(mask_token_id)
        if not masked.any():
            break
        remaining_steps = steps - step
        active_remask_fraction = remask_rate(
            remask_fraction,
            remask_decay_start,
            step,
            steps,
        )
        if active_remask_fraction > 0.0 and step > 0 and remaining_steps > 1:
            for row in range(x.size(0)):
                editable = ~masked[row] & ~fixed[row]
                for token_id in protected_token_ids or ():
                    editable &= x[row].ne(token_id)
                remask_count = math.floor(editable.sum().item() * active_remask_fraction)
                if remask_count:
                    positions = torch.where(editable)[0]
                    chosen = positions[
                        revealed_confidence[row, positions].topk(
                            remask_count, largest=False
                        ).indices
                    ]
                    x[row, chosen] = mask_token_id
                    revealed_confidence[row, chosen] = float("-inf")
            masked = x.eq(mask_token_id)
        t = masked.float().mean(dim=1)
        # The model supports projecting only selected hidden states.  Sampling
        # never needs logits for already revealed tokens, so keeping the
        # vocabulary dimension to masked positions avoids a large B x L x V
        # allocation at inference time.
        logits = model(x, t, masked) / max(temperature, 1e-5)
        if top_k is not None:
            values = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values
            logits = logits.masked_fill(logits < values[..., [-1]], float("-inf"))
        probabilities = F.softmax(logits, dim=-1)
        predictions = torch.zeros_like(x)
        confidence = torch.full(x.shape, float("-inf"), device=device)
        if isinstance(generator, Sequence):
            if len(generator) != x.size(0):
                raise ValueError("generator count must match batch size")
            offset = 0
            masked_counts = masked.sum(dim=1).tolist()
            for row, (row_generator, count) in enumerate(
                zip(generator, masked_counts, strict=True)
            ):
                row_probabilities = probabilities[offset : offset + count]
                row_predictions = torch.multinomial(
                    row_probabilities, 1, generator=row_generator
                ).squeeze(-1)
                predictions[row, masked[row]] = row_predictions
                confidence[row, masked[row]] = row_probabilities.gather(
                    -1, row_predictions[:, None]
                ).squeeze(-1)
                offset += count
        else:
            masked_predictions = torch.multinomial(probabilities, 1, generator=generator).squeeze(
                -1
            )
            predictions[masked] = masked_predictions
            confidence[masked] = probabilities.gather(-1, masked_predictions[:, None]).squeeze(-1)
        for row in range(x.size(0)):
            positions = torch.where(masked[row])[0]
            reveal = math.ceil(positions.numel() / remaining_steps)
            chosen = positions[confidence[row, positions].topk(reveal).indices]
            x[row, chosen] = predictions[row, chosen]
            revealed_confidence[row, chosen] = confidence[row, chosen]
        if callback is not None:
            callback(step + 1, x.clone())
    return x
