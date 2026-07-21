import torch
import torch.nn as nn

from diffusion import sample_masked


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
