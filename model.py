"""A compact Transformer shared by autoregressive and diffusion objectives."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from config import ModelConfig


class LayerNorm(nn.Module):
    def __init__(self, size: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.bias = nn.Parameter(torch.zeros(size)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class SelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = config.n_head
        self.dropout = config.dropout
        self.causal = config.causal
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, channels = x.shape
        q, k, v = self.qkv(x).split(channels, dim=2)
        head_size = channels // self.n_head
        shape = (batch, length, self.n_head, head_size)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.causal,
        )
        y = y.transpose(1, 2).contiguous().view(batch, length, channels)
        return self.resid_dropout(self.proj(y))


class MLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x), approximate="tanh")))


class Block(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, config.bias)
        self.attn = SelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig, gradient_checkpointing: bool = False) -> None:
        super().__init__()
        self.config = config
        self.gradient_checkpointing = gradient_checkpointing
        self.token_embedding = nn.Embedding(config.input_vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm = LayerNorm(config.n_embd, config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.time_mlp = (
            nn.Sequential(
                nn.Linear(1, config.n_embd),
                nn.SiLU(),
                nn.Linear(config.n_embd, config.n_embd),
            )
            if config.time_conditioning and not config.causal
            else None
        )
        self.apply(self._init_weights)
        for name, parameter in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(parameter, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward_features(self, idx: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        _, length = idx.shape
        if length > self.config.block_size:
            raise ValueError(f"sequence length {length} exceeds block_size")
        positions = torch.arange(length, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)
        if self.time_mlp is not None:
            if t is None:
                t = idx.eq(self.config.mask_token_id).float().mean(dim=1)
            x = x + self.time_mlp(t[:, None])[:, None, :]
        x = self.dropout(x)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.norm(x)

    def forward(
        self,
        idx: torch.Tensor,
        t: torch.Tensor | None = None,
        output_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.forward_features(idx, t)
        if output_mask is not None:
            hidden = hidden[output_mask]
        return self.lm_head(hidden)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        if not self.config.causal:
            raise ValueError("generate() is only for autoregressive models")
        for _ in range(max_new_tokens):
            context = idx[:, -self.config.block_size :]
            logits = self(context)[:, -1] / max(temperature, 1e-5)
            if top_k is not None:
                threshold = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                logits = logits.masked_fill(logits < threshold, float("-inf"))
            next_token = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
