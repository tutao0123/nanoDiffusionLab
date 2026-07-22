"""A compact Transformer shared by autoregressive and diffusion objectives."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from config import ModelConfig

KVCache = tuple[torch.Tensor, torch.Tensor]
GeneratorLike = torch.Generator | Sequence[torch.Generator] | None


def sample_rows(probabilities: torch.Tensor, generator: GeneratorLike) -> torch.Tensor:
    """Sample one item per row, optionally with a stable generator for every sequence."""
    if isinstance(generator, Sequence):
        if len(generator) != probabilities.size(0):
            raise ValueError("generator count must match batch size")
        return torch.stack(
            [
                torch.multinomial(probabilities[row], 1, generator=generator[row])
                for row in range(probabilities.size(0))
            ]
        )
    return torch.multinomial(probabilities, 1, generator=generator)


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

    def forward_cached(
        self,
        x: torch.Tensor,
        past_kv: KVCache | None = None,
    ) -> tuple[torch.Tensor, KVCache]:
        """Run causal attention while appending keys and values to an inference cache."""
        if not self.causal:
            raise ValueError("KV caching is only supported for causal attention")
        batch, length, channels = x.shape
        q, k, v = self.qkv(x).split(channels, dim=2)
        head_size = channels // self.n_head
        shape = (batch, length, self.n_head, head_size)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        if past_kv is None:
            full_k, full_v = k, v
            is_causal = True
        else:
            if length != 1:
                raise ValueError("cached decoding accepts one new token at a time")
            full_k = torch.cat((past_kv[0], k), dim=2)
            full_v = torch.cat((past_kv[1], v), dim=2)
            # The single query is the final position and may attend to every cached key.
            is_causal = False
        y = F.scaled_dot_product_attention(q, full_k, full_v, is_causal=is_causal)
        y = y.transpose(1, 2).contiguous().view(batch, length, channels)
        return self.resid_dropout(self.proj(y)), (full_k, full_v)


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

    def forward_cached(
        self,
        x: torch.Tensor,
        past_kv: KVCache | None = None,
    ) -> tuple[torch.Tensor, KVCache]:
        attention, cache = self.attn.forward_cached(self.ln_1(x), past_kv)
        x = x + attention
        return x + self.mlp(self.ln_2(x)), cache


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

    def forward_cached(
        self,
        idx: torch.Tensor,
        caches: list[KVCache] | None = None,
    ) -> tuple[torch.Tensor, list[KVCache]]:
        """Return causal logits and updated per-layer inference caches."""
        if not self.config.causal:
            raise ValueError("KV caching is only supported for autoregressive models")
        batch, length = idx.shape
        del batch
        past_length = 0 if caches is None else caches[0][0].size(2)
        if caches is not None and len(caches) != len(self.blocks):
            raise ValueError("cache count must match the number of Transformer blocks")
        if past_length + length > self.config.block_size:
            raise ValueError("cached sequence exceeds block_size")
        positions = torch.arange(past_length, past_length + length, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)
        x = self.dropout(x)
        updated = []
        for layer, block in enumerate(self.blocks):
            past_kv = None if caches is None else caches[layer]
            x, cache = block.forward_cached(x, past_kv)
            updated.append(cache)
        return self.lm_head(self.norm(x)), updated

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
        *,
        use_cache: bool = True,
        generator: GeneratorLike = None,
    ) -> torch.Tensor:
        if not self.config.causal:
            raise ValueError("generate() is only for autoregressive models")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if use_cache:
            if idx.size(1) + max_new_tokens > self.config.block_size:
                raise ValueError("cached generation cannot exceed block_size")
            if max_new_tokens == 0:
                return idx
            logits, caches = self.forward_cached(idx)
            for step in range(max_new_tokens):
                next_logits = logits[:, -1] / max(temperature, 1e-5)
                if top_k is not None:
                    threshold = torch.topk(next_logits, min(top_k, next_logits.size(-1))).values[
                        :, [-1]
                    ]
                    next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))
                next_token = sample_rows(F.softmax(next_logits, dim=-1), generator)
                idx = torch.cat((idx, next_token), dim=1)
                if step + 1 < max_new_tokens:
                    logits, caches = self.forward_cached(next_token, caches)
            return idx
        for _ in range(max_new_tokens):
            context = idx[:, -self.config.block_size :]
            logits = self(context)[:, -1] / max(temperature, 1e-5)
            if top_k is not None:
                threshold = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                logits = logits.masked_fill(logits < threshold, float("-inf"))
            next_token = sample_rows(F.softmax(logits, dim=-1), generator)
            idx = torch.cat((idx, next_token), dim=1)
        return idx

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
