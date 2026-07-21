"""Character and memory-mapped token data loaders."""

from __future__ import annotations

import bisect
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


class CharacterData:
    """Small in-memory character dataset used by smoke tests and examples."""

    def __init__(self, path: str | Path, block_size: int, device: torch.device) -> None:
        self.path = Path(path)
        text = self.path.read_text(encoding="utf-8")
        chars = sorted(set(text))
        if len(text) < block_size + 2:
            raise ValueError(f"dataset needs at least {block_size + 2} characters")
        self.stoi = {char: index for index, char in enumerate(chars)}
        self.itos = chars
        self.tokens = torch.tensor([self.stoi[char] for char in text], dtype=torch.long)
        self.split = int(0.9 * len(self.tokens))
        self.block_size = block_size
        self.device = device

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def batch(
        self,
        split: str,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        data = self.tokens[: self.split] if split == "train" else self.tokens[self.split :]
        if len(data) < self.block_size + 1:
            data = self.tokens
        starts = torch.randint(len(data) - self.block_size, (batch_size,), generator=generator)
        batch = torch.stack([data[int(i) : int(i) + self.block_size] for i in starts])
        return batch.to(self.device, non_blocking=True)

    def save_metadata(self, out_dir: Path) -> None:
        payload = {"format": "char", "stoi": self.stoi, "itos": self.itos}
        (out_dir / "meta.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def description(self) -> dict[str, Any]:
        return {"format": "char", "path": str(self.path), "tokens": len(self.tokens)}


class TokenizedData:
    """Sample fixed-length batches from uint16/uint32 token shards."""

    def __init__(self, manifest_path: str | Path, block_size: int, device: torch.device) -> None:
        self.manifest_path = Path(manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("format") != "nanodiffusion-token-shards/1":
            raise ValueError("unsupported token manifest format")
        self.block_size = block_size
        self.device = device
        self.dtype = np.dtype(self.manifest["dtype"])
        self.vocab_size = int(self.manifest["tokenizer"]["vocab_size"])
        self._shards: dict[str, list[np.memmap]] = {}
        self._cumulative_starts: dict[str, list[int]] = {}
        for split, split_info in self.manifest["splits"].items():
            shards = []
            cumulative = []
            total = 0
            for item in split_info["shards"]:
                path = self.manifest_path.parent / item["path"]
                shard = np.memmap(path, dtype=self.dtype, mode="r")
                if len(shard) != int(item["tokens"]):
                    raise ValueError(f"token count mismatch for {path}")
                valid_starts = len(shard) - self.block_size + 1
                if valid_starts <= 0:
                    continue
                shards.append(shard)
                total += valid_starts
                cumulative.append(total)
            if not shards:
                raise ValueError(f"split {split!r} has no shard longer than block_size")
            self._shards[split] = shards
            self._cumulative_starts[split] = cumulative

    def batch(
        self,
        split: str,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        cumulative = self._cumulative_starts[split]
        global_starts = torch.randint(cumulative[-1], (batch_size,), generator=generator)
        rows = []
        for value in global_starts.tolist():
            shard_index = bisect.bisect_right(cumulative, value)
            previous = cumulative[shard_index - 1] if shard_index else 0
            local_start = value - previous
            values = self._shards[split][shard_index][local_start : local_start + self.block_size]
            rows.append(torch.tensor(values, dtype=torch.long))
        return torch.stack(rows).to(self.device, non_blocking=True)

    def save_metadata(self, out_dir: Path) -> None:
        payload = {
            "format": "tokenized",
            "tokenizer": self.manifest["tokenizer"],
            "manifest": str(self.manifest_path.resolve()),
            "dataset": self.manifest["dataset"],
        }
        (out_dir / "meta.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def description(self) -> dict[str, Any]:
        return {
            "format": "tokenized",
            "manifest": str(self.manifest_path.resolve()),
            "dataset": self.manifest["dataset"],
            "tokenizer": self.manifest["tokenizer"],
            "splits": {
                name: {"tokens": value["tokens"], "documents": value["documents"]}
                for name, value in self.manifest["splits"].items()
            },
        }
