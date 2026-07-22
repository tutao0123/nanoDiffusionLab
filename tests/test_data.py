import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from data import TokenizedData


def write_shard(path: Path, values: list[int]) -> dict[str, object]:
    array = np.asarray(values, dtype=np.uint16)
    array.tofile(path)
    return {
        "path": path.name,
        "tokens": len(values),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_tokenized_data_reads_manifest_and_replays_batches(tmp_path: Path) -> None:
    train = write_shard(tmp_path / "train.bin", list(range(64)))
    validation = write_shard(tmp_path / "validation.bin", list(range(100, 164)))
    manifest = {
        "format": "nanodiffusion-token-shards/1",
        "dtype": "uint16",
        "dataset": {"name": "test", "revision": "fixed"},
        "tokenizer": {"name": "test", "vocab_size": 256, "eot_token_id": 255},
        "splits": {
            "train": {"documents": 1, "tokens": 64, "shards": [train]},
            "validation": {"documents": 1, "tokens": 64, "shards": [validation]},
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    data = TokenizedData(manifest_path, block_size=8, device=torch.device("cpu"))

    first = data.batch("train", 4, generator=torch.Generator().manual_seed(7))
    second = data.batch("train", 4, generator=torch.Generator().manual_seed(7))

    assert torch.equal(first, second)
    assert first.shape == (4, 8)
    assert torch.all(first[:, 1:] - first[:, :-1] == 1)


def test_tokenized_data_rejects_short_splits(tmp_path: Path) -> None:
    shard = write_shard(tmp_path / "short.bin", [1, 2, 3])
    manifest = {
        "format": "nanodiffusion-token-shards/1",
        "dtype": "uint16",
        "dataset": {"name": "test", "revision": "fixed"},
        "tokenizer": {"name": "test", "vocab_size": 8, "eot_token_id": 7},
        "splits": {"train": {"documents": 1, "tokens": 3, "shards": [shard]}},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    try:
        TokenizedData(path, block_size=8, device=torch.device("cpu"))
    except ValueError as error:
        assert "no shard longer" in str(error)
    else:
        raise AssertionError("expected short split to be rejected")
