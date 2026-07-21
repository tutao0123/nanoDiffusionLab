"""Download, tokenize, and shard the pinned TinyStories dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

DATASET = "roneneldan/TinyStories"
REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/tinystories-gpt2")
    parser.add_argument("--shard-tokens", type=int, default=100_000_000)
    parser.add_argument("--batch-documents", type=int, default=1_024)
    parser.add_argument("--workers", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    parser.add_argument("--download-workers", type=int, default=5)
    parser.add_argument("--max-train-documents", type=int)
    parser.add_argument("--max-validation-documents", type=int)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class ShardWriter:
    def __init__(self, root: Path, split: str, shard_tokens: int) -> None:
        self.root = root / split
        self.root.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.shard_tokens = shard_tokens
        self.index = 0
        self.current_tokens = 0
        self.total_tokens = 0
        self.file: Any = None
        self.path: Path | None = None
        self.shards: list[dict[str, Any]] = []

    def _open(self) -> None:
        self.path = self.root / f"{self.index:05d}.bin"
        self.file = self.path.open("wb")
        self.current_tokens = 0

    def _close(self) -> None:
        if self.file is None or self.path is None:
            return
        self.file.close()
        relative = self.path.relative_to(self.root.parent)
        self.shards.append(
            {"path": str(relative), "tokens": self.current_tokens, "sha256": sha256(self.path)}
        )
        self.file = None
        self.path = None
        self.index += 1

    def write(self, tokens: list[int]) -> None:
        offset = 0
        while offset < len(tokens):
            if self.file is None:
                self._open()
            available = self.shard_tokens - self.current_tokens
            piece = tokens[offset : offset + available]
            np.asarray(piece, dtype=np.uint16).tofile(self.file)
            count = len(piece)
            offset += count
            self.current_tokens += count
            self.total_tokens += count
            if self.current_tokens == self.shard_tokens:
                self._close()

    def finish(self) -> None:
        self._close()


def process_split(
    dataset: Any,
    encoding: Any,
    root: Path,
    split: str,
    shard_tokens: int,
    batch_documents: int,
    workers: int,
    limit: int | None,
) -> dict[str, Any]:
    writer = ShardWriter(root, split, shard_tokens)
    documents = min(len(dataset), limit) if limit is not None else len(dataset)
    for start in range(0, documents, batch_documents):
        texts = dataset[start : min(start + batch_documents, documents)]["text"]
        batches = encoding.encode_ordinary_batch(texts, num_threads=workers)
        for tokens in batches:
            writer.write(tokens + [encoding.eot_token])
        if start == 0 or (start // batch_documents) % 100 == 0:
            print(f"{split}: {min(start + len(texts), documents):,}/{documents:,} documents")
    writer.finish()
    return {"documents": documents, "tokens": writer.total_tokens, "shards": writer.shards}


def main() -> None:
    args = parse_args()
    # Xet is substantially slower than regular HTTPS on the target training server.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    try:
        import tiktoken
        from datasets import load_dataset
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise SystemExit('install data dependencies with: pip install -e ".[data]"') from error

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    encoding = tiktoken.get_encoding("gpt2")
    snapshot = Path(
        snapshot_download(
            repo_id=DATASET,
            repo_type="dataset",
            revision=REVISION,
            allow_patterns="data/*.parquet",
            max_workers=args.download_workers,
        )
    )
    data_files = {
        "train": sorted(str(path) for path in (snapshot / "data").glob("train-*.parquet")),
        "validation": sorted(
            str(path) for path in (snapshot / "data").glob("validation-*.parquet")
        ),
    }
    if not data_files["train"] or not data_files["validation"]:
        raise RuntimeError("pinned TinyStories snapshot did not contain expected parquet files")
    dataset = load_dataset("parquet", data_files=data_files)
    splits = {
        "train": process_split(
            dataset["train"],
            encoding,
            output,
            "train",
            args.shard_tokens,
            args.batch_documents,
            args.workers,
            args.max_train_documents,
        ),
        "validation": process_split(
            dataset["validation"],
            encoding,
            output,
            "validation",
            args.shard_tokens,
            args.batch_documents,
            args.workers,
            args.max_validation_documents,
        ),
    }
    manifest = {
        "format": "nanodiffusion-token-shards/1",
        "dtype": "uint16",
        "dataset": {"name": DATASET, "revision": REVISION},
        "tokenizer": {
            "name": "gpt2",
            "vocab_size": encoding.n_vocab,
            "eot_token_id": encoding.eot_token,
        },
        "splits": splits,
    }
    with tempfile.NamedTemporaryFile("w", dir=output, delete=False, encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")
        temporary = Path(file.name)
    temporary.replace(output / "manifest.json")
    print(f"wrote {output / 'manifest.json'}")


if __name__ == "__main__":
    main()
