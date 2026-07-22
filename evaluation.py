"""Deterministic generation-evaluation helpers shared by scripts and tests."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

JUDGE_DIMENSIONS = ("relevance", "coherence", "fluency", "non_repetition", "completeness")


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def load_validation_prompts(
    manifest_path: str | Path,
    count: int,
    prompt_length: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Select fixed document prefixes from token shards without crossing EOT boundaries."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dtype = np.dtype(manifest["dtype"])
    eot = int(manifest["tokenizer"]["eot_token_id"])
    documents: list[list[int]] = []
    pending: list[int] = []
    for shard_info in manifest["splits"]["validation"]["shards"]:
        shard = np.memmap(manifest_path.parent / shard_info["path"], dtype=dtype, mode="r")
        starts = np.concatenate(([0], np.flatnonzero(shard == eot) + 1))
        ends = np.concatenate((np.flatnonzero(shard == eot), [len(shard)]))
        for index, (start, end) in enumerate(zip(starts, ends, strict=True)):
            values = shard[start:end].astype(np.int64).tolist()
            if index == 0 and pending:
                values = pending + values
                pending = []
            if end == len(shard) and (not len(shard) or int(shard[-1]) != eot):
                pending = values
            elif len(values) >= prompt_length:
                documents.append(values)
    if pending and len(pending) >= prompt_length:
        documents.append(pending)
    if len(documents) < count:
        raise ValueError(f"requested {count} prompts but only found {len(documents)} documents")
    chosen = random.Random(seed).sample(range(len(documents)), count)
    return [
        {
            "prompt_id": index,
            "document_index": document_index,
            "token_ids": documents[document_index][:prompt_length],
        }
        for index, document_index in enumerate(chosen)
    ]


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sample_key(record: dict[str, Any]) -> str:
    steps = record.get("steps") or 0
    return f"{record['objective']}:seed{record['train_seed']}:steps{steps}:p{record['prompt_id']}"


def local_text_metrics(token_ids: list[int], eot_token_id: int) -> dict[str, float]:
    """Compute transparent diversity and repetition metrics on one continuation."""
    truncated = token_ids[:]
    ended = eot_token_id in truncated
    if ended:
        truncated = truncated[: truncated.index(eot_token_id)]
    result: dict[str, float] = {
        "length": float(len(truncated)),
        "ended": float(ended),
        "unique_token_ratio": len(set(truncated)) / max(len(truncated), 1),
    }
    for n in (1, 2, 3):
        ngrams = [tuple(truncated[index : index + n]) for index in range(len(truncated) - n + 1)]
        result[f"distinct_{n}"] = len(set(ngrams)) / max(len(ngrams), 1)
    fourgrams = [tuple(truncated[index : index + 4]) for index in range(len(truncated) - 3)]
    counts = Counter(fourgrams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    result["repeated_4gram_ratio"] = repeated / max(len(fourgrams), 1)
    return result


def judge_messages(prompt: str, candidate_a: str, candidate_b: str) -> list[dict[str, str]]:
    schema = {
        "winner": "A, B, or tie",
        "scores": {
            "A": {dimension: "integer 1-5" for dimension in JUDGE_DIMENSIONS},
            "B": {dimension: "integer 1-5" for dimension in JUDGE_DIMENSIONS},
        },
        "reason": "one short sentence",
    }
    system = (
        "You are a strict evaluator of short children's story continuations. Compare two anonymous "
        "continuations for the same prompt. Judge prompt relevance, causal coherence, grammatical "
        "fluency, absence of repetition, and story completeness. Ignore model identity and output "
        "only a JSON object matching the requested schema."
    )
    user = (
        f"PROMPT:\n{prompt}\n\nCANDIDATE A:\n{candidate_a}\n\nCANDIDATE B:\n"
        f"{candidate_b}\n\nJSON SCHEMA EXAMPLE:\n{json.dumps(schema)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def validate_judgment(value: dict[str, Any]) -> dict[str, Any]:
    winner = value.get("winner")
    if winner not in {"A", "B", "tie"}:
        raise ValueError("judge winner must be A, B, or tie")
    scores = value.get("scores")
    if not isinstance(scores, dict):
        raise ValueError("judge response has no scores")
    normalized: dict[str, dict[str, int]] = {}
    for side in ("A", "B"):
        if not isinstance(scores.get(side), dict):
            raise ValueError(f"judge response has no {side} scores")
        normalized[side] = {}
        for dimension in JUDGE_DIMENSIONS:
            score = int(scores[side][dimension])
            if score < 1 or score > 5:
                raise ValueError("judge scores must be between 1 and 5")
            normalized[side][dimension] = score
    return {"winner": winner, "scores": normalized, "reason": str(value.get("reason", ""))}


def bootstrap_mean_interval(
    values: list[float], seed: int = 4242, samples: int = 2000
) -> tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[indices].mean(axis=1)
    return float(array.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def cohens_kappa(first: list[str], second: list[str]) -> float:
    if len(first) != len(second) or not first:
        raise ValueError("kappa requires equal non-empty label lists")
    labels = ("A", "B", "tie")
    observed = sum(left == right for left, right in zip(first, second, strict=True)) / len(first)
    expected = sum(
        (first.count(label) / len(first)) * (second.count(label) / len(second)) for label in labels
    )
    return 1.0 if expected == 1.0 else (observed - expected) / (1.0 - expected)
