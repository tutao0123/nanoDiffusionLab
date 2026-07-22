import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from evaluation import (
    bootstrap_mean_interval,
    cohens_kappa,
    load_validation_prompts,
    local_text_metrics,
    stable_seed,
    validate_judgment,
)
from scripts.benchmark_generation import request_judgment


def test_prompt_selection_respects_document_boundaries(tmp_path: Path) -> None:
    shard_dir = tmp_path / "validation"
    shard_dir.mkdir()
    tokens = np.asarray([1, 2, 3, 9, 4, 5, 6, 9, 7, 8, 0, 9], dtype=np.uint16)
    tokens.tofile(shard_dir / "00000.bin")
    manifest = {
        "format": "nanodiffusion-token-shards/1",
        "dtype": "uint16",
        "tokenizer": {"eot_token_id": 9},
        "splits": {"validation": {"shards": [{"path": "validation/00000.bin"}]}},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    first = load_validation_prompts(manifest_path, count=3, prompt_length=3, seed=7)
    second = load_validation_prompts(manifest_path, count=3, prompt_length=3, seed=7)

    assert first == second
    assert {tuple(record["token_ids"]) for record in first} == {
        (1, 2, 3),
        (4, 5, 6),
        (7, 8, 0),
    }


def test_local_metrics_detect_ending_and_repetition() -> None:
    metrics = local_text_metrics([1, 2, 1, 2, 1, 2, 1, 99, 4], eot_token_id=99)

    assert metrics["ended"] == 1.0
    assert metrics["length"] == 7.0
    assert metrics["distinct_1"] == pytest.approx(2 / 7)
    assert metrics["repeated_4gram_ratio"] > 0


def test_judgment_validation_and_statistics_are_deterministic() -> None:
    scores = {
        side: {
            name: 4
            for name in ("relevance", "coherence", "fluency", "non_repetition", "completeness")
        }
        for side in ("A", "B")
    }
    result = validate_judgment({"winner": "tie", "scores": scores, "reason": "equal"})
    assert result["winner"] == "tie"
    assert stable_seed("sample", 1) == stable_seed("sample", 1)
    assert bootstrap_mean_interval([0.0, 1.0], samples=100)[0] == 0.5
    assert cohens_kappa(["A", "B", "tie"], ["A", "B", "tie"]) == 1.0

    with pytest.raises(ValueError, match="winner"):
        validate_judgment({"winner": "unknown", "scores": scores})


def test_deepseek_request_uses_json_non_thinking_mode() -> None:
    payload = {
        "winner": "A",
        "scores": {
            side: {
                name: 4
                for name in (
                    "relevance",
                    "coherence",
                    "fluency",
                    "non_repetition",
                    "completeness",
                )
            }
            for side in ("A", "B")
        },
        "reason": "A is clearer.",
    }
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        usage = SimpleNamespace(model_dump=lambda: {"total_tokens": 10})
        message = SimpleNamespace(content=json.dumps(payload))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=usage,
            model="deepseek-v4-flash",
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    pair = {
        "pair_id": "seed1337:steps8:p0",
        "train_seed": 1337,
        "steps": 8,
        "prompt": "Once upon a time",
        "A_label": "ar-seed1337",
        "A_objective": "autoregressive",
        "A_text": "there was a fox.",
        "B_label": "mdlm-seed1337",
        "B_objective": "masked_diffusion",
        "B_text": "there was a dog.",
    }

    result = request_judgment(client, pair, "deepseek-v4-flash", max_retries=1)

    assert result["judgment"]["winner"] == "A"
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["extra_body"]["thinking"] == {"type": "disabled"}
    assert captured["temperature"] == 0
