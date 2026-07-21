import json
from pathlib import Path

import torch

from experiment import JsonlLogger, atomic_json, atomic_torch_save


def test_atomic_artifacts_and_jsonl(tmp_path: Path) -> None:
    atomic_json(tmp_path / "record.json", {"value": 3})
    atomic_torch_save(tmp_path / "state.pt", {"tensor": torch.arange(3)})
    logger = JsonlLogger(tmp_path / "metrics.jsonl")
    logger.write({"step": 1, "loss": 2.0})
    logger.write({"step": 2, "loss": 1.0})

    assert json.loads((tmp_path / "record.json").read_text())["value"] == 3
    loaded = torch.load(tmp_path / "state.pt", weights_only=True)
    assert torch.equal(loaded["tensor"], torch.arange(3))
    events = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [event["step"] for event in events] == [1, 2]
