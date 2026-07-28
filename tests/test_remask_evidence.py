import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "reports/data/tinystories_106m_remask_results.json"
REPORT_PATH = ROOT / "reports/tinystories_106m_remask_generation.md"
GATE_SUMMARY_PATH = ROOT / "reports/data/remask_validation_100.json"
GATE_REPORT_PATH = ROOT / "reports/remask_validation_100.md"


def test_remask_report_matches_bounded_summary() -> None:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    report = REPORT_PATH.read_text(encoding="utf-8")
    settings = summary["settings"]

    assert summary["generation_records"] == 2000
    assert summary["flash_judgments"] == 4000
    assert summary["pro_judgments"] == 200
    assert settings["training_seeds"] == [1337, 2027]
    assert settings["prompt_sha256"] in report
    assert "2,000 new remask continuations" in report
    assert "DeepSeek V4 Flash on 4,000 pairs" in report
    assert "200-pair audit" in report
    assert "seeds 1337 and 2027" in report


def test_remask_performance_summary_has_complete_grid() -> None:
    rows = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))["performance"]

    assert {
        (row["label"], row["train_seed"], row["batch_size"])
        for row in rows
    } == {
        (label, seed, batch)
        for label in ("ar-cached", "mdlm-baseline", "mdlm-remask")
        for seed in (1337, 2027)
        for batch in (1, 8, 32)
    }


def test_remask_gate_report_matches_bounded_summary() -> None:
    summary = json.loads(GATE_SUMMARY_PATH.read_text(encoding="utf-8"))
    report = GATE_REPORT_PATH.read_text(encoding="utf-8")

    assert summary["generation_records"] == 1200
    assert summary["settings"]["training_seeds"] == [1337, 2027]
    assert summary["selection"]["selected_variant"] == "decay-protect-10"
    assert summary["selection"]["gate_passed"] is True
    assert "Total: 1,200 continuations" in report
    assert "10% remask with EOT protection and final-quarter decay" in report
