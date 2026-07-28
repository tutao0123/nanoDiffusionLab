from scripts.validate_remask import aggregate_records, select_candidate, variant_key


def metrics(ended: float, distinct_3: float, repetition: float) -> dict[str, float]:
    return {
        "length": 10.0,
        "ended": ended,
        "unique_token_ratio": 0.5,
        "distinct_1": 0.5,
        "distinct_2": 0.6,
        "distinct_3": distinct_3,
        "repeated_4gram_ratio": repetition,
    }


def test_remask_record_keys_are_stable_and_variant_specific() -> None:
    fixed = {
        "name": "candidate",
        "remask_fraction": 0.1,
        "remask_decay_start": 1.0,
        "protect_eot": False,
    }
    protected = {**fixed, "protect_eot": True}
    different_fraction = {**fixed, "remask_fraction": 0.05}

    assert variant_key("seed", fixed, 7) == (
        "seed:candidate:fraction=0.1:decay=1:protect_eot=0:p7"
    )
    assert variant_key("seed", fixed, 7) != variant_key("seed", protected, 7)
    assert variant_key("seed", fixed, 7) != variant_key("seed", different_fraction, 7)


def test_aggregate_records_groups_metrics_by_variant() -> None:
    records = [
        {"variant": "baseline", "local_metrics": metrics(1.0, 0.7, 0.2)},
        {"variant": "baseline", "local_metrics": metrics(0.0, 0.9, 0.0)},
    ]

    summary = aggregate_records(records)["baseline"]

    assert summary["samples"] == 2.0
    assert summary["ended"] == 0.5
    assert summary["distinct_3"] == 0.8
    assert summary["repeated_4gram_ratio"] == 0.1


def test_candidate_gate_selects_lowest_repetition_among_passing_variants() -> None:
    aggregate = {
        "baseline": metrics(0.2, 0.8, 0.1),
        "passes": metrics(0.18, 0.81, 0.07),
        "best": metrics(0.19, 0.82, 0.05),
        "eot-regression": metrics(0.1, 0.9, 0.01),
    }
    gate = {
        "minimum_relative_repetition_reduction": 0.2,
        "maximum_eot_completion_drop": 0.02,
        "maximum_distinct_3_drop": 0.01,
    }

    selected, decisions = select_candidate(aggregate, gate)

    assert selected == "best"
    assert decisions["passes"]["passed"] is True
    assert decisions["eot-regression"]["passed"] is False
