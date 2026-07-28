from scripts.benchmark_remask import build_pairs, candidate_key, pro_subset, record_kind


def record(kind: str, seed: int, prompt_id: int) -> dict:
    objective = "autoregressive" if kind == "ar" else "masked_diffusion"
    return {
        "kind": kind if kind == "remask" else None,
        "objective": objective,
        "train_seed": seed,
        "prompt_id": prompt_id,
        "prompt": "prompt",
        "continuation": kind,
    }


def test_candidate_keys_are_seed_and_prompt_specific() -> None:
    assert candidate_key(1337, 9) == "remask:seed1337:p9"
    assert candidate_key(1337, 9) != candidate_key(2027, 9)


def test_build_pairs_creates_both_comparisons_with_balanced_labels() -> None:
    base = [record("ar", 1337, 0), record("baseline", 1337, 0)]
    remask = [record("remask", 1337, 0)]

    pairs = build_pairs(base, remask)

    assert {pair["comparison"] for pair in pairs} == {
        "remask-vs-ar",
        "remask-vs-baseline",
    }
    expected_labels = ({"remask", "ar"}, {"remask", "baseline"})
    assert all({pair["A_label"], pair["B_label"]} in expected_labels for pair in pairs)
    assert record_kind(remask[0]) == "remask"


def test_pro_subset_selects_requested_count_per_comparison() -> None:
    pairs = [
        {"comparison": comparison, "pair_id": f"{comparison}:{index}"}
        for comparison in ("remask-vs-ar", "remask-vs-baseline")
        for index in range(5)
    ]

    selected = pro_subset(pairs, 2)

    assert len(selected) == 4
    assert sum(pair["comparison"] == "remask-vs-ar" for pair in selected) == 2
