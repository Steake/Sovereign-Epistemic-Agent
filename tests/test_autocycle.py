from __future__ import annotations

import sqlite3

from epistemic_tribunal.evaluation.autocycle import (
    detect_dominant_blocker,
    evaluate_generated_fixtures,
    generated_eqbsl_fixtures,
    load_arc_tasks_from_dataset,
)


def test_detect_dominant_blocker_prioritises_unrecoverable_wrong_select() -> None:
    results = [
        {
            "task_id": "u1",
            "decision": "select",
            "gt_match": False,
            "any_correct": False,
            "eqbsl": {"task_reason": {"reason_code": "select_strong_expectation"}},
        },
        {
            "task_id": "b1",
            "decision": "abstain",
            "gt_match": None,
            "any_correct": True,
            "eqbsl": {"task_reason": {"reason_code": "abstain_low_gap"}},
        },
    ]
    blocker = detect_dominant_blocker(results)
    assert blocker["blocker"] == "unrecoverable_wrong_select"
    assert blocker["task_ids"] == ["u1"]


def test_generated_fixture_bank_contains_expected_cases() -> None:
    fixtures = generated_eqbsl_fixtures()
    fixture_ids = {fixture["fixture_id"] for fixture in fixtures}
    assert "same_answer_tie_strong_select" in fixture_ids
    assert "recoverable_low_gap_should_select" in fixture_ids
    assert "memory_heavy_low_gap_abstain" in fixture_ids


def test_generated_fixture_bank_filters_by_failure_class() -> None:
    fixtures = generated_eqbsl_fixtures(failure_classes={"low_gap_recoverable_case"})
    assert {fixture["fixture_id"] for fixture in fixtures} == {"recoverable_low_gap_should_select"}
    assert all(fixture["failure_class"] == "low_gap_recoverable_case" for fixture in fixtures)


def test_load_arc_tasks_from_dataset_uses_ledger_order(tmp_path) -> None:
    dataset_dir = tmp_path / "tasks"
    dataset_dir.mkdir()
    (dataset_dir / "arc_a.json").write_text(
        '{"train":[{"input":[[0]],"output":[[1]]}],"test":[{"input":[[0]],"output":[[1]]}]}',
        encoding="utf-8",
    )
    (dataset_dir / "arc_b.json").write_text(
        '{"train":[{"input":[[0]],"output":[[2]]}],"test":[{"input":[[0]],"output":[[2]]}]}',
        encoding="utf-8",
    )

    db_path = tmp_path / "arc.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE tasks (task_id TEXT, domain TEXT, description TEXT, train_examples_count INTEGER, created_at TEXT)")
    conn.execute("INSERT INTO tasks VALUES (?, ?, '', 1, '')", ("arc_b", "arc_like"))
    conn.execute("INSERT INTO tasks VALUES (?, ?, '', 1, '')", ("arc_a", "arc_like"))
    conn.commit()
    conn.close()

    tasks = load_arc_tasks_from_dataset(str(db_path), str(dataset_dir))
    assert list(tasks.keys()) == ["arc_b", "arc_a"]
    assert tasks["arc_b"].ground_truth == [[2]]
    assert tasks["arc_a"].ground_truth == [[1]]


def test_tuned_eqbsl_fixture_eval_passes_recoverable_low_gap_guard() -> None:
    report = evaluate_generated_fixtures("configs/gsm8k_memory_amplified_eqbsl_tuned.yaml")
    failures = {row["fixture_id"] for row in report["fixtures"] if not row["passed"]}
    assert "recoverable_low_gap_should_select" not in failures
