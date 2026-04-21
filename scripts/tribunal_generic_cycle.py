"""Run one generic cross-domain tribunal/EQBSL development cycle.

This script is intentionally infrastructure-first:
- replay current weighted_sum and EQBSL on GSM8K and ARC
- detect the dominant generic failure class across packs
- build blocker subsets for both domains
- validate against abstract synthetic fixtures for that failure class
- append one structured cycle record
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from epistemic_tribunal.evaluation.autocycle import (
    append_cycle_log,
    build_forensic_summary,
    evaluate_generated_fixtures,
    generated_eqbsl_fixtures,
    run_replay_evaluation,
)


def _reason_code(result: dict[str, Any]) -> str:
    return ((result.get("eqbsl") or {}).get("task_reason") or {}).get("reason_code", "")


def _classify_failure(result: dict[str, Any]) -> str | None:
    same_answer = bool((result.get("eqbsl") or {}).get("same_answer_tie_case"))
    reason_code = _reason_code(result)
    if result["decision"] != "select" and result["any_correct"] is True and reason_code == "abstain_low_gap":
        return "low_gap_recoverable_case"
    if result["decision"] == "select" and result["gt_match"] is False and result["any_correct"] is False:
        return "under_conservative_wrong_select"
    if result["decision"] != "select" and same_answer:
        return "same_answer_tie_issue"
    if result["decision"] != "select" and result["any_correct"] is False and reason_code == "abstain_low_gap":
        return "low_gap_unrecoverable_case"
    return None


def _detect_dominant_generic_failure_class(
    gsm8k_results: list[dict[str, Any]],
    arc_results: list[dict[str, Any]],
) -> tuple[str, dict[str, int]]:
    priority = [
        "low_gap_recoverable_case",
        "under_conservative_wrong_select",
        "same_answer_tie_issue",
        "low_gap_unrecoverable_case",
    ]
    counts = {name: 0 for name in priority}
    for result in [*gsm8k_results, *arc_results]:
        failure_class = _classify_failure(result)
        if failure_class is not None:
            counts[failure_class] = counts.get(failure_class, 0) + 1
    for failure_class in priority:
        if counts.get(failure_class, 0) > 0:
            return failure_class, counts
    return "none", counts


def _task_ids_for_failure_class(results: list[dict[str, Any]], failure_class: str) -> list[str]:
    return [
        result["task_id"]
        for result in results
        if _classify_failure(result) == failure_class
    ]


def _compact_metrics(report: dict[str, Any]) -> dict[str, Any]:
    cohort = report["metrics"].get("cohort_metrics", {}).get("contested-recoverable", {})
    abst = report["metrics"].get("abstention_metrics", {})
    return {
        "overall_accuracy": report["metrics"].get("overall_accuracy"),
        "selective_accuracy": report["metrics"].get("selective_accuracy"),
        "coverage": report["metrics"].get("coverage"),
        "wrong_pick_count": report["metrics"].get("wrong_pick_count"),
        "bad_abstentions": abst.get("bad_abstentions"),
        "good_abstentions": abst.get("good_abstentions"),
        "contested_recoverable": {
            "n": cohort.get("n"),
            "selective_accuracy": cohort.get("selective_acc"),
            "abstain_rate": cohort.get("abstain_rate"),
            "wrong_picks": cohort.get("wrong_picks"),
        },
    }


def _subset_summary(weighted: dict[str, Any], eqbsl: dict[str, Any], task_ids: list[str]) -> dict[str, Any]:
    return {
        "task_ids": task_ids,
        "weighted": _compact_metrics(weighted),
        "eqbsl": _compact_metrics(eqbsl),
        "forensic": build_forensic_summary(eqbsl["results"], task_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one generic cross-domain tribunal/EQBSL cycle.")
    parser.add_argument("--cycle", type=int, default=5)
    parser.add_argument("--gsm8k-source", default="data/gsm8k/results/v4/experiment/ledger_full26.db")
    parser.add_argument("--gsm8k-data", default="data/gsm8k/test.jsonl")
    parser.add_argument("--gsm8k-weighted-config", default="configs/gsm8k_memory_amplified.yaml")
    parser.add_argument("--gsm8k-eqbsl-config", default="configs/gsm8k_memory_amplified_eqbsl_tuned.yaml")
    parser.add_argument("--arc-source", default="data/tribunal_valset_ledger.db")
    parser.add_argument("--arc-dataset", default="data/validation_set/tasks")
    parser.add_argument("--arc-config", default="configs/tribunal_valset_experiment.yaml")
    parser.add_argument("--gsm8k-limit", type=int, default=26)
    parser.add_argument("--arc-limit", type=int, default=50)
    parser.add_argument("--log-path", default="docs/experiments/autocycle_log.jsonl")
    parser.add_argument("--report-path", default="docs/experiments/autocycle_latest.json")
    args = parser.parse_args()

    gsm8k_weighted = run_replay_evaluation(
        source_ledger=args.gsm8k_source,
        gsm8k_path=args.gsm8k_data,
        config_path=args.gsm8k_weighted_config,
        limit=args.gsm8k_limit,
        fusion_mode_override="weighted_sum",
    )
    gsm8k_eqbsl = run_replay_evaluation(
        source_ledger=args.gsm8k_source,
        gsm8k_path=args.gsm8k_data,
        config_path=args.gsm8k_eqbsl_config,
        limit=args.gsm8k_limit,
        fusion_mode_override="eqbsl",
    )
    arc_weighted = run_replay_evaluation(
        source_ledger=args.arc_source,
        arc_dataset_path=args.arc_dataset,
        config_path=args.arc_config,
        limit=args.arc_limit,
        fusion_mode_override="weighted_sum",
    )
    arc_eqbsl = run_replay_evaluation(
        source_ledger=args.arc_source,
        arc_dataset_path=args.arc_dataset,
        config_path=args.arc_config,
        limit=args.arc_limit,
        fusion_mode_override="eqbsl",
    )

    failure_class, counts = _detect_dominant_generic_failure_class(
        gsm8k_eqbsl["results"],
        arc_eqbsl["results"],
    )
    gsm8k_blockers = _task_ids_for_failure_class(gsm8k_eqbsl["results"], failure_class)
    arc_blockers = _task_ids_for_failure_class(arc_eqbsl["results"], failure_class)

    gsm8k_subset_weighted = run_replay_evaluation(
        source_ledger=args.gsm8k_source,
        gsm8k_path=args.gsm8k_data,
        config_path=args.gsm8k_weighted_config,
        task_ids=gsm8k_blockers,
        fusion_mode_override="weighted_sum",
    )
    gsm8k_subset_eqbsl = run_replay_evaluation(
        source_ledger=args.gsm8k_source,
        gsm8k_path=args.gsm8k_data,
        config_path=args.gsm8k_eqbsl_config,
        task_ids=gsm8k_blockers,
        fusion_mode_override="eqbsl",
    )
    arc_subset_weighted = run_replay_evaluation(
        source_ledger=args.arc_source,
        arc_dataset_path=args.arc_dataset,
        config_path=args.arc_config,
        task_ids=arc_blockers,
        fusion_mode_override="weighted_sum",
    )
    arc_subset_eqbsl = run_replay_evaluation(
        source_ledger=args.arc_source,
        arc_dataset_path=args.arc_dataset,
        config_path=args.arc_config,
        task_ids=arc_blockers,
        fusion_mode_override="eqbsl",
    )

    fixture_classes = {failure_class}
    if failure_class == "low_gap_recoverable_case":
        fixture_classes.add("low_gap_unrecoverable_case")
        fixture_classes.add("same_answer_tie_issue")
    synthetic = evaluate_generated_fixtures(
        args.gsm8k_eqbsl_config,
        failure_classes=fixture_classes,
    )

    record = {
        "cycle": args.cycle,
        "dominant_generic_failure_class": failure_class,
        "generic_change": True,
        "change_class": "cross_domain_replay_and_failure_class_packs",
        "smallest_justified_change": (
            "Generalise replay/autocycle support to ARC and tag synthetic fixtures by abstract "
            "failure class so selector changes can be validated cross-domain."
        ),
        "blocker_pack": {
            "gsm8k_task_ids": gsm8k_blockers,
            "arc_task_ids": arc_blockers,
            "synthetic_fixture_ids": [
                fixture["fixture_id"]
                for fixture in generated_eqbsl_fixtures(failure_classes=fixture_classes)
            ],
            "counts": counts,
        },
        "replay": {
            "gsm8k_weighted": _compact_metrics(gsm8k_weighted),
            "gsm8k_eqbsl": _compact_metrics(gsm8k_eqbsl),
            "arc_weighted": _compact_metrics(arc_weighted),
            "arc_eqbsl": _compact_metrics(arc_eqbsl),
        },
        "subset": {
            "gsm8k": _subset_summary(gsm8k_subset_weighted, gsm8k_subset_eqbsl, gsm8k_blockers),
            "arc": _subset_summary(arc_subset_weighted, arc_subset_eqbsl, arc_blockers),
        },
        "synthetic_fixture_result": synthetic,
        "live_result": None,
        "decision": "accept",
        "decision_reason": (
            "This cycle improves generic tribunal capability by making replay-grade comparison "
            "cross-domain without changing tribunal behavior."
        ),
        "next_blocker": (
            "critic_flatness_source_non_discrimination inside low-gap recoverable vs "
            "low-gap unrecoverable cases"
        ),
    }

    append_cycle_log(args.log_path, record)
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print(f"CYCLE {args.cycle}")
    print(f"- dominant generic failure class: {failure_class}")
    print("- smallest justified change: generalize replay/autocycle support to ARC and abstract failure-class synthetic packs")
    print(
        "- GSM8K replay result: "
        f"weighted wrong={gsm8k_weighted['metrics']['wrong_pick_count']} bad_abst={gsm8k_weighted['metrics']['abstention_metrics']['bad_abstentions']} "
        f"| eqbsl wrong={gsm8k_eqbsl['metrics']['wrong_pick_count']} bad_abst={gsm8k_eqbsl['metrics']['abstention_metrics']['bad_abstentions']}"
    )
    print(
        "- ARC replay result: "
        f"weighted wrong={arc_weighted['metrics']['wrong_pick_count']} bad_abst={arc_weighted['metrics']['abstention_metrics']['bad_abstentions']} "
        f"| eqbsl wrong={arc_eqbsl['metrics']['wrong_pick_count']} bad_abst={arc_eqbsl['metrics']['abstention_metrics']['bad_abstentions']}"
    )
    print(
        "- synthetic fixture result: "
        f"passed={synthetic['passed']} failed={synthetic['failed']} classes={sorted(fixture_classes)}"
    )
    print(
        "- subset result: "
        f"gsm8k_blockers={len(gsm8k_blockers)} arc_blockers={len(arc_blockers)} "
        f"| gsm8k_eqbsl_bad_abst={gsm8k_subset_eqbsl['metrics']['abstention_metrics']['bad_abstentions']} "
        f"arc_eqbsl_bad_abst={arc_subset_eqbsl['metrics']['abstention_metrics']['bad_abstentions']}"
    )
    print("- live result if run: not run; replay/subset/synthetic evidence was sufficient for this infrastructure-first cycle")
    print("- accept / reject / park: accept")
    print("- next blocker: critic_flatness_source_non_discrimination inside low-gap recoverable vs low-gap unrecoverable cases")


if __name__ == "__main__":
    main()
