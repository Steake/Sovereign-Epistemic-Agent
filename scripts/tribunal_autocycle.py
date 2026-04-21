"""Run one autonomous tribunal/EQBSL development cycle.

The script is intentionally conservative:
- detect the dominant blocker on the current EQBSL config
- build a compact blocker slice
- compare weighted baseline, current EQBSL, and one candidate EQBSL config
- validate the candidate on generated EQBSL fixtures
- append one structured log record
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from epistemic_tribunal.evaluation.autocycle import (
    append_cycle_log,
    build_seed_signatures,
    build_forensic_summary,
    decide_cycle_outcome,
    detect_dominant_blocker,
    evaluate_generated_fixtures,
    run_replay_evaluation,
)


def _expand_slice(current_results: list[dict[str, Any]], blocker: dict[str, Any]) -> list[str]:
    task_ids = set(blocker["task_ids"])
    if blocker["blocker"] == "unrecoverable_wrong_select":
        for result in current_results:
            eqbsl = result.get("eqbsl", {})
            top_gap = float(eqbsl.get("top_gap") or 1.0)
            if eqbsl.get("same_answer_tie_case") and top_gap <= 0.05:
                task_ids.add(result["task_id"])
    elif blocker["blocker"] in {"bad_abstention", "same_answer_tie_nonselect", "low_gap_nonselect"}:
        for result in current_results:
            eqbsl = result.get("eqbsl", {})
            if (eqbsl.get("task_reason") or {}).get("reason_code") == "abstain_low_gap":
                task_ids.add(result["task_id"])
    return sorted(task_ids)


def _compact_metrics(report: dict[str, Any]) -> dict[str, Any]:
    cohort = report["metrics"].get("cohort_metrics", {}).get("contested-recoverable", {})
    abst = report["metrics"].get("abstention_metrics", {})
    eq = report["metrics"].get("eqbsl_diagnostics", {})
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
        "eqbsl": {
            "mean_coalition_belief": eq.get("mean_coalition_belief"),
            "mean_coalition_uncertainty": eq.get("mean_coalition_uncertainty"),
            "same_answer_tie_cases_correctly_selected": eq.get(
                "same_answer_tie_cases_correctly_selected"
            ),
            "abstentions_caused_by_high_uncertainty": eq.get(
                "abstentions_caused_by_high_uncertainty"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one autonomous tribunal/EQBSL cycle.")
    parser.add_argument(
        "--source",
        default="data/gsm8k/results/v4/experiment/ledger_full26.db",
        help="Truth-source replay ledger.",
    )
    parser.add_argument(
        "--gsm8k",
        default="data/gsm8k/test.jsonl",
        help="GSM8K JSONL used for ground truth lookup.",
    )
    parser.add_argument(
        "--weighted-config",
        default="configs/gsm8k_memory_amplified.yaml",
        help="Weighted-sum baseline config.",
    )
    parser.add_argument(
        "--current-eqbsl-config",
        default="configs/gsm8k_memory_amplified_eqbsl_tuned.yaml",
        help="Current EQBSL config under evaluation.",
    )
    parser.add_argument(
        "--candidate-eqbsl-config",
        default="configs/gsm8k_memory_amplified_eqbsl.yaml",
        help="Candidate EQBSL config for this cycle.",
    )
    parser.add_argument("--limit", type=int, default=26)
    parser.add_argument("--cycle", type=int, default=1)
    parser.add_argument(
        "--log-path",
        default="docs/experiments/autocycle_log.jsonl",
        help="JSONL log path for structured cycle records.",
    )
    parser.add_argument(
        "--report-path",
        default="docs/experiments/autocycle_latest.json",
        help="Optional JSON report artifact.",
    )
    args = parser.parse_args()

    shared_full_seed = build_seed_signatures(
        source_ledger=args.source,
        gsm8k_path=args.gsm8k,
        config_path=args.weighted_config,
        limit=args.limit,
    )

    weighted_full = run_replay_evaluation(
        source_ledger=args.source,
        gsm8k_path=args.gsm8k,
        config_path=args.weighted_config,
        limit=args.limit,
        seed_signatures=shared_full_seed["signatures"],
    )
    current_full = run_replay_evaluation(
        source_ledger=args.source,
        gsm8k_path=args.gsm8k,
        config_path=args.current_eqbsl_config,
        limit=args.limit,
        seed_signatures=shared_full_seed["signatures"],
    )
    candidate_full = run_replay_evaluation(
        source_ledger=args.source,
        gsm8k_path=args.gsm8k,
        config_path=args.candidate_eqbsl_config,
        limit=args.limit,
        seed_signatures=shared_full_seed["signatures"],
    )

    blocker = detect_dominant_blocker(current_full["results"])
    slice_task_ids = _expand_slice(current_full["results"], blocker)
    shared_subset_seed = build_seed_signatures(
        source_ledger=args.source,
        gsm8k_path=args.gsm8k,
        config_path=args.weighted_config,
        task_ids=slice_task_ids,
    )

    weighted_subset = run_replay_evaluation(
        source_ledger=args.source,
        gsm8k_path=args.gsm8k,
        config_path=args.weighted_config,
        task_ids=slice_task_ids,
        seed_signatures=shared_subset_seed["signatures"],
    )
    current_subset = run_replay_evaluation(
        source_ledger=args.source,
        gsm8k_path=args.gsm8k,
        config_path=args.current_eqbsl_config,
        task_ids=slice_task_ids,
        seed_signatures=shared_subset_seed["signatures"],
    )
    candidate_subset = run_replay_evaluation(
        source_ledger=args.source,
        gsm8k_path=args.gsm8k,
        config_path=args.candidate_eqbsl_config,
        task_ids=slice_task_ids,
        seed_signatures=shared_subset_seed["signatures"],
    )

    current_generated = evaluate_generated_fixtures(args.current_eqbsl_config)
    candidate_generated = evaluate_generated_fixtures(args.candidate_eqbsl_config)
    outcome = decide_cycle_outcome(
        current_full=current_full,
        candidate_full=candidate_full,
        current_subset=current_subset,
        candidate_subset=candidate_subset,
        current_generated=current_generated,
        candidate_generated=candidate_generated,
    )

    forensic = build_forensic_summary(current_full["results"], slice_task_ids)
    record = {
        "cycle": args.cycle,
        "hypothesis": "Tightening low-gap EQBSL abstention should reduce unrecoverable wrong selections.",
        "change_class": "EQBSL decision-policy threshold tuning",
        "weighted_config": args.weighted_config,
        "current_eqbsl_config": args.current_eqbsl_config,
        "candidate_eqbsl_config": args.candidate_eqbsl_config,
        "dominant_blocker": blocker,
        "slice_task_ids": slice_task_ids,
        "forensic_summary": forensic,
        "replay": {
            "weighted_full": _compact_metrics(weighted_full),
            "current_full": _compact_metrics(current_full),
            "candidate_full": _compact_metrics(candidate_full),
        },
        "subset": {
            "weighted": _compact_metrics(weighted_subset),
            "current": _compact_metrics(current_subset),
            "candidate": _compact_metrics(candidate_subset),
        },
        "generated_fixtures": {
            "current": current_generated,
            "candidate": candidate_generated,
        },
        "live_result": None,
        "outcome": outcome,
        "next_blocker": (
            "source discrimination / oracle limits on low-gap same-answer-tie families"
            if outcome["decision"] != "accept"
            else "re-run blocker detection after candidate promotion"
        ),
    }

    append_cycle_log(args.log_path, record)
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print(f"CYCLE {args.cycle}")
    print(f"- dominant blocker: {blocker['blocker']} ({blocker['count']})")
    print("- smallest justified change: raise low-gap abstention protection by using the safer EQBSL candidate config")
    print(
        "- replay result: "
        f"current wrong={current_full['metrics']['wrong_pick_count']} bad_abst={current_full['metrics']['abstention_metrics']['bad_abstentions']} | "
        f"candidate wrong={candidate_full['metrics']['wrong_pick_count']} bad_abst={candidate_full['metrics']['abstention_metrics']['bad_abstentions']}"
    )
    print(
        "- subset result: "
        f"current wrong={current_subset['metrics']['wrong_pick_count']} bad_abst={current_subset['metrics']['abstention_metrics']['bad_abstentions']} | "
        f"candidate wrong={candidate_subset['metrics']['wrong_pick_count']} bad_abst={candidate_subset['metrics']['abstention_metrics']['bad_abstentions']}"
    )
    print(
        "- generated-task result: "
        f"current failed={current_generated['failed']} | candidate failed={candidate_generated['failed']}"
    )
    print("- live result if run: not run; replay/subset/generated evidence did not justify promotion")
    print(f"- accept/reject/park: {outcome['decision']} ({outcome['reason']})")
    print(f"- next blocker: {record['next_blocker']}")


if __name__ == "__main__":
    main()
