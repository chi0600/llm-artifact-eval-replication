#!/usr/bin/env python3
"""Fixture checks for MBPP/EvalPlus Stage 6 re-evaluation comparison."""

from __future__ import annotations

import tempfile
from pathlib import Path

from run_mbpp_evalplus import (
    OUTCOME_ACCEPTED,
    OUTCOME_CONDITIONAL,
    OUTCOME_REJECTED_BASE,
    OUTCOME_REJECTED_EXECUTION,
    compare_evaluation_runs,
    write_re_evaluation_report,
)


def fixture_row(
    task_id: str,
    outcome: str,
    *,
    base_pass: bool,
    plus_pass: bool,
    failed_stage: str = "",
    failed_gate: str = "",
    failure_cause: str = "",
    dataset_hash: str = "fixture",
    prompt_hash: str = "prompt-a",
    completion_hash: str = "completion-a",
    model: str = "fixture-model",
    provider: str = "fixture",
) -> dict[str, object]:
    return {
        "run_id": "fixture_run",
        "task_id": task_id,
        "pipeline_outcome": outcome,
        "decision_category": outcome,
        "base_pass": base_pass,
        "plus_pass": plus_pass,
        "base_status": "pass" if base_pass else "fail",
        "plus_status": "pass" if plus_pass else "fail",
        "failed_stage": failed_stage,
        "failed_gate": failed_gate,
        "failure_cause": failure_cause,
        "model": model,
        "provider": provider,
        "dataset_hash": dataset_hash,
        "prompt_hash": prompt_hash,
        "completion_hash": completion_hash,
    }


def signal_ids(summary: dict[str, object]) -> set[str]:
    return {
        str(signal["id"])
        for signal in summary["evaluation_specification_review_signals"]  # type: ignore[index]
    }


def recommendation_ids(summary: dict[str, object]) -> set[str]:
    return {
        str(recommendation["id"])
        for recommendation in summary["evaluation_specification_recommendations"]  # type: ignore[index]
    }


def main() -> int:
    baseline = [
        fixture_row("A", OUTCOME_ACCEPTED, base_pass=True, plus_pass=True),
        fixture_row(
            "B",
            OUTCOME_REJECTED_BASE,
            base_pass=False,
            plus_pass=False,
            failed_stage="base_oracle",
            failed_gate="base_oracle",
            failure_cause="base_oracle_failure",
        ),
        fixture_row("C", OUTCOME_ACCEPTED, base_pass=True, plus_pass=True),
        fixture_row(
            "D",
            OUTCOME_REJECTED_BASE,
            base_pass=False,
            plus_pass=False,
            failed_stage="base_oracle",
            failed_gate="base_oracle",
            failure_cause="base_oracle_failure",
        ),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        no_change = compare_evaluation_runs(
            baseline,
            list(baseline),
            tmp / "baseline" / "results.jsonl",
            tmp / "current" / "results.jsonl",
        )
        assert no_change["change_counts"]["regressed"] == 0
        assert no_change["change_counts"]["improved"] == 0
        assert no_change["change_counts"]["unchanged"] == 4
        assert no_change["evaluation_specification_review_signals"] == []
        assert no_change["evaluation_specification_recommendations"] == []

        current = [
            fixture_row(
                "A",
                OUTCOME_CONDITIONAL,
                base_pass=True,
                plus_pass=False,
                failed_stage="extended_oracle",
                failed_gate="extended_oracle",
                failure_cause="extended_oracle_failure",
                dataset_hash="fixture-v2",
                prompt_hash="prompt-b",
                model="fixture-model-v2",
            ),
            fixture_row(
                "B",
                OUTCOME_ACCEPTED,
                base_pass=True,
                plus_pass=True,
                dataset_hash="fixture-v2",
                prompt_hash="prompt-a",
                model="fixture-model-v2",
            ),
            fixture_row(
                "C",
                OUTCOME_REJECTED_EXECUTION,
                base_pass=False,
                plus_pass=False,
                failed_stage="execution",
                failed_gate="harness_setup_ok",
                failure_cause="runtime_error",
                dataset_hash="fixture-v2",
                prompt_hash="prompt-a",
                model="fixture-model-v2",
            ),
            fixture_row(
                "D",
                OUTCOME_CONDITIONAL,
                base_pass=True,
                plus_pass=False,
                failed_stage="extended_oracle",
                failed_gate="extended_oracle",
                failure_cause="extended_oracle_failure",
                dataset_hash="fixture-v2",
                prompt_hash="prompt-a",
                model="fixture-model-v2",
            ),
        ]
        changed = compare_evaluation_runs(
            baseline,
            current,
            tmp / "baseline" / "results.jsonl",
            tmp / "current" / "results.jsonl",
        )
        assert changed["change_counts"]["regressed"] == 2
        assert changed["change_counts"]["improved"] == 2
        signals = signal_ids(changed)
        assert "dataset_hash_changed" in signals
        assert "prompt_hash_changed" in signals
        assert "model_metadata_changed" in signals
        assert "accepted_task_regressed" in signals
        assert "new_failure_causes" in signals
        assert "strict_pass_rate_drop" in signals
        recommendations = recommendation_ids(changed)
        assert "document_model_replacement_context" in recommendations
        assert "add_accepted_regressions_as_sentinels" in recommendations
        assert "review_new_failure_handling" in recommendations
        assert "keep_task_level_regression_guards" in recommendations

        report_path = tmp / "re_evaluation_report.md"
        write_re_evaluation_report(report_path, changed)
        report = report_path.read_text(encoding="utf-8")
        assert "Stage 6 Re-evaluation Report" in report
        assert "Suggested Evaluation Specification Updates" in report

    print("Stage 6 comparison checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
