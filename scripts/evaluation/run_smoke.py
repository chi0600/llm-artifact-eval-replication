#!/usr/bin/env python3
"""Run a deterministic smoke evaluation and write run outputs plus report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from validate_contracts import (
    validate_artifact_spec_doc,
    validate_result_doc,
    validate_run_config_doc,
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ensure_fields(item: dict[str, Any], required_fields: list[str]) -> bool:
    return all(field in item and item[field] not in (None, "") for field in required_fields)


def gate_pass(item: dict[str, Any], gate_id: str) -> bool:
    artifact_text = str(item.get("generated_artifact", ""))
    context_text = str(item.get("input_context", ""))

    if gate_id == "format_schema":
        return bool(artifact_text.strip())
    if gate_id == "source_fidelity":
        return "API" in artifact_text and "report" in artifact_text.lower()
    if gate_id == "sequence_consistency":
        return "->" in artifact_text and "-->" in artifact_text
    if gate_id == "execution_probe":
        return len(artifact_text.splitlines()) >= 3
    if gate_id == "quality_readability":
        return len(artifact_text) > 40 and len(context_text) > 40
    return True


def compute_scores(item: dict[str, Any], scoring: list[dict[str, Any]]) -> tuple[dict[str, float], float]:
    artifact_text = str(item.get("generated_artifact", ""))
    base = 0.88
    if "TODO" in artifact_text:
        base -= 0.2

    score_map: dict[str, float] = {}
    total_weight = 0.0
    weighted_sum = 0.0
    for module in scoring:
        sid = module["id"]
        weight = float(module["weight"])
        min_v = float(module["min"])
        max_v = float(module["max"])
        score = max(min_v, min(max_v, base))
        score_map[sid] = round(score, 3)
        weighted_sum += score * weight
        total_weight += weight

    overall = weighted_sum / total_weight if total_weight else 0.0
    return score_map, round(overall, 3)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run smoke evaluation")
    parser.add_argument("--config", required=True, help="Path to run config JSON")
    parser.add_argument("--output-root", required=True, help="Directory for run outputs")
    parser.add_argument("--report-dir", required=True, help="Directory for report markdown")
    args = parser.parse_args()

    root = Path.cwd()
    cfg_path = root / args.config
    config = load_json(cfg_path)

    config_errors = validate_run_config_doc(config)
    if config_errors:
        for err in config_errors:
            print(f"Run config error: {err}")
        return 1

    spec_path = root / config["artifact_spec"]
    spec = load_json(spec_path)
    spec_errors = validate_artifact_spec_doc(spec)
    if spec_errors:
        for err in spec_errors:
            print(f"Artifact spec error: {err}")
        return 1

    dataset_path = root / config["dataset"]["path"]
    dataset = load_jsonl(dataset_path)

    run_id = config.get("run_id") or datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    output_dir = root / args.output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    enabled = set(config["gates_enabled"])
    all_results: list[dict[str, Any]] = []

    for item in dataset:
        required_ok = ensure_fields(item, spec["required_fields"])
        gates: list[dict[str, Any]] = []

        for gate in spec["gates"]:
            if gate["id"] not in enabled:
                continue
            passed = required_ok and gate_pass(item, gate["id"])
            gates.append(
                {
                    "id": gate["id"],
                    "stage": gate["stage"],
                    "oracle_type": gate["oracle_type"],
                    "blocking": gate["blocking"],
                    "passed": passed,
                    "evidence": "smoke evaluation heuristic"
                }
            )

        scores, overall_score = compute_scores(item, spec["scoring"])
        blocking_ok = all(g["passed"] for g in gates if g["blocking"])
        min_score = float(config["aggregation_policy"]["min_score"])
        decision_rule = config["aggregation_policy"]["decision_rule"]

        if decision_rule == "min_score_only":
            verdict = "pass" if overall_score >= min_score else "fail"
        else:
            verdict = "pass" if (blocking_ok and overall_score >= min_score) else "fail"

        result = {
            "run_id": run_id,
            "artifact_id": str(item.get("artifact_id", "unknown")),
            "verdict": verdict,
            "gates": gates,
            "scores": scores,
            "overall_score": overall_score,
            "diagnostics": {
                "pipeline_stage": "aggregation",
                "notes": "Smoke-run deterministic decision policy",
                "warnings": []
            }
        }

        result_errors = validate_result_doc(result)
        if result_errors:
            for err in result_errors:
                print(f"Result error: {err}")
            return 1

        all_results.append(result)

    results_path = output_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as fh:
        for row in all_results:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")

    pass_count = sum(1 for row in all_results if row["verdict"] == "pass")
    avg_score = round(sum(row["overall_score"] for row in all_results) / max(1, len(all_results)), 3)
    summary = {
        "run_id": run_id,
        "artifacts": len(all_results),
        "pass_count": pass_count,
        "fail_count": len(all_results) - pass_count,
        "average_overall_score": avg_score,
        "result_path": str(results_path.relative_to(root))
    }

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    report_dir = root / args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{run_id}-summary.md"
    report = [
        f"# Smoke Evaluation Summary: {run_id}",
        "",
        f"- Artifacts: {summary['artifacts']}",
        f"- Pass: {summary['pass_count']}",
        f"- Fail: {summary['fail_count']}",
        f"- Average overall score: {summary['average_overall_score']}",
        "",
        "## Artifacts",
        "",
        "| Artifact ID | Verdict | Overall Score |",
        "| --- | --- | ---: |"
    ]
    for row in all_results:
        report.append(f"| {row['artifact_id']} | {row['verdict']} | {row['overall_score']} |")

    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"Smoke run completed: {run_id}")
    print(f"Results: {results_path}")
    print(f"Summary: {summary_path}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
