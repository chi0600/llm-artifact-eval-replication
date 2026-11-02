#!/usr/bin/env python3
"""Validate artifact spec, run config, and optional result instances against contracts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def validate_artifact_spec_doc(doc: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    _require(isinstance(doc, dict), "artifact spec must be an object", errors)
    required = ["artifact_type", "required_fields", "gates", "scoring", "aggregation"]
    for key in required:
        _require(key in doc, f"artifact spec missing '{key}'", errors)

    if "required_fields" in doc:
        _require(isinstance(doc["required_fields"], list) and doc["required_fields"], "required_fields must be a non-empty array", errors)
    if "gates" in doc:
        _require(isinstance(doc["gates"], list) and doc["gates"], "gates must be a non-empty array", errors)
        for idx, gate in enumerate(doc["gates"]):
            _require(isinstance(gate, dict), f"gates[{idx}] must be an object", errors)
            for key in ["id", "stage", "oracle_type", "blocking"]:
                _require(key in gate, f"gates[{idx}] missing '{key}'", errors)
    if "scoring" in doc:
        _require(isinstance(doc["scoring"], list) and doc["scoring"], "scoring must be a non-empty array", errors)
        for idx, score in enumerate(doc["scoring"]):
            _require(isinstance(score, dict), f"scoring[{idx}] must be an object", errors)
            for key in ["id", "weight", "min", "max"]:
                _require(key in score, f"scoring[{idx}] missing '{key}'", errors)
    if "aggregation" in doc:
        agg = doc["aggregation"]
        _require(isinstance(agg, dict), "aggregation must be an object", errors)
        if isinstance(agg, dict):
            for key in ["policy", "pass_threshold"]:
                _require(key in agg, f"aggregation missing '{key}'", errors)
    if "monitoring" in doc:
        monitoring = doc["monitoring"]
        _require(isinstance(monitoring, dict), "monitoring must be an object", errors)
        if isinstance(monitoring, dict):
            for key in ["comparison_keys", "monitored_metrics", "review_triggers"]:
                _require(key in monitoring, f"monitoring missing '{key}'", errors)
                if key in monitoring:
                    _require(
                        isinstance(monitoring[key], list) and monitoring[key],
                        f"monitoring.{key} must be a non-empty array",
                        errors,
                    )

    return errors


def validate_run_config_doc(doc: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = ["artifact_spec", "dataset", "model", "gates_enabled", "aggregation_policy"]
    for key in required:
        _require(key in doc, f"run config missing '{key}'", errors)

    dataset = doc.get("dataset")
    if dataset is not None:
        _require(isinstance(dataset, dict), "dataset must be an object", errors)
        if isinstance(dataset, dict):
            for key in ["path", "format"]:
                _require(key in dataset, f"dataset missing '{key}'", errors)

    model = doc.get("model")
    if model is not None:
        _require(isinstance(model, dict), "model must be an object", errors)
        if isinstance(model, dict):
            for key in ["provider", "name", "temperature", "max_tokens"]:
                _require(key in model, f"model missing '{key}'", errors)

    gates = doc.get("gates_enabled")
    if gates is not None:
        _require(isinstance(gates, list) and gates, "gates_enabled must be a non-empty array", errors)

    policy = doc.get("aggregation_policy")
    if policy is not None:
        _require(isinstance(policy, dict), "aggregation_policy must be an object", errors)
        if isinstance(policy, dict):
            for key in ["decision_rule", "min_score"]:
                _require(key in policy, f"aggregation_policy missing '{key}'", errors)

    return errors


def validate_result_doc(doc: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = ["run_id", "artifact_id", "verdict", "gates", "scores", "overall_score", "diagnostics"]
    for key in required:
        _require(key in doc, f"result missing '{key}'", errors)

    gates = doc.get("gates")
    if gates is not None:
        _require(isinstance(gates, list) and gates, "gates must be a non-empty array", errors)
        if isinstance(gates, list):
            for idx, gate in enumerate(gates):
                _require(isinstance(gate, dict), f"gates[{idx}] must be an object", errors)
                if isinstance(gate, dict):
                    for key in ["id", "stage", "oracle_type", "blocking", "passed"]:
                        _require(key in gate, f"gates[{idx}] missing '{key}'", errors)

    scores = doc.get("scores")
    if scores is not None:
        _require(isinstance(scores, dict) and scores, "scores must be a non-empty object", errors)

    diagnostics = doc.get("diagnostics")
    if diagnostics is not None:
        _require(isinstance(diagnostics, dict), "diagnostics must be an object", errors)
        if isinstance(diagnostics, dict):
            for key in ["pipeline_stage", "notes"]:
                _require(key in diagnostics, f"diagnostics missing '{key}'", errors)

    return errors


def validate_with_jsonschema(instance: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    try:
        import jsonschema
    except ImportError:
        return ["jsonschema_not_installed"]

    try:
        jsonschema.validate(instance=instance, schema=schema)
        return []
    except jsonschema.ValidationError as exc:
        return [f"jsonschema validation error: {exc.message}"]


def run_validation(instance: dict[str, Any], schema: dict[str, Any], fallback_validator: Any) -> list[str]:
    errors = validate_with_jsonschema(instance, schema)
    if errors == ["jsonschema_not_installed"]:
        return fallback_validator(instance)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate evaluation contract files")
    parser.add_argument("--artifact-spec", required=True, help="Path to artifact spec JSON")
    parser.add_argument("--run-config", required=True, help="Path to run config JSON")
    parser.add_argument("--result", help="Optional path to result JSON")
    args = parser.parse_args()

    root = Path.cwd()
    schema_dir = root / "evaluation" / "contracts"

    artifact_schema = load_json(schema_dir / "artifact-spec.schema.json")
    run_schema = load_json(schema_dir / "run-config.schema.json")
    result_schema = load_json(schema_dir / "result.schema.json")

    artifact_doc = load_json(root / args.artifact_spec)
    run_doc = load_json(root / args.run_config)

    failures: list[str] = []

    artifact_errors = run_validation(artifact_doc, artifact_schema, validate_artifact_spec_doc)
    failures.extend([f"artifact spec: {msg}" for msg in artifact_errors])

    run_errors = run_validation(run_doc, run_schema, validate_run_config_doc)
    failures.extend([f"run config: {msg}" for msg in run_errors])

    if args.result:
        result_doc = load_json(root / args.result)
        result_errors = run_validation(result_doc, result_schema, validate_result_doc)
        failures.extend([f"result: {msg}" for msg in result_errors])

    if failures:
        print("Contract validation failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Contract validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
