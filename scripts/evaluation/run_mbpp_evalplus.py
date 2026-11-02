#!/usr/bin/env python3
"""Run the MBPP/EvalPlus reference-pipeline instantiation.

The full run uses EvalPlus for MBPP/MBPP+ oracles and Hugging Face's
OpenAI-compatible router for one Qwen3-Coder completion per selected task.
The fixture run exercises the same pipeline classification without network or
third-party dependencies.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import multiprocessing
import os
import platform
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from validate_contracts import validate_artifact_spec_doc, validate_run_config_doc


OUTCOME_ACCEPTED = "Accepted"
OUTCOME_CONDITIONAL = "Conditional"
OUTCOME_REJECTED_PRE_ORACLE = "Rejected before oracle"
OUTCOME_REJECTED_EXECUTION = "Rejected during execution"
OUTCOME_REJECTED_BASE = "Rejected by base oracle"

OUTCOME_ORDER = [
    OUTCOME_ACCEPTED,
    OUTCOME_CONDITIONAL,
    OUTCOME_REJECTED_PRE_ORACLE,
    OUTCOME_REJECTED_EXECUTION,
    OUTCOME_REJECTED_BASE,
]

OUTCOME_SEVERITY = {
    OUTCOME_ACCEPTED: 2,
    OUTCOME_CONDITIONAL: 1,
    OUTCOME_REJECTED_PRE_ORACLE: 0,
    OUTCOME_REJECTED_EXECUTION: 0,
    OUTCOME_REJECTED_BASE: 0,
}

PASS = "pass"
FAIL = "fail"
TIMEOUT = "timeout"
NOT_RUN = "not_run"
LOCAL_OPENAI_PROVIDERS = {"ollama_openai"}

MONITORED_FIELDS = [
    "base_pass",
    "strict_mbpp_plus_pass",
    "decision_category",
    "failed_stage",
    "failed_gate",
    "failure_cause",
]

RESULT_FIELDS = [
    "run_id",
    "task_id",
    "prompt",
    "entry_point",
    "raw_output_path",
    "raw_output_text",
    "extracted_code",
    "validity_check_result",
    "validity_checks",
    "mbpp_base_result",
    "mbpp_plus_result",
    "decision_category",
    "pipeline_outcome",
    "decision_stage",
    "failed_stage",
    "failed_gate",
    "base_status",
    "plus_status",
    "base_pass",
    "plus_pass",
    "failure_stage",
    "failure_cause",
    "failure_reason",
    "base_fail_count",
    "base_total_count",
    "plus_fail_count",
    "plus_total_count",
    "model",
    "provider",
    "prompt_hash",
    "completion_hash",
    "dataset_hash",
    "selection_rank",
]


FIXTURE_PROMPT = "Write a Python function add_one(x) that returns x plus one."

FIXTURE_PROBLEMS: Dict[str, Dict[str, Any]] = {
    "Fixture/accepted": {
        "task_id": "Fixture/accepted",
        "prompt": FIXTURE_PROMPT,
        "entry_point": "add_one",
        "base_tests": [([1], 2), ([2], 3)],
        "plus_tests": [([10], 11)],
    },
    "Fixture/conditional": {
        "task_id": "Fixture/conditional",
        "prompt": FIXTURE_PROMPT,
        "entry_point": "add_one",
        "base_tests": [([1], 2), ([2], 3)],
        "plus_tests": [([10], 11)],
    },
    "Fixture/pre_oracle": {
        "task_id": "Fixture/pre_oracle",
        "prompt": FIXTURE_PROMPT,
        "entry_point": "add_one",
        "base_tests": [([1], 2)],
        "plus_tests": [([10], 11)],
    },
    "Fixture/execution": {
        "task_id": "Fixture/execution",
        "prompt": FIXTURE_PROMPT,
        "entry_point": "add_one",
        "base_tests": [([1], 2)],
        "plus_tests": [([10], 11)],
    },
    "Fixture/base_reject": {
        "task_id": "Fixture/base_reject",
        "prompt": FIXTURE_PROMPT,
        "entry_point": "add_one",
        "base_tests": [([1], 2), ([2], 3)],
        "plus_tests": [([10], 11)],
    },
}


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(json.dumps(value, ensure_ascii=True, sort_keys=True))


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def selection_key(version: str, task_id: str, seed: int) -> str:
    return sha256_text(f"mbpp-plus-{version}:{task_id}:{seed}")


def validate_basic_contracts(root: Path, config: Dict[str, Any]) -> None:
    config_errors = validate_run_config_doc(config)
    if config_errors:
        raise ValueError("Run config error: " + "; ".join(config_errors))

    spec_path = root / config["artifact_spec"]
    spec = load_json(spec_path)
    spec_errors = validate_artifact_spec_doc(spec)
    if spec_errors:
        raise ValueError("Artifact spec error: " + "; ".join(spec_errors))


def validate_mbpp_config(config: Dict[str, Any]) -> None:
    dataset = config.get("dataset", {})
    for key in ["source", "version", "selection_seed", "sample_size"]:
        if key not in dataset:
            raise ValueError(f"dataset missing '{key}'")

    source = dataset["source"]
    if source not in {"evalplus", "fixture"}:
        raise ValueError("dataset.source must be 'evalplus' or 'fixture'")

    if source == "evalplus":
        model = config.get("model", {})
        for key in ["base_url", "top_p"]:
            if key not in model:
                raise ValueError(f"model missing '{key}' for evalplus generation")
        if model.get("provider") not in LOCAL_OPENAI_PROVIDERS and "api_key_env" not in model:
            raise ValueError("model missing 'api_key_env' for hosted evalplus generation")


def load_api_key(root: Path, model_cfg: Dict[str, Any]) -> Tuple[str, str]:
    if model_cfg.get("provider") in LOCAL_OPENAI_PROVIDERS:
        return "ollama", "local provider default"

    api_key_env = model_cfg["api_key_env"]
    api_key = os.environ.get(api_key_env, "").strip()
    if api_key:
        return api_key, api_key_env

    token_file = os.environ.get(f"{api_key_env}_FILE") or model_cfg.get("api_key_file")
    if token_file:
        token_path = Path(token_file)
        if not token_path.is_absolute():
            token_path = root / token_path
        if token_path.exists():
            file_key = token_path.read_text(encoding="utf-8").strip()
            if file_key:
                return file_key, str(token_path)

    raise RuntimeError(
        f"{api_key_env} is not set and no token was found in "
        f"{model_cfg.get('api_key_file', f'${api_key_env}_FILE')}. "
        f"Set {api_key_env}, or put the token in an ignored local file such as .hf_token."
    )


def output_dir_for(root: Path, config: Dict[str, Any]) -> Tuple[Path, Path]:
    output = config.get("output", {})
    output_root = root / output.get("output_root", "evaluation/runs")
    report_dir = root / output.get("report_dir", "evaluation/reports")
    return output_root / config.get("run_id", utc_now()), report_dir


def extract_code(raw_response: str, entry_point: str) -> Tuple[str, str]:
    raw = (raw_response or "").strip()
    if not raw:
        return "", "empty response"

    fenced = re.findall(r"```(?:python|py)?\s*(.*?)```", raw, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        entry_pattern = re.compile(rf"\bdef\s+{re.escape(entry_point)}\s*\(")
        for block in fenced:
            if entry_pattern.search(block):
                return block.strip(), "python fence containing entry point"
        return fenced[0].strip(), "first fenced code block"

    return raw, "raw response"


def remove_top_level_asserts(code: str) -> Tuple[str, int]:
    """Exclude model-generated module-level asserts from benchmark evaluation."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, 0
    assertions = [node for node in tree.body if isinstance(node, ast.Assert)]
    if not assertions:
        return code, 0
    lines = code.splitlines(keepends=True)
    for node in reversed(assertions):
        del lines[node.lineno - 1 : node.end_lineno]
    return "".join(lines).strip(), len(assertions)


def validate_python_candidate(code: str, entry_point: str) -> Tuple[bool, bool, str]:
    if not code.strip():
        return False, False, "no extracted code"

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, False, f"syntax error: {exc.msg} at line {exc.lineno}"

    entry_defined = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry_point
        for node in tree.body
    )
    if not entry_defined:
        return True, False, f"missing top-level function '{entry_point}'"
    return True, True, "syntax and entry point valid"


def validity_check(check_id: str, stage: str, passed: bool, evidence: str) -> Dict[str, Any]:
    return {
        "id": check_id,
        "stage": stage,
        "passed": bool(passed),
        "evidence": evidence,
    }


def validity_result(checks: List[Dict[str, Any]]) -> str:
    return "pass" if all(check["passed"] for check in checks) else "fail"


def raw_response_from_sample(sample: Dict[str, Any]) -> str:
    return sample.get("raw_response") or sample.get("completion") or sample.get("solution") or ""


def annotate_samples(samples: List[Dict[str, Any]], samples_path: Path) -> List[Dict[str, Any]]:
    annotated: List[Dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        row = dict(sample)
        row["_source_ref"] = f"{samples_path}:line {index}"
        annotated.append(row)
    return annotated


def _execution_probe_worker(code: str, entry_point: str, queue: multiprocessing.Queue) -> None:
    try:
        try:
            from evalplus.eval.utils import reliability_guard

            reliability_guard()
        except Exception:
            pass

        namespace: Dict[str, Any] = {}
        exec(code, namespace)
        candidate = namespace.get(entry_point)
        if not callable(candidate):
            queue.put((False, f"entry point '{entry_point}' did not resolve to a callable"))
            return
        queue.put((True, "execution probe passed"))
    except BaseException as exc:
        queue.put((False, f"{type(exc).__name__}: {exc}"))


def execution_probe(code: str, entry_point: str, timeout_seconds: float) -> Tuple[bool, str]:
    queue: multiprocessing.Queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_execution_probe_worker,
        args=(code, entry_point, queue),
    )
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        time.sleep(0.1)
    if process.is_alive():
        process.kill()
        time.sleep(0.1)
    if process.exitcode is None:
        return False, "execution probe timeout"
    if queue.empty():
        return False, "execution probe ended without a result"
    ok, reason = queue.get()
    return bool(ok), str(reason)


def build_prompt(problem: Dict[str, Any], entry_point: str) -> str:
    return "\n".join(
        [
            "You are solving an MBPP Python programming task.",
            "Return only Python code. Do not include Markdown fences or explanations.",
            f"Define the function `{entry_point}` and any helpers required.",
            "",
            "Task:",
            str(problem["prompt"]).strip(),
        ]
    )


def load_evalplus_problems(config: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], str]:
    try:
        from evalplus.data import get_mbpp_plus, get_mbpp_plus_hash
    except ImportError as exc:
        raise RuntimeError(
            "EvalPlus is required for the full MBPP run. Install with: "
            "python3 -m pip install -r requirements-evaluation.txt"
        ) from exc

    version = config["dataset"]["version"]
    problems = get_mbpp_plus(version=version)
    dataset_hash = get_mbpp_plus_hash(version=version)
    return problems, dataset_hash


def apply_evalplus_runtime_env(config: Dict[str, Any]) -> None:
    execution = config.get("execution", {})
    configured_limit = execution.get("evalplus_max_memory_bytes")
    if configured_limit is not None:
        os.environ["EVALPLUS_MAX_MEMORY_BYTES"] = str(configured_limit)
        return

    if platform.uname().system == "Darwin" and "EVALPLUS_MAX_MEMORY_BYTES" not in os.environ:
        os.environ["EVALPLUS_MAX_MEMORY_BYTES"] = "-1"
        print("Using EVALPLUS_MAX_MEMORY_BYTES=-1 on macOS to avoid resource.setrlimit failures")


def validate_evalplus_manifest(
    manifest: List[Dict[str, Any]],
    problems: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
    dataset_hash: str,
) -> None:
    dataset = config["dataset"]
    version = dataset["version"]
    seed = int(dataset["selection_seed"])
    expected_size = int(dataset["sample_size"])
    expected_ids = sorted(problems, key=lambda task_id: selection_key(version, task_id, seed))[:expected_size]
    actual_ids = [row["task_id"] for row in manifest]

    if actual_ids != expected_ids:
        raise ValueError("Manifest task IDs do not match the configured reproducible selection")

    for rank, row in enumerate(manifest, start=1):
        task_id = row["task_id"]
        problem = problems[task_id]
        expected_prompt_hash = sha256_text(problem["prompt"])
        expected_key = selection_key(version, task_id, seed)
        checks = {
            "selection_rank": rank,
            "entry_point": problem["entry_point"],
            "prompt_hash": expected_prompt_hash,
            "dataset_hash": dataset_hash,
            "selection_key": expected_key,
        }
        for key, expected in checks.items():
            if row.get(key) != expected:
                raise ValueError(f"Manifest mismatch for {task_id}: {key}={row.get(key)!r}, expected {expected!r}")


def load_fixture_problems() -> Tuple[Dict[str, Dict[str, Any]], str]:
    return FIXTURE_PROBLEMS, "fixture"


def generate_samples(
    root: Path,
    manifest: List[Dict[str, Any]],
    problems: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
    samples_path: Path,
    existing_rows: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The OpenAI Python client is required for Hugging Face router generation. "
            "Install with: python3 -m pip install -r requirements-evaluation.txt"
        ) from exc

    model_cfg = config["model"]
    api_key, api_key_source = load_api_key(root, model_cfg)
    print(f"Using API key from {api_key_source}")

    client = OpenAI(base_url=model_cfg["base_url"], api_key=api_key)
    rows: List[Dict[str, Any]] = list(existing_rows or [])
    existing_task_ids = {row["task_id"] for row in rows}
    for row in manifest:
        task_id = row["task_id"]
        if task_id in existing_task_ids:
            continue
        problem = problems[task_id]
        prompt = build_prompt(problem, row["entry_point"])
        response = client.chat.completions.create(
            model=model_cfg["name"],
            messages=[{"role": "user", "content": prompt}],
            temperature=float(model_cfg["temperature"]),
            top_p=float(model_cfg.get("top_p", 1)),
            max_tokens=int(model_cfg["max_tokens"]),
        )
        choice = response.choices[0]
        raw_response = choice.message.content or ""
        sample = {
            "task_id": task_id,
            "entry_point": row["entry_point"],
            "model": getattr(response, "model", model_cfg["name"]),
            "provider": model_cfg["provider"],
            "prompt_hash": row["prompt_hash"],
            "prompt_template_hash": sha256_text(build_prompt({"prompt": "{prompt}"}, "{entry_point}")),
            "raw_response": raw_response,
            "completion_hash": sha256_text(raw_response),
            "response_id": getattr(response, "id", None),
            "finish_reason": getattr(choice, "finish_reason", None),
            "generated_at": utc_now(),
        }
        rows.append(sample)
        write_jsonl(samples_path, rows)
        print(f"Generated {len(rows)}/{len(manifest)}: {task_id}")

    return rows


def load_or_generate_samples(
    root: Path,
    manifest: List[Dict[str, Any]],
    problems: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
    output_dir: Path,
    samples_override: Optional[Path] = None,
    no_generate: bool = False,
) -> Tuple[List[Dict[str, Any]], Path]:
    generation = config.get("generation", {})
    configured_samples = str(samples_override) if samples_override else generation.get("samples_path")
    samples_path = root / configured_samples if configured_samples and not Path(configured_samples).is_absolute() else Path(configured_samples) if configured_samples else output_dir / "samples.jsonl"
    reuse = bool(generation.get("reuse_existing_samples", True))
    required_ids = {row["task_id"] for row in manifest}

    if samples_path.exists() and reuse:
        existing = load_jsonl(samples_path)
        existing_ids = {row["task_id"] for row in existing}
        if required_ids.issubset(existing_ids):
            return annotate_samples(existing, samples_path), samples_path
        missing = sorted(required_ids - existing_ids)
        if no_generate:
            preview = ", ".join(missing[:10])
            suffix = "" if len(missing) <= 10 else f", ... ({len(missing)} total)"
            raise RuntimeError(f"--no-generate was set, but samples are missing: {preview}{suffix}")
        if config["dataset"]["source"] == "fixture":
            return annotate_samples(existing, samples_path), samples_path
        print(f"Resuming generation for {len(missing)} missing tasks")
    else:
        if no_generate:
            raise RuntimeError(f"--no-generate was set, but samples file does not exist: {samples_path}")
        existing = []

    if config["dataset"]["source"] == "fixture":
        raise RuntimeError(f"Fixture samples not found: {samples_path}")

    samples_path.parent.mkdir(parents=True, exist_ok=True)
    generated = generate_samples(root, manifest, problems, config, samples_path, existing)
    return annotate_samples(generated, samples_path), samples_path


def fixture_test_suite(code: str, problem: Dict[str, Any], suite_name: str) -> Tuple[str, List[bool], str]:
    namespace: Dict[str, Any] = {}
    try:
        exec(code, namespace)
        fn = namespace[problem["entry_point"]]
    except BaseException as exc:
        return FAIL, [], f"{type(exc).__name__}: {exc}"

    details: List[bool] = []
    for args, expected in problem[suite_name]:
        try:
            actual = fn(*args)
            details.append(actual == expected)
        except BaseException:
            details.append(False)

    return (PASS if all(details) else FAIL), details, "fixture tests completed"


def evalplus_expected_output(
    problems: Dict[str, Dict[str, Any]],
    dataset_hash: str,
    manifest: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    try:
        from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
        from evalplus.evaluate import get_groundtruth
    except ImportError as exc:
        raise RuntimeError(
            "EvalPlus is required for the full MBPP run. Install with: "
            "python3 -m pip install -r requirements-evaluation.txt"
        ) from exc

    selected_ids = [row["task_id"] for row in manifest]
    selected = {task_id: problems[task_id] for task_id in selected_ids}
    subset_hash = sha256_text(dataset_hash + ":" + ",".join(selected_ids))[:12]
    return get_groundtruth(selected, f"{dataset_hash}-subset-{subset_hash}", MBPP_OUTPUT_NOT_NONE_TASKS)


def evalplus_test_suites(
    code: str,
    problem: Dict[str, Any],
    expected_output: Dict[str, Any],
    config: Dict[str, Any],
) -> Tuple[str, List[bool], str, List[bool]]:
    try:
        from evalplus.evaluate import check_correctness
    except ImportError as exc:
        raise RuntimeError(
            "EvalPlus is required for the full MBPP run. Install with: "
            "python3 -m pip install -r requirements-evaluation.txt"
        ) from exc

    execution = config.get("execution", {})
    result = check_correctness(
        "mbpp",
        0,
        problem,
        code,
        expected_output,
        base_only=False,
        fast_check=not bool(execution.get("test_details", True)),
        identifier=problem["task_id"],
        min_time_limit=float(execution.get("min_time_limit", 1.0)),
        gt_time_limit_factor=float(execution.get("gt_time_limit_factor", 4.0)),
    )
    base_status, base_details = result["base"]
    plus_status, plus_details = result["plus"]
    return base_status, list(base_details), plus_status, list(plus_details)


def fail_count(status: str, details: List[bool], total_tests: int) -> int:
    if status == NOT_RUN:
        return 0
    if status == PASS:
        return 0
    if not details:
        return total_tests
    passed = sum(1 for item in details if item)
    return max(0, total_tests - passed)


def oracle_result(
    oracle_name: str,
    status: str,
    details: List[bool],
    total_tests: int,
    not_run_reason: str = "",
) -> Dict[str, Any]:
    normalized_details = [bool(item) for item in details]
    failed = fail_count(status, normalized_details, total_tests)
    if status == NOT_RUN:
        summary = not_run_reason or "oracle not run"
    elif status == PASS:
        summary = "all tests passed"
    elif status == TIMEOUT:
        summary = "oracle execution timed out"
    else:
        summary = f"{failed} of {total_tests} tests failed"

    return {
        "oracle": oracle_name,
        "status": status,
        "passed": status == PASS,
        "reached": status != NOT_RUN,
        "timeout": status == TIMEOUT,
        "runtime_error": False,
        "fail_count": failed,
        "total_count": total_tests,
        "details": normalized_details,
        "failure_summary": summary,
    }


def format_nested_cell(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    return value


def classify_outcome(
    base_status: str,
    plus_status: str,
    pre_oracle_failed: bool,
    execution_failed: bool,
    failure_reason: str,
) -> Tuple[str, str, str]:
    if pre_oracle_failed:
        return OUTCOME_REJECTED_PRE_ORACLE, "pre_oracle", failure_reason
    if execution_failed or base_status == TIMEOUT:
        return OUTCOME_REJECTED_EXECUTION, "execution", failure_reason
    if base_status != PASS:
        return OUTCOME_REJECTED_BASE, "base_oracle", "original MBPP tests failed"
    if plus_status == PASS:
        return OUTCOME_ACCEPTED, "extended_oracle", ""
    return OUTCOME_CONDITIONAL, "extended_oracle", "MBPP+ extended tests failed or timed out"


def execution_failure_cause(reason: str) -> str:
    lowered = reason.lower()
    if "timeout" in lowered:
        return "timeout"
    if "unsafe" in lowered or "security" in lowered or "reliability_guard" in lowered:
        return "unsafe_execution"
    if "did not resolve to a callable" in lowered:
        return "harness_setup_failure"
    return "runtime_error"


def result_gates(
    response_captured: bool,
    code_extracted: bool,
    syntax_valid: bool,
    entry_point_valid: bool,
    execution_ok: bool,
    base_status: str,
    plus_status: str,
    outcome: str,
) -> List[Dict[str, Any]]:
    return [
        {"id": "response_capture", "stage": "validity", "oracle_type": "schema", "blocking": True, "passed": response_captured},
        {"id": "code_extraction", "stage": "normalization", "oracle_type": "rule", "blocking": True, "passed": code_extracted},
        {"id": "syntax_validation", "stage": "validity", "oracle_type": "schema", "blocking": True, "passed": syntax_valid},
        {"id": "entry_point_validation", "stage": "validity", "oracle_type": "rule", "blocking": True, "passed": entry_point_valid},
        {"id": "execution_probe", "stage": "execution", "oracle_type": "test", "blocking": True, "passed": execution_ok},
        {"id": "base_oracle", "stage": "execution", "oracle_type": "test", "blocking": True, "passed": base_status == PASS},
        {"id": "extended_oracle", "stage": "execution", "oracle_type": "test", "blocking": False, "passed": plus_status == PASS},
        {"id": "decision_classification", "stage": "quality", "oracle_type": "rule", "blocking": False, "passed": outcome == OUTCOME_ACCEPTED},
    ]


def validate_mbpp_result(row: Dict[str, Any]) -> None:
    missing = [field for field in RESULT_FIELDS if field not in row]
    if missing:
        raise ValueError(f"result missing fields: {', '.join(missing)}")
    if row["pipeline_outcome"] not in OUTCOME_ORDER:
        raise ValueError(f"unsupported pipeline outcome: {row['pipeline_outcome']}")


def evaluate_samples(
    manifest: List[Dict[str, Any]],
    samples: List[Dict[str, Any]],
    problems: Dict[str, Dict[str, Any]],
    dataset_hash: str,
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    samples_by_task = {sample["task_id"]: sample for sample in samples}
    missing = [row["task_id"] for row in manifest if row["task_id"] not in samples_by_task]
    if missing:
        raise ValueError(f"samples missing selected tasks: {', '.join(missing)}")

    source = config["dataset"]["source"]
    expected_output: Optional[Dict[str, Dict[str, Any]]] = None
    if source == "evalplus":
        apply_evalplus_runtime_env(config)
        expected_output = evalplus_expected_output(problems, dataset_hash, manifest)

    timeout_seconds = float(config.get("execution", {}).get("execution_probe_timeout_seconds", 5))
    results: List[Dict[str, Any]] = []
    for manifest_row in manifest:
        task_id = manifest_row["task_id"]
        problem = problems[task_id]
        sample = samples_by_task[task_id]
        raw_response = raw_response_from_sample(sample)
        code, extraction_note = extract_code(raw_response, manifest_row["entry_point"])
        code, removed_assertions = remove_top_level_asserts(code)
        if removed_assertions:
            extraction_note += f"; removed {removed_assertions} top-level model assertion(s)"
        syntax_valid, entry_point_valid, validation_note = validate_python_candidate(code, manifest_row["entry_point"])

        response_captured = bool(raw_response.strip())
        code_extracted = bool(code.strip())
        pre_oracle_failed = not (response_captured and code_extracted and syntax_valid and entry_point_valid)
        execution_ok = False
        execution_note = "not run"
        base_status = NOT_RUN
        plus_status = NOT_RUN
        base_details: List[bool] = []
        plus_details: List[bool] = []

        if not pre_oracle_failed:
            execution_ok, execution_note = execution_probe(code, manifest_row["entry_point"], timeout_seconds)

        if execution_ok:
            if source == "fixture":
                base_status, base_details, _ = fixture_test_suite(code, problem, "base_tests")
                if base_status == PASS:
                    plus_status, plus_details, _ = fixture_test_suite(code, problem, "plus_tests")
                else:
                    plus_status = NOT_RUN
            else:
                assert expected_output is not None
                base_status, base_details, plus_status, plus_details = evalplus_test_suites(
                    code,
                    problem,
                    expected_output[task_id],
                    config,
                )

        validity_checks = [
            validity_check(
                "raw_output_exists",
                "validity",
                response_captured,
                "raw model output is present" if response_captured else "raw model output is empty",
            ),
            validity_check(
                "code_extracted",
                "normalization",
                code_extracted,
                extraction_note,
            ),
            validity_check(
                "ast_parse_ok",
                "validity",
                syntax_valid,
                validation_note if not syntax_valid else "extracted code parses with ast.parse",
            ),
            validity_check(
                "expected_function_present",
                "validity",
                entry_point_valid,
                f"found top-level function '{manifest_row['entry_point']}'" if entry_point_valid else validation_note,
            ),
            validity_check(
                "harness_setup_ok",
                "execution",
                execution_ok,
                execution_note if not execution_ok else "candidate resolved in execution probe",
            ),
        ]

        failure_reason = validation_note
        failed_pre_oracle = None
        if pre_oracle_failed:
            failed_pre_oracle = next((check for check in validity_checks[:4] if not check["passed"]), None)
            failure_reason = failed_pre_oracle["evidence"] if failed_pre_oracle else extraction_note
        elif not execution_ok:
            failure_reason = execution_note

        base_total = len(problem.get("base_tests", problem.get("base_input", [])))
        plus_total = len(problem.get("plus_tests", problem.get("plus_input", [])))
        completion_hash = sample.get("completion_hash") or sha256_text(raw_response)
        base_pass = base_status == PASS
        plus_pass = plus_status == PASS
        base_result = oracle_result(
            "original_mbpp",
            base_status,
            base_details,
            base_total,
            "base oracle not run because pre-oracle or execution checks failed",
        )
        plus_result = oracle_result(
            "mbpp_plus",
            plus_status,
            plus_details,
            plus_total,
            "extended oracle not run because base oracle did not pass",
        )

        outcome, decision_stage, normalized_failure = classify_outcome(
            base_status,
            plus_status,
            pre_oracle_failed,
            not execution_ok and not pre_oracle_failed,
            failure_reason,
        )
        failed_stage = ""
        failed_gate = ""
        failure_cause = ""
        if outcome == OUTCOME_ACCEPTED:
            normalized_failure = ""
        elif outcome == OUTCOME_CONDITIONAL:
            normalized_failure = plus_result["failure_summary"]
            failed_stage = "extended_oracle"
            failed_gate = "extended_oracle"
            failure_cause = "extended_oracle_failure"
            if plus_status == TIMEOUT:
                failure_cause = "extended_oracle_timeout"
        elif outcome == OUTCOME_REJECTED_BASE:
            normalized_failure = base_result["failure_summary"]
            failed_stage = "base_oracle"
            failed_gate = "base_oracle"
            failure_cause = "base_oracle_failure"
        elif outcome == OUTCOME_REJECTED_EXECUTION and base_status == TIMEOUT:
            normalized_failure = base_result["failure_summary"]
            failed_stage = "execution"
            failed_gate = "base_oracle"
            failure_cause = "timeout"
        elif outcome == OUTCOME_REJECTED_EXECUTION:
            failed_stage = "execution"
            failed_gate = "harness_setup_ok"
            failure_cause = execution_failure_cause(normalized_failure)
        elif outcome == OUTCOME_REJECTED_PRE_ORACLE:
            failed_stage = failed_pre_oracle["stage"] if failed_pre_oracle else "pre_oracle"
            failed_gate = failed_pre_oracle["id"] if failed_pre_oracle else "pre_oracle"
            if failed_gate == "raw_output_exists":
                failure_cause = "raw_output_missing"
            elif failed_gate == "code_extracted":
                failure_cause = "code_extraction_failure"
            elif failed_gate == "ast_parse_ok":
                failure_cause = "syntax_error"
            elif failed_gate == "expected_function_present":
                failure_cause = "missing_expected_function"
            elif failed_gate == "harness_setup_ok":
                failure_cause = "harness_setup_failure"
            else:
                failure_cause = "pre_oracle_failure"

        row = {
            "run_id": config["run_id"],
            "task_id": task_id,
            "prompt": problem["prompt"],
            "entry_point": manifest_row["entry_point"],
            "raw_output_path": sample.get("_source_ref", ""),
            "raw_output_text": raw_response,
            "extracted_code": code,
            "validity_check_result": validity_result(validity_checks),
            "validity_checks": validity_checks,
            "mbpp_base_result": base_result,
            "mbpp_plus_result": plus_result,
            "decision_category": outcome,
            "pipeline_outcome": outcome,
            "decision_stage": decision_stage,
            "failed_stage": failed_stage,
            "failed_gate": failed_gate,
            "base_status": base_status,
            "plus_status": plus_status,
            "base_pass": base_pass,
            "plus_pass": plus_pass,
            "failure_stage": failed_stage,
            "failure_cause": failure_cause,
            "failure_reason": normalized_failure,
            "base_fail_count": base_result["fail_count"],
            "base_total_count": base_result["total_count"],
            "plus_fail_count": plus_result["fail_count"],
            "plus_total_count": plus_result["total_count"],
            "model": sample.get("model", config["model"]["name"]),
            "provider": sample.get("provider", config["model"]["provider"]),
            "prompt_hash": manifest_row["prompt_hash"],
            "completion_hash": completion_hash,
            "dataset_hash": dataset_hash,
            "selection_rank": manifest_row.get("selection_rank"),
            "syntax_valid": syntax_valid,
            "entry_point_valid": entry_point_valid,
            "execution_probe_passed": execution_ok,
            "extraction_note": extraction_note,
            "gates": result_gates(
                response_captured,
                code_extracted,
                syntax_valid,
                entry_point_valid,
                execution_ok,
                base_status,
                plus_status,
                outcome,
            ),
            "scores": {
                "base_pass": 1.0 if base_pass else 0.0,
                "strict_mbpp_plus_pass": 1.0 if base_pass and plus_pass else 0.0,
            },
            "overall_score": 1.0 if outcome == OUTCOME_ACCEPTED else 0.0,
            "diagnostics": {
                "decision_stage": decision_stage,
                "failed_stage": failed_stage,
                "failed_gate": failed_gate,
                "failure_cause": failure_cause,
                "notes": normalized_failure,
                "warnings": [] if source == "fixture" else ["EvalPlus executes generated code; use an isolated environment for real runs."],
            },
        }
        validate_mbpp_result(row)
        results.append(row)
        print(f"{task_id}: {outcome}")

    return results


def write_results_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_nested_cell(row.get(field, "")) for field in writer.fieldnames})


def rate(count: int, total: int) -> float:
    return round(count / total, 4) if total else 0.0


def summarize(config: Dict[str, Any], dataset_hash: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    outcome_counts = {outcome: 0 for outcome in OUTCOME_ORDER}
    for row in results:
        outcome_counts[row["pipeline_outcome"]] += 1

    base_pass_count = sum(1 for row in results if row["base_status"] == PASS)
    base_reached_count = sum(1 for row in results if row["base_status"] != NOT_RUN)
    strict_pass_count = sum(1 for row in results if row["base_status"] == PASS and row["plus_status"] == PASS)
    failure_stage_counts = Counter(row.get("failed_stage", "") for row in results if row.get("failed_stage"))
    failed_gate_counts = Counter(row.get("failed_gate", "") for row in results if row.get("failed_gate"))
    failure_cause_counts = Counter(row.get("failure_cause", "") for row in results if row.get("failure_cause"))
    return {
        "run_id": config["run_id"],
        "dataset": config["dataset"]["name"],
        "dataset_version": config["dataset"]["version"],
        "dataset_hash": dataset_hash,
        "model": config["model"]["name"],
        "provider": config["model"]["provider"],
        "artifacts": total,
        "base_oracle_reached_count": base_reached_count,
        "base_pass_count": base_pass_count,
        "base_pass_rate": rate(base_pass_count, total),
        "base_pass_rate_all_tasks": rate(base_pass_count, total),
        "base_pass_rate_reached_tasks": rate(base_pass_count, base_reached_count),
        "strict_mbpp_plus_pass_count": strict_pass_count,
        "strict_mbpp_plus_pass_rate": rate(strict_pass_count, total),
        "outcome_counts": outcome_counts,
        "outcome_rates": {outcome: rate(count, total) for outcome, count in outcome_counts.items()},
        "failure_stage_counts": dict(sorted(failure_stage_counts.items())),
        "failed_gate_counts": dict(sorted(failed_gate_counts.items())),
        "failure_cause_counts": dict(sorted(failure_cause_counts.items())),
    }


def pct(value: float) -> str:
    return f"{value * 100:.1f}\\%"


def markdown_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def bool_cell(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "pass"}
    return bool(value)


def result_outcome(row: Dict[str, Any]) -> str:
    return str(row.get("pipeline_outcome") or row.get("decision_category") or "")


def outcome_severity(outcome: str) -> Optional[int]:
    if outcome in OUTCOME_SEVERITY:
        return OUTCOME_SEVERITY[outcome]
    if outcome.startswith("Rejected"):
        return 0
    return None


def base_passed(row: Dict[str, Any]) -> bool:
    return row.get("base_status") == PASS or bool_cell(row.get("base_pass"))


def strict_mbpp_plus_passed(row: Dict[str, Any]) -> bool:
    return (row.get("base_status") == PASS and row.get("plus_status") == PASS) or (
        bool_cell(row.get("base_pass")) and bool_cell(row.get("plus_pass"))
    )


def monitored_values(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "base_pass": base_passed(row),
        "strict_mbpp_plus_pass": strict_mbpp_plus_passed(row),
        "decision_category": result_outcome(row),
        "failed_stage": row.get("failed_stage", ""),
        "failed_gate": row.get("failed_gate", ""),
        "failure_cause": row.get("failure_cause", ""),
    }


def result_rows_by_task(rows: List[Dict[str, Any]], source_label: str) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        task_id = str(row.get("task_id", ""))
        if not task_id:
            raise ValueError(f"{source_label} result row is missing task_id")
        if task_id in indexed:
            raise ValueError(f"{source_label} contains duplicate task_id: {task_id}")
        indexed[task_id] = row
    return indexed


def aggregate_result_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    outcome_counts = {outcome: 0 for outcome in OUTCOME_ORDER}
    for row in rows:
        outcome = result_outcome(row)
        if outcome not in outcome_counts:
            outcome_counts[outcome] = 0
        outcome_counts[outcome] += 1

    base_count = sum(1 for row in rows if base_passed(row))
    strict_count = sum(1 for row in rows if strict_mbpp_plus_passed(row))
    failed_gates = Counter(str(row.get("failed_gate", "")) for row in rows if row.get("failed_gate"))
    failure_causes = Counter(str(row.get("failure_cause", "")) for row in rows if row.get("failure_cause"))
    return {
        "artifacts": total,
        "base_pass_count": base_count,
        "base_pass_rate": rate(base_count, total),
        "strict_mbpp_plus_pass_count": strict_count,
        "strict_mbpp_plus_pass_rate": rate(strict_count, total),
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "failed_gate_counts": dict(sorted(failed_gates.items())),
        "failure_cause_counts": dict(sorted(failure_causes.items())),
    }


def unique_row_values(rows: List[Dict[str, Any]], key: str) -> List[str]:
    return sorted({str(row.get(key)) for row in rows if row.get(key) not in {None, ""}})


def compare_task_row(
    task_id: str,
    baseline: Optional[Dict[str, Any]],
    current: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if baseline is None:
        current_values = monitored_values(current or {})
        return {
            "task_id": task_id,
            "change_type": "new_task",
            "baseline": None,
            "current": current_values,
            "prompt_hash_changed": False,
            "completion_hash_changed": False,
        }
    if current is None:
        baseline_values = monitored_values(baseline)
        return {
            "task_id": task_id,
            "change_type": "missing_task",
            "baseline": baseline_values,
            "current": None,
            "prompt_hash_changed": False,
            "completion_hash_changed": False,
        }

    baseline_values = monitored_values(baseline)
    current_values = monitored_values(current)
    baseline_severity = outcome_severity(str(baseline_values["decision_category"]))
    current_severity = outcome_severity(str(current_values["decision_category"]))

    if baseline_values == current_values:
        change_type = "unchanged"
    elif baseline_severity is not None and current_severity is not None and current_severity > baseline_severity:
        change_type = "improved"
    elif baseline_severity is not None and current_severity is not None and current_severity < baseline_severity:
        change_type = "regressed"
    else:
        change_type = "changed_without_ordering"

    return {
        "task_id": task_id,
        "change_type": change_type,
        "baseline": baseline_values,
        "current": current_values,
        "prompt_hash_changed": baseline.get("prompt_hash") != current.get("prompt_hash"),
        "completion_hash_changed": baseline.get("completion_hash") != current.get("completion_hash"),
        "baseline_completion_hash": baseline.get("completion_hash", ""),
        "current_completion_hash": current.get("completion_hash", ""),
    }


def aggregate_delta(baseline: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
    numeric_metrics = [
        "base_pass_count",
        "base_pass_rate",
        "strict_mbpp_plus_pass_count",
        "strict_mbpp_plus_pass_rate",
    ]
    deltas: Dict[str, Any] = {
        metric: round(current.get(metric, 0) - baseline.get(metric, 0), 4)
        for metric in numeric_metrics
    }
    outcome_delta: Dict[str, int] = {}
    outcomes = set(baseline.get("outcome_counts", {})) | set(current.get("outcome_counts", {}))
    for outcome in sorted(outcomes):
        outcome_delta[outcome] = current.get("outcome_counts", {}).get(outcome, 0) - baseline.get("outcome_counts", {}).get(outcome, 0)
    deltas["outcome_counts"] = outcome_delta
    return deltas


def add_review_signal(signals: List[Dict[str, Any]], signal_id: str, message: str, details: Dict[str, Any]) -> None:
    signals.append(
        {
            "id": signal_id,
            "severity": "review",
            "message": message,
            "details": details,
        }
    )


def add_recommendation(
    recommendations: List[Dict[str, Any]],
    rec_id: str,
    priority: str,
    triggered_by: List[str],
    suggestion: str,
    affected_tasks: List[str],
    spec_target: str,
) -> None:
    if any(recommendation["id"] == rec_id for recommendation in recommendations):
        return
    recommendations.append(
        {
            "id": rec_id,
            "priority": priority,
            "triggered_by": triggered_by,
            "suggestion": suggestion,
            "affected_tasks": affected_tasks,
            "spec_target": spec_target,
        }
    )


def signal_ids(summary: Dict[str, Any]) -> set[str]:
    return {str(signal.get("id", "")) for signal in summary.get("evaluation_specification_review_signals", [])}


def suggested_regression_tasks(summary: Dict[str, Any]) -> List[str]:
    return [
        row["task_id"]
        for row in summary.get("task_comparisons", [])
        if row.get("change_type") == "regressed"
    ]


def suggest_evaluation_spec_updates(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    signals = signal_ids(summary)
    recommendations: List[Dict[str, Any]] = []
    regression_tasks = suggested_regression_tasks(summary)
    aggregate_delta_view = summary.get("aggregate", {}).get("delta", {})

    if "model_metadata_changed" in signals:
        add_recommendation(
            recommendations,
            "document_model_replacement_context",
            "medium",
            ["model_metadata_changed"],
            "Treat this comparison as a model-replacement evaluation and keep model/provider identity as explicit monitoring context before interpreting changes as drift.",
            [],
            "monitoring.context_fields",
        )

    if "accepted_task_regressed" in signals:
        affected_tasks = []
        for signal in summary.get("evaluation_specification_review_signals", []):
            if signal.get("id") == "accepted_task_regressed":
                affected_tasks = list(signal.get("details", {}).get("task_ids", []))
                break
        add_recommendation(
            recommendations,
            "add_accepted_regressions_as_sentinels",
            "high",
            ["accepted_task_regressed"],
            "Add previously accepted tasks that became conditional or rejected as sentinel regression cases in future runs.",
            affected_tasks,
            "monitoring.sentinel_tasks",
        )

    pass_rate_triggers = sorted(signals & {"base_pass_rate_drop", "strict_pass_rate_drop"})
    if pass_rate_triggers:
        add_recommendation(
            recommendations,
            "review_pass_rate_thresholds",
            "high",
            pass_rate_triggers,
            "Review aggregate pass-rate thresholds and add a regression gate for drops in base or strict pass rates.",
            [],
            "monitoring.review_triggers",
        )

    failure_triggers = sorted(signals & {"new_failed_gates", "new_failure_causes"})
    if failure_triggers:
        affected_values: List[str] = []
        for signal in summary.get("evaluation_specification_review_signals", []):
            details = signal.get("details", {})
            affected_values.extend(details.get("failed_gates", []))
            affected_values.extend(details.get("failure_causes", []))
        add_recommendation(
            recommendations,
            "review_new_failure_handling",
            "medium",
            failure_triggers,
            "Review whether newly observed failed gates or failure causes should be added to the evaluation specification, diagnostics, or escalation policy.",
            sorted(set(affected_values)),
            "failure_causes",
        )

    if "strict_failures_increased_base_stable" in signals:
        add_recommendation(
            recommendations,
            "strengthen_extended_oracle_regression_guard",
            "medium",
            ["strict_failures_increased_base_stable"],
            "Strengthen the extended-oracle regression guard because MBPP+ failures increased even though original MBPP base pass count stayed stable.",
            [],
            "monitoring.review_triggers",
        )

    aggregate_improved = (
        aggregate_delta_view.get("base_pass_count", 0) > 0
        or aggregate_delta_view.get("strict_mbpp_plus_pass_count", 0) > 0
        or aggregate_delta_view.get("base_pass_rate", 0) > 0
        or aggregate_delta_view.get("strict_mbpp_plus_pass_rate", 0) > 0
    )
    if regression_tasks and aggregate_improved:
        add_recommendation(
            recommendations,
            "keep_task_level_regression_guards",
            "high",
            ["regressions_with_aggregate_improvement"],
            "Preserve task-level regression guards alongside aggregate metrics, because individual tasks regressed even though aggregate performance improved.",
            regression_tasks,
            "monitoring.review_triggers",
        )

    return recommendations


def resolve_compare_results_path(root: Path, compare_to: str) -> Path:
    path = Path(compare_to)
    if not path.is_absolute():
        path = root / path
    if path.is_dir():
        path = path / "results.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"comparison results not found: {path}")
    if path.name != "results.jsonl":
        raise ValueError("--compare-to must point to a run directory or a results.jsonl file")
    return path


def compare_evaluation_runs(
    baseline_rows: List[Dict[str, Any]],
    current_rows: List[Dict[str, Any]],
    baseline_path: Path,
    current_path: Path,
) -> Dict[str, Any]:
    baseline_by_task = result_rows_by_task(baseline_rows, "baseline")
    current_by_task = result_rows_by_task(current_rows, "current")
    baseline_ids = set(baseline_by_task)
    current_ids = set(current_by_task)
    matched_ids = sorted(baseline_ids & current_ids)
    all_ids = sorted(baseline_ids | current_ids)

    comparisons = [
        compare_task_row(task_id, baseline_by_task.get(task_id), current_by_task.get(task_id))
        for task_id in all_ids
    ]
    change_counts = {change_type: 0 for change_type in [
        "unchanged",
        "improved",
        "regressed",
        "changed_without_ordering",
        "new_task",
        "missing_task",
    ]}
    for comparison in comparisons:
        change_counts[comparison["change_type"]] += 1

    baseline_matched = [baseline_by_task[task_id] for task_id in matched_ids]
    current_matched = [current_by_task[task_id] for task_id in matched_ids]
    baseline_aggregate = aggregate_result_rows(baseline_matched)
    current_aggregate = aggregate_result_rows(current_matched)

    prompt_hash_changes = [
        comparison["task_id"]
        for comparison in comparisons
        if comparison["change_type"] not in {"new_task", "missing_task"} and comparison["prompt_hash_changed"]
    ]
    completion_hash_changes = [
        comparison["task_id"]
        for comparison in comparisons
        if comparison["change_type"] not in {"new_task", "missing_task"} and comparison["completion_hash_changed"]
    ]

    metadata = {
        "baseline": {
            "run_ids": unique_row_values(baseline_rows, "run_id"),
            "dataset_hashes": unique_row_values(baseline_rows, "dataset_hash"),
            "models": unique_row_values(baseline_rows, "model"),
            "providers": unique_row_values(baseline_rows, "provider"),
        },
        "current": {
            "run_ids": unique_row_values(current_rows, "run_id"),
            "dataset_hashes": unique_row_values(current_rows, "dataset_hash"),
            "models": unique_row_values(current_rows, "model"),
            "providers": unique_row_values(current_rows, "provider"),
        },
    }
    compatibility_checks = [
        {
            "id": "dataset_hash",
            "status": "same" if metadata["baseline"]["dataset_hashes"] == metadata["current"]["dataset_hashes"] else "changed",
            "baseline": metadata["baseline"]["dataset_hashes"],
            "current": metadata["current"]["dataset_hashes"],
        },
        {
            "id": "model_metadata",
            "status": "same"
            if metadata["baseline"]["models"] == metadata["current"]["models"]
            and metadata["baseline"]["providers"] == metadata["current"]["providers"]
            else "changed",
            "baseline": {
                "models": metadata["baseline"]["models"],
                "providers": metadata["baseline"]["providers"],
            },
            "current": {
                "models": metadata["current"]["models"],
                "providers": metadata["current"]["providers"],
            },
        },
        {
            "id": "prompt_hash",
            "status": "same" if not prompt_hash_changes else "changed",
            "changed_task_count": len(prompt_hash_changes),
            "changed_tasks": prompt_hash_changes,
        },
        {
            "id": "completion_hash",
            "status": "same" if not completion_hash_changes else "changed",
            "changed_task_count": len(completion_hash_changes),
            "changed_tasks": completion_hash_changes,
        },
    ]

    review_signals: List[Dict[str, Any]] = []
    for check in compatibility_checks:
        if check["id"] in {"dataset_hash", "model_metadata", "prompt_hash"} and check["status"] == "changed":
            add_review_signal(
                review_signals,
                f"{check['id']}_changed",
                f"{check['id'].replace('_', ' ')} changed between baseline and current run.",
                check,
            )

    if current_aggregate["base_pass_rate"] < baseline_aggregate["base_pass_rate"]:
        add_review_signal(
            review_signals,
            "base_pass_rate_drop",
            "Original MBPP base pass rate dropped on matched tasks.",
            {
                "baseline_rate": baseline_aggregate["base_pass_rate"],
                "current_rate": current_aggregate["base_pass_rate"],
                "delta": round(current_aggregate["base_pass_rate"] - baseline_aggregate["base_pass_rate"], 4),
            },
        )
    if current_aggregate["strict_mbpp_plus_pass_rate"] < baseline_aggregate["strict_mbpp_plus_pass_rate"]:
        add_review_signal(
            review_signals,
            "strict_pass_rate_drop",
            "MBPP+ strict pass rate dropped on matched tasks.",
            {
                "baseline_rate": baseline_aggregate["strict_mbpp_plus_pass_rate"],
                "current_rate": current_aggregate["strict_mbpp_plus_pass_rate"],
                "delta": round(
                    current_aggregate["strict_mbpp_plus_pass_rate"]
                    - baseline_aggregate["strict_mbpp_plus_pass_rate"],
                    4,
                ),
            },
        )

    accepted_regressions = [
        comparison["task_id"]
        for comparison in comparisons
        if (comparison.get("baseline") or {}).get("decision_category") == OUTCOME_ACCEPTED
        and comparison.get("current")
        and (comparison.get("current") or {}).get("decision_category") != OUTCOME_ACCEPTED
    ]
    if accepted_regressions:
        add_review_signal(
            review_signals,
            "accepted_task_regressed",
            "Previously accepted tasks became conditional or rejected.",
            {"task_ids": accepted_regressions, "count": len(accepted_regressions)},
        )

    baseline_failed_gates = {row.get("failed_gate") for row in baseline_matched if row.get("failed_gate")}
    current_failed_gates = {row.get("failed_gate") for row in current_matched if row.get("failed_gate")}
    new_failed_gates = sorted(current_failed_gates - baseline_failed_gates)
    if new_failed_gates:
        add_review_signal(
            review_signals,
            "new_failed_gates",
            "New failed gates appeared in the current run.",
            {"failed_gates": new_failed_gates},
        )

    baseline_failure_causes = {row.get("failure_cause") for row in baseline_matched if row.get("failure_cause")}
    current_failure_causes = {row.get("failure_cause") for row in current_matched if row.get("failure_cause")}
    new_failure_causes = sorted(current_failure_causes - baseline_failure_causes)
    if new_failure_causes:
        add_review_signal(
            review_signals,
            "new_failure_causes",
            "New failure causes appeared in the current run.",
            {"failure_causes": new_failure_causes},
        )

    baseline_strict_failures = len(matched_ids) - baseline_aggregate["strict_mbpp_plus_pass_count"]
    current_strict_failures = len(matched_ids) - current_aggregate["strict_mbpp_plus_pass_count"]
    if (
        current_strict_failures > baseline_strict_failures
        and current_aggregate["base_pass_count"] == baseline_aggregate["base_pass_count"]
    ):
        add_review_signal(
            review_signals,
            "strict_failures_increased_base_stable",
            "MBPP+ strict failures increased while original MBPP base pass count stayed stable.",
            {
                "baseline_strict_failures": baseline_strict_failures,
                "current_strict_failures": current_strict_failures,
                "base_pass_count": current_aggregate["base_pass_count"],
            },
        )

    summary = {
        "stage": "monitoring_and_re_evaluation",
        "comparison_key": "task_id",
        "monitored_fields": MONITORED_FIELDS,
        "generated_at": utc_now(),
        "baseline_results_path": str(baseline_path),
        "current_results_path": str(current_path),
        "task_counts": {
            "baseline": len(baseline_rows),
            "current": len(current_rows),
            "matched": len(matched_ids),
            "new": len(current_ids - baseline_ids),
            "missing": len(baseline_ids - current_ids),
        },
        "change_counts": change_counts,
        "aggregate": {
            "baseline_matched": baseline_aggregate,
            "current_matched": current_aggregate,
            "delta": aggregate_delta(baseline_aggregate, current_aggregate),
        },
        "metadata": metadata,
        "compatibility_checks": compatibility_checks,
        "evaluation_specification_review_signals": review_signals,
        "task_comparisons": comparisons,
    }
    summary["evaluation_specification_recommendations"] = suggest_evaluation_spec_updates(summary)
    return summary


def append_task_change_table(lines: List[str], title: str, rows: List[Dict[str, Any]], limit: int = 20) -> None:
    lines.extend(["", f"## {title}", ""])
    if not rows:
        lines.append("No tasks in this category.")
        return
    lines.extend(["| Task | Baseline | Current | Notes |", "| --- | --- | --- | --- |"])
    for row in rows[:limit]:
        baseline = row.get("baseline") or {}
        current = row.get("current") or {}
        notes = []
        if row.get("prompt_hash_changed"):
            notes.append("prompt hash changed")
        if row.get("completion_hash_changed"):
            notes.append("completion hash changed")
        if baseline.get("failure_cause") != current.get("failure_cause"):
            notes.append(
                f"cause: {baseline.get('failure_cause') or 'none'} -> {current.get('failure_cause') or 'none'}"
            )
        lines.append(
            "| {task} | {baseline} | {current} | {notes} |".format(
                task=row["task_id"],
                baseline=baseline.get("decision_category", "missing"),
                current=current.get("decision_category", "missing"),
                notes=", ".join(notes) if notes else "none",
            )
        )
    if len(rows) > limit:
        lines.append(f"| ... | ... | ... | {len(rows) - limit} additional tasks omitted |")


def append_recommendations(lines: List[str], recommendations: List[Dict[str, Any]]) -> None:
    lines.extend(["", "## Suggested Evaluation Specification Updates", ""])
    if not recommendations:
        lines.append("No evaluation-specification updates are suggested for this comparison.")
        return
    lines.extend(
        [
            "| Priority | Suggested update | Spec target | Triggered by | Affected tasks |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for recommendation in recommendations:
        affected_tasks = recommendation.get("affected_tasks", [])
        affected_text = ", ".join(affected_tasks) if affected_tasks else "none"
        triggered_by = ", ".join(recommendation.get("triggered_by", []))
        lines.append(
            "| {priority} | {suggestion} | `{target}` | {triggered_by} | {affected} |".format(
                priority=recommendation.get("priority", ""),
                suggestion=recommendation.get("suggestion", ""),
                target=recommendation.get("spec_target", ""),
                triggered_by=triggered_by,
                affected=affected_text,
            )
        )


def write_re_evaluation_report(path: Path, summary: Dict[str, Any]) -> None:
    aggregate = summary["aggregate"]
    baseline = aggregate["baseline_matched"]
    current = aggregate["current_matched"]
    delta = aggregate["delta"]
    signals = summary["evaluation_specification_review_signals"]
    recommendations = summary.get("evaluation_specification_recommendations", [])
    comparisons = summary["task_comparisons"]
    current_run_ids = summary["metadata"]["current"]["run_ids"]
    current_run_label = current_run_ids[0] if current_run_ids else "current"
    lines = [
        f"# Stage 6 Re-evaluation Report: {current_run_label}",
        "",
        f"- Baseline results: {summary['baseline_results_path']}",
        f"- Current results: {summary['current_results_path']}",
        f"- Matched tasks: {summary['task_counts']['matched']}",
        f"- New tasks: {summary['task_counts']['new']}",
        f"- Missing tasks: {summary['task_counts']['missing']}",
        "",
        "## Aggregate Deltas",
        "",
        "| Metric | Baseline | Current | Delta |",
        "| --- | ---: | ---: | ---: |",
        f"| Original MBPP base pass | {baseline['base_pass_count']} ({markdown_pct(baseline['base_pass_rate'])}) | {current['base_pass_count']} ({markdown_pct(current['base_pass_rate'])}) | {delta['base_pass_count']} ({markdown_pct(delta['base_pass_rate'])}) |",
        f"| MBPP+ strict pass | {baseline['strict_mbpp_plus_pass_count']} ({markdown_pct(baseline['strict_mbpp_plus_pass_rate'])}) | {current['strict_mbpp_plus_pass_count']} ({markdown_pct(current['strict_mbpp_plus_pass_rate'])}) | {delta['strict_mbpp_plus_pass_count']} ({markdown_pct(delta['strict_mbpp_plus_pass_rate'])}) |",
        "",
        "## Change Counts",
        "",
        "| Change type | Count |",
        "| --- | ---: |",
    ]
    for change_type, count in summary["change_counts"].items():
        lines.append(f"| {change_type} | {count} |")

    lines.extend(["", "## Evaluation Specification Review Signals", ""])
    if not signals:
        lines.append("No evaluation-specification review signals were triggered.")
    else:
        lines.extend(["| Signal | Message |", "| --- | --- |"])
        for signal in signals:
            lines.append(f"| {signal['id']} | {signal['message']} |")

    append_recommendations(lines, recommendations)

    for check in summary["compatibility_checks"]:
        if check["status"] != "same":
            lines.extend(["", f"### Compatibility: {check['id']}", ""])
            lines.append("```json")
            lines.append(json.dumps(check, indent=2, sort_keys=True))
            lines.append("```")

    append_task_change_table(
        lines,
        "Regressions",
        [row for row in comparisons if row["change_type"] == "regressed"],
    )
    append_task_change_table(
        lines,
        "Improvements",
        [row for row in comparisons if row["change_type"] == "improved"],
    )
    append_task_change_table(
        lines,
        "Changed Without Ordering",
        [row for row in comparisons if row["change_type"] == "changed_without_ordering"],
    )
    append_task_change_table(
        lines,
        "New Or Missing Tasks",
        [row for row in comparisons if row["change_type"] in {"new_task", "missing_task"}],
    )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_latex_table(path: Path, summary: Dict[str, Any]) -> None:
    rows = [
        ("Original MBPP base pass", summary["base_pass_count"], summary["base_pass_rate"]),
        ("MBPP+ strict pass", summary["strict_mbpp_plus_pass_count"], summary["strict_mbpp_plus_pass_rate"]),
    ]
    for outcome in OUTCOME_ORDER:
        rows.append((outcome, summary["outcome_counts"][outcome], summary["outcome_rates"][outcome]))

    lines = [
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "Metric & Count & Rate \\\\",
        "\\midrule",
    ]
    for label, count, row_rate in rows:
        lines.append(f"{label} & {count} & {pct(row_rate)} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def summary_rows(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = [
        {
            "view": "aggregate",
            "metric": "Original MBPP base pass",
            "count": summary["base_pass_count"],
            "denominator": summary["artifacts"],
            "rate": summary["base_pass_rate_all_tasks"],
        },
        {
            "view": "aggregate",
            "metric": "Original MBPP base pass among reached tasks",
            "count": summary["base_pass_count"],
            "denominator": summary["base_oracle_reached_count"],
            "rate": summary["base_pass_rate_reached_tasks"],
        },
        {
            "view": "aggregate",
            "metric": "MBPP+ strict pass",
            "count": summary["strict_mbpp_plus_pass_count"],
            "denominator": summary["artifacts"],
            "rate": summary["strict_mbpp_plus_pass_rate"],
        },
    ]
    for outcome in OUTCOME_ORDER:
        rows.append(
            {
                "view": "pipeline",
                "metric": outcome,
                "count": summary["outcome_counts"][outcome],
                "denominator": summary["artifacts"],
                "rate": summary["outcome_rates"][outcome],
            }
        )
    for stage, count in summary.get("failure_stage_counts", {}).items():
        rows.append(
            {
                "view": "failure_stage",
                "metric": stage,
                "count": count,
                "denominator": summary["artifacts"],
                "rate": rate(count, summary["artifacts"]),
            }
        )
    for gate, count in summary.get("failed_gate_counts", {}).items():
        rows.append(
            {
                "view": "failed_gate",
                "metric": gate,
                "count": count,
                "denominator": summary["artifacts"],
                "rate": rate(count, summary["artifacts"]),
            }
        )
    for cause, count in summary.get("failure_cause_counts", {}).items():
        rows.append(
            {
                "view": "failure_cause",
                "metric": cause,
                "count": count,
                "denominator": summary["artifacts"],
                "rate": rate(count, summary["artifacts"]),
            }
        )
    return rows


def write_summary_csv(path: Path, summary: Dict[str, Any]) -> None:
    fieldnames = ["view", "metric", "count", "denominator", "rate"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows(summary):
            writer.writerow(row)


def extracted_filename(task_id: str) -> str:
    return task_id.replace("/", "_").replace("\\", "_") + ".py"


def write_extracted_code(output_dir: Path, rows: List[Dict[str, Any]]) -> Path:
    extracted_dir = output_dir / "extracted"
    extracted_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        code = row.get("extracted_code", "")
        if not code:
            continue
        path = extracted_dir / extracted_filename(row["task_id"])
        path.write_text(code.rstrip() + "\n", encoding="utf-8")
    return extracted_dir


def write_traceability_report(path: Path, summary: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    stage_counts = summary.get("failure_stage_counts", {})
    gate_counts = summary.get("failed_gate_counts", {})
    cause_counts = summary.get("failure_cause_counts", {})
    lines = [
        f"# Traceability Report: {summary['run_id']}",
        "",
        "## Aggregate Benchmark View",
        "",
        f"- Original MBPP base pass over all tasks: {summary['base_pass_count']}/{summary['artifacts']} ({markdown_pct(summary['base_pass_rate_all_tasks'])})",
        f"- Original MBPP base pass among reached tasks: {summary['base_pass_count']}/{summary['base_oracle_reached_count']} ({markdown_pct(summary['base_pass_rate_reached_tasks'])})",
        f"- MBPP+ strict pass over all tasks: {summary['strict_mbpp_plus_pass_count']}/{summary['artifacts']} ({markdown_pct(summary['strict_mbpp_plus_pass_rate'])})",
        "",
        "## Pipeline Outcome View",
        "",
        "| Outcome | Count | Rate |",
        "| --- | ---: | ---: |",
    ]
    for outcome in OUTCOME_ORDER:
        lines.append(
            f"| {outcome} | {summary['outcome_counts'][outcome]} | {markdown_pct(summary['outcome_rates'][outcome])} |"
        )

    lines.extend(["", "## Failure Stage Counts", "", "| Failed stage | Count |", "| --- | ---: |"])
    if not stage_counts:
        lines.append("| none | 0 |")
    for stage, count in sorted(stage_counts.items()):
        lines.append(f"| {stage} | {count} |")

    lines.extend(["", "## Failed Gate Counts", "", "| Failed gate | Count |", "| --- | ---: |"])
    if not gate_counts:
        lines.append("| none | 0 |")
    for gate, count in sorted(gate_counts.items()):
        lines.append(f"| {gate} | {count} |")

    lines.extend(["", "## Failure Cause Counts", "", "| Failure cause | Count |", "| --- | ---: |"])
    if not cause_counts:
        lines.append("| none | 0 |")
    for cause, count in sorted(cause_counts.items()):
        lines.append(f"| {cause} | {count} |")

    lines.extend(["", "## Example Records", ""])
    for outcome in OUTCOME_ORDER:
        example = next((row for row in rows if row["decision_category"] == outcome), None)
        if not example:
            continue
        lines.extend(
            [
                f"### {outcome}",
                "",
                f"- Task: `{example['task_id']}`",
                f"- Entry point: `{example['entry_point']}`",
                f"- Decision stage: `{example['decision_stage']}`",
                f"- Failed stage: `{example['failed_stage'] or 'none'}`",
                f"- Failed gate: `{example['failed_gate'] or 'none'}`",
                f"- Failure cause: `{example['failure_cause'] or 'none'}`",
                f"- Validity: `{example['validity_check_result']}`",
                f"- Base oracle: `{example['mbpp_base_result']['status']}` ({example['mbpp_base_result']['failure_summary']})",
                f"- Extended oracle: `{example['mbpp_plus_result']['status']}` ({example['mbpp_plus_result']['failure_summary']})",
                f"- Failure reason: {example['failure_reason'] or 'none'}",
                "",
            ]
        )

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_report(
    path: Path,
    summary: Dict[str, Any],
    result_path: Path,
    table_path: Path,
    summary_csv_path: Path,
    traceability_path: Path,
) -> None:
    lines = [
        f"# MBPP/EvalPlus Summary: {summary['run_id']}",
        "",
        f"- Dataset: {summary['dataset']} {summary['dataset_version']} ({summary['dataset_hash']})",
        f"- Model: {summary['model']}",
        f"- Provider: {summary['provider']}",
        f"- Artifacts: {summary['artifacts']}",
        f"- Original MBPP base pass: {summary['base_pass_count']} ({markdown_pct(summary['base_pass_rate'])})",
        f"- MBPP+ strict pass: {summary['strict_mbpp_plus_pass_count']} ({markdown_pct(summary['strict_mbpp_plus_pass_rate'])})",
        f"- Results: {result_path}",
        f"- Summary CSV: {summary_csv_path}",
        f"- LaTeX table: {table_path}",
        f"- Traceability report: {traceability_path}",
        "",
        "## Pipeline Outcomes",
        "",
        "| Outcome | Count | Rate |",
        "| --- | ---: | ---: |",
    ]
    for outcome in OUTCOME_ORDER:
        lines.append(
            f"| {outcome} | {summary['outcome_counts'][outcome]} | {markdown_pct(summary['outcome_rates'][outcome])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_manifest(output_dir: Path, manifest: List[Dict[str, Any]]) -> Path:
    path = output_dir / "manifest.jsonl"
    write_jsonl(path, manifest)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MBPP/EvalPlus pipeline instantiation")
    parser.add_argument("--config", required=True, help="Path to MBPP/EvalPlus run config JSON")
    parser.add_argument(
        "--samples",
        help="Optional JSONL sample file with one saved model output per selected task",
    )
    parser.add_argument(
        "--no-generate",
        action="store_true",
        help="Use only saved samples and fail if any selected task is missing",
    )
    parser.add_argument(
        "--compare-to",
        help="Optional previous run directory or results.jsonl for Stage 6 re-evaluation comparison",
    )
    args = parser.parse_args()

    root = Path.cwd()
    config = load_json(root / args.config)
    validate_basic_contracts(root, config)
    validate_mbpp_config(config)

    manifest = load_jsonl(root / config["dataset"]["path"])
    output_dir, report_dir = output_dir_for(root, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    if config["dataset"]["source"] == "fixture":
        problems, dataset_hash = load_fixture_problems()
    else:
        problems, dataset_hash = load_evalplus_problems(config)
        expected_hash = config["dataset"].get("hash")
        if expected_hash and expected_hash != dataset_hash:
            raise ValueError(f"EvalPlus dataset hash changed: {dataset_hash}, expected {expected_hash}")
        validate_evalplus_manifest(manifest, problems, config, dataset_hash)

    samples_override = Path(args.samples) if args.samples else None
    samples, samples_path = load_or_generate_samples(
        root,
        manifest,
        problems,
        config,
        output_dir,
        samples_override=samples_override,
        no_generate=args.no_generate,
    )
    results = evaluate_samples(manifest, samples, problems, dataset_hash, config)
    summary = summarize(config, dataset_hash, results)

    copy_manifest(output_dir, manifest)
    results_path = output_dir / "results.jsonl"
    csv_path = output_dir / "results.csv"
    summary_path = output_dir / "summary.json"
    summary_csv_path = output_dir / "summary.csv"
    table_path = output_dir / "summary_table.tex"
    traceability_path = output_dir / "traceability_report.md"
    report_path = report_dir / f"{config['run_id']}-summary.md"

    write_jsonl(results_path, results)
    write_results_csv(csv_path, results)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_summary_csv(summary_csv_path, summary)
    write_latex_table(table_path, summary)
    extracted_dir = write_extracted_code(output_dir, results)
    write_traceability_report(traceability_path, summary, results)
    write_report(report_path, summary, results_path, table_path, summary_csv_path, traceability_path)

    re_evaluation_summary_path: Optional[Path] = None
    re_evaluation_report_path: Optional[Path] = None
    if args.compare_to:
        baseline_results_path = resolve_compare_results_path(root, args.compare_to)
        baseline_results = load_jsonl(baseline_results_path)
        re_evaluation_summary = compare_evaluation_runs(
            baseline_results,
            results,
            baseline_results_path,
            results_path,
        )
        re_evaluation_summary_path = output_dir / "re_evaluation_summary.json"
        re_evaluation_report_path = output_dir / "re_evaluation_report.md"
        re_evaluation_summary_path.write_text(
            json.dumps(re_evaluation_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_re_evaluation_report(re_evaluation_report_path, re_evaluation_summary)

    print(f"MBPP/EvalPlus run completed: {config['run_id']}")
    print(f"Samples: {samples_path}")
    print(f"Results: {results_path}")
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")
    print(f"Summary CSV: {summary_csv_path}")
    print(f"LaTeX table: {table_path}")
    print(f"Traceability report: {traceability_path}")
    print(f"Extracted code: {extracted_dir}")
    print(f"Report: {report_path}")
    if re_evaluation_summary_path and re_evaluation_report_path:
        print(f"Re-evaluation summary: {re_evaluation_summary_path}")
        print(f"Re-evaluation report: {re_evaluation_report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
