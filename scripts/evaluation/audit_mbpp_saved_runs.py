#!/usr/bin/env python3
"""Audit saved MBPP cleaning and compare direct EvalPlus API results; no generation.

Run after both *_cleaned framework runs exist, inside sandbox.sb:
  python scripts/evaluation/audit_mbpp_saved_runs.py
The native path calls the official check_correctness API on the 50-task subset;
EvalPlus's full-dataset CLI requires a sample for every dataset task.
"""
import ast
import json
from pathlib import Path

from evalplus.evaluate import check_correctness
from evalplus.sanitize import sanitize
from run_mbpp_evalplus import (
    apply_evalplus_runtime_env, compare_evaluation_runs, extract_code,
    load_evalplus_problems, evalplus_expected_output, load_jsonl,
    raw_response_from_sample, remove_top_level_asserts, sha256_text,
    write_jsonl, write_re_evaluation_report,
)

ROOT = Path.cwd()
EVIDENCE = ROOT / 'evaluation/reports/mbpp_framework_value_evidence'
RUNS = ['mbpp_evalplus_qwen25_coder_7b_local_50', 'mbpp_evalplus_qwen3_coder_50']
CONFIGS = ['mbpp-evalplus-qwen25-coder-7b-local-50', 'mbpp-evalplus-qwen3-coder-50']


def main():
    # Audit files are new evidence; refuse to silently replace an earlier audit.
    destinations = [EVIDENCE / 'cleaning-audit.jsonl', EVIDENCE / 'native-parity.jsonl']
    for ident in RUNS:
        destinations.extend(EVIDENCE / f'{ident}-{suffix}' for suffix in
            ['native-samples.jsonl', 'native-results.jsonl', 'cleaning-comparison.json'])
    existing = [str(path) for path in destinations if path.exists()]
    if existing:
        raise FileExistsError('Refusing to overwrite audit evidence: ' + ', '.join(existing))
    audits = []
    parity = []
    for ident, config_name in zip(RUNS, CONFIGS):
        config = json.loads((ROOT / f'evaluation/configs/{config_name}-cleaned.json').read_text())
        apply_evalplus_runtime_env(config)
        problems, dataset_hash = load_evalplus_problems(config)
        manifest = load_jsonl(ROOT / config['dataset']['path'])
        expected = evalplus_expected_output(problems, dataset_hash, manifest)
        original = load_jsonl(ROOT / f'evaluation/runs/{ident}/results.jsonl')
        results_path = ROOT / f'evaluation/runs/{ident}_cleaned/results.jsonl'
        cleaned_rows = load_jsonl(results_path)
        by_id = {r['task_id']: r for r in cleaned_rows}
        manifest_by_id = {r['task_id']: r for r in manifest}
        samples = load_jsonl(ROOT / f'evaluation/runs/{ident}/samples.jsonl')
        assert len(samples) == len(manifest) == len(cleaned_rows) == 50
        assert {s['task_id'] for s in samples} == set(manifest_by_id) == set(by_id)
        native_samples, native_rows = [], []
        for line, sample in enumerate(samples, 1):
            task = sample['task_id']; entry = manifest_by_id[task]['entry_point']
            extracted, note = extract_code(raw_response_from_sample(sample), entry)
            cleaned, removed = remove_top_level_asserts(extracted)
            assert cleaned == by_id[task]['extracted_code'], task
            nodes = ast.parse(extracted).body
            # Check line deletion did not discard non-assert AST statements.
            original_nonassert = [ast.dump(n, include_attributes=False) for n in nodes if not isinstance(n, ast.Assert)]
            cleaned_nodes = [ast.dump(n, include_attributes=False) for n in ast.parse(cleaned).body]
            official = sanitize(extracted, entrypoint=entry)
            audits.append({
                'run_id': ident, 'task_id': task, 'sample_line': line,
                'extraction_note': note, 'removed_assertions': removed,
                'removed_assertion_source': [ast.get_source_segment(extracted,n) for n in nodes if isinstance(n, ast.Assert)],
                'nonassert_ast_preserved': original_nonassert == cleaned_nodes,
                'cleaned_sha256': sha256_text(cleaned),
                'official_sanitizer_sha256': sha256_text(official),
                'official_sanitizer_same_ast': ast.dump(ast.parse(official)) == ast.dump(ast.parse(cleaned)),
                'official_sanitized_code': official,
            })
            native_samples.append({'task_id':task, 'solution':cleaned})
            e = config['execution']
            result = check_correctness('mbpp', 0, problems[task], cleaned, expected[task],
                base_only=False, fast_check=not e['test_details'], identifier=task,
                min_time_limit=e['min_time_limit'], gt_time_limit_factor=e['gt_time_limit_factor'])
            row = {'task_id':task, 'code_sha256':sha256_text(cleaned),
                'base_status':result['base'][0], 'base_details':[bool(x) for x in result['base'][1]],
                'plus_status':result['plus'][0], 'plus_details':[bool(x) for x in result['plus'][1]]}
            native_rows.append(row)
            print(f"{ident} {task}: native audit complete", flush=True)
            framework = by_id[task]
            parity.append({'run_id':ident, 'task_id':task,
                'status_match': row['base_status']==framework['base_status'] and row['plus_status']==framework['plus_status'],
                'details_match': row['base_details']==framework['mbpp_base_result']['details'] and row['plus_details']==framework['mbpp_plus_result']['details']})
        write_jsonl(EVIDENCE / f'{ident}-native-samples.jsonl',native_samples)
        write_jsonl(EVIDENCE / f'{ident}-native-results.jsonl',native_rows)
        comparison = compare_evaluation_runs(original,cleaned_rows,ROOT/f'evaluation/runs/{ident}/results.jsonl',results_path)
        (EVIDENCE/f'{ident}-cleaning-comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    write_jsonl(EVIDENCE/'cleaning-audit.jsonl',audits)
    write_jsonl(EVIDENCE/'native-parity.jsonl',parity)
    print(json.dumps({'tasks':len(parity),'status_mismatches':[p for p in parity if not p['status_match']], 'details_mismatches':[p for p in parity if not p['details_match']]}))


if __name__ == '__main__':
    main()
