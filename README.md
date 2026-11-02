# Replication Materials: Context-Specific Evaluation of AI Systems

This repository contains a **curated, publication-oriented subset** of reproducibility materials for the manuscript *Toward Context-Specific Evaluation of AI Systems in Industrial Workflows: Workshop Insights and a Reference Framework*.

The original research repository remains private. This repository was created **without copying its Git history** and excludes original workshop transcripts, identifying participant data, internal industrial requirements, credentials and local machine paths.

## Contents

- `scripts/evaluation/`: evaluation pipeline, contract validation and regression tests.
- `evaluation/specs/`, `evaluation/contracts/`, `evaluation/configs/`: evaluation specifications and configurations.
- `evaluation/datasets/`: reproducible selection manifest for 50 public MBPP+ tasks and a self-contained smoke fixture.
- `evaluation/runs/`: saved public-benchmark model outputs, sanitized evaluation records and aggregate summaries.
- `workshop/codebook/`: publication-safe definitions for all **41 human-reviewed descriptive codes**, without statement IDs or participant metadata (original inclusion/exclusion rules are available for the first 30 only).
- `DATA_AVAILABILITY.md`: what can and cannot be reproduced from the shared material.

The MBPP/EvalPlus example demonstrates how existing executable tests can be organized with validity checks, decision outcomes and records. It is **not** evidence that the framework improves test accuracy or operational efficiency.

## Quick start (from the repository root)

Install the Python requirements:

```bash
python3 -m pip install -r requirements-evaluation.txt
```

Run the built-in, network-free fixture:

```bash
python3 scripts/evaluation/run_mbpp_evalplus.py --config evaluation/configs/mbpp-evalplus-smoke.json
```

Re-evaluate **saved** Qwen3-Coder outputs for the public 50-task MBPP+ manifest, without model API calls:

```bash
python3 scripts/evaluation/run_mbpp_evalplus.py \
  --config evaluation/configs/mbpp-evalplus-qwen3-coder-50.json \
  --samples evaluation/runs/mbpp_evalplus_qwen3_coder_50/samples.jsonl \
  --no-generate
```

For local Qwen2.5-Coder, replace the config and samples paths with `mbpp-evalplus-qwen25-coder-7b-local-50.json` and `evaluation/runs/mbpp_evalplus_qwen25_coder_7b_local_50/samples.jsonl`.

Run the regression tests:

```bash
python3 -m unittest discover -s scripts/evaluation -p 'test_*.py'
```

`EvalPlus` downloads public benchmark data as needed. Re-evaluation can depend on dependency versions, runtime environment and execution limits. Check the manifest's dataset version/hash and configuration when interpreting discrepancies.

**Security:** The evaluation process can execute model-generated Python. Run it in an isolated, resource-limited environment, not on a machine with sensitive credentials or trusted data. Do not run untrusted completions unsandboxed.

## Data boundaries

Workshop transcripts and statement-level records are withheld because they may disclose participants or organization-specific context. The 3GPP industrial case uses non-public underlying material and is not distributed here. See [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md).

This is a selected set of review materials, **not** a complete reproduction of the qualitative analysis or the industrial case. A repository URL alone does not imply that withheld data can be independently checked. A publication license has not yet been selected; consult the maintainers before assuming reuse rights.
