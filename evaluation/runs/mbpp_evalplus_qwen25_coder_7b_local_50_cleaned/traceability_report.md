# Traceability Report: mbpp_evalplus_qwen25_coder_7b_local_50_cleaned

## Aggregate Benchmark View

- Original MBPP base pass over all tasks: 43/50 (86.0%)
- Original MBPP base pass among reached tasks: 43/50 (86.0%)
- MBPP+ strict pass over all tasks: 36/50 (72.0%)

## Pipeline Outcome View

| Outcome | Count | Rate |
| --- | ---: | ---: |
| Accepted | 36 | 72.0% |
| Conditional | 7 | 14.0% |
| Rejected before oracle | 0 | 0.0% |
| Rejected during execution | 0 | 0.0% |
| Rejected by base oracle | 7 | 14.0% |

## Failure Stage Counts

| Failed stage | Count |
| --- | ---: |
| base_oracle | 7 |
| extended_oracle | 7 |

## Failed Gate Counts

| Failed gate | Count |
| --- | ---: |
| base_oracle | 7 |
| extended_oracle | 7 |

## Failure Cause Counts

| Failure cause | Count |
| --- | ---: |
| base_oracle_failure | 7 |
| extended_oracle_failure | 7 |

## Example Records

### Accepted

- Task: `Mbpp/418`
- Entry point: `Find_Max`
- Decision stage: `extended_oracle`
- Failed stage: `none`
- Failed gate: `none`
- Failure cause: `none`
- Validity: `pass`
- Base oracle: `pass` (all tests passed)
- Extended oracle: `pass` (all tests passed)
- Failure reason: none

### Conditional

- Task: `Mbpp/589`
- Entry point: `perfect_squares`
- Decision stage: `extended_oracle`
- Failed stage: `extended_oracle`
- Failed gate: `extended_oracle`
- Failure cause: `extended_oracle_failure`
- Validity: `pass`
- Base oracle: `pass` (all tests passed)
- Extended oracle: `fail` (66 of 103 tests failed)
- Failure reason: 66 of 103 tests failed

### Rejected by base oracle

- Task: `Mbpp/721`
- Entry point: `maxAverageOfPath`
- Decision stage: `base_oracle`
- Failed stage: `base_oracle`
- Failed gate: `base_oracle`
- Failure cause: `base_oracle_failure`
- Validity: `pass`
- Base oracle: `fail` (4 of 4 tests failed)
- Extended oracle: `fail` (94 of 101 tests failed)
- Failure reason: 4 of 4 tests failed
