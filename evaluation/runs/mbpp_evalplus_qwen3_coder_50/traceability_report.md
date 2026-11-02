# Traceability Report: mbpp_evalplus_qwen3_coder_50

## Aggregate Benchmark View

- Original MBPP base pass over all tasks: 46/50 (92.0%)
- Original MBPP base pass among reached tasks: 46/50 (92.0%)
- MBPP+ strict pass over all tasks: 42/50 (84.0%)

## Pipeline Outcome View

| Outcome | Count | Rate |
| --- | ---: | ---: |
| Accepted | 42 | 84.0% |
| Conditional | 4 | 8.0% |
| Rejected before oracle | 0 | 0.0% |
| Rejected during execution | 0 | 0.0% |
| Rejected by base oracle | 4 | 8.0% |

## Failure Stage Counts

| Failed stage | Count |
| --- | ---: |
| base_oracle | 4 |
| extended_oracle | 4 |

## Failed Gate Counts

| Failed gate | Count |
| --- | ---: |
| base_oracle | 4 |
| extended_oracle | 4 |

## Failure Cause Counts

| Failure cause | Count |
| --- | ---: |
| base_oracle_failure | 4 |
| extended_oracle_failure | 4 |

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
- Extended oracle: `fail` (39 of 103 tests failed)
- Failure reason: 39 of 103 tests failed

### Rejected by base oracle

- Task: `Mbpp/311`
- Entry point: `set_left_most_unset_bit`
- Decision stage: `base_oracle`
- Failed stage: `base_oracle`
- Failed gate: `base_oracle`
- Failure cause: `base_oracle_failure`
- Validity: `pass`
- Base oracle: `fail` (3 of 3 tests failed)
- Extended oracle: `fail` (91 of 109 tests failed)
- Failure reason: 3 of 3 tests failed
