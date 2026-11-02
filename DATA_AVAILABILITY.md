# Data availability and limitations

This public repository is a curated replication subset; the original research repository is private.

## Publicly shared
- Source code, evaluation specifications, contracts and run configurations for the public MBPP/EvalPlus instantiation.
- A fixed 50-task selection manifest for the publicly available `MbppPlus` dataset.
- Saved model completions for Qwen3-Coder and locally run Qwen2.5-Coder-7B, aggregate summaries, and path-sanitized evaluation records.
- Redacted definitions for all **41 descriptive codes** in the human-reviewed summary. The original inclusion/exclusion rules are retained for the first 30 codes; the 11 codes added during human review have no formal rules in the source summary. Counts, speaker/context coverage and segment IDs are removed.

## Not shared
- Original workshop transcripts, organizer notes, speaker information, detailed statement-level evidence, and completed human coding review workbooks. Pseudonymization alone cannot guarantee that individuals or companies are unidentifiable.
- Proprietary industrial requirements, company-specific engineering context, underlying 3GPP case data and non-public outputs.
- Private repository Git history, local paths, credentials and investigator-specific environments.

## Reproducibility scope
The public MBPP/EvalPlus case supports inspection and rerunning evaluation on already saved public-benchmark completions. It does not guarantee identical model generation or identical results across every execution environment.

The qualitative synthesis and confidential industrial case can be understood from the publication and the non-sensitive material shared here, but **cannot be fully independently reproduced** from this repository. Access to withheld data would require separate ethical, privacy, organizational and confidentiality approvals. No access is promised.

Please verify journal requirements concerning anonymous review, external repository links, licensing and data availability statements before submission.
