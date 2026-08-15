# Task 64 C2 Preregistration V4.27.1

## Purpose

Task 64 C2 restores strict protocol-before-seed ordering after Task 63 C2. It reserves five fresh final seeds after the complete final-evaluation manifest is frozen.

The Task 64 C1 seeds remain permanently preserved as `SUPERSEDED_UNUSED_FOR_FINAL_EVALUATION` and are prohibited from Task 65. Development seeds 7, 42, 99, 123, and 2026 are also ineligible.

## Frozen deterministic selection

Candidates are generated in ascending candidate-index order from SHA256 over:

`LFighter|task64_c2_v4271|parent_tag=task63-c2-final-manifest-frozen-v4262|candidate={candidate_index}`

A candidate survives only when it is unique, is not in either prohibited seed set, and has zero exact numeric-token occurrences in the eligible repository corpus. Manual substitution, best-seed selection, and outcome-based selection are forbidden.

## Data boundary

Task 64 C2 performs no training, no final model evaluation, no SHAP evaluation, and no LLM call. Reserved natural and diagnostic test arrays remain unopened and unmaterialized.

## Task 65 handoff

Only the five seeds frozen by the Task 64 C2 manifest may be used in Task 65. Once Task 64 C2 is frozen, the seed set cannot be changed because of any Task 65 result. Negative and boundary results must be retained.
