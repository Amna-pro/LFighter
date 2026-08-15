# Task 64 Preregistration V4.27

## Purpose

Task 64 reserves untouched final model seeds after the complete Task 63 protocol freeze and before any one-shot final natural or diagnostic test evaluation.

The roadmap identifies seeds 7, 42, 99, 123, and 2026 as development-used and therefore ineligible as fully untouched final seeds. Task 64 does not train a model, inspect final outcomes, run SHAP, call an LLM, or materialize reserved test arrays.

## Frozen selection rule

Five final seeds are reserved. Candidates are generated deterministically from SHA256 over the frozen namespace `LFighter|task64_v4270|parent_tag=task63-c1-final-protocol-frozen-v4261|candidate={candidate_index}`. Candidate indices are considered in ascending order.

A candidate is accepted only if it is unique, is not one of the five development seeds, and has zero exact numeric-token usage hits in the audited pre-Task64 repository corpus. Any exact occurrence conservatively rejects the candidate, even if the occurrence may be unrelated to random seeding. Manual substitution, best-seed selection, and outcome-based selection are prohibited.

## Prior-usage audit

The selector audits file names and supported text-file contents in the repository while excluding `.git`, `.venv`, caches, Task 64 files, Task 64 outputs, and paths containing reserved-test array names. Binary arrays are never loaded. If an eligible text file cannot be read, Task 64 fails rather than silently weakening the untouched-seed claim.

## Final boundary

After the Task 64 manifest is frozen, the reserved seed set cannot be changed because of Task 65 outcomes. Task 65 may use only the frozen Task 64 seed manifest for the final seed dimension. Negative and boundary results remain reportable and no post-outcome retuning is allowed.
