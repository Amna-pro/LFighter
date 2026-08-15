# Task 63 C2 Protocol Completion Amendment V4.26.2

## Why this amendment exists
Task 63 V4.26.0 correctly froze the method boundary, claim ledger, one-shot Task 65 rule, Holm control, and prohibition on post-outcome tuning. However, its machine-readable protocol also referred to a separate final-evaluation manifest containing the exact confirmatory attack panel, endpoint, inferential test, bootstrap seed, and practical-significance thresholds. That manifest was not actually materialized before Task 64 C1.

Task 64 C1 was nevertheless outcome-free: it performed no training, opened no reserved natural/diagnostic test path, loaded no binary test array, made no SHAP evaluation, and made no LLM call. Therefore no scientific outcome has been inspected. This amendment records the ordering deviation rather than hiding it.

## Clean recovery rule
The five Task 64 C1 seeds are permanently recorded as `SUPERSEDED_UNUSED_FOR_FINAL_EVALUATION`. They must never be used in Task 65. After this amendment is frozen, Task 64 C2 must deterministically reserve a new five-seed set from a new namespace and independently prove zero prior use. Only that new Task 64 C2 manifest may enter Task 65.

## Final confirmatory scope
Task 65 is restricted to the previously implemented frozen untargeted label-poisoning panel:

* all_to_one_benign
* cyclic_shift
* multiclass_partial_cycle
* pairwise_swap
* random_flip

The poison fraction is 1.0. The malicious coalition is the original eight-client development coalition `[1,7,8,10,14,15,17,18]`. The chronology is four trusted clean warmup rounds followed by four monitored attack rounds. No Task 46-54 condition may be introduced.

The final comparison arms are: one clean FedAvg reference per final seed, matched attacked plain FedAvg, and matched attacked frozen LFighter `center_plus_residual`. Task 43 robust-aggregation baselines remain valid development/boundary evidence, but are not promoted into this cross-attack final panel because no frozen cross-attack implementation was established before the final gate.

## Final test endpoint
Both full reserved datasets are evaluated: diagnostic and natural. Round 8 is the sole primary endpoint. Rounds 5-8 may be reported only as prespecified secondary trajectories; no best-round selection is allowed. All eight classes must be reported with precision, recall, F1, support, and confusion matrices.

## Statistics frozen before Task 65
Pairing is by new final seed, attack, dataset, and endpoint. The primary uncertainty procedure is a deterministic paired percentile bootstrap with 20,000 replicates and bootstrap seed `650428`. The exact paired sign-flip test is two-sided and is reported as resolution-limited with five seeds; the minimum attainable two-sided p-value is 0.0625. Holm family-wise adjustment is applied across the five attacks within each dataset and primary metric family. Effect reporting includes paired mean difference, paired median difference, Cohen dz when defined, and relative attack-excess reduction.

Practical thresholds are frozen as follows: clean mean macro-F1 loss no worse than 0.01 with no individual seed loss above 0.02; benign FPR at most 0.05; mean malicious-client recall at least 0.50; and mean relative attack-excess reduction at least 0.25 with positive mitigation in at least four of five final seeds. Failure of a threshold is retained as a limitation and cannot trigger retuning.

## Data boundary
This amendment performs no training and may not materialize `X_test_natural`, `y_test_natural`, `X_test_diagnostic`, or `y_test_diagnostic`. Task 65 remains blocked until this amendment and a replacement Task 64 C2 seed reservation are both frozen.
