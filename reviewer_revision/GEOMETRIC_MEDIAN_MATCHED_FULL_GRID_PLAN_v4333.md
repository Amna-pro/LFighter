# Geometric Median Matched Full Grid Plan v4.33.3

## Scope

Scale the already verified v4.33.2 geometric-median comparator from the single attacked
preflight to the complete frozen primary reviewer grid without changing the implementation.

Grid:
* seeds: 1379954285, 1886033230, 480705558, 1377035733, 1707771978
* attacks: all_to_one_benign, cyclic_shift, multiclass_partial_cycle, pairwise_swap, random_flip
* 25 attacked conditions total
* four continuation rounds per condition, global rounds 5 through 8

The audited `all_to_one_benign / seed_1379954285` preflight is reused. The remaining 24
conditions are run with the frozen v4.33.2 runner.

## Frozen geometric-median settings

* source: scripts/audit_robust_centers_v319a.py
* source SHA256: A8BDA1CB3BAEBB97A00CBBBE86F190D12FF7185A07A08A622CCF52BCD38DC76E
* initialization: NumPy coordinate-wise median
* deterministic Weiszfeld iteration
* maximum iterations: 60
* relative tolerance: 1e-7
* near-point threshold: 1e-12
* float64 update geometry
* equal-client geometric-median objective
* all 20 submitted updates included
* no BATR detector
* no client rejection
* no reconstruction
* no sample-count weighting
* no attack-specific tuning

## Matched conditions

Each condition reuses the same fixed partition, seed-specific frozen W4 checkpoint, exact
Task-65 poison plan, ResMLP architecture, local optimizer/hyperparameters, and local RNG rule
already used by the other matched reviewer comparators.

## Data-access and outcome policy

Training and validation arrays only. Final diagnostic and natural test arrays are not materialized.

Validation outcomes and algorithm convergence status are retained exactly as observed.
No attack/seed condition may be removed and no geometric-median parameter may be changed after
viewing outcomes.

No final-test evaluation and no formal cross-method inferential analysis are performed here.
