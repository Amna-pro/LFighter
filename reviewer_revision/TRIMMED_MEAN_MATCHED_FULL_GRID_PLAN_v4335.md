# Trimmed Mean Matched Full Grid Plan v4.33.5

## Scope

Scale the already verified v4.33.4 trimmed-mean comparator from the single attacked preflight
to the complete frozen primary reviewer grid without changing the implementation.

Grid:
* seeds: 1379954285, 1886033230, 480705558, 1377035733, 1707771978
* attacks: all_to_one_benign, cyclic_shift, multiclass_partial_cycle, pairwise_swap, random_flip
* 25 attacked conditions total
* four continuation rounds per condition, global rounds 5 through 8

The audited `all_to_one_benign / seed_1379954285` preflight is reused. The remaining 24
conditions are run with the frozen v4.33.4 runner.

## Frozen trimmed-mean settings

* implementation: scripts/verify_task43_c1_baselines_v414.py
* historical setting source: scripts/run_task43_c2_seed7_screen_v414.py
* submitted clients: 20
* trim_fraction: 0.2
* symmetric coordinate-wise trim
* trim count per side: 4
* retained values per coordinate: 12
* all submitted updates participate before per-coordinate trimming
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

Validation outcomes are retained exactly as observed. No attack/seed condition may be removed
and the trim fraction may not be changed after viewing outcomes.

No final-test evaluation and no formal cross-method inferential analysis are performed here.
