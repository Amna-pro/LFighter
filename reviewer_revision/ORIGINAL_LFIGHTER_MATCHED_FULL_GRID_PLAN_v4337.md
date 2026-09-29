# Original LFighter Matched Full Grid Plan v4.33.7

## Scope

Scale the already verified v4.33.6 Original LFighter comparator from the single attacked
preflight to the complete frozen primary reviewer grid without changing the implementation.

Grid:
* seeds: 1379954285, 1886033230, 480705558, 1377035733, 1707771978
* attacks: all_to_one_benign, cyclic_shift, multiclass_partial_cycle, pairwise_swap, random_flip
* 25 attacked conditions total
* four continuation rounds per condition, global rounds 5 through 8

The audited `all_to_one_benign / seed_1379954285` preflight is reused. The remaining 24
conditions are run with the frozen v4.33.6 runner.

## Frozen Original LFighter settings

* source: src/lfighter_original_v34.py
* source SHA256: A2690DECAE8A4589EA4A620171AD195E2F5FE8B8E742F529BEA9685C1167D9EC
* kmeans_seed: 0
* KMeans clusters: 2
* n_init: 10
* final-classifier two-class salience rule from historical V3.4 implementation
* original cluster dissimilarity rule
* equal average of admitted local states
* malicious client labels are used only for diagnostics, never for clustering or admission
* no attack-specific tuning

## Matched conditions

Each condition reuses the same fixed partition, seed-specific frozen W4 checkpoint, exact
Task-65 poison plan, ResMLP architecture, local optimizer/hyperparameters, and local RNG rule
already used by the other matched reviewer comparators.

## Data-access and outcome policy

Training and validation arrays only. Final diagnostic and natural test arrays are not materialized.

Validation outcomes, selected class pairs, admitted/rejected counts, malicious rejection recall,
benign false-rejection rate, and fallback events are retained exactly as observed.

No condition may be removed and no Original LFighter parameter may be changed after viewing
outcomes.

No final-test evaluation and no formal cross-method inferential analysis are performed here.
