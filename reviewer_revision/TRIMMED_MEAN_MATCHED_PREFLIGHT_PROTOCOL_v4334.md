# Trimmed Mean Matched Comparator Preflight Protocol v4.33.4

## Purpose

Add a matched coordinate-wise trimmed-mean robust aggregation comparator under the same frozen
reviewer conditions used for Plain FedAvg, BATR-FL, P4P, coordinate-wise median, Multi-Krum,
and geometric median.

## Frozen historical setting

The reviewer comparator uses the exact Task-43 baseline setting rather than the separate robust-
center audit setting.

Task 43 explicitly froze:
- `TRIMMED_MEAN_FRACTION = 0.2`
- 20 submitted clients
- symmetric coordinate-wise trimming
- 4 smallest and 4 largest client values removed per coordinate
- 12 values retained and averaged per coordinate

The implementation is imported directly from
`scripts/verify_task43_c1_baselines_v414.py`.

The separate `scripts/audit_robust_centers_v319a.py` default `trim_f=8` belonged to a different
robust-center audit. It is not used to redefine the actual Task-43 comparator after outcomes.

## Preflight condition

- attack: all_to_one_benign
- model seed: 1379954285
- clients: 20
- frozen warmup: global round 4
- continuation: global rounds 5 through 8
- exact recovered Task-65 poison plan
- training and validation only
- final diagnostic/natural test arrays are not materialized

No parameter tuning or outcome-based selection is permitted.

## Matching constraints

- model: ResMLP
- local epochs: 1
- batch size: 2048
- evaluation batch size: 4096
- learning rate: 0.0003
- weight decay: 0.0001
- maximum class weight: 4.0
- gradient clip norm: 5.0
- CPU threads: 6
- local RNG seed: model_seed + global_round * 1000 + client_id

The baseline receives all 20 submitted updates and applies coordinate-wise symmetric trimmed mean
with trim_fraction=0.2. It uses no detector, client rejection, reconstruction, or sample-count
weighting.

Validation performance is not an implementation gate.
