# Coordinate Median Matched Comparator Protocol v4.32.8

## Purpose

Add a strong robust aggregation baseline under the same frozen reviewer comparison conditions
used by the final BATR FL evaluation and the matched P4P comparator.

This stage is a validation-only implementation preflight for one attacked condition:

* attack: all_to_one_benign
* model seed: 1379954285
* clients: 20
* malicious coalition: 1, 7, 8, 10, 14, 15, 17, 18
* warmup: frozen clean global round 4
* continuation: global rounds 5 through 8
* poison plan: exact recovered Task 65 plain FedAvg poison manifest
* training data: train only
* evaluation during the run: validation only
* final diagnostic and natural test arrays: not materialized

No parameter tuning or outcome-based selection is permitted.

## Coordinate median definition

The comparator reuses the already verified project implementation from
`src/trusted_update_reconstruction_v312.py`.

For every floating model-update coordinate, the server stacks the corresponding value from all
20 submitted client updates and takes `torch.median(..., dim=0).values`.

The resulting aggregated update is converted back to a model state with the project's existing
`state_from_update` helper.

This baseline:

* aggregates all 20 submitted updates,
* does not run the BATR FL detector,
* does not classify clients,
* does not reject clients,
* does not reconstruct clients,
* does not use sample-count weights in the coordinate median,
* does not use attack-specific tuning.

This matches the semantics documented and used in Task 43 for the coordinate-median arm.

## Matching constraints

The runner preserves the matched P4P/BATR local-training settings:

* model: resmlp
* number of clients: 20
* continuation rounds: 4
* local epochs: 1
* batch size: 2048
* evaluation batch size: 4096
* learning rate: 0.0003
* weight decay: 0.0001
* maximum class weight: 4.0
* gradient clip norm: 5.0
* CPU threads: 6
* local RNG seed: model_seed + global_round * 1000 + client_id

The runner verifies the fixed partition hash through the existing partition and clean-seed
records and verifies that the warmup metadata seed and partition match.

## Preflight decision rule

The preflight passes implementation validation only if:

1. the branch completes all four monitored rounds,
2. the completion metadata records `test_sets_accessed = false`,
3. the completion metadata records `attack_specific_retuning = false`,
4. the exact recovered poison hash matches the frozen manifest-recovery evidence,
5. the round-8 checkpoint exists and records the expected seed, attack, partition, poison hash,
   and arm,
6. validation tables have the expected cardinalities,
7. the working comparison code and the reused robust-aggregation source identities are frozen.

The validation macro F1 values themselves do not determine whether the implementation passes.

After a successful preflight, the same frozen runner can be orchestrated across the remaining
5-attack by 5-seed grid without changing the aggregation rule.
