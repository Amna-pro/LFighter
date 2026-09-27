# Matched P4P Full Grid Scaling Note v4.32.5

## Purpose

This note freezes the orchestration for the remaining matched P4P comparator conditions after the
single audited preflight condition completed successfully.

The P4P scientific configuration itself was frozen before the first attacked outcome and is not
changed here. The first observed condition is preserved exactly and is not rerun.

## Grid

Five Task 65 final seeds:

1379954285, 1886033230, 480705558, 1377035733, 1707771978

Five frozen label poisoning attacks:

all_to_one_benign, cyclic_shift, multiclass_partial_cycle, pairwise_swap, random_flip

Twenty logical clients with malicious coalition:

1, 7, 8, 10, 14, 15, 17, 18

The common matched training settings remain:

* continuation rounds: 4, corresponding to global rounds 5 through 8
* batch size: 2048
* evaluation batch size: 4096
* learning rate: 0.0003
* weight decay: 0.0001
* maximum class weight: 4.0
* gradient clip norm: 5.0
* threads: 6
* aggregation: sample count weighted FedAvg over the P4P trusted set

The frozen P4P configuration remains:

* MAD k: 3.0
* DBSCAN eps: 0.5
* DBSCAN min samples: 5
* Isolation Forest contamination: 0.1
* Isolation Forest estimators: 100
* KMeans clusters: 2
* ensemble vote threshold: 2
* temporal suspicion decay: 0.2
* permanent suspicion threshold: 2.0
* detector random state: 42
* KMeans n_init: 10

## Provenance rules

1. Every condition must reuse the verified Task 65 recovered attack manifest for the same attack and seed.
2. Every condition must reuse the verified seed-specific Task 65 warmup checkpoint and P4P probe bootstrap.
3. The already audited all_to_one_benign seed 1379954285 condition is reused without rerun.
4. Existing completed conditions may be skipped only after their completion metadata passes validation.
5. A partially existing output directory without completion metadata causes an immediate stop. Nothing is deleted automatically.
6. No P4P parameter may be changed in response to observed outcomes.
7. The runner must report test_sets_accessed false and attack_specific_retuning false for every condition.
8. The final summary is descriptive across all 25 conditions. Formal paired statistical comparisons are a later frozen analysis stage.
