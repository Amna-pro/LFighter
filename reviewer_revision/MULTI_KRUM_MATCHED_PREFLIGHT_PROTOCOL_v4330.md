# Multi-Krum Matched Comparator Preflight Protocol v4.33.0

## Purpose
Add a matched Krum-family robust aggregation comparator under the same frozen reviewer conditions used for Plain FedAvg, BATR FL, P4P, and coordinate-wise median.

This is a validation-only preflight for one attacked condition: all_to_one_benign, model seed 1379954285, 20 clients, malicious clients 1, 7, 8, 10, 14, 15, 17, 18, assumed Byzantine count f=8, project Multi-Krum selection count m=n-f-2=10, frozen round-4 warmup, and global rounds 5 through 8. It reuses the exact recovered Task 65 poison manifest. Training uses train data and during-run evaluation uses validation only; final diagnostic and natural test arrays are not materialized.

No parameter tuning or outcome-based selection is permitted.

## Frozen project implementation
Reuse `multi_krum` from `scripts/verify_task43_c1_baselines_v414.py`. The frozen implementation flattens each update, computes pairwise squared Euclidean distances, scores each update by the sum of distances to its m nearest other updates, deterministically orders by (score, client index), selects the m lowest-score updates, and averages those selected updates.

For n=20 and f=8, m=10. The aggregated floating update is converted back to a model state using `state_from_update`.

The baseline does not run the BATR FL detector, does not reconstruct clients, does not use sample-count weighting inside Multi-Krum, and does not perform attack-specific tuning. The value f=8 is inherited from the already frozen Task 43 configuration and is not chosen from reviewer-run outcomes.

## Matched settings
resmlp; 20 clients; four continuation rounds; one local epoch; batch size 2048; evaluation batch size 4096; learning rate 0.0003; weight decay 0.0001; maximum class weight 4.0; gradient clip norm 5.0; six CPU threads; local RNG seed = model_seed + global_round*1000 + client_id.

## Preflight pass rule
The preflight passes only if the historical Task 43 unit tests pass, all four monitored rounds complete, exactly 10 updates are selected each round, source/partition/poison identities match, output cardinalities are correct, and metadata confirms no final-test access and no attack-specific retuning. Validation performance is not a pass/fail criterion.
