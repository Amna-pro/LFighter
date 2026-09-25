# BATR FL Reviewer Revision Protocol Amendment C1 v4.32.2

Status: PRE OUTCOME CORRECTION. This amendment must be committed and tagged before reviewer extension implementation preflight or outcome execution.

Parent freeze: `reviewer-v4321-preoutcome-protocol-frozen` at commit `2e6673f`.

## Reason for amendment

A source level implementation audit performed after the v4.32.1 protocol freeze found that the historical trusted warmup runner retains the final round 4 branch checkpoint, but does not retain a distinct round 3 global checkpoint. The v4.32.1 P4P specification stated that both round 3 and round 4 checkpoints already existed. That statement is therefore operationally incorrect.

No reviewer extension attack outcome had been run or inspected when this discrepancy was discovered. The correction below is based only on source code and historical checkpoint structure.

## C1. P4P initial probe bootstrap correction

P4P requires the first monitored probe

`v_5 = W_4 - W_3`.

The reviewer implementation shall recover `W_3` by deterministic replay of the trusted clean warmup using the exact frozen model seed, client partition, model architecture, local training hyperparameters, local training seeds, and four round chronology used by `run_true_warmup_v310.py`.

The replay must continue through round 4 solely for verification. Before a P4P branch is eligible to run, the replayed round 4 model must match the frozen historical round 4 branch checkpoint.

Required checks:

1. model seed matches the frozen warmup metadata
2. partition hash matches the frozen warmup metadata
3. replay uses train data only and does not materialize either final test array
4. replayed round 4 state has exactly the same state keys as the frozen round 4 checkpoint
5. maximum absolute floating parameter difference between replayed and frozen round 4 states is at most `1e-7`
6. if this equivalence check fails, P4P execution aborts before any attacked reviewer outcome is run
7. once equivalence passes, the replayed round 3 state and frozen round 4 state define the first P4P probe

The replay is a chronology reconstruction step, not a new defense tuning stage.

## C2. P4P deterministic implementation rules

The following implementation details are frozen before outcomes because the P4P paper leaves library level behavior implicit:

1. K Means uses `K=2`, `random_state=42`, and `n_init=10`.
2. If K Means produces fewer than two effective clusters because all probe responses are identical, it contributes no suspicious vote.
3. If the two K Means clusters have equal size, the cluster with the lower mean cosine response is treated as suspicious.
4. DBSCAN uses `eps=0.5` and `min_samples=5`. Label `-1` is the suspicious noise label.
5. Isolation Forest uses contamination `0.1`, `random_state=42`, `n_estimators=100`, and the semantic outlier decision. In scikit learn this means prediction label `-1` is suspicious.
6. Ensemble anomaly status requires at least two suspicious votes among K Means, DBSCAN, and Isolation Forest.
7. Temporal suspicion starts at zero at the beginning of the monitored window and uses decay `0.2` and permanent ban threshold `2.0`.
8. Magnitude rejection is immediate. A magnitude rejected update is not passed to the ensemble and is not aggregated in that round.
9. If no client remains in the trusted set, the branch aborts. No hidden fallback aggregation is substituted.
10. The matched comparator aggregates the retained trusted clients using the experiment's original sample count weights, renormalized over the retained set, as frozen in v4.32.1.

## C3. Runtime phase definitions for P4P

The P4P reviewer runner records `time.perf_counter()` phase timings for:

1. local client training
2. P4P probe construction and cosine response computation
3. P4P magnitude, ensemble, and temporal detector scoring
4. mitigation selection of the trusted set
5. trusted set aggregation
6. validation evaluation

These timings are descriptive and are not used for algorithm selection.

## C4. Outcome boundary

This amendment changes no attack mapping, seed, metric, hypothesis, threshold, or final claim. It corrects an implementation prerequisite discovered before reviewer extension outcomes.

No reviewer extension attacked run is permitted until this amendment and the corresponding implementation are separately frozen in Git.
