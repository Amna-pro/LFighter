# Matched FLAME Comparator Preflight Protocol v4.33.8

## Purpose

Add a reviewer-matched FLAME comparator using the preserved project implementation in
`src/aggregation.py`, while holding the same frozen reviewer training protocol used by the
other matched baselines.

This is a reproduction of the preserved project FLAME implementation under a newly frozen
reviewer environment. The exact historical HDBSCAN package version was not recoverable from
the repository requirements; the repository only required `hdbscan>=0.8`. Before any FLAME
reviewer outcome was observed, pip dry-run resolved HDBSCAN 0.8.44, and that exact version
was then installed and frozen for this reviewer comparator.

## Frozen source / environment

* `src/aggregation.py`
  SHA256: `C964742E0279EF98CC3B3D0CE36C62282AF1BE9C7B94D3B8B484240085333F0D`
* reviewer-frozen HDBSCAN: `0.8.44`
* Python observed before reviewer FLAME outcomes: `3.10.11`
* torch: `2.13.0+cpu`
* numpy: `1.26.4`
* scipy: `1.15.3`
* scikit-learn: `1.7.2`
* joblib: `1.5.3`

Do not describe HDBSCAN 0.8.44 as the recovered historical package version.

## Frozen preserved FLAME rule

For 20 submitted client models in each monitored round:

1. Flatten the current global model and all local models.
2. Compute global-minus-local model vectors.
3. Compute the model-wise cosine-similarity matrix.
4. Run external-package `hdbscan.HDBSCAN` with:
   * `min_cluster_size = int(20 * 0.5) + 1 = 11`
   * `min_samples = 1`
   * `allow_single_cluster = True`
5. Treat HDBSCAN label `-1` as rejected/outlier. If all 20 labels are `-1`, admit all 20,
   exactly as the preserved source does.
6. Compute each local model's Euclidean distance to the current global model.
7. Set clipping threshold `st` to the median of those 20 Euclidean distances.
8. For every admitted client, apply the preserved clipping rule
   `global + (local - global) * min(1, st / distance)`.
9. Equally average admitted clipped model states.
10. Add adaptive Gaussian noise using the preserved source:
    * `lambda = 0.001`
    * `noise_scalar = 1.0`
    * `sigma = lambda * st * noise_scalar`
    * the preserved implementation passes `std=(sigma**2)` to `normal_`; this behavior is
      retained exactly rather than silently changed.

The historical simulation initialized `noise_scalar=1.0` and halved it only after observing
a NaN test loss. The matched reviewer comparator does not use final test behavior as a training
control, so `noise_scalar` remains fixed at 1.0. If a numerical failure occurs, preserve the
partial output and stop; do not halve the parameter or retune.

## Reviewer-frozen noise RNG schedule

The FLAME source draws Gaussian noise from PyTorch's RNG but does not accept an explicit seed.
For reproducible reviewer comparison, before each FLAME aggregation call the runner sets:

`flame_noise_seed = model_seed + global_round * 1000 + 999`

This RNG schedule is frozen before any FLAME reviewer outcome and is not an algorithmic
performance parameter.

## Matched preflight condition

* attack: `all_to_one_benign`
* model seed: `1379954285`
* clients: 20
* malicious clients: frozen 8-client coalition from the exact Task-65 attack manifest
* frozen warmup checkpoint: global round 4
* continuation: global rounds 5, 6, 7, 8
* exact recovered Task-65 poison plan
* train and validation arrays only
* no final diagnostic or natural test arrays materialized

## Matched local training

* model: frozen ResMLP
* local epochs: 1
* batch size: 2048
* evaluation batch size: 4096
* learning rate: 0.0003
* weight decay: 0.0001
* maximum class weight: 4.0
* gradient clip norm: 5.0
* CPU threads: 6
* local RNG seed: `model_seed + global_round * 1000 + client_id`

## Outcome policy

Validation utility, HDBSCAN admission/rejection, malicious rejection recall, benign false
rejection rate, clipping threshold, clipping factors, and fallback behavior are descriptive
outputs only. They are not tuning gates.

No attack-specific retuning, no final-test evaluation, and no scientific outcome gate are
allowed in this preflight.
