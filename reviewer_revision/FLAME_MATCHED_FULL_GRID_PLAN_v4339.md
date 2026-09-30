# Matched FLAME Five-Attack Five-Seed Grid Plan v4.33.9

Scale the verified v4.33.8 FLAME comparator to the complete frozen reviewer grid without changing
the implementation or environment.

Grid: five final seeds × five attacks = 25 conditions, with global rounds 5–8. Reuse the audited
all_to_one_benign / seed_1379954285 preflight exactly; run the other 24 conditions with the frozen
v4.33.8 runner.

Frozen FLAME settings: src/aggregation.py SHA256
C964742E0279EF98CC3B3D0CE36C62282AF1BE9C7B94D3B8B484240085333F0D;
reviewer-frozen HDBSCAN 0.8.44 (not claimed as a recovered historical pin); min_cluster_size=11,
min_samples=1, allow_single_cluster=True; median Euclidean-distance clipping; equal average of
admitted clipped states; lambda=0.001; noise_scalar=1.0; preserved normal_(std=sigma**2) behavior;
reviewer noise RNG seed=model_seed+global_round*1000+999; no noise-scalar adaptation.

Hold fixed the W4 checkpoint, partition, architecture, local optimizer/hyperparameters, exact Task-65
poison plan, attack definition, validation procedure, and local RNG schedule. Training and validation
only; do not materialize final diagnostic or natural test arrays. Retain all outcomes. No retuning,
final-test evaluation, or formal cross-method inference in this stage.
