# BATR-FL Reconstruction/Mitigation Ablation Preflight v4.34.0e

## Correction from v4.34.0d

The v4.34.0d setup attempt stopped before its own protocol freeze and before any scientific outcome.
Its AST patcher correctly found the inherited V3.20B.1 CLI guard, but a verification step then looked
for the full error message as one contiguous source substring. The original source expresses that
message as adjacent string literals, so the verification failed even though the AST target was valid.

v4.34.0e changes only that pre-outcome patcher verification. It removes the smallest AST `if` block
whose subtree raises the exact combined message
`--reconstruction-calibration-dir is required for trusted_reconstruction`, then verifies structurally
that no such raise remains. No detector, mitigation, training, attack, seed, threshold, safety, or
evaluation setting is changed.

The previously frozen v4.34.0c protocol at commit `cac0238` remains the parent evidence state. Its
failed center-only launch is preserved. The failed v4.34.0d setup files are backed up outside the repo
before v4.34.0e is installed.

## Preflight arms

### center_only
Flagged client updates are replaced by the current trusted coordinate-median center. Historical
client residuals are not loaded or added. Original client sample-count aggregation weights remain.

### hard_rejection
Flagged clients are excluded from aggregation. Sample-count weights are renormalized over admitted
clients. No reconstruction is performed.

## Frozen detector and experimental settings

- 20 logical clients with full participation.
- Fixed Dirichlet alpha = 0.5 partition.
- Common real round-4 warmup checkpoint.
- Balanced server probe: 48 examples per class, probe seed 3701.
- EMA decay 0.65.
- EMA q99 plus instantaneous q95 thresholds from frozen trusted warmup behavior.
- Flag rule: `max(ema_ratio, instant_ratio) > 1`.
- Malicious coalition: clients `1,7,8,10,14,15,17,18`.
- Attack: `all_to_one_benign`.
- Poison fraction 1.0.
- Model seed = attack seed = 1379954285.
- Global rounds 5-8 only.
- Batch size 2048; evaluation batch size 4096.
- Learning rate 0.0003; weight decay 0.0001.
- Maximum class weight 4.0; gradient clip norm 5.0.
- CPU threads 6.
- Minimum trusted-client safety guard remains 8.
- Malicious labels are reporting-only.
- No reserved final-test access.
- No attack-specific retuning.
- No scientific outcome gate.

The detector rule and calibration are frozen. The two mitigation arms are closed-loop variants, so
realized detector decisions after the first monitored round may diverge because the global model
trajectory changes.
