# BATR-FL Warmup-Calibration Recovery + Reconstruction Ablation Preflight v4.34.0f

## Why v4.34.0f is needed

The frozen v4.34.0e reviewer protocol was committed before outcomes, but its first center-only launch
stopped before monitored training because the reviewer Task-65 warmup recovery contains the exact
round-4 checkpoint and trusted profile NPZ but not the full historical calibration CSV set required
by the V3.20B.1 detector path.

The deleted historical Task-65 directory is not used as an unverifiable source. Instead, v4.34.0f
replays the already-frozen four-round trusted warmup using the historical `run_true_warmup_v310.py`
implementation and the same data, partition, clean-seed record, model seed, probe, and training
hyperparameters. This replay is performed before any ablation outcome.

The recovered warmup is accepted only if all of the following pre-outcome checks pass:

1. The recovered round-4 model state is tensor-exact to the already verified reviewer Task-65 W4.
2. The recovered `trusted_client_profiles.npz` arrays are exactly equal to the already verified
   reviewer Task-65 trusted profiles.
3. Partition hash is
   `5f14674b0c775f06c3400c5e1e3b9649b78c82d4257c2293411f5c6fe53bed48`.
4. Probe hash is
   `4f4243b5725fd80f8f17be179711241a0fd9b459acc5ea6f1725b880c54865bf`.
5. The recovered detector calibration reproduces the historical V3.20B.1 thresholds to six decimals:
   EMA q99 = `2.749450`, instantaneous q95 = `3.977139`.
6. `feature_calibration.csv` and `clean_leave_one_round_out_scores.csv` exist and are finite.
7. Reserved test arrays are not materialized.

If any check fails, execution stops before either ablation arm.

## Frozen ablation arms

### center_only
The frozen detector flags clients. A flagged update is replaced by the current trusted
coordinate-median center only. Historical client residuals are disabled. All original client
sample-count weights remain in aggregation.

### hard_rejection
The same frozen detector flags clients. Flagged clients are removed from aggregation and
sample-count weights are renormalized over the admitted clients. No reconstruction is performed.

## Frozen experiment settings

- 20 controlled logical clients, full participation.
- Fixed Dirichlet alpha 0.5 partition.
- Four trusted warmup rounds.
- Balanced server probe: 48 samples per class, probe seed 3701.
- EMA decay 0.65.
- Detector: EMA q99 plus instantaneous q95, flag if max ratio > 1.
- Malicious clients: 1,7,8,10,14,15,17,18.
- Attack: all_to_one_benign.
- Poison fraction: 1.0.
- Model seed = attack seed = 1379954285.
- Monitored global rounds 5-8.
- Batch size 2048; evaluation batch size 4096.
- Learning rate 0.0003; weight decay 0.0001.
- Max class weight 4.0; gradient clip norm 5.0.
- CPU threads 6.
- Minimum trusted-client safety guard: 8.
- Malicious labels are reporting-only.
- No attack-specific retuning.
- No final-test access.
- No scientific outcome gate.

The v4.34.0e reviewer runner is reused unchanged after the warmup calibration recovery is verified.
Its source hash remains part of the v4.34.0e frozen evidence.
