# BATR-FL Reconstruction/Mitigation Ablation Preflight v4.34.0c

## Correction note

The first v4.34.0 setup attempt stopped before protocol freeze and before any scientific run because
it tried to discover a final-seed reconstruction-calibration bundle that is not present in the
reviewer repository. No v4.34.0 outcome was produced.

That bundle is not scientifically required by either requested ablation:
- `center_only` uses the trusted coordinate-median center and deliberately omits historical residuals.
- `hard_rejection` removes flagged clients and performs no reconstruction.

This corrected pre-outcome protocol derives both ablations directly from the exact frozen
V3.20B.1 detector/training runner without loading a residual-profile calibration bundle.


## v4.34.0c setup correction

The preceding v4.34.0a setup attempt stopped before protocol freeze and before any scientific outcome.
Its builder searched for the verified source anchor using a literal backslash-n sequence. v4.34.0c
changes only that pre-outcome builder anchor to a newline-independent exact substring search. No
detector, mitigation, training, attack, seed, threshold, or evaluation setting changes.


## v4.34.0c setup correction

The preceding v4.34.0b setup attempt stopped before protocol freeze and before any scientific outcome.
The derived-runner builder passed the literal two-character string `\\n` to `Path.write_text(..., newline=...)`,
which Python 3.10 rejects as an illegal newline value. v4.34.0c changes only that file-writing argument to
a real newline (`"\n"`). No detector, mitigation, training, attack, seed, threshold, or evaluation setting changes.

## Purpose

Compare two post-detection mitigation actions while keeping the detector, attack plan, common
warmup checkpoint, partition, probe, thresholds, local training hyperparameters, coalition, and
random seeds frozen.

### Arm 1: center_only
The frozen detector identifies flagged clients. Unflagged-client updates form the trusted
coordinate-median center. Each flagged client's update is replaced by that center only.
No historical residual is used. Original sample-count aggregation weights are kept.

### Arm 2: hard_rejection
The same frozen detector identifies flagged clients. Flagged clients are removed from aggregation.
The remaining clients retain their original sample-count weights, renormalized over the retained
set, following the historical hard-gate semantics.

## Frozen settings

- 20 clients, full participation.
- Dirichlet-alpha-0.5 frozen partition.
- Common real round-4 warmup checkpoint.
- Probe size 48 per class, probe seed 3701.
- EMA decay 0.65.
- EMA q99 + instantaneous q95 detector.
- Flag rule `max(ema_ratio, instant_ratio) > 1`.
- Frozen malicious coalition `1,7,8,10,14,15,17,18`.
- Attack `all_to_one_benign`.
- Poison fraction 1.0.
- Model/attack seed 1379954285.
- Global rounds 5-8.
- Batch size 2048, evaluation batch size 4096.
- Learning rate 0.0003, weight decay 0.0001.
- Max class weight 4.0, gradient clip norm 5.0, CPU threads 6.
- Minimum trusted-client safety guard remains 8 in both arms.
- Malicious labels are reporting-only.

The detector algorithm and thresholds are frozen, not the realized per-round flag decisions.
After round 5 the two closed-loop trajectories may diverge, so later flags may legitimately differ.

## Frozen source identities

- `scripts/run_frozen_untargeted_defense_v320b1.py`
  SHA256 `5F3852FC13959301B31ADF47F57DF3B65456FB028A647978B355F976E5AEE951`
- `src/trusted_update_reconstruction_v312.py`
  SHA256 `AF5083FF363AE2DE23F802FBA5778111171C23C95537793BF2453F5235CEC42C`
- `src/independent_anchor_v310.py`
  SHA256 `9FEE9BB10E6B4381B30E5488AAB7650E8E2D3919529F4F7652526BD9E3B452FA`
- historical hard-gate reference `scripts/run_aggregation_headroom_v3112.py`
  SHA256 `BFE57C2564D4B38773C390BDBA30788CD7FEE048BCADECB1D03625A0234EF154`

The frozen source files above are not edited.

## Test isolation and outcome policy

Task-65 test-access guard is enabled. Only train/validation arrays are permitted.
This is a structural/reproducibility preflight. Performance is not a pass/fail gate.
No attack-specific retuning, threshold changes, final-test evaluation, or outcome-dependent code changes.
Unfavorable outcomes are retained.
