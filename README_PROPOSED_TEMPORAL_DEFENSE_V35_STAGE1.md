# Proposed Temporal Defense V3.5, Stage 1

This package starts development of the independent proposed defense. It is not the final paper result and it does not reproduce the original LFighter rule.

The stage-1 method combines:

1. robust update-magnitude evidence,
2. update-direction distance from a coordinate-wise median update,
3. class-conditional prediction drift on a fixed validation probe,
4. gradient-times-input explanation drift,
5. temporal suspicion memory,
6. soft risk-aware aggregation weights,
7. persistent-client quarantine with a minimum-admission safeguard.

Malicious-client labels are used only for retrospective evaluation. They are not used to calculate aggregation weights or rejection decisions. Test sets are never used for round selection. The prediction and explanation probes come only from the validation split and are saved in CSV manifests.

## Stage-1 protocol

First run short clean and attacked smoke tests. If both complete, run one full clean seed and one full attacked seed. Review the security and utility results before any five-seed experiment. This prevents wasting several hours on an unvalidated defense configuration.

## Expected files

- `scripts/run_proposed_temporal_defense_v35.py`
- `src/proposed_temporal_defense_v35.py`
- `README_PROPOSED_TEMPORAL_DEFENSE_V35_STAGE1.md`

Every run saves CSV tables, PNG figures, PDF figures, checkpoints, attack manifests, probe manifests, and metadata.
