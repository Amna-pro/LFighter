# CIC IoT-DIAD Attack Strength Grid V3.0

This stage calibrates the first targeted label-flipping attack before any LFighter defense comparison.

Attack pair:

- Source class: DDoS
- Target class: DoS

Grid:

- Malicious clients: 10%, 20%, and 40%
- Poisoned DDoS rows on malicious clients: 50% and 100%
- Total runs: 6
- Model seed: 42
- Attack seed: 42
- Fixed V2.7 and V2.8 client partition
- Learning rate: 0.0003
- One local epoch
- Full client participation

Scientific controls:

- The 10%, 20%, and 40% malicious-client sets are nested.
- The 50% poisoned rows are a subset of the 100% poisoned rows for each client.
- Clean class weights remain fixed.
- Features are never modified.
- Each run selects its best checkpoint using validation macro F1 only.
- Attack-strength ranking uses validation attack effect, not test performance.
- Natural and diagnostic test results are reported only for interpretation.
- Runs are resumable with `--skip-existing`.

Recommended command:

```cmd
python scripts\run_attack_strength_grid_v30.py --data-file "%USERPROFILE%\LFighter-research\data\processed\cic_iot_diad_2024_v2_1\arrays\behavioral_only.npz" --partition-file "%USERPROFILE%\LFighter-research\results\cic_iot_diad_federated_tuning_v27_alpha05_seed42\partitions\client_partitions.npz" --clean-seed-dir "%USERPROFILE%\LFighter-research\results\cic_iot_diad_federated_multiseed_v28_fixed_partition\seed_runs\seed_42" --output-dir "%USERPROFILE%\LFighter-research\results\cic_iot_diad_attack_strength_grid_v30_ddos_to_dos_seed42" --source-class DDoS --target-class DoS --malicious-fractions 0.10,0.20,0.40 --poison-fractions 0.50,1.00 --anchor-fraction 0.20 --min-source-samples 1000 --attack-seed 42 --model-seed 42 --num-clients 20 --rounds 30 --participation-rate 1.0 --local-epochs 1 --batch-size 2048 --evaluation-batch-size 4096 --learning-rate 0.0003 --weight-decay 0.0001 --threads 6 --early-stopping-patience 8 --skip-existing
```

Outputs include:

- nested malicious-client ranking
- exact nested client sets
- exact poisoned-row hashes
- six complete V2.9.1 attack runs
- validation-only attack-strength ranking
- clean and attacked overall metrics
- source-target attack rates
- CSV source tables
- JSON metadata
- checkpoints
- PNG and PDF figures

This is a single-seed attack calibration stage. After review, selected weak, moderate, and strong settings will be repeated over multiple seeds.
