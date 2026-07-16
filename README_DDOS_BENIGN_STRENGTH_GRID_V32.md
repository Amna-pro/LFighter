# CIC IoT-DIAD DDoS-to-Benign Strength Grid V3.2

V3.1.1 selected DDoS to Benign as the strongest validation-controlled targeted label-flipping pair.

V3.2 calibrates its strength using nine nested settings:

- Malicious clients: 10%, 20%, and 40%
- Poisoned DDoS rows on malicious clients: 25%, 50%, and 100%
- Fixed client partition
- Model seed 42
- Attack seed 42
- Learning rate 0.0003
- One local epoch
- Full participation

Controls:

- malicious-client sets are nested
- poisoned rows are nested within every client
- clean class weights remain fixed
- features are unchanged
- best checkpoints use validation macro F1 only
- strength ranking uses validation attack-rate increase only
- natural and diagnostic tests are reported after selection

Recommended command:

```cmd
python scripts\run_ddos_benign_strength_grid_v32.py --data-file "%USERPROFILE%\LFighter-research\data\processed\cic_iot_diad_2024_v2_1\arrays\behavioral_only.npz" --partition-file "%USERPROFILE%\LFighter-research\results\cic_iot_diad_federated_tuning_v27_alpha05_seed42\partitions\client_partitions.npz" --clean-seed-dir "%USERPROFILE%\LFighter-research\results\cic_iot_diad_federated_multiseed_v28_fixed_partition\seed_runs\seed_42" --output-dir "%USERPROFILE%\LFighter-research\results\cic_iot_diad_ddos_benign_strength_grid_v32_seed42" --source-class DDoS --target-class Benign --malicious-fractions 0.10,0.20,0.40 --poison-fractions 0.25,0.50,1.00 --anchor-fraction 0.20 --min-source-samples 1000 --attack-seed 42 --model-seed 42 --num-clients 20 --rounds 30 --participation-rate 1.0 --local-epochs 1 --batch-size 2048 --evaluation-batch-size 4096 --learning-rate 0.0003 --weight-decay 0.0001 --threads 6 --early-stopping-patience 8 --skip-existing
```

The controller is resumable. Completed settings are reused. An incomplete setting is removed and restarted automatically.
