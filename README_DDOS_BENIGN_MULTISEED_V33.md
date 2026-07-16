# CIC IoT-DIAD DDoS-to-Benign Multi-Seed V3.3

V3.2 froze three monotonic attack settings using validation-only calibration:

- Weak: 10% malicious clients, 25% local DDoS poisoning
- Moderate: 20% malicious clients, 50% local DDoS poisoning
- Strong: 40% malicious clients, 100% local DDoS poisoning

V3.3 repeats each setting over the same five model seeds used by the clean
federated baseline:

`42, 123, 2026, 7, 99`

Scientific controls:

- exact V2.7 and V2.8 client partition
- paired clean reference for each model seed
- fixed malicious-client IDs within each strength
- fixed attack seed 42
- fixed poisoned-row set within each strength
- independent validation-only best-round selection for each seed
- natural and diagnostic tests not used for strength selection
- mean, standard deviation, and 95% Student t intervals
- paired Wilcoxon clean-versus-attack tests
- Friedman and Holm-adjusted pairwise strength comparisons
- paired Cohen dz effect sizes
- resumable execution

Recommended command:

```cmd
python scripts\run_ddos_benign_multiseed_v33.py --data-file "%USERPROFILE%\LFighter-research\data\processed\cic_iot_diad_2024_v2_1\arrays\behavioral_only.npz" --partition-file "%USERPROFILE%\LFighter-research\results\cic_iot_diad_federated_tuning_v27_alpha05_seed42\partitions\client_partitions.npz" --clean-multiseed-dir "%USERPROFILE%\LFighter-research\results\cic_iot_diad_federated_multiseed_v28_fixed_partition" --v32-results-dir "%USERPROFILE%\LFighter-research\results\cic_iot_diad_ddos_benign_strength_grid_v32_seed42" --output-dir "%USERPROFILE%\LFighter-research\results\cic_iot_diad_ddos_benign_multiseed_v33" --seeds 42,123,2026,7,99 --source-class DDoS --target-class Benign --min-source-samples 1000 --attack-seed 42 --num-clients 20 --rounds 30 --participation-rate 1.0 --local-epochs 1 --batch-size 2048 --evaluation-batch-size 4096 --learning-rate 0.0003 --weight-decay 0.0001 --threads 6 --early-stopping-patience 8 --skip-existing
```

The script contains 15 runs. Completed runs are reused. An incomplete run is
removed and restarted automatically.
