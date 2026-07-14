# CIC IoT-DIAD Federated Multi-Seed V2.8

This stage evaluates the V2.7-selected clean FedAvg configuration over five seeds while keeping the exact V2.7 Dirichlet client partition fixed.

Selected configuration:

- Residual MLP
- square-root class-weighted cross-entropy
- learning rate 0.0003
- one local epoch
- 20 clients
- Dirichlet alpha 0.5 partition from V2.7
- full client participation
- validation-only best-round selection

Outputs:

- per-seed round and local-client metrics
- per-seed best checkpoints
- natural and diagnostic test metrics
- per-class metrics and confusion matrices
- mean, standard deviation, and 95% Student t confidence intervals
- DDoS to DoS clean attack baseline
- partition hash verification
- CSV and JSON source artifacts
- matching PNG and PDF figures
- resumable execution with `--skip-existing`

Recommended command:

```cmd
python scripts\run_federated_multiseed_v28.py --data-file "%USERPROFILE%\LFighter-research\data\processed\cic_iot_diad_2024_v2_1\arrays\behavioral_only.npz" --partition-file "%USERPROFILE%\LFighter-research\results\cic_iot_diad_federated_tuning_v27_alpha05_seed42\partitions\client_partitions.npz" --output-dir "%USERPROFILE%\LFighter-research\results\cic_iot_diad_federated_multiseed_v28_fixed_partition" --seeds 42,123,2026,7,99 --num-clients 20 --rounds 30 --participation-rate 1.0 --local-epochs 1 --batch-size 2048 --evaluation-batch-size 4096 --learning-rate 0.0003 --weight-decay 0.0001 --threads 6 --early-stopping-patience 8 --skip-existing
```

The script saves each completed seed independently. If the computer shuts down, rerun the same command with `--skip-existing`; completed seeds will be reused.
