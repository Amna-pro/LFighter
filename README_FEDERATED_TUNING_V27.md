# CIC IoT-DIAD Federated Tuning V2.7

This stage performs validation-only clean FedAvg hyperparameter screening on one fixed Dirichlet client partition.

Candidates:

- learning rate 0.0003, one local epoch
- learning rate 0.0008, one local epoch
- learning rate 0.0015, one local epoch
- learning rate 0.0008, two local epochs

Scientific safeguards:

- the same client partition is used for every candidate
- client indices and SHA-256 partition hash are saved
- candidate ranking uses validation metrics only
- test sets are not evaluated until the winner is fixed
- only the selected candidate is evaluated on natural and diagnostic tests
- CSV, JSON, PNG, PDF, checkpoints, and compressed prediction tables are saved

Recommended command:

```cmd
python scripts\screen_federated_clean_v27.py --data-file "%USERPROFILE%\LFighter-research\data\processed\cic_iot_diad_2024_v2_1\arrays\behavioral_only.npz" --output-dir "%USERPROFILE%\LFighter-research\results\cic_iot_diad_federated_tuning_v27_alpha05_seed42" --num-clients 20 --dirichlet-alpha 0.5 --rounds 30 --participation-rate 1.0 --batch-size 2048 --evaluation-batch-size 4096 --weight-decay 0.0001 --seed 42 --threads 6 --early-stopping-patience 8
```
