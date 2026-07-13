# CIC IoT-DIAD Clean Federated Baseline V2.6

This stage validates clean FedAvg before poisoning and defense experiments.

Scientific safeguards:

- Residual MLP with the validation-selected square-root class weighting
- reproducible Dirichlet non-IID clients
- saved partition indices and SHA-256 partition hash
- validation-only communication-round selection
- no natural or diagnostic test use during model selection
- natural and diagnostic test evaluation after selection
- full client, round, per-class, calibration, and runtime records
- CSV tables
- matching PNG and PDF figures
- compressed sample-level prediction CSVs
- reusable best checkpoint and client partition

Recommended smoke test:

```cmd
python scripts\run_federated_clean_v26.py --data-file "%USERPROFILE%\LFighter-research\data\processed\cic_iot_diad_2024_v2_1\arrays\behavioral_only.npz" --output-dir "%USERPROFILE%\LFighter-research\results\cic_iot_diad_federated_clean_v26_smoke" --num-clients 20 --dirichlet-alpha 0.5 --rounds 3 --participation-rate 1.0 --local-epochs 1 --batch-size 2048 --seed 42 --threads 6
```

This smoke test is an engineering validation, not a final paper result.
