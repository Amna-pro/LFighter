# CIC IoT-DIAD Attack Pair Screen V3.1.1 Patch

This patch corrects two issues found during V3.1 review:

1. The earlier runner reused the clean DDoS-to-DoS source-target table when
   evaluating a different target class. V2.9.2 now evaluates the clean
   checkpoint dynamically for the requested source and target IDs.

2. The earlier pair controller reused eight DDoS-eligible clients, but two of
   those clients did not contain the required 1,000 DoS rows. The run therefore
   stopped before DoS-to-Benign and never produced the final ranking.

V3.1.1 selects one reproducible client set from the intersection of clients
eligible for both DDoS and DoS. Under the current fixed partition, that
intersection contains exactly eight clients:

`1, 8, 14, 15, 16, 17, 18, 19`

All three attacks are rerun with the exact same client IDs:

- DDoS to DoS
- DDoS to Benign
- DoS to Benign

Recommended command:

```cmd
python scripts\run_attack_pair_screen_v311.py --data-file "%USERPROFILE%\LFighter-research\data\processed\cic_iot_diad_2024_v2_1\arrays\behavioral_only.npz" --partition-file "%USERPROFILE%\LFighter-research\results\cic_iot_diad_federated_tuning_v27_alpha05_seed42\partitions\client_partitions.npz" --clean-seed-dir "%USERPROFILE%\LFighter-research\results\cic_iot_diad_federated_multiseed_v28_fixed_partition\seed_runs\seed_42" --output-dir "%USERPROFILE%\LFighter-research\results\cic_iot_diad_attack_pair_screen_v311_shared_clients_seed42" --malicious-client-fraction 0.40 --poison-fraction 1.00 --min-source-samples 1000 --attack-seed 42 --model-seed 42 --num-clients 20 --rounds 30 --participation-rate 1.0 --local-epochs 1 --batch-size 2048 --evaluation-batch-size 4096 --learning-rate 0.0003 --weight-decay 0.0001 --threads 6 --early-stopping-patience 8 --skip-existing
```

Use the new V3.1.1 output directory. Do not merge the incomplete V3.1 results
with the corrected results.
