# CIC IoT-DIAD Targeted Label Flip V2.9

This stage validates the first targeted federated poisoning scenario:

- source class: DDoS
- target class: DoS
- fixed V2.7/V2.8 Dirichlet partition
- 20 percent malicious clients
- malicious clients selected reproducibly from eligible DDoS-bearing clients
- 50 percent of their DDoS rows statically relabeled as DoS
- model seed 42
- attack seed 42
- learning rate 0.0003
- one local epoch
- full participation

The attack is availability-aware but not outcome-selected. Clients are sampled
without replacement from those containing at least 1,000 DDoS rows.

Primary checkpoint selection uses validation macro F1 only. Test results are
computed after checkpoint selection. The last-round checkpoint is also reported
as a secondary diagnostic.

Recommended command:

```cmd
python scripts\run_targeted_label_flip_v29.py --data-file "%USERPROFILE%\LFighter-research\data\processed\cic_iot_diad_2024_v2_1\arrays\behavioral_only.npz" --partition-file "%USERPROFILE%\LFighter-research\results\cic_iot_diad_federated_tuning_v27_alpha05_seed42\partitions\client_partitions.npz" --clean-seed-dir "%USERPROFILE%\LFighter-research\results\cic_iot_diad_federated_multiseed_v28_fixed_partition\seed_runs\seed_42" --output-dir "%USERPROFILE%\LFighter-research\results\cic_iot_diad_targeted_label_flip_v29_ddos_to_dos_seed42" --source-class DDoS --target-class DoS --malicious-client-fraction 0.20 --poison-fraction 0.50 --min-source-samples 1000 --attack-seed 42 --model-seed 42 --num-clients 20 --rounds 30 --participation-rate 1.0 --local-epochs 1 --batch-size 2048 --evaluation-batch-size 4096 --learning-rate 0.0003 --weight-decay 0.0001 --threads 6 --early-stopping-patience 8
```

Outputs include:

- exact malicious-client list
- per-client poison counts
- poisoned local and global indices
- partition and poison-index hashes
- round-level validation macro F1 and attack success
- best-validation and last-round checkpoints
- natural and diagnostic metrics
- per-class metrics and confusion matrices
- paired clean-versus-attack deltas
- CSV and JSON source artifacts
- publication-quality PNG and PDF figures

This is a single-seed engineering and scientific validation, not a final paper result.
