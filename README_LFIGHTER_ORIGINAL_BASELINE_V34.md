# CIC IoT-DIAD Original LFighter Baseline V3.4

## Purpose

V3.4 evaluates the original LFighter defense on the frozen CIC IoT-DIAD 2024 benchmark.
It runs paired clean and strong DDoS-to-Benign experiments over model seeds:

`42, 123, 2026, 7, 99`

The strong attack is frozen from V3.3:

- 20 logical clients
- Full participation
- Malicious clients: `1, 7, 8, 10, 14, 15, 17, 18`
- 40% malicious-client fraction
- 100% of each malicious client's DDoS rows flipped to Benign
- Attack seed 42
- Exact V2.7 client partition and V3.3 poisoned-row selection

## Fidelity to the original LFighter implementation

The following steps reproduce the original multiclass LFighter rule:

1. Compute global-minus-local updates for the final classifier weight and bias.
2. Rank classes by summed final-layer update magnitude.
3. Select the two most active classes.
4. Cluster client updates for those two classes using KMeans with two clusters.
5. Compute the original cluster-dissimilarity expression.
6. Select the good cluster using the original comparison rule.
7. Equally average the admitted local models.

The added logs, security metrics, CSV files, and figures do not change the defense decision.

## Outputs

Each seed has one clean run and one strong-attack run. V3.4 saves:

- Round-level validation and attack metrics
- Client admission and rejection decisions
- Malicious-client rejection recall
- Benign-client retention and false rejection
- Detection precision and F1
- Identified class pair per round
- Class salience per round
- Cluster diagnostics per round
- Natural and diagnostic test metrics
- Per-class metrics and confusion matrices
- Best-validation and last-round checkpoints
- CSV source tables
- Publication-ready PNG and PDF figures
- Reproducibility metadata and hashes
- Paired comparisons with clean and attacked FedAvg

## Windows CMD installation

Run each command separately.

```cmd
cd /d "%USERPROFILE%\LFighter-research"
```

```cmd
call .venv\Scripts\activate.bat
```

Commit V3.3 before starting V3.4:

```cmd
git status
```

```cmd
git add scripts\run_targeted_label_flip_v292.py scripts\run_ddos_benign_multiseed_v33.py README_DDOS_BENIGN_MULTISEED_V33.md
```

```cmd
git commit -m "Add DDoS-to-Benign five-seed attack study V3.3"
```

```cmd
git push -u origin HEAD
```

Create the V3.4 branch:

```cmd
git switch -c feature/cic-iot-diad-lfighter-v34
```

Extract this ZIP to Downloads, then install it:

```cmd
call "%USERPROFILE%\Downloads\CIC_IoT_DIAD_LFighter_Original_Baseline_V34\install_lfighter_original_v34_windows.bat"
```

Check the installed files:

```cmd
python -m py_compile scripts\run_lfighter_original_v34.py scripts\run_lfighter_multiseed_v34.py scripts\run_targeted_label_flip_v292.py src\lfighter_original_v34.py
```

## Full five-seed run

```cmd
python scripts\run_lfighter_multiseed_v34.py --data-file "%USERPROFILE%\LFighter-research\data\processed\cic_iot_diad_2024_v2_1\arrays\behavioral_only.npz" --partition-file "%USERPROFILE%\LFighter-research\results\cic_iot_diad_federated_tuning_v27_alpha05_seed42\partitions\client_partitions.npz" --clean-multiseed-dir "%USERPROFILE%\LFighter-research\results\cic_iot_diad_federated_multiseed_v28_fixed_partition" --fedavg-attack-v33-dir "%USERPROFILE%\LFighter-research\results\cic_iot_diad_ddos_benign_multiseed_v33" --output-dir "%USERPROFILE%\LFighter-research\results\cic_iot_diad_lfighter_original_v34" --seeds 42,123,2026,7,99 --source-class DDoS --target-class Benign --malicious-clients 1,7,8,10,14,15,17,18 --poison-fraction 1.0 --min-source-samples 1000 --attack-seed 42 --num-clients 20 --rounds 30 --participation-rate 1.0 --local-epochs 1 --batch-size 2048 --evaluation-batch-size 4096 --learning-rate 0.0003 --weight-decay 0.0001 --threads 6 --early-stopping-patience 8 --kmeans-seed 0 --skip-existing
```

The controller runs 10 experiments, five clean and five strongly attacked. On the current CPU-only computer, allow several hours. Completed runs are reused, and incomplete runs are removed and restarted.

## After completion

Display the main results:

```cmd
type "%USERPROFILE%\LFighter-research\results\cic_iot_diad_lfighter_original_v34\tables\seed_level_defense_results.csv"
```

```cmd
type "%USERPROFILE%\LFighter-research\results\cic_iot_diad_lfighter_original_v34\tables\aggregate_mean_std_ci.csv"
```

```cmd
type "%USERPROFILE%\LFighter-research\results\cic_iot_diad_lfighter_original_v34\tables\paired_method_tests.csv"
```

Create the result ZIP:

```cmd
powershell -NoProfile -Command "$b='$env:USERPROFILE\LFighter-research\results\cic_iot_diad_lfighter_original_v34'; if(-not (Test-Path $b)){Write-Error 'V3.4 results folder not found'; exit 1}; Compress-Archive -Path \"$b\*\" -DestinationPath '$env:USERPROFILE\Downloads\CIC_IoT_DIAD_LFighter_Original_V34_Results.zip' -Force; Write-Host 'V3.4 results ZIP created'"
```

## Interpretation rule

Do not judge LFighter only by global accuracy. The primary questions are:

- Does LFighter reduce the attacked DDoS-to-Benign rate compared with attacked FedAvg?
- Does it preserve DDoS recall?
- How many malicious clients does it reject?
- How many benign clients does it incorrectly reject?
- Does it identify the DDoS and Benign class pair?
- What clean macro-F1 cost does filtering introduce?

V3.4 is the original-defense baseline. The proposed temporal, explanation-aware LFighter should only be implemented after these results are reviewed.
