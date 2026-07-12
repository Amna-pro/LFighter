# CIC IoT-DIAD Shift Audit V2.3

This stage diagnoses why model performance drops under the leakage-safe natural test.

It produces:

- class support and prior-shift tables
- Jensen-Shannon and total-variation divergence
- per-class, per-feature KS, Wasserstein, SMD, and PSI shift metrics
- top shifted features
- minority-class support reliability
- PCA domain projection
- CSV tables
- matching PNG and PDF figures

The script uses prepared NPZ arrays and does not modify the raw dataset.

Example command:

```cmd
python scripts\audit_iot_diad_shift_v23.py --data-file "%USERPROFILE%\LFighter-research\data\processed\cic_iot_diad_2024_v2_1\arrays\behavioral_only.npz" --feature-names "%USERPROFILE%\LFighter-research\data\processed\cic_iot_diad_2024_v2_1\preprocessors\behavioral_only\selected_features.csv" --screening-per-class "%USERPROFILE%\LFighter-research\results\cic_iot_diad_model_screening_seed42\tables\per_class_metrics.csv" --output-dir "%USERPROFILE%\LFighter-research\results\cic_iot_diad_shift_audit_v23" --sample-per-class 5000 --pca-sample-per-class 1000 --top-k 15 --seed 42
```
