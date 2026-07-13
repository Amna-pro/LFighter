# CIC IoT-DIAD Neural Multi-Seed V2.5

This stage resolves the one-seed uncertainty among:

- resmlp_ce
- resmlp_sqrt
- resmlp_focal

It runs five paired seeds using the existing V2.4 screening implementation and selects the final candidate using validation metrics only.

Outputs include:

- per-seed raw metrics
- mean, standard deviation, median, range, and 95% confidence interval
- validation-only final ranking
- paired Wilcoxon tests
- Friedman test
- per-class uncertainty
- runtime and reliability analysis
- CSV tables
- matching PNG and PDF figures
- saved checkpoints and seed-level artifacts

The natural EM correction remains exploratory and is not used for selection.

Recommended command:

```cmd
python scripts\run_neural_multiseed_v25.py --data-file "%USERPROFILE%\LFighter-research\data\processed\cic_iot_diad_2024_v2_1\arrays\behavioral_only.npz" --output-dir "%USERPROFILE%\LFighter-research\results\cic_iot_diad_neural_multiseed_v25_behavioral" --candidates resmlp_ce,resmlp_sqrt,resmlp_focal --seeds 42,123,2026,7,99 --threads 6 --epochs 30 --patience 6 --batch-size 2048 --skip-existing
```
