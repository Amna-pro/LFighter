# CIC IoT-DIAD Neural Multi-Seed V2.5.1

This patch fixes unequal-candidate aggregation when a reused seed folder contains
extra candidates from an earlier screening run.

Changes:
- filters all aggregated tables to the candidates explicitly requested
- verifies that every requested candidate has every requested validation seed
- writes candidate_seed_coverage.csv
- keeps the original trained seed folders and checkpoints
- reruns aggregation and figures without retraining when --skip-existing is used
