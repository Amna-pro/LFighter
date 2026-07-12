# CIC IoT-DIAD Neural Reliability Screening V2.4

This stage is a one-seed architecture and loss screening experiment. It is not a final paper estimate.

## Why this stage exists

The V2.3 audit found:

- severe class-prior shift between training and the natural test
- class-conditional feature shift, especially for Recon, Web-Based, and DDoS
- very low natural-test support for BruteForce
- excessive minority-class false positives under fully balanced class weighting

## Candidates

- MLP with unweighted cross-entropy
- MLP with clipped square-root class weights
- residual MLP with unweighted cross-entropy
- residual MLP with clipped square-root class weights
- residual MLP with focal loss
- 1D CNN with unweighted cross-entropy
- 1D CNN with clipped square-root class weights
- 1D CNN with focal loss

All models preserve an explicit final classifier layer for later LFighter gradient analysis.

## Methodological safeguards

- model selection uses validation macro F1 only
- test labels are never used to select a model
- temperature scaling is fitted on validation data
- natural-test EM prior adjustment uses unlabeled model probabilities only
- actual natural-test priors are saved only for retrospective evaluation
- EM results are exploratory because covariate shift was also detected

## Outputs

- CSV tables
- PNG figures
- PDF figures
- checkpoints
- validation-only candidate ranking
- per-class metrics
- raw and EM-adjusted natural-test results
- diagnostic-test results
- calibration and runtime analysis

## Recommended Windows CMD command

```cmd
python scripts\screen_neural_iot_diad_v24.py --data-file "%USERPROFILE%\LFighter-research\data\processed\cic_iot_diad_2024_v2_1\arrays\behavioral_only.npz" --output-dir "%USERPROFILE%\LFighter-research\results\cic_iot_diad_neural_screening_v24_seed42" --candidates mlp_ce,mlp_sqrt,resmlp_ce,resmlp_sqrt,resmlp_focal,cnn1d_ce,cnn1d_sqrt,cnn1d_focal --seed 42 --threads 6 --epochs 30 --patience 6 --batch-size 2048
```
