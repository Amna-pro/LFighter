# BATR-FL Reviewer Results for Paper Integration

## New experiment
Four-arm reconstruction-mitigation ablation: 4 arms x 5 attacks x 5 primary seeds = 100 validation conditions. Primary endpoint: global round 8.

## Four arms
1. center_plus_residual
2. center_only
3. hard_rejection
4. down_weighting

## Main round-8 descriptive results

| Arm | Macro-F1 | Balanced Accuracy | Source-to-Target Rate | Benign FPR | Malicious Recall |\n|---|---:|---:|---:|---:|---:|\n| center_only | 0.368357 | 0.375855 | 0.078168 | 0.066667 | 0.995000 |\n| center_plus_residual | 0.364377 | 0.371054 | 0.069236 | 0.026667 | 0.995000 |\n| down_weighting | 0.379718 | 0.383419 | 0.129772 | 0.050000 | 0.995000 |\n| hard_rejection | 0.371491 | 0.374773 | 0.132254 | 0.046667 | 0.995000 |

## Statistical conclusion
Minimum attainable attack-specific two-sided exact sign-flip p-value: 0.0625. No raw pairwise comparison reached p < 0.05. No Holm-adjusted family reached significance. Seed-cluster analysis also found no significant pairwise superiority.

## Frozen claim boundary
No pairwise mitigation-arm superiority is established at alpha=0.05. Report the ablation as a descriptive trade-off: down-weighting has the highest mean round-8 validation macro-F1, while center-plus-residual has the lowest mean round-8 source-to-target rate and benign FPR.

## Safe manuscript wording
The four-arm ablation revealed a mitigation trade-off rather than statistically established superiority of a single strategy. Down-weighting achieved the highest descriptive mean round-8 validation macro-F1, whereas center-plus-residual achieved the lowest mean source-to-target rate and benign false-positive rate. Exact paired sign-flip tests across five primary seeds did not establish pairwise superiority at alpha = 0.05, and no comparison remained significant after Holm correction.

## Important restriction
Do not state that center-plus-residual significantly outperformed the other mitigation arms. Do not state that down-weighting significantly outperformed center-plus-residual. These differences are descriptive under the frozen five-seed protocol.
