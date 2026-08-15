# Task 66 C0 — Final Statistics Implementation Freeze V4.29.0

Task 65 final outcomes are already frozen under `task65-c2-final-results-audited-v4283`. Task 66 therefore may not alter the preregistered statistical method.

Confirmatory inference uses **round 8 only**. The pairing unit is the same final seed, attack, dataset, and endpoint. All five frozen final seeds are required.

The frozen inference engine implements:

- 20,000-replicate paired percentile bootstrap on seed-level contrasts, RNG seed 650428, 95% confidence intervals.
- Exact two-sided paired sign-flip inference over all 32 sign patterns for five seeds.
- Minimum attainable two-sided p-value 0.0625, so conventional p<0.05 rejection is resolution-impossible with five seeds; p-values are reported rather than hidden.
- Holm family-wise correction across the five attacks separately inside each dataset × primary-metric family.
- Paired mean, paired median, and Cohen dz when defined.
- No failed seed or attack may be dropped. Non-finite seed-level values make the corresponding confirmatory estimate/family not estimable rather than silently reducing n.
- Round 5–8 trajectories are descriptive secondary outputs only; they are never used for endpoint selection.

For utility metrics, the seed-level contrast is defended minus plain. `attack_excess_removed_fraction` is analyzed directly against zero. Detector recall and FPR have no plain-arm counterpart in the frozen Task65 paired table, so their confirmatory contrasts are expressed against the already-frozen practical thresholds: recall − 0.50 and 0.05 − FPR. Their observed levels are reported alongside those margins. These detector metrics are training-side and dataset-invariant; the same frozen values appear in both dataset families because the Task63 Holm specification is dataset-stratified.

The Task63 clean-utility limits are not recomputed on reserved test data because the frozen Task65 comparison panel contains no clean-defense arm. They remain carried-forward development safety constraints.

Task 66 reads only frozen Task65 CSV/JSON evidence. It does not open the NPZ, load models, train, run SHAP, call an LLM, or rerun Task65.
