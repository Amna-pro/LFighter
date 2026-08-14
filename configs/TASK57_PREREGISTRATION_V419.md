# Task 57 preregistration: attribution recovery after reconstruction

## Scientific question

Task 57 tests whether the trusted reconstruction state moves the validated SHAP attribution profile toward the paired seed specific preattack clean reference. This is a post detection forensic analysis. It does not train a model, modify the detector, select a new defense, or open reserved test arrays.

## Upstream authorization

The analysis is permitted only because Task 56 finished with PASS, all six validation domains passed, all 37 integrity checks passed, and the Task 56 decision explicitly allowed Task 57 primary claims.

## Frozen panel

The panel contains four seeds, three preregistered coalition families, coalition size 10, and twelve paired family seed triplets. Each triplet contains the clean reference, suspicious, and reconstructed attribution states. The primary rows are the same sixteen true DDoS validation rows used by Tasks 55 and 56. The primary tensor is `source_minus_target_margin_shap`, where the source is DDoS class 2 and the target is Benign class 0.

## Five state roadmap gap

The research roadmap requested clean, attacked, rejected, reconstructed, and oracle clean states. The frozen Task 55 and Task 56 evidence contains only clean reference, suspicious, and reconstructed attributions. It contains no exact paired rejected attribution artifact and no round 8 counterfactual oracle clean attribution artifact.

Task 57 therefore does not rename, infer, or synthesize either missing state. The primary claim is limited to movement toward the frozen round 4 preattack clean reference. It must not be described as recovery of an unobserved oracle clean model. This limitation is retained in every decision and paper handoff.

## Primary profile and distances

For each physical state, the primary feature profile is the mean absolute margin SHAP value across the sixteen true DDoS rows. For each family seed triplet:

1. Attack distance is normalized L1 distance from suspicious to clean.
2. Reconstructed distance is normalized L1 distance from reconstructed to clean.
3. Distance reduction equals attack distance minus reconstructed distance.
4. Recovery fraction equals distance reduction divided by attack distance.

The denominator is the L1 norm of the clean primary profile plus `1e-12`.

Secondary metrics are change in Spearman similarity, change in top 10 feature Jaccard similarity, signed profile distance reduction, mean DDoS minus Benign logit margin, and DDoS prediction rate.

## Measurement noise gate

Task 56 produced 84 repeated run comparisons. Their preregistered normalized L1 drift has a frozen 95th percentile of `0.12184864742664225`. Task 57 first requires a measurable attack signal. The median suspicious to clean distance must exceed this value and at least eight of twelve triplets must individually exceed it. If this gate fails, the recovery result is INCONCLUSIVE rather than a recovery failure.

## Primary recovery gate

PASS requires all of the following:

1. The attack signal gate passes.
2. At least nine of twelve triplets have positive distance reduction.
3. Every one of the four seed averaged reductions is positive.
4. Median recovery fraction is at least 0.25.
5. The exact one sided triplet sign flip test is at most 0.05.
6. The hierarchical bootstrap 95 percent lower bound for median recovery fraction is above zero.

The hierarchical bootstrap resamples seeds and then families within seeds for 20,000 replicates using seed 5719. A seed blocked sign flip test over all sixteen patterns is reported as a dependence sensitivity analysis. Its smallest possible one sided p value is 0.0625, so it is not used as an impossible 0.05 pass gate.

If the attack signal gate passes and the median recovery is positive but one or more primary gates fail, the result is PARTIAL. If the attack signal gate passes and median recovery is nonpositive, the result is FAIL. Negative and boundary results are retained.

## Poisoning associated features

All 69 features remain in the result table. A descriptive poisoning associated feature must shift in the same direction in at least nine of twelve triplets and its median absolute attack shift must exceed its feature specific Task 56 repeat noise reference. The fixed paper list contains at most ten features, ranked by median absolute attack displacement with feature name as the deterministic tie breaker.

This is a prespecified descriptive ranking, not confirmatory feature discovery. No feature may be chosen from a visually attractive case study.

## Dependence and uncertainty

The twelve family seed triplets are the frozen condition panel, but the three families within a seed share the same clean checkpoint. Results therefore report both triplet level evidence and seed clustered sensitivity. Claims generalize to this frozen panel, not automatically to all poisoning attacks or unseen data.

## Prohibited actions

No training, new SHAP evaluation, detector change, method reopening, reserved test materialization, post hoc threshold change, missing state substitution, or Task 58 publication figure selection is permitted in Task 57 C0 or C1.
