# Task 56 preregistration: XAI stability and faithfulness

## Purpose

Task 56 validates whether the frozen Task 55 SHAP explanations are reproducible and causally informative. It does not evaluate reconstruction recovery, reopen the detector, train models, or access reserved test arrays.

The parent evidence is `task55-c3-confirmatory-xai-frozen-v4173` at commit `8226e0ca76c9344113e2c5d53039119ea37d3eed`.

## Frozen analysis panel

The panel contains four seeds, three coalition families, coalition size ten, 28 unique checkpoint states, 36 logical comparisons, 69 behavioral features, a 128 row balanced background, and a 128 row balanced validation probe. Primary analysis uses the 16 true DDoS probe rows and the DDoS minus Benign SHAP margin.

## Validation domains

### Repeated run stability

The frozen Task 55 run with random seed 5517 is compared with three new expected gradients runs using seeds 56101, 56102, and 56103 for every physical checkpoint. The feature profile is mean absolute margin SHAP over true DDoS rows. Metrics are Spearman correlation, top ten Jaccard overlap, and normalized L1 drift.

This domain passes only if median Spearman is at least 0.90, the fifth percentile is at least 0.75, median top ten Jaccard is at least 0.67, and median normalized L1 drift is at most 0.25.

### Background sensitivity

Two alternate balanced backgrounds are selected with seeds 5601 and 5602. Each contains 16 training rows per class and is disjoint from both the frozen baseline background and the other alternate background. SHAP uses the frozen random seed 5517.

This domain passes only if median Spearman is at least 0.80, the fifth percentile is at least 0.60, and median top ten Jaccard is at least 0.54.

### Probe size sensitivity

Nested true DDoS probe subsets of sizes 8 and 12 are selected from the frozen 16 rows with seed 5603 and compared with the full 16 row profile. This domain passes only if median Spearman is at least 0.80, the fifth percentile is at least 0.60, and median top ten Jaccard is at least 0.54.

### Feature perturbation faithfulness

Each feature is replaced with its frozen baseline background mean. The impact is the mean absolute change in the DDoS minus Benign logit margin over true DDoS rows. Mean absolute SHAP importance is correlated with this observed impact over all 69 features.

Cumulative perturbations use the top 1, 5, 10, and 20 SHAP ranked features. They are compared with 256 frozen random feature panels for each size using seed 5604.

This domain passes only if median state Spearman is at least 0.50, at least 75 percent of states have positive Spearman, median top ten random percentile is at least 0.90, and at least 75 percent of states exceed the random median.

### Class specificity

For each true DDoS sample, its absolute margin SHAP profile is compared with a leave one out DDoS centroid and a pooled non DDoS centroid. The state metric is the median within class correlation minus between class correlation.

This domain passes only if the median state difference is at least 0.10 and at least 75 percent of states have positive differences.

### Cross seed consistency

Corresponding clean, suspicious, and reconstructed profiles are compared across all six seed pairs for seven physical conditions, producing 42 comparisons. This domain passes only if median Spearman is at least 0.50 and at least 75 percent of comparisons exceed 0.30.

### Computational cost

Wall time and process peak resident memory are recorded per new SHAP state evaluation and summarized. These measurements are machine dependent and therefore do not control the scientific pass decision.

## Decision policy

Repeated run stability and perturbation faithfulness are core domains. Background sensitivity, probe size sensitivity, class specificity, and cross seed consistency are robustness domains.

The overall result is PASS only when both core domains and all four robustness domains pass. It is PARTIAL when both core domains and at least two robustness domains pass. It is FAIL when either core domain fails or fewer than two robustness domains pass.

Task 57 may make a primary recovery claim only after PASS. A PARTIAL result permits exploratory recovery analysis with explicit limitations. A FAIL blocks a recovery claim.

All thresholds, seeds, subsets, random controls, and rules are frozen before Task 56 outcome analysis. Negative results are retained. No post hoc threshold changes are permitted.

## Safety boundary

Only `X_train`, `y_train`, `X_val`, and `y_val` may be materialized. `X_test_natural`, `y_test_natural`, `X_test_diagnostic`, and `y_test_diagnostic` remain forbidden. Task 56 performs no training and has no detector authority.
