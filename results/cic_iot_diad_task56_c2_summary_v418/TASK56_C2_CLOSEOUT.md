# Task 56 C2 XAI validation closeout

Overall preregistered result: **PASS**

No model training or reserved test-array access occurred. All negative results are retained.

## Domain decisions

- repeated_run_stability: **PASS** (4/4 gates)
- background_sensitivity: **PASS** (3/3 gates)
- probe_size_sensitivity: **PASS** (3/3 gates)
- feature_perturbation_faithfulness: **PASS** (4/4 gates)
- class_specificity: **PASS** (2/2 gates)
- cross_seed_consistency: **PASS** (2/2 gates)

## Preregistered metrics

| Domain | Metric | Value | Threshold | Pass |
|---|---|---:|---:|:---:|
| repeated_run_stability | median_spearman | 0.9960 | >= 0.9000 | Yes |
| repeated_run_stability | fifth_percentile_spearman | 0.9925 | >= 0.7500 | Yes |
| repeated_run_stability | median_top10_jaccard | 0.8182 | >= 0.6700 | Yes |
| repeated_run_stability | median_normalized_l1_drift | 0.0906 | <= 0.2500 | Yes |
| background_sensitivity | median_spearman | 0.9817 | >= 0.8000 | Yes |
| background_sensitivity | fifth_percentile_spearman | 0.9694 | >= 0.6000 | Yes |
| background_sensitivity | median_top10_jaccard | 1.0000 | >= 0.5400 | Yes |
| probe_size_sensitivity | median_spearman | 0.9826 | >= 0.8000 | Yes |
| probe_size_sensitivity | fifth_percentile_spearman | 0.9730 | >= 0.6000 | Yes |
| probe_size_sensitivity | median_top10_jaccard | 0.8182 | >= 0.5400 | Yes |
| feature_perturbation_faithfulness | median_state_spearman | 0.8961 | >= 0.5000 | Yes |
| feature_perturbation_faithfulness | fraction_states_positive_spearman | 1.0000 | >= 0.7500 | Yes |
| feature_perturbation_faithfulness | median_top10_random_percentile | 1.0000 | >= 0.9000 | Yes |
| feature_perturbation_faithfulness | fraction_states_top10_above_random_median | 1.0000 | >= 0.7500 | Yes |
| class_specificity | median_state_difference | 0.1706 | >= 0.1000 | Yes |
| class_specificity | fraction_states_positive | 1.0000 | >= 0.7500 | Yes |
| cross_seed_consistency | median_spearman | 0.9592 | >= 0.5000 | Yes |
| cross_seed_consistency | fraction_comparisons_above_0p3 | 1.0000 | >= 0.7500 | Yes |

## Task 57 gate

- Primary recovery claim allowed: **True**
- Exploratory-only status: **False**
- Recovery claim blocked: **False**
