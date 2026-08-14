# Task 58 figure captions

**Figure F1. XAI validation dashboard.** Complete Task 56 evidence across 28 physical states, including repeated run stability, alternate background sensitivity, probe size sensitivity, feature perturbation faithfulness, class specificity, and cross seed consistency. The dashed drift threshold is the frozen repeated run q95 used by Task 57.

**Figure F2. Attribution recovery across every paired triplet.** Normalized L1 distance between each suspicious or reconstructed attribution profile and its paired round 4 preattack clean reference. All twelve combinations of four seeds and three coalition families are shown. Lower is closer to the paired clean reference.

**Figure F3. Prediction behavior across states.** DDoS minus Benign logit margin and correct DDoS prediction rate for all twelve triplets. Thin curves are individual triplets and the black curve is the cohort median.

**Figure F4. Frozen top ten poisoning associated features.** Median absolute attack shift and residual reconstructed shift for the ten features selected by the prespecified Task 57 ranking. Percent annotations report feature level recovery. No feature was selected during figure generation.

**Figure F5. Deterministic representative case.** Per feature SHAP contributions for the true DDoS probe row selected by the frozen mechanical rule. The triplet is closest to the cohort median recovery fraction and the row is closest to the suspicious state median DDoS minus Benign margin. Green supports the source target margin and red opposes it.

**Scope limitation.** Rejected and round 8 oracle clean attribution states were unavailable. No substitute states were created. Clean refers to the paired round 4 preattack reference, so recovery means movement toward that reference rather than proof of oracle clean restoration.
