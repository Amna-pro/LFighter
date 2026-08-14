# Task 58 publication figure preregistration

Task 58 converts the already frozen Task 56 validation evidence and Task 57 attribution recovery evidence into publication figures. It performs no training, no new SHAP evaluation, and no reserved test access.

Five figure families are frozen before generation. The validation dashboard uses every available Task 56 validation row. The recovery and prediction trajectory figures use all twelve family and seed triplets. The feature figure uses the frozen Task 57 top ten prespecified ranking. The individual case is selected mechanically as the triplet closest to the cohort median recovery fraction, with ties resolved by seed and family, followed by the true DDoS probe row closest to that suspicious state median margin.

Manual case replacement is prohibited. Rejected and round 8 oracle clean attribution states are unavailable and cannot be simulated or relabeled. The clean comparator remains the paired round 4 preattack clean reference. Each figure is exported as a 300 DPI PNG and a vector PDF. Captions must state the panel size and relevant limitation.
