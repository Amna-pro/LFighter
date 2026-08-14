# Task 57 C2 attribution recovery closeout

Scientific result: **PASS**

The frozen analysis reused 28 validated attribution states and evaluated 12 clean, suspicious, reconstructed triplets across four seeds and three coalition families. It performed no training and no new SHAP evaluations.

The median suspicious to clean distance was `0.738776` against the Task 56 repeat noise q95 of `0.121849`. `12` of 12 attack distances exceeded that noise reference.

The median recovery fraction was `0.397277` with a hierarchical 95 percent interval of `[0.367060, 0.609167]`. `12` of 12 triplets and `4` of 4 seed means had positive distance reduction. The one sided exact triplet sign flip p value was `0.000244`. The resolution limited seed blocked p value was `0.062500`.

The prespecified feature rule identified `39` poisoning associated features, of which `10` are retained in the fixed report table.

Rejected and oracle clean attributions remain unavailable. The supported claim is movement toward the frozen preattack clean reference, not recovery of an unobserved oracle clean model. Publication figures and case studies remain deferred to Task 58.
