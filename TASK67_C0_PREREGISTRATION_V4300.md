# Task 67 C0 — Publication Output Implementation Freeze V4.30.0

Task 67 is a deterministic presentation layer over the frozen Task 66 evidence. It does **not** rerun Task 65, rerun Task 66 inference, open the reserved NPZ, load models, retrain, recompute SHAP, or call an LLM.

The publication generator is frozen to produce **30 figure stems**, each in both PNG (300 dpi) and vector PDF, plus seven publication tables. The panel is intentionally comprehensive: it includes positive, null, negative, resolution-limited, and non-estimable findings rather than selecting only favorable outcomes.

The figures cover:
- all six primary metric families,
- both diagnostic and natural datasets,
- seed-level variability for core utility metrics,
- exact raw and Holm-adjusted p-value structure,
- Cohen dz where defined,
- positive-seed counts,
- bootstrap confidence intervals,
- round 5–8 descriptive trajectories,
- runtime and peak-RSS summaries.

Scientific claim boundaries are carried forward unchanged. In particular, with five final seeds the exact two-sided sign-flip test has minimum attainable p = 0.0625, so the publication layer must not convert consistent directionality into a conventional p<0.05 significance claim. Non-estimable attack-excess cells and worst-class-recall negative/null findings remain visible.

No best-seed or best-round selection is permitted.
