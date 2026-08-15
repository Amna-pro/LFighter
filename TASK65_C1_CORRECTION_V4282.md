# Task 65 C1 Implementation Correction V4.28.2

V4.28.1 passed a static preflight but was **not committed, tagged, trained, or used to access final test arrays**.

A deeper source-level verification before freeze found two execution-wiring defects:

1. the plain runner saves `continuation_last_round_model.pt`, while the frozen defense runner saves `v320b1_last_round_model.pt`; V4.28.1 only captured the first name;
2. the defense runner appends `calibration/trusted_update_reconstruction_profiles.pt` to the path supplied via `--reconstruction-calibration-dir`; V4.28.1 supplied the nested `calibration` folder instead of the reconstruction builder output root.

V4.28.2 corrects both points before any Task 65 training or reserved-test access. It also strengthens preflight checks and deterministic incomplete-stage restart handling.

No scientific outcome, final metric, or reserved test array was observed during this correction.

## Windows handoff wrapper correction before freeze

The first V4.28.2 setup wrapper stopped at its initial local-tag check because Windows CMD consumed the caret in annotated-tag syntax (`^{commit}`) inside `FOR /F`. The stop occurred before package installation, training, or final-test access. The wrapper and the future one-shot batch now use `git rev-list -n 1 <tag>` instead, avoiding CMD caret parsing.
