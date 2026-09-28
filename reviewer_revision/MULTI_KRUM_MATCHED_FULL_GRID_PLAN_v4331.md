# Multi-Krum Matched Full Grid Plan v4.33.1

Scale the verified v4.33.0 Multi-Krum comparator to the full frozen reviewer grid without changing the implementation.

Grid: five final model seeds x five attacks = 25 attacked conditions, global rounds 5 through 8. The audited all_to_one_benign / seed 1379954285 preflight is reused; the remaining 24 conditions are newly executed.

Frozen Multi-Krum settings: n=20 submitted updates, assumed Byzantine count f=8, selection count m=n-f-2=10. The exact historical implementation in scripts/verify_task43_c1_baselines_v414.py is reused. Selected updates are averaged without sample-count weighting. There is no BATR detector, reconstruction, or attack-specific tuning.

Every condition reuses the corresponding frozen clean round-4 warmup checkpoint, the fixed partition, the exact recovered Task-65 poison plan, the same ResMLP architecture, optimizer, local hyperparameters, RNG rule, and four-round continuation.

This stage loads train and validation arrays only. Final diagnostic and natural test arrays are not materialized. Validation outcomes and selected-client identities are retained exactly as observed; no condition is dropped and no parameter is changed after observing outcomes. No final-test evaluation or formal cross-method inferential analysis is performed in this stage.
