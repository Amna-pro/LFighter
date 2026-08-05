# Task 42 Preregistration, Model-Update Poisoning Extension

**Protocol version:** 4.13.0 (draft — adjust version string to match your repo's convention before freezing)
**Parent frozen evidence:** Task 40, tag `task40-v3.20b1-freeze`; Task 41C, tag `task41c-c3-h2-correction-v1`
**Status:** DRAFT — not yet preregistered or committed
**Reserved test access:** Prohibited until the final gate

## Purpose

Tasks 39–41 tested attacks that corrupt a client's *local training data* (label flips, untargeted poisoning, backdoor triggers) and let ordinary local training produce the malicious update. Task 42 tests a different attacker capability: **direct manipulation of the submitted update itself**, independent of what the local data implies. This matters because LFighter's detector operates on prediction-transition signatures derived from local training behavior — an attacker who bypasses local training entirely and crafts the update directly may evade that signal in ways label/trigger poisoning cannot.

## Frozen research questions

1. Does the frozen Task 40 detector (`D0`), calibrated on label/prediction-transition drift, detect update-space attacks that never touch local training data?
2. Can a norm- or direction-based update-space signal (independent of prediction transitions) detect these attacks while preserving benign-client precision?
3. Do standard robust-aggregation baselines (coordinate median, trimmed mean, Multi-Krum) outperform LFighter's detection-plus-reconstruction approach on this specific attack family, and does that comparison hold across attack strengths?
4. Does the selected extension (if any) remain effective against an adaptive attacker who constrains its update to evade norm/direction-based detection?

## Frozen hypotheses

- **H1, detection:** Mean malicious-client recall ≥ 0.50 and maximum benign-client FPR ≤ 0.05, matching the Task 41C bar for consistency across attack-breadth stages.
- **H2, mitigation:** For each confirmatory attack type, mean primary-metric damage-removed fraction (matching Task 40's `mean_damage_removed_fraction` convention) ≥ 25% relative to plain FedAvg, with improvement in at least 3 of 4 seeds.
- **H3, clean utility:** Mean clean validation macro-F1 loss ≤ 0.01; no seed loses more than 0.02.
- **H4, adaptive robustness:** Under a norm/direction-constrained adaptive attacker, damage-removed fraction remains positive in at least 3 of 4 seeds, mean malicious recall ≥ 0.35, maximum benign FPR ≤ 0.05.

## Frozen data policy

Development uses only training, validation, clean warmup, independent-anchor, and balanced server-probe data. `X_test_natural`, `y_test_natural`, `X_test_diagnostic`, `y_test_diagnostic` are forbidden until a separately committed final-test gate — identical to every prior stage.

## Frozen seeds and clients

Seeds `7, 99, 123, 2026` (development-only seed 42 excluded from confirmatory claims, matching prior practice). The 20-client partition and malicious-client set `[1, 7, 8, 10, 14, 15, 17, 18]` remain unchanged.

## Attack panel (draft — confirm strengths before freezing)

- **Sign flipping:** submitted update replaced by its negation (`-update`).
- **Update scaling:** submitted update scaled by a fixed factor (candidates: 5×, 10×, 20× — pick after a quick magnitude sanity check against clean update norms).
- **Additive Gaussian perturbation:** clean update plus zero-mean Gaussian noise at fixed relative magnitude (candidates: noise std = 0.5×, 1×, 2× the clean-update norm).
- **Model replacement:** submitted update crafted so the aggregated model moves toward a fixed attacker-chosen target state (classic Bhagoji-style replacement, scaled to survive averaging).
- **Constrained/norm-bounded update (reserved for H4 adaptive testing only):** any of the above, clipped to stay within the empirical clean-update norm range observed during warmup — this is the adaptive/evasive variant, not a separate confirmatory arm.

## Defense candidates

- `D0_frozen_task40_lfighter` — unchanged mandatory reference (unmodified from Task 40/41).
- `D1_update_norm_deviation` — client score from deviation of current update norm against that client's own historical warmup norm baseline. **Calibration confirmed free**: `results\cic_iot_diad_frozen_reconstruction_v3123_multiseed\seed_<N>\calibration\tables\trusted_warmup_replay_local_updates.csv` already contains per-client, per-warmup-round `update_norm` (4 rounds × 20 clients per seed) — a per-client mean/std baseline is a direct reprocessing of this existing table, no new training required.
- `D2_update_direction_deviation` — client score from cosine deviation between the current update and that client's historical residual direction. **Calibration confirmed free**: `client_residual_profiles[client_id]` in `trusted_update_reconstruction_profiles.pt` already stores the full per-parameter historical residual tensor (matching the model's `state_dict()` structure) for every client — flatten and normalize for the reference direction, no new calibration run required.
- `D3_equal_rank_fusion` — equal-weight rank fusion of D0, D1, D2 (retained only if both D1 and D2 individually clear the hard gates in the C2-equivalent screen; otherwise dropped, per the same rule that excluded D1/D3 from Task 41C's confirmatory stage).

## Comparison arms (robust-aggregation baselines, per roadmap Task 51)

Plain FedAvg, coordinate-wise median, trimmed mean, Multi-Krum. These are standalone comparison arms, not LFighter candidates — evaluated under identical seeds/attacks/pairing to answer research question 3 directly.

## Candidate selection rule (identical structure to Task 41C, for methodological consistency)

1. Reject any candidate with benign FPR above 0.05.
2. Reject any candidate with mean clean macro-F1 loss above 0.01, or any seed-7 round loss above 0.02.
3. Among survivors, maximize malicious-client recall across all preregistered attack-type × seed-7 development conditions.
4. Break ties by damage-removed fraction, then runtime overhead.
5. If every candidate fails, record failure. No new candidate may be invented inside the same preregistered task.

## Staged execution (mirrors Task 41C's C0–C5 structure)

- **D0 (this document):** Preregistration and integrity only.
- **D1 (calibration, confirmed zero-training):** Reprocess `trusted_warmup_replay_local_updates.csv` (per-client norm baseline for `D1_update_norm_deviation`) and `trusted_update_reconstruction_profiles.pt` (per-client residual direction for `D2_update_direction_deviation`) for all four seeds. Pure reprocessing of existing Task 40 artifacts — no attack construction, no model training. Output: a candidate-threshold table structurally identical to Task 41C's `task41c1c_candidate_thresholds.csv`.
- **D2:** Seed-7 development screening of every preregistered candidate against all four attack types.
- **D3:** Confirmatory multiseed evaluation on the primary attack panel.
- **D4:** Adaptive multiseed evaluation on the norm-constrained variant.
- **D5:** One final reserved-test evaluation after every choice is frozen and committed.

## Statistics and reporting

Identical to Task 41C: exact paired comparisons by attack type, strength, seed, partition, warmup, and round; deterministic paired-bootstrap 95% confidence intervals; every seed and every failed condition reported; CSV/JSON tables plus PNG/PDF figures; SHA-256 source/evidence manifest at every stage.

## Prohibited post-hoc changes

Identical to Task 41C: no seed, malicious-client, partition, attack-strength, candidate-weight, or threshold changes after freeze; no failed attack or seed dropped; no defense added after confirmatory outcomes are observed.

## Open decisions before this can be frozen (resolve, then commit)

1. Confirm exact attack-strength grid (scaling factors, noise magnitudes) — currently placeholders above.
2. ~~Confirm whether D1/D2's calibration can reuse Task 40's warmup artifacts unmodified~~ — **RESOLVED**: both are free reprocessing of existing artifacts (`trusted_warmup_replay_local_updates.csv` for D1, `trusted_update_reconstruction_profiles.pt` for D2). No new calibration training required. Confirmed against `seed_7`'s bundle on 2026-08-05; verify the same column/key structure holds for seeds 99, 123, 2026 before writing the D1-stage script.
3. Decide the version-string convention (`4.13.0` used here is a guess based on the 4.10/4.11/4.12 pattern for Tasks 41A/B/C — confirm against your actual numbering scheme).
