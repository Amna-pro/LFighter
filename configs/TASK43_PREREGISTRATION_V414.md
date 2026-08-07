# Task 43 Preregistration, Byzantine Attacks and Robust-Aggregation Baselines

**Protocol version:** 4.14.0 (draft — following the 4.10/4.11/4.12/4.13 per-task major-slot convention confirmed from the codebase's own EXPERIMENT_VERSION history)
**Parent frozen evidence:** Task 42, tag `task42-c2-no-candidate-survived-v413`
**Status:** DRAFT — not yet preregistered or committed
**Reserved test access:** Prohibited until the final gate

## Purpose and Scope Boundary

Several classic Byzantine attack names (constant, sign-inversion, Gaussian) conceptually overlap with attacks Task 42 already tested (model-replacement, sign-flip, additive-noise) under different terminology. Re-testing them here under the "Byzantine" label would duplicate already-answered questions rather than extend them. Task 43 is scoped instead to two genuinely new contributions:

1. **Coordinated/omniscient attacks** — Task 42's malicious clients acted independently and identically; no coalition-level coordination was tested. Task 43 tests whether a coalition that *shares information* (e.g., splits a target perturbation across multiple clients so no single client's update looks anomalous, but the aggregate effect is significant) evades detection where independent attacks did not.
2. **Standard robust-aggregation baselines** — coordinate-wise median, trimmed mean, and Multi-Krum are implemented and evaluated for the first time in this project, directly answering the roadmap's long-deferred Task 51 question: does LFighter's detect-and-reconstruct approach add value beyond established robust-FL defenses, under the attack family (Byzantine) those defenses were specifically designed for?

Two classic Byzantine variants that do **not** meaningfully overlap with Task 42 are retained: **constant** (a fixed, non-honest-update-dependent vector, unlike Task 42's attacks which all transform an honest update) and **extreme-value** (per-coordinate extremization, a qualitatively different geometry than uniform scaling).

## Frozen research questions

1. Does LFighter's detect-and-reconstruct approach outperform standard robust-aggregation baselines (coordinate median, trimmed mean, Multi-Krum) under classic Byzantine attacks, using the same clean-utility and detection metrics applied throughout this project?
2. Does a coordinated/omniscient attacker — splitting effect across the coalition rather than each client acting independently — evade detection where Task 42's independent-attacker model did not?
3. Do update-independent attacks (constant, extreme-value) reveal different detector failure modes than the update-*dependent* attacks Task 42 already characterized (scaling, noise, replacement all transform an honest update; constant/extreme-value ignore it entirely)?

## Frozen hypotheses

- **H1, detection:** Mean malicious-client recall ≥ 0.50 and maximum benign-client FPR ≤ 0.05 — **carried over unchanged from Task 40/41C/42 for direct comparability**, not re-derived per attack family. (Task 42's failure analysis flagged this ceiling as worth reconsidering per-family in future work; Task 43 deliberately keeps it fixed to test whether that fixed bar is itself the limiting factor, one open question at a time rather than changing two things at once.)
- **H2, mitigation:** For each attack type, mean damage-removed fraction (macro-F1 convention, matching Tasks 40/42) ≥ 25% relative to plain FedAvg, improvement in ≥ 3 of 4 seeds.
- **H3, clean utility:** Mean clean validation macro-F1 loss ≤ 0.01, no seed losing more than 0.02.
- **H4, baseline superiority:** LFighter (D0, unmodified) achieves strictly higher damage-removed fraction than the best-performing robust-aggregation baseline on at least 3 of the 4 attack types, at equal or lower benign impact (clean macro-F1 loss).

## Frozen data policy

Identical to every prior stage: development/warmup/probe data only; `X_test_natural`, `y_test_natural`, `X_test_diagnostic`, `y_test_diagnostic` forbidden until a separately committed final-test gate.

## Frozen seeds and clients

Seeds `7, 99, 123, 2026`. The 20-client partition and malicious-client set `[1, 7, 8, 10, 14, 15, 17, 18]` remain unchanged.

## Attack panel

- **Constant Byzantine:** each malicious client submits a fixed vector, independent of local training entirely — magnitude set at the 95th-percentile clean update norm (2.37, from Task 42's C1b pooled distribution), direction drawn once (frozen per malicious client, shared across all rounds within a run) from a fixed random seed, **not** derived from that client's own honest update at all. This is the qualitative difference from every Task 42 attack, which all transformed an honest update.
- **Extreme-value Byzantine:** each malicious client submits a vector with every coordinate set to `±k · σ_coordinate`, where `σ_coordinate` is that parameter's own standard deviation across the clean warmup update matrices (already captured, Task 42 C1b infrastructure) and sign is chosen per-coordinate to maximize disruption (opposite the sign of the trusted-center coordinate). `k ∈ {2, 5, 10}`.
- **Coordinated/omniscient (the core new contribution):** the malicious coalition shares a single target perturbation and **splits it evenly** across all 8 malicious clients (each contributes `target/8` rather than each independently submitting a full-strength attack), so that any single client's update looks close to honest-scale, while the aggregate malicious contribution (after federated averaging) still reaches the same effective magnitude a single-attacker version would. Target perturbation reuses Task 42's collapse-to-Benign direction for continuity, at boost factors `{3, 5, 10}` (same justification as Task 42: `1/f ≈ 2.97`, sample-weighted).

## Defense candidates and comparison arms

- `D0_frozen_task40_lfighter` — unchanged mandatory reference.
- **New baseline implementations** (not LFighter variants — standalone aggregation rules, replacing weighted averaging entirely, no detection/reconstruction step):
  - `coordinate_median` — per-parameter coordinate-wise median across all 20 submitted updates.
  - `trimmed_mean` — per-parameter mean after discarding the top/bottom `f_trim` fraction of values (β = malicious-fraction estimate; use β=0.4 matching this coalition's true 8/20 client-count fraction, the conservative literature-standard choice when the attacker fraction is known or upper-bounded).
  - `multi_krum` — select the `m = 20 − 8 − 2 = 10` client updates with smallest sum-of-squared-distances to their `n − f − 2` nearest neighbors, average only those.
- `plain_fedavg` — undefended reference, as always.

No new D1/D2/D3-style candidates are preregistered for Task 43 — the emphasis is the baseline comparison, not another custom detector, per the scope boundary above.

## Candidate selection rule

Identical structure to Tasks 41C/42: reject on FPR/clean-utility gates where applicable to detection-based arms (D0); baselines are compared on damage-removed fraction and clean utility directly, since they have no "recall"/"FPR" concept (they never classify clients as malicious, they aggregate robustly regardless). H4 is evaluated as a direct pairwise comparison, not a survivor-selection process.

## Staged execution

- **C0 (this document):** Preregistration and integrity only.
- **C1:** Implement and unit-verify the three baseline aggregation rules against synthetic known-answer cases (e.g., coordinate median of a known small vector set) before any real federated run — this is new infrastructure and should be validated in isolation first, matching the caution applied to Task 42's parameter-order alignment.
- **C2:** Seed-7 development screen, all attack types x {D0, plain, coordinate_median, trimmed_mean, multi_krum}.
- **C3:** Confirmatory multiseed on the surviving/leading configuration(s).
- **C4:** Adaptive variant (reserved — coordinated attacker that also knows the specific baseline in use and crafts accordingly).
- **C5:** Final reserved-test gate.

## Statistics and reporting

Identical to Tasks 41C/42: exact paired comparisons, deterministic paired-bootstrap CIs, every seed and failed condition reported, full CSV/JSON/figure/manifest output.

## Prohibited post-hoc changes

Identical to every prior stage: no seed, coalition, partition, attack-strength, or threshold changes after freeze; no failed attack or seed dropped; Task 42 remains unchanged and is not reopened.

## Open decisions before this can be frozen

1. **Trimmed-mean's β parameter** assumes the true malicious fraction (8/20) is known when selecting how much to trim — this is a strong assumption real deployments wouldn't have. Worth deciding whether to also test a *misspecified* β (e.g., trimming only 20% when 40% is actually malicious) as a robustness check, or keep this out of scope for Task 43 and flag it as future work.
2. **Multi-Krum's `f` parameter** (assumed-Byzantine-count) has the same "assumes known coalition size" issue — same decision needed.
3. Confirm the version prefix (`4.14.0` follows the established per-task pattern; flag if a different convention is intended).
