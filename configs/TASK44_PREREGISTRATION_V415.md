# Task 44 Preregistration, Poisoning Strength Variation

**Protocol version:** 4.15.0 (next slot after Task 43's 4.14)
**Parent frozen evidence:** Task 40 (`task40-v3.20b1-freeze`), Task 43 (`task43-c2-h4-pass-h1-narrow-fail-v414`)
**Status:** DRAFT — not yet preregistered or committed
**Reserved test access:** Prohibited until the final gate

## Purpose

Every attack family tested through Task 43 used a single fixed intensity (full poison fraction, full boost, full coalition size). This leaves an open question the roadmap explicitly requires answering (Task 44): is D0's behavior at that one fixed point representative, or does it sit at an unusual extreme? In particular, both Task 42 and Task 43 found D0's false-positive rate exceeding the frozen 0.05 ceiling specifically under their *strongest* tested conditions — Task 44 tests directly whether that FPR cost scales down at weaker, more realistic attack intensities, turning a single-point result into an actual response curve.

## Scope decision

Rather than re-testing strength variation across every attack family (which would substantially duplicate Tasks 40–43), Task 44 is scoped to the **frozen Task 40 untargeted label-poisoning panel** specifically. This is the strongest choice available for three reasons: (1) it already has a complete, qualified, 4-seed fixed-strength baseline to extend rather than build from scratch (`task40-v3.20b1-freeze`), (2) untargeted poisoning is the attack family with the cleanest, most literature-standard notion of "poison fraction" as a strength parameter, and (3) it keeps this task's scope achievable rather than combinatorially exploding across every attack type tested so far.

## Frozen research questions

1. How does D0's malicious-client recall change as poison fraction decreases from 1.0 (Task 40's tested value) toward more marginal levels?
2. How does D0's benign-client FPR change across the same range — does it fall below the 0.05 ceiling at any tested reduced strength?
3. Is there an identifiable minimum poison fraction below which the attack becomes undetectable (recall drops meaningfully) but also stops being harmful (damage-removed becomes moot because the attack itself causes negligible damage)? I.e., does a "dead zone" exist where the attack is too weak to matter either way?

## Frozen hypotheses

- **H1, monotonicity:** Malicious recall is non-increasing as poison fraction decreases (weaker attacks are at least as hard to detect as stronger ones, never harder).
- **H2, FPR relief:** Benign FPR at poison fraction 0.25 is strictly lower than at poison fraction 1.00, for at least 3 of the 5 previously-qualified untargeted attacks.
- **H3, damage floor:** At poison fraction 0.25, mean plain-FedAvg damage (relative to clean) is still positive and non-trivial (>5% macro-F1 loss) for at least 3 of the 5 attacks — confirming weaker attacks are still worth defending against, not testing a strength so low it is scientifically uninteresting.

## Frozen data policy, seeds, and clients

Identical to every prior stage. Confirmatory seeds `7, 99, 123, 2026`. Same 20-client partition. Same malicious coalition `[1, 7, 8, 10, 14, 15, 17, 18]` — coalition *size* is not varied here (that is Task 45's explicit scope; varying both strength and coalition size simultaneously would test two things at once).

## Frozen strength grid

Poison fraction `{0.25, 0.50, 0.75, 1.00}`, applied identically to all 5 attacks already qualified in Task 40 (`all_to_one_benign`, `cyclic_shift`, `multiclass_partial_cycle`, `pairwise_swap`, `random_flip`). The 1.00 point reuses Task 40's existing frozen results directly rather than rerunning — only the three new fractions (0.25/0.50/0.75) require new execution.

## Candidate

D0 only (frozen, unmodified) — this task characterizes the existing frozen detector's response curve, it does not screen new candidate variants.

## Selection rule

Not a candidate-selection task. Reports the full recall/FPR/damage curve across all 4 strength points × 5 attacks, and evaluates H1–H3 as stated. No pass/fail gate analogous to Tasks 41C–43's lexicographic rule — the deliverable is the curve itself plus the three hypothesis evaluations.

## Staged execution

- **C0 (this document):** Preregistration and integrity only.
- **C1:** Verify the existing 1.00-fraction Task 40 results are directly reusable without rerun (partition/seed/checkpoint match check) — avoid regenerating anything already frozen.
- **C2:** Seed-7 development run at 0.25/0.50/0.75 fractions, all 5 attacks, D0 only (15 new attack-branch combinations, reusing Task 40's plain FedAvg and D0 grid architecture).
- **C3:** Confirmatory multiseed at the same 3 new fractions.
- **C4:** Final reserved-test gate (only if warranted based on C3 findings).

## Statistics and reporting

Identical conventions to every prior stage: exact paired comparisons against Task 40's frozen clean/attacked references, full CSV/JSON/figure/manifest output, all four fractions reported regardless of outcome.

## Prohibited post-hoc changes

Identical to every prior stage. Task 40's frozen 1.00-fraction results are not rerun, retuned, or altered — only referenced as the anchor point for the new curve.

## Open decision before this can be frozen

Confirm the version prefix (`4.15.0`, following the established per-task pattern) and confirm the scope decision (Task 40's untargeted panel only, not re-testing Tasks 41–43's attack families at varying strength) is the right call, or whether a different attack family should anchor this instead.
