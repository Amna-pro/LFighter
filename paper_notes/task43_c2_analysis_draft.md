# Task 43 C2 Analysis: Byzantine Attacks and Robust-Aggregation Baselines

*Draft for the Results section. All figures trace to `task43c2_candidate_selection.csv`, `task43c2_h4_baseline_comparison.csv`, and `task43c2_selection_decision.json` (tag pending — see commit checklist below).*

## 1. Overview

Task 43 tested two things Task 42 could not: (a) whether D0's per-client behavioral detection can be evaded by a *coordinated* coalition that splits one target perturbation across all 8 malicious clients so no individual contribution looks large, and (b) how D0 compares against standard robust-aggregation baselines (coordinate-median, trimmed-mean, Multi-Krum) — the long-deferred Roadmap Task 51 comparison, implemented and unit-verified for the first time in this project (tag `task43-c1-baselines-verified-v414`).

## 2. Result — read precisely, not flattened

| Arm | Mean clean F1 loss | Survives H3 | Mean damage-removed | Max benign FPR | Survives H1 | Mean recall |
|---|---|---|---|---|---|---|
| D0 (frozen) | 0.000289 | **Yes** | **0.962** | 0.167 | **No** | **1.000** |
| coordinate_median | 0.029 | No | 0.448 | n/a | n/a | n/a |
| trimmed_mean | 0.016 | No | **−0.409** | n/a | n/a | n/a |
| multi_krum | 0.052 | No | 0.168 | n/a | n/a | n/a |

**H4 (baseline superiority): passes 3 of 3 attack types**, decisively — D0 strictly beats the best available baseline on every attack type tested, at equal-or-lower clean-utility cost every time (`task43c2_h4_baseline_comparison.csv`).

**Formally, D0 does not survive H1's frozen FPR gate** (max 0.167 against the 0.05 ceiling carried over from Task 40). Per the preregistered rule, applied exactly as written and consistent with Tasks 41C and 42, this closes the C2 stage without proceeding to C3 confirmatory multiseed.

## 3. Why this is a materially different result from Task 42, not a repeat of it

Task 42 closed with every candidate weak on detection, mitigation, or both — a clean, unambiguous negative. Task 43's result does not have that character. D0 achieves the strongest mitigation figure recorded anywhere in this project (96.2% mean damage-removed), never misses a malicious client (recall = 1.0, including the minimum across every condition), and costs essentially nothing in clean utility (0.03% mean loss). Critically, the FPR violation is not general: **`constant` and `extreme_value` attacks produce FPR = 0.000 throughout** — D0 has a perfect security profile against two of the three attack types tested. The violation is confined specifically to `coordinated_split`, and grows with boost strength (0.083 at 3×/5×, rising to 0.167 at higher settings), consistent with a narrower, more specific reconstruction-feedback effect than the catastrophic, general instabilities seen elsewhere in this project (Task 41C's `D2`, Task 42's `D2`).

This is the **second independent attack family** — following Task 42's model-update poisoning — where D0's substantive detection quality is excellent, but a single frozen ceiling, calibrated once against label-poisoning behavior in Task 40, is the limiting factor rather than any weakness in the detector itself. Two independent closures showing the identical pattern is meaningfully stronger evidence than either alone that the ceiling itself, not D0, deserves methodological reconsideration — a recommendation for future work and the paper's limitations section, not grounds to reopen this frozen result.

## 4. Baseline comparison — the first in this project's history

None of the three standard robust-aggregation baselines are viable as tested. All three fail clean utility before any attack even begins (H3), which is an inherent structural cost of these methods (discarding or downweighting some legitimate signal by design), not an implementation defect — the aggregation math was independently unit-verified against hand-computed cases before this run (tag `task43-c1-baselines-verified-v414`).

`trimmed_mean`'s negative mean damage-removed fraction (−0.409, actively worse than plain FedAvg) has a precise, traceable mechanism: its trim setting (4 lowest + 4 highest, a symmetric 20%/20% split) assumes malicious mass could land on either side of the sorted distribution. Both `extreme_value` and `coordinated_split` instead push all 8 malicious clients toward the *same* direction, so all 8 malicious values cluster on one side — trimming only 4 from that side leaves 4 malicious values still contaminating the average. `coordinate_median` does not share this flaw (it depends only on the single middle-ranked value, not a fixed trim count), which is consistent with its comparatively better, though still H3-failing, performance.

## 5. An honest, unresolved gap — flagged, not glossed over

Unlike D0, none of the three baseline arms log which clients were actually retained or discarded each round — only D0 has per-round audit data. `multi_krum` shows an unexplained non-monotonic pattern under `extreme_value` (catastrophic collapse at 2×, full recovery at 5×/10×) that is plausibly explained by all 8 malicious clients submitting *identical* vectors each round (no per-client jitter), creating a zero-internal-distance cluster whose pairwise-to-honest distance grows with attack strength — but this mechanism is **not confirmed** without selection-level logging, and should not be asserted as fact in the paper without either adding that logging and rerunning, or explicitly scoping the claim as a plausible-but-unverified hypothesis.

## 6. Scope of the claim

**Supported:** D0, unmodified, substantively outperforms every tested robust-aggregation baseline on every attack type, at equal-or-lower clean-utility cost, and detects coordinated/split attacks as reliably as independent ones — coordination did not evade its per-client behavioral detection here. D0's one weakness is narrow and specific: rising false positives under a *directionally-coordinated* coalition at higher boost strengths, not a general detection failure.

**Not supported:** that D0 "passes" Task 43 by the frozen formal rule — it does not, and this section should not be read as claiming otherwise. Nor does this result establish that the 0.05 FPR ceiling is wrong — only that it is now twice in a row the specific, identifiable limiting factor for D0 against non-label-poisoning attack families, which is evidence toward reconsidering it, not proof.

## 7. Preregistration correction, documented rather than silently fixed

`TASK43_PREREGISTRATION_V414.md`'s H4 referenced "3 of the 4 attack types," but the frozen panel has three types (constant, extreme_value, coordinated_split), not four — a wording error caught during analysis. H4 was evaluated against a majority-of-3 threshold (≥2) as the natural correction; the actual result (3 of 3) satisfies this regardless of the exact threshold chosen.

---

*Roadmap status: Task 43's C2 stage is closed. Per the frozen rule, D0 does not proceed to C3 confirmatory multiseed on this attack panel. The cross-task FPR-ceiling pattern (Task 42 and Task 43 both) is recommended as a methodological item for the paper's limitations section and as a candidate for a dedicated future recalibration study, not as grounds to revisit either frozen result.*
