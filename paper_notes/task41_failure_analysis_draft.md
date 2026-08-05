# Task 41 Failure and Limitation Analysis: Backdoor and Trigger-Based Poisoning

*Draft for the Results / Limitations section. All figures below are drawn directly from frozen experiment outputs (git tags `task41b-frozen-v4.11b9`, `task40-v3.20b1-freeze`, `task41c-c3-confirmatory-failure-v4.12c3`, `task41c-c3-h2-correction-v1`). Every claim, including the clean-round FPR drift in §4, has been directly verified against `task41c3_all_seed_round_metrics.csv`.*

## 1. Overview

Following the completed Task 40 evaluation of untargeted label-poisoning attacks, Task 41 tested the frozen LFighter defense — the temporal 8×8 transition-signature detector with independent historical anchor and `center_plus_residual` trusted reconstruction — against backdoor and trigger-based poisoning. This attack family differs fundamentally from the label-poisoning attacks evaluated in Tasks 39–40: rather than corrupting labels broadly or for a fixed source class, a backdoor attacker preserves ordinary model behavior on clean inputs and activates only when a specific, low-rate feature trigger is present. This makes the attack considerably harder to detect through prediction-transition monitoring alone, since the malicious behavior is deliberately rare and localized.

Task 41 proceeded in two stages: Task 41B, which tested the unmodified frozen Task 40 defense against realistic in-range IoT traffic triggers, and Task 41C, a separately preregistered extension that developed and confirmatory-tested trigger-aware detector candidates after Task 41B's result.

## 2. Task 41B: The Frozen Task 40 Detector Does Not Detect Backdoors

Task 41B evaluated the unmodified, frozen Task 40 detector (`D0`) against two frozen trigger families (`flow_iat_exact`, `active_idle_exact`) at multiple low poison rates, using the same 20-client Dirichlet partition, malicious coalition (`clients [1, 7, 8, 10, 14, 15, 17, 18]`), and four-round trusted warmup as every prior stage.

**Finding:** The frozen detector does not consistently mitigate stealthy 1% dirty-label backdoors. This is consistent with the design rationale documented in the roadmap (Task 21): the detector's core signal — class-conditional prediction-transition drift — was calibrated against label-flipping and untargeted-poisoning attacks that visibly shift a client's overall prediction distribution. A backdoor attacker's local training data is otherwise clean; the poisoned samples are a small, trigger-gated minority, producing far weaker transition-signature drift than the attacks the detector was designed against.

This result was retained as a negative finding per the project's evidence-retention policy and motivated the Task 41C extension rather than a silent redesign of the frozen Task 40 method.

## 3. Task 41C: A Preregistered Trigger-Aware Extension

### 3.1 Preregistration

Task 41C was preregistered in full before any development began (`TASK41C_PREREGISTRATION_V412C0`, tag `task41c-c0-preregistered-v4.12c0`), fixing in advance:

- Four confirmatory seeds (7, 99, 123, 2026) and the existing malicious coalition and partition.
- An attack panel of low-rate dirty-label attacks at 0.5%, 1%, and 2% poison fraction across two trigger families.
- Four candidate detectors: `D0` (frozen Task 40 reference, mandatory), `D1` (trigger-response-shift), `D2` (trigger-gradient-alignment), `D3` (equal-weight rank fusion of D0/D1/D2).
- A lexicographic, non-negotiable selection rule: reject any candidate with benign false-positive rate (FPR) above 0.05 or mean clean macro-F1 loss above 0.01; among survivors, maximize malicious-client recall; break ties by ASR reduction, then runtime.
- Three falsifiable hypotheses for the confirmatory stage:
  - **H1 (detection):** mean malicious-client recall ≥ 0.50 and maximum benign FPR ≤ 0.05.
  - **H2 (mitigation):** for each trigger family, mean triggered attack-success-rate (ASR) reduction ≥ 25% relative to paired plain FedAvg, in at least 3 of 4 confirmatory seeds.
  - **H3 (clean utility):** mean clean validation macro-F1 loss ≤ 0.01, no single seed losing more than 0.02.

### 3.2 Stage C2: Seed-7 Development Screen

A single-seed (seed 7) development screen tested all four candidates across the full attack panel. Two candidates failed the hard gates outright and were excluded from further consideration: `D1` (maximum benign FPR 0.25) and `D3` (maximum benign FPR 0.15) — both roughly 3–5× over the 0.05 ceiling. Of the two survivors:

| Candidate | Max benign FPR | Mean malicious recall (seed 7) | Mean relative ASR reduction |
|---|---|---|---|
| D0 (frozen reference) | 0.05 | 0.042 | −0.010 |
| D2 (trigger-gradient-alignment) | 0.00 | 0.536 | 0.302 |

`D2` was selected under the frozen lexicographic rule (highest recall among survivors). A pattern was already visible at this stage: `D2`'s mitigation was consistently **negative** at the lowest (0.5%) poison rate across both trigger families, becoming positive only at 1% and 2% poison rates.

### 3.3 Stage C3: Confirmatory Multiseed Result — Failure

`D2` was then evaluated confirmatory-multiseed (all four preregistered seeds), paired against `D0` as the mandatory reference, with every threshold and profile frozen exactly as calibrated in C1c3/C2 — no retuning.

| Candidate | Max benign FPR | Mean malicious recall (4 seeds) | Mean relative ASR reduction | Clean utility loss |
|---|---|---|---|---|
| D0 (frozen reference) | 0.05 | 0.077 | −0.023 | ≈0 |
| D2 (selected candidate) | **0.30** | 0.497 | 0.111 | ≈0 (slightly negative, i.e. no measurable cost) |

**H1 (detection): FAILED.** Mean malicious-client recall (0.497) fell fractionally short of the 0.50 requirement, and maximum benign FPR (0.30) exceeded the 0.05 ceiling by a factor of six. Because H1 requires *both* conditions, this alone is decisive.

**H2 (mitigation): FAILED**, under the correctly-specified per-seed methodology (see §3.4). Per trigger family, mean relative ASR reduction by seed:

| Seed | `flow_iat_exact` | `active_idle_exact` |
|---|---|---|
| 7 | 0.362 (pass) | 0.243 (fail — just under 0.25) |
| 99 | −0.025 (fail) | 0.131 (fail) |
| 123 | 0.214 (fail — just under 0.25) | 0.361 (pass) |
| 2026 | −0.689 (fail) | 0.291 (pass) |
| **Seeds passing** | **1 / 4** (need ≥ 3) | **2 / 4** (need ≥ 3) |

Neither trigger family reaches the required 3-of-4 seed threshold.

**H3 (clean utility): PASSED.** `D2` did not measurably degrade clean validation performance in any seed.

**Confirmatory verdict: `CONFIRMATORY_FAILURE_RECORD_AND_STOP`.** Per the preregistration, this closes Task 41C without retuning or inventing a replacement candidate inside the same preregistered task.

### 3.4 A Methodology Correction, Retained Transparently

The original C3 summarization code computed the H2 per-trigger-family check by pooling all four seeds together before thresholding, rather than applying the preregistered per-seed test. This was identified during review, corrected in a standalone reprocessing script (`audit_task41c_c3_h2_correction_v1.py`, tag `task41c-c3-h2-correction-v1`) that reprocesses only the already-frozen paired-round data — no retraining or reopening of the underlying experiment — and is committed alongside, not in place of, the original record. The correction does not change the overall verdict: H1 already fails independently and decisively (FPR six times the ceiling), and the corrected H2 computation confirms failure by an even wider margin than the original pooled calculation suggested.

## 4. Pattern Analysis

Two consistent, cross-seed patterns emerge from the confirmatory data, both scientifically informative beyond the simple pass/fail outcome:

**Poison-rate dependence.** At the lowest, stealthiest poison rate (0.5%), `D2`'s mitigation is negative or negligible in 7 of 8 seed×trigger-family combinations (all but seed 2026 / `active_idle_exact`, which is only marginally positive at 0.007). Mitigation strengthens substantially at 1% and especially 2% poison rates. This indicates `D2`'s trigger-gradient-alignment signal requires a poisoning rate well above what a patient real-world attacker would use — the defense is weakest exactly where a backdoor attacker would prefer to operate.

**Detector instability, confirmed.** `D2`'s maximum benign FPR across the confirmatory run (0.30) occurs specifically in the `clean_reference` (no-attack) branch of seed 99, and is not an isolated spike but a **progressive drift with no attack present at any point**: FPR rises from 0.10 (round 2) to 0.15 (round 3) to 0.30 (round 4) across four monitoring rounds of purely clean continuation training. A second, milder instance appears in seed 2026's clean branch (FPR 0.10 at round 4). By contrast, every recorded FPR value across all *attack* conditions and all seeds is ≤ 0.083 — lower than the worst clean-branch value. This is a materially different and more serious finding than "over-sensitive under attack": `D2`'s score for at least some benign clients drifts upward over successive rounds of ordinary, non-adversarial training, independent of any poisoning. Verified directly from `task41c3_all_seed_round_metrics.csv`:

| condition_id | model_seed | monitoring_round | benign_false_positive_rate |
|---|---|---|---|
| clean_reference | 99 | 4 | **0.300** |
| clean_reference | 99 | 3 | 0.150 |
| clean_reference | 99 | 2 | 0.100 |
| clean_reference | 2026 | 4 | 0.100 |
| (worst attack-condition value, any seed) | — | — | 0.083 |

## 5. Interpretation and Scope of the Claim

Task 41's honest conclusion is **not** that LFighter fails universally, nor that trigger-aware detection is impossible. It is a precisely scoped negative result: under the frozen assumptions carried through every prior stage (this partition, this coalition, this warmup length, this probe set), neither the unmodified Task 40 transition-signature detector (41B) nor a purpose-built trigger-gradient-alignment extension (41C) provides detection that is simultaneously (a) strong enough to meet a 50% recall bar, (b) stable enough to avoid false-alarming on clean traffic — `D2` demonstrably drifts toward false-flagging benign clients over successive clean rounds even with no attack present — and (c) consistent across seeds, particularly against low-rate, realistic-strength triggers.

This is consistent with, and extends, the project's own earlier signal-discovery finding (Task 21) that prediction-transition drift is a comparatively weak signal for subtle, localized attacks. It suggests that detecting backdoor/trigger poisoning within this framework may require a qualitatively different signal (e.g., input-space trigger reconstruction, activation-clustering style methods, or spectral signatures) rather than an alignment-based refinement of the existing transition/gradient features — a candidate direction for future work rather than a claim this study can support.

## 6. What This Section Supports and Does Not Support

**Supported:** LFighter's frozen detection paradigm, and one reasonable trigger-aware extension of it, do not reliably defend against backdoor/trigger attacks under the tested assumptions, particularly at low poison rates.

**Not supported by this evidence:** any claim that backdoor defense is impossible for LFighter-style systems in general, or that the `D2` mechanism is without any value (it did significantly outperform the frozen baseline's near-zero recall, just not enough to clear the preregistered bar).

**Recommended framing for the paper:** report both 41B and 41C as retained negative results with full quantitative detail (as above), explicitly scope the claim to the tested conditions, and note the poison-rate dependence and (pending verification) the clean-round FPR instability as concrete, specific limitations rather than a vague "did not work."

---

*Roadmap status: Task 41 closed with documented negative result. Per the master roadmap, the next attack-breadth item is Task 42 (direct model-update poisoning: sign flipping, update scaling, additive perturbation, model replacement).*
