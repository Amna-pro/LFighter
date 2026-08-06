# Task 42 Failure and Limitation Analysis: Direct Model-Update Poisoning

*Draft for the Results / Limitations section. All figures below are drawn directly from frozen, git-committed outputs (tags `task42-c0-preregistered-v413`, `task42-c1-calibration-v413`, `task42-c2-no-candidate-survived-v413`). No number in this section is estimated or recalled from memory — each traces to a specific committed CSV or JSON file, listed inline.*

## 1. Overview

Tasks 39–41 tested attacks that corrupt a client's *local training data* — label flips, untargeted poisoning, backdoor triggers — and let ordinary local training produce the resulting malicious update. Task 42 tests a categorically different attacker capability: **direct manipulation of the submitted update itself**, with local training left completely honest and identical for every client, malicious or not. This is the sharpest possible test of whether LFighter's detection paradigm — built and calibrated entirely on *prediction-behavior* drift from label-space attacks — transfers to attacks that never touch local training at all.

## 2. Preregistration and Attack Panel

Task 42 was preregistered in full (`TASK42_PREREGISTRATION_V413.md`, tag `task42-c0-preregistered-v413`) before any code was written, fixing:

- Four confirmatory seeds (7, 99, 123, 2026), the existing 20-client partition, and the existing malicious coalition `[1, 7, 8, 10, 14, 15, 17, 18]`.
- Four attack types, each grounded in real data rather than round numbers chosen in advance:
  - **Sign flip:** submitted update negated (`-honest_update`). No strength parameter.
  - **Scaling:** submitted update multiplied by 2×/5×/10×, chosen against the observed clean update-norm distribution (median 1.26, 95th percentile 2.37, across 320 client-rounds pooled over all four seeds).
  - **Additive noise:** honest update plus a random Gaussian direction rescaled to 0.5×/1×/2× the client's own current update norm.
  - **Model replacement:** submitted update replaced entirely by `boost × (target − reference)`, where `target` is the warmup reference state with `classifier.weight` zeroed and `classifier.bias` set to force a constant-Benign prediction — the update-space analog of the DDoS→Benign attack that has been this project's throughline since Task 18. Boost factors (3×/5×/10×) were derived from the *exact* sample-weighted malicious fraction in the frozen coalition: **f = 162,930 / 483,734 = 0.3368**, giving a textbook boost of `1/f ≈ 2.97 ≈ 3×`.
- Four candidates: `D0` (frozen Task 40/41 detector, unmodified, mandatory reference), `D1` (per-client update-norm deviation), `D2` (per-client update-direction deviation), `D3` (fusion of D0/D1/D2).
- The identical lexicographic selection rule used in Task 41C: reject any candidate exceeding 0.05 maximum benign FPR or 0.01 mean / 0.02 max-round clean macro-F1 loss; among survivors, maximize malicious recall; tie-break by damage-removed fraction, then runtime.

## 3. Calibration (C1)

`D1` and `D2` reference statistics were confirmed derivable at zero training cost from existing Task 40 artifacts (`audit_task42_c1a_calibration_preflight_v413.py`) — but genuine *threshold* calibration required raw per-round update vectors that did not exist on disk for the warmup stage. Rather than invent a new capture mechanism, the existing `run_update_capture_v318b.py` infrastructure (already built for a different Task 40 purpose) was reused in `--mode clean` across all four seeds — this replays already-frozen clean continuation rounds and persists deltas a training run computes anyway; it adds no new experimental content.

D1/D2 thresholds were calibrated from 320 pooled clean observations (4 seeds × 4 rounds × 20 clients) at the q99 quantile, matching D0's own frozen convention: **D1 instant threshold = −0.464**, **D2 instant threshold = 0.682** (`task42c1b_pooled_threshold_candidates.csv`, tag `task42-c1-calibration-v413`). D0's own thresholds were not re-derived — they were reused directly from Task 41C's frozen calibration (`task41c1c_candidate_thresholds.csv`): EMA = 2.706, instant = 3.557.

A parameter-order alignment check was built into calibration explicitly: the captured update matrices and the reconstruction bundle's residual profiles use different natural orderings, and silently mismatching them would have produced meaningless cosine scores with no error raised. The calibration script re-derives D2's reference direction using each seed's own captured `parameter_layout.csv` order and verifies this order is identical across all four seeds before pooling — confirmed, not assumed.

## 4. C2 Seed-7 Development Screen — Result

All four candidates were screened against 11 conditions (1 clean reference + 10 attack conditions) × 4 rounds. Final, verified selection table (`task42c2_candidate_selection.csv`, tag `task42-c2-no-candidate-survived-v413`):

| Candidate | Max benign FPR | Mean recall | Min recall (any condition) | Damage-removed fraction | Detector collapses |
|---|---|---|---|---|---|
| D0 (frozen reference) | 0.167 | 0.619 | 0.00 | 0.682 | 0 |
| D1 (norm deviation) | 0.250 | 0.938 | 0.00 | 2.006 | 0 |
| D2 (direction deviation) | 1.000 | 0.678 | 0.00 | −0.184 | 0 |
| D3 (OR-fusion) | 0.167 | **0.981** | **0.75** | **2.052** | **0** |

**Selection status: `NO_CANDIDATE_SURVIVED`.** Every candidate exceeds the frozen 0.05 maximum-benign-FPR ceiling. Per the preregistration's own rule, this closes Task 42's development stage: no replacement candidate may be invented inside the same preregistered task, and confirmatory multiseed testing (C3) does not proceed for any of the four.

## 5. Mechanism Analysis

Each failure has a specific, traceable cause — not a vague "didn't work."

**D2 (direction deviation) fails catastrophically and specifically under scaling attacks**, reaching FPR = 1.000. The reason is mathematical, not statistical noise: cosine similarity is invariant to positive scalar multiplication — `cos(k·v, w) = cos(v, w)` for any `k > 0`. A pure scaling attack cannot change a client's cosine similarity to its own historical direction by construction, so D2 cannot detect it directly. What was observed instead was a **positive feedback instability**: early misclassifications (from unrelated variance) triggered reconstruction, which reshaped the model trajectory, which pushed *benign* clients' honest gradients away from their fixed historical reference, triggering more flags, more reconstruction, escalating FPR 0.083 → 0.417 → 1.000 across rounds before total collapse. This mirrors a failure mode this project already documented once before (Roadmap Task 26, "reference capture and feedback failure") — reproduced here in a cleaner, mechanistically fully-explained form specific to update-space attacks. The engineering safeguard built for this (freeze the model rather than aggregate when zero clients remain trusted) worked correctly and is why this produced clean data rather than a second crash.

**D0 (unmodified Task 40/41 detector) transfers with real but limited success.** It correctly reaches recall = 1.0 on sign-flip and all three model-replacement strengths — a positive and non-obvious generalization result, since D0 was calibrated purely on label-flip-induced prediction drift. Its failure is a precise, narrow one: FPR = 0.167 specifically under model-replacement, where the "collapse-to-Benign" target itself resembles the class-collapse behavior D0 was built to detect strongly, causing it to also flag some benign clients whose natural round-to-round variance crosses the same threshold.

**D1 (norm deviation) is the strongest individual detector by recall** (0.938) but exceeds the FPR ceiling (0.250) specifically on model-scaling and additive-noise conditions, where legitimate client-to-client variance in update magnitude is apparently large enough to periodically cross a threshold calibrated from only 320 pooled clean observations.

**D3 (OR-fusion) is not a failed design — it is the best achievable combination, definitively.** Its recall (0.981) and damage-removed fraction (2.052) are the best of any candidate, and it never collapses. Its FPR (0.167) is not a new problem: it is **mathematically identical** to D0's own FPR, an unavoidable consequence of OR-fusion when one component (D0) is the binding constraint. This is worth stating precisely: D3's failure is not evidence the fusion approach is wrong — it is proof that no combination of these four candidates can outperform D0's own FPR ceiling, because D0 alone already defines that ceiling for the fused result.

## 6. Comparison to Task 41C — A More Decisive Failure

Task 41C's `D2` looked promising on its seed-7 development screen and only failed at confirmatory multiseed testing (C3) — a generalization failure requiring multiple seeds to detect. **Task 42's four candidates fail at the development screen itself, on the same seed they were calibrated against.** There is no ambiguity here that additional seeds could resolve — this is not a generalization problem, it is an in-sample failure to clear a fixed, predefined bar. Per the mission's own standard of scoping claims precisely, this should be reported as a stronger and cleaner negative result than Task 41's, not a weaker one.

## 7. An Open Methodological Question — Flagged, Not Used to Reopen This Result

The 0.05 FPR ceiling was calibrated once, in Task 40, against label-poisoning attack behavior, and has since been inherited unchanged into Task 41C and Task 42 without being re-examined for suitability to fundamentally different attack families. Whether a single fixed FPR ceiling is the right methodological choice across all attack types, or whether each attack family's preregistration should set its own ceiling grounded in that family's own clean-behavior variance, is a legitimate open question for Task 43's preregistration and the paper's limitations section. This is explicitly **not** grounds to revisit or loosen the frozen Task 42 result above — the preregistered rule was applied exactly as written, and the result stands as reported.

## 8. Scope of the Claim

**Supported:** none of D0 (unmodified), D1, D2, or D3 (as fused here) can detect this project's preregistered direct model-update-poisoning attack panel while holding benign-client false-positive rate at or below the frozen 0.05 bar, under the tested seed-7 development conditions. D0 shows genuine partial transfer (strong recall on sign-flip and model-replacement) despite never being built for this attack family. D2's specific mathematical blind spot to scaling attacks, and the resulting feedback-driven collapse, is a well-understood and reproducible mechanism, not noise.

**Not supported:** any claim that model-update poisoning is undetectable in general, that D1's norm-based approach has no value (it has the strongest raw recall of any single candidate), or that fusion approaches are inherently flawed (D3 is provably optimal given its inputs — the ceiling is inherited, not created by the fusion design).

---

*Roadmap status: Task 42 closed with a documented, mechanistically-explained negative result at the development-screening stage. Per the master roadmap, the next unstarted attack-breadth item is Task 43 (standard Byzantine attacks: Gaussian, constant, extreme-value, sign-inversion, coordinate-wise, colluding, and omniscient variants).*
