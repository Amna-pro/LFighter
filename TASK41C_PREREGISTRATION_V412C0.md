# Task 41C Preregistration, Backdoor-Defense Extension

**Protocol version:** 4.12C.0
**Parent frozen evidence:** Task 41B, tag `task41b-frozen-v4.11b9`, commit `f29b026`
**Status:** Preregistered before any Task 41C experiment
**Reserved test access:** Prohibited until the final C5 gate

## Purpose

Task 41B established that the frozen Task 40 LFighter detector and `center_plus_residual` reconstruction do not consistently mitigate stealthy 1% dirty-label backdoors. Task 41C is a separate extension. It does not modify, reinterpret, or reopen Task 41B.

Task 41C evaluates whether explicitly backdoor-aware client evidence can improve detection and mitigation while preserving benign-client precision, clean utility, and exact experimental pairing.

## Frozen research questions

1. Can a backdoor-aware client detector improve malicious-client recall under low-rate trigger poisoning while keeping benign-client FPR at or below 5%?
2. Can a separately calibrated backdoor-aware reconstruction policy reduce triggered ASR consistently across trigger families and seeds?
3. Does the selected extension remain effective against detector-aware and reconstruction-aware adaptive attackers?

## Frozen hypotheses

- **H1, detection:** Mean malicious-client recall will be at least 0.50 and maximum benign-client FPR will not exceed 0.05.
- **H2, mitigation:** For each confirmatory trigger family, mean triggered ASR will fall by at least 25% relative to exact paired FedAvg, with improvement in at least three of four seeds.
- **H3, clean utility:** Mean clean validation macro-F1 loss will not exceed 0.01, and no seed will lose more than 0.02.
- **H4, adaptive robustness:** Under adaptive attacks, ASR reduction will remain positive in at least three of four seeds, mean malicious-client recall will be at least 0.35, and maximum benign FPR will not exceed 0.05.

## Frozen data policy

Development may use only training, validation, clean warmup, independent anchor, balanced server probes, and previously frozen trigger specifications. `X_test_natural`, `y_test_natural`, `X_test_diagnostic`, and `y_test_diagnostic` are forbidden until a separately committed final-test gate.

## Frozen seeds and clients

Seeds are `7, 99, 123, 2026`. The 20-client partition and malicious-client set `[1, 7, 8, 10, 14, 15, 17, 18]` remain unchanged.

## Attack panel

The preregistered panel includes low-rate dirty-label attacks at 0.5%, 1%, and 2%, clean-label source-preserving attacks at 1% and 2%, a detector-aware white-box objective, and a reconstruction-aware scaled-update objective. The frozen trigger families are `flow_iat_exact` and `active_idle_exact`.

## Defense candidates

- `D0_frozen_task40_lfighter`, unchanged mandatory reference.
- `D1_trigger_response_shift`, based on class-conditional prediction shifts on frozen triggered server probes.
- `D2_trigger_gradient_alignment`, based on client-update alignment with trigger-target gradient directions.
- `D3_equal_rank_fusion`, equal-weight rank fusion of D0, D1, and D2.

All transforms and thresholds must be calibrated from clean warmup and independent-anchor evidence only. Attack-specific or trigger-specific thresholds are prohibited.

## Comparison arms

Plain FedAvg, frozen Task 40 LFighter, coordinate median, trimmed mean, Multi-Krum, RFA geometric median, FoolsGold, D1, D2, and D3.

## Candidate selection rule

Selection is lexicographic and frozen:

1. Reject any candidate with benign FPR above 0.05.
2. Reject any candidate with mean clean macro-F1 loss above 0.01 or any seed-7 round loss above 0.02.
3. Among survivors, maximize malicious-client recall across all preregistered A1 seed-7 conditions.
4. Break ties by relative ASR reduction, then runtime overhead.
5. If every candidate fails, Task 41C records failure. No new candidate may be invented inside the same preregistered task.

## Staged execution

- **C0:** Preregistration and integrity only.
- **C1:** Clean-only calibration.
- **C2:** Seed-7 development screening of every preregistered candidate.
- **C3:** Confirmatory multiseed evaluation on A1 and A2.
- **C4:** Adaptive multiseed evaluation on A3 and A4.
- **C5:** One final reserved-test evaluation after every choice is frozen and committed.

## Statistics and reporting

Every comparison is exact paired by trigger, poison fraction, seed, partition, warmup, poison plan, and round. Report deterministic paired-bootstrap 95% confidence intervals, exact sign-flip tests where applicable, Holm correction within hypothesis families, every seed, every failed condition, runtime, communication overhead, and memory overhead.

Every stage must produce CSV and JSON tables plus publication-quality PNG and PDF figures and a SHA-256 source/evidence manifest.

## Prohibited post-hoc changes

No seed, malicious-client, partition, trigger, poison-fraction, candidate-weight, or threshold changes are allowed after this freeze. No failed attack or seed may be dropped. No defense may be added after confirmatory outcomes are observed. Task 41B must remain unchanged.

## Final claim policy

Task 41C development results are not final paper claims. Final claims require the C5 gate, a single reserved-test evaluation, and no subsequent retuning. Negative and boundary results must be reported.
