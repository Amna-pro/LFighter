# BATR FL Reviewer Revision Preregistration Draft v4.32.0

Status: DRAFT ONLY. Do not run reviewer revision outcome experiments until this document is reviewed, the published temporal/probe comparator is selected and validated, and a frozen commit/tag is created.

Base scientific state: commit 726093f, `feature/cic-iot-diad-task68c2-final-audit-correction-v4312`.

## 1. Purpose

This protocol defines reviewer-requested confirmatory and sensitivity work for the BATR FL manuscript. It is a post-review extension of the already completed Tasks 44 to 68. It must not be described as part of the original untouched five-seed final evaluation.

The original five-seed final panel remains historically frozen. New reviewer-extension experiments must use predeclared configurations, retain negative results, and prohibit outcome-dependent tuning.

## 2. Existing evidence that will be reused without rerunning

1. Task 44 poisoning-strength curves at poison fractions 0.25, 0.50, 0.75 and 1.00.
2. Task 45 malicious-coalition-size study.
3. Task 65 original one-shot final evaluation.
4. Task 66 frozen final statistics.
5. Task 67 publication tables and resource summaries.
6. Task 68 independent final claim-to-evidence audit.

## 3. Frozen core configuration

Unless an experiment explicitly varies one factor below, retain:

- Dataset: CIC IoT DIAD 2024 processed behavioral-only representation.
- Logical clients: 20.
- Dirichlet alpha: 0.5.
- Full participation.
- Warmup: global rounds 1 to 4.
- Original monitored window: global rounds 5 to 8.
- Malicious coalition for primary matched comparisons: clients [1, 7, 8, 10, 14, 15, 17, 18].
- Poison fraction: 1.0.
- Attacks: all_to_one_benign, cyclic_shift, multiclass_partial_cycle, pairwise_swap, random_flip.
- Batch size: 2048.
- Evaluation batch size: 4096.
- Learning rate: 0.0003.
- Weight decay: 0.0001.
- Threads: 6.
- Probe: 48 examples per class, seed 3701.
- EMA decay: 0.65.
- Detector thresholds: EMA q99 and instantaneous q95 computed from clean warmup calibration only.
- Reconstruction reference: center_plus_residual.
- Original sample-count weights preserved.
- No best-round selection. Round 8 remains the original primary endpoint.

## 4. Source-audit interpretation to be stated in the manuscript

- Gamma 0.65 is the EMA decay, not a trust-weighting gamma. The temporal update gives 65% weight to the previous EMA state and 35% to the current instantaneous score.
- The defended continuation derives the EMA q99 threshold and instantaneous q95 threshold from clean leave-one-round-out warmup scores.
- The class-balanced probe is constructed from the validation split, with 48 examples per class and seed 3701. Its hash is checked against the warmup branch point.
- The reconstruction guard aborts by raising an error when fewer than 8 unflagged/trusted clients remain. No alternate mitigation rule is substituted.
- The value 8 is an operational guard in the existing implementation; no theoretical derivation was recovered. The revised manuscript must not present it as a Byzantine robustness guarantee.
- Existing round timing measures complete monitored-round wall time. Task 65 branch timing measures whole subprocess wall time and peak RSS. Neither isolates server-side defense overhead.

## 5. Reviewer Extension A: Matched strong baselines

### A1. Required matched arms

Run under identical seed, partition, warmup checkpoint, attack plan, local optimizer, local epochs, batch size, learning rate, evaluation protocol and monitored rounds:

1. Plain FedAvg.
2. Frozen BATR FL center_plus_residual.
3. Coordinate-wise median.
4. Multi-Krum using the preregistered known Byzantine-count setting where mathematically valid.
5. RFA / geometric median using the already verified deterministic implementation.
6. Original LFighter adapted as the class-aware label-flip comparator, only after exact interface equivalence is verified.
7. P4P, implemented from Khang et al. (2026) as the closest probe/temporal comparator.

### A1.1. Frozen P4P comparator specification

The matched P4P arm will preserve the published defense logic while running inside the BATR FL federation so that model architecture, client partition, local optimization, warmup checkpoint, attacks, monitored rounds and evaluation are matched.

For each monitored round:

1. Construct the probe from the historical global trajectory:
   `v_t = W_t - W_{t-1}`.
   Because BATR FL already has warmup checkpoints through global round 4, the first reviewer-comparison round can construct its initial P4P probe from the round-4 and round-3 global checkpoints.

2. Compute every client update norm and apply MAD magnitude filtering.
   Published coefficient: `k = 3.0`.
   Operational implementation: median norm plus/minus `3 * MAD`; only updates inside the acceptance interval proceed.

3. For each magnitude-passing update, compute the server-side scalar probe response:
   `r_i^t = cosine(delta_W_i^t, v_t)`.

4. Apply the published 1D ensemble:
   - K-Means with `K = 2`; the minority cluster contributes one suspicious vote.
   - DBSCAN with `eps = 0.5`, `minPts = 5`; noise points contribute one suspicious vote.
   - Isolation Forest with contamination `0.1`; samples classified as outliers contribute one suspicious vote.
   - A client is anomalous when it receives at least 2 of 3 votes.

5. Update temporal suspicion:
   `s_i^t = max(0, s_i^{t-1} + delta_i^t - gamma_p4p)`,
   with `gamma_p4p = 0.2`.
   Published permanent-removal threshold: `theta_p4p = 2.0`.

6. Aggregate only updates that:
   - pass magnitude filtering,
   - are not currently anomalous,
   - and have temporal suspicion `<= 2.0`.
   P4P therefore acts as a binary filtering defense in the matched implementation; suspicious current updates are not reconstructed.

7. Aggregate the trusted set by FedAvg using the BATR FL experiment's original sample-count weights. This preserves the manuscript's matched-comparison requirement while retaining P4P's published trusted-set filtering logic.

### A1.2. P4P implementation disambiguation rules

The paper is sufficiently detailed for a faithful comparator, but several notation-level ambiguities must be frozen before execution:

- Isolation Forest: use the semantic outlier decision rather than relying on the sign convention shown in the paper pseudocode, because library label conventions differ.
- If K-Means yields equally sized clusters, designate the cluster with the lower mean probe-response cosine as suspicious. This is a deterministic tie rule needed for reproducibility and is not used for tuning.
- The magnitude filter is implemented as the median norm plus/minus `k * MAD`, with `k = 3.0`, following the paper's MAD description and stated threshold coefficient.
- P4P suspicion scores are initialized to zero at the beginning of the monitored comparison window.
- No P4P hyperparameter is tuned on BATR FL reviewer-extension outcomes.
- P4P's original training settings (15 rounds, 5 local epochs, batch 32, SGD learning rate 0.01, 30% malicious clients) are not copied into the matched arm, because doing so would confound the comparison. Only its defense mechanism and published defense hyperparameters are transferred.

### A2. Outcomes

For every attack and seed report:

- diagnostic and natural macro F1
- balanced accuracy
- worst-class recall
- per-class recall
- confusion matrix
- attack-specific error where defined
- malicious-client recall and benign-client FPR where the method has a client-detection decision
- runtime and peak RSS
- failure or invalidity conditions

For methods without client detection, detector metrics are not applicable and must not be fabricated.


## 6. Reviewer Extension B: Reconstruction/mitigation ablation

Keep the BATR detector algorithm, probe, calibration, thresholds and attack configuration frozen. Change only the mitigation response.

Arms:

1. no mitigation / suspicious update unchanged
2. hard rejection of flagged updates with remaining sample weights renormalized
3. fixed down-weighting of flagged updates by 0.10, chosen before outcome access to match the previously frozen minimum-trust constant used in development work
4. center_only replacement
5. center_plus_residual replacement

The main comparison is trajectory-level. Each arm recomputes the same frozen detector rule on its own resulting trajectory. Do not reuse outcome-dependent flags from another arm.

Primary ablation endpoint: Round 8 validation utility and attack damage reduction. Final test arrays are not needed for this ablation unless separately frozen later.

## 7. Reviewer Extension C: Trusted-warmup contamination sensitivity

This is a post-review sensitivity study, not part of the original final protocol.

Use development/validation data only.

- Seeds: 7, 99, 123, 2026.
- Same 8-client malicious coalition.
- Monitored poison fraction: 1.0.
- Warmup contamination fractions: 0.00, 0.10, 0.25, 0.50.
- Attacks: all_to_one_benign and random_flip.
- Apply contamination during warmup to the same malicious-client identities using the corresponding label corruption mapping.
- Rebuild the historical anchor, residuals and clean-calibration objects from the contaminated warmup exactly as the method would see them.
- Do not recalibrate after reviewing attacked outcomes.

Report detector recall, benign FPR, utility, attack-specific error, reconstruction frequency and any aborts.

## 8. Reviewer Extension D: Detector-probe-size sensitivity

Use development/validation data only.

- Probe sizes per class: 12, 24, 48, 96.
- Construct deterministic nested probe subsets from the validation split with a frozen selection seed.
- Recompute clean calibration independently for each probe size because the score distribution changes with probe size.
- Keep EMA decay and q99/q95 quantile rules fixed.
- Seeds: 7, 99, 123, 2026.
- Attacks: all five primary label-corruption mappings.

Report malicious recall, benign FPR, utility, attack-specific error, runtime and probe inference cost.

No probe size is selected as a new default based on these outcomes. The original 48-per-class method remains the frozen reference.

## 9. Reviewer Extension E: Longer-horizon residual-aging study

Use development/validation data only.

- Warmup remains rounds 1 to 4.
- Extend monitored continuation from 4 rounds to 12 rounds, producing global rounds 5 to 16.
- Keep the stored warmup residual fixed for the entire continuation.
- Use all five primary attacks and seeds 7, 99, 123, 2026.
- No threshold or residual refresh is allowed.

Report round-wise malicious recall, benign FPR, macro F1, balanced accuracy, worst-class recall, attack-specific error, reconstructed-client count and residual/center scale diagnostics.

This experiment evaluates longer-horizon residual reuse. It must not be described as a true concept-drift experiment unless the data distribution itself is changed by a separately preregistered, scientifically justified drift mechanism.

## 10. Reviewer Extension F: Larger prospective seed cohort

The original five final seeds remain unchanged and are reported as the original frozen cohort.

Add ten prospectively reserved reviewer-extension model seeds:

1660346836
1009144290
983376964
352494761
722985454
419358667
2034409938
443861967
1552002875
822602003

These were deterministically generated before reviewer-extension outcome runs from NumPy `default_rng(4320)` with integer range [1, 2147483647).

Before execution:

1. Verify none overlap original development or final model seeds.
2. Freeze the list and its SHA256 manifest.
3. Verify all required client partitions/checkpoints can be constructed without test-array access.
4. Freeze source hashes and attack plans.

The ten new seeds form the reviewer-extension confirmatory cohort. The original five-seed cohort is reported separately. A pooled 15-seed analysis may be shown as a secondary combined analysis, clearly labeled as combining the original and reviewer-extension cohorts.

## 11. Statistical plan

For the ten new reviewer-extension seeds:

- show every seed value
- report mean and median paired contrast
- report deterministic paired bootstrap 95% confidence intervals
- report exact two-sided paired sign-flip p values where applicable
- report effect sizes
- apply Holm correction within predeclared metric families
- retain zero and negative effects
- do not use bootstrap resampling to pretend there are more independent seeds

The original five-seed statistics remain unchanged.

## 12. Runtime protocol

Add `time.perf_counter()` instrumentation for distinct phases:

1. local client training
2. server probe inference and signature construction
3. detector scoring
4. mitigation/reconstruction
5. aggregation
6. validation evaluation

Record per-round phase timings. Report full distributions across seeds, attacks and rounds, not only one total runtime.

Define server-side BATR overhead as the sum of probe/signature, detector-scoring and reconstruction time. Do not claim that BATR is faster than FedAvg merely because a whole-branch wall-clock total is lower.

Record hardware, operating system, Python, PyTorch, CPU/GPU and thread count.

## 13. Minimum-unflagged-client guard

Retain the current `< 8` abort guard for exact reproduction of the frozen method.

For the manuscript:

- call it an operational safety guard
- state that the implementation raises an error and performs no fallback substitution
- do not claim a theoretical Byzantine threshold
- report the observed number of aborts in every reviewer-extension study

If a separate threshold-sensitivity study is added, it must be preregistered and must not be used to replace 8 after observing outcomes.

## 14. EMA and threshold justification

No final-outcome tuning is allowed.

The manuscript will state that:

- EMA decay 0.65 was inherited from the development-stage frozen temporal detector and fixed before the original final evaluation.
- q99 for the EMA branch and q95 for the instantaneous rescue branch were frozen before the original final evaluation.
- the reviewer-extension studies preserve these values unless a dedicated sensitivity study is explicitly labeled post hoc and non-selective.

A gamma/quantile sensitivity grid is not required for the main revision because the reviewer allowed either sensitivity analysis or a clearly documented fixed pre-evaluation rule. If later added, it must not choose a new default.

## 15. Integrity and chronology rules

- Never overwrite Tasks 44 to 68.
- Write all reviewer-extension outputs to new directories.
- Freeze a new Git commit/tag before any reviewer-extension attacked outcomes are opened.
- Do not describe new reviewer-extension seeds as part of the original preregistration.
- Do not describe post-review sensitivity studies as untouched final evaluation.
- Preserve failed runs and negative results.
- If an implementation bug requires correction, freeze the correction before rerunning all affected conditions from scratch.
- Do not access final test arrays for sensitivity studies that can be answered on development/validation data.

## 16. Comparator decision

P4P (Khang et al., 2026) is frozen as the closest published probe/temporal comparator. Its published equations, workflow, hyperparameters and aggregation behavior have been reviewed from the full paper.

Before any outcome experiment, the P4P adapter must pass synthetic unit tests and a clean-trajectory/interface preflight. No reviewer-extension outcome should be opened until those checks and the complete protocol are committed and tagged.
