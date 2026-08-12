# Task 45 Preregistration, Malicious Coalition Size Variation

**Protocol version:** 4.16.0
**Parent frozen checkpoint:** `task44-c2-strength-curve-decided-v415` at `cc84bc343508ad355064b03e89645fa165a6daad`
**Status:** Preregistered before any Task 45 experiment
**Reserved test access:** Prohibited until a separately committed final gate

## Purpose and scope

Task 45 measures how the frozen LFighter defense changes as the malicious population grows from a single client to half of the 20 client federation. Only coalition size and preregistered coalition identity vary. The attack remains the primary DDoS to Benign targeted label flip, poison fraction remains 1.0, and the frozen D0 detector plus `center_plus_residual` reconstruction remains unchanged.

The size grid is `{1, 2, 4, 6, 8, 10}`. Size 10 is included because 13 clients in the frozen partition have at least 1000 clean DDoS rows, so multiple valid size 10 identities exist.

## Weakest design point and its control

Changing coalition size can be confounded by changing which clients are malicious. A single identity per size would therefore not support a coalition size claim. Task 45 uses three fixed coalition families, with coalitions nested inside each family. This isolates the size progression within a family while allowing identity sensitivity to be measured between families.

Three families are the minimum fixed replication used here to limit the already large execution matrix. They do not establish exhaustive identity robustness. Seeds are repeated measurements within each family and must never be described as additional coalition identities.

## Frozen attack and defense

The attack pair is DDoS to Benign with poison fraction 1.0. A client is eligible only if its frozen training partition contains at least 1000 clean DDoS rows. The expected eligible IDs are `[1, 3, 5, 7, 8, 10, 13, 14, 15, 16, 17, 18, 19]`.

The plain arm uses `run_exact_plain_fedavg_v3132.py`. The defended arm uses `run_frozen_reconstruction_v3123.py` with `trusted_reconstruction`. Neither frozen runner is modified. Both must be invoked through the Task 45 development only loader adapter because their inherited shared loader materializes every array in the NPZ, including reserved test arrays, even though the runners only reference training and validation arrays. The adapter exposes only `X_train`, `y_train`, `X_val`, and `y_val` and raises immediately on any other request.

## Frozen coalition families

For every family and size `k`, the coalition is the sorted set of the first `k` IDs in that family’s frozen ordering.

### Family A, development anchor

Frozen order: `[7, 15, 1, 14, 18, 10, 17, 8, 13, 19, 16, 3, 5]`.

The first eight IDs reproduce the original development coalition exactly as a set: `[1, 7, 8, 10, 14, 15, 17, 18]`. The eight anchor IDs were ranked by ascending SHA256 of `task45_v416|family_A|client_NNN`. Remaining eligible IDs were ranked separately by `task45_v416|family_A|extension|client_NNN`, ensuring the size 8 anchor is preserved.

### Family B, hash ranked replicate

Frozen order: `[19, 16, 13, 15, 7, 5, 10, 8, 18, 14, 17, 1, 3]`.

Ranking key: `task45_v416|family_B|client_NNN`.

### Family C, hash ranked replicate

Frozen order: `[8, 16, 5, 3, 1, 18, 7, 15, 14, 19, 13, 17, 10]`.

Ranking key: `task45_v416|family_C|client_NNN`.

All rankings use ascending SHA256 hexadecimal digest with ascending client ID as a deterministic tie break. Ranking uses no attack result, validation metric, test array, or model outcome. The manifest generator must reproduce these exact orders from the frozen training partition or stop.

## Research questions

1. How do plain attack damage, malicious client recall, benign client false positive rate, and reconstruction effectiveness change with coalition size?
2. How sensitive are the curves to coalition identity at the same size?
3. At what sizes, if any, does contamination of the current round consensus reduce the effectiveness of the frozen detector or reconstruction?

## Frozen hypotheses

* **H1, detection robustness:** At every size, mean malicious client recall across all families and seeds is at least 0.50 and maximum benign client FPR is at most 0.05.
* **H2, mitigation:** At every size with mean plain FedAvg attack excess greater than 0.05, mean damage removed fraction by frozen D0 is positive.
* **H3, attack scale:** Within at least two of the three nested families, mean plain FedAvg attack excess is nondecreasing across sizes when averaged over seeds `7`, `99`, `123`, and `2026`.

These hypotheses are evaluated as written. Failure is retained as a boundary result and does not authorize a threshold, family, or method change.

## Pairing and statistics

Every plain and defended comparison is paired by family, coalition size, seed, partition, warmup checkpoint, poison index hash, and round. Attack seed equals model seed, matching the frozen multiseed protocol. Coalition membership is identical across seeds.

Results must show every family separately and summarize across exactly three coalition identities per size. Report between family range and standard deviation. Use deterministic paired bootstrap confidence intervals and exact paired sign flip tests only where their sample counts and estimands permit, with Holm correction inside each hypothesis family.

## Staged execution

* **C0:** Create and audit the preregistration, selector, and development only loader adapter. No training or attack execution. Commit all C0 files and audit outputs before continuing.
* **C1:** Count source rows, freeze the coalition manifest, verify array isolation, run static interface checks and dry runs, and verify whether the original Family A size 8 evidence is exactly reusable. No training.
* **C2:** Run all 18 family by size conditions on seed 7, with exact paired plain and frozen D0 arms.
* **C3:** Run the same frozen 18 conditions on seeds 99, 123, and 2026.
* **C4:** Produce the full curves, identity sensitivity analysis, statistics, and H1 through H3 decisions using development and validation artifacts only.
* **C5:** Reserved tests only under a separate preregistered and committed final gate, followed by no retuning.

## Stopping rules

Stop immediately if a reserved test array is requested or materialized before C5. Stop if the partition does not reproduce the 13 expected eligible IDs, a family is not nested, identities coincide at any size, an explicit malicious client has fewer than 1000 clean DDoS rows, or any exact pair hash differs. Stop if Family A at size 8 is not the original development coalition. Do not drop failed conditions or change the protocol after C0.

## Inherited Task 44 metadata caveat

The committed Task 44 Markdown still contains a DRAFT status line although Task 44 completed and its decision is tagged. Task 45 records this inconsistency but does not edit, replace, or reinterpret the frozen Task 44 evidence.

## Roadmap scope

The formal roadmap still lists Tasks 46 through 54. A prior discussion proposed pivoting to XAI after Task 45, but that was a scope recommendation rather than a committed roadmap amendment. Task 45 makes no claim that later roadmap tasks are cancelled.
