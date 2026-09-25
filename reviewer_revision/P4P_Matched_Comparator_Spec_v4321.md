# P4P Matched Comparator Specification for BATR FL Reviewer Revision v4.32.1

## Source

Thai Tuan Khang, Tran Huu Duc, Dang Van Huynh, Van-Hau Pham, and Phan The Duy, "P4P: A probe-guided anti-poisoning defense for federated learning-based intrusion detection in IoT networks under non-IID data," Journal of Network and Computer Applications, 251, 104502, 2026.

## Goal

Implement P4P as a matched external comparator inside the frozen BATR FL experimental environment. Preserve the P4P defense mechanism, but do not import its unrelated model architecture, optimizer, training length, or malicious-fraction settings.

## Published P4P defense

### Probe
`v_t = W_t - W_{t-1}`

### Client probe response
`r_i^t = <delta_W_i^t, v_t> / (||delta_W_i^t||_2 ||v_t||_2)`

### Magnitude filter
Use client update L2 norms with a Median Absolute Deviation acceptance interval.

Published coefficient:
`k = 3.0`

Matched deterministic implementation:
`median_norm = median(norms)`
`mad = median(abs(norms - median_norm))`
`T_low = median_norm - 3.0 * mad`
`T_high = median_norm + 3.0 * mad`
Accept when `T_low <= norm_i <= T_high`.

### Ensemble anomaly detector
Operate on the 1D probe responses of clients that passed the magnitude filter.

K-Means:
- clusters = 2
- minority cluster gets one suspicious vote
- equal-size tie rule for this matched implementation: lower-mean-cosine cluster is suspicious

DBSCAN:
- eps = 0.5
- minPts = 5
- noise points get one suspicious vote

Isolation Forest:
- contamination = 0.1
- semantically classified outliers get one suspicious vote

Final current-round anomaly:
- suspicious if votes >= 2

### Temporal suspicion
`score_i_t = max(0, score_i_prev + delta_i_t - 0.2)`

where `delta_i_t = 1` for a current-round anomaly and 0 otherwise.

Permanent-removal threshold:
`theta = 2.0`

Initialize reviewer-comparison suspicion scores to 0.

### Trusted-set rule
A current update is eligible only when:
- it passes magnitude filtering
- it is not a current-round ensemble anomaly
- its temporal suspicion score is <= 2.0

### Aggregation
Use FedAvg on only the trusted updates.

Inside the matched BATR FL harness, preserve the original BATR FL sample-count weights for those trusted updates and renormalize over the retained set.

## Matched chronology

BATR FL warmup provides checkpoints through rounds 1 to 4.

For the first monitored round:
`probe = W_round4 - W_round3`

After trusted-set aggregation produces the next global checkpoint, the next P4P probe is the difference between the newest two global checkpoints.

This avoids any undefined first-round probe without adding a new warmup stage.

## What remains fixed from BATR FL

- CIC IoT DIAD 2024 processed dataset
- same 20 logical clients
- same Dirichlet alpha 0.5 partition
- same exact client data
- same warmup checkpoint
- same local model architecture
- same optimizer and training hyperparameters
- same attacks
- same attack identities and poison fraction
- same random seeds
- same monitored rounds
- same evaluation sets and metrics
- same hardware for matched timing
- same test-access chronology

## What is intentionally NOT copied from the P4P paper

The P4P paper's own experimental settings are not part of the matched comparator:
- 15 total rounds
- 5 local epochs
- batch size 32
- SGD learning rate 0.01
- 30% malicious clients
- random seed 42
- its own 128/64 MLP
- its own 30-feature preprocessing

Copying those would destroy experimental matching.

## Required preflight before any attack outcome run

1. Unit-test MAD filtering on hand-computed vectors.
2. Unit-test probe cosine responses.
3. Unit-test K-Means vote, DBSCAN vote, Isolation Forest vote and 2-of-3 majority.
4. Unit-test temporal suspicion recurrence and permanent-removal threshold.
5. Unit-test trusted-set FedAvg against a hand-computed weighted example.
6. Verify no final test arrays are materialized during implementation/preflight.
7. Run clean branch only and verify deterministic completion, output schema and timing instrumentation.
8. Freeze source hashes, config and Git tag.
9. Only then run matched attack outcomes.

## Important paper ambiguities handled explicitly

1. Isolation Forest label sign differs across APIs. The implementation follows the semantic meaning "outlier", not a raw numeric label convention.
2. The paper gives K-Means minority-cluster logic but no equal-size tie rule. The reviewer implementation uses lower mean cosine only as a deterministic tie breaker.
3. The paper describes P4P in places as dynamically weighting clients, but Algorithm 2 and Eq. 9 use binary inclusion/exclusion. The comparator follows Algorithm 2 and Eq. 9.
4. The paper discusses a probe broadcast, while the mathematical defense computes cosine server-side from received updates. The matched implementation performs the computation server-side and adds no client-side defense computation.
5. The paper does not specify a first-round probe initializer. BATR FL's existing warmup makes this unnecessary because round-3 and round-4 global checkpoints already exist.

## Status

Comparator specification complete.

Do not run attack outcomes until this specification and the reviewer preregistration are committed and tagged.
