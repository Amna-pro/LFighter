# BATR-FL v4.34.0f Ablation Audit Correction a1

## Why this correction exists

The v4.34.0f warmup recovery and both mitigation-ablation runs completed. The original final audit
then failed only on `guard_no_reserved_test_tokens`.

That check was too strict: the Task-65 guard log intentionally records the *names* of the reserved
test arrays under `test_array_names_verified_only` while explicitly recording
`test_arrays_materialized: false`. The guard implementation materializes only
`X_train`, `y_train`, `X_val`, and `y_val`.

Therefore, the presence of strings such as `X_test_natural` in the JSONL log is not evidence of test
access. It is evidence that the guard verified the archive schema without materializing those arrays.

This correction changes only the audit interpretation. It does not rerun warmup, center-only,
hard-rejection, local training, attack generation, detection, aggregation, or evaluation.

## Frozen evidence being audited

Parent HEAD/tag before the correction:
- `34772df` / `reviewer-v4340f-warmup-calibration-recovery-verified`

Original frozen audit script SHA256:
- `9D9D52B3B16B06CDD2FAC5BE18578C2DB2D3BCF75439ABE79B9287E9F75C3198`

Observed guard log SHA256:
- `67909A8B30ADF3E3332FD58A754364312DF4B0B66830AECEA41839EDCFF3C61C`

## Correct semantic guard rule

The corrected audit requires all of the following:

1. At least one `guarded_load_protocol_arrays` event exists.
2. Every record carrying `test_arrays_materialized` records it as `false`.
3. No reserved test-array key occurs in any `materialized_keys` list.
4. For guarded protocol-array loads, `materialized_keys` are a subset of:
   `X_train`, `y_train`, `X_val`, `y_val`.
5. For guarded protocol-array loads, `test_array_names_verified_only` contains exactly:
   `X_test_natural`, `y_test_natural`, `X_test_diagnostic`, `y_test_diagnostic`.

The original token-presence rule is removed because it confuses name verification with array
materialization.

## Scientific status

No scientific setting is changed.
No attack or model run is repeated.
No result is selected or discarded based on performance.
No reserved final-test array is permitted to be materialized.
