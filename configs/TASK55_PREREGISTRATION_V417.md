# Task 55 preregistration: post detection SHAP and XAI pipeline

Version 4.17.0

Parent freeze: `task45-c4-coalition-size-partial-v4164` at `9c8a7dc107191e153e1e7eff2062546c3c7de633`.

## Scientific role

Task 55 implements feature level forensic explanation after the deterministic LFighter detector and reconstruction decision. XAI has no authority to flag clients, change thresholds, modify trust, replace updates, or control aggregation. No model training or method reopening is permitted.

The pipeline explains the 69 standardized behavioral network traffic inputs used by the residual MLP. It does not explain detector transition signature features, client descriptor vectors, identifiers, ports, protocol, labels, or capture metadata.

## Leakage control

Only `X_train`, `y_train`, `X_val`, and `y_val` may be materialized. `X_test_natural`, `y_test_natural`, `X_test_diagnostic`, and `y_test_diagnostic` remain reserved and forbidden. The initial XAI study is development and validation only. A final test gate is not opened by Task 55.

## Frozen explainer

The primary explainer is `shap.GradientExplainer` from SHAP 0.48.0. This release supports the frozen Python 3.10 environment. The expected gradients method is selected because the frozen PyTorch residual MLP is differentiable and contains LayerNorm, GELU, residual blocks, and dropout. All models run in evaluation mode.

Attributions are computed on logits. The primary output is the DDoS source logit minus the Benign target logit. Secondary outputs explain the DDoS source logit and Benign target logit separately. Expected gradients use 128 samples, batch size 64, no local smoothing, and random seed 5517.

## Frozen background and probe

The background is selected without replacement from training data using 16 examples per class and seed 5501, for 128 rows total. The probe is selected without replacement from validation data using 16 examples per class and seed 5502, also 128 rows total. The identical background and probe indices are used for every model state and seed.

Primary attack transition analysis uses true DDoS probe rows. Rows from all eight classes are retained for class specificity diagnostics and Task 56 validation.

## Frozen state comparison

Every unit compares three paired states:

1. Clean reference: the seed matched common round 4 warmup checkpoint.
2. Suspicious: the final plain attack checkpoint from Task 45.
3. Reconstructed: the final trusted reconstruction checkpoint from Task 45.

The primary panel is Task 45 coalition size 10 across all three frozen coalition families and seeds 7, 99, 123, and 2026. This produces 12 paired conditions and 36 state evaluations. Seed 7 is the development pilot; seeds 99, 123, and 2026 remain confirmatory for the frozen implementation.

## Stored evidence

The pipeline stores immutable background and probe index manifests, float32 compressed attribution tensors, long form Parquet records, CSV summaries, checkpoint hashes, package versions, runtime, and JSON decisions. Every attribution record is keyed by family, size, seed, state, probe index, class, feature, checkpoint hash, and explainer provenance.

Task 56 will separately validate repeated run stability, rank correlation, perturbation faithfulness, class specificity, background sensitivity, probe size sensitivity, cross seed consistency, runtime, and memory. Task 57 will evaluate whether reconstruction moves attributions toward the paired clean reference. Task 58 will create figures and non cherry picked case studies only after those gates close.

## Stop rules

Stop immediately if a reserved array is requested, a feature name or count mismatch occurs, a checkpoint is absent or incompatible, the paired Task 45 evidence is incomplete, the pinned SHAP version is unavailable, output shapes are inconsistent, or reproducibility checks fail. Negative and boundary results are retained.
