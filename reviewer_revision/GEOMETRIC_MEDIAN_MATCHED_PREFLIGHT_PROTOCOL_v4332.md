# Geometric Median Matched Comparator Preflight Protocol v4.33.2

Purpose: add a matched geometric-median robust aggregation comparator under the same frozen reviewer conditions used for Plain FedAvg, BATR-FL, P4P, coordinate-wise median, and Multi-Krum.

This implementation uses the project's existing deterministic Weiszfeld geometric median from `scripts/audit_robust_centers_v319a.py`. It is a geometric-median / RFA-family baseline; report the implemented arm as **geometric median** unless a separate published RFA parameterization is later reproduced exactly.

Preflight condition:
* attack: all_to_one_benign
* model seed: 1379954285
* clients: 20
* warmup checkpoint: global round 4
* continuation: rounds 5 through 8
* exact recovered Task-65 poison plan
* train + validation only; no final diagnostic/natural test arrays

Frozen historical algorithm:
* source SHA256 A8BDA1CB3BAEBB97A00CBBBE86F190D12FF7185A07A08A622CCF52BCD38DC76E
* maximum iterations: 60
* tolerance: 1e-7
* near-point threshold: 1e-12
* initialization: NumPy coordinate-wise median
* geometry dtype: float64
* equal-client geometric-median objective
* relative stopping rule: step <= tolerance * max(||center||_2, 1.0)

The adapter flattens every floating model update using sorted update keys, applies the historical function without modifying it, and maps the center back to the same tensor shapes.

Matched local training remains unchanged: ResMLP, 20 clients, one local epoch, batch 2048, evaluation batch 4096, lr 3e-4, weight decay 1e-4, max class weight 4.0, gradient clipping 5.0, six CPU threads, and local RNG seed model_seed + global_round*1000 + client_id.

No detector, client rejection, reconstruction, sample-count weighting, attack-specific tuning, or test-set access is allowed.

Preflight validity is implementation-only. Performance is not a gate. Convergence is recorded but is also not a selection gate; if the frozen 60-iteration solver does not satisfy the stopping tolerance, its final iterate and non-convergence status are retained without retuning.
