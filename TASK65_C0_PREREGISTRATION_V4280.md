# Task 65 C0 Preflight V4.28.0

This is a static, outcome-blind preflight only. It does not run training and does not open the processed NPZ container.

It verifies the frozen Task 63 C2 evaluation manifest, the frozen Task 64 C2 replacement seed set, the exact five attack names, the eight malicious clients, poison fraction 1.0, round 8 primary endpoint, and the presence/source fingerprints of the already-frozen experiment runners.

The preflight also checks whether the frozen training runners call `load_protocol_arrays`. If they do, Task 65 C1 must use an explicit train/validation isolation wrapper so that final natural/diagnostic arrays are not inspected during training. Final test arrays are to be loaded only by the explicit final checkpoint evaluation stage.

No Task 65 C1 execution is authorized by this package.
