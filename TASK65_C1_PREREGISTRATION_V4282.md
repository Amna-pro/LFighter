# Task 65 C1 Corrected Implementation Freeze V4.28.2

This stage freezes the executable Task 65 one-shot implementation before final data access.

Training child processes are guarded so that the generic protocol loader materializes only train/validation arrays. Round 5–8 checkpoints are captured from both frozen historical checkpoint names without changing model state or training control flow.

All incomplete training stages may be deterministically restarted only before the final-access marker exists. Once final access starts, no further training, retuning, seed substitution, best-round selection, attack dropping, or rerun is permitted.

The final evaluator is the only component permitted to materialize the natural and diagnostic arrays.

The Task 63 clean-utility numerical thresholds are retained unchanged. Because the exact frozen Task 65 comparison arms do not contain a separate clean-defense arm, no new clean-defense arm is introduced at the final gate; those clean-utility limits remain carried-forward development safety constraints rather than a newly recomputed reserved-test metric.
