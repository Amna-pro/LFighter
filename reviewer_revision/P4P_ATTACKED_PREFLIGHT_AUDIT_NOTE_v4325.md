# P4P Attacked Preflight Audit Note v4.32.5

The first matched attacked P4P condition was executed only after the complete 5 attack by
5 final seed Task 65 attack-manifest recovery was frozen.

Condition:

* attack: all_to_one_benign
* model seed: 1379954285
* clients: 20
* malicious clients: 1, 7, 8, 10, 14, 15, 17, 18
* monitored rounds: 5 through 8

This audit does not rerun the experiment. It verifies frozen partition identity, poison
plan identity, warmup checkpoint identity, P4P configuration, exact poisoned indices and
replacement labels, client-decision coverage, summary arithmetic, and output hashes.

The observed result is preserved without parameter tuning. No conclusion about overall
P4P performance is drawn from this single seed. Scaling to the full matched P4P grid is
allowed only after this audit passes and is frozen.
