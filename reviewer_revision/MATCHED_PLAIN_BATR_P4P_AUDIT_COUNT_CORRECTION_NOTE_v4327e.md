# Matched Plain FedAvg vs BATR FL vs P4P Audit Count Correction v4.32.7e

The v4.32.7 matched comparison results were already written before its final audit stopped.
The v4.32.7b audit then independently checked those existing results but used an incorrect
expected count for the new P4P-related statistical field checks.

The correct count is:

2 comparisons x 2 metrics x 2 datasets x 5 attacks x 11 checked fields = 440.

The attempted v4.32.7c and v4.32.7d launchers stopped before modifying the repository.
v4.32.7c failed in an inline Python command because CMD delayed expansion altered the command.
v4.32.7d failed while attempting to transform the audit script at runtime.

This v4.32.7e package avoids runtime source transformation. It contains the corrected audit
script directly. The only scientific-code difference from v4.32.7b is the audit predicate:

new_p4p_statistics_checks_exact_220 = (nc == 220)

becomes

new_p4p_statistics_checks_exact_440 = (nc == 440)

The original v4.32.7 result CSVs are verified by SHA256 before the audit, are not rerun, and
are not overwritten. No model training, test inference, parameter retuning, attack regeneration,
checkpoint selection, or statistical-formula change occurs.
