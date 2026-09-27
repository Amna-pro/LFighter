# Reviewer Recovery Correction D v4.32.4d

## Scope

The first full 5 attack by 5 seed manifest recovery attempt stopped on the first new
condition, all_to_one_benign with model seed 1379954285, during post-write validation
through the frozen V3.20B.1 exact poison-plan loader.

No local model training, P4P attack evaluation, BATR FL attack evaluation, natural-test
evaluation, or diagnostic-test evaluation was run.

## Cause

The recovery constructor and loader agreed on the attack plan, selected indices, labels,
integer counts, and poison digest. The failure was caused only by an exact pandas
DataFrame comparison of two derived floating-rate columns after a CSV write/read
round trip.

Examples in the failed assertion differed only at approximately 1e-16, such as:

0.916271406460632 versus 0.9162714064606321

This is serialization-level floating-point representation, not a change in attack
membership, poisoning counts, labels, seeds, coalition, mapping, or poison fraction.

## Correction

The recovery implementation is corrected only in its validation layer.

1. Structural and discrete manifest columns remain exact.
2. Poisoned local positions and replacement labels remain exact array comparisons.
3. The poison index SHA256 must remain exact.
4. The frozen exact loader must still accept the reconstructed manifest.
5. Derived floating-rate columns are checked with zero relative tolerance and an
   absolute tolerance of 5e-15, and are independently recomputed from the exact integer
   numerator and denominator columns.
6. No scientific parameter is changed.
7. The failed partial output is hashed and recorded before it is removed.
8. Only the known incomplete all_to_one_benign seed 1379954285 recovery directory may
   be removed automatically, and only if the corresponding PASS evidence file does not
   exist and no model/checkpoint file is present.

The grid remains a manifest-recovery operation only. Attacked P4P remains blocked until
all 25 manifest conditions pass.
