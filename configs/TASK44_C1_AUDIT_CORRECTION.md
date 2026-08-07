# Task 44 C1 Audit Correction

**Does not modify or replace `task44c1_anchor_audit_decision.json` — that file remains as originally frozen. This is a separate, dated correction, matching the same pattern used for Task 41C's H2 methodology correction.**

## What the original C1 audit got wrong

`audit_task44_c1_strength_anchor_v415.py`'s automated check only searched for lines containing both "poison" and "fraction" and counted matches as evidence the parameter was a genuine, working, variable input. This was an insufficient check: it detected the parameter's *existence* but not whether it was actually enforced to a single fixed value.

## What manual inspection of the real source confirms

`run_exact_untargeted_qualification_grid_v320a3.py`, lines 70–73:

```python
if abs(a.poison_fraction - 1.0) > 1e-12:
    raise ValueError(
        "V3.20A.3 is frozen to poison fraction 1.0"
    )
```

This is a hard, unconditional guard. The script does not merely default to 1.0 — it actively refuses to execute at any other value, immediately raising an error before any other logic runs. This is a deliberate safety lock protecting the frozen V3.20A.3 methodology from being reopened, consistent with this project's broader discipline (matching guards seen elsewhere, e.g. `V3.20B is frozen to poison fraction 1.0` in the sibling defense-grid script).

## Corrected conclusion

Task 44's C2 stage **cannot** reuse the existing `run_exact_untargeted_qualification_grid_v320a3.py` / `run_frozen_untargeted_defense_grid_v320b1.py` scripts directly at new poison-fraction values, since both actively refuse to run at anything other than 1.0. Modifying these frozen scripts is not an option under this project's standing rule against altering frozen evidence.

**Genuinely new code is required** for Task 44's C2 stage. The next audit step, before writing that code, is confirming exactly where the reusable *attack-construction logic itself* lives (the actual label-flip / poisoning-plan generation functions), separate from the CLI-level guard — if that logic lives in an importable `src/` module as pure functions, the new C2 script may be able to reuse those functions directly (bypassing the grid-orchestration script's guard entirely, since the guard sits in argument validation, not in the poisoning logic itself), rather than reimplementing untargeted poisoning from scratch. This has not yet been confirmed and should not be assumed.
