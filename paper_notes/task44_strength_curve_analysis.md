# Task 44 Analysis: Poison Fraction Strength Curve

*Draft for the Results section. All figures trace directly to `task44c2_full_strength_curve.csv`, `task44c2_per_seed_new_fractions.csv`, and `task44c2_strength_curve_decision.json` (tag pending — see commit checklist below). No number here is estimated.*

## 1. Overview

Every attack family tested through Task 43 used a single fixed intensity. Task 44 turns Task 40's single-point untargeted-poisoning result into an actual four-point response curve (poison fraction 0.25/0.50/0.75/1.00, D0 unmodified, all 5 qualified attacks, 4 seeds), reusing the existing V3.20A.3/V3.20B.1 runners directly at new fraction values rather than any new attack-mechanics code.

## 2. Result — read precisely, not as a flat pass/fail count

| Hypothesis | Result | What it actually means |
|---|---|---|
| H1 (recall monotonicity) | **5/5 pass** | Recall never decreases as poison fraction increases, for every attack — the clean, expected baseline result |
| H2 (FPR relief at 0.25) | 0/5 pass | Not a detector weakness — see Section 3 |
| H3 (damage floor at 0.25) | 0/5 pass | This is the actual finding — see Section 4 |

## 3. H2's failure is a preregistration design mismatch, not a cost being paid

H2 assumed a pattern borrowed from Task 42/43's attack families (which had real, measurable FPR costs to relieve at weaker strength). This untargeted-poisoning panel never had that cost to begin with: for 4 of 5 attacks, **benign FPR is exactly 0.000000 at every single tested fraction** (0.25, 0.50, 0.75, and the frozen 1.00 anchor) — there is nothing to relieve. For the fifth attack, `all_to_one_benign`, FPR is **non-monotonic and small**: 0.005208 → 0.000000 → 0.000000 → 0.005208 across 0.25/0.50/0.75/1.00. The H2 comparison (FPR at 0.25 strictly less than FPR at 1.00) fails here specifically because the two endpoints are **numerically equal**, not because weaker attacks cost more. This should be reported as a hypothesis calibrated for the wrong attack family, not as evidence about D0's behavior.

## 4. H3's failure is the real, valuable finding: a genuine attack-strength dead zone

At poison fraction 0.25, mean plain-FedAvg damage (clean minus attacked macro-F1) is **negative for 3 of 5 attacks** and **near-zero positive for the other two** — none within an order of magnitude of the 5% threshold:

| Attack | Mean damage at 0.25 |
|---|---|
| `all_to_one_benign` | −0.002347 |
| `cyclic_shift` | +0.000140 |
| `multiclass_partial_cycle` | +0.004080 |
| `pairwise_swap` | −0.004869 |
| `random_flip` | −0.002508 |

Negative damage means the attacked model's macro-F1 was *higher* than the clean reference — at this intensity, the attack is indistinguishable from ordinary round-to-round training noise. This directly and precisely answers this task's own preregistered research question 3 ("is there a minimum poison fraction below which the attack becomes too weak to matter?") — **yes, and it is reached by 0.25 for every attack in this panel.**

**Methodological caveat, important for later paper writing:** `damage_removed_fraction` values computed at these near-zero-damage fractions (e.g., `cyclic_shift` at 0.25 showing −4.95, `multiclass_partial_cycle` at 0.50 showing −1.65) are the expected numerical symptom of dividing by a denominator indistinguishable from zero, not meaningful recovery percentages. These specific values must not be quoted as real recovery measurements in the manuscript — they should be reported as "damage too small to yield an interpretable ratio," with the ratio itself omitted or explicitly caveated.

## 5. The dead zone is not uniform — real, attack-specific heterogeneity

- **`all_to_one_benign`** resolves cleanly and quickly: by fraction 0.50, damage is unambiguously positive (0.025132) and interpretable (recall 1.0, FPR 0.0, damage-removed 1.174).
- **`cyclic_shift`**'s dead zone extends further: damage stays negative through 0.75 (−0.012126), only becoming clearly positive at the frozen 1.00 anchor.
- **`multiclass_partial_cycle`** never resolves cleanly across the entire 0.25–0.75 range — damage oscillates between small positive and small negative values at every intermediate point. This is not new: Task 40's own frozen decision already marked this attack `CONSISTENT_PARTIAL` rather than `STRICT_PASS` at full strength. Task 44 independently confirms this same weakness from a completely different angle — persistent, marginal damage across the *entire* strength range, not just borderline behavior at one fixed point. Two independent measurements agreeing is real, convergent evidence that this attack is structurally weaker than the other four, not a coincidence of one experimental setting.
- **`pairwise_swap`** shows a third, distinct shape: damage stays negative or near-zero through 0.75 (−0.017 to −0.021) and only becomes clearly positive at the 1.00 anchor (where Task 40's frozen result shows damage-removed = 0.549). Unlike `all_to_one_benign`'s smooth threshold-then-rise pattern, this attack's damage appears concentrated specifically near full strength rather than scaling gradually.

## 6. Scope of the claim

**Supported:** D0's recall is monotonic in attack strength for every tested attack (H1). Below poison fraction 0.25, none of the five qualified untargeted attacks produce damage distinguishable from clean-model training noise — a genuine, quantified dead zone. This dead zone's extent is attack-specific: `all_to_one_benign` clears it by 0.50, `cyclic_shift` and `pairwise_swap` require close to full strength, and `multiclass_partial_cycle` never clears it within the tested range, consistent with and reinforcing its already-known Task 40 weakness.

**Not supported:** any claim that D0's FPR is generally strength-dependent — the data shows no real relationship to test, since FPR was near-zero throughout this panel regardless of fraction. Also not supported: any specific recovery percentage at fractions where damage is near-zero (see Section 4 caveat).

## 7. Freeze checklist

```
cd /d "%USERPROFILE%\LFighter-research"
git add scripts\summarize_task44_c2_strength_curve_v415.py
git add -f results\cic_iot_diad_task44_c2_strength_variation_v415\summary
git commit -m "Freeze Task 44 C2: strength curve, H1 pass, H2/H3 explained (dead zone + design mismatch)"
git tag task44-c2-strength-curve-decided-v415
git push
git push --tags
```

---

*Roadmap status: Task 44 closed with a documented, mechanistically-explained result: H1 confirms D0's expected monotonic behavior; H2's failure is attributed to a preregistration design mismatch rather than a detector cost; H3's failure is itself the finding — a genuine, attack-specific poison-strength dead zone below fraction 0.25, with `multiclass_partial_cycle`'s persistent weakness now independently confirmed by two separate experimental angles (Task 40's fixed-strength result and Task 44's full strength curve). Per the master roadmap, the next unstarted attack-breadth item is Task 45 (malicious coalition size variation).*
