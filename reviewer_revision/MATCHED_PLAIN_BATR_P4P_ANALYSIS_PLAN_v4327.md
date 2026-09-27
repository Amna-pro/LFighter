# Matched Plain FedAvg vs BATR FL vs P4P Analysis Plan v4.32.7

## Scope

This is a post outcome analysis of already frozen round 8 test evidence.

No model training, checkpoint evaluation, test inference, defense retuning, baseline retuning,
checkpoint selection, attack regeneration, or seed selection is permitted in this stage.

The comparison uses the same five final seeds, five attacks, global round 8, and the same
diagnostic and natural test splits.

## Sources

BATR FL and attacked Plain FedAvg values come from the frozen Task 65 and Task 66 evidence.

The Task 66 seed-level table is used as the primary bridge:

* `reported_level` is the defended BATR FL level.
* `seed_level_contrast` is BATR FL minus attacked Plain FedAvg.
* attacked Plain FedAvg is therefore `reported_level - seed_level_contrast`.

These identities are independently checked against Task 65
`raw_checkpoint_metrics.csv` at global round 8.

P4P values come from the frozen v4.32.6 round 8 reserved-test metrics.

## Primary metrics

The matched inferential comparison is restricted to the same two primary utility metrics
already used in Task 66:

1. macro F1
2. balanced accuracy

Higher values are better for both metrics.

## Comparisons

For each dataset, attack, metric, and seed:

1. BATR FL minus attacked Plain FedAvg
2. P4P minus attacked Plain FedAvg
3. BATR FL minus P4P

Positive contrast means the first named method has the larger metric value.

The original Task 66 BATR FL minus Plain FedAvg inferential rows are preserved rather than
replaced. The two new P4P-related comparisons use the same inferential family:

* paired mean contrast
* paired median contrast
* deterministic 20,000-replicate paired bootstrap percentile 95 percent CI
* exact two-sided sign-flip test across five paired seeds
* paired Cohen dz
* positive seed count
* Holm adjustment across the five attacks separately for each dataset, metric, and comparison

With five paired seeds, the minimum attainable nonzero exact two-sided sign-flip p value is
0.0625. Therefore no p below 0.05 is possible with this cohort. Effect sizes, directions,
confidence intervals, and consistency across seeds must be reported without converting this
resolution limit into a claim of statistical significance.

## Claim boundary

This analysis establishes a matched comparison among Plain FedAvg, BATR FL, and P4P only.

It does not establish superiority over all robust FL defenses. Additional matched baselines and
reconstruction ablations remain required by the reviewer.

Negative and mixed results must be retained.
