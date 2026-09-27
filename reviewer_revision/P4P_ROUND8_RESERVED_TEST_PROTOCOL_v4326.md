# P4P Round 8 Reserved Test Evaluation Protocol v4.32.6

## Purpose

Evaluate the already frozen matched P4P comparator checkpoints on the same reserved
diagnostic and natural test splits used by the Task 65 final evaluation.

This stage performs evaluation only. It does not train a model, choose a checkpoint,
change a defense parameter, or rerun any P4P federated branch.

## Frozen endpoint and condition grid

The endpoint is global round 8 for every condition.

Five final seeds:

1379954285, 1886033230, 480705558, 1377035733, 1707771978

Five attacks:

all_to_one_benign, cyclic_shift, multiclass_partial_cycle, pairwise_swap, random_flip

The already audited first P4P preflight condition is read from its preserved preflight
directory. The remaining 24 conditions are read from the frozen full-grid directory.

## Evaluation implementation

The P4P runner built the model with:

build_model("resmlp", X_train.shape[1], NUM_CLASSES)

and every saved P4P checkpoint contains `model_state_dict`.

The reserved-test evaluator therefore builds the same `resmlp` architecture and loads
only the frozen round 8 `model_state_dict`.

Metrics are computed through the existing `federated_iot_v26.evaluation_artifacts`
implementation. This preserves the project's established definitions of accuracy,
balanced accuracy, macro F1, weighted F1, MCC, log loss, 15-bin ECE, class metrics,
and confusion matrices.

No new metric definition is introduced for the primary comparison.

## Test access chronology

Before this protocol was frozen, a diagnostic inspection command opened the protocol
NPZ and printed only array keys, shapes, and dtypes for the train, validation,
diagnostic-test, and natural-test arrays.

That structural inspection materialized the arrays in Python, so the test arrays are
not described as previously unopened or untouched in this reviewer stage.

However:

1. no P4P test prediction was produced,
2. no P4P diagnostic or natural test metric was calculated,
3. no P4P checkpoint was selected from test performance,
4. global round 8 had already been fixed as the endpoint,
5. all P4P detector and aggregation parameters had already been frozen,
6. the complete 5 attack by 5 seed P4P training grid had already been frozen.

This v4.32.6 run is therefore the first P4P outcome evaluation on the reserved test
splits, but not the first structural access to their array objects.

## One-shot rule

The evaluator refuses to run if its output directory or completion marker already
exists. All 25 round 8 checkpoints are evaluated in one pass.

No best-round selection is permitted. No condition may be excluded because of its
result. Negative results are retained.

## Outputs

The evaluator saves:

1. one metric row per dataset, attack, and seed,
2. one class metric row per dataset, attack, seed, and class,
3. one confusion row per dataset, attack, seed, true class, and predicted class,
4. the access-start record,
5. the completion record with source and output hashes.

Expected cardinalities:

* 50 metric rows,
* 400 per-class rows,
* 3200 confusion rows.

Formal cross-method statistical testing is not performed in this stage. It is reserved
for the subsequent matched Plain FedAvg versus BATR FL versus P4P analysis.
