# Original LFighter Matched Comparator Preflight Protocol v4.33.6

## Purpose

Add a matched comparator using the project's faithful V3.4 adaptation of the original LFighter
multiclass LFD.aggregate rule, while holding the reviewer protocol fixed.

This comparator is intentionally different from BATR-FL. It reproduces the original LFighter
decision logic already preserved in `src/lfighter_original_v34.py`.

## Frozen source identities

* `src/lfighter_original_v34.py`
  SHA256: `A2690DECAE8A4589EA4A620171AD195E2F5FE8B8E742F529BEA9685C1167D9EC`
* `scripts/run_lfighter_original_v34.py`
  SHA256: `5849B80703A15A234C39BF202573817573291FC9482DFFA1B0AE068BE79CEA3E`
* `scripts/run_lfighter_multiseed_v34.py`
  SHA256: `48B5A9409019F14427BF7E79E7672F894778E4730168AE860A513CF3144FF2BA`

## Frozen original LFighter rule

For each monitored round:

1. Compute global-minus-local updates for the final classifier weight and bias.
2. Rank classes by summed classifier-row update norm plus absolute bias-update magnitude.
3. Select the two most active classes.
4. Flatten those two classifier-row updates for each client.
5. Run `KMeans(n_clusters=2, random_state=0, n_init=10)`.
6. Compute the original cluster-dissimilarity score.
7. Select the cluster according to the original V3.4 rule.
8. Equally average the admitted local model states.
9. If there are fewer than two unique client vectors, admit all clients.
10. If the chosen admitted cluster is empty, admit all clients.

`malicious_client_ids` are passed only for diagnostic labels/metrics returned by the historical
implementation; they do not influence clustering, cluster choice, admission, or aggregation.

## Matched preflight condition

* attack: all_to_one_benign
* model seed: 1379954285
* clients: 20
* frozen warmup checkpoint: global round 4
* continuation: global rounds 5 through 8
* exact recovered Task-65 poison plan
* training and validation only
* no final diagnostic/natural test arrays materialized

## Matched local training

* model: ResMLP
* local epochs: 1
* batch size: 2048
* evaluation batch size: 4096
* learning rate: 0.0003
* weight decay: 0.0001
* maximum class weight: 4.0
* gradient clip norm: 5.0
* CPU threads: 6
* local RNG seed: model_seed + global_round * 1000 + client_id

## Outcome policy

The implementation and source identities are frozen before viewing this reviewer-run outcome.
Validation performance, malicious rejection recall, benign false-rejection rate, number of admitted
clients, selected class pair, and fallback events are descriptive outputs only and are not tuning
gates.

No attack-specific retuning and no final-test evaluation are allowed.
