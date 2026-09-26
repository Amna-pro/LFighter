# Reviewer Protocol Amendment C2 v4.32.3

## Status

Pre outcome correction. This amendment is required because the historical Task65 final seed warmup files are no longer present in the recovered repository, ZIP archive, RAR search, Git object history, or preservation folders examined during the reviewer revision.

No new reviewer attack outcome was opened before this correction.

## What was recovered and verified

The CIC IoT DIAD behavioral array was recovered with SHA256 `1475d26cbddfcaf874c5f00ad83d027b1de529e06be6bf0c330d89abe26752fd`.

The federated partition file was recovered with raw SHA256 `417607da09e1d1b1108a48dedc5125b2ad543f67facdaf9a98c91b89de792c7c` and logical partition SHA256 `5f14674b0c775f06c3400c5e1e3b9649b78c82d4257c2293411f5c6fe53bed48` for 20 clients and 483734 training rows.

Archived Task65 execution logs preserve, for every original final seed, the exact model seed, partition hash, probe hash, four warmup validation macro F1 values rounded to four decimals, four DDoS to Benign rates rounded to four decimal percent, frozen EMA threshold rounded to six decimals, clean calibration FPR, and profile SHA256.

For seed 1379954285 the archived original reconstruction calibration additionally recorded an independent four round warmup replay with maximum absolute model state difference `0`, relative state L2 difference `0`, maximum round macro F1 difference `0`, maximum source target difference `2.77555756156e-17`, and replay equivalence `True` against the historical checkpoint while that checkpoint existed.

## Correction to C1

C1 required a newly replayed round4 state to be compared directly to the historical Task65 round4 checkpoint. That file cannot now be recovered. The direct file comparison requirement is therefore replaced by a historical fingerprint and independent replay verification procedure.

This is a provenance recovery procedure, not an outcome based model change.

## Frozen recovery procedure

For each original Task65 final seed:

1. Use only the verified train and validation arrays. Do not materialize either reserved test array.
2. Use the verified 20 client logical partition.
3. Use the original frozen V3.10 model, optimizer, class weight, gradient clipping, local epoch, batch size, learning rate, weight decay, probe size, probe seed, EMA decay, and clean threshold quantile.
4. Replay trusted clean warmup rounds 1 through 4 and capture W3 and W4.
5. Reconstruct the same 8 by 8 client probe signatures and clean calibration artifacts needed to compute the historical warmup fingerprints.
6. Require exact equality of partition hash and probe hash.
7. Require the newly generated profile SHA256 to equal the archived Task65 profile SHA256 for that seed.
8. Require the frozen EMA threshold rounded to six decimals to equal the archived value.
9. Require clean calibration FPR to equal `0.0375` within `1e-12`.
10. Require every warmup validation macro F1 rounded to four decimals to equal the archived value.
11. Require every DDoS to Benign rate expressed as percent and rounded to four decimals to equal the archived value.
12. Perform a second independent deterministic warmup replay and require the second W4 to match the first W4 with maximum absolute state difference at most `1e-7`.
13. Save W3 only after every historical fingerprint and independent W4 replay check passes.
14. Record the reconstructed W4 checkpoint hash and the evidence checks in a pre outcome JSON marker.
15. Abort on any mismatch. Do not relax tolerances or change hyperparameters after seeing a mismatch.

## Interpretation

A reconstructed W4 is not described as the original retained Task65 checkpoint. It is described as a deterministic pre outcome reconstruction of the historical Task65 warmup state, validated against archived historical fingerprints and independent replay evidence.

The original Task65 results remain historical evidence and are not rerun or replaced. The reconstructed warmup exists only to initialize matched reviewer comparators that require a W3 to W4 trajectory.

## Seed metadata for reviewer comparators

Reviewer specific seed metadata may be generated under a new reviewer results directory. Such records must contain the original final seed, the verified logical partition hash, the provenance label `reviewer_reconstructed_task65_warmup_v4323`, and `test_sets_accessed = false`. These records are not represented as the original Task65 seed metadata files.

## Outcome boundary

No poisoned reviewer branch may run until this amendment, the historical fingerprint file, and the v4.32.3 bootstrap implementation are committed and tagged.
