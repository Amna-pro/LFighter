# Reviewer Protocol Amendment C3 v4.32.4

## Purpose

The original Task 65 plain attack manifest binaries are not available in the recovered working directories or searched ZIP archives. The exact Task 65 attack-construction source is, however, preserved in Git history and in the Task 65 preregistration.

A Windows checkout may store tracked Python files with CRLF line endings even though the Git object and preregistered source hash use LF bytes. Therefore raw working-tree SHA256 is not used as the scientific source identity. Source identity is established by both:

1. Git content equivalence to Task 65 commit `2e40296`, and
2. SHA256 after CRLF to LF normalization.

The frozen Task 65 plain source normalizes to:
`c579d04815a7e7a72a934926c8092056e83714702322a38a97f2b3b4dab38b75`.

The historical V3.20B.1 exact-plan loader at the same Task 65 commit normalizes to:
`9103a41b8152034be2745cd9fbcdbc320b2d935023ceb93eb7b06d93cf7e5b33`.

## Frozen recovery rule

1. The recovery uses the exact Git-equivalent Task 65 attack constructor in `scripts/run_exact_untargeted_plain_v320a3.py`.
2. The exact-plan loader in `scripts/run_frozen_untargeted_defense_v320b1.py` must also remain Git-equivalent to Task 65 commit `2e40296`.
3. The recovered CIC IoT DIAD array file must match the frozen raw SHA256.
4. The recovered partition must match the frozen logical partition SHA256.
5. Only `y_train` may be materialized from the protocol NPZ during manifest recovery. Natural and diagnostic test arrays must not be materialized.
6. For each recovered condition, `attack_seed` equals `model_seed`, poison fraction is 1.0, the primary malicious coalition is [1, 7, 8, 10, 14, 15, 17, 18], and the attack mapping is the requested frozen V3.20A.3 attack type.
7. The frozen attack constructor is executed twice independently in memory. The poison digest, selected local positions, replacement labels, and manifest table must agree exactly.
8. The recovered files must pass the existing frozen exact Task 65 poison-plan loader without any outcome-based adjustment.
9. Recovered outputs are described as deterministic reviewer reconstructions of the historical attack manifests, not as recovered historical binaries.
10. This recovery workflow does not execute local model training, P4P attack evaluation, BATR FL attack evaluation, natural test evaluation, or diagnostic test evaluation.
11. No defense parameter, attack mapping, threshold, seed, coalition, or poisoning fraction may be changed in response to later outcomes.

## Initial preflight condition

The first recovery condition is model seed 1379954285, attack seed 1379954285, attack type `random_flip`, 20 clients, malicious clients [1, 7, 8, 10, 14, 15, 17, 18], and poison fraction 1.0.

Only after this condition passes deterministic reconstruction and exact-loader validation may the remaining frozen Task 65 manifest conditions be recovered.

## Evidence boundary

Passing this preflight establishes deterministic reconstruction compatibility and provenance for the attack plan. It is not an attack-performance result.
