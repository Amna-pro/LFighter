from __future__ import annotations

import re

NEGATION_CUES = re.compile(
    r"\b(not|without|never|no evidence of|does not|do not|"
    r"distinct from|different from|separate from|rather than)\b",
    re.I,
)
ORACLE_TRIGGER = re.compile(
    r"\b(?:achiev(?:e|ed|ing)|restor(?:e|ed|ing)|confirm(?:s|ed|ing)?|"
    r"demonstrat(?:e|ed|ing))\s+(?:an?\s+|the\s+)?(?:exact\s+)?oracle[\s-]clean",
    re.I,
)
# Also catch the noun-phrase form making a positive claim without any verb,
# e.g. a hypothetical "Full oracle clean restoration was achieved here."
# -- kept separate so both forms are checked, still sentence-scoped.
ORACLE_MENTION = re.compile(r"oracle[\s-]clean", re.I)


def split_sentences(text: str) -> list[str]:
    # Deliberately simple -- these are short, single-paragraph LLM summaries
    # with no abbreviations that would break a period-based split.
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def count_unsupported_oracle_claims(text: str) -> int:
    """Sentence-scoped: an oracle-clean mention only counts as an unsupported
    claim if that same sentence has NO negation cue anywhere in it.
    Fixes the old adjacency-only check, which missed negation across a
    coordinating "or" (e.g. "does not establish X or confirm oracle clean Y").
    """
    violations = 0
    for sentence in split_sentences(text):
        if ORACLE_MENTION.search(sentence) and not NEGATION_CUES.search(sentence):
            violations += 1
    return violations


# ---- Test corpus: all 30 real texts from the actual Task 62 C2 run that
# contain "oracle", copied verbatim from the run output. ----
_REAL_TEXTS = [
    "Observed attack effects were substantial, while defended processing and attribution reconstruction showed strong recovery toward paired preattack clean reference behavior. The findings do not establish malicious intent or oracle clean restoration. Evidence is limited to development and validation settings, and reserved test arrays remain closed.",
    "The case documents targeted suppression of attack predictions toward the Benign class, with substantial effect across the reported attack condition and strong recovery in the defended condition. Reconstruction shifted the attribution profile toward a paired preattack clean reference. The findings do not establish malicious intent or oracle clean restoration, and are limited to development and validation because reserved test arrays remain closed.",
    "The case shows a substantial targeted suppression effect on DDoS predictions toward the Benign class, alongside strong recovery in the defended analysis. Reconstruction moved the attribution profile toward the paired preattack clean reference. These findings do not establish malicious intent or oracle clean restoration. Evidence is limited to development and validation, while reserved test arrays remain closed.",
    "The review describes targeted suppression of DDoS predictions toward the Benign class, with separate attack, paired clean, and defended analyses reported elsewhere. Reconstruction shifted the attribution profile toward the paired preattack clean reference, indicating strong recovery within the observed scope. This does not establish malicious intent or oracle clean restoration. Evidence remains limited to development and validation, while reserved test arrays remain closed.",
    "The forensic review found targeted suppression of DDoS predictions toward the Benign class in the examined attack condition. Recovery measures substantially improved the attribution profile relative to the affected state, moving it toward the paired preattack clean reference. The findings are limited to development and validation evidence; reserved test arrays remain unavailable. The reconstruction does not establish malicious intent or confirm oracle clean restoration.",
    "The assessment describes a moderate attack effect alongside strong recovery in the reported reconstruction. Attribution moved toward a paired preattack clean reference, while the findings do not establish malicious intent or oracle clean restoration. Evidence remains limited to development and validation materials, and reserved test arrays were not examined. The separate report records the underlying branches, detector results, and attribution details.",
    "The assessment indicates a moderate attack effect involving suppression of predictions toward the benign class, followed by strong recovery after reconstruction. Attribution moved toward the paired preattack clean reference, supporting recovery of the profile without establishing malicious intent or oracle clean restoration. Findings are limited to development and validation evidence; reserved test arrays remain closed.",
    "The assessment describes a moderate effect on prediction behavior with strong recovery in the reconstructed attribution profile toward the paired preattack reference. The available evidence does not establish malicious intent or oracle-clean restoration. Findings are limited to development and validation materials, while reserved test arrays remain closed; consequently, the reported recovery should be interpreted within those stated evidentiary bounds.",
    "The assessment documents a substantial attack effect that suppresses DDoS predictions toward the Benign class, while recovery remains weak. Reconstruction shifted the attribution profile toward a paired preattack clean reference, but this does not establish malicious intent or oracle clean restoration. Findings are limited to development and validation evidence, and reserved test arrays remain closed.",
    "The assessment describes a limited attack effect with partial recovery. Reconstruction moved the attribution profile toward a paired preattack clean reference, without establishing malicious intent or oracle clean restoration. Evidence remains limited to development and validation, while reserved test arrays remain closed.",
    "The review describes a limited attack effect involving suppression of DDoS predictions toward the Benign class, with partial recovery in attribution reconstruction toward a paired preattack clean reference. The available evidence does not establish malicious intent or oracle clean restoration. Findings are limited to development and validation materials, while reserved test arrays remain closed.",
    "Observed results indicate a limited adverse effect with partial recovery after reconstruction. The recovered attribution profile moved toward the paired preattack clean reference. This comparison does not establish malicious intent or oracle clean restoration. Evidence is confined to development and validation, while reserved test arrays remain unavailable for assessment.",
    "The case reflects a moderate attack effect with partial recovery. The attack affected prediction outcomes, while reconstruction moved the attribution profile toward the paired preattack clean reference. Separate reporting covers the relevant branches, detector performance, and attribution recovery. These findings do not establish malicious intent or oracle clean restoration. Evidence is limited to development and validation, and reserved test arrays remain closed.",
    "The reported attack effect was moderate, with partial recovery under the evaluated conditions. Suppression was observed in the affected prediction outcomes, while reconstruction moved the attribution profile toward the paired preattack reference. The available evidence does not establish malicious intent or oracle clean restoration. Conclusions are limited to development and validation evidence, and reserved test arrays remain closed.",
    "The case documents targeted suppression of DDoS predictions toward the Benign class, with a substantial observed attack effect and limited recovery. Reconstruction shifted the attribution profile toward a paired preattack clean reference, but does not establish malicious intent or oracle clean restoration. The available evidence is confined to development and validation, while reserved test arrays remain closed.",
    "The case documents targeted suppression of DDoS predictions toward the Benign class. Findings cover plain attack, paired clean, and defended conditions, with detector performance and attribution recovery reported separately. Reconstruction shifted the attribution profile toward the paired preattack clean reference, but recovery remained limited. This does not establish malicious intent or oracle clean restoration. Evidence is confined to development and validation; reserved test arrays remain closed.",
    "Assessment indicates a limited suppression effect toward the Benign class, with partial recovery in the defended condition. Reconstruction brought the attribution profile closer to the paired preattack clean reference, without demonstrating malicious intent or oracle clean restoration. Findings are confined to development and validation evidence; reserved test arrays remain closed.",
    "The assessment documents a limited effect on prediction behavior and partial recovery in the reconstructed attribution profile relative to its paired preattack clean reference. Findings are confined to the evaluated development and validation evidence. The separate records provide the relevant branch, detection, and attribution details. The analysis does not establish malicious intent or oracle clean restoration, and reserved test arrays remain closed.",
    "The assessment describes a limited attack effect with partial recovery. Suppression of predictions toward the stated class was observed across the reported branches, while reconstruction moved attribution closer to a paired preattack reference. The findings do not establish intent or oracle clean restoration. Evidence is confined to development and validation, with reserved test arrays remaining closed.",
    "The case reflects a moderate attack effect involving targeted suppression of DDoS predictions toward the Benign class, with partial recovery following reconstruction. The reconstructed attribution profile moved toward the paired preattack clean reference. This finding does not establish malicious intent or oracle clean restoration. Evidence is limited to development and validation materials, while reserved test arrays remain closed.",
    "The report describes a moderate attack effect involving suppression of predictions toward the Benign class, with partial recovery after reconstruction. The recovered attribution profile moved toward the paired preattack clean reference. These findings do not establish malicious intent or restoration to an oracle clean state. Evidence is limited to development and validation materials, while reserved test arrays remain closed.",
    "The report documents substantial suppression of prediction outcomes with partial recovery under defensive analysis. Attribution reconstruction moved toward a paired preattack reference while remaining distinct from any conclusion of malicious intent or oracle-clean restoration. Findings are limited to development and validation evidence; reserved test arrays remain unavailable for this assessment.",
    "The case documents targeted suppression of DDoS predictions toward the Benign class, with a substantial attack effect and partial recovery under the defended branch. Reconstruction moved the attribution profile toward the paired preattack clean reference. The findings do not establish malicious intent or oracle clean restoration. Evidence is limited to development and validation, while reserved test arrays remain closed.",
    "The report describes targeted suppression of prediction outputs, with reconstruction moving the attribution profile toward a paired preattack clean reference. The observed effect was limited, and recovery was partial rather than conclusive. The findings do not establish malicious intent or oracle clean restoration. Evidence is limited to development and validation contexts, while reserved test arrays remain closed.",
    "The assessment describes a limited attack effect with partial recovery in the evaluated setting. Reconstruction shifted the attribution profile toward the paired preattack reference, while not establishing malicious intent or oracle clean restoration. Findings are limited to development and validation evidence. Reserved test arrays remain closed, so conclusions do not extend beyond the reported scope.",
    "The report describes a limited attack effect involving suppression of DDoS predictions toward the Benign class, alongside partial recovery through reconstruction. The recovered attribution profile moved toward a paired preattack clean reference. This comparison does not establish malicious intent or oracle clean restoration. The available evidence is limited to development and validation settings, while reserved test arrays remain closed.",
    "The review addresses suppression of DDoS predictions toward the Benign class, with a moderate attack effect and partial recovery reflected across the separately reported branches. Reconstruction moved the attribution profile toward the paired preattack clean reference. This comparison does not establish malicious intent or oracle clean restoration. The evidence is limited to development and validation, while reserved test arrays remain closed.",
    "The analysis indicates a moderate attack effect with partial recovery under the evaluated conditions. Reconstruction moved the attribution profile toward the paired preattack reference, while not demonstrating restoration to an oracle-clean state or establishing malicious intent. Findings are limited to development and validation evidence; reserved test arrays remain closed, and broader generalization is not established.",
    "This forensic review indicates a moderate attack effect with partial recovery under reconstruction. Results are presented across compared conditions and attribution findings. Reconstruction shifted the attribution profile toward the paired preattack reference, but does not demonstrate malicious intent or oracle-clean restoration. Conclusions are limited to development and validation evidence; reserved test arrays remain unavailable.",
]

# A genuinely unsafe sentence the fixed checker must still catch -- proves
# this isn't a blanket whitelist.
_SYNTHETIC_TRUE_VIOLATION = (
    "Reconstruction achieved oracle clean restoration for this client, "
    "confirming the defended model matches the exact pre-attack state."
)

if __name__ == "__main__":
    false_positive_count_old_would_have_flagged = 1  # the known 123B repeat 3 case
    real_violations_found = sum(count_unsupported_oracle_claims(t) for t in _REAL_TEXTS)
    synthetic_violations_found = count_unsupported_oracle_claims(_SYNTHETIC_TRUE_VIOLATION)

    print(f"Real texts checked: {len(_REAL_TEXTS)}")
    print(f"Real violations found by FIXED checker: {real_violations_found} (must be 0)")
    print(f"Synthetic true-violation sentence flagged: {synthetic_violations_found} (must be >= 1)")
    print()
    ok = real_violations_found == 0 and synthetic_violations_found >= 1
    print("RESULT:", "PASS -- fix confirmed correct on both fronts" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
