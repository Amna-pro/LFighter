from __future__ import annotations

import re
from typing import Any

REDUCTION_RE = re.compile(r"reduction of (\d+\.\d+)")
RECOVERY_RE = re.compile(r"recovery fraction (\d+\.\d+)")


def _magnitude_bucket(reduction: float) -> str:
    if reduction < 0.20:
        return "limited"
    if reduction < 0.40:
        return "moderate"
    return "substantial"


def _recovery_bucket(recovery: float) -> str:
    if recovery < 0.30:
        return "weak"
    if recovery < 0.50:
        return "partial"
    return "strong"


def derive_capsule_phrase(locked_content: dict[str, Any]) -> str:
    """Derive a zero-digit qualitative phrase from the already-locked facts.

    Reads the reduction and recovery-fraction values that are already
    present as English sentences in locked_content["facts"] (indices 3
    and 2 respectively, matching the fixed contract schema used
    throughout Task 62), buckets each into one of three coarse,
    non-numeric levels, and composes them into a short phrase for
    insertion into the user-prompt capsule. The LLM never sees a digit;
    it only sees words like "substantial" and "strong".

    Thresholds (0.20/0.40 for reduction, 0.30/0.50 for recovery) were
    set from the real observed gaps across all 12 preregistered cases,
    not chosen arbitrarily -- see task62_capsule_threshold_derivation.md
    for the full gap analysis this was derived from.
    """
    facts = locked_content["facts"]
    reduction_match = REDUCTION_RE.search(facts[3]["statement"])
    recovery_match = RECOVERY_RE.search(facts[2]["statement"])
    if reduction_match is None or recovery_match is None:
        raise ValueError(
            "Could not locate reduction/recovery figures in the expected "
            "fact positions -- contract schema may have changed; do not "
            "silently fall back to a static phrase."
        )
    reduction = float(reduction_match.group(1))
    recovery = float(recovery_match.group(1))
    magnitude = _magnitude_bucket(reduction)
    recovery_level = _recovery_bucket(recovery)
    return f"a {magnitude} attack effect with {recovery_level} recovery"


# Self-test against all 12 known cases from the real C1 contracts.
# Run directly (`python task62_capsule_v4251.py`) to verify before wiring
# this into the execution script.
_KNOWN_CASES = {
    "seed_123__A_development_anchor__size_10": (0.483134, 0.587239, "substantial", "strong"),
    "seed_123__B_hash_ranked__size_10": (0.101783, 0.631095, "limited", "strong"),
    "seed_123__C_hash_ranked__size_10": (0.263053, 0.634439, "moderate", "strong"),
    "seed_2026__A_development_anchor__size_10": (0.502938, 0.260068, "substantial", "weak"),
    "seed_2026__B_hash_ranked__size_10": (0.102796, 0.399214, "limited", "partial"),
    "seed_2026__C_hash_ranked__size_10": (0.286420, 0.379252, "moderate", "partial"),
    "seed_7__A_development_anchor__size_10": (0.492711, 0.241595, "substantial", "weak"),
    "seed_7__B_hash_ranked__size_10": (0.096894, 0.395341, "limited", "partial"),
    "seed_7__C_hash_ranked__size_10": (0.307549, 0.392606, "moderate", "partial"),
    "seed_99__A_development_anchor__size_10": (0.486922, 0.367060, "substantial", "partial"),
    "seed_99__B_hash_ranked__size_10": (0.079278, 0.425979, "limited", "partial"),
    "seed_99__C_hash_ranked__size_10": (0.298760, 0.461212, "moderate", "partial"),
}

if __name__ == "__main__":
    failures = 0
    for case_id, (reduction, recovery, expected_mag, expected_rec) in _KNOWN_CASES.items():
        fake_locked = {
            "facts": [
                {"statement": "placeholder"},
                {"statement": "placeholder"},
                {"statement": f"Normalized attribution distance decreased, with recovery fraction {recovery}."},
                {"statement": f"The defended rate was X, an absolute reduction of {reduction} from the plain attack branch."},
            ]
        }
        phrase = derive_capsule_phrase(fake_locked)
        expected = f"a {expected_mag} attack effect with {expected_rec} recovery"
        status = "OK" if phrase == expected else "MISMATCH"
        if phrase != expected:
            failures += 1
        print(f"{status:9s} {case_id}: {phrase}")
    print()
    print(f"{len(_KNOWN_CASES) - failures}/{len(_KNOWN_CASES)} cases matched expected bucket assignment")
    raise SystemExit(1 if failures else 0)
