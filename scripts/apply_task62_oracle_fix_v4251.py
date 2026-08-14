from __future__ import annotations

from pathlib import Path

TARGET = Path("scripts/summarize_task62_c2_hybrid_v425.py")

IMPORT_ANCHOR = "    valid_completion,\n    write_json,\n)"
NEW_IMPORT_LINE = "\nfrom task62_oracle_check_fix_v4251 import count_unsupported_oracle_claims"

OLD_LINE = '                "unsupported_oracle_claim_count": count_patterns(executive_decision_text, ORACLE_PATTERNS),'
NEW_LINE = '                "unsupported_oracle_claim_count": count_unsupported_oracle_claims(executive_decision_text),'


def main() -> int:
    if not TARGET.exists():
        raise SystemExit(f"Target file not found: {TARGET} -- run this from the repository root")

    text = TARGET.read_text(encoding="utf-8")

    import_count = text.count(IMPORT_ANCHOR)
    if import_count != 1:
        raise SystemExit(
            f"REFUSING TO EDIT: import anchor found {import_count} time(s), expected exactly 1. "
            "The file may have changed since this patch was written -- do not proceed blindly."
        )

    line_count = text.count(OLD_LINE)
    if line_count != 1:
        raise SystemExit(
            f"REFUSING TO EDIT: target line found {line_count} time(s), expected exactly 1. "
            "The file may have changed since this patch was written -- do not proceed blindly."
        )

    already_imported = "from task62_oracle_check_fix_v4251 import count_unsupported_oracle_claims" in text
    if already_imported:
        raise SystemExit("This file already has the fix applied -- refusing to patch twice.")

    patched = text.replace(IMPORT_ANCHOR, IMPORT_ANCHOR + NEW_IMPORT_LINE)
    patched = patched.replace(OLD_LINE, NEW_LINE)

    TARGET.write_text(patched, encoding="utf-8")

    verify = TARGET.read_text(encoding="utf-8")
    assert "from task62_oracle_check_fix_v4251 import count_unsupported_oracle_claims" in verify, (
        "Post-write verification of the import failed -- inspect the file manually before running anything."
    )
    assert NEW_LINE in verify, "Post-write verification of the line swap failed."
    assert OLD_LINE not in verify, "Old line still present after patch -- something is wrong."

    print("PATCH APPLIED AND VERIFIED:")
    print(f"  - import added: from task62_oracle_check_fix_v4251 import count_unsupported_oracle_claims")
    print(f"  - line 112 (unsupported_oracle_claim_count) now calls the fixed checker")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
