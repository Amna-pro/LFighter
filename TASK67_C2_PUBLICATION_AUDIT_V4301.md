# Task 67 C2 — Publication Output Integrity Audit V4.30.1

This is an audit-only stage over the already-generated Task 67 C1 publication artifacts.

It does not regenerate figures or tables, rerun Task 66 inference, rerun Task 65, open the reserved NPZ, load models, train, recompute SHAP, or invoke an LLM.

The audit verifies:
- exact 30 figure stems with one PNG and one PDF per stem,
- exact seven publication tables,
- completion-marker counts and SHA-256 hashes,
- PNG readability, nonblank content, image dimensions, and approximately 300-dpi metadata,
- PDF signature and nontrivial file size,
- byte-for-byte preservation of Task 66 tables 1–6,
- preservation of claim-boundary table 7,
- no best-seed / best-round selection,
- negative, null, and non-estimable results retained.

The audit writes only a compact integrity JSON and a figure-file manifest.
