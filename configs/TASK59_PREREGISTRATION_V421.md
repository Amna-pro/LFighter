# Task 59 grounded evidence schema preregistration

Task 59 defines a versioned JSON Schema for forensic evidence passed to later deterministic and LLM report generators. It does not call an LLM and does not reopen any defense, attack, reconstruction, or XAI decision.

The schema separates observed facts from interpretation. Every fact requires provenance references, every interpretation requires evidence basis references and an uncertainty level, and every source file requires its SHA256 digest. Missing evidence remains explicit. Unknown detector score or threshold values cannot be fabricated merely to complete a report.

The authority object is fixed by schema constants. Any record that grants a reporting model permission to flag clients, change thresholds, modify trust, reconstruct updates, control aggregation, or claim malicious intent must fail validation. Unknown properties, out of range rates, missing provenance, and control characters must also fail.

The canonical example reuses the deterministic Task 58 representative case, seed 7 and coalition family B at size 10. This is an interface validation example, not a new scientific outcome or selected success case.
