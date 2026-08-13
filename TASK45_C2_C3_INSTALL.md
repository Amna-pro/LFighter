# Task 45 C2 audit and C3 confirmatory runner

This package contains the read only seed 7 C2 audit and the frozen C3 runner for seeds 99, 123, and 2026.

The C2 audit verifies all 18 coalition conditions, all 36 paired branches, poison index equality, four round metric schemas, the frozen reconstruction policy, development only array isolation, and absence of reserved test access. It writes compact audit evidence and a seed 7 development summary without changing any experiment branch.

The C3 runner is locked to the same 18 coalition conditions and the three preregistered confirmatory seeds. It requires the C2 freeze tag and a clean working tree. It stops on incomplete output unless the operator explicitly requests an overwrite after inspection.

Reserved natural and diagnostic test arrays remain prohibited.
