# Task 56 C0 and C1 installation

This package freezes the Task 56 validation protocol and audits local readiness. It does not calculate Task 56 scientific outcomes.

Run from Windows Command Prompt:

```bat
cd /d "%USERPROFILE%\LFighter-research"
call .venv\Scripts\activate.bat
python -m pip install --upgrade-strategy only-if-needed -r requirements_task56_v418.txt
call scripts\run_task56_c0_c1_v418.bat
```

The preflight verifies the frozen Task 55 evidence, all 28 checkpoints, all attribution tensors, allowed data arrays, alternate background feasibility, nested probe subsets, random faithfulness controls, package versions, and a small GradientExplainer smoke test. Reserved test arrays are never materialized.

Send the complete output for review before committing or tagging.
