# Task 59 C0 and C1 installation

This package freezes and validates the LFighter grounded forensic evidence schema. It creates a strict JSON Schema, a canonical evidence record built from frozen Task 45 through Task 58 artifacts, a field mapping, a data dictionary, negative validation cases, and independent audit evidence.

It performs no LLM call, training, new SHAP evaluation, or reserved test access.

Run from the repository root with the virtual environment active:

```bat
python -m pip install --upgrade-strategy only-if-needed -r requirements_task59_v421.txt
call scripts\run_task59_c0_c1_v421.bat
```

Review the complete output before committing or tagging.
