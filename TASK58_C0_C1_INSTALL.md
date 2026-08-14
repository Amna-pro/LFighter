# Task 58 C0 and C1 installation

This package freezes the publication figure rules, generates five publication figure families from the already frozen Task 56 and Task 57 evidence, and independently audits the generated files. It performs no training and no new SHAP evaluation.

Run from the repository root with the project virtual environment active:

```bat
python -m pip install --upgrade-strategy only-if-needed -r requirements_task58_v420.txt
call scripts\run_task58_c0_c1_v420.bat
```

Review the complete output before committing or tagging. The batch file stages only the Task 58 protocol, scripts, and generated Task 58 evidence.
