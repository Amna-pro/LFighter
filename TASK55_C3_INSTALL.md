# Task 55 C3 confirmatory multiseed SHAP run

This package runs the frozen Task 55 protocol for confirmatory seeds 99, 123, and 2026.

The run evaluates 21 unique checkpoint states and materializes 27 logical family/state comparisons. It never trains a model and only loads `X_train`, `y_train`, `X_val`, and `y_val`. Reserved test arrays remain closed.

The runner is checkpoint resumable. Each completed state is protected by a completion marker and SHA256 hashes. If Windows or the PC closes, rerun the same batch file; verified states are skipped and an incomplete state is safely recomputed.

Run from Windows Command Prompt:

```bat
cd /d "%USERPROFILE%\LFighter-research"
call .venv\Scripts\activate.bat
python -m pip install --upgrade-strategy only-if-needed -r requirements_task55_c3_v417.txt
call scripts\run_task55_c3_v417.bat
```

Success requires all C3 audit checks to pass and `READY FOR TASK 56 VALIDATION: True`.
