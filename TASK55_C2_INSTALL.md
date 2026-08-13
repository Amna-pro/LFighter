# Task 55 C2 seed 7 SHAP pilot

This package runs the frozen Task 55 seed 7 pilot for coalition size 10 across the three Task 45 coalition families. It evaluates one shared clean checkpoint plus paired suspicious and reconstructed checkpoints, for seven unique checkpoint computations and nine logical state comparisons.

The runner is resumable. A completed checkpoint state is verified and skipped when the command is run again. An interrupted state is recomputed. No model training occurs. Only training and validation arrays are materialized; all reserved test arrays remain closed.

Run from Windows Command Prompt:

```cmd
cd /d "%USERPROFILE%\LFighter-research"
call .venv\Scripts\activate.bat

powershell -NoProfile -Command "Expand-Archive -LiteralPath '%USERPROFILE%\Downloads\LFighter_Task55_C2_v417.zip' -DestinationPath '%USERPROFILE%\LFighter-research' -Force"

python -m pip install --upgrade-strategy only-if-needed -r requirements_task55_c2_v417.txt

call scripts\run_task55_c2_v417.bat
```

The run may take time on CPU. Leave Command Prompt open. If Windows restarts or the process stops, run the same batch command again.

Expected final gates:

```text
UNIQUE CHECKPOINT STATES VERIFIED: 7
LOGICAL STATE COMPARISONS VERIFIED: 9
ATTRIBUTION TENSORS FINITE: True
EXACT DERIVED MARGIN IDENTITY: True
RESERVED TEST ARRAYS MATERIALIZED: False
READY FOR C3 CONFIRMATORY MULTISEED: True
```

Do not commit if any audit check fails. Send the complete terminal output for review.
