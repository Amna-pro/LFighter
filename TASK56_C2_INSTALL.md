# Task 56 C2 — Stability and faithfulness validation

This package executes the frozen 140-evaluation SHAP validation plan, computes every preregistered Task 56 domain, and independently audits the complete evidence. It never trains a model and never materializes reserved test arrays.

From Windows Command Prompt:

```bat
cd /d "%USERPROFILE%\LFighter-research"
call .venv\Scripts\activate.bat

powershell -NoProfile -Command "Expand-Archive -LiteralPath '%USERPROFILE%\Downloads\LFighter_Task56_C2_v4182.zip' -DestinationPath '%USERPROFILE%\LFighter-research' -Force"

python -m pip install --upgrade-strategy only-if-needed -r requirements_task56_c2_v418.txt

call scripts\run_task56_c2_v418.bat
```

The run is checkpoint-resumable. If Windows restarts or the process is interrupted, execute the same batch command again; verified evaluations are skipped. A scientific PASS, PARTIAL, or FAIL is retained. Only an incomplete or corrupt integrity audit causes the batch to fail.
