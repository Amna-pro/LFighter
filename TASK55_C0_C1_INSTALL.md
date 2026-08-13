# Task 55 C0 and C1 installation

This package freezes the post detection XAI protocol, verifies the local Task 45 checkpoint panel, freezes balanced training background and validation probe indices, and runs a two sample SHAP GradientExplainer compatibility test. It performs no training and never materializes reserved test arrays.

Run these commands from Windows Command Prompt:

```cmd
cd /d "%USERPROFILE%\LFighter-research"
call .venv\Scripts\activate.bat

git switch feature/cic-iot-diad-task55-xai-v417 2>nul || git switch -c feature/cic-iot-diad-task55-xai-v417 task45-c4-coalition-size-partial-v4164

powershell -NoProfile -Command "Expand-Archive -LiteralPath '%USERPROFILE%\Downloads\LFighter_Task55_C0_C1_v417.zip' -DestinationPath '%USERPROFILE%\LFighter-research' -Force"

python -m pip install --upgrade-strategy only-if-needed -r requirements_task55_v417.txt

call scripts\run_task55_c0_c1_v417.bat
```

Expected final gates:

```text
READY FOR C1 LOCAL INVENTORY: True
FEATURES VERIFIED: 69
CHECKPOINT FILES VERIFIED: 28
SHAP VERSION: 0.48.0
GRADIENT EXPLAINER SMOKE: True
RESERVED TEST ARRAYS MATERIALIZED: False
READY FOR C2 SEED 7 PILOT: True
```

Do not commit if any check fails. Send the complete terminal output for review. The next package will run the seed 7 three family explanation pilot using the frozen indices and checkpoints.
