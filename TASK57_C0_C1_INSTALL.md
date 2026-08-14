# Task 57 C0 and C1 installation

This package freezes the Task 57 attribution recovery protocol and audits the available evidence without evaluating Task 57 recovery outcomes.

## Install and run from Windows CMD

```bat
cd /d "%USERPROFILE%\LFighter-research"
call .venv\Scripts\activate.bat

git switch feature/cic-iot-diad-task57-xai-recovery-v419 2>nul || git switch -c feature/cic-iot-diad-task57-xai-recovery-v419 task56-c2-xai-validation-frozen-v4182

powershell -NoProfile -Command "Expand-Archive -LiteralPath '%USERPROFILE%\Downloads\LFighter_Task57_C0_C1_v419.zip' -DestinationPath '%USERPROFILE%\LFighter-research' -Force"

python -m pip install --upgrade-strategy only-if-needed -r requirements_task57_v419.txt
call scripts\run_task57_c0_c1_v419.bat
```

## Expected scientific scope

The complete primary panel contains 28 physical attribution states and 12 clean, suspicious, reconstructed triplets. Rejected and oracle clean states are explicitly unavailable. They are not substituted. The claim is limited to recovery toward the frozen preattack clean reference.

Task 57 C0 and C1 perform no training, no new SHAP evaluation, no recovery outcome analysis, and no reserved test materialization.

Send the complete CMD output for review before committing or tagging.
