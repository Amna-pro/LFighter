# Task 45 C1 preflight and C2 seed 7 runner

This package adds the C1 artifact preflight and the frozen C2 seed 7 orchestrator. Installing the files does not start training.

## Install

```cmd
cd /d "%USERPROFILE%\LFighter-research"
call .venv\Scripts\activate.bat

powershell -NoProfile -Command "Expand-Archive -LiteralPath '%USERPROFILE%\Downloads\LFighter_Task45_C1_C2_v416.zip' -DestinationPath '%USERPROFILE%\LFighter-research' -Force"
```

Run `audit_task45_c1_preflight_v416.py` before any C2 execution. The C2 runner refuses to execute unless the working tree is clean and tag `task45-c1-preflight-frozen-v4161` is an ancestor of `HEAD`.
