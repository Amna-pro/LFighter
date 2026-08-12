# Task 45 C0 installation and audit

This package contains only preregistration and integrity code. It does not start training.

## Included files

* `configs/TASK45_PREREGISTRATION_V416.md`
* `configs/task45_preregistration_v4160.json`
* `scripts/audit_task45_c0_preregistration_v416.py`
* `scripts/select_task45_coalitions_v416.py`
* `scripts/run_task45_dev_only_adapter_v416.py`

## Install from Windows CMD

Place `LFighter_Task45_C0_v416.zip` in the Windows Downloads folder, then run:

```cmd
cd /d "%USERPROFILE%\LFighter-research"
call .venv\Scripts\activate.bat

powershell -NoProfile -Command "Expand-Archive -LiteralPath '%USERPROFILE%\Downloads\LFighter_Task45_C0_v416.zip' -DestinationPath '%USERPROFILE%\LFighter-research' -Force"

git status --short --branch
```

Expected new files are the five files listed above plus this installation note. No existing frozen file should be modified.

## Run the C0 audit

```cmd
cd /d "%USERPROFILE%\LFighter-research"
call .venv\Scripts\activate.bat

python scripts\audit_task45_c0_preregistration_v416.py --project-root . --output-dir results\cic_iot_diad_task45_c0_preregistration_v416
```

The audit must report all checks passed, `READY FOR C0 FREEZE COMMIT: True`, `TASK 45 TRAINING STARTED: False`, and `RESERVED TEST ARRAYS MATERIALIZED: False`.

Do not run the selector or any training command until the C0 audit output has been reviewed and the C0 freeze commit has been created.
