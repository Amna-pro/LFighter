@echo off
setlocal
cd /d "%~dp0\.."

set "CFG=configs\task56_preregistration_v4180.json"
set "DATA=data\processed\cic_iot_diad_2024_v2_1\arrays\behavioral_only.npz"
set "FEATURES=data\processed\cic_iot_diad_2024_v2_1\preprocessors\behavioral_only\selected_features.csv"
set "INDICES=results\cic_iot_diad_task56_c1_preflight_v418\tables\task56c1_frozen_validation_indices.npz"
set "PLAN=results\cic_iot_diad_task56_c1_preflight_v418\tables\task56c1_validation_plan.csv"
set "READY=results\cic_iot_diad_task56_c1_preflight_v418\tables\task56c1_state_readiness.csv"
set "C0DEC=results\cic_iot_diad_task56_c0_preregistration_v418\task56c0_preregistration_decision.json"
set "C1DEC=results\cic_iot_diad_task56_c1_preflight_v418\task56c1_preflight_decision.json"
set "WARMUP=results\cic_iot_diad_true_warmup_anchor_v3101_multiseed"
set "T45C2=results\cic_iot_diad_task45_c2_seed7_v416"
set "T45C3=results\cic_iot_diad_task45_c3_multiseed_v416"
set "T55C2=results\cic_iot_diad_task55_c2_seed7_v417"
set "T55C3=results\cic_iot_diad_task55_c3_multiseed_v417"
set "EXEC=results\cic_iot_diad_task56_c2_execution_v418"
set "SUMMARY=results\cic_iot_diad_task56_c2_summary_v418"
set "AUDIT=results\cic_iot_diad_task56_c2_audit_v418"

set "RUN=python scripts\run_task56_c2_validation_v418.py --project-root . --config "%CFG%" --data-file "%DATA%" --feature-file "%FEATURES%" --frozen-index-file "%INDICES%" --validation-plan "%PLAN%" --checkpoint-readiness "%READY%" --warmup-root "%WARMUP%" --task45-c2-root "%T45C2%" --task45-c3-root "%T45C3%" --output-root "%EXEC%" --threads 6"

echo ===== TASK 56 C2 DRY RUN =====
%RUN% --dry-run
if errorlevel 1 exit /b 1

echo ===== TASK 56 C2 SHAP VALIDATION RUN =====
%RUN%
if errorlevel 1 exit /b 1

echo ===== TASK 56 C2 PREREGISTERED SUMMARY =====
python scripts\summarize_task56_c2_validation_v418.py --project-root . --config "%CFG%" --data-file "%DATA%" --frozen-index-file "%INDICES%" --checkpoint-readiness "%READY%" --warmup-root "%WARMUP%" --task45-c2-root "%T45C2%" --task45-c3-root "%T45C3%" --task55-c2-root "%T55C2%" --task55-c3-root "%T55C3%" --execution-root "%EXEC%" --output-dir "%SUMMARY%" --threads 6
if errorlevel 1 exit /b 1

echo ===== TASK 56 C2 INDEPENDENT AUDIT =====
python scripts\audit_task56_c2_validation_v418.py --project-root . --config "%CFG%" --c0-decision "%C0DEC%" --c1-decision "%C1DEC%" --validation-plan "%PLAN%" --execution-root "%EXEC%" --summary-root "%SUMMARY%" --output-dir "%AUDIT%"
if errorlevel 1 exit /b 1

git add TASK56_C2_INSTALL.md requirements_task56_c2_v418.txt scripts\run_task56_c2_validation_v418.py scripts\summarize_task56_c2_validation_v418.py scripts\audit_task56_c2_validation_v418.py scripts\run_task56_c2_v418.bat
git add -f "%EXEC%" "%SUMMARY%" "%AUDIT%"
git diff --cached --check
if errorlevel 1 exit /b 1

echo ===== TASK 56 C2 READY FOR REVIEW =====
git diff --cached --stat
git status --short --branch
echo NEXT ACTION: Send the complete output for review before commit and tag.
exit /b 0
