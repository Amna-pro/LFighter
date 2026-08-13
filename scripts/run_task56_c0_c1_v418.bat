@echo off
setlocal
cd /d "%~dp0\.."

set "CFG=configs\task56_preregistration_v4180.json"
set "PROTOCOL=configs\TASK56_PREREGISTRATION_V418.md"
set "DATA=data\processed\cic_iot_diad_2024_v2_1\arrays\behavioral_only.npz"
set "FEATURES=data\processed\cic_iot_diad_2024_v2_1\preprocessors\behavioral_only\selected_features.csv"
set "INDICES=results\cic_iot_diad_task55_c1_preflight_v417\tables\task55c1_frozen_sample_indices.npz"
set "INVENTORY=results\cic_iot_diad_task55_c1_preflight_v417\tables\task55c1_checkpoint_inventory.csv"
set "WARMUP=results\cic_iot_diad_true_warmup_anchor_v3101_multiseed"
set "T45C2=results\cic_iot_diad_task45_c2_seed7_v416"
set "T45C3=results\cic_iot_diad_task45_c3_multiseed_v416"
set "T55C2=results\cic_iot_diad_task55_c2_seed7_v417"
set "T55C3=results\cic_iot_diad_task55_c3_multiseed_v417"
set "T55AUDIT=results\cic_iot_diad_task55_c3_audit_v417"
set "C0OUT=results\cic_iot_diad_task56_c0_preregistration_v418"
set "C1OUT=results\cic_iot_diad_task56_c1_preflight_v418"

echo ===== RUNNING TASK 56 C0 AUDIT =====
python scripts\audit_task56_c0_preregistration_v418.py --project-root . --config "%CFG%" --protocol "%PROTOCOL%" --output-dir "%C0OUT%"
if errorlevel 1 exit /b 1

echo ===== RUNNING TASK 56 C1 PREFLIGHT =====
python scripts\preflight_task56_c1_v418.py --project-root . --config "%CFG%" --data-file "%DATA%" --feature-file "%FEATURES%" --task55-index-file "%INDICES%" --checkpoint-inventory "%INVENTORY%" --warmup-root "%WARMUP%" --task45-c2-root "%T45C2%" --task45-c3-root "%T45C3%" --task55-c2-root "%T55C2%" --task55-c3-root "%T55C3%" --task55-c3-audit-root "%T55AUDIT%" --output-dir "%C1OUT%"
if errorlevel 1 exit /b 1

git add TASK56_C0_C1_INSTALL.md requirements_task56_v418.txt configs\TASK56_PREREGISTRATION_V418.md configs\task56_preregistration_v4180.json scripts\audit_task56_c0_preregistration_v418.py scripts\preflight_task56_c1_v418.py scripts\run_task56_c0_c1_v418.bat
git add -f "%C0OUT%" "%C1OUT%"
git diff --cached --check
if errorlevel 1 exit /b 1

echo ===== TASK 56 C0 AND C1 READY FOR REVIEW =====
git diff --cached --stat
git status --short --branch
echo NEXT ACTION: Send the complete output for review before commit and tag.
exit /b 0
