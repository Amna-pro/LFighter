@echo off
setlocal
cd /d "%~dp0\.."

set "CFG=configs\task55_preregistration_v4170.json"
set "DATA=data\processed\cic_iot_diad_2024_v2_1\arrays\behavioral_only.npz"
set "FEATURES=data\processed\cic_iot_diad_2024_v2_1\preprocessors\behavioral_only\selected_features.csv"
set "INDICES=results\cic_iot_diad_task55_c1_preflight_v417\tables\task55c1_frozen_sample_indices.npz"
set "INVENTORY=results\cic_iot_diad_task55_c1_preflight_v417\tables\task55c1_checkpoint_inventory.csv"
set "C1DEC=results\cic_iot_diad_task55_c1_preflight_v417\task55c1_preflight_decision.json"
set "C2DEC=results\cic_iot_diad_task55_c2_audit_v417\task55c2_audit_decision.json"
set "C2ROOT=results\cic_iot_diad_task55_c2_seed7_v417"
set "WARMUP=results\cic_iot_diad_true_warmup_anchor_v3101_multiseed"
set "T45=results\cic_iot_diad_task45_c3_multiseed_v416"
set "OUT=results\cic_iot_diad_task55_c3_multiseed_v417"
set "AUDIT=results\cic_iot_diad_task55_c3_audit_v417"

echo ===== TASK 55 C3 DRY RUN =====
python scripts\run_task55_c3_multiseed_v417.py --project-root . --config "%CFG%" --data-file "%DATA%" --feature-file "%FEATURES%" --sample-index-file "%INDICES%" --checkpoint-inventory "%INVENTORY%" --warmup-root "%WARMUP%" --task45-c3-root "%T45%" --output-root "%OUT%" --dry-run
if errorlevel 1 exit /b 1

echo ===== TASK 55 C3 CONFIRMATORY MULTISEED RUN =====
python scripts\run_task55_c3_multiseed_v417.py --project-root . --config "%CFG%" --data-file "%DATA%" --feature-file "%FEATURES%" --sample-index-file "%INDICES%" --checkpoint-inventory "%INVENTORY%" --warmup-root "%WARMUP%" --task45-c3-root "%T45%" --output-root "%OUT%"
if errorlevel 1 exit /b 1

echo ===== TASK 55 C3 AUDIT =====
python scripts\audit_task55_c3_multiseed_v417.py --project-root . --config "%CFG%" --c1-decision "%C1DEC%" --c1-checkpoint-inventory "%INVENTORY%" --c2-decision "%C2DEC%" --c2-root "%C2ROOT%" --c3-root "%OUT%" --output-dir "%AUDIT%"
if errorlevel 1 exit /b 1

git add TASK55_C3_INSTALL.md requirements_task55_c3_v417.txt scripts\run_task55_c3_multiseed_v417.py scripts\audit_task55_c3_multiseed_v417.py scripts\run_task55_c3_v417.bat
git add -f "%OUT%" "%AUDIT%"
git diff --cached --check
if errorlevel 1 exit /b 1

echo ===== TASK 55 C3 READY FOR REVIEW =====
git diff --cached --stat
git status --short --branch
echo NEXT ACTION: Send the complete output for review before commit and tag.
exit /b 0
