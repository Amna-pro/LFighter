@echo off
setlocal

cd /d "%~dp0\.."
if errorlevel 1 exit /b 1

for /f "delims=" %%B in ('git branch --show-current') do set "TASK55_BRANCH=%%B"
if /I not "%TASK55_BRANCH%"=="feature/cic-iot-diad-task55-xai-v417" (
  echo ERROR: Expected branch feature/cic-iot-diad-task55-xai-v417 but found %TASK55_BRANCH%
  exit /b 2
)

python -c "import shap, pyarrow; assert shap.__version__ == '0.48.0', shap.__version__"
if errorlevel 1 (
  echo ERROR: Required Task 55 C2 packages are missing or incompatible.
  echo RUN: python -m pip install --upgrade-strategy only-if-needed -r requirements_task55_c2_v417.txt
  exit /b 3
)

set "T55_DATA=data\processed\cic_iot_diad_2024_v2_1\arrays\behavioral_only.npz"
set "T55_FEATURES=data\processed\cic_iot_diad_2024_v2_1\preprocessors\behavioral_only\selected_features.csv"
set "T55_CONFIG=configs\task55_preregistration_v4170.json"
set "T55_INDICES=results\cic_iot_diad_task55_c1_preflight_v417\tables\task55c1_frozen_sample_indices.npz"
set "T55_WARMUP=results\cic_iot_diad_true_warmup_anchor_v3101_multiseed"
set "T55_TASK45=results\cic_iot_diad_task45_c2_seed7_v416"
set "T55_OUT=results\cic_iot_diad_task55_c2_seed7_v417"
set "T55_AUDIT=results\cic_iot_diad_task55_c2_audit_v417"

echo ===== TASK 55 C2 DRY RUN =====
python scripts\run_task55_c2_seed7_v417.py --project-root . --config "%T55_CONFIG%" --data-file "%T55_DATA%" --feature-file "%T55_FEATURES%" --sample-index-file "%T55_INDICES%" --warmup-root "%T55_WARMUP%" --task45-c2-root "%T55_TASK45%" --output-root "%T55_OUT%" --threads 6 --dry-run
if errorlevel 1 exit /b 10

echo ===== TASK 55 C2 SEED 7 PILOT =====
python scripts\run_task55_c2_seed7_v417.py --project-root . --config "%T55_CONFIG%" --data-file "%T55_DATA%" --feature-file "%T55_FEATURES%" --sample-index-file "%T55_INDICES%" --warmup-root "%T55_WARMUP%" --task45-c2-root "%T55_TASK45%" --output-root "%T55_OUT%" --threads 6
if errorlevel 1 exit /b 11

echo ===== TASK 55 C2 AUDIT =====
python scripts\audit_task55_c2_seed7_v417.py --project-root . --config "%T55_CONFIG%" --c1-decision results\cic_iot_diad_task55_c1_preflight_v417\task55c1_preflight_decision.json --c1-checkpoint-inventory results\cic_iot_diad_task55_c1_preflight_v417\tables\task55c1_checkpoint_inventory.csv --c2-root "%T55_OUT%" --output-dir "%T55_AUDIT%"
if errorlevel 1 exit /b 12

git add TASK55_C2_INSTALL.md requirements_task55_c2_v417.txt scripts\run_task55_c2_seed7_v417.py scripts\audit_task55_c2_seed7_v417.py scripts\run_task55_c2_v417.bat
if errorlevel 1 exit /b 20
git add -f "%T55_OUT%" "%T55_AUDIT%"
if errorlevel 1 exit /b 21
git diff --cached --check
if errorlevel 1 exit /b 22

echo ===== TASK 55 C2 READY FOR REVIEW =====
git diff --cached --stat
git status --short --branch
echo NEXT ACTION: Send the complete output for review before commit and tag.
exit /b 0
