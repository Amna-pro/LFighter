@echo off
setlocal

cd /d "%~dp0\.."
if errorlevel 1 exit /b 1

for /f "delims=" %%B in ('git branch --show-current') do set "TASK55_BRANCH=%%B"
if /I not "%TASK55_BRANCH%"=="feature/cic-iot-diad-task55-xai-v417" (
  echo ERROR: Expected branch feature/cic-iot-diad-task55-xai-v417 but found %TASK55_BRANCH%
  exit /b 2
)

python -c "import shap; assert shap.__version__ == '0.48.0', shap.__version__"
if errorlevel 1 (
  echo ERROR: SHAP 0.48.0 is required.
  echo RUN: python -m pip install --upgrade-strategy only-if-needed -r requirements_task55_v417.txt
  exit /b 3
)

set "TASK55_DATA=data\processed\cic_iot_diad_2024_v2_1\arrays\behavioral_only.npz"
set "TASK55_FEATURES=data\processed\cic_iot_diad_2024_v2_1\preprocessors\behavioral_only\selected_features.csv"
set "TASK55_WARMUP=results\cic_iot_diad_true_warmup_anchor_v3101_multiseed"
set "TASK55_C2=results\cic_iot_diad_task45_c2_seed7_v416"
set "TASK55_C3=results\cic_iot_diad_task45_c3_multiseed_v416"
set "TASK55_CONFIG=configs\task55_preregistration_v4170.json"
set "TASK55_PROTOCOL=configs\TASK55_PREREGISTRATION_V417.md"
set "TASK55_C0_OUT=results\cic_iot_diad_task55_c0_preregistration_v417"
set "TASK55_C1_OUT=results\cic_iot_diad_task55_c1_preflight_v417"

echo ===== RUNNING TASK 55 C0 AUDIT =====
python scripts\audit_task55_c0_preregistration_v417.py --project-root . --config "%TASK55_CONFIG%" --protocol "%TASK55_PROTOCOL%" --output-dir "%TASK55_C0_OUT%"
if errorlevel 1 exit /b 10

echo ===== RUNNING TASK 55 C1 LOCAL INVENTORY =====
python scripts\preflight_task55_c1_inventory_v417.py --project-root . --config "%TASK55_CONFIG%" --data-file "%TASK55_DATA%" --feature-file "%TASK55_FEATURES%" --warmup-root "%TASK55_WARMUP%" --c2-root "%TASK55_C2%" --c3-root "%TASK55_C3%" --output-dir "%TASK55_C1_OUT%"
if errorlevel 1 exit /b 11

git add TASK55_C0_C1_INSTALL.md requirements_task55_v417.txt configs\TASK55_PREREGISTRATION_V417.md configs\task55_preregistration_v4170.json scripts\audit_task55_c0_preregistration_v417.py scripts\preflight_task55_c1_inventory_v417.py scripts\run_task55_c0_c1_v417.bat
if errorlevel 1 exit /b 20
git add -f "%TASK55_C0_OUT%" "%TASK55_C1_OUT%"
if errorlevel 1 exit /b 21

git diff --cached --check
if errorlevel 1 exit /b 22

echo ===== TASK 55 C0 AND C1 READY FOR REVIEW =====
git diff --cached --stat
git status --short --branch
echo NEXT ACTION: Send the full output for review before commit and tag.
exit /b 0
