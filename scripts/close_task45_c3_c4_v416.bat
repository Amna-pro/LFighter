@echo off
setlocal

cd /d "%~dp0.."
if errorlevel 1 exit /b 1

set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  echo ERROR: Project virtual environment Python was not found.
  exit /b 1
)

set "T45_MANIFEST=results\cic_iot_diad_task45_c1_coalitions_v416\tables\task45_coalition_manifest.csv"
set "T45_WARMUP=results\cic_iot_diad_true_warmup_anchor_v3101_multiseed"
set "T45_C2=results\cic_iot_diad_task45_c2_seed7_v416"
set "T45_C3=results\cic_iot_diad_task45_c3_multiseed_v416"
set "T45_C3_AUDIT=results\cic_iot_diad_task45_c3_audit_v416"
set "T45_C4=results\cic_iot_diad_task45_c4_summary_v416"
set "T45_C3_TAG=task45-c3-confirmatory-frozen-v4163"

git rev-parse -q --verify "refs/tags/%T45_C3_TAG%" >nul 2>&1
if errorlevel 1 goto run_c3_audit
for /f %%i in ('git rev-list -n 1 "%T45_C3_TAG%"') do set "TAG_COMMIT=%%i"
for /f %%i in ('git rev-parse HEAD') do set "HEAD_COMMIT=%%i"
if not "%TAG_COMMIT%"=="%HEAD_COMMIT%" (
  echo ERROR: Existing C3 tag does not point to current HEAD.
  exit /b 1
)
goto run_c4

:run_c3_audit
echo ===== RUNNING TASK 45 C3 AUDIT =====
"%PYTHON_EXE%" scripts\audit_task45_c3_multiseed_v416.py --project-root . --coalition-manifest "%T45_MANIFEST%" --c2-root "%T45_C2%" --c3-root "%T45_C3%" --output-dir "%T45_C3_AUDIT%"
if errorlevel 1 exit /b 1

git add TASK45_C3_C4_INSTALL.md scripts\audit_task45_c3_multiseed_v416.py scripts\summarize_task45_c4_coalition_curve_v416.py scripts\close_task45_c3_c4_v416.bat
if errorlevel 1 exit /b 1
git add -f "%T45_C3_AUDIT%"
if errorlevel 1 exit /b 1
git diff --cached --check
if errorlevel 1 exit /b 1

git diff --cached --quiet
if not errorlevel 1 (
  echo ERROR: C3 audit produced no staged evidence and the C3 tag is missing.
  exit /b 1
)
git commit -m "Freeze Task 45 confirmatory coalition matrix"
if errorlevel 1 exit /b 1
git tag -a "%T45_C3_TAG%" -m "Freeze Task 45 C3 confirmatory multiseed evidence"
if errorlevel 1 exit /b 1
goto run_c4

:run_c4
git status --porcelain | findstr . >nul
if not errorlevel 1 (
  echo ERROR: Working tree is not clean before C4.
  git status --short --branch
  exit /b 1
)

echo ===== RUNNING TASK 45 C4 SUMMARY =====
"%PYTHON_EXE%" scripts\summarize_task45_c4_coalition_curve_v416.py --project-root . --coalition-manifest "%T45_MANIFEST%" --warmup-root "%T45_WARMUP%" --c2-root "%T45_C2%" --c3-root "%T45_C3%" --c3-audit-dir "%T45_C3_AUDIT%" --output-dir "%T45_C4%"
if errorlevel 1 exit /b 1

git add -f "%T45_C4%"
if errorlevel 1 exit /b 1
git diff --cached --check
if errorlevel 1 exit /b 1

echo ===== TASK 45 CLOSEOUT READY =====
git diff --cached --stat
git status --short --branch
echo NEXT ACTION: Review the output, then commit and tag the staged C4 closeout.
exit /b 0
