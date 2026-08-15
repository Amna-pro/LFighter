@echo off
setlocal
cd /d "%~dp0\.."

echo ===== TASK 65 C2 POST-ACCESS AUDIT ONLY =====
if not exist ".venv\Scripts\python.exe" (
  echo ERROR: .venv Python not found.
  exit /b 2
)

if not exist "results\cic_iot_diad_task65_c1_one_shot_v4282\TASK65_FINAL_ACCESS_STARTED.json" (
  echo ERROR: Final-access marker missing.
  exit /b 2
)
if not exist "results\cic_iot_diad_task65_c1_one_shot_v4282\TASK65_FINAL_EVALUATION_COMPLETE.json" (
  echo ERROR: Final-evaluation completion marker missing.
  exit /b 2
)

".venv\Scripts\python.exe" -m py_compile scripts\audit_task65_c2_postaccess_v4283.py
if errorlevel 1 exit /b %errorlevel%

".venv\Scripts\python.exe" scripts\audit_task65_c2_postaccess_v4283.py
if errorlevel 1 exit /b %errorlevel%

echo.
type results\cic_iot_diad_task65_c2_postaccess_audit_v4283\task65c2_postaccess_integrity_audit_v4283.json
echo.
echo TASK 65 C2 AUDIT COMPLETE.
echo DO NOT RERUN THE TASK65 C1 FINAL EVALUATOR.
exit /b 0
