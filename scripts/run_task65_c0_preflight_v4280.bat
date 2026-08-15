@echo off
setlocal
cd /d "%~dp0\.."

if not exist ".venv\Scripts\python.exe" (
  echo ERROR: .venv\Scripts\python.exe not found.
  exit /b 2
)

echo ===== TASK 65 C0 OUTCOME-BLIND PREFLIGHT =====
".venv\Scripts\python.exe" scripts\preflight_task65_c0_v4280.py
if errorlevel 1 exit /b %errorlevel%

echo.
echo ===== PREFLIGHT DECISION =====
type results\cic_iot_diad_task65_c0_preflight_v4280\task65c0_preflight_decision.json

echo.
echo ===== STATIC EXECUTION INVENTORY =====
type results\cic_iot_diad_task65_c0_preflight_v4280\task65c0_execution_inventory.json

echo.
echo TASK 65 C0 COMPLETE - DO NOT START C1 YET
exit /b 0
