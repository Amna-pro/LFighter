@echo off
setlocal
cd /d "%~dp0\.."
".venv\Scripts\python.exe" -m py_compile ^
 scripts\task65_guard_v4282\sitecustomize.py ^
 scripts\run_task65_c1_training_v4282.py ^
 scripts\evaluate_task65_c1_one_shot_v4282.py ^
 scripts\audit_task65_c1_results_v4282.py ^
 scripts\audit_task65_c1_implementation_v4282.py
if errorlevel 1 exit /b %errorlevel%
".venv\Scripts\python.exe" scripts\audit_task65_c1_implementation_v4282.py
if errorlevel 1 exit /b %errorlevel%
type results\cic_iot_diad_task65_c1_implementation_preflight_v4282\task65c1_implementation_audit_v4282.json
echo.
echo V4.28.2 STATIC PREFLIGHT COMPLETE - NO TRAINING / NO FINAL TEST ACCESS.
exit /b 0
