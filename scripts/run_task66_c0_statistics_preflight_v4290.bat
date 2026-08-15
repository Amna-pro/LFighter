@echo off
setlocal
cd /d "%~dp0\.."
echo ===== TASK 66 C0 STATISTICS IMPLEMENTATION PREFLIGHT =====
echo NO STATISTICS COMPUTED / NO TASK65 CSV LOADED / NO NPZ ACCESS
".venv\Scripts\python.exe" -m py_compile scripts\run_task66_c1_final_statistics_v4290.py scripts\audit_task66_c0_statistics_implementation_v4290.py
if errorlevel 1 exit /b %errorlevel%
".venv\Scripts\python.exe" scripts\audit_task66_c0_statistics_implementation_v4290.py
if errorlevel 1 exit /b %errorlevel%
echo.
type results\cic_iot_diad_task66_c0_statistics_preflight_v4290\task66c0_statistics_implementation_audit_v4290.json
echo.
echo TASK 66 C0 PREFLIGHT COMPLETE.
echo DO NOT RUN TASK66 C1 YET.
exit /b 0
