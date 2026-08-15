@echo off
setlocal
cd /d "%~dp0\.."
echo ===== TASK 68 C0 INDEPENDENT FINAL AUDIT PREFLIGHT =====
echo STATIC ONLY - NO FINAL AUDIT EXECUTION - NO SCIENTIFIC CSV READ
".venv\Scripts\python.exe" -m py_compile scripts\run_task68_c1_independent_final_audit_v4310.py scripts\audit_task68_c0_implementation_v4310.py
if errorlevel 1 exit /b %errorlevel%
".venv\Scripts\python.exe" scripts\audit_task68_c0_implementation_v4310.py
if errorlevel 1 exit /b %errorlevel%
echo.
type results\cic_iot_diad_task68_c0_final_audit_preflight_v4310\task68c0_final_audit_implementation_preflight_v4310.json
echo.
echo TASK68 C0 PREFLIGHT COMPLETE.
echo DO NOT RUN TASK68 C1 YET.
exit /b 0
