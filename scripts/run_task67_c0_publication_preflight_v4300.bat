@echo off
setlocal
cd /d "%~dp0\.."
echo ===== TASK 67 C0 PUBLICATION IMPLEMENTATION PREFLIGHT =====
echo NO FIGURES / NO TABLES / NO STATISTICS RERUN / NO NPZ ACCESS
".venv\Scripts\python.exe" -m py_compile scripts\run_task67_c1_publication_outputs_v4300.py scripts\audit_task67_c0_publication_implementation_v4300.py
if errorlevel 1 exit /b %errorlevel%
".venv\Scripts\python.exe" scripts\audit_task67_c0_publication_implementation_v4300.py
if errorlevel 1 exit /b %errorlevel%
echo.
type results\cic_iot_diad_task67_c0_publication_preflight_v4300\task67c0_publication_implementation_audit_v4300.json
echo.
echo TASK 67 C0 PREFLIGHT COMPLETE.
echo DO NOT RUN TASK67 C1 YET.
exit /b 0
