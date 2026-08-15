@echo off
setlocal
cd /d "%~dp0\.."
echo ===== TASK 66 C2 AUDIT ONLY =====
echo NO STATISTICS RERUN / NO TASK65 RERUN / NO NPZ ACCESS
".venv\Scripts\python.exe" -m py_compile scripts\audit_task66_c2_final_statistics_v4291.py
if errorlevel 1 exit /b %errorlevel%
".venv\Scripts\python.exe" scripts\audit_task66_c2_final_statistics_v4291.py
if errorlevel 1 exit /b %errorlevel%
echo.
type results\cic_iot_diad_task66_c2_statistics_audit_v4291\task66c2_statistics_integrity_audit_v4291.json
echo.
echo TASK 66 C2 AUDIT COMPLETE.
echo DO NOT RERUN TASK66 C1.
exit /b 0
