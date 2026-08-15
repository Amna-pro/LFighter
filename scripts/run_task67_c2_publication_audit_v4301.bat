@echo off
setlocal
cd /d "%~dp0\.."
echo ===== TASK 67 C2 AUDIT ONLY =====
echo NO FIGURE REGENERATION / NO TASK66 RERUN / NO NPZ ACCESS
".venv\Scripts\python.exe" -m py_compile scripts\audit_task67_c2_publication_outputs_v4301.py
if errorlevel 1 exit /b %errorlevel%
".venv\Scripts\python.exe" scripts\audit_task67_c2_publication_outputs_v4301.py
if errorlevel 1 exit /b %errorlevel%
echo.
type results\cic_iot_diad_task67_c2_publication_audit_v4301\task67c2_publication_integrity_audit_v4301.json
echo.
echo TASK 67 C2 AUDIT COMPLETE.
echo DO NOT RERUN TASK67 C1.
exit /b 0
