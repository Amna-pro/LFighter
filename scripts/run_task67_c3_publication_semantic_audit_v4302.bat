@echo off
setlocal
cd /d "%~dp0\.."
echo ===== TASK 67 C3 AUDIT-ONLY CORRECTION =====
echo NO OUTPUT MODIFICATION / NO REGENERATION / NO TASK66 RERUN / NO NPZ
".venv\Scripts\python.exe" -m py_compile scripts\audit_task67_c3_publication_semantics_v4302.py
if errorlevel 1 exit /b %errorlevel%
".venv\Scripts\python.exe" scripts\audit_task67_c3_publication_semantics_v4302.py
if errorlevel 1 exit /b %errorlevel%
echo.
type results\cic_iot_diad_task67_c3_publication_audit_v4302\task67c3_publication_integrity_audit_v4302.json
echo.
echo TASK 67 C3 AUDIT COMPLETE.
echo DO NOT RERUN TASK67 C1.
exit /b 0
