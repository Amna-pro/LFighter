@echo off
setlocal
cd /d "%~dp0\.."
echo ===== TASK 68 C2 V4.31.2 AUDIT-ONLY CORRECTION =====
echo NO TASK68 C1 RERUN / NO SCIENTIFIC RECOMPUTATION / NO NPZ ACCESS
".venv\Scripts\python.exe" -m py_compile scripts\audit_task68_c2_final_correction_v4312.py
if errorlevel 1 exit /b %errorlevel%
".venv\Scripts\python.exe" scripts\audit_task68_c2_final_correction_v4312.py
if errorlevel 1 exit /b %errorlevel%
echo.
type results\cic_iot_diad_task68_c2_final_audit_v4312\TASK68_C2_FINAL_AUDIT_CORRECTED_COMPLETE.json
echo.
echo TASK 68 C2 V4.31.2 CORRECTION COMPLETE.
echo DO NOT RERUN TASK68 C1.
exit /b 0
