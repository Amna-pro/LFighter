@echo off
setlocal
cd /d "%~dp0\.."

if not exist ".venv\Scripts\python.exe" (
  echo ERROR: .venv\Scripts\python.exe not found.
  exit /b 2
)

echo ===== TASK 63 C2 FINAL EVALUATION MANIFEST COMPLETION =====
.venv\Scripts\python.exe scripts\audit_task63_c2_manifest_v4262.py
if errorlevel 1 exit /b %errorlevel%

echo.
echo TASK 63 C2 COMPLETE - DO NOT START TASK 65 YET
echo NEXT: freeze this amendment, then run Task 64 C2 replacement seed reservation.
exit /b 0
