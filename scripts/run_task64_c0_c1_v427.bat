@echo off
setlocal
cd /d "%~dp0\.."

if not exist ".venv\Scripts\python.exe" (
  echo ERROR: .venv\Scripts\python.exe not found.
  exit /b 2
)

set "PY=.venv\Scripts\python.exe"

echo ===== TASK 64 C0/C1 UNTOUCHED SEED RESERVATION =====
"%PY%" scripts\reserve_task64_untouched_seeds_v427.py --project-root "%CD%"
if errorlevel 1 exit /b %errorlevel%

echo.
echo ===== TASK 64 INDEPENDENT AUDIT =====
"%PY%" scripts\audit_task64_seed_reservation_v427.py
if errorlevel 1 exit /b %errorlevel%

echo.
echo TASK 64 C0/C1 COMPLETE
exit /b 0
