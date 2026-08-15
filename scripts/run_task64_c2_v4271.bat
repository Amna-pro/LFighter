@echo off
setlocal
cd /d "%~dp0\.."

if not exist ".venv\Scripts\python.exe" (
  echo ERROR: .venv\Scripts\python.exe not found.
  exit /b 2
)

set "PY=.venv\Scripts\python.exe"

echo ===== TASK 64 C2 REPLACEMENT UNTOUCHED SEED RESERVATION =====
"%PY%" scripts\reserve_task64_c2_untouched_seeds_v4271.py --project-root "%CD%"
if errorlevel 1 exit /b %errorlevel%

echo.
echo ===== TASK 64 C2 INDEPENDENT AUDIT =====
"%PY%" scripts\audit_task64_c2_seed_reservation_v4271.py
if errorlevel 1 exit /b %errorlevel%

echo.
echo TASK 64 C2 COMPLETE - DO NOT START TASK 65 UNTIL THIS RESULT IS FROZEN
exit /b 0
