@echo off
setlocal EnableExtensions
cd /d "%~dp0\.."
if not exist ".venv\Scripts\python.exe" exit /b 2

for /f %%H in ('git rev-parse HEAD') do set "HEADSHA=%%H"
for /f %%T in ('git rev-list -n 1 task65-c1-implementation-frozen-v4282') do set "TAGSHA=%%T"
if not defined TAGSHA (
  echo ERROR: frozen Task65 C1 V4.28.2 implementation tag does not resolve.
  exit /b 2
)
if /I not "%HEADSHA%"=="%TAGSHA%" (
  echo ERROR: HEAD is not exactly the frozen Task65 C1 V4.28.2 implementation tag.
  exit /b 2
)

git status --porcelain > "%TEMP%\task65_dirty.txt"
for %%A in ("%TEMP%\task65_dirty.txt") do if not "%%~zA"=="0" (
  echo ERROR: repository is not clean.
  type "%TEMP%\task65_dirty.txt"
  del "%TEMP%\task65_dirty.txt" >nul 2>&1
  exit /b 2
)
del "%TEMP%\task65_dirty.txt" >nul 2>&1

echo [1/3] Frozen training with reserved final arrays blocked...
".venv\Scripts\python.exe" scripts\run_task65_c1_training_v4282.py
if errorlevel 1 exit /b %errorlevel%

echo [2/3] ONE-SHOT reserved final evaluation...
".venv\Scripts\python.exe" scripts\evaluate_task65_c1_one_shot_v4282.py
if errorlevel 1 exit /b %errorlevel%

echo [3/3] Integrity audit...
".venv\Scripts\python.exe" scripts\audit_task65_c1_results_v4282.py
if errorlevel 1 exit /b %errorlevel%

echo TASK 65 C1 COMPLETE. DO NOT RERUN.
exit /b 0
