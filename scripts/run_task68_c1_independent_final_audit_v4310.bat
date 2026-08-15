@echo off
setlocal EnableExtensions
cd /d "%~dp0\.."
for /f %%H in ('git rev-parse HEAD') do set "HEADSHA=%%H"
for /f %%T in ('git rev-list -n 1 task68-c0-independent-final-audit-frozen-v4310') do set "TAGSHA=%%T"
if not defined TAGSHA (
  echo ERROR: Task68 C0 frozen tag missing.
  exit /b 2
)
if /I not "%HEADSHA%"=="%TAGSHA%" (
  echo ERROR: HEAD must equal the frozen Task68 C0 audit tag.
  exit /b 2
)
git status --porcelain > "%TEMP%\task68dirty.txt"
for %%A in ("%TEMP%\task68dirty.txt") do if not "%%~zA"=="0" (
  echo ERROR: repository is not clean.
  type "%TEMP%\task68dirty.txt"
  del "%TEMP%\task68dirty.txt" >nul 2>&1
  exit /b 2
)
del "%TEMP%\task68dirty.txt" >nul 2>&1
".venv\Scripts\python.exe" scripts\run_task68_c1_independent_final_audit_v4310.py
exit /b %errorlevel%
