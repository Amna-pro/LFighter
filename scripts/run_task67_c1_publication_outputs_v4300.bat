@echo off
setlocal EnableExtensions
cd /d "%~dp0\.."
for /f %%H in ('git rev-parse HEAD') do set "HEADSHA=%%H"
for /f %%T in ('git rev-list -n 1 task67-c0-publication-implementation-frozen-v4300') do set "TAGSHA=%%T"
if not defined TAGSHA (
  echo ERROR: Task67 C0 frozen tag missing.
  exit /b 2
)
if /I not "%HEADSHA%"=="%TAGSHA%" (
  echo ERROR: HEAD must equal the Task67 C0 frozen tag.
  exit /b 2
)
git status --porcelain > "%TEMP%\task67dirty.txt"
for %%A in ("%TEMP%\task67dirty.txt") do if not "%%~zA"=="0" (
  echo ERROR: repository is not clean.
  type "%TEMP%\task67dirty.txt"
  del "%TEMP%\task67dirty.txt" >nul 2>&1
  exit /b 2
)
del "%TEMP%\task67dirty.txt" >nul 2>&1
".venv\Scripts\python.exe" scripts\run_task67_c1_publication_outputs_v4300.py
exit /b %errorlevel%
