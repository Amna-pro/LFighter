@echo off
setlocal EnableExtensions
cd /d "%~dp0\.."
for /f %%H in ('git rev-parse HEAD') do set "HEADSHA=%%H"
for /f %%T in ('git rev-list -n 1 task66-c0-statistics-implementation-frozen-v4290') do set "TAGSHA=%%T"
if not defined TAGSHA (
  echo ERROR: frozen Task66 C0 tag missing.
  exit /b 2
)
if /I not "%HEADSHA%"=="%TAGSHA%" (
  echo ERROR: HEAD must exactly equal Task66 C0 frozen implementation tag.
  exit /b 2
)
git status --porcelain > "%TEMP%\task66dirty.txt"
for %%A in ("%TEMP%\task66dirty.txt") do if not "%%~zA"=="0" (
  echo ERROR: repository is not clean.
  type "%TEMP%\task66dirty.txt"
  del "%TEMP%\task66dirty.txt" >nul 2>&1
  exit /b 2
)
del "%TEMP%\task66dirty.txt" >nul 2>&1
".venv\Scripts\python.exe" scripts\run_task66_c1_final_statistics_v4290.py
exit /b %errorlevel%
