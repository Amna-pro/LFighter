@echo off
setlocal
cd /d "%~dp0\.."

echo ===== RUNNING TASK 58 C0 PREREGISTRATION AUDIT =====
python scripts\audit_task58_c0_preregistration_v420.py --project-root .
if errorlevel 1 exit /b 1

echo ===== RUNNING TASK 58 C1 PUBLICATION FIGURES =====
python scripts\generate_task58_c1_xai_figures_v420.py --project-root .
if errorlevel 1 exit /b 1

echo ===== RUNNING TASK 58 C1 PUBLICATION AUDIT =====
python scripts\audit_task58_c1_publication_v420.py --project-root .
if errorlevel 1 exit /b 1

git add TASK58_C0_C1_INSTALL.md requirements_task58_v420.txt configs\TASK58_PREREGISTRATION_V420.md configs\task58_preregistration_v4200.json scripts\audit_task58_c0_preregistration_v420.py scripts\generate_task58_c1_xai_figures_v420.py scripts\audit_task58_c1_publication_v420.py scripts\run_task58_c0_c1_v420.bat
git add -f results\cic_iot_diad_task58_c0_preregistration_v420 results\cic_iot_diad_task58_c1_publication_v420 results\cic_iot_diad_task58_c1_audit_v420
if errorlevel 1 exit /b 1

git diff --cached --check
if errorlevel 1 exit /b 1

echo ===== TASK 58 C0 AND C1 READY FOR REVIEW =====
git diff --cached --stat
git status --short --branch
echo NEXT ACTION: Send the complete output for review before commit and tag.
endlocal
