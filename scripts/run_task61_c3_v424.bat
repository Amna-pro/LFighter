@echo off
setlocal
cd /d "%~dp0.."

echo ===== RUNNING TASK 61 C3 POST HOC FAILURE DIAGNOSTIC =====
python scripts\run_task61_c3_failure_diagnostic_v424.py
if errorlevel 1 exit /b 1

echo ===== RUNNING TASK 61 C3 INDEPENDENT AUDIT =====
python scripts\audit_task61_c3_failure_diagnostic_v424.py
if errorlevel 1 exit /b 1

git add TASK61_C3_INSTALL.md requirements_task61_c3_v424.txt configs\task61_c3_diagnostic_protocol_v4240.json scripts\run_task61_c3_failure_diagnostic_v424.py scripts\audit_task61_c3_failure_diagnostic_v424.py scripts\run_task61_c3_v424.bat
git add -f results\cic_iot_diad_task61_c3_failure_diagnostic_v424 results\cic_iot_diad_task61_c3_audit_v424

echo ===== TASK 61 C3 READY FOR REVIEW =====
git diff --cached --stat
git status --short --branch
echo NEXT ACTION: Send the complete output for review before commit and tag.
endlocal
