@echo off
setlocal
cd /d "%~dp0\.."

echo ===== RUNNING TASK 57 C2 RECOVERY ANALYSIS =====
python scripts\run_task57_c2_recovery_v419.py --project-root . --config configs\task57_preregistration_v4190.json --c1-root results\cic_iot_diad_task57_c1_preflight_v419 --task55-c2-root results\cic_iot_diad_task55_c2_seed7_v417 --task55-c3-root results\cic_iot_diad_task55_c3_multiseed_v417 --task56-execution-root results\cic_iot_diad_task56_c2_execution_v418 --task56-summary-root results\cic_iot_diad_task56_c2_summary_v418 --output-dir results\cic_iot_diad_task57_c2_summary_v419
if errorlevel 1 exit /b 1

echo ===== RUNNING TASK 57 C2 INTEGRITY AUDIT =====
python scripts\audit_task57_c2_recovery_v419.py --project-root . --config configs\task57_preregistration_v4190.json --c1-root results\cic_iot_diad_task57_c1_preflight_v419 --task55-c2-root results\cic_iot_diad_task55_c2_seed7_v417 --task55-c3-root results\cic_iot_diad_task55_c3_multiseed_v417 --summary-root results\cic_iot_diad_task57_c2_summary_v419 --output-dir results\cic_iot_diad_task57_c2_audit_v419
if errorlevel 1 exit /b 1

git add TASK57_C2_INSTALL.md requirements_task57_c2_v419.txt scripts\run_task57_c2_recovery_v419.py scripts\audit_task57_c2_recovery_v419.py scripts\run_task57_c2_v419.bat
if errorlevel 1 exit /b 1
git add -f results\cic_iot_diad_task57_c2_summary_v419 results\cic_iot_diad_task57_c2_audit_v419
if errorlevel 1 exit /b 1

git diff --cached --check
if errorlevel 1 exit /b 1

echo ===== TASK 57 C2 READY FOR REVIEW =====
git diff --cached --stat
git status --short --branch
echo NEXT ACTION: Send the complete output for review before commit and tag.
endlocal
