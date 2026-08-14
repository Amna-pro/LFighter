@echo off
setlocal
cd /d "%~dp0\.."

echo ===== RUNNING TASK 57 C0 AUDIT =====
python scripts\audit_task57_c0_preregistration_v419.py --project-root . --config configs\task57_preregistration_v4190.json --protocol configs\TASK57_PREREGISTRATION_V419.md --output-dir results\cic_iot_diad_task57_c0_preregistration_v419
if errorlevel 1 exit /b 1

echo ===== RUNNING TASK 57 C1 PREFLIGHT =====
python scripts\preflight_task57_c1_v419.py --project-root . --config configs\task57_preregistration_v4190.json --checkpoint-inventory results\cic_iot_diad_task55_c1_preflight_v417\tables\task55c1_checkpoint_inventory.csv --task55-c2-root results\cic_iot_diad_task55_c2_seed7_v417 --task55-c3-root results\cic_iot_diad_task55_c3_multiseed_v417 --task55-c3-audit-root results\cic_iot_diad_task55_c3_audit_v417 --task56-summary-root results\cic_iot_diad_task56_c2_summary_v418 --task56-audit-root results\cic_iot_diad_task56_c2_audit_v418 --output-dir results\cic_iot_diad_task57_c1_preflight_v419
if errorlevel 1 exit /b 1

git add TASK57_C0_C1_INSTALL.md configs\TASK57_PREREGISTRATION_V419.md configs\task57_preregistration_v4190.json requirements_task57_v419.txt scripts\audit_task57_c0_preregistration_v419.py scripts\preflight_task57_c1_v419.py scripts\run_task57_c0_c1_v419.bat
if errorlevel 1 exit /b 1
git add -f results\cic_iot_diad_task57_c0_preregistration_v419 results\cic_iot_diad_task57_c1_preflight_v419
if errorlevel 1 exit /b 1

git diff --cached --check
if errorlevel 1 exit /b 1

echo ===== TASK 57 C0 AND C1 READY FOR REVIEW =====
git diff --cached --stat
git status --short --branch
echo NEXT ACTION: Send the complete output for review before commit and tag.
endlocal
