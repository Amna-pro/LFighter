@echo off
setlocal
cd /d "%~dp0\.."

echo ===== RUNNING TASK 61 C0 PREREGISTRATION AUDIT =====
python scripts\audit_task61_c0_preregistration_v423.py
if errorlevel 1 exit /b 1

echo ===== BUILDING TASK 61 C1 MULTICASE EVIDENCE MATRIX =====
python scripts\build_task61_c1_evidence_matrix_v423.py
if errorlevel 1 exit /b 1

echo ===== RUNNING TASK 61 C1 EVIDENCE MATRIX AUDIT =====
python scripts\audit_task61_c1_evidence_matrix_v423.py
if errorlevel 1 exit /b 1

git add TASK61_C0_C1_INSTALL.md requirements_task61_v423.txt configs\TASK61_PREREGISTRATION_V423.md configs\task61_preregistration_v4230.json scripts\audit_task61_c0_preregistration_v423.py scripts\build_task61_c1_evidence_matrix_v423.py scripts\audit_task61_c1_evidence_matrix_v423.py scripts\run_task61_c0_c1_v423.bat
git add -f results\cic_iot_diad_task61_c0_preregistration_v423 results\cic_iot_diad_task61_c1_evidence_matrix_v423 results\cic_iot_diad_task61_c1_audit_v423

echo ===== TASK 61 C0 AND C1 READY FOR REVIEW =====
git diff --cached --stat
git status --short --branch
echo NEXT ACTION: Send the complete output for review before commit and tag.
endlocal
