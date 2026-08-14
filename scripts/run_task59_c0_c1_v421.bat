@echo off
setlocal
cd /d "%~dp0\.."

echo ===== RUNNING TASK 59 C0 PREREGISTRATION AUDIT =====
python scripts\audit_task59_c0_preregistration_v421.py --project-root .
if errorlevel 1 exit /b 1

echo ===== BUILDING TASK 59 C1 GROUNDED EVIDENCE SCHEMA =====
python scripts\build_task59_c1_evidence_schema_v421.py --project-root .
if errorlevel 1 exit /b 1

echo ===== RUNNING TASK 59 C1 INDEPENDENT AUDIT =====
python scripts\audit_task59_c1_evidence_schema_v421.py --project-root .
if errorlevel 1 exit /b 1

git add TASK59_C0_C1_INSTALL.md requirements_task59_v421.txt configs\TASK59_PREREGISTRATION_V421.md configs\task59_preregistration_v4210.json schemas\lfighter_forensic_evidence_v1.schema.json scripts\audit_task59_c0_preregistration_v421.py scripts\build_task59_c1_evidence_schema_v421.py scripts\audit_task59_c1_evidence_schema_v421.py scripts\run_task59_c0_c1_v421.bat
git add -f results\cic_iot_diad_task59_c0_preregistration_v421 results\cic_iot_diad_task59_c1_schema_v421 results\cic_iot_diad_task59_c1_audit_v421
if errorlevel 1 exit /b 1

git diff --cached --check
if errorlevel 1 exit /b 1

echo ===== TASK 59 C0 AND C1 READY FOR REVIEW =====
git diff --cached --stat
git status --short --branch
echo NEXT ACTION: Send the complete output for review before commit and tag.
endlocal
