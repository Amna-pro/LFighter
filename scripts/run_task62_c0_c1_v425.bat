@echo off
setlocal
cd /d "%~dp0.."

echo ===== RUNNING TASK 62 C0 PREREGISTRATION AUDIT =====
python scripts\audit_task62_c0_preregistration_v425.py
if errorlevel 1 exit /b 1

echo ===== BUILDING TASK 62 C1 HYBRID CONTENT MATRIX =====
python scripts\build_task62_c1_hybrid_matrix_v425.py
if errorlevel 1 exit /b 1

echo ===== RUNNING TASK 62 C1 INDEPENDENT AUDIT =====
python scripts\audit_task62_c1_hybrid_matrix_v425.py
if errorlevel 1 exit /b 1

git add TASK62_C0_C1_INSTALL.md requirements_task62_v425.txt configs\TASK62_PREREGISTRATION_V425.md configs\task62_preregistration_v4250.json prompts\task62_hybrid_narrative_system_v425.txt prompts\task62_hybrid_narrative_user_v425.txt schemas\lfighter_hybrid_narrative_v1.schema.json scripts\audit_task62_c0_preregistration_v425.py scripts\build_task62_c1_hybrid_matrix_v425.py scripts\audit_task62_c1_hybrid_matrix_v425.py scripts\run_task62_c0_c1_v425.bat
git add -f results\cic_iot_diad_task62_c0_preregistration_v425 results\cic_iot_diad_task62_c1_hybrid_matrix_v425 results\cic_iot_diad_task62_c1_audit_v425

echo ===== TASK 62 C0 AND C1 READY FOR REVIEW =====
git diff --cached --stat
git status --short --branch
echo NEXT ACTION: Send the complete output for review before commit and tag.
endlocal
