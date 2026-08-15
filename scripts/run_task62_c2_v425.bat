@echo off
setlocal
cd /d "%~dp0.."

echo ===== RUNNING TASK 62 C2 EXECUTION PREFLIGHT =====
python scripts\preflight_task62_c2_v425.py
if errorlevel 1 exit /b 1

echo ===== RUNNING TASK 62 C2 HYBRID REPORT EXECUTION =====
if defined OPENAI_API_KEY (
  python scripts\run_task62_c2_hybrid_v425.py
) else (
  powershell -NoProfile -Command "$code=1; $s=Read-Host 'Enter OpenAI API key' -AsSecureString; $b=[Runtime.InteropServices.Marshal]::SecureStringToBSTR($s); try {$env:OPENAI_API_KEY=[Runtime.InteropServices.Marshal]::PtrToStringBSTR($b); python scripts\run_task62_c2_hybrid_v425.py; $code=$LASTEXITCODE} finally {[Runtime.InteropServices.Marshal]::ZeroFreeBSTR($b); Remove-Item Env:OPENAI_API_KEY -ErrorAction SilentlyContinue}; exit $code"
)
if errorlevel 1 exit /b 1

echo ===== SUMMARIZING TASK 62 C2 AUTOMATED EVALUATION =====
python scripts\summarize_task62_c2_hybrid_v425.py
if errorlevel 1 exit /b 1

echo ===== RUNNING TASK 62 C2 INDEPENDENT AUDIT =====
python scripts\audit_task62_c2_hybrid_v425.py
if errorlevel 1 exit /b 1

git add TASK62_C2_INSTALL.md requirements_task62_c2_v425.txt scripts\task62_c2_common_v425.py scripts\preflight_task62_c2_v425.py scripts\run_task62_c2_hybrid_v425.py scripts\summarize_task62_c2_hybrid_v425.py scripts\audit_task62_c2_hybrid_v425.py scripts\run_task62_c2_v425.bat
git add -f results\cic_iot_diad_task62_c2_preflight_v425 results\cic_iot_diad_task62_c2_hybrid_reports_v425 results\cic_iot_diad_task62_c2_summary_v425 results\cic_iot_diad_task62_c2_audit_v425

echo ===== TASK 62 C2 READY FOR REVIEW =====
git diff --cached --check
if errorlevel 1 exit /b 1
git diff --cached --stat
git status --short --branch
echo NEXT ACTION: Send the complete output for review before commit and tag.
endlocal
