@echo off
setlocal
cd /d "%~dp0.."

echo ===== RUNNING TASK 61 C2 PREFLIGHT =====
python scripts\preflight_task61_c2_v423.py
if errorlevel 1 exit /b 1

echo ===== RUNNING TASK 61 C2 MULTIREPORT MATRIX =====
if defined OPENAI_API_KEY (
  python scripts\run_task61_c2_multireport_v423.py
) else (
  powershell -NoProfile -Command "$code=1; $s=Read-Host 'Enter OpenAI API key' -AsSecureString; $b=[Runtime.InteropServices.Marshal]::SecureStringToBSTR($s); try {$env:OPENAI_API_KEY=[Runtime.InteropServices.Marshal]::PtrToStringBSTR($b); python scripts\run_task61_c2_multireport_v423.py; $code=$LASTEXITCODE} finally {[Runtime.InteropServices.Marshal]::ZeroFreeBSTR($b); Remove-Item Env:OPENAI_API_KEY -ErrorAction SilentlyContinue}; exit $code"
)
if errorlevel 1 exit /b 1

echo ===== SUMMARIZING TASK 61 C2 AUTOMATED EVALUATION =====
python scripts\summarize_task61_c2_evaluation_v423.py
if errorlevel 1 exit /b 1

echo ===== RUNNING TASK 61 C2 INDEPENDENT AUDIT =====
python scripts\audit_task61_c2_evaluation_v423.py
if errorlevel 1 exit /b 1

git add TASK61_C2_INSTALL.md requirements_task61_c2_v423.txt scripts\preflight_task61_c2_v423.py scripts\run_task61_c2_multireport_v423.py scripts\summarize_task61_c2_evaluation_v423.py scripts\audit_task61_c2_evaluation_v423.py scripts\run_task61_c2_v423.bat
git add -f results\cic_iot_diad_task61_c2_llm_reports_v423 results\cic_iot_diad_task61_c2_summary_v423 results\cic_iot_diad_task61_c2_audit_v423

echo ===== TASK 61 C2 READY FOR REVIEW =====
git diff --cached --check
if errorlevel 1 exit /b 1
git diff --cached --stat
git status --short --branch
echo NEXT ACTION: Send the complete output for review before commit and tag.
endlocal
