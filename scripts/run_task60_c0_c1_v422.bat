@echo off
setlocal
cd /d "%~dp0\.."

echo ===== RUNNING TASK 60 C0 PREREGISTRATION AUDIT =====
python scripts\audit_task60_c0_preregistration_v422.py
if errorlevel 1 exit /b 1

echo ===== RUNNING TASK 60 C1 DRY RUN =====
python scripts\run_task60_c1_report_generators_v422.py --dry-run
if errorlevel 1 exit /b 1

echo ===== RUNNING TASK 60 C1 SINGLE LLM CALL =====
if defined OPENAI_API_KEY (
  python scripts\run_task60_c1_report_generators_v422.py
) else (
  powershell -NoProfile -ExecutionPolicy Bypass -Command "$s=Read-Host 'Enter OpenAI API key' -AsSecureString; $b=[Runtime.InteropServices.Marshal]::SecureStringToBSTR($s); $code=1; try {$env:OPENAI_API_KEY=[Runtime.InteropServices.Marshal]::PtrToStringBSTR($b); & python 'scripts\run_task60_c1_report_generators_v422.py'; $code=$LASTEXITCODE} finally {[Runtime.InteropServices.Marshal]::ZeroFreeBSTR($b); Remove-Item Env:OPENAI_API_KEY -ErrorAction SilentlyContinue}; exit $code"
)
if errorlevel 1 exit /b 1

echo ===== RUNNING TASK 60 C1 INDEPENDENT AUDIT =====
python scripts\audit_task60_c1_report_generators_v422.py
if errorlevel 1 exit /b 1

git add TASK60_C0_C1_INSTALL.md requirements_task60_v422.txt configs\TASK60_PREREGISTRATION_V422.md configs\task60_preregistration_v4220.json prompts\task60_forensic_report_system_v422.txt prompts\task60_forensic_report_user_v422.txt schemas\lfighter_forensic_report_v1.schema.json scripts\task60_report_contract_v422.py scripts\audit_task60_c0_preregistration_v422.py scripts\run_task60_c1_report_generators_v422.py scripts\audit_task60_c1_report_generators_v422.py scripts\run_task60_c0_c1_v422.bat results\cic_iot_diad_task60_c0_preregistration_v422 results\cic_iot_diad_task60_c1_reporting_v422 results\cic_iot_diad_task60_c1_audit_v422

echo ===== TASK 60 C0 AND C1 READY FOR REVIEW =====
git diff --cached --stat
git status --short --branch
echo NEXT ACTION: Send the complete output for review before commit and tag.
endlocal
