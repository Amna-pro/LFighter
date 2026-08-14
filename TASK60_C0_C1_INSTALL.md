# Task 60 C0 and C1 installation

Run from Windows Command Prompt after Task 59 is frozen and pushed.

```bat
cd /d "%USERPROFILE%\LFighter-research"
call .venv\Scripts\activate.bat
git switch feature/cic-iot-diad-task60-forensic-reporting-v422 2>nul || git switch -c feature/cic-iot-diad-task60-forensic-reporting-v422 task59-c1-evidence-schema-frozen-v4211
powershell -NoProfile -Command "$z=Get-ChildItem '%USERPROFILE%\Downloads' -File -Filter 'LFighter_Task60_C0_C1_v422*.zip' | Sort-Object LastWriteTime -Descending | Select-Object -First 1; if (!$z) { throw 'Task 60 package not found in Downloads' }; Write-Host 'Using:' $z.FullName; Expand-Archive -LiteralPath $z.FullName -DestinationPath '%USERPROFILE%\LFighter-research' -Force"
python -m pip install --upgrade-strategy only-if-needed -r requirements_task60_v422.txt
call scripts\run_task60_c0_c1_v422.bat
```

If `OPENAI_API_KEY` is not already defined, the batch file requests it through a hidden PowerShell prompt. The key is passed only to the report process and is not saved. A completed result is reused to avoid an accidental second billed call.
