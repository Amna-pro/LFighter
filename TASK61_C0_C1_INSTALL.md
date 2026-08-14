# Task 61 C0 and C1 installation

Complete and freeze Task 60 first. Then run the following commands from Windows Command Prompt.

```bat
cd /d "%USERPROFILE%\LFighter-research"
call .venv\Scripts\activate.bat
git switch feature/cic-iot-diad-task61-llm-evaluation-v423 2>nul || git switch -c feature/cic-iot-diad-task61-llm-evaluation-v423 task60-c1-forensic-reporting-frozen-v4221
powershell -NoProfile -Command "$z=Get-ChildItem '%USERPROFILE%\Downloads' -File -Filter 'LFighter_Task61_C0_C1_v423*.zip' | Sort-Object LastWriteTime -Descending | Select-Object -First 1; if (!$z) { throw 'Task 61 package not found in Downloads' }; Write-Host 'Using:' $z.FullName; Expand-Archive -LiteralPath $z.FullName -DestinationPath '%USERPROFILE%\LFighter-research' -Force"
python -m pip install --upgrade-strategy only-if-needed -r requirements_task61_v423.txt
call scripts\run_task61_c0_c1_v423.bat
```

This stage makes zero API calls. It freezes and audits the 12 case evidence matrix and the 36 call evaluation plan before Task 61 C2.
