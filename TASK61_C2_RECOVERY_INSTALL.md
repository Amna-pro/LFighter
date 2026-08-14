# Task 61 C2 recovery installation

This package preserves the aborted first attempt and restarts the entire 36 report matrix under one amended 1100 token output cap. None of the 24 completed reports from the aborted attempt are reused in the confirmatory comparison.

```bat
cd /d "%USERPROFILE%\LFighter-research"
call .venv\Scripts\activate.bat
powershell -NoProfile -Command "$z=Get-ChildItem '%USERPROFILE%\Downloads' -File -Filter 'LFighter_Task61_C2_Recovery_v4232*.zip' | Sort-Object LastWriteTime -Descending | Select-Object -First 1; if (!$z) { throw 'Task 61 C2 recovery package not found in Downloads' }; Write-Host 'Using:' $z.FullName; Expand-Archive -LiteralPath $z.FullName -DestinationPath '%USERPROFILE%\LFighter-research' -Force"
python -m pip install --upgrade-strategy only-if-needed -r requirements_task61_c2_recovery_v4232.txt
call scripts\run_task61_c2_recovery_v4232.bat
```

The recovery preflight must verify exactly 24 completed original reports, no completed report 25, the frozen C1 tag, and the supplied console failure evidence. The fresh run performs 36 new calls under an amended USD 1.50 accounting ceiling. It does not select, replace, or reuse outputs from the aborted attempt.

Every fresh call is journaled before submission. A normal interruption can resume from verified completed outputs. If an API call starts but does not produce a complete audited output, the batch stops and blocks an automatic retry so another hidden attempt cannot enter the study.
