# Task 61 C2 installation

Run only after Task 61 C0 and C1 are committed and tagged.

```bat
cd /d "%USERPROFILE%\LFighter-research"
call .venv\Scripts\activate.bat
powershell -NoProfile -Command "$z=Get-ChildItem '%USERPROFILE%\Downloads' -File -Filter 'LFighter_Task61_C2_v4231*.zip' | Sort-Object LastWriteTime -Descending | Select-Object -First 1; if (!$z) { throw 'Task 61 C2 package not found in Downloads' }; Write-Host 'Using:' $z.FullName; Expand-Archive -LiteralPath $z.FullName -DestinationPath '%USERPROFILE%\LFighter-research' -Force"
python -m pip install --upgrade-strategy only-if-needed -r requirements_task61_c2_v423.txt
call scripts\run_task61_c2_v423.bat
```

The runner makes at most 36 calls and resumes from verified completion markers after interruption. It never replaces or selects among completed reports. C2 performs automated evaluation and prepares a blinded package for at least two independent human reviewers. No LLM superiority claim is permitted before the later human review closeout.
