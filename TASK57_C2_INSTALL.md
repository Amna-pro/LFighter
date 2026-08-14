# Task 57 C2 installation

This package runs the preregistered Task 57 attribution recovery analysis after the C0 and C1 freeze tag. It reuses existing Task 55 and Task 56 attribution tensors. It performs no training and no new SHAP evaluations.

## Install and run from Windows CMD

```bat
cd /d "%USERPROFILE%\LFighter-research"
call .venv\Scripts\activate.bat

powershell -NoProfile -Command "Expand-Archive -LiteralPath '%USERPROFILE%\Downloads\LFighter_Task57_C2_v419.zip' -DestinationPath '%USERPROFILE%\LFighter-research' -Force"

python -m pip install --upgrade-strategy only-if-needed -r requirements_task57_c2_v419.txt
call scripts\run_task57_c2_v419.bat
```

The summary retains all twelve triplets and all 69 features. Rejected and oracle clean states remain explicitly unavailable. The supported claim concerns movement toward the frozen preattack clean reference only.

Task 58 remains responsible for publication figures and fixed case studies. Send the complete CMD output for review before committing or tagging.
