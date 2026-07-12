# CIC IoT-DIAD 2024, Step 1 Clean Centralized Baseline

This package adds the first implementation stage to LFighter. It does not yet add federated training, poisoning, SHAP, or an LLM.

## What this stage does

- Reads all 129 flow-based CSV files in chunks.
- Recovers the five small headerless DoS CSV files.
- Assigns the eight broad labels from folder names.
- Excludes Flow ID, source IP, destination IP, timestamp, and the original placeholder label from model features.
- Replaces infinity with missing values.
- Removes duplicate sampled rows.
- Fits median imputation, zero-variance feature removal, and standardization on training data only.
- Saves stratified train, validation, and test arrays.
- Trains a class-weighted CPU MLP.
- Reports accuracy, balanced accuracy, macro F1, per-class metrics, and a confusion matrix.

## Install into the repository

Extract this ZIP directly into:

```text
C:\Users\Amm\LFighter-research
```

The ZIP adds new files and does not overwrite the original LFighter source files.

## Prepare the dataset

Open CMD, activate the existing environment, then run:

```cmd
cd /d "%USERPROFILE%\LFighter-research"
.venv\Scripts\activate
python scripts\prepare_iot_diad.py --dataset-root "%USERPROFILE%\Documents\Ahmed Client work\Ah\Dataset\IoT device identification and anomaly detection dataset (CIC IoT-DIAD 2024)"
```

Expected output folder:

```text
C:\Users\Amm\LFighter-research\data\processed\cic_iot_diad_2024
```

The scan reads roughly 9.8 GB and can take several minutes. Do not close the CMD window.

## Train the clean baseline

After preparation completes:

```cmd
python scripts\train_iot_diad_baseline.py
```

Expected results folder:

```text
C:\Users\Amm\LFighter-research\results\cic_iot_diad_centralized
```

Upload these files after training:

```text
summary.json
classification_report.csv
confusion_matrix.csv
training_history.csv
```

Also upload the preparation metadata:

```text
data\processed\cic_iot_diad_2024\metadata.json
```

## Save the implementation in Git

After both scripts run successfully:

```cmd
git status
git add src\iot_diad_dataset.py src\tabular_models.py scripts config requirements_iot_diad.txt README_STEP1.md
git commit -m "Add CIC IoT-DIAD centralized baseline"
```

Do not push until the results are reviewed.
