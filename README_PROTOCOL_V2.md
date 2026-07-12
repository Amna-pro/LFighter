# CIC IoT-DIAD 2024, Data Protocol V2

This package replaces the prototype random row split with a research-oriented protocol.

## Scientific controls

- Entire capture files are held out where each broad class has at least three captures.
- Classes with fewer than three captures use chronological row blocks, not random rows.
- Exact prepared feature duplicates are removed across train, validation, and test.
- A row-hash overlap audit must report zero overlap before the script completes.
- Imputation, variance filtering, feature selection, and scaling are fitted on training data only.
- Three feature configurations are produced, including a strict behavioral version that excludes ports and protocol.
- A natural-distribution test set and a class-diagnostic test set are saved separately.

## Research artifacts

Every stage produces source CSV tables and publication figures in both PNG and PDF.

## Run preparation

```cmd
cd /d "%USERPROFILE%\LFighter-research"
.venv\Scripts\activate
python scripts\prepare_iot_diad_v2.py --dataset-root "%USERPROFILE%\Documents\Ahmed Client work\Ah\Dataset\IoT device identification and anomaly detection dataset (CIC IoT-DIAD 2024)"
```

Expected output folder:

```text
data\processed\cic_iot_diad_2024_v2
```

## Run the centralized benchmark after preparation is reviewed

```cmd
python scripts\benchmark_iot_diad_v2.py
```

The default benchmark uses four models and five random seeds, SGD logistic regression, Random Forest, Extra Trees, and a PyTorch MLP. On CPU this can take substantial time. The preparation results should be reviewed before starting it.

## Generated outputs

- `tables/*.csv`
- `figures/*.png`
- `figures/*.pdf`
- `arrays/*.npz`
- `samples/*.parquet`
- `preprocessors/*`
- `metadata_v2.json`
