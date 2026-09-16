# Seoul and Mobile Data Analysis Design

## Goal

Create a reproducible preprocessing and exploratory data analysis workflow for the Seoul commercial-district CSV files and `data/raw/mobile.xlsx`.

## Scope

- Read all Seoul CSVs under `data/raw/seoul` and all sheets in `mobile.xlsx`.
- Normalize column names, numeric/date types, missing values, duplicates, and basic outliers without inventing business values.
- Write cleaned datasets under `data/processed/seoul` and `data/processed/mobile`.
- Generate quality summaries, descriptive statistics, and PNG charts under `reports/`.
- Keep Seoul and mobile datasets separate unless a validated common key exists.

## Design

`src/preprocess.py` will expose deterministic loaders/cleaners and a CLI that writes processed CSVs and a JSON quality report. `src/eda.py` will read the processed outputs, generate compact summaries and charts, and write a Markdown report. The existing notebook will remain a lightweight inspection entry point rather than the source of truth.

## Validation

- Every source CSV is processed once and row counts are recorded.
- Every ZIP-backed CSV is already extracted and readable.
- Processed outputs have no duplicate column names and numeric columns are numeric where conversion is lossless.
- Scripts run from the repository root with the project Python environment.
