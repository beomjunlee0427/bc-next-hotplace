## Project structure

```text
data/raw/          Original input data
data/processed/    Cleaned analysis data
notebooks/         Interactive analysis notebooks
reports/           Validation reports
src/               Python scripts
.venv/             Project virtual environment
```

Run the preprocessing script from the project root:

```powershell
uv run python src/profile_data.py
uv run python src/preprocess.py
uv run python src/eda.py
```

The cleaned Seoul tables are written to `data/processed/seoul/`, the cleaned
mobile workbook sheets to `data/processed/mobile/`, and quality/EDA outputs to
`reports/`. Seoul and mobile sources are kept separate because no validated
common join key was assumed.

## BC카드 프로젝트 산출물

```powershell
uv run python src/build_project_data.py
```

This creates `data/processed/project/merged_panel_data.csv`,
`feature_data.csv`, and `target_data.csv`, plus inventory and quality reports
under `reports/project/`. The current target is next-quarter sales growth; it
is a proxy for the requested three-month target because the available Seoul
source is quarterly.

`data/processed/preprocessed_data.csv`는 2026년 1~6월 BC카드 데이터로
분류하여 현재 학습·평가에서는 제외하고, 향후 예측 결과를 실제 상권 후보로
선정하는 단계에서만 사용합니다.
