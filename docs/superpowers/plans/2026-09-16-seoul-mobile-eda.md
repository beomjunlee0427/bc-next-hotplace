# Seoul and Mobile Data Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build reproducible preprocessing and EDA outputs for Seoul commercial-district data and `mobile.xlsx`.

**Architecture:** Focused Python modules load and clean one source family at a time, write processed CSVs plus quality metadata, then a separate EDA module creates summaries and charts. Seoul and mobile data remain separate unless a validated key is discovered.

**Tech Stack:** Python 3.14, pandas, openpyxl, matplotlib, JSON/Markdown reports.

**Spec:** `docs/superpowers/specs/2026-09-16-seoul-mobile-eda-design.md`

## Global Constraints

- Read source data only from `data/raw/seoul` and `data/raw/mobile.xlsx`.
- Write cleaned data only under `data/processed` and reports under `reports`.
- Never silently drop rows; record dropped/changed counts in quality metadata.
- Run commands from the repository root.

### Task 1: Add analysis dependencies and source profiling

**Files:**
- Modify: `pyproject.toml`
- Create: `src/profile_data.py`

- [ ] Add `openpyxl` to runtime dependencies and keep matplotlib available for EDA.
- [ ] Implement a profiler that lists mobile sheets and source CSV row/column counts without loading whole files unnecessarily.
- [ ] Run the profiler and save its output to `reports/source_profile.json`.

### Task 2: Implement preprocessing

**Files:**
- Create: `src/preprocess.py`
- Create: `data/processed/seoul/.gitkeep`
- Create: `data/processed/mobile/.gitkeep`

- [ ] Implement `normalize_columns`, `read_seoul_csv`, `clean_seoul_frame`, `read_mobile_workbook`, and `clean_mobile_frame`.
- [ ] Convert numeric-looking fields with coercion metrics, preserve identifiers as strings, trim text, and retain missing values as missing.
- [ ] Write one processed CSV per source table and `data_quality.json` with source rows, output rows, missingness, duplicate rows, and conversion failures.
- [ ] Provide a CLI entry point: `python src/preprocess.py`.

### Task 3: Implement EDA outputs

**Files:**
- Create: `src/eda.py`
- Modify: `reports/.gitkeep` if present

- [ ] Read processed CSVs and create distribution/trend charts for row counts, missingness, yearly coverage, and top categories where columns are available.
- [ ] Write `reports/eda_report.md`, `reports/summary_statistics.csv`, and PNGs under `reports/figures`.
- [ ] Ensure the module can run after preprocessing with `python src/eda.py`.

### Task 4: Update the notebook and documentation

**Files:**
- Modify: `notebooks/preprocess.ipynb`
- Modify: `README.md`

- [ ] Add notebook cells that call the scripts and inspect the generated report.
- [ ] Document commands, output locations, and the fact that cross-source joins are not assumed.

### Task 5: Verify

- [ ] Run `python src/profile_data.py`.
- [ ] Run `python src/preprocess.py`.
- [ ] Run `python src/eda.py`.
- [ ] Confirm processed file counts, non-empty reports, and generated figures.
