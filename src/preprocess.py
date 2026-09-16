"""Preprocess Seoul CSVs and the mobile Excel workbook."""
from __future__ import annotations
import json
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def csv_encoding(path: Path) -> str:
    for encoding in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            path.read_text(encoding=encoding, errors="strict")
            return encoding
        except UnicodeDecodeError:
            continue
    return "utf-8"

def normalize_columns(columns) -> list[str]:
    result = []
    for i, col in enumerate(columns):
        value = re.sub(r"\s+", "_", str(col).strip()).lower()
        value = re.sub(r"[^0-9a-zA-Z가-힣_]+", "_", value).strip("_") or f"column_{i+1:02d}"
        if value in result: value = f"{value}_{i+1:02d}"
        result.append(value)
    return result

def clean_seoul_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    frame.columns = normalize_columns(frame.columns)
    before = len(frame); frame = frame.drop_duplicates().copy(); failures = {}
    for col in frame.columns:
        if frame[col].dtype == "object":
            frame[col] = frame[col].astype("string").str.strip()
            numeric = pd.to_numeric(frame[col].str.replace(",", "", regex=False), errors="coerce")
            nonempty = frame[col].notna() & frame[col].ne("")
            if nonempty.any() and numeric[nonempty].notna().mean() >= 0.95:
                failures[col] = int(numeric[nonempty].isna().sum()); frame[col] = numeric
    return frame, {"source_rows": before, "output_rows": len(frame), "duplicate_rows_removed": before-len(frame), "numeric_conversion_failures": failures, "missing_cells": int(frame.isna().sum().sum())}

def clean_mobile_frame(frame: pd.DataFrame, sheet_index: int) -> tuple[pd.DataFrame, dict]:
    names = ["geo_1", "geo_2", "period_raw", "male", "female", "total"] if len(frame.columns)==6 else ["geo", "period_raw", "male", "female", "total"] if len(frame.columns)==5 else normalize_columns(frame.columns)
    frame = frame.copy(); frame.columns = names; before = len(frame)
    frame = frame.dropna(how="all").drop_duplicates().copy(); frame["period_raw"] = frame["period_raw"].astype("string").str.strip()
    extracted = frame["period_raw"].str.extract(r"(?P<year>\d{4})[.\-/](?P<month>\d{1,2})[.\-/](?P<quarter>\d)")
    for col in ["year", "month", "quarter", "male", "female", "total"]:
        if col in frame: frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = pd.concat([frame, extracted.astype("Int64")], axis=1); frame["source_sheet_index"] = sheet_index
    return frame, {"source_rows": before, "output_rows": len(frame), "duplicate_rows_removed": before-len(frame), "missing_cells": int(frame.isna().sum().sum())}

def main() -> None:
    out_root = ROOT / "data/processed"; seoul_out, mobile_out = out_root/"seoul", out_root/"mobile"
    seoul_out.mkdir(parents=True, exist_ok=True); mobile_out.mkdir(parents=True, exist_ok=True); quality={"seoul":[],"mobile":[]}
    for source in sorted((ROOT/"data/raw/seoul").rglob("*.csv")):
        frame = pd.read_csv(source, encoding=csv_encoding(source), low_memory=False); clean, metrics = clean_seoul_frame(frame); target=seoul_out/f"{source.stem}.csv"; clean.to_csv(target,index=False,encoding="utf-8-sig")
        quality["seoul"].append({"source":str(source.relative_to(ROOT)),"output":str(target.relative_to(ROOT)),**metrics})
    workbook=ROOT/"data/raw/mobile.xlsx"
    for i, sheet in enumerate(pd.ExcelFile(workbook).sheet_names, 1):
        frame=pd.read_excel(workbook,sheet_name=sheet); clean,metrics=clean_mobile_frame(frame,i); target=mobile_out/f"mobile_sheet_{i:02d}.csv"; clean.to_csv(target,index=False,encoding="utf-8-sig"); quality["mobile"].append({"source":str(workbook.relative_to(ROOT)),"sheet":sheet,"output":str(target.relative_to(ROOT)),**metrics})
    (out_root/"data_quality.json").write_text(json.dumps(quality,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(quality,ensure_ascii=False,indent=2))

if __name__ == "__main__": main()
