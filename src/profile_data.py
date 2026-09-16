"""Lightweight source inventory for the Seoul and mobile inputs."""
from __future__ import annotations
import json
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

def main() -> None:
    seoul = []
    for path in sorted((ROOT / "data/raw/seoul").rglob("*.csv")):
        frame = pd.read_csv(path, nrows=0, encoding=csv_encoding(path))
        seoul.append({"file": str(path.relative_to(ROOT)), "columns": len(frame.columns), "column_names": frame.columns.tolist()})
    mobile = []
    workbook = ROOT / "data/raw/mobile.xlsx"
    for sheet in pd.ExcelFile(workbook).sheet_names:
        frame = pd.read_excel(workbook, sheet_name=sheet, nrows=0)
        mobile.append({"sheet": sheet, "columns": len(frame.columns), "column_names": [str(c) for c in frame.columns]})
    out = {"seoul_csvs": seoul, "mobile_sheets": mobile}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/source_profile.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
