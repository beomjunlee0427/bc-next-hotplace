"""Generate compact EDA tables, figures, and a Markdown report."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def main() -> None:
    report=ROOT/"reports"; figures=report/"figures"; figures.mkdir(parents=True,exist_ok=True); rows=[]; sections=[]
    for path in sorted((ROOT/"data/processed/seoul").glob("*.csv")):
        frame=pd.read_csv(path,encoding="utf-8-sig",nrows=200_000,low_memory=False); rows.append({"dataset":path.stem,"rows_sampled":len(frame),"columns":len(frame.columns),"missing_cells_sample":int(frame.isna().sum().sum())})
    for path in sorted((ROOT/"data/processed/mobile").glob("*.csv")):
        frame=pd.read_csv(path,encoding="utf-8-sig",low_memory=False); rows.append({"dataset":path.stem,"rows_sampled":len(frame),"columns":len(frame.columns),"missing_cells_sample":int(frame.isna().sum().sum())})
        if {"year","total"}.issubset(frame.columns):
            yearly=frame.groupby("year",dropna=True)["total"].sum(); sections.append(f"### {path.stem} 연도별 합계\n\n{yearly.to_string()}"); yearly.plot(kind="bar",title=f"{path.stem}: total by year"); plt.tight_layout(); plt.savefig(figures/f"{path.stem}_yearly_total.png",dpi=150); plt.close()
    summary=pd.DataFrame(rows); summary.to_csv(report/"summary_statistics.csv",index=False,encoding="utf-8-sig")
    (report/"eda_report.md").write_text("\n".join(["# Seoul and Mobile EDA","","## Dataset overview","",summary.to_csv(index=False),"","## Mobile trends","",*sections,"","Figures are in `reports/figures/`. Cross-source joins were not assumed." ]),encoding="utf-8"); print(summary.to_string(index=False))

if __name__ == "__main__": main()
