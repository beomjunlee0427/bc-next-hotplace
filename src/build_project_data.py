"""Build inventory, Seoul panel, model features, and target data."""
from __future__ import annotations
import json, re
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; RAW=ROOT/"data/raw/seoul"; OUT=ROOT/"data/processed/project"; REPORTS=ROOT/"reports/project"

def encoding(path: Path)->str:
    for enc in ("utf-8-sig","cp949","euc-kr"):
        try: path.read_text(encoding=enc); return enc
        except UnicodeDecodeError: pass
    return "utf-8"

def source_type(path: Path)->str:
    name=path.name
    if "추정매출" in name: return "sales"
    if "점포-행정동" in name: return "stores_admin"
    if "점포-상권" in name: return "stores_trdar"
    if "길단위인구" in name: return "population"
    if "소비-행정동" in name: return "consumption"
    return "unknown"

def year_from_name(path: Path):
    m=re.search(r"(20\d{2})",path.name); return int(m.group(1)) if m else None

def read_source(path: Path, kind: str)->pd.DataFrame:
    keep = range(7) if kind == "sales" else range(12) if kind == "stores_admin" else range(4) if kind == "population" else range(14)
    frame=pd.read_csv(path,encoding=encoding(path),low_memory=False,usecols=list(keep)).drop_duplicates().copy()
    n=len(frame.columns)
    if kind in {"sales","stores_admin"}:
        frame.columns=["quarter_code","admin_code","admin_name","service_code","service_name"]+[f"value_{i:02d}" for i in range(n-5)]
    elif kind in {"population","consumption"}:
        frame.columns=["quarter_code","admin_code","admin_name"]+[f"value_{i:02d}" for i in range(n-3)]
    frame["quarter_code"]=frame["quarter_code"].astype("string").str.strip(); frame["quarter"]=pd.to_numeric(frame["quarter_code"],errors="coerce").astype("Int64"); frame["year"]=(frame["quarter"]//10).astype("Int64"); frame["q"]=(frame["quarter"]%10).astype("Int64")
    for col in frame.columns:
        if col.startswith("value_"): frame[col]=pd.to_numeric(frame[col].astype("string").str.replace(",","",regex=False),errors="coerce")
    frame["source_file"]=path.name; return frame

def inventory()->dict:
    files=[]
    for path in sorted(RAW.rglob("*.csv")):
        header=pd.read_csv(path,nrows=0,encoding=encoding(path)); files.append({"file":str(path.relative_to(ROOT)),"source_type":source_type(path),"year":year_from_name(path),"columns":len(header.columns),"column_names":[str(c) for c in header.columns]})
    mobile=ROOT/"data/raw/mobile.xlsx"
    if mobile.exists():
        for sheet in pd.ExcelFile(mobile).sheet_names: files.append({"file":str(mobile.relative_to(ROOT)),"source_type":"mobile_sheet","sheet":sheet})
    bc=ROOT/"data/raw/bc"
    if bc.exists():
        for path in sorted(bc.rglob("*")):
            if path.is_file(): files.append({"file":str(path.relative_to(ROOT)),"source_type":"bc_candidate","bytes":path.stat().st_size})
    result={"file_count":len(files),"files":files}; REPORTS.mkdir(parents=True,exist_ok=True); (REPORTS/"data_inventory.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8"); pd.DataFrame(files).to_csv(REPORTS/"data_inventory.csv",index=False,encoding="utf-8-sig"); return result

def build_panel()->tuple[pd.DataFrame,dict]:
    grouped={k:[] for k in ("sales","stores_admin","population","consumption")}
    for path in sorted(RAW.rglob("*.csv")):
        kind=source_type(path)
        if kind in grouped: grouped[kind].append(read_source(path,kind))
    sales=pd.concat(grouped["sales"],ignore_index=True).rename(columns={"value_00":"sales_amount","value_01":"sales_count"})
    stores=pd.concat(grouped["stores_admin"],ignore_index=True).rename(columns={"value_00":"store_count","value_03":"new_store_count","value_05":"closed_store_count"})
    population=pd.concat(grouped["population"],ignore_index=True).rename(columns={"value_00":"population_total"})
    consumption=pd.concat(grouped["consumption"],ignore_index=True).rename(columns={"value_00":"consumption_total"})
    keys=["quarter","admin_code"]; area=population.groupby(keys,as_index=False)[["population_total"]].sum(min_count=1); area=area.merge(consumption.groupby(keys,as_index=False)[["consumption_total"]].sum(min_count=1),on=keys,how="outer")
    panel=sales.merge(stores[["quarter","admin_code","service_code","store_count","new_store_count","closed_store_count"]],on=["quarter","admin_code","service_code"],how="left"); panel=panel.merge(area,on=keys,how="left")
    panel["quarter_start"]=pd.PeriodIndex(panel["year"].astype(str)+"Q"+panel["q"].astype(str),freq="Q").to_timestamp(); panel=panel.sort_values(["admin_code","service_code","quarter_start"]).reset_index(drop=True)
    grp=panel.groupby(["admin_code","service_code"]); panel["sales_growth_qoq"]=grp["sales_amount"].pct_change(); panel["sales_count_growth_qoq"]=grp["sales_count"].pct_change(); panel["store_growth_qoq"]=grp["store_count"].pct_change(); panel["sales_amount_lag1"]=grp["sales_amount"].shift(1); panel["sales_amount_lag2"]=grp["sales_amount"].shift(2); panel["sales_amount_roll4"]=grp["sales_amount"].transform(lambda x:x.shift(1).rolling(4,min_periods=2).mean()); panel["future_sales_amount"]=grp["sales_amount"].shift(-1); panel["future_sales_growth"]=panel["future_sales_amount"]/panel["sales_amount"]-1
    cutoff=pd.to_numeric(panel.groupby("quarter")["future_sales_growth"].transform(lambda x:x.quantile(.80)),errors="coerce"); growth=pd.to_numeric(panel["future_sales_growth"],errors="coerce"); panel["target_top20"]=(growth.notna() & cutoff.notna() & (growth >= cutoff)).astype("Int8"); panel=panel.drop(columns=["future_sales_amount"])
    quality={"rows":len(panel),"columns":len(panel.columns),"duplicate_rows":int(panel.duplicated(["quarter","admin_code","service_code"]).sum()),"missing_cells":int(panel.isna().sum().sum()),"quarters":sorted(panel["quarter"].dropna().unique().tolist())}; return panel,quality

def main()->None:
    OUT.mkdir(parents=True,exist_ok=True); REPORTS.mkdir(parents=True,exist_ok=True); inv=inventory(); panel,quality=build_panel(); panel.to_csv(OUT/"merged_panel_data.csv",index=False,encoding="utf-8-sig")
    features=panel.drop(columns=["future_sales_growth","target_top20","source_file"],errors="ignore"); targets=panel[["quarter","admin_code","service_code","future_sales_growth","target_top20"]]; features.to_csv(OUT/"feature_data.csv",index=False,encoding="utf-8-sig"); targets.to_csv(OUT/"target_data.csv",index=False,encoding="utf-8-sig")
    quality["inventory_file_count"]=inv["file_count"]; (REPORTS/"data_quality_report.json").write_text(json.dumps(quality,ensure_ascii=False,indent=2),encoding="utf-8"); pd.DataFrame([quality]).to_csv(REPORTS/"data_quality_report.csv",index=False,encoding="utf-8-sig"); print(json.dumps(quality,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
