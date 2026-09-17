"""Train and evaluate leakage-safe next-hotplace classifiers."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/project"
OUT = ROOT / "reports/model"


def top_k_precision(y_true: pd.Series, scores: pd.Series, fraction: float = 0.20) -> float:
    n = max(1, int(len(scores) * fraction))
    chosen = scores.nlargest(n).index
    return float(y_true.loc[chosen].mean())


def evaluate(name: str, model: Pipeline, x: pd.DataFrame, y: pd.Series, meta: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    scores = pd.Series(model.predict_proba(x)[:, 1], index=x.index, name="predicted_probability")
    pred = (scores >= 0.5).astype("int8")
    metrics = {
        "model": name,
        "rows": int(len(y)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "pr_auc": float(average_precision_score(y, scores)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "top20_precision": top_k_precision(y, scores),
    }
    predictions = meta[["quarter", "admin_code", "admin_name", "service_code", "service_name"]].copy()
    predictions["actual_target_top20"] = y
    predictions["predicted_probability"] = scores
    predictions["predicted_target_top20"] = pred
    predictions["model"] = name
    return metrics, predictions


def main() -> None:
    frame = pd.read_csv(DATA / "merged_panel_data.csv", encoding="utf-8-sig", low_memory=False)
    frame = frame.replace([float("inf"), float("-inf")], pd.NA).dropna().copy()
    frame["period"] = frame["quarter"].astype(str)
    train = frame[frame["quarter"] <= 20242]
    validation = frame[frame["quarter"].between(20243, 20244)]
    test = frame[frame["quarter"] >= 20251]

    drop = {"future_sales_growth", "target_top20", "source_file", "quarter_start", "period", "admin_name", "service_name"}
    categorical = ["admin_code", "service_code"]
    numeric = [c for c in frame.columns if c not in drop and c not in categorical]
    preprocessor = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numeric", StandardScaler(), numeric),
    ])
    models = {
        "logistic_regression": LogisticRegression(max_iter=300, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=150, max_depth=14, min_samples_leaf=3, class_weight="balanced", n_jobs=-1, random_state=42),
    }
    metrics = []
    predictions = []
    for model_name, estimator in models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        pipe.fit(train.drop(columns=list(drop), errors="ignore"), train["target_top20"].astype("int8"))
        for split_name, split in (("validation", validation), ("test", test)):
            x = split.drop(columns=list(drop), errors="ignore")
            y = split["target_top20"].astype("int8")
            result, pred = evaluate(model_name, pipe, x, y, split)
            result["split"] = split_name
            metrics.append(result)
            pred["split"] = split_name
            predictions.append(pred)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(OUT / "model_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(predictions, ignore_index=True).sort_values("predicted_probability", ascending=False).to_csv(OUT / "predictions.csv", index=False, encoding="utf-8-sig")
    summary = {"train_rows": len(train), "validation_rows": len(validation), "test_rows": len(test), "features_numeric": numeric, "categorical_features": categorical, "metrics": metrics}
    (OUT / "model_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(metrics).to_string(index=False))


if __name__ == "__main__":
    main()
