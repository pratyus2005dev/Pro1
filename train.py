\
import argparse, json, os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from joblib import dump
import yaml

from .utils import safe_read_excel
from .features import build_pairwise_dataset

def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main(args):
    cfg = load_config(args.config)
    src_path = cfg["data"]["source_excel"]
    tgt_path = cfg["data"]["target_excel"]
    ground_truth = cfg["ground_truth"]
    random_state = cfg.get("random_state", 42)

    src_df = safe_read_excel(src_path)
    tgt_df = safe_read_excel(tgt_path)

    # Build dataset
    X, y, keys = build_pairwise_dataset(src_df, tgt_df, ground_truth)

    # Select numeric feature columns
    feature_cols = [c for c in X.columns if c not in ("src_col","tgt_col")]
    X_num = X[feature_cols].astype(float).fillna(0.0)

    X_train, X_test, y_train, y_test = train_test_split(X_num, y, test_size=0.3, random_state=random_state, stratify=y)

    # Candidate models
    pipelines = {
        "logreg": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
        "linsvm": Pipeline([("scaler", StandardScaler(with_mean=False)), ("clf", LinearSVC(class_weight="balanced"))]),
        "gboost": Pipeline([("clf", GradientBoostingClassifier())]),
        "rf": Pipeline([("clf", RandomForestClassifier(class_weight="balanced", n_estimators=400, random_state=random_state))]),
    }

    param_grid = {
        "logreg": {"clf__C":[0.1,1,10]},
        "linsvm": {"clf__C":[0.1,1,10]},
        "gboost": {"clf__n_estimators":[100,200], "clf__learning_rate":[0.05,0.1,0.2], "clf__max_depth":[2,3]},
        "rf": {"clf__max_depth":[None,5,10], "clf__min_samples_split":[2,5,10]},
    }

    best_name, best_model, best_f1 = None, None, -1
    results = {}

    for name, pipe in pipelines.items():
        grid = GridSearchCV(pipe, param_grid[name], scoring="f1", cv=3, n_jobs=-1, verbose=0)
        grid.fit(X_train, y_train)
        y_pred = grid.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        results[name] = {"best_params": grid.best_params_, "f1": float(f1)}
        if f1 > best_f1:
            best_f1 = f1
            best_model = grid.best_estimator_
            best_name = name

    # Fit best on full data
    best_model.fit(X_num, y)

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True, parents=True)
    dump(best_model, models_dir / "best_model.joblib")

    # Save feature column names
    (models_dir / "feature_cols.json").write_text(json.dumps(feature_cols, indent=2))

    # Save training report
    report = {
        "best_model": best_name,
        "cv_results": results,
        "n_pairs": int(len(y)),
        "class_balance": {"positives": int(int(y.sum())), "negatives": int(len(y) - int(y.sum()))},
    }
    Path("output").mkdir(exist_ok=True, parents=True)
    (Path("output") / "training_report.json").write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)
