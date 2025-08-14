
import argparse, json
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import load
from typing import Dict, List, Tuple

from .utils import safe_read_excel
from .features import build_pairwise_dataset

def load_feature_cols(path: str) -> List[str]:
    import json
    with open(path, "r") as f:
        return json.load(f)

def infer_mapping(model_path: str, source_excel: str, target_excel: str, threshold: float = 0.45, out_path: str = "output/inferred_mapping.csv"):
    model = load(model_path)
    src_df = safe_read_excel(source_excel)
    tgt_df = safe_read_excel(target_excel)

    # Dummy empty ground truth for feature building
    X, _, keys = build_pairwise_dataset(src_df, tgt_df, ground_truth={})
    feature_cols = [c for c in X.columns if c not in ("src_col","tgt_col")]
    X_num = X[feature_cols].astype(float).fillna(0.0)

    # Some models (LinearSVC) do not support predict_proba; fall back to decision_function
    has_proba = hasattr(model, "predict_proba")
    if has_proba:
        scores = model.predict_proba(X_num)[:,1]
    else:
        decision = model.decision_function(X_num)
        # scale to 0..1 via min-max for readability
        mn, mx = decision.min(), decision.max()
        scores = (decision - mn) / (mx - mn + 1e-9)

    pairs = pd.DataFrame({"src_col": X["src_col"], "tgt_col": X["tgt_col"], "score": scores})
    # For each source column, pick the target with max score
    mapping_rows = []
    for s, grp in pairs.groupby("src_col"):
        top = grp.sort_values("score", ascending=False).head(1).iloc[0]
        pred = top["tgt_col"] if top["score"] >= threshold else "NO_MATCH"
        mapping_rows.append({"source_column": s, "predicted_target": pred, "confidence": float(top["score"])})
    mapping = pd.DataFrame(mapping_rows).sort_values("source_column")
    Path(out_path).parent.mkdir(exist_ok=True, parents=True)
    mapping.to_csv(out_path, index=False)
    return mapping

def main(args):
    df = infer_mapping(args.model, args.source_excel, args.target_excel, args.threshold, args.out)
    print(df.to_string(index=False))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="models/best_model.joblib")
    p.add_argument("--source_excel", type=str, required=True)
    p.add_argument("--target_excel", type=str, required=True)
    p.add_argument("--threshold", type=float, default=0.45)
    p.add_argument("--out", type=str, default="output/inferred_mapping.csv")
    args = p.parse_args()
    main(args)
