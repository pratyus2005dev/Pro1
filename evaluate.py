\
import argparse, json
import pandas as pd
from pathlib import Path
from joblib import load
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support

from .utils import safe_read_excel
from .features import build_pairwise_dataset

def main(args):
    # Load config
    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    src_path = cfg["data"]["source_excel"]
    tgt_path = cfg["data"]["target_excel"]
    ground_truth = cfg["ground_truth"]

    src_df = safe_read_excel(src_path)
    tgt_df = safe_read_excel(tgt_path)

    X, y, keys = build_pairwise_dataset(src_df, tgt_df, ground_truth)

    feature_cols = [c for c in X.columns if c not in ("src_col","tgt_col")]
    X_num = X[feature_cols].astype(float).fillna(0.0)

    model = load("models/best_model.joblib")

    # Predict labels from probabilities/decision
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_num)[:,1]
    else:
        decision = model.decision_function(X_num)
        mn, mx = decision.min(), decision.max()
        proba = (decision - mn) / (mx - mn + 1e-9)

    # Choose threshold from config
    thr = cfg.get("threshold_no_match", 0.45)
    y_pred = (proba >= thr).astype(int)

    report = classification_report(y, y_pred, output_dict=True)
    Path("output").mkdir(exist_ok=True, parents=True)
    (Path("output") / "eval_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    args = p.parse_args()
    main(args)
