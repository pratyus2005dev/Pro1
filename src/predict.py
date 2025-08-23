from __future__ import annotations
import json
from typing import Dict, List
import pandas as pd
from joblib import load
from .ddl_parser import load_ddl
from .data_loader import load_tables
from .featurizer import column_features
from .utils import coarse_type

def predict_mappings(
    ddl_path: str,
    source_root: str, source_files: Dict[str, str],
    target_root: str, target_files: Dict[str, str],
    table_pairs: List[List[str]],
    model_in: str,
    top_k: int,
    threshold: float,
    out_csv: str,
    out_json: str
):
    mdl = load(model_in)
    model = mdl["model"]; feat_cols = mdl["features"]

    ddl = load_ddl(ddl_path)
    src = load_tables(source_root, source_files)
    tgt = load_tables(target_root, target_files)

    rows = []
    for s_table, t_table in table_pairs:
        for s_col in src[s_table].columns:
            cand = []
            for t_col in tgt[t_table].columns:
                feats = column_features(
                    s_table, s_col, src[s_table][s_col], ddl[s_table].get(s_col, "string"),
                    t_table, t_col, tgt[t_table][t_col], ddl[t_table].get(t_col, "string")
                )
                X = [[feats[c] for c in feat_cols]]
                score = float(model.predict_proba(X)[0,1]) if hasattr(model, "predict_proba") else float(model.decision_function(X)[0])
                cand.append((t_col, score, feats))
            cand.sort(key=lambda x: x[1], reverse=True)
            top = cand[:top_k]
            best = top[0] if top else None
            rows.append({
                "source_table": s_table,
                "source_column": s_col,
                "target_table": t_table,
                "best_target_column": best[0] if best and best[1] >= threshold else None,
                "best_score": best[1] if best else None,
                "alternates": [{ "t_col": t, "score": s } for t,s,_ in top],
            })

    df = pd.DataFrame([{
        "source_table": r["source_table"],
        "source_column": r["source_column"],
        "target_table": r["target_table"],
        "predicted_target_column": r["best_target_column"],
        "score": r["best_score"]
    } for r in rows])
    df.to_csv(out_csv, index=False)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    return {"rows": len(rows), "csv": out_csv, "json": out_json}
