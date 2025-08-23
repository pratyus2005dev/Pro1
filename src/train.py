from __future__ import annotations
import json, random
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from joblib import dump
from .ddl_parser import load_ddl
from .data_loader import load_tables
from .featurizer import column_features, to_frame
from .model import build_model
from .utils import coarse_type

def load_synonyms(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_training_pairs(
    table_pairs: List[List[str]],
    src_tables: Dict[str, pd.DataFrame],
    tgt_tables: Dict[str, pd.DataFrame],
    ddl: Dict[str, Dict[str, str]],
    synonyms: Dict[str, List[str]],
    negative_ratio: int,
) -> pd.DataFrame:
    rows = []
    positives = set()
    # positives
    for s_full, targets in synonyms.items():
        s_table, s_col = s_full.split("::")
        for t_full in targets:
            t_table, t_col = t_full.split("::")
            # build if within declared table_pairs
            if [s_table, t_table] not in table_pairs:
                continue
            s_series = src_tables[s_table][s_col] if s_col in src_tables[s_table].columns else pd.Series(dtype="object")
            t_series = tgt_tables[t_table][t_col] if t_col in tgt_tables[t_table].columns else pd.Series(dtype="object")
            feats = column_features(s_table, s_col, s_series, ddl[s_table].get(s_col, "string"),
                                    t_table, t_col, t_series, ddl[t_table].get(t_col, "string"))
            rows.append({**feats,
                         "s_table": s_table, "s_col": s_col,
                         "t_table": t_table, "t_col": t_col,
                         "label": 1})
            positives.add((s_table, s_col, t_table, t_col))

    # negatives: mismatches within allowed table pairs, type-aware
    rng = random.Random(0)
    for s_table, t_table in table_pairs:
        s_cols = list(src_tables[s_table].columns)
        t_cols = list(tgt_tables[t_table].columns)
        for s_col in s_cols:
            # pick negatives for this source column
            candidates = []
            for t_col in t_cols:
                if (s_table, s_col, t_table, t_col) in positives:
                    continue
                # prefer type-incompatible & name-dissimilar pairs as negatives
                candidates.append(t_col)
            rng.shuffle(candidates)
            for t_col in candidates[:negative_ratio]:
                s_series = src_tables[s_table][s_col]
                t_series = tgt_tables[t_table][t_col]
                feats = column_features(s_table, s_col, s_series, ddl[s_table].get(s_col, "string"),
                                        t_table, t_col, t_series, ddl[t_table].get(t_col, "string"))
                rows.append({**feats,
                             "s_table": s_table, "s_col": s_col,
                             "t_table": t_table, "t_col": t_col,
                             "label": 0})
    return to_frame(rows)

def train(
    ddl_path: str,
    source_root: str, source_files: Dict[str, str],
    target_root: str, target_files: Dict[str, str],
    table_pairs: List[List[str]],
    synonyms_path: str,
    negative_ratio: int,
    test_size: float,
    random_state: int,
    model_out: str
):
    ddl = load_ddl(ddl_path)
    src = load_tables(source_root, source_files)
    tgt = load_tables(target_root, target_files)
    syn = load_synonyms(synonyms_path)

    df = build_training_pairs(table_pairs, src, tgt, ddl, syn, negative_ratio)

    feature_cols = [c for c in df.columns if c not in ("label","s_table","s_col","t_table","t_col")]
    X = df[feature_cols].values
    y = df["label"].values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    clf = build_model(random_state=random_state)
    clf.fit(X_tr, y_tr)

    proba = clf.predict_proba(X_te)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_te)
    auc = roc_auc_score(y_te, proba)
    ap = average_precision_score(y_te, proba)

    dump({"model": clf, "features": feature_cols}, model_out)
    return {"train_rows": int(len(df)), "features": feature_cols, "auc": float(auc), "ap": float(ap)}
