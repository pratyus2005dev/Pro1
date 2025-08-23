import argparse
import os
from pathlib import Path
from typing import List, Tuple, Dict
import json

import pandas as pd
from rapidfuzz import fuzz
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import joblib

# ------------------------------
# Utility helpers
# ------------------------------

def parse_ddl(ddl_path: Path) -> Dict[str, Dict[str, str]]:
    """Very naive DDL parser that extracts table & column data types.

    Currently supports PostgreSQL / MySQL style CREATE TABLE statements.
    """
    import sqlparse
    with open(ddl_path, "r", encoding="utf-8") as fh:
        ddl = fh.read()

    parsed = sqlparse.parse(ddl)
    metadata: Dict[str, Dict[str, str]] = {}
    for stmt in parsed:
        if stmt.get_type() != "CREATE":
            continue
        tokens = [t for t in stmt.tokens if not t.is_whitespace]
        if not tokens:
            continue
        # Table name usually follows 'TABLE'
        table_name = None
        for i, tok in enumerate(tokens):
            if tok.ttype is None and tok.value.upper() == "TABLE":
                # Next token is table name
                table_name = tokens[i + 1].value.strip('"`')
                break
        if not table_name:
            continue
        col_defs = {}
        # columns between first parenthesis pair
        parenthesis = None
        for tok in tokens:
            if tok.ttype is None and tok.value.startswith("("):
                parenthesis = tok
                break
        if parenthesis:
            # naive split by comma at top-level
            cols_raw = [c.strip() for c in parenthesis.value.strip("() ").split(",")]
            for col_raw in cols_raw:
                parts = col_raw.split()
                if len(parts) >= 2:
                    col_name = parts[0].strip('"`')
                    data_type = parts[1]
                    col_defs[col_name] = data_type
        metadata[table_name] = col_defs
    return metadata


def fuzzy_name_score(a: str, b: str) -> float:
    """Token set ratio scaled to [0,1]."""
    return fuzz.token_set_ratio(a, b) / 100.0


def data_type_compatibility(dt1: str, dt2: str) -> int:
    """Simple compatibility - 1 if same family else 0."""
    numeric = {"int", "integer", "bigint", "smallint", "float", "double", "decimal"}
    text = {"varchar", "char", "text", "string"}
    date = {"date", "timestamp", "datetime"}

    def family(dt: str) -> str:
        dt = dt.lower()
        for fam, group in {"numeric": numeric, "text": text, "date": date}.items():
            if any(dt.startswith(g) for g in group):
                return fam
        return "other"

    return 1 if family(dt1) == family(dt2) else 0


def sample_value_similarity(s1: pd.Series, s2: pd.Series) -> float:
    """Jaccard similarity of unique sample values (stringified)."""
    set1 = set(s1.dropna().astype(str).unique()[:100])
    set2 = set(s2.dropna().astype(str).unique()[:100])
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union else 0.0

# ------------------------------
# Feature Engineering
# ------------------------------

def build_feature_matrix(
    src_df: pd.DataFrame,
    tgt_df: pd.DataFrame,
    src_meta: Dict[str, str],
    tgt_meta: Dict[str, str],
) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    """Return feature matrix X and list of (src_col, tgt_col) pairs."""
    rows = []
    pairs = []
    for s_col, s_type in src_meta.items():
        for t_col, t_type in tgt_meta.items():
            features = {
                "fuzzy_name": fuzzy_name_score(s_col, t_col),
                "type_compat": data_type_compatibility(s_type, t_type),
                "sample_sim": sample_value_similarity(src_df[s_col] if s_col in src_df else pd.Series(dtype=str),
                                                      tgt_df[t_col] if t_col in tgt_df else pd.Series(dtype=str)),
            }
            rows.append(features)
            pairs.append((s_col, t_col))
    X = pd.DataFrame(rows)
    return X, pairs

# ------------------------------
# Model handling
# ------------------------------

def train_model(train_csv: Path, output_model: Path):
    """Train gradient boosting classifier from reference mapping csv.

    train_csv should have columns: source_column, target_column, match (1/0), plus optional cols.
    """
    df = pd.read_csv(train_csv)

    X = df[[c for c in df.columns if c not in {"match"}]]
    y = df["match"]

    # Identify categorical vs numeric
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ])

    clf = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc"
    )

    model = Pipeline([
        ("pre", preprocessor),
        ("clf", clf)
    ])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    val_pred = model.predict_proba(X_val)[:, 1]
    print(f"Validation AUC: {roc_auc_score(y_val, val_pred):.4f}")

    joblib.dump(model, output_model)
    print(f"Model saved to {output_model}")

# ------------------------------
# Mapping prediction logic
# ------------------------------

def generate_mapping(
    src_file: Path,
    tgt_file: Path,
    ddl_path: Path,
    model_path: Path,
    output_csv: Path,
):
    src_df = pd.read_csv(src_file)
    tgt_df = pd.read_csv(tgt_file)
    metadata = parse_ddl(ddl_path)

    # Choose first table for simplicity
    if not metadata:
        raise ValueError("No table definitions found in DDL")
    table_name = list(metadata.keys())[0]
    src_meta = metadata[table_name]
    tgt_meta = metadata[table_name]

    X, pairs = build_feature_matrix(src_df, tgt_df, src_meta, tgt_meta)

    model = joblib.load(model_path)
    proba = model.predict_proba(X)[:, 1]

    results = pd.DataFrame(pairs, columns=["source_column", "target_column"])
    results["fuzzy_score"] = X["fuzzy_name"]
    results["ml_score"] = proba
    results["final_score"] = 0.5 * results["fuzzy_score"] + 0.5 * results["ml_score"]

    # Determine best mapping per source column (one-to-one for now)
    idx = results.groupby("source_column")["final_score"].idxmax()
    best = results.loc[idx].sort_values("final_score", ascending=False)
    best.to_csv(output_csv, index=False)
    print(f"Mapping results saved to {output_csv}")

# ------------------------------
# CLI
# ------------------------------

def main():
    parser = argparse.ArgumentParser(description="Automated source→target column mapping tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_p = subparsers.add_parser("train", help="Train a new model from reference mappings")
    train_p.add_argument("train_csv", type=Path, help="CSV with labeled mapping data")
    train_p.add_argument("model_out", type=Path, help="Path to save trained model.joblib")

    map_p = subparsers.add_parser("map", help="Generate mapping for new source/target files")
    map_p.add_argument("source_file", type=Path)
    map_p.add_argument("target_file", type=Path)
    map_p.add_argument("ddl", type=Path)
    map_p.add_argument("model", type=Path)
    map_p.add_argument("output_csv", type=Path)

    args = parser.parse_args()

    if args.command == "train":
        train_model(args.train_csv, args.model_out)
    elif args.command == "map":
        generate_mapping(args.source_file, args.target_file, args.ddl, args.model, args.output_csv)


if __name__ == "__main__":
    main()