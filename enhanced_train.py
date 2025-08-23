from __future__ import annotations
import json, random
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
import joblib
from .ddl_parser import load_ddl
from .data_loader import load_tables
from .enhanced_featurizer import EnhancedFeaturizer, to_frame
from .enhanced_model import build_model

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
    featurizer: EnhancedFeaturizer
) -> pd.DataFrame:
    rows = []
    positives = set()
    
    # Generate positive examples from synonyms
    for s_full, targets in synonyms.items():
        s_table, s_col = s_full.split("::")
        for t_full in targets:
            t_table, t_col = t_full.split("::")
            
            # Check if within declared table_pairs
            if [s_table, t_table] not in table_pairs:
                continue
                
            # Get data series
            s_series = src_tables[s_table][s_col] if s_col in src_tables[s_table].columns else pd.Series(dtype="object")
            t_series = tgt_tables[t_table][t_col] if t_col in tgt_tables[t_table].columns else pd.Series(dtype="object")
            
            # Calculate enhanced features
            feats = featurizer.column_features(
                s_table, s_col, s_series, ddl[s_table].get(s_col, "string"),
                t_table, t_col, t_series, ddl[t_table].get(t_col, "string"),
                ddl
            )
            
            rows.append({
                **feats,
                "s_table": s_table, "s_col": s_col,
                "t_table": t_table, "t_col": t_col,
                "label": 1
            })
            positives.add((s_table, s_col, t_table, t_col))

    # Generate negative examples
    rng = random.Random(0)
    for s_table, t_table in table_pairs:
        s_cols = list(src_tables[s_table].columns)
        t_cols = list(tgt_tables[t_table].columns)
        
        for s_col in s_cols:
            # Pick negative targets for this source column
            candidates = []
            for t_col in t_cols:
                if (s_table, s_col, t_table, t_col) in positives:
                    continue
                candidates.append(t_col)
            
            rng.shuffle(candidates)
            for t_col in candidates[:negative_ratio]:
                s_series = src_tables[s_table][s_col]
                t_series = tgt_tables[t_table][t_col]
                
                # Calculate enhanced features
                feats = featurizer.column_features(
                    s_table, s_col, s_series, ddl[s_table].get(s_col, "string"),
                    t_table, t_col, t_series, ddl[t_table].get(t_col, "string"),
                    ddl
                )
                
                rows.append({
                    **feats,
                    "s_table": s_table, "s_col": s_col,
                    "t_table": t_table, "t_col": t_col,
                    "label": 0
                })
    
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
    model_out: str,
    use_ensemble: bool = True,
    tune_hyperparameters: bool = False
):
    """Enhanced training function with improved features and model selection"""
    
    print("Loading data...")
    ddl = load_ddl(ddl_path)
    src = load_tables(source_root, source_files)
    tgt = load_tables(target_root, target_files)
    syn = load_synonyms(synonyms_path)
    
    print("Initializing featurizer...")
    featurizer = EnhancedFeaturizer()
    
    print("Building training pairs...")
    df = build_training_pairs(table_pairs, src, tgt, ddl, syn, negative_ratio, featurizer)
    
    print(f"Training dataset size: {len(df)} rows")
    print(f"Positive examples: {df['label'].sum()}")
    print(f"Negative examples: {len(df) - df['label'].sum()}")
    
    # Extract features
    feature_cols = [c for c in df.columns if c not in ("label", "s_table", "s_col", "t_table", "t_col")]
    X = df[feature_cols].values
    y = df["label"].values
    
    print(f"Feature count: {len(feature_cols)}")
    print(f"Features: {feature_cols}")
    
    # Split data
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set size: {len(X_tr)}")
    print(f"Test set size: {len(X_te)}")
    
    # Build and train model
    print("Building model...")
    model = build_model(random_state=random_state, use_ensemble=use_ensemble)
    
    print("Training model...")
    model_info = model.train(X_tr, y_tr, tune_hyperparameters=tune_hyperparameters)
    
    # Evaluate on test set
    print("Evaluating model...")
    y_pred_proba = model.predict_proba(X_te)
    test_auc = roc_auc_score(y_te, y_pred_proba)
    test_ap = average_precision_score(y_te, y_pred_proba)
    
    # Generate classification report
    y_pred = model.predict(X_te, threshold=0.5)
    class_report = classification_report(y_te, y_pred, output_dict=True)
    
    # Save model and metadata
    print(f"Saving model to {model_out}...")
    model.save(model_out)
    
    # Save featurizer separately
    featurizer_path = model_out.replace('.pkl', '_featurizer.pkl')
    joblib.dump(featurizer, featurizer_path)
    
    # Compile results
    results = {
        "train_rows": int(len(df)),
        "test_rows": int(len(X_te)),
        "features": feature_cols,
        "feature_count": len(feature_cols),
        "positive_examples": int(df['label'].sum()),
        "negative_examples": int(len(df) - df['label'].sum()),
        "model_info": model_info,
        "test_metrics": {
            "auc": float(test_auc),
            "ap": float(test_ap),
            "classification_report": class_report
        },
        "model_path": model_out,
        "featurizer_path": featurizer_path
    }
    
    print("Training completed successfully!")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test AP: {test_ap:.4f}")
    print(f"Best model: {model_info['model_type']}")
    
    return results

def evaluate_model_performance(
    model_path: str,
    featurizer_path: str,
    ddl_path: str,
    source_root: str, source_files: Dict[str, str],
    target_root: str, target_files: Dict[str, str],
    table_pairs: List[List[str]],
    synonyms_path: str
):
    """Evaluate model performance on validation data"""
    
    print("Loading model and featurizer...")
    model = build_model().load(model_path)
    featurizer = joblib.load(featurizer_path)
    
    print("Loading validation data...")
    ddl = load_ddl(ddl_path)
    src = load_tables(source_root, source_files)
    tgt = load_tables(target_root, target_files)
    syn = load_synonyms(synonyms_path)
    
    # Generate validation pairs
    validation_pairs = build_training_pairs(
        table_pairs, src, tgt, ddl, syn, negative_ratio=2, featurizer=featurizer
    )
    
    # Extract features
    feature_cols = [c for c in validation_pairs.columns if c not in ("label", "s_table", "s_col", "t_table", "t_col")]
    X_val = validation_pairs[feature_cols].values
    y_val = validation_pairs["label"].values
    
    # Predict
    y_pred_proba = model.predict_proba(X_val)
    y_pred = model.predict(X_val, threshold=0.5)
    
    # Calculate metrics
    val_auc = roc_auc_score(y_val, y_pred_proba)
    val_ap = average_precision_score(y_val, y_pred_proba)
    val_report = classification_report(y_val, y_pred, output_dict=True)
    
    results = {
        "validation_metrics": {
            "auc": float(val_auc),
            "ap": float(val_ap),
            "classification_report": val_report
        },
        "validation_samples": len(X_val),
        "positive_samples": int(y_val.sum()),
        "negative_samples": int(len(y_val) - y_val.sum())
    }
    
    print("Validation Results:")
    print(f"AUC: {val_auc:.4f}")
    print(f"AP: {val_ap:.4f}")
    
    return results