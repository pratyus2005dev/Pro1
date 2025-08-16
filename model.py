from __future__ import annotations
from typing import Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

def build_model(random_state: int = 42):
    if _HAS_XGB:
        # solid, regularized baseline
        return XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=1.5,
            reg_alpha=0.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state
        )
    # fallback
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=4,
        class_weight="balanced",
        random_state=random_state
    )
