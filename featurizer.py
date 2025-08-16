from __future__ import annotations
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from .utils import tokens, jaccard, seq_ratio, coarse_type, string_value_jaccard, numeric_stat_similarity, date_range_overlap

def column_features(
    s_table: str, s_col: str, s_series: pd.Series, s_dtype: str,
    t_table: str, t_col: str, t_series: pd.Series, t_dtype: str
) -> Dict[str, float]:
    s_type = coarse_type(s_dtype)
    t_type = coarse_type(t_dtype)

    feats = {
        "name_seq_ratio": seq_ratio(s_col, t_col),
        "name_token_jaccard": jaccard(tokens(s_col), tokens(t_col)),
        "coarse_type_match": 1.0 if s_type == t_type else 0.0,
        "null_rate_diff": abs(s_series.isna().mean() - t_series.isna().mean()),
        "strlen_med_diff": 0.0,
        "string_val_jaccard": 0.0,
        "numeric_stat_sim": 0.0,
        "date_overlap": 0.0,
    }

    if s_type == "string" and t_type == "string":
        s_len = s_series.dropna().astype(str).str.len()
        t_len = t_series.dropna().astype(str).str.len()
        feats["strlen_med_diff"] = float(abs(s_len.median() - t_len.median()) / (max(s_len.median(), t_len.median(), 1)))
        feats["string_val_jaccard"] = string_value_jaccard(s_series, t_series)
    elif s_type == "numeric" and t_type == "numeric":
        feats["numeric_stat_sim"] = numeric_stat_similarity(s_series, t_series)
    elif s_type == "date" and t_type == "date":
        feats["date_overlap"] = date_range_overlap(s_series, t_series)

    return feats

def to_frame(feature_rows: List[Dict]) -> pd.DataFrame:
    return pd.DataFrame(feature_rows)
