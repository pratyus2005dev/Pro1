from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
from rapidfuzz import fuzz as rapidfuzz
import re
from .utils import (
    tokens, jaccard, seq_ratio, coarse_type, string_value_jaccard, 
    numeric_stat_similarity, date_range_overlap, extract_patterns,
    calculate_cardinality, calculate_uniqueness, calculate_distribution_similarity
)

def fuzzy_name_scores(s_col: str, t_col: str) -> Dict[str, float]:
    """Calculate multiple fuzzy matching scores for column names."""
    s_clean = re.sub(r'[^a-zA-Z0-9]', '', s_col.lower())
    t_clean = re.sub(r'[^a-zA-Z0-9]', '', t_col.lower())
    
    return {
        "fuzzy_ratio": fuzz.ratio(s_col, t_col) / 100.0,
        "fuzzy_partial_ratio": fuzz.partial_ratio(s_col, t_col) / 100.0,
        "fuzzy_token_sort_ratio": fuzz.token_sort_ratio(s_col, t_col) / 100.0,
        "fuzzy_token_set_ratio": fuzz.token_set_ratio(s_col, t_col) / 100.0,
        "rapidfuzz_ratio": rapidfuzz.ratio(s_col, t_col) / 100.0,
        "rapidfuzz_partial_ratio": rapidfuzz.partial_ratio(s_col, t_col) / 100.0,
        "name_seq_ratio": seq_ratio(s_col, t_col),
        "name_token_jaccard": jaccard(tokens(s_col), tokens(t_col)),
        "clean_name_ratio": fuzz.ratio(s_clean, t_clean) / 100.0 if s_clean and t_clean else 0.0,
    }

def metadata_features(
    s_table: str, s_col: str, s_series: pd.Series, s_dtype: str,
    t_table: str, t_col: str, t_series: pd.Series, t_dtype: str
) -> Dict[str, float]:
    """Extract metadata-based features for column mapping."""
    s_type = coarse_type(s_dtype)
    t_type = coarse_type(t_dtype)
    
    # Basic metadata
    s_null_rate = s_series.isna().mean()
    t_null_rate = t_series.isna().mean()
    s_cardinality = calculate_cardinality(s_series)
    t_cardinality = calculate_cardinality(t_series)
    s_uniqueness = calculate_uniqueness(s_series)
    t_uniqueness = calculate_uniqueness(t_series)
    
    return {
        "coarse_type_match": 1.0 if s_type == t_type else 0.0,
        "null_rate_diff": abs(s_null_rate - t_null_rate),
        "null_rate_similarity": 1.0 - abs(s_null_rate - t_null_rate),
        "cardinality_ratio": min(s_cardinality, t_cardinality) / max(s_cardinality, t_cardinality) if max(s_cardinality, t_cardinality) > 0 else 0.0,
        "uniqueness_diff": abs(s_uniqueness - t_uniqueness),
        "uniqueness_similarity": 1.0 - abs(s_uniqueness - t_uniqueness),
        "length_diff": abs(len(s_series) - len(t_series)) / max(len(s_series), len(t_series)) if max(len(s_series), len(t_series)) > 0 else 0.0,
        "table_name_similarity": seq_ratio(s_table, t_table),
    }

def data_similarity_features(
    s_series: pd.Series, t_series: pd.Series, s_type: str, t_type: str
) -> Dict[str, float]:
    """Extract data-based similarity features."""
    features = {
        "strlen_med_diff": 0.0,
        "string_val_jaccard": 0.0,
        "numeric_stat_sim": 0.0,
        "date_overlap": 0.0,
        "distribution_similarity": 0.0,
        "pattern_similarity": 0.0,
    }
    
    if s_type == "string" and t_type == "string":
        s_len = s_series.dropna().astype(str).str.len()
        t_len = t_series.dropna().astype(str).str.len()
        if not s_len.empty and not t_len.empty:
            features["strlen_med_diff"] = float(abs(s_len.median() - t_len.median()) / (max(s_len.median(), t_len.median(), 1)))
        features["string_val_jaccard"] = string_value_jaccard(s_series, t_series)
        features["pattern_similarity"] = extract_patterns(s_series, t_series)
        
    elif s_type == "numeric" and t_type == "numeric":
        features["numeric_stat_sim"] = numeric_stat_similarity(s_series, t_series)
        features["distribution_similarity"] = calculate_distribution_similarity(s_series, t_series)
        
    elif s_type == "date" and t_type == "date":
        features["date_overlap"] = date_range_overlap(s_series, t_series)
    
    return features

def column_features(
    s_table: str, s_col: str, s_series: pd.Series, s_dtype: str,
    t_table: str, t_col: str, t_series: pd.Series, t_dtype: str
) -> Dict[str, float]:
    """Enhanced column features combining fuzzy, metadata, and data similarity."""
    s_type = coarse_type(s_dtype)
    t_type = coarse_type(t_dtype)
    
    # Fuzzy name matching features
    fuzzy_scores = fuzzy_name_scores(s_col, t_col)
    
    # Metadata features
    metadata_feats = metadata_features(s_table, s_col, s_series, s_dtype,
                                     t_table, t_col, t_series, t_dtype)
    
    # Data similarity features
    data_feats = data_similarity_features(s_series, t_series, s_type, t_type)
    
    # Combine all features
    all_features = {**fuzzy_scores, **metadata_feats, **data_feats}
    
    # Add composite scores
    all_features["fuzzy_avg_score"] = np.mean([
        fuzzy_scores["fuzzy_ratio"],
        fuzzy_scores["fuzzy_token_sort_ratio"],
        fuzzy_scores["rapidfuzz_ratio"]
    ])
    
    all_features["overall_similarity"] = np.mean([
        all_features["fuzzy_avg_score"],
        all_features["coarse_type_match"],
        all_features["null_rate_similarity"],
        all_features["uniqueness_similarity"]
    ])
    
    return all_features

def to_frame(feature_rows: List[Dict]) -> pd.DataFrame:
    return pd.DataFrame(feature_rows)
