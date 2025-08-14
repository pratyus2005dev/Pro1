\
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from .utils import normalize_name, tokenize, jaccard, seqmatch_ratio, jaccard_ngrams, detect_type_and_patterns

def column_profile(df: pd.DataFrame, col: str) -> Dict[str, float]:
    s = df[col]
    prof = detect_type_and_patterns(s)
    # add length stats
    lens = s.dropna().astype(str).apply(len)
    prof.update({
        "mean_len": lens.mean() if len(lens) else 0.0,
        "std_len": lens.std() if len(lens) else 0.0,
    })
    return prof

def pair_features(src_name: str, tgt_name: str, src_prof: Dict[str, float], tgt_prof: Dict[str, float]) -> Dict[str, float]:
    # name similarities
    a_norm, b_norm = normalize_name(src_name), normalize_name(tgt_name)
    tokens_a, tokens_b = tokenize(src_name), tokenize(tgt_name)
    feats = {
        "name_jaccard_tokens": jaccard(tokens_a, tokens_b),
        "name_seqmatch": seqmatch_ratio(a_norm, b_norm),
        "name_3gram_jaccard": jaccard_ngrams(a_norm, b_norm, n=3),
        "name_4gram_jaccard": jaccard_ngrams(a_norm, b_norm, n=4),
    }
    # profile diffs
    for k in ["is_date_like","is_numeric_like","has_email_like","has_phone_like","avg_len","pct_digits","pct_alpha","pct_special","nunique_ratio","mean_len","std_len"]:
        sa, sb = src_prof.get(k, 0.0), tgt_prof.get(k, 0.0)
        feats[f"absdiff_{k}"] = abs(sa - sb)
        feats[f"prod_{k}"] = sa * sb
    return feats

def build_pairwise_dataset(src_df: pd.DataFrame, tgt_df: pd.DataFrame, ground_truth: Dict[str, str]) -> Tuple[pd.DataFrame, pd.Series, List[Tuple[str, str]]]:
    # Precompute profiles
    src_cols = list(src_df.columns)
    tgt_cols = list(tgt_df.columns)
    src_profiles = {c: column_profile(src_df, c) for c in src_cols}
    tgt_profiles = {c: column_profile(tgt_df, c) for c in tgt_cols}

    rows = []
    y = []
    keys = []
    for s in src_cols:
        for t in tgt_cols:
            feats = pair_features(s, t, src_profiles[s], tgt_profiles[t])
            feat_row = {"src_col": s, "tgt_col": t}
            feat_row.update(feats)
            rows.append(feat_row)
            y.append(1 if ground_truth.get(s) == t else 0)
            keys.append((s,t))
    X = pd.DataFrame(rows)
    Y = pd.Series(y, name="label")
    return X, Y, keys
