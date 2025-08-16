import re
import difflib
from typing import Iterable, Set, Tuple
import numpy as np
import pandas as pd

NAME_SPLIT = re.compile(r"[_\W]+")

def coarse_type(dtype_str: str) -> str:
    ds = dtype_str.lower()
    if any(k in ds for k in ["date", "datetime", "time"]):
        return "date"
    if any(k in ds for k in ["int", "float", "decimal", "double", "numeric"]):
        return "numeric"
    return "string"

def clean_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()

def tokens(name: str) -> Set[str]:
    return {t for t in NAME_SPLIT.split(clean_name(name)) if t}

def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb: return 1.0
    if not sa or not sb: return 0.0
    return len(sa & sb) / len(sa | sb)

def seq_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, clean_name(a), clean_name(b)).ratio()

def string_value_jaccard(a: pd.Series, b: pd.Series, sample_size: int = 500) -> float:
    sa = set(a.dropna().astype(str).head(sample_size).str.lower().unique())
    sb = set(b.dropna().astype(str).head(sample_size).str.lower().unique())
    if not sa and not sb: return 1.0
    if not sa or not sb: return 0.0
    return len(sa & sb) / len(sa | sb)

def numeric_stat_similarity(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce"); b = pd.to_numeric(b, errors="coerce")
    a, b = a.dropna(), b.dropna()
    if len(a) < 5 or len(b) < 5:
        return 0.5
    am, asd = a.mean(), a.std()
    bm, bsd = b.mean(), b.std()
    # normalize difference into [0,1]
    denom = (abs(am) + abs(bm) + 1e-6)
    mean_sim = 1.0 - min(abs(am - bm) / denom, 1.0)
    sd_denom = (abs(asd) + abs(bsd) + 1e-6)
    std_sim = 1.0 - min(abs(asd - bsd) / sd_denom, 1.0)
    return float((mean_sim + std_sim) / 2)

def date_range_overlap(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_datetime(a, errors="coerce"); b = pd.to_datetime(b, errors="coerce")
    if a.dropna().empty or b.dropna().empty:
        return 0.5
    a0, a1 = a.min(), a.max()
    b0, b1 = b.min(), b.max()
    left = max(a0, b0); right = min(a1, b1)
    if pd.isna(left) or pd.isna(right) or right < left:
        return 0.0
    total = (max(a1, b1) - min(a0, b0)).days + 1
    inter = (right - left).days + 1
    return float(max(0.0, min(1.0, inter / max(total, 1))))
