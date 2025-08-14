\
import re
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

def normalize_name(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("_", " ").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def tokenize(s: str) -> List[str]:
    s = normalize_name(s)
    return [t for t in re.split(r"[^a-z0-9]+", s) if t]

def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0

def seqmatch_ratio(a: str, b: str) -> float:
    # difflib SequenceMatcher ratio
    import difflib
    return difflib.SequenceMatcher(None, a, b).ratio()

def char_ngrams(s: str, n: int = 3) -> set:
    s = normalize_name(s)
    if len(s) < n:
        return {s}
    return {s[i:i+n] for i in range(len(s) - n + 1)}

def jaccard_ngrams(a: str, b: str, n: int = 3) -> float:
    A = char_ngrams(a, n)
    B = char_ngrams(b, n)
    inter = len(A & B)
    union = len(A | B) or 1
    return inter / union

def detect_type_and_patterns(series: pd.Series, sample_size: int = 1000) -> Dict[str, float]:
    """Return simple profile stats useful for matching."""
    s = series.dropna().astype(str).head(sample_size)
    n = len(s)
    if n == 0:
        return {
            "is_date_like": 0.0, "is_numeric_like": 0.0, "has_email_like": 0.0, "has_phone_like": 0.0,
            "avg_len": 0.0, "pct_digits": 0.0, "pct_alpha": 0.0, "pct_special": 0.0, "nunique_ratio": 0.0
        }
    import datetime as _dt
    def is_date(x):
        try:
            pd.to_datetime(x, errors="raise")
            return True
        except Exception:
            return False
    def is_float(x):
        try:
            float(str(x).replace(",","").replace(" ",""))
            return True
        except Exception:
            return False
    email_re = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    phone_re = re.compile(r"^\+?[0-9\-\s\(\)]{6,}$")

    chars = s.apply(lambda x: list(str(x)))
    lens = s.apply(lambda x: len(str(x)))
    digits = s.apply(lambda x: sum(ch.isdigit() for ch in str(x)))
    alpha  = s.apply(lambda x: sum(ch.isalpha() for ch in str(x)))
    special= s.apply(lambda x: sum(not ch.isalnum() for ch in str(x)))

    stats = {
        "is_date_like": s.apply(is_date).mean(),
        "is_numeric_like": s.apply(is_float).mean(),
        "has_email_like": s.apply(lambda x: bool(email_re.match(str(x)))).mean(),
        "has_phone_like": s.apply(lambda x: bool(phone_re.match(str(x)))).mean(),
        "avg_len": lens.mean(),
        "pct_digits": (digits.sum() / max(1, lens.sum())),
        "pct_alpha":  (alpha.sum() / max(1, lens.sum())),
        "pct_special":(special.sum()/ max(1, lens.sum())),
        "nunique_ratio": s.nunique() / n
    }
    return stats

def safe_read_excel(path: str) -> pd.DataFrame:
    # Supports CSV or Excel by extension
    path = str(path)
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    else:
        return pd.read_excel(path)
