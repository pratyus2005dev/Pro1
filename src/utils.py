import re
import difflib
from typing import Iterable, Set, Tuple, List
import numpy as np
import pandas as pd
from collections import Counter

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

def calculate_cardinality(series: pd.Series) -> float:
    """Calculate the cardinality (number of unique values) of a series."""
    return float(series.nunique())

def calculate_uniqueness(series: pd.Series) -> float:
    """Calculate the uniqueness ratio (unique values / total values) of a series."""
    if len(series) == 0:
        return 0.0
    return float(series.nunique() / len(series))

def extract_patterns(s_series: pd.Series, t_series: pd.Series, sample_size: int = 1000) -> float:
    """Extract and compare patterns in string data."""
    def get_patterns(series: pd.Series) -> Set[str]:
        patterns = set()
        sample = series.dropna().astype(str).head(sample_size)
        
        for val in sample:
            # Extract common patterns
            if re.match(r'^\d+$', val):
                patterns.add('numeric_only')
            elif re.match(r'^[A-Za-z]+$', val):
                patterns.add('alpha_only')
            elif re.match(r'^[A-Za-z0-9]+$', val):
                patterns.add('alphanumeric')
            elif re.match(r'^[A-Za-z0-9\s]+$', val):
                patterns.add('alphanumeric_with_spaces')
            elif re.match(r'^[A-Za-z0-9\-_]+$', val):
                patterns.add('alphanumeric_with_special')
            elif '@' in val and '.' in val:
                patterns.add('email_like')
            elif re.match(r'^\d{4}-\d{2}-\d{2}$', val):
                patterns.add('date_format')
            elif re.match(r'^\d{2}/\d{2}/\d{4}$', val):
                patterns.add('date_format_alt')
            elif len(val) <= 3:
                patterns.add('short_string')
            elif len(val) > 50:
                patterns.add('long_string')
            else:
                patterns.add('mixed_pattern')
        
        return patterns
    
    s_patterns = get_patterns(s_series)
    t_patterns = get_patterns(t_series)
    
    if not s_patterns and not t_patterns:
        return 1.0
    if not s_patterns or not t_patterns:
        return 0.0
    
    return len(s_patterns & t_patterns) / len(s_patterns | t_patterns)

def calculate_distribution_similarity(s_series: pd.Series, t_series: pd.Series, bins: int = 10) -> float:
    """Calculate distribution similarity between two numeric series using histogram comparison."""
    try:
        s_numeric = pd.to_numeric(s_series, errors="coerce").dropna()
        t_numeric = pd.to_numeric(t_series, errors="coerce").dropna()
        
        if len(s_numeric) < 5 or len(t_numeric) < 5:
            return 0.5
        
        # Create histograms
        s_hist, _ = np.histogram(s_numeric, bins=bins, density=True)
        t_hist, _ = np.histogram(t_numeric, bins=bins, density=True)
        
        # Calculate cosine similarity
        dot_product = np.dot(s_hist, t_hist)
        norm_s = np.linalg.norm(s_hist)
        norm_t = np.linalg.norm(t_hist)
        
        if norm_s == 0 or norm_t == 0:
            return 0.0
        
        return float(dot_product / (norm_s * norm_t))
    except:
        return 0.5

def calculate_entropy(series: pd.Series) -> float:
    """Calculate the entropy of a series."""
    value_counts = series.value_counts()
    total = len(series)
    if total == 0:
        return 0.0
    
    entropy = 0.0
    for count in value_counts:
        p = count / total
        if p > 0:
            entropy -= p * np.log2(p)
    
    return float(entropy)

def calculate_mutual_information(s_series: pd.Series, t_series: pd.Series) -> float:
    """Calculate mutual information between two series."""
    try:
        # Create contingency table
        contingency = pd.crosstab(s_series, t_series)
        
        # Calculate mutual information
        total = contingency.sum().sum()
        if total == 0:
            return 0.0
        
        mi = 0.0
        for i in range(len(contingency)):
            for j in range(len(contingency.columns)):
                p_ij = contingency.iloc[i, j] / total
                p_i = contingency.iloc[i, :].sum() / total
                p_j = contingency.iloc[:, j].sum() / total
                
                if p_ij > 0 and p_i > 0 and p_j > 0:
                    mi += p_ij * np.log2(p_ij / (p_i * p_j))
        
        return float(mi)
    except:
        return 0.0
