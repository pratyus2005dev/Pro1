from __future__ import annotations
import os
import pandas as pd
from typing import Dict

def load_tables(root: str, file_map: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    out = {}
    for table, fname in file_map.items():
        path = os.path.join(root, fname)
        df = pd.read_csv(path, low_memory=False)
        out[table] = df
    return out

def profile_column(series: pd.Series) -> dict:
    s = series.dropna()
    info = {
        "n": int(series.size),
        "null_ratio": float(series.isna().mean()),
        "dtype": str(series.dtype),
    }
    if s.empty:
        return {**info, "unique": 0}
    info["unique"] = int(s.nunique(dropna=True))
    if pd.api.types.is_string_dtype(series):
        lengths = s.astype(str).str.len()
        info["strlen_med"] = float(lengths.median())
        info["sample_values"] = [str(v)[:64] for v in s.sample(min(100, len(s)), random_state=0).unique()[:50]]
    elif pd.api.types.is_numeric_dtype(series):
        info["mean"] = float(s.mean())
        info["std"] = float(s.std() or 0.0)
    elif pd.api.types.is_datetime64_any_dtype(series):
        info["min"] = str(pd.to_datetime(s, errors="coerce").min())
        info["max"] = str(pd.to_datetime(s, errors="coerce").max())
    return info
