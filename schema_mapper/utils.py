import os
import re
from typing import Set, List, Tuple, Optional

import pandas as pd
from text_unidecode import unidecode


_identifier_splitter = re.compile(r"[^A-Za-z0-9]+")


def normalize_identifier(name: str) -> str:
	if name is None:
		return ""
	name_ascii = unidecode(name)
	name_ascii = name_ascii.strip().strip('"').strip("`").strip("'")
	# Drop schema prefixes like schema.table
	if "." in name_ascii:
		name_ascii = name_ascii.split(".")[-1]
	return name_ascii.lower()


def tokenize_identifier(name: str) -> Set[str]:
	norm = normalize_identifier(name)
	tokens = [t for t in _identifier_splitter.split(norm) if t]
	return set(tokens)


def infer_table_name_from_path(path: str) -> str:
	base = os.path.basename(path)
	stem = os.path.splitext(base)[0]
	return normalize_identifier(stem)


def read_csv_sample(path: str, sample_rows: Optional[int] = 5000) -> pd.DataFrame:
	try:
		if sample_rows is None:
			return pd.read_csv(path)
		# For large files, let pandas read in chunks then sample
		df = pd.read_csv(path, nrows=sample_rows)
		return df
	except Exception as exc:
		raise RuntimeError(f"Failed to read CSV at {path}: {exc}")


def try_to_numeric(series: pd.Series) -> pd.Series:
	return pd.to_numeric(series, errors="coerce")


def is_series_numeric(series: pd.Series) -> bool:
	return pd.api.types.is_numeric_dtype(series)


def is_series_string(series: pd.Series) -> bool:
	return pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)


def top_value_counts(series: pd.Series, top_k: int = 20) -> List[Tuple[str, int]]:
	vc = series.dropna().astype(str).value_counts().head(top_k)
	return [(str(idx), int(cnt)) for idx, cnt in vc.items()]