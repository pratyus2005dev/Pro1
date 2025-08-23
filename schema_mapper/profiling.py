from typing import Dict, List

import numpy as np
import pandas as pd

from .types import DatasetProfiles, TableProfile, ColumnProfile, DDLSchema
from .utils import (
	infer_table_name_from_path,
	read_csv_sample,
	is_series_numeric,
	is_series_string,
	try_to_numeric,
	tokenize_identifier,
)


_DEF_SAMPLE_ROWS = 5000


def _profile_table(df: pd.DataFrame, table_name: str) -> TableProfile:
	table_profile = TableProfile(table_name=table_name)
	for col_name in df.columns:
		series = df[col_name]
		inferred_dtype = str(series.dtype)
		is_num = is_series_numeric(series)
		is_str = is_series_string(series)
		sample_count = int(series.shape[0])
		num_null = int(series.isna().sum())
		null_fraction = float(num_null / sample_count) if sample_count > 0 else 0.0
		unique_count = int(series.nunique(dropna=True))
		unique_fraction = float(unique_count / sample_count) if sample_count > 0 else 0.0

		length_mean = None
		length_std = None
		numeric_mean = None
		numeric_std = None
		numeric_min = None
		numeric_max = None

		if is_str:
			lengths = series.dropna().astype(str).str.len()
			if lengths.shape[0] > 0:
				length_mean = float(lengths.mean())
				length_std = float(lengths.std(ddof=0) if lengths.shape[0] > 1 else 0.0)

		if is_num:
			numeric = try_to_numeric(series)
			if numeric.dropna().shape[0] > 0:
				numeric_mean = float(numeric.mean())
				numeric_std = float(numeric.std(ddof=0) if numeric.shape[0] > 1 else 0.0)
				numeric_min = float(numeric.min())
				numeric_max = float(numeric.max())

		top_values = []
		try:
			vc = series.dropna().astype(str).value_counts().head(20)
			top_values = [(str(k), int(v)) for k, v in vc.items()]
		except Exception:
			pass

		column_profile = ColumnProfile(
			table_name=table_name,
			column_name=str(col_name),
			inferred_dtype=inferred_dtype,
			is_numeric=is_num,
			is_string=is_str,
			sample_count=sample_count,
			null_fraction=null_fraction,
			num_unique=unique_count,
			unique_fraction=unique_fraction,
			length_mean=length_mean,
			length_std=length_std,
			numeric_mean=numeric_mean,
			numeric_std=numeric_std,
			numeric_min=numeric_min,
			numeric_max=numeric_max,
			top_values=top_values,
			name_tokens=tokenize_identifier(str(col_name)),
		)
		table_profile.columns[str(col_name)] = column_profile
	return table_profile


def _load_and_profile(paths: List[str]) -> Dict[str, TableProfile]:
	profiles: Dict[str, TableProfile] = {}
	for path in paths:
		df = read_csv_sample(path, sample_rows=_DEF_SAMPLE_ROWS)
		table = infer_table_name_from_path(path)
		profiles[table] = _profile_table(df, table)
	return profiles


def profile_sources_and_targets(source_paths: List[str], target_paths: List[str], ddl_schema: DDLSchema) -> DatasetProfiles:
	source_profiles = _load_and_profile(source_paths)
	target_profiles = _load_and_profile(target_paths)
	return DatasetProfiles(source=source_profiles, target=target_profiles, ddl=ddl_schema)