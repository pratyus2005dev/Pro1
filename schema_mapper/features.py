from typing import Dict, Tuple, List

import numpy as np
from rapidfuzz import fuzz

from .types import ColumnProfile, PairFeatures


_NUMERIC_TYPES = {"int", "integer", "bigint", "smallint", "tinyint", "decimal", "numeric", "float", "double", "real"}
_STRING_TYPES = {"varchar", "char", "nvarchar", "text", "string"}
_DATE_TYPES = {"date", "datetime", "timestamp"}


def _jaccard(a: set, b: set) -> float:
	if not a and not b:
		return 1.0
	if not a or not b:
		return 0.0
	inter = len(a & b)
	union = len(a | b)
	return float(inter / union) if union > 0 else 0.0


def _type_compatibility(src: ColumnProfile, tgt: ColumnProfile) -> float:
	# Simple heuristic using inferred dtypes
	if src.is_numeric and tgt.is_numeric:
		return 1.0
	if src.is_string and tgt.is_string:
		return 0.8
	return 0.2


def _length_similarity(src: ColumnProfile, tgt: ColumnProfile) -> float:
	if src.length_mean is None or tgt.length_mean is None:
		return 0.5
	m1, m2 = float(src.length_mean), float(tgt.length_mean)
	if m1 == 0 and m2 == 0:
		return 1.0
	return float(1.0 / (1.0 + abs(m1 - m2) / (1.0 + max(m1, m2))))


def _numeric_moment_similarity(src: ColumnProfile, tgt: ColumnProfile) -> float:
	if src.numeric_mean is None or tgt.numeric_mean is None:
		return 0.5
	m1, m2 = float(src.numeric_mean), float(tgt.numeric_mean)
	s1 = float(src.numeric_std or 0.0)
	s2 = float(tgt.numeric_std or 0.0)
	mean_sim = 1.0 / (1.0 + abs(m1 - m2) / (1.0 + max(abs(m1), abs(m2), 1.0)))
	std_sim = 1.0 / (1.0 + abs(s1 - s2) / (1.0 + max(s1, s2, 1.0)))
	return float(0.7 * mean_sim + 0.3 * std_sim)


def _unique_fraction_similarity(src: ColumnProfile, tgt: ColumnProfile) -> float:
	return float(1.0 - abs(src.unique_fraction - tgt.unique_fraction))


def _null_fraction_similarity(src: ColumnProfile, tgt: ColumnProfile) -> float:
	return float(1.0 - abs(src.null_fraction - tgt.null_fraction))


def compute_pair_features(src: ColumnProfile, tgt: ColumnProfile) -> PairFeatures:
	name_fuzz = fuzz.token_set_ratio(src.column_name, tgt.column_name) / 100.0
	token_jaccard = _jaccard(src.name_tokens, tgt.name_tokens)
	type_compat = _type_compatibility(src, tgt)
	length_sim = _length_similarity(src, tgt)
	numeric_sim = _numeric_moment_similarity(src, tgt)
	unique_sim = _unique_fraction_similarity(src, tgt)
	null_sim = _null_fraction_similarity(src, tgt)

	return {
		"fuzzy_name": float(name_fuzz),
		"token_jaccard": float(token_jaccard),
		"type_compat": float(type_compat),
		"length_sim": float(length_sim),
		"numeric_sim": float(numeric_sim),
		"unique_sim": float(unique_sim),
		"null_sim": float(null_sim),
	}


def feature_names() -> List[str]:
	return [
		"fuzzy_name",
		"token_jaccard",
		"type_compat",
		"length_sim",
		"numeric_sim",
		"unique_sim",
		"null_sim",
	]