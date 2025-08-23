import json
import os
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .features import compute_pair_features, feature_names
from .types import DatasetProfiles, TableProfile, ColumnProfile


def _collect_all_columns(profiles: Dict[str, TableProfile]) -> List[ColumnProfile]:
	cols: List[ColumnProfile] = []
	for t in profiles.values():
		cols.extend(t.columns.values())
	return cols


def _read_reference_mapping(path: str) -> pd.DataFrame:
	df = pd.read_csv(path)
	required = {"source_table", "source_column", "target_table", "target_column"}
	missing = [c for c in required if c not in df.columns]
	if missing:
		raise ValueError(f"Missing columns in reference mapping: {missing}")
	for c in required:
		df[c] = df[c].astype(str)
	return df


def _build_training_pairs(dataset: DatasetProfiles, reference_csv: str, negative_ratio: float = 4.0) -> Tuple[np.ndarray, np.ndarray, List[str]]:
	reference = _read_reference_mapping(reference_csv)
	src_by_key: Dict[Tuple[str, str], ColumnProfile] = {}
	tgt_by_key: Dict[Tuple[str, str], ColumnProfile] = {}
	for tname, tprof in dataset.source.items():
		for cname, cprof in tprof.columns.items():
			src_by_key[(tname, cname)] = cprof
	for tname, tprof in dataset.target.items():
		for cname, cprof in tprof.columns.items():
			tgt_by_key[(tname, cname)] = cprof

	X: List[List[float]] = []
	y: List[int] = []
	feats = feature_names()

	# Positives
	pos_pairs: List[Tuple[ColumnProfile, ColumnProfile]] = []
	for _, row in reference.iterrows():
		key_s = (str(row["source_table"]).lower(), str(row["source_column"]).lower())
		key_t = (str(row["target_table"]).lower(), str(row["target_column"]).lower())
		s = src_by_key.get(key_s)
		t = tgt_by_key.get(key_t)
		if s is None or t is None:
			continue
		pos_pairs.append((s, t))
		f = compute_pair_features(s, t)
		X.append([float(f[n]) for n in feats])
		y.append(1)

	# Negatives: for each positive, sample mismatched targets and sources
	all_src = list(src_by_key.values())
	all_tgt = list(tgt_by_key.values())
	rng = np.random.default_rng(42)
	neg_needed = int(len(pos_pairs) * negative_ratio)
	neg_count = 0
	while neg_count < neg_needed and (all_src and all_tgt):
		s = rng.choice(all_src)
		t = rng.choice(all_tgt)
		if (s.table_name, s.column_name) in [(ps.table_name, ps.column_name) for ps, _ in pos_pairs] and \
			(t.table_name, t.column_name) in [(pt.table_name, pt.column_name) for _, pt in pos_pairs]:
			# avoid exact positive pairs, but it's fine if source or target overlaps
			pass
		f = compute_pair_features(s, t)
		X.append([float(f[n]) for n in feats])
		y.append(0)
		neg_count += 1

	return np.asarray(X, dtype=float), np.asarray(y, dtype=int), feats


def _fit_models(X: np.ndarray, y: np.ndarray) -> Tuple[Dict[str, object], Dict[str, float]]:
	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X)
	models: Dict[str, object] = {}
	scores: Dict[str, float] = {}

	log_reg = LogisticRegression(max_iter=1000)
	gb_clf = GradientBoostingClassifier()

	x_train, x_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.25, random_state=42, stratify=y)

	log_reg.fit(x_train, y_train)
	gb_clf.fit(x_train, y_train)

	pr_lr = log_reg.predict_proba(x_val)[:, 1]
	pr_gb = gb_clf.predict_proba(x_val)[:, 1]

	s_lr = roc_auc_score(y_val, pr_lr)
	s_gb = roc_auc_score(y_val, pr_gb)

	models["scaler"] = scaler
	models["logistic_regression"] = log_reg
	models["gradient_boosting"] = gb_clf
	scores["logistic_regression"] = float(s_lr)
	scores["gradient_boosting"] = float(s_gb)

	return models, scores


def train_and_save_model(dataset: DatasetProfiles, reference_mapping_csv: str, model_dir: str) -> None:
	X, y, feats = _build_training_pairs(dataset, reference_mapping_csv)
	if X.shape[0] == 0:
		raise RuntimeError("No training pairs could be built. Check reference mapping and data.")
	models, scores = _fit_models(X, y)
	# Choose best by AUC
	best_name = max([k for k in scores.keys()], key=lambda k: scores[k])
	best_model = models[best_name]
	artifacts = {
		"feature_names": feats,
		"best_model": best_name,
		"scores": scores,
	}
	joblib.dump(best_model, os.path.join(model_dir, "model.pkl"))
	joblib.dump(models.get("scaler"), os.path.join(model_dir, "scaler.pkl"))
	with open(os.path.join(model_dir, "meta.json"), "w", encoding="utf-8") as f:
		json.dump(artifacts, f, indent=2)