import json
import os
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from .features import compute_pair_features, feature_names
from .types import DatasetProfiles, TableProfile, ColumnProfile


_DEF_TOPK = 3


def _load_artifacts(model_dir: str):
	with open(os.path.join(model_dir, "meta.json"), "r", encoding="utf-8") as f:
		meta = json.load(f)
	scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
	model = joblib.load(os.path.join(model_dir, "model.pkl"))
	return meta, scaler, model


def _iter_columns(profiles: Dict[str, TableProfile]):
	for tname, tprof in profiles.items():
		for cname, cprof in tprof.columns.items():
			yield (tname, cname, cprof)


def _score_all_pairs(dataset: DatasetProfiles, scaler, model) -> pd.DataFrame:
	rows: List[Dict[str, object]] = []
	feat_names = feature_names()
	for st, sc, sprof in _iter_columns(dataset.source):
		for tt, tc, tprof in _iter_columns(dataset.target):
			f = compute_pair_features(sprof, tprof)
			x = np.asarray([[float(f[n]) for n in feat_names]], dtype=float)
			x_scaled = scaler.transform(x)
			ml_score = float(model.predict_proba(x_scaled)[0, 1])
			rows.append({
				"source_table": st,
				"source_column": sc,
				"target_table": tt,
				"target_column": tc,
				"ml_score": ml_score,
				"fuzzy_score": float(f.get("fuzzy_name", 0.0)),
				"combined_score": float(0.5 * ml_score + 0.5 * float(f.get("fuzzy_name", 0.0))),
				"details": json.dumps(f),
			})
	return pd.DataFrame(rows)


def _best_by_group(df: pd.DataFrame, group_cols: List[str], score_col: str, topk: int = 1) -> pd.DataFrame:
	return (
		df.sort_values(score_col, ascending=False)
		.groupby(group_cols, as_index=False)
		.head(topk)
	)


def _aggregate_links(df: pd.DataFrame, key_cols: List[str], agg_cols: List[str], score_col: str) -> pd.DataFrame:
	# Aggregate rows by key_cols; collect other cols as lists with best scores
	agg_funcs = {c: list for c in agg_cols}
	agg_funcs[score_col] = max
	g = df.groupby(key_cols, as_index=False).agg(agg_funcs)
	return g


def generate_mappings_and_save(dataset: DatasetProfiles, model_dir: str, output_csv: str) -> None:
	meta, scaler, model = _load_artifacts(model_dir)
	df = _score_all_pairs(dataset, scaler, model)
	if df.shape[0] == 0:
		pd.DataFrame([], columns=[
			"mapping_type","source_tables","source_columns","target_tables","target_columns","fuzzy_score","ml_score","combined_score","details"
		]).to_csv(output_csv, index=False)
		return

	# One-to-one: best target for each source, and best source for each target, intersect
	best_t_for_s = _best_by_group(df, ["source_table", "source_column"], "combined_score", topk=1)
	best_s_for_t = _best_by_group(df, ["target_table", "target_column"], "combined_score", topk=1)
	one_to_one = pd.merge(
		best_t_for_s,
		best_s_for_t,
		left_on=["source_table", "source_column", "target_table", "target_column"],
		right_on=["source_table", "source_column", "target_table", "target_column"],
		suffixes=("_l", "_r"),
	)
	one_to_one = one_to_one[[
		"source_table","source_column","target_table","target_column","fuzzy_score","ml_score","combined_score","details"
	]]

	# One-to-many: take topK targets per source
	one_to_many = _best_by_group(df, ["source_table","source_column"], "combined_score", topk=_DEF_TOPK)
	one_to_many = one_to_many[[
		"source_table","source_column","target_table","target_column","fuzzy_score","ml_score","combined_score","details"
	]]

	# Many-to-one: take topK sources per target
	many_to_one = _best_by_group(df, ["target_table","target_column"], "combined_score", topk=_DEF_TOPK)
	many_to_one = many_to_one[[
		"source_table","source_column","target_table","target_column","fuzzy_score","ml_score","combined_score","details"
	]]

	# Many-to-many: aggregate topK both ways and combine
	agg_src = _aggregate_links(one_to_many, ["source_table","source_column"], ["target_table","target_column"], "combined_score")
	agg_tgt = _aggregate_links(many_to_one, ["target_table","target_column"], ["source_table","source_column"], "combined_score")

	records: List[Dict[str, object]] = []

	def _record(mt: str, st: List[str], sc: List[str], tt: List[str], tc: List[str], scores: List[float], fuzz: List[float], details: List[str]):
		records.append({
			"mapping_type": mt,
			"source_tables": ";".join(st),
			"source_columns": ";".join(sc),
			"target_tables": ";".join(tt),
			"target_columns": ";".join(tc),
			"fuzzy_score": float(np.mean(fuzz) if fuzz else 0.0),
			"ml_score": float(np.mean(scores) if scores else 0.0),
			"combined_score": float(np.mean(scores) if scores else 0.0),
			"details": json.dumps(details),
		})

	# Emit one-to-one
	for _, r in one_to_one.iterrows():
		_record("one_to_one", [r["source_table"]], [r["source_column"]], [r["target_table"]], [r["target_column"]], [r["ml_score"]], [r["fuzzy_score"]], [r["details"]])

	# Emit one-to-many per source
	for (st, sc), g in one_to_many.groupby(["source_table","source_column"]):
		_record("one_to_many", [st], [sc], list(g["target_table"].astype(str)), list(g["target_column"].astype(str)), list(g["ml_score"].astype(float)), list(g["fuzzy_score"].astype(float)), list(g["details"].astype(str)))

	# Emit many-to-one per target
	for (tt, tc), g in many_to_one.groupby(["target_table","target_column"]):
		_record("many_to_one", list(g["source_table"].astype(str)), list(g["source_column"].astype(str)), [tt], [tc], list(g["ml_score"].astype(float)), list(g["fuzzy_score"].astype(float)), list(g["details"].astype(str)))

	# Many-to-many: where both one-to-many and many-to-one include the pair
	# Simple heuristic: combine topK pairings where mutual presence occurs
	pairs = set(zip(one_to_many["source_table"], one_to_many["source_column"], one_to_many["target_table"], one_to_many["target_column"]))
	pairs &= set(zip(many_to_one["source_table"], many_to_one["source_column"], many_to_one["target_table"], many_to_one["target_column"]))
	for st, sc, tt, tc in pairs:
		g = df[(df.source_table==st)&(df.source_column==sc)&(df.target_table==tt)&(df.target_column==tc)]
		if g.shape[0] == 0:
			continue
		r = g.sort_values("combined_score", ascending=False).iloc[0]
		_record("many_to_many", [st], [sc], [tt], [tc], [float(r["ml_score"])], [float(r["fuzzy_score"])], [str(r["details"])])

	out_df = pd.DataFrame.from_records(records, columns=[
		"mapping_type","source_tables","source_columns","target_tables","target_columns","fuzzy_score","ml_score","combined_score","details"
	])
	out_df.to_csv(output_csv, index=False)