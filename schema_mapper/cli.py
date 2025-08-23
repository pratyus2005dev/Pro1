import argparse
import json
import os
from typing import List

from .ddl_parser import parse_ddl_files
from .profiling import profile_sources_and_targets
from .training import train_and_save_model
from .mapping import generate_mappings_and_save


def _parse_multi_path_arg(values: List[str]) -> List[str]:
	paths: List[str] = []
	for v in values:
		# Support comma-separated lists in addition to repeated flags
		parts = [p.strip() for p in v.split(",") if p.strip()]
		paths.extend(parts)
	return [os.path.abspath(p) for p in paths]



def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Schema Mapper: ML-driven source-to-target column mapping using data and metadata",
	)
	sub = parser.add_subparsers(dest="command", required=True)

	common = argparse.ArgumentParser(add_help=False)
	common.add_argument("--source-files", nargs="+", required=True, help="Absolute paths to source CSV files (headers required). Comma-separated or repeat flag.")
	common.add_argument("--target-files", nargs="+", required=True, help="Absolute paths to target CSV files (headers required). Comma-separated or repeat flag.")
	common.add_argument("--ddl", nargs="+", required=True, help="Absolute paths to one or more DDL SQL files with CREATE TABLE statements.")
	common.add_argument("--model-dir", required=True, help="Directory to save or load model artifacts.")

	p_train = sub.add_parser("train", parents=[common], help="Train the model from reference mapping and save.")
	p_train.add_argument("--reference-mapping", required=True, help="CSV with columns source_table,source_column,target_table,target_column")

	p_map = sub.add_parser("map", parents=[common], help="Score and generate mapping suggestions using an existing model.")
	p_map.add_argument("--output", required=True, help="Output CSV for mapping suggestions.")

	p_auto = sub.add_parser("auto", parents=[common], help="Train then map in one step.")
	p_auto.add_argument("--reference-mapping", required=True, help="CSV with columns source_table,source_column,target_table,target_column")
	p_auto.add_argument("--output", required=True, help="Output CSV for mapping suggestions.")

	return parser


def main() -> None:
	parser = build_parser()
	args = parser.parse_args()

	source_files = _parse_multi_path_arg(args.source_files)
	target_files = _parse_multi_path_arg(args.target_files)
	ddl_files = _parse_multi_path_arg(args.ddl)
	model_dir = os.path.abspath(args.model_dir)
	os.makedirs(model_dir, exist_ok=True)

	# Parse DDL and profile CSVs (shared across commands)
	ddl_schema = parse_ddl_files(ddl_files)
	profiles = profile_sources_and_targets(source_files, target_files, ddl_schema)

	if args.command == "train":
		reference_mapping = os.path.abspath(args.reference_mapping)
		train_and_save_model(profiles, reference_mapping, model_dir)
		print(json.dumps({"status": "ok", "model_dir": model_dir}))
	elif args.command == "map":
		output_csv = os.path.abspath(args.output)
		generate_mappings_and_save(profiles, model_dir, output_csv)
		print(json.dumps({"status": "ok", "output": output_csv}))
	elif args.command == "auto":
		reference_mapping = os.path.abspath(args.reference_mapping)
		output_csv = os.path.abspath(args.output)
		train_and_save_model(profiles, reference_mapping, model_dir)
		generate_mappings_and_save(profiles, model_dir, output_csv)
		print(json.dumps({"status": "ok", "model_dir": model_dir, "output": output_csv}))
	else:
		parser.error("Unknown command")


if __name__ == "__main__":
	main()