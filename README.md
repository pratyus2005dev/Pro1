### Schema Mapper: Data- and Metadata-driven Column Mapping

This tool automates source-to-target schema mapping for data migrations. It:
- Parses DDL to extract table/column metadata
- Profiles CSV data for statistical features
- Engineers fuzzy name and data similarity features
- Trains ML models (Logistic Regression and Gradient Boosting) and selects the best
- Scores column pairs and proposes one-to-one, one-to-many, many-to-one, and many-to-many mappings
- Outputs a CSV with fuzzy name score, ML score, and combined score

#### Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

#### Inputs
- Source CSVs: one or more files (headers required)
- Target CSVs: one or more files (headers required)
- DDL SQL: one or more files containing CREATE TABLE definitions
- Reference mapping CSV for training: columns `source_table,source_column,target_table,target_column`

#### CLI
```bash
python -m schema_mapper.cli --help | cat
```

Train and map in one step:
```bash
python -m schema_mapper.cli auto \
  --source-files /abs/path/src_*.csv \
  --target-files /abs/path/tgt_*.csv \
  --ddl /abs/path/schema.ddl.sql \
  --reference-mapping /abs/path/reference_mapping.csv \
  --model-dir /abs/path/model_dir \
  --output /abs/path/mappings.csv
```

Train only:
```bash
python -m schema_mapper.cli train \
  --source-files /abs/path/src_*.csv \
  --target-files /abs/path/tgt_*.csv \
  --ddl /abs/path/schema.ddl.sql \
  --reference-mapping /abs/path/reference_mapping.csv \
  --model-dir /abs/path/model_dir
```

Map with an existing model:
```bash
python -m schema_mapper.cli map \
  --source-files /abs/path/src_*.csv \
  --target-files /abs/path/tgt_*.csv \
  --ddl /abs/path/schema.ddl.sql \
  --model-dir /abs/path/model_dir \
  --output /abs/path/mappings.csv
```

Output CSV columns:
- `mapping_type` in {one_to_one, one_to_many, many_to_one, many_to_many}
- `source_tables`, `source_columns` (semicolon-separated for multiple)
- `target_tables`, `target_columns`
- `fuzzy_score`, `ml_score`, `combined_score`
- `details` (JSON with feature diagnostics)

