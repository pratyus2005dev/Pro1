# Quick Usage Guide

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip3 install --break-system-packages -r requirements.txt
```

### 2. Run with Example Data
```bash
python3 schema_mapper.py \
    --source-ddl example_source.sql \
    --target-ddl example_target.sql \
    --reference-mappings example_reference_mappings.json \
    --output example_mappings.csv \
    --top-n 3
```

### 3. View Results
```bash
python3 demo.py
```

## 📁 Files Included

- **schema_mapper.py** - Main application
- **requirements.txt** - Python dependencies
- **example_source.sql** - Sample source database schema
- **example_target.sql** - Sample target database schema
- **example_reference_mappings.json** - Training data with known mappings
- **run_example.py** - Automated example runner
- **demo.py** - Results visualization script

## 🎯 What It Does

The tool analyzes database schemas and automatically suggests column mappings between source and target databases using:

1. **Fuzzy String Matching** - Multiple algorithms for name similarity
2. **Machine Learning** - Gradient boosting, Random Forest, and XGBoost models
3. **Semantic Analysis** - Data types, constraints, and naming patterns
4. **Confidence Scoring** - High/Medium/Low confidence levels

## 📊 Output Format

Results are saved as CSV with columns:
- source_table, source_column
- target_table, target_column  
- fuzzy_score, ml_score, combined_score
- mapping_type (1:1, 1:many, many:1, many:many)
- confidence (high, medium, low)
- Data type information and semantic similarity

## 🔧 Customization

### Use Your Own Data
1. Replace example DDL files with your schemas
2. Optionally provide reference mappings JSON for better accuracy
3. Adjust `--top-n` parameter for more/fewer suggestions per column

### Advanced Options
- Modify fuzzy matching algorithms in `FuzzyMatcher` class
- Add custom ML features in `MLModelTrainer._extract_features()`
- Adjust confidence thresholds in `_determine_confidence()`

## ✅ Expected Results

With the example data, you should see:
- ~156 mappings generated
- ~18% high confidence mappings
- ~80% medium confidence mappings
- Perfect matches like: `email_address` → `email_addr` (91% score)
- Semantic matches like: `is_discontinued` → `discontinued_flag`

## 🐛 Troubleshooting

**No tables parsed**: Check DDL syntax, ensure proper CREATE TABLE statements
**Low accuracy**: Provide reference mappings, ensure similar business domains
**Memory issues**: Reduce schema size or adjust `top_n` parameter

---

**Ready to use!** The tool provides automated suggestions - always review results before production use.