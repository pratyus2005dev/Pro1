# Advanced Column Mapping ML System

This project provides a **comprehensive machine learning-based schema mapping tool** that automatically aligns columns between source and target databases using advanced ML models, fuzzy matching, and multiple mapping types (one-one, one-many, many-one, many-many).

## 🚀 Key Features

### **Advanced ML Models**
- **XGBoost**: High-performance gradient boosting
- **LightGBM**: Fast gradient boosting framework
- **CatBoost**: Advanced gradient boosting with categorical features
- **Random Forest**: Ensemble of decision trees
- **Ensemble Methods**: Voting classifiers combining multiple models
- **Hyperparameter Tuning**: Automatic optimization of model parameters

### **Comprehensive Fuzzy Matching**
- **Multiple Algorithms**: FuzzyWuzzy, RapidFuzz, Sequence Matcher
- **Token-based Matching**: Jaccard similarity, token sorting
- **Pattern Recognition**: Data pattern extraction and comparison
- **Name Normalization**: Advanced text preprocessing

### **Multiple Mapping Types**
- **One-to-One**: Traditional column mappings
- **One-to-Many**: Single source column maps to multiple target columns
- **Many-to-One**: Multiple source columns map to single target column
- **Many-to-Many**: Complex multi-column relationships

### **Enhanced Feature Engineering**
- **Metadata Features**: Cardinality, uniqueness, null rates, data types
- **Data Similarity**: Distribution analysis, pattern matching, value overlap
- **Statistical Features**: Mean, variance, entropy, mutual information
- **Composite Scores**: Weighted combinations of multiple similarity measures

### **Comprehensive Reporting**
- **CSV Output**: Detailed mapping results with scores
- **Excel Reports**: Multi-sheet reports with statistics
- **JSON Output**: Structured data for API integration
- **Visualizations**: Score distributions, correlation matrices, performance charts

## 📂 Project Structure

```
schema-mapper-ml/
├─ README.md
├─ requirements.txt
├─ example_usage.py
├─ main.py
├─ ddl/
│ └─ DDL.sql                    # Schema definitions
├─ configs/
│ ├─ config.yaml               # Configuration settings
│ └─ synonyms.json             # Known column mappings
├─ data/
│ ├─ guidewire/                # Source CSV files
│ └─ insurenow/                # Target CSV files
├─ models/
│ └─ matcher.pkl               # Trained models
├─ outputs/
│ ├─ comprehensive_mappings.csv
│ ├─ mapping_summary.csv
│ └─ visualizations/
└─ src/
├─ config.py                   # Configuration loader
├─ ddl_parser.py               # DDL parsing utilities
├─ data_loader.py              # Data loading functions
├─ utils.py                    # Utility functions
├─ featurizer.py               # Feature extraction
├─ model.py                    # ML model definitions
├─ mapping_engine.py           # Mapping engine with multiple types
├─ report_generator.py         # Report generation
├─ train.py                    # Training pipeline
└─ predict.py                  # Prediction pipeline
```

## ⚙️ Installation

### 1. Create Environment
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
# or
venv\Scripts\activate      # Windows
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
python example_usage.py
```

## 🛠️ Configuration

### `configs/config.yaml`
```yaml
# Data paths
ddl_path: ddl/DDL.sql

source:
  root: data/guidewire
  files:
    Guidewire_Policy: Guidewire_Policy.csv
    Guidewire_Customer: Guidewire_Customer.csv

target:
  root: data/insurenow
  files:
    InsureNow_Contract: InsureNow_Contract.csv
    InsureNow_Client: InsureNow_Client.csv

# Table mappings
table_pairs:
  - [Guidewire_Policy, InsureNow_Contract]
  - [Guidewire_Customer, InsureNow_Client]

# Training settings
train:
  model_out: models/matcher.pkl
  negative_ratio: 4
  test_size: 0.2
  random_state: 42

# Prediction settings
predict:
  model_in: models/matcher.pkl
  top_k: 3
  threshold: 0.5
  out_csv: outputs/mapping_suggestions.csv
  out_json: outputs/mapping_suggestions.json
```

### `configs/synonyms.json`
```json
{
  "Guidewire_Policy::policy_id": ["InsureNow_Contract::contract_key"],
  "Guidewire_Customer::first_name": ["InsureNow_Client::given_name"],
  "Guidewire_Customer::last_name": ["InsureNow_Client::family_name"]
}
```

## 🏋️ Training

### Basic Training
```bash
python main.py train --config configs/config.yaml
```

### Training with Specific Model
```bash
python main.py train \
  --config configs/config.yaml \
  --model-type xgboost \
  --tune-hyperparameters
```

### Available Model Types
- `xgboost`: XGBoost gradient boosting
- `lightgbm`: LightGBM gradient boosting
- `catboost`: CatBoost gradient boosting
- `random_forest`: Random Forest ensemble
- `gradient_boosting`: Scikit-learn gradient boosting
- `logistic_regression`: Logistic regression
- `svm`: Support Vector Machine
- `neural_network`: Multi-layer perceptron
- `ensemble`: Voting ensemble of multiple models

## 🔍 Prediction & Mapping

### Basic Prediction
```bash
python main.py predict --config configs/config.yaml
```

### Prediction with Custom Threshold
```bash
python main.py predict \
  --config configs/config.yaml \
  --threshold 0.7 \
  --include-all-types
```

### Generate Comprehensive Report
```bash
python main.py generate-report \
  --config configs/config.yaml \
  --threshold 0.5 \
  --output-format excel \
  --include-visualizations
```

## 📊 Output Formats

### CSV Output Columns
- `mapping_type`: Type of mapping (one_one, one_many, many_one, many_many)
- `source_table`: Source table name
- `source_column`: Source column name
- `target_table`: Target table name
- `target_columns`: Target column(s)
- `ml_score`: Machine learning prediction score
- `fuzzy_score`: Fuzzy matching score
- `combined_score`: Weighted combination of ML and fuzzy scores
- `confidence`: Overall confidence in the mapping
- `num_targets`: Number of target columns
- `num_sources`: Number of source columns
- `best_alternative_score`: Score of best alternative mapping
- `score_rank`: Rank of this mapping among alternatives
- `is_primary_mapping`: Whether this is the primary mapping

### Excel Report Sheets
1. **Comprehensive_Mappings**: All mapping results
2. **Summary_Mappings**: Best mapping per source column
3. **Statistics**: Summary statistics and metrics

## 🎯 Advanced Usage

### Model Comparison
```bash
python main.py compare-models \
  --config configs/config.yaml \
  --models xgboost lightgbm ensemble \
  --output-dir outputs/comparison
```

### Mapping Analysis
```bash
python main.py analyze-mappings \
  --config configs/config.yaml \
  --threshold 0.5 \
  --output-dir outputs/analysis
```

### Validation
```bash
python main.py validate-mappings \
  --config configs/config.yaml \
  --ground-truth ground_truth.csv \
  --threshold 0.5
```

## 🔧 Customization

### Adding New Features
Extend `featurizer.py` to add custom similarity measures:

```python
def custom_similarity(s_series, t_series):
    # Your custom similarity calculation
    return similarity_score

# Add to column_features function
features["custom_sim"] = custom_similarity(s_series, t_series)
```

### Adding New Models
Extend `model.py` to add custom ML models:

```python
def build_custom_model():
    return YourCustomModel()

# Add to ColumnMappingModel.build_model method
elif self.model_type == "custom":
    self.model = build_custom_model()
```

### Custom Mapping Types
Extend `mapping_engine.py` to add new mapping types:

```python
def find_custom_mappings(self, threshold=0.5):
    # Your custom mapping logic
    return custom_mappings
```

## 📈 Performance Metrics

The system provides comprehensive performance metrics:

- **AUC-ROC**: Area under the ROC curve
- **Average Precision**: Precision-recall curve area
- **Cross-validation Scores**: Mean and standard deviation
- **Feature Importance**: Model feature rankings
- **Mapping Statistics**: Distribution of mapping types and scores

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Memory Issues**: Reduce data size or use smaller models
   ```bash
   python main.py train --model-type random_forest
   ```

3. **Poor Performance**: 
   - Add more synonyms to `configs/synonyms.json`
   - Adjust threshold values
   - Try different model types
   - Increase training data

4. **No Mappings Found**: 
   - Lower the threshold
   - Check table pairs configuration
   - Verify data file paths

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review example usage in `example_usage.py`
3. Open an issue on GitHub
4. Contact the development team

---

**Note**: This system is designed for database migration scenarios where you need to map columns between different schemas. It works best when you have some known mappings (synonyms) to train the model and sufficient data to extract meaningful features.

