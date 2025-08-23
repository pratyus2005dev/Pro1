# Enhanced Schema Mapping ML System

A comprehensive, production-ready machine learning system for automated schema mapping between source and target databases. This enhanced version provides advanced fuzzy matching, multiple ML algorithms, and support for various mapping types (one-to-one, one-to-many, many-to-one, many-to-many).

## 🚀 Key Features

### Advanced Fuzzy Matching
- **Multiple Algorithms**: Levenshtein, Jaro-Winkler, Token-based, Sequence-based
- **Weighted Scoring**: Intelligent combination of different fuzzy metrics
- **Semantic Similarity**: TF-IDF based semantic analysis

### Machine Learning Models
- **Multiple Algorithms**: XGBoost, LightGBM, CatBoost, Random Forest, SVM, Neural Networks
- **Ensemble Methods**: Automatic model selection and ensemble creation
- **Hyperparameter Tuning**: Automated optimization of model parameters
- **Cross-validation**: Robust model evaluation

### Mapping Types Support
- **One-to-One**: Traditional column-to-column mapping
- **One-to-Many**: Single source column maps to multiple target columns
- **Many-to-One**: Multiple source columns map to single target column
- **Many-to-Many**: Complex multi-column relationships

### Comprehensive Analysis
- **Data Profiling**: Statistical analysis of column characteristics
- **Metadata Analysis**: DDL parsing and constraint analysis
- **Quality Scoring**: Confidence metrics and risk assessment
- **Detailed Reporting**: HTML reports, CSV exports, JSON outputs

## 📦 Installation

```bash
# Install dependencies
pip install -r requirements.txt

# For development
pip install -e .
```

## 🏗️ Project Structure

```
enhanced-schema-mapper/
├── README.md
├── ENHANCED_README.md
├── requirements.txt
├── config.yaml
├── synonyms.json
├── enhanced_main.py          # Main CLI interface
├── enhanced_featurizer.py    # Advanced feature engineering
├── enhanced_model.py         # Multi-model framework
├── enhanced_train.py         # Enhanced training pipeline
├── enhanced_predict.py       # Advanced prediction system
├── mapping_engine.py         # Mapping optimization engine
├── ddl_parser.py            # DDL parsing utilities
├── data_loader.py           # Data loading utilities
├── utils.py                 # Utility functions
├── models/                  # Trained models
├── outputs/                 # Generated outputs
├── data/                    # Sample data
│   ├── source/
│   └── target/
└── ddl/                     # DDL files
```

## ⚙️ Configuration

### config.yaml
```yaml
# Data paths
ddl_path: ddl/schema.sql
source:
  root: data/source
  files:
    customers: customers.csv
    orders: orders.csv
target:
  root: data/target
  files:
    clients: clients.csv
    transactions: transactions.csv

# Table mappings
table_pairs:
  - [customers, clients]
  - [orders, transactions]

# Training parameters
train:
  negative_ratio: 4
  test_size: 0.2
  random_state: 42

# Prediction parameters
predict:
  threshold: 0.5
  top_k: 3
```

### synonyms.json
```json
{
  "customers::customer_id": ["clients::client_id"],
  "customers::first_name": ["clients::given_name"],
  "customers::last_name": ["clients::family_name"],
  "orders::order_id": ["transactions::transaction_id"],
  "orders::amount": ["transactions::total_amount"]
}
```

## 🚀 Usage

### 1. Training the Model

```bash
# Basic training
python enhanced_main.py train --config config.yaml

# Training with ensemble and hyperparameter tuning
python enhanced_main.py train \
  --config config.yaml \
  --use-ensemble \
  --tune-hyperparameters \
  --output-dir models/enhanced
```

### 2. Generating Mappings

```bash
# Generate mappings with default threshold
python enhanced_main.py predict \
  --config config.yaml \
  --model-path models/enhanced_matcher.pkl \
  --featurizer-path models/enhanced_matcher_featurizer.pkl

# Generate mappings with custom threshold
python enhanced_main.py predict \
  --config config.yaml \
  --threshold 0.7 \
  --output-dir results
```

### 3. Comprehensive Reporting

```bash
# Generate detailed reports
python enhanced_main.py report \
  --config config.yaml \
  --output-dir reports
```

### 4. Model Evaluation

```bash
# Evaluate model performance
python enhanced_main.py evaluate \
  --config config.yaml \
  --model-path models/enhanced_matcher.pkl
```

### 5. Data Analysis

```bash
# Analyze source and target data
python enhanced_main.py analyze-data \
  --config config.yaml \
  --output-dir analysis
```

### 6. Model Comparison

```bash
# Compare multiple models
python enhanced_main.py compare-models \
  --config config.yaml \
  --model-paths "models/model1.pkl,models/model2.pkl" \
  --output-dir comparison
```

## 📊 Output Files

### Main Outputs
- `enhanced_mapping_suggestions.csv`: Best mappings with scores
- `enhanced_mapping_suggestions.json`: Detailed mapping information
- `detailed_mapping_analysis.csv`: Comprehensive analysis with all features

### Quality Analysis
- `mapping_quality_analysis.csv`: Quality assessment and recommendations
- `mapping_summary.json`: Statistical summary of mappings
- `mapping_report.html`: Interactive HTML report

### Training Outputs
- `training_metrics.json`: Training performance metrics
- `model_comparison.json`: Model comparison results
- `data_analysis.json`: Source/target data analysis

## 🔧 Advanced Features

### Feature Engineering

The system extracts comprehensive features:

**Fuzzy Matching Features:**
- Levenshtein ratio
- Partial ratio
- Token sort/set ratios
- Jaro-Winkler similarity
- Sequence ratio
- Token Jaccard similarity

**Semantic Features:**
- TF-IDF based semantic similarity
- N-gram analysis
- Context-aware matching

**Data Profiling Features:**
- Null rate differences
- Cardinality ratios
- Value overlap analysis
- Length similarity
- Pattern matching

**Metadata Features:**
- Primary key similarity
- Foreign key relationships
- Unique constraint matching
- Nullable constraint matching
- Data type compatibility

### Model Selection

The system automatically selects the best performing model:

1. **Individual Model Evaluation**: Tests all available models
2. **Cross-validation**: Uses 5-fold CV for robust evaluation
3. **Ensemble Creation**: Combines top-performing models
4. **Hyperparameter Tuning**: Optimizes model parameters

### Mapping Optimization

**One-to-One Mappings:**
- Hungarian algorithm for optimal assignment
- Global optimization across all table pairs

**One-to-Many/Many-to-One:**
- Score-based ranking
- Configurable limits per source/target

**Quality Assessment:**
- Confidence scoring
- Risk level classification
- Recommendation generation

## 📈 Performance Metrics

The system provides comprehensive evaluation metrics:

- **ROC-AUC**: Overall model performance
- **Average Precision**: Ranking quality
- **Classification Report**: Precision, recall, F1-score
- **Confidence Distribution**: Mapping reliability
- **Quality Metrics**: High-confidence mapping ratios

## 🛠️ Customization

### Adding New Features

```python
# In enhanced_featurizer.py
def custom_feature(self, s_col: str, t_col: str) -> float:
    # Your custom feature logic
    return similarity_score

# Add to column_features method
features['custom_feature'] = self.custom_feature(s_col, t_col)
```

### Adding New Models

```python
# In enhanced_model.py
def build_custom_model(self):
    return CustomClassifier(
        param1=value1,
        param2=value2
    )

# Add to build_models method
models['custom_model'] = self.build_custom_model()
```

### Custom Mapping Logic

```python
# In mapping_engine.py
def custom_mapping_optimization(self, candidates):
    # Your custom optimization logic
    return optimized_mappings
```

## 🔍 Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce batch size or use data sampling
2. **Slow Training**: Use fewer models or disable hyperparameter tuning
3. **Poor Performance**: Check data quality and feature engineering
4. **Import Errors**: Ensure all dependencies are installed

### Performance Tips

1. **Data Preprocessing**: Clean and normalize your data
2. **Feature Selection**: Remove irrelevant features
3. **Model Selection**: Use ensemble for better performance
4. **Threshold Tuning**: Adjust based on your requirements

## 📚 API Reference

### EnhancedFeaturizer
```python
featurizer = EnhancedFeaturizer()
features = featurizer.column_features(
    s_table, s_col, s_series, s_dtype,
    t_table, t_col, t_series, t_dtype,
    ddl
)
```

### EnhancedModel
```python
model = build_model(use_ensemble=True)
model_info = model.train(X, y, tune_hyperparameters=True)
predictions = model.predict_proba(X_test)
```

### MappingEngine
```python
engine = MappingEngine(featurizer, model, ddl)
mappings = engine.generate_final_mappings(
    source_tables, target_tables, table_pairs, threshold=0.5
)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the API documentation

---

**Note**: This enhanced system is designed for production use and includes comprehensive error handling, logging, and validation. It's suitable for large-scale schema mapping projects and can be easily integrated into existing data migration pipelines.