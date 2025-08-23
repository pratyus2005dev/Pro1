# Schema Mapping Automation Tool

A comprehensive Python tool for automated database schema mapping using machine learning and fuzzy matching algorithms. This tool helps automate the process of mapping columns between source and target database schemas during data migration projects.

## Features

### 🎯 Core Capabilities
- **DDL Parsing**: Automatically extracts table and column metadata from SQL DDL files
- **Fuzzy Matching**: Uses multiple string similarity algorithms for column name matching
- **Machine Learning**: Employs gradient boosting, random forest, and XGBoost models for intelligent mapping
- **Multiple Mapping Types**: Supports 1:1, 1:many, many:1, and many:many mappings
- **Confidence Scoring**: Provides confidence levels (high, medium, low) for each mapping
- **Reference Learning**: Can learn from existing mapping examples for better accuracy

### 🔧 Technical Features
- **Multiple ML Models**: Gradient Boosting, Random Forest, and XGBoost ensemble
- **Comprehensive Fuzzy Matching**: 
  - Levenshtein distance
  - Fuzzy ratio and partial ratio
  - Token sort and set ratios
  - Sequence matching
  - TF-IDF cosine similarity
- **Semantic Analysis**: Considers data types, constraints, and naming patterns
- **Metadata Integration**: Uses both structural and semantic features for mapping

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Quick Setup
1. Clone or download the project files
2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dependencies
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- xgboost >= 1.5.0
- sqlparse >= 0.4.0
- fuzzywuzzy >= 0.18.0
- python-Levenshtein >= 0.12.0

## Usage

### Basic Usage
```bash
python schema_mapper.py --source-ddl source_schema.sql --target-ddl target_schema.sql --output mappings.csv
```

### Advanced Usage with Reference Mappings
```bash
python schema_mapper.py \
    --source-ddl source_schema.sql \
    --target-ddl target_schema.sql \
    --reference-mappings known_mappings.json \
    --output mappings.csv \
    --top-n 5
```

### Command Line Arguments
- `--source-ddl`: Path to source DDL file (required)
- `--target-ddl`: Path to target DDL file (required) 
- `--output`: Output CSV file path (default: mappings.csv)
- `--reference-mappings`: Path to reference mappings JSON file (optional)
- `--top-n`: Number of top mappings per source column (default: 3)

### Quick Example
Run the included example:
```bash
python run_example.py
```

## Input Formats

### DDL Files
Standard SQL CREATE TABLE statements:
```sql
CREATE TABLE customers (
    customer_id INT PRIMARY KEY AUTO_INCREMENT,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    email_address VARCHAR(255) UNIQUE,
    phone_number VARCHAR(20),
    date_of_birth DATE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);
```

### Reference Mappings (Optional)
JSON file with known correct mappings for training:
```json
[
  {
    "source_table": "customers",
    "source_column": "customer_id", 
    "target_table": "client_profiles",
    "target_column": "client_id",
    "is_correct": 1,
    "mapping_type": "1:1",
    "confidence": "high",
    "notes": "Primary key mapping"
  }
]
```

## Output Format

The tool generates a CSV file with the following columns:

| Column | Description |
|--------|-------------|
| source_table | Source table name |
| source_column | Source column name |
| target_table | Target table name |
| target_column | Target column name |
| fuzzy_score | Fuzzy matching score (0-1) |
| ml_score | Machine learning similarity score (0-1) |
| combined_score | Weighted combination of fuzzy and ML scores |
| mapping_type | Type of mapping (1:1, 1:many, many:1, many:many) |
| confidence | Confidence level (high, medium, low) |
| source_data_type | Source column data type |
| target_data_type | Target column data type |
| type_match | Whether data types match exactly |
| semantic_similarity | Semantic similarity score based on metadata |

## Algorithm Details

### Fuzzy Matching Components
1. **Levenshtein Distance**: Character-level edit distance
2. **Fuzzy Ratio**: Overall string similarity
3. **Partial Ratio**: Substring matching
4. **Token Sort/Set Ratio**: Word-order independent matching
5. **Sequence Matching**: Longest common subsequence
6. **TF-IDF Cosine Similarity**: Character n-gram based similarity

### Machine Learning Features
- Name similarity scores
- Data type compatibility
- Length/precision ratios
- Nullable flag matching
- Constraint similarity
- Name pattern recognition
- Table name similarity

### Scoring Algorithm
```
Combined Score = 0.6 × Fuzzy Score + 0.4 × ML Score
```

Confidence levels:
- **High**: Combined score ≥ 0.8
- **Medium**: Combined score ≥ 0.6
- **Low**: Combined score < 0.6

## Examples

### Example 1: Basic Column Mapping
Source: `customer_id` → Target: `client_id`
- Fuzzy Score: 0.75
- ML Score: 0.85
- Combined Score: 0.79
- Confidence: Medium
- Mapping Type: 1:1

### Example 2: Semantic Mapping
Source: `first_name` → Target: `given_name`
- Fuzzy Score: 0.45
- ML Score: 0.92
- Combined Score: 0.64
- Confidence: Medium
- Mapping Type: 1:1

## Architecture

### Core Components

1. **DDLParser**: Parses SQL DDL files and extracts metadata
2. **FuzzyMatcher**: Implements multiple string similarity algorithms
3. **MLModelTrainer**: Trains and manages machine learning models
4. **SchemaMappingEngine**: Orchestrates the entire mapping process

### Data Flow
```
DDL Files → DDLParser → Table/Column Metadata → 
FuzzyMatcher + MLModelTrainer → SchemaMappingEngine → 
Mapping Results → CSV Output
```

## Customization

### Adding New Fuzzy Algorithms
Extend the `FuzzyMatcher.calculate_fuzzy_score()` method:
```python
def calculate_fuzzy_score(self, source_name: str, target_name: str) -> float:
    scores = []
    # Add your custom algorithm
    custom_score = your_custom_algorithm(source_name, target_name)
    scores.append(custom_score)
    # ... existing algorithms
```

### Custom ML Features
Extend the `MLModelTrainer._extract_features()` method:
```python
def _extract_features(self, source_col: ColumnInfo, target_col: ColumnInfo) -> List[float]:
    features = []
    # Add your custom features
    custom_feature = calculate_custom_feature(source_col, target_col)
    features.append(custom_feature)
    # ... existing features
```

## Performance Considerations

### Scalability
- **Small schemas** (< 50 tables): < 1 minute
- **Medium schemas** (50-200 tables): 1-5 minutes  
- **Large schemas** (200+ tables): 5-30 minutes

### Memory Usage
- Approximately 100MB per 1000 columns
- ML models require additional 50-100MB

### Optimization Tips
1. Use reference mappings for better accuracy
2. Filter out system/audit columns if not needed
3. Run on schemas with similar business domains
4. Adjust `top_n` parameter based on needs

## Troubleshooting

### Common Issues

**Error: "No module named 'fuzzywuzzy'"**
```bash
pip install fuzzywuzzy python-Levenshtein
```

**Error: "Unable to parse DDL file"**
- Ensure DDL file uses standard SQL syntax
- Check for proper table/column definitions
- Remove complex constraints that may cause parsing issues

**Low accuracy scores**
- Provide reference mappings for training
- Ensure schemas are from similar business domains
- Check for consistent naming conventions

### Debug Mode
Add debug prints to see intermediate results:
```python
# In schema_mapper.py, add after line 742:
print(f"Debug: Processing {source_col.name} -> candidates: {len(candidates)}")
```

## Contributing

### Areas for Enhancement
1. **Additional ML Models**: Deep learning approaches
2. **Data Profiling**: Statistical analysis of actual data
3. **GUI Interface**: Web-based interface for easier usage
4. **Database Connectivity**: Direct connection to live databases
5. **Advanced Mapping Types**: Complex transformations and aggregations

### Code Structure
- `schema_mapper.py`: Main application
- `requirements.txt`: Dependencies
- `example_*.sql`: Sample DDL files
- `example_reference_mappings.json`: Sample training data
- `run_example.py`: Demo script

## License

This project is provided as-is for educational and commercial use. Please ensure compliance with all dependency licenses.

## Support

For questions, issues, or feature requests:
1. Check the troubleshooting section
2. Review the example files
3. Examine the algorithm details for customization options

---

**Note**: This tool provides automated suggestions for schema mapping. Always review and validate the results before using in production environments. The accuracy depends on the quality of input schemas and reference mappings provided.

