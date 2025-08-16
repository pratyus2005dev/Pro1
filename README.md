# Guidewire → InsureNow Schema Mapping (ML Project)

This project builds a **machine learning–based schema mapping tool** to automatically align columns between **Guidewire** (source) and **InsureNow** (target).  
It uses metadata, column name similarity, and statistical profiling of values to suggest the most likely column mappings.

---

## 🚀 Features

- Parses your **DDL.sql** to understand table structures and valid table pairs.
- Generates training data automatically using **weak supervision** (seed synonyms in `configs/synonyms.json`).
- Trains a **classifier (XGBoost or RandomForest)** to predict if a source column matches a target column.
- Produces **mapping suggestions** with scores + alternate candidates.
- Easily swap **sample CSVs → real datasets** without changing code (just edit `configs/config.yaml`).
- Outputs results as both **CSV** and **JSON** for downstream integration.

---

## 📂 Project Structure

schema-mapper-ml/
├─ README.md
├─ requirements.txt
├─ ddl/
│ └─ DDL.sql # schema definition
├─ configs/
│ ├─ config.yaml # paths + settings
│ └─ synonyms.json # seed mappings
├─ data/
│ ├─ guidewire/ # Guidewire CSVs
│ └─ insurenow/ # InsureNow CSVs
├─ models/
│ └─ matcher.pkl # trained model
├─ outputs/
│ ├─ mapping_suggestions.csv
│ └─ mapping_suggestions.json
└─ src/
├─ config.py
├─ ddl_parser.py
├─ data_loader.py
├─ utils.py
├─ featurizer.py
├─ model.py
├─ train.py
└─ predict.py
└─ main.py # CLI entrypoint

---

## ⚙️ Installation

### 1. Create environment  
> Recommended: use a fresh virtual environment so TensorFlow or other libs don’t conflict.

```bash
python -m venv venv
venv\Scripts\activate   # on Windows
source venv/bin/activate   # on Linux/Mac
🛠️ Configuration
configs/config.yaml

Defines dataset paths, table pairs, model paths, thresholds, etc.

Example:

source:
  root: data/guidewire
  files:
    Guidewire_Policy: Guidewire_Policy.csv
target:
  root: data/insurenow
  files:
    InsureNow_Contract: InsureNow_Contract.csv
table_pairs:
  - [Guidewire_Policy, InsureNow_Contract]

configs/synonyms.json

Provides known column mappings (used as positive training labels).

Example:

{
  "Guidewire_Policy::policy_id": ["InsureNow_Contract::contract_key"],
  "Guidewire_Customer::first_name": ["InsureNow_Client::given_name"]
}

🏋️ Training

Run training with your datasets:

python main.py train --config configs/config.yaml


This will:

Parse the DDL.

Load CSVs.

Build training pairs (positives from synonyms, negatives auto-generated).

Train the classifier.

Save model to models/matcher.pkl.

🔍 Prediction

Run prediction on new datasets:

python main.py predict --config configs/config.yaml


Outputs:

outputs/mapping_suggestions.csv (flat best-match table)

outputs/mapping_suggestions.json (detailed with top-k alternates & scores)

