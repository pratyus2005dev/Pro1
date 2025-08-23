#!/usr/bin/env python3
"""
Example usage of the Enhanced Column Mapping ML System

This script demonstrates how to use the advanced column mapping system
with multiple ML models, fuzzy matching, and different mapping types.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

from model import ColumnMappingModel, build_model
from mapping_engine import ColumnMappingEngine, MappingType
from report_generator import ReportGenerator
from featurizer import column_features, fuzzy_name_scores
from utils import calculate_cardinality, calculate_uniqueness

def create_sample_data():
    """Create sample source and target data for demonstration."""
    
    # Create sample source data
    source_data = {
        'Guidewire_Policy': pd.DataFrame({
            'policy_id': ['POL001', 'POL002', 'POL003'],
            'customer_name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
            'policy_type': ['Auto', 'Home', 'Life'],
            'premium_amount': [1200.50, 850.75, 2000.00],
            'start_date': ['2023-01-01', '2023-02-15', '2023-03-10'],
            'status': ['Active', 'Active', 'Pending']
        }),
        'Guidewire_Customer': pd.DataFrame({
            'customer_id': ['CUST001', 'CUST002', 'CUST003'],
            'first_name': ['John', 'Jane', 'Bob'],
            'last_name': ['Doe', 'Smith', 'Johnson'],
            'email_address': ['john@email.com', 'jane@email.com', 'bob@email.com'],
            'phone_number': ['555-0101', '555-0102', '555-0103'],
            'address_line1': ['123 Main St', '456 Oak Ave', '789 Pine Rd']
        })
    }
    
    # Create sample target data
    target_data = {
        'InsureNow_Contract': pd.DataFrame({
            'contract_key': ['CON001', 'CON002', 'CON003'],
            'client_full_name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
            'insurance_type': ['Automobile', 'Homeowners', 'Life'],
            'annual_premium': [1200.50, 850.75, 2000.00],
            'effective_date': ['2023-01-01', '2023-02-15', '2023-03-10'],
            'contract_status': ['Active', 'Active', 'Pending']
        }),
        'InsureNow_Client': pd.DataFrame({
            'client_id': ['CLI001', 'CLI002', 'CLI003'],
            'given_name': ['John', 'Jane', 'Bob'],
            'family_name': ['Doe', 'Smith', 'Johnson'],
            'email': ['john@email.com', 'jane@email.com', 'bob@email.com'],
            'contact_phone': ['555-0101', '555-0102', '555-0103'],
            'street_address': ['123 Main St', '456 Oak Ave', '789 Pine Rd']
        })
    }
    
    # Create directories and save data
    os.makedirs('data/guidewire', exist_ok=True)
    os.makedirs('data/insurenow', exist_ok=True)
    
    for table_name, df in source_data.items():
        df.to_csv(f'data/guidewire/{table_name}.csv', index=False)
    
    for table_name, df in target_data.items():
        df.to_csv(f'data/insurenow/{table_name}.csv', index=False)
    
    return source_data, target_data

def create_sample_ddl():
    """Create a sample DDL file."""
    ddl_content = """
-- Sample DDL for demonstration
CREATE TABLE Guidewire_Policy (
    policy_id VARCHAR(50),
    customer_name VARCHAR(100),
    policy_type VARCHAR(50),
    premium_amount DECIMAL(10,2),
    start_date DATE,
    status VARCHAR(20)
);

CREATE TABLE Guidewire_Customer (
    customer_id VARCHAR(50),
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    email_address VARCHAR(100),
    phone_number VARCHAR(20),
    address_line1 VARCHAR(200)
);

CREATE TABLE InsureNow_Contract (
    contract_key VARCHAR(50),
    client_full_name VARCHAR(100),
    insurance_type VARCHAR(50),
    annual_premium DECIMAL(10,2),
    effective_date DATE,
    contract_status VARCHAR(20)
);

CREATE TABLE InsureNow_Client (
    client_id VARCHAR(50),
    given_name VARCHAR(50),
    family_name VARCHAR(50),
    email VARCHAR(100),
    contact_phone VARCHAR(20),
    street_address VARCHAR(200)
);
"""
    
    os.makedirs('ddl', exist_ok=True)
    with open('ddl/DDL.sql', 'w') as f:
        f.write(ddl_content)

def create_sample_synonyms():
    """Create sample synonyms for training."""
    synonyms = {
        "Guidewire_Policy::policy_id": ["InsureNow_Contract::contract_key"],
        "Guidewire_Policy::customer_name": ["InsureNow_Contract::client_full_name"],
        "Guidewire_Policy::policy_type": ["InsureNow_Contract::insurance_type"],
        "Guidewire_Policy::premium_amount": ["InsureNow_Contract::annual_premium"],
        "Guidewire_Policy::start_date": ["InsureNow_Contract::effective_date"],
        "Guidewire_Policy::status": ["InsureNow_Contract::contract_status"],
        "Guidewire_Customer::customer_id": ["InsureNow_Client::client_id"],
        "Guidewire_Customer::first_name": ["InsureNow_Client::given_name"],
        "Guidewire_Customer::last_name": ["InsureNow_Client::family_name"],
        "Guidewire_Customer::email_address": ["InsureNow_Client::email"],
        "Guidewire_Customer::phone_number": ["InsureNow_Client::contact_phone"],
        "Guidewire_Customer::address_line1": ["InsureNow_Client::street_address"]
    }
    
    os.makedirs('configs', exist_ok=True)
    import json
    with open('configs/synonyms.json', 'w') as f:
        json.dump(synonyms, f, indent=2)

def demonstrate_feature_extraction():
    """Demonstrate the enhanced feature extraction capabilities."""
    print("=== Feature Extraction Demonstration ===")
    
    # Sample column data
    source_col = "customer_name"
    target_col = "client_full_name"
    source_data = pd.Series(['John Doe', 'Jane Smith', 'Bob Johnson'])
    target_data = pd.Series(['John Doe', 'Jane Smith', 'Bob Johnson'])
    
    # Calculate fuzzy name scores
    fuzzy_scores = fuzzy_name_scores(source_col, target_col)
    print(f"Fuzzy Name Scores for '{source_col}' -> '{target_col}':")
    for score_name, score_value in fuzzy_scores.items():
        print(f"  {score_name}: {score_value:.3f}")
    
    # Calculate metadata features
    source_cardinality = calculate_cardinality(source_data)
    source_uniqueness = calculate_uniqueness(source_data)
    target_cardinality = calculate_cardinality(target_data)
    target_uniqueness = calculate_uniqueness(target_data)
    
    print(f"\nMetadata Features:")
    print(f"  Source cardinality: {source_cardinality}")
    print(f"  Source uniqueness: {source_uniqueness:.3f}")
    print(f"  Target cardinality: {target_cardinality}")
    print(f"  Target uniqueness: {target_uniqueness:.3f}")
    
    # Calculate comprehensive features
    features = column_features(
        "Guidewire_Policy", source_col, source_data, "VARCHAR(100)",
        "InsureNow_Contract", target_col, target_data, "VARCHAR(100)"
    )
    
    print(f"\nComprehensive Features:")
    for feature_name, feature_value in features.items():
        print(f"  {feature_name}: {feature_value:.3f}")

def demonstrate_model_comparison():
    """Demonstrate different ML models."""
    print("\n=== Model Comparison Demonstration ===")
    
    # Create sample training data
    X = np.random.rand(100, 15)  # 100 samples, 15 features
    y = np.random.randint(0, 2, 100)  # Binary labels
    
    models_to_test = ["xgboost", "lightgbm", "random_forest", "ensemble"]
    
    for model_type in models_to_test:
        print(f"\nTesting {model_type.upper()} model:")
        try:
            model = build_model(model_type=model_type)
            results = model.fit(X, y)
            print(f"  CV Mean Score: {results['cv_mean']:.3f}")
            print(f"  CV Std Score: {results['cv_std']:.3f}")
            if results['feature_importance']:
                print(f"  Feature importance available: Yes")
        except Exception as e:
            print(f"  Error: {e}")

def demonstrate_mapping_engine():
    """Demonstrate the mapping engine capabilities."""
    print("\n=== Mapping Engine Demonstration ===")
    
    # Load sample data
    source_data, target_data = create_sample_data()
    
    # Create a simple model for demonstration
    model = build_model(model_type="random_forest")
    
    # Create mapping engine
    engine = ColumnMappingEngine(model)
    
    # Define table pairs
    table_pairs = [
        ["Guidewire_Policy", "InsureNow_Contract"],
        ["Guidewire_Customer", "InsureNow_Client"]
    ]
    
    # Generate mapping report
    mapping_report = engine.generate_mapping_report(
        source_data, target_data, table_pairs, threshold=0.3
    )
    
    print(f"Mapping Results:")
    print(f"  One-to-One mappings: {len(mapping_report['one_one'])}")
    print(f"  One-to-Many mappings: {len(mapping_report['one_many'])}")
    print(f"  Many-to-One mappings: {len(mapping_report['many_one'])}")
    print(f"  Many-to-Many mappings: {len(mapping_report['many_many'])}")
    
    # Show some example mappings
    if mapping_report['one_one']:
        print(f"\nExample One-to-One Mapping:")
        mapping = mapping_report['one_one'][0]
        print(f"  Source: {mapping.source_table}.{mapping.source_column}")
        print(f"  Target: {mapping.target_table}.{', '.join(mapping.target_columns)}")
        print(f"  Confidence: {mapping.confidence:.3f}")

def demonstrate_report_generation():
    """Demonstrate the report generation capabilities."""
    print("\n=== Report Generation Demonstration ===")
    
    # Create sample data and files
    source_data, target_data = create_sample_data()
    create_sample_ddl()
    create_sample_synonyms()
    
    # Create a simple model
    model = build_model(model_type="random_forest")
    
    # Create report generator
    generator = ReportGenerator(
        ddl_path="ddl/DDL.sql",
        source_root="data/guidewire",
        target_root="data/insurenow",
        model_path="models/demo_model.pkl"
    )
    
    # Save the model for demonstration
    os.makedirs("models", exist_ok=True)
    model.save("models/demo_model.pkl")
    
    # Define table pairs and file mappings
    table_pairs = [
        ["Guidewire_Policy", "InsureNow_Contract"],
        ["Guidewire_Customer", "InsureNow_Client"]
    ]
    
    source_files = {
        "Guidewire_Policy": "Guidewire_Policy.csv",
        "Guidewire_Customer": "Guidewire_Customer.csv"
    }
    
    target_files = {
        "InsureNow_Contract": "InsureNow_Contract.csv",
        "InsureNow_Client": "InsureNow_Client.csv"
    }
    
    try:
        # Generate comprehensive CSV report
        csv_path = generator.generate_comprehensive_csv(
            table_pairs, source_files, target_files, threshold=0.3
        )
        print(f"Generated comprehensive CSV: {csv_path}")
        
        # Generate summary report
        summary_path = generator.generate_summary_report(
            table_pairs, source_files, target_files, threshold=0.3
        )
        print(f"Generated summary CSV: {summary_path}")
        
        # Show sample of the report
        df = pd.read_csv(csv_path)
        print(f"\nSample of generated report ({len(df)} rows):")
        print(df.head(3).to_string())
        
    except Exception as e:
        print(f"Error generating report: {e}")

def main():
    """Main demonstration function."""
    print("Enhanced Column Mapping ML System - Demonstration")
    print("=" * 50)
    
    # Create sample data
    print("Creating sample data...")
    create_sample_data()
    create_sample_ddl()
    create_sample_synonyms()
    
    # Demonstrate features
    demonstrate_feature_extraction()
    demonstrate_model_comparison()
    demonstrate_mapping_engine()
    demonstrate_report_generation()
    
    print("\n" + "=" * 50)
    print("Demonstration completed!")
    print("\nTo use the system with your own data:")
    print("1. Place your source CSV files in data/guidewire/")
    print("2. Place your target CSV files in data/insurenow/")
    print("3. Update configs/synonyms.json with known mappings")
    print("4. Run: python main.py train --config config.yaml")
    print("5. Run: python main.py predict --config config.yaml")
    print("6. Run: python main.py generate_report --config config.yaml")

if __name__ == "__main__":
    main()