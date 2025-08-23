#!/usr/bin/env python3
"""
Example usage of the Enhanced Schema Mapping ML System

This script demonstrates how to use the enhanced system programmatically
for training, prediction, and analysis.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_featurizer import EnhancedFeaturizer
from enhanced_model import build_model
from mapping_engine import MappingEngine
from enhanced_train import train as train_fn
from enhanced_predict import predict_mappings
from ddl_parser import load_ddl
from data_loader import load_tables

def create_sample_data():
    """Create sample data for demonstration"""
    
    # Create directories
    Path("data/source").mkdir(parents=True, exist_ok=True)
    Path("data/target").mkdir(parents=True, exist_ok=True)
    
    # Sample customers data
    customers_data = {
        'customer_id': [1, 2, 3, 4, 5],
        'first_name': ['John', 'Jane', 'Bob', 'Alice', 'Charlie'],
        'last_name': ['Doe', 'Smith', 'Johnson', 'Brown', 'Wilson'],
        'email_address': ['john@email.com', 'jane@email.com', 'bob@email.com', 'alice@email.com', 'charlie@email.com'],
        'phone_number': ['555-0101', '555-0102', '555-0103', '555-0104', '555-0105'],
        'date_of_birth': ['1990-01-01', '1992-02-02', '1988-03-03', '1995-04-04', '1985-05-05'],
        'registration_date': ['2020-01-01', '2020-02-01', '2019-12-01', '2020-03-01', '2019-11-01'],
        'status': ['active', 'active', 'inactive', 'active', 'active'],
        'total_orders': [10, 5, 15, 8, 12],
        'lifetime_value': [1000.00, 500.00, 1500.00, 800.00, 1200.00]
    }
    
    # Sample clients data (target)
    clients_data = {
        'client_id': [1, 2, 3, 4, 5],
        'given_name': ['John', 'Jane', 'Bob', 'Alice', 'Charlie'],
        'family_name': ['Doe', 'Smith', 'Johnson', 'Brown', 'Wilson'],
        'email': ['john@email.com', 'jane@email.com', 'bob@email.com', 'alice@email.com', 'charlie@email.com'],
        'contact_phone': ['555-0101', '555-0102', '555-0103', '555-0104', '555-0105'],
        'birth_date': ['1990-01-01', '1992-02-02', '1988-03-03', '1995-04-04', '1985-05-05'],
        'member_since': ['2020-01-01', '2020-02-01', '2019-12-01', '2020-03-01', '2019-11-01'],
        'account_status': ['active', 'active', 'inactive', 'active', 'active'],
        'order_count': [10, 5, 15, 8, 12],
        'total_spent': [1000.00, 500.00, 1500.00, 800.00, 1200.00]
    }
    
    # Sample orders data
    orders_data = {
        'order_id': [1, 2, 3, 4, 5],
        'customer_id': [1, 2, 1, 3, 4],
        'order_date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
        'order_status': ['completed', 'pending', 'completed', 'shipped', 'pending'],
        'total_amount': [150.00, 75.50, 200.00, 125.25, 300.00],
        'shipping_address': ['123 Main St', '456 Oak Ave', '789 Pine Rd', '321 Elm St', '654 Maple Dr'],
        'billing_address': ['123 Main St', '456 Oak Ave', '789 Pine Rd', '321 Elm St', '654 Maple Dr'],
        'payment_method': ['credit_card', 'paypal', 'credit_card', 'bank_transfer', 'credit_card'],
        'delivery_notes': ['Leave at door', 'Call before delivery', 'Office hours only', 'Ring doorbell', 'Gate code 1234']
    }
    
    # Sample transactions data (target)
    transactions_data = {
        'transaction_id': [1, 2, 3, 4, 5],
        'client_id': [1, 2, 1, 3, 4],
        'transaction_date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
        'transaction_status': ['completed', 'pending', 'completed', 'shipped', 'pending'],
        'total_amount': [150.00, 75.50, 200.00, 125.25, 300.00],
        'shipping_address': ['123 Main St', '456 Oak Ave', '789 Pine Rd', '321 Elm St', '654 Maple Dr'],
        'billing_address': ['123 Main St', '456 Oak Ave', '789 Pine Rd', '321 Elm St', '654 Maple Dr'],
        'payment_type': ['credit_card', 'paypal', 'credit_card', 'bank_transfer', 'credit_card'],
        'delivery_instructions': ['Leave at door', 'Call before delivery', 'Office hours only', 'Ring doorbell', 'Gate code 1234']
    }
    
    # Save sample data
    pd.DataFrame(customers_data).to_csv('data/source/customers.csv', index=False)
    pd.DataFrame(clients_data).to_csv('data/target/clients.csv', index=False)
    pd.DataFrame(orders_data).to_csv('data/source/orders.csv', index=False)
    pd.DataFrame(transactions_data).to_csv('data/target/transactions.csv', index=False)
    
    print("✅ Sample data created successfully!")

def example_training():
    """Example of training the enhanced model"""
    
    print("\n🚀 Starting Enhanced Model Training...")
    
    # Configuration
    config = {
        'ddl_path': 'sample_ddl.sql',
        'source_root': 'data/source',
        'source_files': {'customers': 'customers.csv', 'orders': 'orders.csv'},
        'target_root': 'data/target',
        'target_files': {'clients': 'clients.csv', 'transactions': 'transactions.csv'},
        'table_pairs': [['customers', 'clients'], ['orders', 'transactions']],
        'synonyms_path': 'enhanced_synonyms.json',
        'negative_ratio': 4,
        'test_size': 0.2,
        'random_state': 42,
        'model_out': 'models/enhanced_example.pkl',
        'use_ensemble': True,
        'tune_hyperparameters': False
    }
    
    # Ensure models directory exists
    Path('models').mkdir(exist_ok=True)
    
    try:
        # Train the model
        results = train_fn(**config)
        
        print("✅ Training completed successfully!")
        print(f"📊 Training Results:")
        print(f"  - Training samples: {results['train_rows']}")
        print(f"  - Test samples: {results['test_rows']}")
        print(f"  - Features: {results['feature_count']}")
        print(f"  - Test AUC: {results['test_metrics']['auc']:.4f}")
        print(f"  - Test AP: {results['test_metrics']['ap']:.4f}")
        print(f"  - Best model: {results['model_info']['model_type']}")
        
        return results
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None

def example_prediction():
    """Example of using the trained model for prediction"""
    
    print("\n🔍 Starting Enhanced Prediction...")
    
    # Configuration
    config = {
        'ddl_path': 'sample_ddl.sql',
        'source_root': 'data/source',
        'source_files': {'customers': 'customers.csv', 'orders': 'orders.csv'},
        'target_root': 'data/target',
        'target_files': {'clients': 'clients.csv', 'transactions': 'transactions.csv'},
        'table_pairs': [['customers', 'clients'], ['orders', 'transactions']],
        'model_in': 'models/enhanced_example.pkl',
        'featurizer_in': 'models/enhanced_example_featurizer.pkl',
        'threshold': 0.5,
        'out_csv': 'outputs/example_mappings.csv',
        'out_json': 'outputs/example_mappings.json',
        'out_detailed_csv': 'outputs/example_detailed.csv'
    }
    
    # Ensure outputs directory exists
    Path('outputs').mkdir(exist_ok=True)
    
    try:
        # Generate predictions
        results = predict_mappings(**config)
        
        print("✅ Prediction completed successfully!")
        print(f"📊 Prediction Results:")
        print(f"  - One-to-one mappings: {results['mapping_counts']['one_to_one']}")
        print(f"  - One-to-many mappings: {results['mapping_counts']['one_to_many']}")
        print(f"  - Many-to-one mappings: {results['mapping_counts']['many_to_one']}")
        print(f"  - Total candidates: {results['mapping_counts']['total_candidates']}")
        
        return results
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return None

def example_direct_usage():
    """Example of using the components directly"""
    
    print("\n🔧 Direct Component Usage Example...")
    
    try:
        # Load data
        ddl = load_ddl('sample_ddl.sql')
        source_tables = load_tables('data/source', {'customers': 'customers.csv'})
        target_tables = load_tables('data/target', {'clients': 'clients.csv'})
        
        # Initialize components
        featurizer = EnhancedFeaturizer()
        model = build_model(use_ensemble=False)  # Use single model for simplicity
        
        # Create mapping engine
        engine = MappingEngine(featurizer, model, ddl)
        
        # Generate mappings
        mappings = engine.generate_final_mappings(
            source_tables, target_tables, [['customers', 'clients']], threshold=0.5
        )
        
        print("✅ Direct usage completed successfully!")
        print(f"📊 Generated {len(mappings['one_to_one'])} one-to-one mappings")
        
        # Show some example mappings
        for mapping in mappings['one_to_one'][:3]:
            print(f"  {mapping.source_column} -> {mapping.target_column} (score: {mapping.combined_score:.3f})")
        
        return mappings
        
    except Exception as e:
        print(f"❌ Direct usage failed: {e}")
        return None

def main():
    """Main example function"""
    
    print("🎯 Enhanced Schema Mapping ML System - Example Usage")
    print("=" * 60)
    
    # Create sample data
    create_sample_data()
    
    # Example 1: Training
    training_results = example_training()
    
    if training_results:
        # Example 2: Prediction
        prediction_results = example_prediction()
        
        # Example 3: Direct usage
        direct_results = example_direct_usage()
    
    print("\n🎉 Example usage completed!")
    print("\n📁 Generated files:")
    print("  - data/source/ - Sample source data")
    print("  - data/target/ - Sample target data")
    print("  - models/ - Trained models")
    print("  - outputs/ - Mapping results")
    
    print("\n📚 Next steps:")
    print("  1. Review the generated mapping files")
    print("  2. Adjust the configuration in enhanced_config.yaml")
    print("  3. Use your own data by replacing the sample files")
    print("  4. Experiment with different thresholds and parameters")

if __name__ == "__main__":
    main()