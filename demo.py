#!/usr/bin/env python3
"""
Demo script for the Schema Mapping Automation Tool
"""

import pandas as pd
import os

def demo():
    """Run a demonstration of the schema mapping tool"""
    
    print("🎯 Schema Mapping Automation Tool - Demo")
    print("=" * 50)
    
    # Check if the example results exist
    if not os.path.exists('example_mappings.csv'):
        print("❌ Example mappings not found. Please run the tool first:")
        print("python3 schema_mapper.py --source-ddl example_source.sql --target-ddl example_target.sql --reference-mappings example_reference_mappings.json --output example_mappings.csv")
        return
    
    # Load the results
    df = pd.read_csv('example_mappings.csv')
    
    print(f"📊 Generated {len(df)} total mappings")
    print(f"🎯 Confidence distribution:")
    
    confidence_counts = df['confidence'].value_counts()
    for conf, count in confidence_counts.items():
        print(f"   {conf.capitalize()}: {count} ({count/len(df)*100:.1f}%)")
    
    print(f"\n🔥 Top 10 High-Confidence Mappings:")
    print("-" * 80)
    
    high_conf = df[df['confidence'] == 'high'].sort_values('combined_score', ascending=False).head(10)
    
    for _, row in high_conf.iterrows():
        print(f"📌 {row['source_table']}.{row['source_column']} → {row['target_table']}.{row['target_column']}")
        print(f"   Score: {row['combined_score']:.3f} | Fuzzy: {row['fuzzy_score']:.3f} | ML: {row['ml_score']:.3f}")
        print(f"   Types: {row['source_data_type']} → {row['target_data_type']} | Match: {row['type_match']}")
        print()
    
    print("🎨 Mapping Types Distribution:")
    mapping_types = df['mapping_type'].value_counts()
    for mtype, count in mapping_types.items():
        print(f"   {mtype}: {count} ({count/len(df)*100:.1f}%)")
    
    print(f"\n📈 Score Statistics:")
    print(f"   Average Combined Score: {df['combined_score'].mean():.3f}")
    print(f"   Average Fuzzy Score: {df['fuzzy_score'].mean():.3f}")
    print(f"   Average ML Score: {df['ml_score'].mean():.3f}")
    print(f"   Average Semantic Similarity: {df['semantic_similarity'].mean():.3f}")
    
    print(f"\n🔍 Data Type Analysis:")
    type_matches = df['type_match'].sum()
    print(f"   Exact type matches: {type_matches} ({type_matches/len(df)*100:.1f}%)")
    
    # Show some interesting semantic matches
    print(f"\n🧠 Interesting Semantic Matches (Different names, high scores):")
    print("-" * 80)
    
    semantic_matches = df[
        (df['fuzzy_score'] < 0.7) & 
        (df['combined_score'] > 0.8) & 
        (df['confidence'] == 'high')
    ].head(5)
    
    for _, row in semantic_matches.iterrows():
        print(f"🎯 {row['source_column']} → {row['target_column']}")
        print(f"   Low name similarity ({row['fuzzy_score']:.3f}) but high overall score ({row['combined_score']:.3f})")
        print(f"   ML model recognized the semantic relationship!")
        print()
    
    print("✨ Demo completed! Check 'example_mappings.csv' for full results.")

if __name__ == "__main__":
    demo()