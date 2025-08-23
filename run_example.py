#!/usr/bin/env python3
"""
Example usage of the Schema Mapping Automation Tool
"""

import subprocess
import sys
import os

def run_schema_mapping():
    """Run the schema mapping tool with example files"""
    
    # Check if required files exist
    required_files = [
        'schema_mapper.py',
        'example_source.sql',
        'example_target.sql',
        'example_reference_mappings.json'
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            print(f"❌ Required file not found: {file}")
            return False
    
    print("🚀 Running Schema Mapping Tool Example...")
    print("=" * 50)
    
    # Run the schema mapping tool
    cmd = [
        sys.executable, 'schema_mapper.py',
        '--source-ddl', 'example_source.sql',
        '--target-ddl', 'example_target.sql',
        '--reference-mappings', 'example_reference_mappings.json',
        '--output', 'example_mappings.csv',
        '--top-n', '3'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ Schema mapping completed successfully!")
        print("\n📋 Tool Output:")
        print(result.stdout)
        
        if result.stderr:
            print("\n⚠️ Warnings/Errors:")
            print(result.stderr)
        
        # Check if output file was created
        if os.path.exists('example_mappings.csv'):
            print(f"\n📊 Results saved to: example_mappings.csv")
            
            # Show first few lines of the output
            try:
                import pandas as pd
                df = pd.read_csv('example_mappings.csv')
                print(f"\n📈 Generated {len(df)} mappings")
                print("\n🔍 Sample Results (Top 10):")
                print(df.head(10).to_string(index=False))
                
                # Show confidence distribution
                confidence_counts = df['confidence'].value_counts()
                print(f"\n📊 Confidence Distribution:")
                for conf, count in confidence_counts.items():
                    print(f"  {conf}: {count} ({count/len(df)*100:.1f}%)")
                
            except ImportError:
                print("📄 Output file created successfully (pandas not available for preview)")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running schema mapping tool:")
        print(f"Return code: {e.returncode}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                      check=True, capture_output=True)
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def main():
    """Main function"""
    print("🔧 Schema Mapping Automation Tool - Example Runner")
    print("=" * 60)
    
    # Install dependencies first
    if not install_dependencies():
        return
    
    # Run the example
    success = run_schema_mapping()
    
    if success:
        print("\n🎉 Example completed successfully!")
        print("\nNext steps:")
        print("1. Review the generated mappings in 'example_mappings.csv'")
        print("2. Adjust the reference mappings in 'example_reference_mappings.json' if needed")
        print("3. Run with your own DDL files using the same command structure")
        print("\nFor help: python schema_mapper.py --help")
    else:
        print("\n❌ Example failed. Please check the error messages above.")

if __name__ == "__main__":
    main()