from __future__ import annotations
import json
import typer
from pathlib import Path
from typing import Optional
from .config import load_config
from .enhanced_train import train as train_fn, evaluate_model_performance
from .enhanced_predict import predict_mappings, export_mapping_report

app = typer.Typer(help="Enhanced Schema Mapping ML System")

@app.command()
def train(
    config: str = typer.Option(..., help="Path to config.yaml"),
    synonyms: str = typer.Option("synonyms.json", help="Seed synonyms for positives"),
    use_ensemble: bool = typer.Option(True, help="Use ensemble model"),
    tune_hyperparameters: bool = typer.Option(False, help="Perform hyperparameter tuning"),
    output_dir: str = typer.Option("models", help="Output directory for models")
):
    """Train the enhanced schema mapping model"""
    
    print("🚀 Starting Enhanced Schema Mapping Training...")
    
    # Load configuration
    cfg = load_config(config)
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Train the model
    metrics = train_fn(
        ddl_path=cfg.paths.ddl_path,
        source_root=cfg.paths.source_root, 
        source_files=cfg.paths.source_files,
        target_root=cfg.paths.target_root, 
        target_files=cfg.paths.target_files,
        table_pairs=cfg.table_pairs,
        synonyms_path=synonyms,
        negative_ratio=cfg.train.negative_ratio,
        test_size=cfg.train.test_size,
        random_state=cfg.train.random_state,
        model_out=f"{output_dir}/enhanced_matcher.pkl",
        use_ensemble=use_ensemble,
        tune_hyperparameters=tune_hyperparameters
    )
    
    print("\n📊 Training Results:")
    print(json.dumps(metrics, indent=2))
    
    # Save training metrics
    with open(f"{output_dir}/training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

@app.command()
def predict(
    config: str = typer.Option(..., help="Path to config.yaml"),
    model_path: str = typer.Option("models/enhanced_matcher.pkl", help="Path to trained model"),
    featurizer_path: str = typer.Option("models/enhanced_matcher_featurizer.pkl", help="Path to featurizer"),
    threshold: float = typer.Option(0.5, help="Mapping threshold"),
    output_dir: str = typer.Option("outputs", help="Output directory for results")
):
    """Generate schema mappings using the trained model"""
    
    print("🔍 Starting Enhanced Schema Mapping Prediction...")
    
    # Load configuration
    cfg = load_config(config)
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate predictions
    results = predict_mappings(
        ddl_path=cfg.paths.ddl_path,
        source_root=cfg.paths.source_root, 
        source_files=cfg.paths.source_files,
        target_root=cfg.paths.target_root, 
        target_files=cfg.paths.target_files,
        table_pairs=cfg.table_pairs,
        model_in=model_path,
        featurizer_in=featurizer_path,
        threshold=threshold,
        out_csv=f"{output_dir}/enhanced_mapping_suggestions.csv",
        out_json=f"{output_dir}/enhanced_mapping_suggestions.json",
        out_detailed_csv=f"{output_dir}/detailed_mapping_analysis.csv"
    )
    
    print("\n📈 Prediction Results:")
    print(json.dumps(results, indent=2))

@app.command()
def evaluate(
    config: str = typer.Option(..., help="Path to config.yaml"),
    model_path: str = typer.Option("models/enhanced_matcher.pkl", help="Path to trained model"),
    featurizer_path: str = typer.Option("models/enhanced_matcher_featurizer.pkl", help="Path to featurizer"),
    synonyms: str = typer.Option("synonyms.json", help="Seed synonyms for validation")
):
    """Evaluate model performance on validation data"""
    
    print("📊 Starting Model Evaluation...")
    
    # Load configuration
    cfg = load_config(config)
    
    # Evaluate model
    results = evaluate_model_performance(
        model_path=model_path,
        featurizer_path=featurizer_path,
        ddl_path=cfg.paths.ddl_path,
        source_root=cfg.paths.source_root, 
        source_files=cfg.paths.source_files,
        target_root=cfg.paths.target_root, 
        target_files=cfg.paths.target_files,
        table_pairs=cfg.table_pairs,
        synonyms_path=synonyms
    )
    
    print("\n📊 Evaluation Results:")
    print(json.dumps(results, indent=2))

@app.command()
def report(
    config: str = typer.Option(..., help="Path to config.yaml"),
    model_path: str = typer.Option("models/enhanced_matcher.pkl", help="Path to trained model"),
    featurizer_path: str = typer.Option("models/enhanced_matcher_featurizer.pkl", help="Path to featurizer"),
    threshold: float = typer.Option(0.5, help="Mapping threshold"),
    output_dir: str = typer.Option("outputs", help="Output directory for reports")
):
    """Generate comprehensive mapping report"""
    
    print("📋 Generating Comprehensive Mapping Report...")
    
    # Load configuration
    cfg = load_config(config)
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate predictions first
    results = predict_mappings(
        ddl_path=cfg.paths.ddl_path,
        source_root=cfg.paths.source_root, 
        source_files=cfg.paths.source_files,
        target_root=cfg.paths.target_root, 
        target_files=cfg.paths.target_files,
        table_pairs=cfg.table_pairs,
        model_in=model_path,
        featurizer_in=featurizer_path,
        threshold=threshold,
        out_csv=f"{output_dir}/mapping_suggestions.csv",
        out_json=f"{output_dir}/mapping_suggestions.json",
        out_detailed_csv=f"{output_dir}/detailed_analysis.csv"
    )
    
    # Load the mappings for report generation
    import joblib
    from .mapping_engine import MappingEngine
    from .enhanced_featurizer import EnhancedFeaturizer
    from .enhanced_model import build_model
    
    model = build_model().load(model_path)
    featurizer = joblib.load(featurizer_path)
    ddl = cfg.paths.ddl_path  # You'll need to load this properly
    
    # Generate comprehensive report
    report_files = export_mapping_report(
        mappings=results,  # This needs to be the actual mappings object
        output_dir=output_dir
    )
    
    print("\n📋 Report Generated:")
    for report_type, file_path in report_files.items():
        print(f"  - {report_type}: {file_path}")

@app.command()
def quick_mapping(
    source_files: str = typer.Option(..., help="Comma-separated list of source CSV files"),
    target_files: str = typer.Option(..., help="Comma-separated list of target CSV files"),
    ddl_file: str = typer.Option(None, help="DDL file path (optional)"),
    output_dir: str = typer.Option("quick_output", help="Output directory"),
    threshold: float = typer.Option(0.5, help="Mapping threshold")
):
    """Quick mapping without training - uses default model"""
    
    print("⚡ Starting Quick Mapping...")
    
    # Parse file lists
    source_file_list = [f.strip() for f in source_files.split(",")]
    target_file_list = [f.strip() for f in target_files.split(",")]
    
    # Create simple configuration
    source_files_dict = {f"source_{i}": Path(f).name for i, f in enumerate(source_file_list)}
    target_files_dict = {f"target_{i}": Path(f).name for i, f in enumerate(target_file_list)}
    
    # Create table pairs
    table_pairs = [[f"source_{i}", f"target_{i}"] for i in range(min(len(source_file_list), len(target_file_list)))]
    
    # For quick mapping, we'll use a simple approach
    print("⚠️  Quick mapping requires a pre-trained model. Please use 'train' command first.")
    print("Or use the 'predict' command with an existing model.")

@app.command()
def compare_models(
    config: str = typer.Option(..., help="Path to config.yaml"),
    model_paths: str = typer.Option(..., help="Comma-separated list of model paths"),
    output_dir: str = typer.Option("comparison", help="Output directory for comparison")
):
    """Compare multiple trained models"""
    
    print("🔬 Starting Model Comparison...")
    
    # Load configuration
    cfg = load_config(config)
    
    # Parse model paths
    model_path_list = [p.strip() for p in model_paths.split(",")]
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    comparison_results = {}
    
    for model_path in model_path_list:
        print(f"Evaluating model: {model_path}")
        
        # Extract model name
        model_name = Path(model_path).stem
        
        try:
            # Evaluate model
            featurizer_path = model_path.replace('.pkl', '_featurizer.pkl')
            
            results = evaluate_model_performance(
                model_path=model_path,
                featurizer_path=featurizer_path,
                ddl_path=cfg.paths.ddl_path,
                source_root=cfg.paths.source_root, 
                source_files=cfg.paths.source_files,
                target_root=cfg.paths.target_root, 
                target_files=cfg.paths.target_files,
                table_pairs=cfg.table_pairs,
                synonyms_path="synonyms.json"
            )
            
            comparison_results[model_name] = results
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            comparison_results[model_name] = {"error": str(e)}
    
    # Save comparison results
    with open(f"{output_dir}/model_comparison.json", "w") as f:
        json.dump(comparison_results, f, indent=2, default=str)
    
    print(f"\n📊 Comparison results saved to: {output_dir}/model_comparison.json")

@app.command()
def analyze_data(
    config: str = typer.Option(..., help="Path to config.yaml"),
    output_dir: str = typer.Option("analysis", help="Output directory for analysis")
):
    """Analyze source and target data for mapping insights"""
    
    print("📊 Starting Data Analysis...")
    
    # Load configuration
    cfg = load_config(config)
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    from .data_loader import load_tables
    from .ddl_parser import load_ddl
    
    src = load_tables(cfg.paths.source_root, cfg.paths.source_files)
    tgt = load_tables(cfg.paths.target_root, cfg.paths.target_files)
    ddl = load_ddl(cfg.paths.ddl_path)
    
    # Analyze source data
    source_analysis = {}
    for table_name, df in src.items():
        source_analysis[table_name] = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "data_types": df.dtypes.to_dict(),
            "null_counts": df.isnull().sum().to_dict(),
            "unique_counts": df.nunique().to_dict()
        }
    
    # Analyze target data
    target_analysis = {}
    for table_name, df in tgt.items():
        target_analysis[table_name] = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "data_types": df.dtypes.to_dict(),
            "null_counts": df.isnull().sum().to_dict(),
            "unique_counts": df.nunique().to_dict()
        }
    
    # Save analysis
    analysis_results = {
        "source_analysis": source_analysis,
        "target_analysis": target_analysis,
        "ddl_info": ddl
    }
    
    with open(f"{output_dir}/data_analysis.json", "w") as f:
        json.dump(analysis_results, f, indent=2, default=str)
    
    print(f"📊 Data analysis saved to: {output_dir}/data_analysis.json")

if __name__ == "__main__":
    app()