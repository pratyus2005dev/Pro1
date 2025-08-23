from __future__ import annotations
import json
import typer
from pathlib import Path
from typing import Optional, List
import pandas as pd
from src.config import load_config
from src.train import train as train_fn
from src.predict import predict_mappings
from src.mapping_engine import ColumnMappingEngine, MappingType
from src.model import ColumnMappingModel, build_model

app = typer.Typer(help="Advanced Column Mapping ML with Multiple Mapping Types")

@app.command()
def train(
    config: str = typer.Option(..., help="Path to configs/config.yaml"),
    synonyms: str = typer.Option("configs/synonyms.json", help="Seed synonyms for positives"),
    model_type: str = typer.Option("ensemble", help="Model type: xgboost, lightgbm, catboost, ensemble, etc."),
    tune_hyperparameters: bool = typer.Option(False, help="Perform hyperparameter tuning")
):
    """Train the column mapping model with enhanced features."""
    cfg = load_config(config)
    
    # Build the specified model
    model = build_model(model_type=model_type, random_state=cfg.train.random_state)
    
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
        model_out=cfg.train.model_out,
        model=model,
        tune_hyperparameters=tune_hyperparameters
    )
    typer.echo(json.dumps(metrics, indent=2))

@app.command()
def predict(
    config: str = typer.Option(..., help="Path to configs/config.yaml"),
    threshold: float = typer.Option(0.5, help="Mapping threshold"),
    include_all_types: bool = typer.Option(True, help="Include all mapping types")
):
    """Predict column mappings with enhanced scoring and multiple mapping types."""
    cfg = load_config(config)
    
    out = predict_mappings(
        ddl_path=cfg.paths.ddl_path,
        source_root=cfg.paths.source_root, 
        source_files=cfg.paths.source_files,
        target_root=cfg.paths.target_root, 
        target_files=cfg.paths.target_files,
        table_pairs=cfg.table_pairs,
        model_in=cfg.predict.model_in,
        top_k=cfg.predict.top_k,
        threshold=threshold,
        out_csv=cfg.predict.out_csv,
        out_json=cfg.predict.out_json,
        include_all_types=include_all_types
    )
    typer.echo(json.dumps(out, indent=2))

@app.command()
def analyze_mappings(
    config: str = typer.Option(..., help="Path to configs/config.yaml"),
    threshold: float = typer.Option(0.5, help="Mapping threshold"),
    output_dir: str = typer.Option("outputs", help="Output directory for analysis")
):
    """Analyze mappings with detailed statistics and visualizations."""
    cfg = load_config(config)
    
    from src.analyzer import MappingAnalyzer
    
    analyzer = MappingAnalyzer(
        ddl_path=cfg.paths.ddl_path,
        source_root=cfg.paths.source_root,
        target_root=cfg.paths.target_root,
        model_path=cfg.predict.model_in
    )
    
    results = analyzer.analyze_mappings(
        table_pairs=cfg.table_pairs,
        threshold=threshold,
        output_dir=output_dir
    )
    
    typer.echo(json.dumps(results, indent=2))

@app.command()
def compare_models(
    config: str = typer.Option(..., help="Path to configs/config.yaml"),
    models: List[str] = typer.Option(["xgboost", "lightgbm", "ensemble"], help="Models to compare"),
    output_dir: str = typer.Option("outputs", help="Output directory for comparison")
):
    """Compare different ML models for column mapping."""
    cfg = load_config(config)
    
    from src.model_comparison import ModelComparator
    
    comparator = ModelComparator(
        ddl_path=cfg.paths.ddl_path,
        source_root=cfg.paths.source_root,
        target_root=cfg.paths.target_root,
        synonyms_path="configs/synonyms.json"
    )
    
    results = comparator.compare_models(
        models=models,
        table_pairs=cfg.table_pairs,
        output_dir=output_dir
    )
    
    typer.echo(json.dumps(results, indent=2))

@app.command()
def generate_report(
    config: str = typer.Option(..., help="Path to configs/config.yaml"),
    threshold: float = typer.Option(0.5, help="Mapping threshold"),
    output_format: str = typer.Option("csv", help="Output format: csv, json, excel"),
    include_visualizations: bool = typer.Option(True, help="Include visualizations")
):
    """Generate comprehensive mapping report with all mapping types."""
    cfg = load_config(config)
    
    from src.report_generator import ReportGenerator
    
    generator = ReportGenerator(
        ddl_path=cfg.paths.ddl_path,
        source_root=cfg.paths.source_root,
        target_root=cfg.paths.target_root,
        model_path=cfg.predict.model_in
    )
    
    report = generator.generate_report(
        table_pairs=cfg.table_pairs,
        threshold=threshold,
        output_format=output_format,
        include_visualizations=include_visualizations
    )
    
    typer.echo(f"Report generated: {report}")

@app.command()
def validate_mappings(
    config: str = typer.Option(..., help="Path to configs/config.yaml"),
    ground_truth: str = typer.Option(None, help="Path to ground truth mappings file"),
    threshold: float = typer.Option(0.5, help="Mapping threshold")
):
    """Validate mappings against ground truth if available."""
    cfg = load_config(config)
    
    from src.validator import MappingValidator
    
    validator = MappingValidator(
        ddl_path=cfg.paths.ddl_path,
        source_root=cfg.paths.source_root,
        target_root=cfg.paths.target_root,
        model_path=cfg.predict.model_in
    )
    
    if ground_truth:
        results = validator.validate_against_ground_truth(
            ground_truth_path=ground_truth,
            table_pairs=cfg.table_pairs,
            threshold=threshold
        )
    else:
        results = validator.validate_internal_consistency(
            table_pairs=cfg.table_pairs,
            threshold=threshold
        )
    
    typer.echo(json.dumps(results, indent=2))

if __name__ == "__main__":
    app()
