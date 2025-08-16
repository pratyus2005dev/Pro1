from __future__ import annotations
import json
import typer
from src.config import load_config
from src.train import train as train_fn
from src.predict import predict_mappings

app = typer.Typer(help="Guidewire → InsureNow schema mapping ML")

@app.command()
def train(config: str = typer.Option(..., help="Path to configs/config.yaml"),
          synonyms: str = typer.Option("configs/synonyms.json", help="Seed synonyms for positives")):
    cfg = load_config(config)
    metrics = train_fn(
        ddl_path=cfg.paths.ddl_path,
        source_root=cfg.paths.source_root, source_files=cfg.paths.source_files,
        target_root=cfg.paths.target_root, target_files=cfg.paths.target_files,
        table_pairs=cfg.table_pairs,
        synonyms_path=synonyms,
        negative_ratio=cfg.train.negative_ratio,
        test_size=cfg.train.test_size,
        random_state=cfg.train.random_state,
        model_out=cfg.train.model_out
    )
    typer.echo(json.dumps(metrics, indent=2))

@app.command()
def predict(config: str = typer.Option(..., help="Path to configs/config.yaml")):
    cfg = load_config(config)
    out = predict_mappings(
        ddl_path=cfg.paths.ddl_path,
        source_root=cfg.paths.source_root, source_files=cfg.paths.source_files,
        target_root=cfg.paths.target_root, target_files=cfg.paths.target_files,
        table_pairs=cfg.table_pairs,
        model_in=cfg.predict.model_in,
        top_k=cfg.predict.top_k,
        threshold=cfg.predict.threshold,
        out_csv=cfg.predict.out_csv,
        out_json=cfg.predict.out_json
    )
    typer.echo(json.dumps(out, indent=2))

if __name__ == "__main__":
    app()
