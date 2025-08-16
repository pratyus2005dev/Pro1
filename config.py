from __future__ import annotations
import yaml
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class Paths:
    ddl_path: str
    source_root: str
    target_root: str
    source_files: Dict[str, str]
    target_files: Dict[str, str]

@dataclass
class TrainCfg:
    model_out: str
    negative_ratio: int
    test_size: float
    random_state: int

@dataclass
class PredictCfg:
    model_in: str
    top_k: int
    threshold: float
    out_csv: str
    out_json: str

@dataclass
class AppConfig:
    paths: Paths
    table_pairs: List[List[str]]
    train: TrainCfg
    predict: PredictCfg

def load_config(path: str) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return AppConfig(
        paths=Paths(
            ddl_path=cfg["ddl_path"],
            source_root=cfg["source"]["root"],
            target_root=cfg["target"]["root"],
            source_files=cfg["source"]["files"],
            target_files=cfg["target"]["files"],
        ),
        table_pairs=cfg["table_pairs"],
        train=TrainCfg(**cfg["train"]),
        predict=PredictCfg(**cfg["predict"]),
    )
