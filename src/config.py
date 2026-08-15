"""Centralized configuration for the ClinVar ML pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class DataConfig:
    """Data loading and splitting configuration."""
    max_rows: int | None = 200_000
    test_year: int = 2023
    assembly: str = "GRCh38"


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    base_features: List[str] = field(default_factory=lambda: [
        "review_score",
        "is_snv",
        "ref_len",
        "alt_len",
        "length_change",
        "is_insertion",
        "is_deletion",
        "chrom_num",
        "position",
        "pos_bin",
        "gc_content",
        "is_transition",
        "eval_year",
        "eval_month",
    ])
    gene_features: List[str] = field(default_factory=lambda: ["gene_freq", "gene_path_ratio"])


@dataclass
class ModelConfig:
    """Model hyperparameters."""
    random_state: int = 42
    n_splits: int = 5
    logreg_max_iter: int = 2000
    logreg_c: float = 0.5
    rf_n_estimators: int = 500
    rf_max_depth: int = 10
    rf_min_samples_leaf: int = 3
    xgb_n_estimators: int = 500
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.85
    xgb_colsample_bytree: float = 0.85
    xgb_reg_alpha: float = 0.1
    xgb_reg_lambda: float = 1.0


@dataclass
class PathsConfig:
    """File paths configuration."""
    project_root: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = project_root / "data"
    results_dir: Path = project_root / "results"
    models_dir: Path = project_root / "models"
    variant_file: Path = data_dir / "variant_summary.txt"


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


config = ExperimentConfig()
