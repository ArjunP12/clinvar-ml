#!/usr/bin/env python3
"""
Main pipeline module.

Loads ClinVar data, runs GroupKFold CV with three models, performs temporal
external validation, pairwise DeLong significance tests, baseline comparisons,
and an ablation study over feature groups.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config
from src.data_loader import load_clinvar_dataset, temporal_train_test_split
from src.model import train_and_evaluate, train_external_validation, MODELS
from src.evaluate import ablation_report, compare_to_baselines, pairwise_delong_tests
from src.inference import save_model
from src.experiment_logger import log_experiment


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging to console with timestamps."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(description="ClinVar Variant Pathogenicity ML Pipeline")
    parser.add_argument("--data", type=Path, default=config.paths.variant_file, help="Path to ClinVar TSV/TSV.GZ")
    parser.add_argument("--results-dir", type=Path, default=config.paths.results_dir, help="Output directory for results")
    parser.add_argument("--models-dir", type=Path, default=config.paths.models_dir, help="Output directory for models")
    parser.add_argument("--test-year", type=int, default=config.data.test_year, help="Year cutoff for temporal holdout")
    parser.add_argument("--max-rows", type=int, default=config.data.max_rows, help="Max rows to load (None for all)")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING)")
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    results_dir = args.results_dir
    models_dir = args.models_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load ClinVar data
    logger.info("Loading ClinVar from %s", args.data)
    df = load_clinvar_dataset(str(args.data), max_rows=args.max_rows)

    # 2. Temporal holdout split
    train_df, test_df = temporal_train_test_split(df, test_year=args.test_year)
    logger.info("Train set: %d variants, Test set: %d variants", len(train_df), len(test_df))

    # 3. Use full train set with class weights
    n_pos = int(train_df["label"].sum())
    n_neg = int((train_df["label"] == 0).sum())
    logger.info("Using full train set with class weights (%d pathogenic, %d benign)", n_pos, n_neg)

    # 4. Feature set
    feature_cols = config.features.base_features

    # 5. GroupKFold CV
    logger.info("Running baseline models with GroupKFold ...")
    fold_probs, outputs = train_and_evaluate(train_df, feature_cols, output_dir=results_dir)
    logger.info("CV Summary:\n%s", outputs["summary"])

    # 6. External temporal validation
    logger.info("Running external validation on temporal holdout ...")
    ext_results = train_external_validation(train_df, test_df, feature_cols, output_dir=results_dir)

    # 7. Pairwise DeLong tests
    logger.info("Running pairwise DeLong tests ...")
    delong_results = pairwise_delong_tests(fold_probs, train_df["label"].values)
    delong_results.to_csv(results_dir / "delong_tests.csv", index=False)
    logger.info("DeLong results:\n%s", delong_results)

    # 8. Compare against majority-class baseline
    logger.info("Comparing to baselines ...")
    baseline_results = compare_to_baselines(outputs, train_df["label"].values)
    logger.info("Baseline results:\n%s", baseline_results)

    # 9. Ablation study
    logger.info("Running ablation study ...")
    feature_groups = {
        "full": feature_cols + config.features.gene_features,
        "clinical_only": ["review_score"] + config.features.gene_features,
        "structural_only": ["is_snv", "ref_len", "alt_len", "length_change", "is_insertion", "is_deletion"],
        "review_only": ["review_score"],
        "gene_only": config.features.gene_features,
    }
    ablation_df = ablation_report(train_df, feature_groups, output_dir=results_dir)
    ablation_df.to_csv(results_dir / "ablation_report.csv", index=False)
    logger.info("Ablation results:\n%s", ablation_df)

    # 10. Train final model and save bundle
    logger.info("Training final XGBoost model on full training set ...")
    base_cols = [c for c in feature_cols if c not in config.features.gene_features]
    X_train = train_df[base_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y_train = train_df["label"].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    final_model = MODELS["XGBoost"](y_train)
    final_model.fit(X_train_s, y_train)

    save_model(final_model, feature_cols, scaler, models_dir / "best_xgb.pkl")
    logger.info("Final model bundle saved to %s", models_dir / "best_xgb.pkl")

    # 11. Log experiment
    cv_summary = outputs["summary"].to_dict()
    external_auc = {k: v["roc_auc"] for k, v in ext_results.items()}
    experiment_metrics = {
        "cv_summary": cv_summary,
        "external_auc": external_auc,
        "delong_tests": delong_results.to_dict(orient="records"),
        "baseline_comparison": baseline_results.to_dict(orient="records"),
        "ablation_summary": ablation_df.set_index(["feature_set", "model"]).to_dict(orient="index"),
    }
    log_experiment(
        config={
            "data": config.data.__dict__,
            "features": config.features.__dict__,
            "model": config.model.__dict__,
        },
        metrics=experiment_metrics,
        output_path=results_dir / "experiment_log.json",
    )
    logger.info("Experiment log saved to %s", results_dir / "experiment_log.json")

    logger.info("Results saved to %s", results_dir.resolve())


if __name__ == "__main__":
    main()
