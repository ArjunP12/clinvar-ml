"""
Evaluation module.

Provides metric summarization, ablation reports, baseline comparisons,
and pairwise DeLong significance tests for model results.
"""

import logging
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def delong_test(
    y_true: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
    n_bootstraps: int = 1000,
    random_state: int = 42,
) -> dict:
    """Approximate DeLong test via bootstrap on AUCs.
    
    Args:
        y_true: Ground-truth binary labels.
        y1: Predicted probabilities from model 1.
        y2: Predicted probabilities from model 2.
        n_bootstraps: Number of bootstrap samples for variance estimation.
        random_state: Random seed for reproducibility.
    
    Returns:
        Dict with auc_1, auc_2, diff, z, p_value, significant_at_0.05.
    """
    rng = np.random.default_rng(random_state)
    auc1 = roc_auc_score(y_true, y1)
    auc2 = roc_auc_score(y_true, y2)

    diffs = []
    for _ in range(n_bootstraps):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        a1 = roc_auc_score(y_true[idx], y1[idx])
        a2 = roc_auc_score(y_true[idx], y2[idx])
        diffs.append(a1 - a2)

    diffs = np.array(diffs)
    se = np.std(diffs, ddof=1)
    z = (auc1 - auc2) / (se + 1e-12)
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / np.sqrt(2))))

    return {
        "auc_1": float(auc1),
        "auc_2": float(auc2),
        "diff": float(auc1 - auc2),
        "z": float(z),
        "p_value": float(p),
        "significant_at_0.05": bool(p < 0.05),
    }


def ablation_report(
    df: pd.DataFrame,
    feature_groups: dict,
    target_col: str = "label",
    group_col: str = "GeneSymbol",
    output_dir: Path = Path("results"),
) -> pd.DataFrame:
    """Run CV for each feature group and return mean/std AUC per model.
    
    Args:
        df: Full training DataFrame.
        feature_groups: Dict mapping group name to list of feature columns.
        target_col: Name of label column.
        group_col: Name of group column for GroupKFold.
        output_dir: Directory to save per-group results.
    
    Returns:
        DataFrame with columns: feature_set, model, mean_auc, std_auc.
    """
    from src.model import train_and_evaluate

    rows = []
    for name, cols in feature_groups.items():
        _, out = train_and_evaluate(df, cols, target_col, group_col, output_dir / "ablations" / name, n_splits=5, do_shap=False)
        s = out["summary"]
        for model in s.index:
            rows.append({
                "feature_set": name,
                "model": model,
                "mean_auc": s.loc[model, ("roc_auc", "mean")],
                "std_auc": s.loc[model, ("roc_auc", "std")],
            })
    return pd.DataFrame(rows).sort_values(["feature_set", "model"])


def compare_to_baselines(cv_outputs: dict, y: np.ndarray) -> pd.DataFrame:
    """Compare model performance against a majority-class baseline.
    
    Args:
        cv_outputs: Dict with 'metrics' DataFrame from train_and_evaluate.
        y: Ground-truth labels.
    
    Returns:
        DataFrame with baseline names and AUC scores.
    """
    if len(np.unique(y)) < 2:
        majority_auc = float("nan")
    else:
        majority_class = int(np.bincount(y).argmax())
        majority_pred = np.full(len(y), majority_class)
        majority_auc = roc_auc_score(y, majority_pred)

    model_aucs = {}
    for model_name, scores in cv_outputs["metrics"].groupby("model"):
        model_aucs[model_name] = scores["roc_auc"].mean()

    results_data = [
        {"baseline": "majority_class", "auc": majority_auc},
    ]

    for model_name, auc in model_aucs.items():
        results_data.append({"baseline": model_name, "auc": auc})

    return pd.DataFrame(results_data)


def pairwise_delong_tests(fold_probs: dict, y: np.ndarray) -> pd.DataFrame:
    """Run pairwise DeLong tests between all models using precomputed fold probabilities.
    
    Args:
        fold_probs: Dict mapping model name to array of predicted probabilities.
        y: Ground-truth labels.
    
    Returns:
        DataFrame with pairwise comparison results.
    """
    if len(np.unique(y)) < 2:
        return pd.DataFrame(columns=["model_1", "model_2", "auc_1", "auc_2", "diff", "p_value", "significant"])

    model_names = list(fold_probs.keys())
    results = []

    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            m1, m2 = model_names[i], model_names[j]
            test_result = delong_test(y, fold_probs[m1], fold_probs[m2])
            results.append({
                "model_1": m1,
                "model_2": m2,
                "auc_1": test_result["auc_1"],
                "auc_2": test_result["auc_2"],
                "diff": test_result["diff"],
                "p_value": test_result["p_value"],
                "significant": test_result["significant_at_0.05"],
            })

    return pd.DataFrame(results)
