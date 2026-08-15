"""
Model training and evaluation module.

Provides GroupKFold cross-validation, external temporal validation,
calibration/ROC/PR plotting, and optional SHAP interpretation for
LogisticRegression, RandomForest, and XGBoost models.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

logger = logging.getLogger(__name__)


# Model builders: lambdas receive y_train so scale_pos_weight can be computed per-fold
MODELS = {
    "LogisticRegression": lambda y_train: LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=42, C=0.5
    ),
    "RandomForest": lambda y_train: RandomForestClassifier(
        n_estimators=500, max_depth=10, min_samples_leaf=3, max_features="sqrt",
        class_weight="balanced", random_state=42, n_jobs=-1
    ),
    "XGBoost": lambda y_train: XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85,
        random_state=42, n_jobs=1, scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
        reg_alpha=0.1, reg_lambda=1.0
    ),
}


def train_and_evaluate(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "label",
    group_col: str = "GeneSymbol",
    output_dir: Union[str, Path] = "results",
    n_splits: int = 5,
    do_shap: bool = True,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Run GroupKFold CV for multiple models. Gene-level features (gene_freq,
    gene_path_ratio) are computed per-fold from training data only to prevent leakage.
    
    Returns fold probabilities and a metrics summary dict.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Exclude gene-level features from the base matrix because they are added per-fold below
    base_cols = [c for c in feature_cols if c not in ("gene_freq", "gene_path_ratio")]
    X = df[base_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y = df[target_col].values
    groups = df[group_col].astype(str).values

    missing = [c for c in feature_cols if c not in df.columns and c not in ("gene_freq", "gene_path_ratio")]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")

    unique_groups = np.unique(groups)
    n_splits = min(n_splits, len(unique_groups))
    if n_splits < 2:
        raise ValueError(f"Need at least 2 groups for GroupKFold, got {len(unique_groups)}")

    gkf = GroupKFold(n_splits=n_splits)
    fold_preds = {name: np.zeros(len(df)) for name in MODELS}
    fold_probs = {name: np.zeros(len(df)) for name in MODELS}

    rows = []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        fold_groups = groups[val_idx]
        fold_group_counts = pd.Series(fold_groups).value_counts()
        logger.info(
            "Fold %d: val genes=%d, min_vars=%d, max_vars=%d, pos_rate=%.3f",
            fold, len(fold_group_counts), fold_group_counts.min(),
            fold_group_counts.max(), y_val.mean()
        )

        # Only inject gene-level features when explicitly requested, and compute them
        # from training data only to avoid leakage into the validation fold.
        X_train_f = X_train.copy()
        X_val_f = X_val.copy()
        if "gene_freq" in feature_cols and "GeneSymbol" in df.columns:
            gene_counts = df.iloc[train_idx]["GeneSymbol"].value_counts()
            X_train_f["gene_freq"] = df.iloc[train_idx]["GeneSymbol"].map(gene_counts).fillna(0).values
            X_train_f["gene_freq"] = np.log1p(X_train_f["gene_freq"].clip(lower=0))
            X_val_f["gene_freq"] = df.iloc[val_idx]["GeneSymbol"].map(gene_counts).fillna(0).values
            X_val_f["gene_freq"] = np.log1p(X_val_f["gene_freq"].clip(lower=0))
        if "gene_path_ratio" in feature_cols and "GeneSymbol" in df.columns:
            train_df_fold = df.iloc[train_idx].copy()
            gene_path = train_df_fold.groupby("GeneSymbol")["label"].mean()
            X_train_f["gene_path_ratio"] = df.iloc[train_idx]["GeneSymbol"].map(gene_path).fillna(y_train.mean()).values
            X_val_f["gene_path_ratio"] = df.iloc[val_idx]["GeneSymbol"].map(gene_path).fillna(y_train.mean()).values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_f)
        X_val_s = scaler.transform(X_val_f)

        for name, builder in MODELS.items():
            model = builder(y_train)
            model.fit(X_train_s, y_train)

            fold_preds[name][val_idx] = model.predict(X_val_s)
            if hasattr(model, "predict_proba"):
                fold_probs[name][val_idx] = model.predict_proba(X_val_s)[:, 1]
            else:
                fold_probs[name][val_idx] = model.decision_function(X_val_s)

            acc = accuracy_score(y_val, fold_preds[name][val_idx])
            auc = roc_auc_score(y_val, fold_probs[name][val_idx])
            rows.append({"model": name, "fold": fold, "accuracy": acc, "roc_auc": auc})
            logger.info("Fold %d | %s | Acc: %.3f | AUC: %.3f", fold, name, acc, auc)

    metrics_df = pd.DataFrame(rows)
    metrics_summary = metrics_df.groupby("model")[["accuracy", "roc_auc"]].agg(["mean", "std"]).round(4)
    metrics_summary.to_csv(output_dir / "cv_metrics_summary.csv")

    _plot_roc_curves(y, fold_probs, output_dir)
    _plot_pr_curves(y, fold_probs, output_dir)
    _plot_calibration(y, fold_probs, output_dir)

    # Pass base_cols to SHAP so feature names match the data columns
    if do_shap and SHAP_AVAILABLE and "XGBoost" in MODELS:
        _run_shap(X, y, base_cols, output_dir, groups)

    return fold_probs, {"metrics": metrics_df, "summary": metrics_summary}


def _run_shap(X: pd.DataFrame, y: np.ndarray, feature_cols: List[str], output_dir: Path, groups: np.ndarray = None) -> None:
    """Run SHAP analysis on XGBoost using a single train/test split to avoid leakage."""
    try:
        from sklearn.model_selection import GroupShuffleSplit
        sss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
        train_idx, val_idx = next(sss.split(X, y, groups=groups))
        model = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=1,
            scale_pos_weight=(y[train_idx] == 0).sum() / max((y[train_idx] == 1).sum(), 1)
        )
        model.fit(X.iloc[train_idx].values, y[train_idx])

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X.iloc[val_idx].values)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X.iloc[val_idx], feature_names=feature_cols, show=False)
        plt.tight_layout()
        plt.savefig(output_dir / "shap_summary.png", dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("SHAP summary plot saved to %s", output_dir / "shap_summary.png")
    except Exception as e:
        logger.warning("SHAP analysis failed: %s", e)


def _plot_roc_curves(y: np.ndarray, probs: Dict[str, np.ndarray], output_dir: Path) -> None:
    """Plot ROC curves for all models."""
    plt.figure(figsize=(8, 6))
    for name, prob in probs.items():
        fpr, tpr, _ = roc_curve(y, prob)
        auc = roc_auc_score(y, prob)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="#95a5a6", label="Random Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves: Gene-Holdout Cross-Validation", weight="bold")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curves.png", dpi=300)
    plt.close()


def _plot_pr_curves(y: np.ndarray, probs: Dict[str, np.ndarray], output_dir: Path) -> None:
    """Plot precision-recall curves for all models."""
    plt.figure(figsize=(8, 6))
    for name, prob in probs.items():
        precision, recall, _ = precision_recall_curve(y, prob)
        plt.plot(recall, precision, lw=2, label=name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves: Gene-Holdout Cross-Validation", weight="bold")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "pr_curves.png", dpi=300)
    plt.close()


def _plot_calibration(y: np.ndarray, probs: Dict[str, np.ndarray], output_dir: Path) -> None:
    """Plot calibration curves for all models."""
    plt.figure(figsize=(8, 6))
    for name, prob in probs.items():
        frac_pos, mean_pred = calibration_curve(y, prob, n_bins=10, strategy="uniform")
        brier = brier_score_loss(y, prob)
        plt.plot(mean_pred, frac_pos, "s-", lw=2, label=f"{name} (Brier={brier:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="#95a5a6", label="Perfectly Calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curves: Gene-Holdout Cross-Validation", weight="bold")
    plt.legend(loc="upper left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "calibration_curves.png", dpi=300)
    plt.close()


def train_external_validation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "label",
    output_dir: Path = Path("results"),
) -> Dict:
    """Train on train_df, evaluate on test_df for temporal/holdout validation."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # gene-level features must be computed from train_df only to avoid leakage.
    if "gene_freq" in feature_cols and "GeneSymbol" in train_df.columns:
        gene_counts = train_df["GeneSymbol"].value_counts()
        train_df = train_df.copy()
        train_df["gene_freq"] = train_df["GeneSymbol"].map(gene_counts).fillna(0).values
        train_df["gene_freq"] = np.log1p(train_df["gene_freq"].clip(lower=0))
        test_df = test_df.copy()
        test_df["gene_freq"] = test_df["GeneSymbol"].map(gene_counts).fillna(0).values
        test_df["gene_freq"] = np.log1p(test_df["gene_freq"].clip(lower=0))
    if "gene_path_ratio" in feature_cols and "GeneSymbol" in train_df.columns:
        gene_path = train_df.groupby("GeneSymbol")["label"].mean()
        train_df = train_df.copy()
        train_df["gene_path_ratio"] = train_df["GeneSymbol"].map(gene_path).fillna(train_df["label"].mean()).values
        test_df = test_df.copy()
        test_df["gene_path_ratio"] = test_df["GeneSymbol"].map(gene_path).fillna(train_df["label"].mean()).values

    X_train = train_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y_train = train_df[target_col].values
    X_test = test_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y_test = test_df[target_col].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = {}
    for name, builder in MODELS.items():
        model = builder(y_train)
        model.fit(X_train_s, y_train)

        y_pred = model.predict(X_test_s)
        y_prob = model.predict_proba(X_test_s)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test_s)

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        results[name] = {"accuracy": acc, "roc_auc": auc, "y_pred": y_pred, "y_prob": y_prob}
        logger.info("External Test | %s | Acc: %.3f | AUC: %.3f", name, acc, auc)

    pd.DataFrame(results).T.to_csv(output_dir / "external_validation.csv")
    return results
