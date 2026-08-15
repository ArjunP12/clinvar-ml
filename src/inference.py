"""Inference module for saving and loading trained models."""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Union, Any


def save_model(model: Any, feature_cols: list, scaler: Any, path: Path) -> None:
    """Serialize model, feature metadata, and scaler for deployment.
    
    Args:
        model: Trained scikit-learn/XGBoost model.
        feature_cols: Ordered list of feature names used for training.
        scaler: Fitted StandardScaler instance.
        path: Path to save the serialized bundle.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "features": feature_cols,
        "scaler": scaler,
    }, path)


def load_model(path: Path) -> Dict:
    """Load serialized model bundle containing model, features, and scaler.
    
    Returns:
        Dict with keys: model, features, scaler.
    """
    return joblib.load(path)


def predict_single(
    bundle: Dict,
    variant_dict: dict,
) -> float:
    """Predict pathogenicity probability for a single variant.
    
    Args:
        bundle: Loaded model dict from load_model().
        variant_dict: Feature values keyed by feature name.
    
    Returns:
        Probability of being pathogenic (0.0 to 1.0).
    """
    feature_cols = bundle["features"]
    scaler = bundle["scaler"]
    model = bundle["model"]
    
    X = pd.DataFrame([variant_dict])[feature_cols]
    X_scaled = scaler.transform(X)
    return model.predict_proba(X_scaled)[0, 1]


def predict_batch(
    bundle: Dict,
    df: pd.DataFrame,
) -> np.ndarray:
    """Predict pathogenicity probabilities for a batch of variants.
    
    Args:
        bundle: Loaded model dict from load_model().
        df: DataFrame containing at least the required feature columns.
    
    Returns:
        Array of probabilities, one per row.
    """
    feature_cols = bundle["features"]
    scaler = bundle["scaler"]
    model = bundle["model"]
    
    X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    X_scaled = scaler.transform(X)
    return model.predict_proba(X_scaled)[:, 1]
