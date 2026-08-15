import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_clinvar_dataset, temporal_train_test_split
from src.inference import load_model, predict_batch, predict_single, save_model
from src.model import MODELS


def test_load_returns_dataframe(tmp_path):
    """Test that load_clinvar_dataset returns a valid DataFrame."""
    from tests.conftest import make_synthetic_clinvar
    path = make_synthetic_clinvar(200)
    try:
        df = load_clinvar_dataset(str(path), max_rows=200)
        assert len(df) > 0
        assert "label" in df.columns
        assert df["label"].isin([0, 1]).all()
    finally:
        path.unlink(missing_ok=True)


def test_required_features_exist(tmp_path):
    """Test that all expected features are present after loading."""
    from tests.conftest import make_synthetic_clinvar
    path = make_synthetic_clinvar(200)
    try:
        df = load_clinvar_dataset(str(path), max_rows=200)
        required = [
            "review_score", "is_snv", "ref_len", "alt_len",
            "length_change", "is_insertion", "is_deletion",
            "chrom_num", "position", "pos_bin", "gc_content",
            "is_transition", "eval_year", "eval_month",
        ]
        for col in required:
            assert col in df.columns, f"Missing feature: {col}"
    finally:
        path.unlink(missing_ok=True)


def test_temporal_split(tmp_path):
    """Test that temporal_train_test_split correctly partitions data."""
    from src.data_loader import temporal_train_test_split
    from tests.conftest import make_synthetic_clinvar
    path = make_synthetic_clinvar(200)
    try:
        df = load_clinvar_dataset(str(path), max_rows=200)
        train_df, test_df = temporal_train_test_split(df, test_year=2023)
        assert len(train_df) + len(test_df) == len(df)
        assert test_df["eval_year"].min() >= 2023
        assert train_df["eval_year"].max() < 2023
    finally:
        path.unlink(missing_ok=True)


def test_model_save_load_roundtrip(tmp_path):
    """Test that a model bundle can be saved and loaded correctly."""
    X = pd.DataFrame({
        "review_score": [0, 1, 2, 3, 4],
        "is_snv": [1, 1, 0, 1, 1],
        "ref_len": [1, 1, 3, 1, 1],
        "alt_len": [1, 1, 1, 1, 2],
    })
    y = np.array([0, 1, 0, 1, 0])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = MODELS["XGBoost"](y)
    model.fit(X_scaled, y)
    
    bundle_path = tmp_path / "test_model.pkl"
    save_model(model, list(X.columns), scaler, bundle_path)
    assert bundle_path.exists()
    
    bundle = load_model(bundle_path)
    assert "model" in bundle
    assert "features" in bundle
    assert "scaler" in bundle
    assert bundle["features"] == list(X.columns)


def test_predict_single(tmp_path):
    """Test single-variant prediction from a saved bundle."""
    X = pd.DataFrame({
        "review_score": [0, 1, 2, 3, 4],
        "is_snv": [1, 1, 0, 1, 1],
        "ref_len": [1, 1, 3, 1, 1],
        "alt_len": [1, 1, 1, 1, 2],
    })
    y = np.array([0, 1, 0, 1, 0])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = MODELS["XGBoost"](y)
    model.fit(X_scaled, y)
    
    bundle = {"model": model, "features": list(X.columns), "scaler": scaler}
    variant = {"review_score": 2, "is_snv": 1, "ref_len": 1, "alt_len": 1}
    prob = predict_single(bundle, variant)
    assert 0.0 <= prob <= 1.0
