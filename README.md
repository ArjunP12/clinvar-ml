# ClinVar Variant Pathogenicity Predictor

An end-to-end Machine Learning pipeline for predicting the clinical significance 
of human genetic variants as **Pathogenic** or **Benign** using **XGBoost**, 
**Random Forest**, and **Logistic Regression** with rigorous **group-aware 
cross-validation** and **temporal holdout testing**.

## Key Results

| Model | CV AUC | External AUC | Notes |
|-------|--------|--------------|-------|
| XGBoost | **0.910** | 0.826 | Best CV, moderate overfitting |
| RandomForest | 0.887 | **0.830** | Best external robustness |
| LogisticRegression | 0.746 | 0.760 | Weak baseline |

- **Statistical significance**: XGBoost significantly outperforms LogisticRegression 
  (DeLong p < 0.001). RandomForest ˜ XGBoost in external validation.
- **Dataset**: 76K variants (52K pathogenic, 24K benign) from ClinVar GRCh38.
- **Validation**: 5-fold GroupKFold (by gene) + temporal holdout (2023+).

## Project Highlights

- **No data leakage**: Gene-level features computed per-fold from training data only.
- **Temporal generalization**: External test set contains newer variants, simulating 
  real-world deployment on unreviewed submissions.
- **Ablation study**: Structural features (indel length, SNV flag) contribute more 
  signal than review status in the broadened dataset.
- **Reproducible**: Fixed random seeds, pinned dependencies, experiment logging.
- **Deployable**: Serialized model bundle includes scaler and feature schema.
- **Tested**: Unit tests with synthetic data, CI via GitHub Actions.

## Quick Start

```bash
# 1. Clone and enter
git clone https://github.com/<your-username>/clinvar-ml.git
cd clinvar-ml

# 2. Create environment
conda env create -f environment.yml
conda activate clinvar-ml

# 3. Download data (~500MB)
make data

# 4. Run pipeline
make run

# 5. Run tests
pytest tests/ -v
```

## Project Structure

```
clinvar-ml/
+-- main.py                    # Entry point
+-- src/
¦   +-- config.py              # Centralized hyperparameters and paths
¦   +-- data_loader.py         # ClinVar parsing + feature engineering
¦   +-- model.py               # GroupKFold CV, baselines, SHAP
¦   +-- evaluate.py            # DeLong tests, ablation, baselines
¦   +-- inference.py           # Model serialization + prediction
¦   +-- experiment_logger.py   # Reproducibility logging
+-- scripts/
¦   +-- download_clinvar.py    # Data downloader
+-- tests/
¦   +-- conftest.py            # Test fixtures
¦   +-- test_data_loader.py    # Unit tests
+-- results/                   # Generated plots and metrics
+-- models/                    # Serialized best models
+-- .github/workflows/ci.yml   # Continuous integration
+-- MODEL_CARD.md              # Model documentation and limitations
+-- environment.yml
+-- requirements.txt
+-- Makefile
```

## Pipeline

1. **Load** — ClinVar GRCh38, binary pathogenic/benign labels.
2. **Engineer** — review score, structural features (indel length, SNV), 
   gene frequency, temporal features.
3. **Split** — Temporal holdout (pre-2023 train, 2023+ test).
4. **Validate** — 5-fold GroupKFold by GeneSymbol.
5. **Compare** — LogisticRegression, RandomForest, XGBoost + majority baseline.
6. **Test** — Pairwise DeLong significance tests.
7. **Ablate** — Feature-set contribution analysis.
8. **Explain** — SHAP summary plot for XGBoost.
9. **Serialize** — Save model bundle with scaler for deployment.
10. **Log** — Save experiment config and metrics to JSON.

## Data Source

[ClinVar variant_summary.txt.gz](https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz) 
from NCBI. Not included in repo due to size.

## Reproducibility

- All random seeds fixed (`random_state=42`).
- Dependencies pinned in `environment.yml`.
- Temporal split ensures external validity.
- Experiment log (`results/experiment_log.json`) captures config, metrics, and git commit.

## Model Card

See [MODEL_CARD.md](MODEL_CARD.md) for intended use, performance, limitations, and ethical considerations.
