# Model Card: ClinVar Variant Pathogenicity Predictor

## Model Details

- **Developer**: <Your Name>
- **Date**: 2026-08-15
- **Version**: 1.0.0
- **Type**: Supervised classification (binary)
- **Framework**: scikit-learn / XGBoost

## Intended Use

This model predicts whether a human genetic variant is **Pathogenic** or **Benign**
based on ClinVar metadata. It is intended for research and educational purposes only.

**Primary users**: Computational biology researchers, bioinformaticians.

**Out-of-scope**: Clinical decision-making, diagnostic use, or patient-facing applications.

## Training Data

- **Source**: NCBI ClinVar ariant_summary.txt.gz`n- **Assembly**: GRCh38
- **Size**: 76,096 variants (52,212 pathogenic, 23,884 benign)
- **Temporal split**: Pre-2023 variants for training, 2023+ for external holdout

## Features

| Feature | Description |
|---------|-------------|
| review_score | Expert panel / guideline / submitter review status |
| is_snv | Single nucleotide variant flag |
| ref_len / alt_len | Reference / alternate allele lengths |
| length_change | Indel length change (clipped to [-50, 50]) |
| is_insertion / is_deletion | Structural variant flags |
| chrom_num / position / pos_bin | Genomic location |
| gc_content | GC content of reference or alternate allele |
| is_transition | Transition vs transversion flag |
| eval_year / eval_month | Temporal evaluation features |
| gene_freq | Log-transformed variant count per gene |
| gene_path_ratio | Historical pathogenic ratio per gene |

## Performance

| Model | CV AUC | External AUC |
|-------|--------|--------------|
| XGBoost | 0.910 | 0.826 |
| RandomForest | 0.887 | 0.830 |
| LogisticRegression | 0.746 | 0.760 |

**Statistical testing**: Pairwise DeLong tests show XGBoost > LogisticRegression (p < 0.001).
RandomForest and XGBoost are not significantly different in external validation.

## Limitations

- **Gene-level features** (gene_freq, gene_path_ratio) do not generalize to unseen genes
  (AUC = 0.500 under GroupKFold).
- **Temporal drift**: External AUC drops 0.08–0.10 relative to CV, indicating distribution shift
  in newer ClinVar submissions.
- **Class imbalance**: Pathogenic variants are 2.7× more frequent than benign in training data.
- **Feature scope**: No protein domain, conservation, or population frequency annotations.

## Ethical Considerations

- This model is for **research use only** and should not be used for clinical diagnosis.
- Predictions may reflect curation biases in ClinVar (e.g., over-representation of well-studied genes).
- Temporal holdout reduces but does not eliminate overfitting to historical submission patterns.
