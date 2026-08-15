"""
Data loader module.

Loads ClinVar variant_summary.txt, applies quality filters (GRCh38),
encodes labels, and derives structural and temporal features for modeling.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def load_clinvar_dataset(variant_summary_path: str, max_rows: int = None) -> pd.DataFrame:
    """Load and clean ClinVar variant_summary.txt into a modeling-ready DataFrame.
    
    Args:
        variant_summary_path: Path to the ClinVar TSV or TSV.GZ file.
        max_rows: Optional row limit for faster iteration during development.
    
    Returns:
        DataFrame with engineered features and binary labels (1=pathogenic, 0=benign).
    
    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If no variants remain after filtering.
    """
    path = Path(variant_summary_path)
    if not path.exists():
        raise FileNotFoundError(f"ClinVar file not found: {variant_summary_path}")

    print(f"[+] Loading ClinVar from {path} ...", flush=True)
    compression = "gzip" if path.suffixes == [".txt", ".gz"] or path.suffix == ".gz" else None
    cols = [
        "VariationID",
        "Type",
        "Name",
        "GeneSymbol",
        "ClinicalSignificance",
        "ReviewStatus",
        "LastEvaluated",
        "RS# (dbSNP)",
        "Chromosome",
        "Start",
        "PositionVCF",
        "ReferenceAlleleVCF",
        "AlternateAlleleVCF",
        "Assembly",
    ]
    df = pd.read_csv(path, sep="\t", usecols=cols, low_memory=False, compression=compression)
    if max_rows:
        df = df.head(max_rows).copy()

    df = df.rename(columns={
        "GeneSymbol": "GeneSymbol",
        "ClinicalSignificance": "clinical_significance",
        "ReviewStatus": "review_status",
        "LastEvaluated": "last_evaluated",
        "Start": "position",
        "Chromosome": "chromosome",
        "ReferenceAlleleVCF": "ref",
        "AlternateAlleleVCF": "alt",
        "Type": "variant_type",
    })

    df = df[df["Assembly"] == "GRCh38"].copy()

    df["chromosome"] = df["chromosome"].replace({"X": "23", "Y": "24", "MT": "25"})
    df["chromosome"] = pd.to_numeric(df["chromosome"], errors="coerce")
    df = df.dropna(subset=["chromosome"]).copy()

    df["position"] = pd.to_numeric(df["position"], errors="coerce")
    df = df.dropna(subset=["position"]).copy()

    df["chrom_num"] = df["chromosome"].astype("Int64")
    df["position"] = df["position"].astype("Int64")

    sig = df["clinical_significance"].astype(str).str.lower()
    df["is_pathogenic"] = sig.str.contains("pathogenic|likely-pathogenic").astype(int)
    df["is_benign"] = sig.str.contains("benign|likely-benign").astype(int)

    df["label"] = np.where(
        df["is_pathogenic"] & ~df["is_benign"], 1,
        np.where(df["is_benign"] & ~df["is_pathogenic"], 0, np.nan)
    )
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)

    df["is_snv"] = (df["ref"].str.len() == 1) & (df["alt"].str.len() == 1)
    df["is_snv"] = df["is_snv"].astype(int)

    df["ref_len"] = df["ref"].str.len()
    df["alt_len"] = df["alt"].str.len()
    df["length_change"] = (df["alt_len"] - df["ref_len"]).clip(-50, 50)
    df["is_insertion"] = (df["length_change"] > 0).astype(int)
    df["is_deletion"] = (df["length_change"] < 0).astype(int)

    review_map = {
        "reviewed by expert panel": 4,
        "practice guideline": 3,
        "criteria provided, multiple submitters, no conflicts": 2,
        "criteria provided, single submitter": 1,
        "criteria provided, conflicting interpretations": 0,
    }
    df["review_score"] = df["review_status"].map(review_map).fillna(0).astype(int)

    df["last_evaluated_dt"] = pd.to_datetime(df["last_evaluated"], errors="coerce")
    df["eval_year"] = df["last_evaluated_dt"].dt.year
    df["eval_month"] = df["last_evaluated_dt"].dt.month

    df["chrom_num"] = df["chrom_num"].astype(int)
    df["position"] = df["position"].astype(int)
    df["pos_bin"] = (df["position"] // 1_000_000).astype(int)
    df["gc_content"] = ((df["ref"] == "G") | (df["ref"] == "C") | (df["alt"] == "G") | (df["alt"] == "C")).astype(int)
    df["is_transition"] = ((df["ref"].isin(["A", "G"]) & df["alt"].isin(["A", "G"])) | (df["ref"].isin(["C", "T"]) & df["alt"].isin(["C", "T"]))).astype(int)

    if len(df) == 0:
        raise ValueError("No variants left after filtering. Check ClinVar file and filters.")

    print(f"[+] Cleaned dataset: {len(df):,} variants ({df['label'].sum():,} pathogenic, {(df['label']==0).sum():,} benign)", flush=True)
    return df.reset_index(drop=True)


def temporal_train_test_split(df: pd.DataFrame, test_year: int = 2023) -> tuple:
    """Split data by LastEvaluated year for external validation.
    
    Args:
        df: Full dataset with eval_year column.
        test_year: Variants evaluated in this year or later form the test set.
    
    Returns:
        Tuple of (train_df, test_df).
    """
    train = df[df["eval_year"] < test_year].copy()
    test = df[df["eval_year"] >= test_year].copy()
    return train, test
