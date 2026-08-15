"""Fixtures and helpers for testing."""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def make_synthetic_clinvar(n_rows: int = 200) -> Path:
    """Create a minimal synthetic ClinVar TSV for testing.
    
    Returns:
        Path to the temporary TSV file.
    """
    np.random.seed(42)
    
    years = np.random.choice([2021, 2022, 2023, 2024], n_rows, p=[0.3, 0.3, 0.2, 0.2])
    
    data = {
        "VariationID": range(n_rows),
        "Type": ["single nucleotide variant"] * n_rows,
        "Name": [f"var{i}" for i in range(n_rows)],
        "GeneSymbol": [f"GENE{i % 20}" for i in range(n_rows)],
        "ClinicalSignificance": np.random.choice(
            ["Pathogenic", "Likely pathogenic", "Benign", "Likely benign"],
            n_rows,
            p=[0.3, 0.2, 0.3, 0.2],
        ),
        "ReviewStatus": np.random.choice(
            [
                "reviewed by expert panel",
                "practice guideline",
                "criteria provided, multiple submitters, no conflicts",
                "criteria provided, single submitter",
                "criteria provided, conflicting interpretations",
            ],
            n_rows,
        ),
        "LastEvaluated": [f"{y}-{i % 12 + 1:02d}-15" for i, y in enumerate(years)],
        "RS# (dbSNP)": [f"rs{i}" for i in range(n_rows)],
        "Chromosome": [str(i % 24 + 1) for i in range(n_rows)],
        "Start": np.random.randint(1000, 1000000, n_rows),
        "PositionVCF": np.random.randint(1000, 1000000, n_rows),
        "ReferenceAlleleVCF": np.random.choice(["A", "C", "G", "T"], n_rows),
        "AlternateAlleleVCF": np.random.choice(["A", "C", "G", "T"], n_rows),
        "Assembly": ["GRCh38"] * n_rows,
    }
    
    df = pd.DataFrame(data)
    
    # Create temp file
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    df.to_csv(tmp.name, sep="\t", index=False)
    tmp.close()
    return Path(tmp.name)
