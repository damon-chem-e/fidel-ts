"""
Column name normalization utilities for dataset compatibility.

Centralizes dataset-specific normalization logic to keep loaders consistent
and avoid duplication across data access layers.
"""


def is_jena_dataset_path(root_path: str) -> bool:
    """
    Check whether a dataset root path corresponds to Jena Atmospheric Physics.

    Args:
        root_path: Dataset root path as string

    Returns:
        True if the path indicates Jena dataset, otherwise False
    """
    return 'Jena_Atmospheric_Physics' in str(root_path)


def normalize_jena_column_name(col_name: str) -> str:
    """
    Normalize Jena column names by removing unit suffixes.

    For Jena only, this drops everything starting at the first '(',
    which avoids Unicode/unit encoding mismatches while preserving the
    base variable name (e.g., "PAR (μmol/m²/s)" -> "PAR").

    Args:
        col_name: Raw column name from parquet or static embeddings

    Returns:
        Normalized column name without unit suffixes
    """
    return col_name.split('(', 1)[0].strip()
