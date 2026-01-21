"""
Column name normalization utilities for dataset compatibility.

Centralizes dataset-specific normalization logic to keep loaders consistent
and avoid duplication across data access layers.
"""

from typing import Dict, Iterable, Set, Tuple


def normalize_jena_column_name(col_name: str) -> str:
    """
    Normalize corrupted Unicode variants in Jena column names to canonical form.

    This converts common corruption patterns so parquet column names match
    static embedding keys.

    Args:
        col_name: Raw column name from parquet or static embeddings

    Returns:
        Normalized column name in canonical Unicode form
    """
    # Start with the original column name.
    normalized = col_name
    # Normalize micro sign variants to the Greek mu.
    normalized = normalized.replace('µ', 'μ')
    # Fix corrupted superscript 3 representations.
    normalized = normalized.replace('m**3', 'm³')
    normalized = normalized.replace('g/m_', 'g/m³')
    # Fix corrupted or missing superscript 2 representations.
    normalized = normalized.replace('W/m_', 'W/m²')
    # Fix corrupted or missing mu and superscript 2 in PAR units.
    normalized = normalized.replace('_mol/m_/s', 'μmol/m²/s')
    normalized = normalized.replace('mol/m_/s', 'μmol/m²/s')
    normalized = normalized.replace('mol/m/s', 'μmol/m²/s')
    return normalized


def build_normalized_embedding_map(
    embedding_keys: Iterable[str],
) -> Tuple[Dict[str, str], Dict[str, Set[str]]]:
    """
    Build a normalized-to-original key map for embedding variables.

    This allows matching parquet column names to embedding keys even when
    the source strings contain corrupted Unicode characters.

    Args:
        embedding_keys: Iterable of embedding variable names

    Returns:
        Tuple of:
        - normalized_map: Dict mapping normalized name -> original embedding key
        - duplicates: Dict of normalized name -> set of original keys (collisions)
    """
    # Track the best mapping from normalized name to original key.
    normalized_map: Dict[str, str] = {}
    # Track collisions where multiple keys normalize to the same string.
    duplicates: Dict[str, Set[str]] = {}
    for key in embedding_keys:
        normalized_key = normalize_jena_column_name(key)
        if normalized_key in normalized_map and normalized_map[normalized_key] != key:
            duplicates.setdefault(normalized_key, set()).update(
                {normalized_map[normalized_key], key}
            )
        normalized_map[normalized_key] = key
    return normalized_map, duplicates
