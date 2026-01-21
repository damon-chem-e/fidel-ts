"""
Embedding metadata management.

Metadata tracks:
- Creation date/time
- Tokenizer name
- Embedding model name
- Aggregation method (cls/average/none)
- Sequence length (if aggregation='none')
- Embedding dimension
- Other configuration parameters
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any


class EmbeddingMetadata:
    """
    Metadata for cached embeddings.
    
    Stores all information needed to identify and validate embedding cache.
    """
    
    def __init__(self,
                 tokenizer_name: str,
                 model_name: str,
                 aggregation_method: str,
                 embedding_dim: int,
                 sequence_length: Optional[int] = None,
                 max_length: int = 512,
                 embedding_version: str = '2.0',
                 created_at: Optional[str] = None,
                 **extra_config):
        """
        Initialize embedding metadata.

        Args:
            tokenizer_name: Name of tokenizer (e.g., 'bert-base-uncased')
            model_name: Name of embedding model (e.g., 'bert-base-uncased')
            aggregation_method: 'cls', 'average', or 'none'
            embedding_dim: Dimension of embeddings
            sequence_length: Max sequence length (if aggregation='none', this is the padded length)
            max_length: Maximum tokenization length
            embedding_version: Version of embedding generation logic (default '2.0' for per-variable embeddings)
                               Version history:
                               - '1.0': Original implementation with concatenated channel descriptions (buggy)
                               - '2.0': Fixed per-variable embeddings with parquet column order alignment
            created_at: ISO format timestamp (auto-generated if None)
            **extra_config: Additional configuration parameters
        """
        self.tokenizer_name = tokenizer_name
        self.model_name = model_name
        self.aggregation_method = aggregation_method
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.max_length = max_length
        self.embedding_version = embedding_version
        self.created_at = created_at or datetime.now().isoformat()
        self.extra_config = extra_config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        result = {
            'tokenizer_name': self.tokenizer_name,
            'model_name': self.model_name,
            'aggregation_method': self.aggregation_method,
            'embedding_dim': self.embedding_dim,
            'sequence_length': self.sequence_length,
            'max_length': self.max_length,
            'embedding_version': self.embedding_version,
            'created_at': self.created_at,
        }
        result.update(self.extra_config)
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmbeddingMetadata':
        """Create metadata from dictionary."""
        # Extract known fields
        known_fields = {
            'tokenizer_name', 'model_name', 'aggregation_method',
            'embedding_dim', 'sequence_length', 'max_length', 'embedding_version', 'created_at'
        }
        kwargs = {k: data.pop(k) for k in list(data.keys()) if k in known_fields}
        # Default to version 1.0 for old caches without version field
        if 'embedding_version' not in kwargs:
            kwargs['embedding_version'] = '1.0'
        # Remaining fields go to extra_config
        kwargs['extra_config'] = data
        return cls(**kwargs)
    
    def to_json(self) -> str:
        """Serialize metadata to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'EmbeddingMetadata':
        """Deserialize metadata from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def save(self, file_path: Path):
        """Save metadata to file."""
        with open(file_path, 'w') as f:
            f.write(self.to_json())
    
    @classmethod
    def load(cls, file_path: Path) -> 'EmbeddingMetadata':
        """Load metadata from file."""
        with open(file_path, 'r') as f:
            return cls.from_json(f.read())
    
    def compute_hash(self) -> str:
        """
        Compute hash identifier for this metadata configuration.

        Uses a subset of metadata fields that determine the embedding characteristics.
        Returns first 16 characters of SHA256 hash for use in folder names.

        IMPORTANT: embedding_version is included in the hash to ensure that changes
        to embedding generation logic produce different cache directories. This prevents
        accidentally using old (potentially buggy) embeddings with new code.
        """
        # Fields that affect embedding computation
        hash_fields = {
            'tokenizer_name': self.tokenizer_name,
            'model_name': self.model_name,
            'aggregation_method': self.aggregation_method,
            'embedding_dim': self.embedding_dim,
            'max_length': self.max_length,
            'embedding_version': self.embedding_version,  # CRITICAL: Invalidate cache on logic changes
        }

        # Include sequence_length only if aggregation='none'
        if self.aggregation_method == 'none':
            hash_fields['sequence_length'] = self.sequence_length

        # Sort for consistent hashing
        hash_str = json.dumps(hash_fields, sort_keys=True)

        # Compute hash
        hash_obj = hashlib.sha256(hash_str.encode())
        return hash_obj.hexdigest()[:16]  # Use first 16 chars for folder name
    
    def matches(self, other: 'EmbeddingMetadata') -> bool:
        """
        Check if this metadata matches another (for cache lookup).
        
        Compares fields that affect embedding computation.
        
        Args:
            other: Another EmbeddingMetadata instance
        
        Returns:
            True if metadata matches (embeddings are compatible)
        """
        return (
            self.tokenizer_name == other.tokenizer_name and
            self.model_name == other.model_name and
            self.aggregation_method == other.aggregation_method and
            self.embedding_dim == other.embedding_dim and
            self.max_length == other.max_length and
            (self.aggregation_method != 'none' or self.sequence_length == other.sequence_length)
        )

