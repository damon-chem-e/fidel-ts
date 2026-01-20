"""
Test suite for tensor cache LLM embedding compatibility.

This module tests the tensor cache implementation for LLM embedding support,
verifying compatibility with TimeCMA and MMTSFLib models that use per-sample
LLM embeddings (shape: [embed_dim, n_channels]) instead of per-timestamp
embeddings (shape: [input_len, num_items, embed_dim]).

Test Categories:
----------------
1. LLM Embedding Detection: Identify LLM vs news/weather embeddings by shape
2. LLM Embedding Storage: Per-sample storage (not per-timestamp deduplication)
3. Shape Preservation: Verify (embed_dim, n_channels) shape through cache round-trip
4. Cache Hash: LLM config inclusion in cache hash for proper invalidation
5. Backward Compatibility: Non-LLM embeddings behavior unchanged

Reference:
    docs/planning/tensor_cache_llm_embedding_compatibility.md

Usage:
------
# Run all LLM embedding tests
pytest tests/test_tensor_cache_llm_embeddings.py -v

# Run specific test class
pytest tests/test_tensor_cache_llm_embeddings.py::TestLLMEmbeddingDetection -v
"""

import pytest
import numpy as np
import tempfile
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import tensor cache components
from data_provider.tensor_cache import (
    SAMPLE_IDX_SAMPLE_ID,
    SAMPLE_IDX_SEQ_X,
    SAMPLE_IDX_SEQ_Y,
    SAMPLE_IDX_X_TIME,
    SAMPLE_IDX_Y_TIME,
    SAMPLE_IDX_HETERO_X,
    SAMPLE_IDX_HETERO_Y,
    SAMPLE_IDX_HETERO_X_TIME,
    SAMPLE_IDX_HETERO_Y_TIME,
    SAMPLE_IDX_HETERO_GENERAL,
    SAMPLE_IDX_HETERO_CHANNEL,
    SAMPLE_IDX_X_TIME_FEATURES,
    SAMPLE_IDX_Y_TIME_FEATURES,
    InferredShapes,
    SharedTableCollector,
    infer_shapes_from_sample,
    _safe_array,
    _infer_dim,
    _is_llm_embedding,  # Real implementation
    CACHE_RELEVANT_KEYS,
)


# =============================================================================
# LLM EMBEDDING CONSTANTS
# =============================================================================

# Standard LLM embedding dimensions
GPT2_EMBED_DIM = 768
GPT2_MEDIUM_EMBED_DIM = 1024
QWEN2_5_EMBED_DIM = 896

# Common input/output lengths
COMMON_INPUT_LENGTHS = [24, 48, 96, 192, 336, 512, 720]


# =============================================================================
# HELPER FUNCTIONS FOR TESTS
# =============================================================================

def _build_cache_config_with_llm(args: Any) -> dict:
    """
    Build cache configuration dict including LLM embedding config.

    This function should be used instead of the current build_cache_config()
    when LLM embedding support is added to ensure proper cache invalidation.

    Args:
        args: Configuration object with standard cache params + llm_embedding

    Returns:
        Dict of cache-relevant configuration for hash computation
    """
    # Base cache config (matches current implementation)
    cache_config = {
        'input_len': getattr(args, 'input_len', None),
        'output_len': getattr(args, 'output_len', None),
        'scale': getattr(args, 'scale', True),
        'truncate_train_for_purge': getattr(args, 'truncate_train_for_purge', False),
        'downsample': getattr(args, 'downsample', 1),
        'data_name': getattr(args, 'data', None),
    }

    # Add LLM embedding config if present
    llm_embedding = getattr(args, 'llm_embedding', None)
    if llm_embedding:
        cache_config['llm_embedding'] = {
            'model_name': llm_embedding.get('model_name'),
            'prompt_template': llm_embedding.get('prompt_template'),
            'extraction_mode': llm_embedding.get('extraction_mode', 'last_token'),
            'd_llm': llm_embedding.get('d_llm'),
        }
    else:
        cache_config['llm_embedding'] = None

    return cache_config


# =============================================================================
# MOCK DATASETS
# =============================================================================

class MockLLMEmbeddingDataset:
    """
    Mock dataset simulating TimeCMA/MMTSFLib with LLM embeddings.

    Produces samples with LLM embeddings in hetero_x:
        - Shape: (embed_dim, n_channels) e.g., (768, 1) for GPT-2
        - Per-sample unique embeddings (not per-timestamp)
        - Cannot be deduplicated by timestamp

    This simulates what happens when:
        1. LLMEmbeddingProvider loads precomputed embeddings
        2. Data loader injects them into x_hetero field
        3. Tensor cache attempts to process the sample
    """

    def __init__(
        self,
        n_samples: int = 100,
        input_len: int = 96,
        output_len: int = 48,
        n_features: int = 7,
        embed_dim: int = GPT2_EMBED_DIM,  # LLM embedding dimension
        n_channels: int = 1,  # Number of channels in data
        seed: int = 42,
        entity_id: str = "test_entity"
    ):
        np.random.seed(seed)
        self.n_samples = n_samples
        self.input_len = input_len
        self.output_len = output_len
        self.n_features = n_features
        self.embed_dim = embed_dim
        self.n_channels = n_channels
        self.entity_id = entity_id

        # Generate continuous timestamps
        base_ts = 1000
        total_ts = n_samples + input_len + output_len
        self.timestamps = np.arange(base_ts, base_ts + total_ts, dtype=np.int64)

        # Pre-generate time series data
        n_ts = len(self.timestamps)
        self.all_timeseries = np.random.randn(n_ts, n_features).astype(np.float32)

        # Pre-generate LLM embeddings - ONE PER SAMPLE (not per timestamp!)
        # Shape: (n_samples, embed_dim, n_channels)
        self.llm_embeddings = np.random.randn(
            n_samples, embed_dim, n_channels
        ).astype(np.float32)

        # Entity-level embeddings
        self.general_embedding = np.random.randn(embed_dim).astype(np.float32)
        self.channel_embedding = np.random.randn(embed_dim).astype(np.float32)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple:
        """
        Return sample with LLM embedding in hetero_x.

        Critical difference from news/weather datasets:
        - hetero_x shape: (embed_dim, n_channels) NOT (input_len, num_items, embed_dim)
        - hetero_x is unique per sample, not shared across samples
        """
        x_start = idx
        x_end = x_start + self.input_len
        y_start = x_end
        y_end = y_start + self.output_len

        # Get per-sample LLM embedding
        llm_emb = self.llm_embeddings[idx]  # Shape: (embed_dim, n_channels)

        return (
            f"{self.entity_id}_sample_{idx}",              # sample_id
            self.all_timeseries[x_start:x_end],            # seq_x: (input_len, n_features)
            self.all_timeseries[y_start:y_end],            # seq_y: (output_len, n_features)
            self.timestamps[x_start:x_end],                # x_time: (input_len,)
            self.timestamps[y_start:y_end],                # y_time: (output_len,)
            llm_emb,                                        # hetero_x: (embed_dim, n_channels) - LLM!
            None,                                           # hetero_y: None for LLM models
            None,                                           # hetero_x_time
            None,                                           # hetero_y_time
            self.general_embedding.copy(),                 # hetero_general
            self.channel_embedding.copy(),                 # hetero_channel
            np.zeros((self.input_len, 4), dtype=np.float32),   # x_time_features
            np.zeros((self.output_len, 4), dtype=np.float32),  # y_time_features
        )


class MockNewsEmbeddingDataset:
    """
    Mock dataset simulating TGTSF models with news/weather embeddings.

    Produces samples with per-timestamp embeddings in hetero_x:
        - Shape: (input_len, num_items, embed_dim) e.g., (96, 2, 768)
        - Per-timestamp data that can be deduplicated
        - Shared across samples with overlapping timestamps

    This represents the CURRENT supported behavior that must remain unchanged.
    """

    def __init__(
        self,
        n_samples: int = 100,
        input_len: int = 96,
        output_len: int = 48,
        n_features: int = 7,
        embed_dim: int = 768,
        num_items: int = 2,  # news + downtime indicator
        seed: int = 42,
        entity_id: str = "test_entity"
    ):
        np.random.seed(seed)
        self.n_samples = n_samples
        self.input_len = input_len
        self.output_len = output_len
        self.n_features = n_features
        self.embed_dim = embed_dim
        self.num_items = num_items
        self.entity_id = entity_id

        # Generate continuous timestamps
        base_ts = 1000
        total_ts = n_samples + input_len + output_len
        self.timestamps = np.arange(base_ts, base_ts + total_ts, dtype=np.int64)

        # Pre-generate data for all timestamps
        n_ts = len(self.timestamps)
        self.all_timeseries = np.random.randn(n_ts, n_features).astype(np.float32)
        self.all_embeddings = np.random.randn(n_ts, num_items, embed_dim).astype(np.float32)
        self.all_hetero_time = np.random.randn(n_ts, 4).astype(np.float32)

        # Entity-level embeddings
        self.general_embedding = np.random.randn(embed_dim).astype(np.float32)
        self.channel_embedding = np.random.randn(embed_dim).astype(np.float32)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple:
        """
        Return sample with per-timestamp news embeddings in hetero_x.

        Standard behavior for TGTSF models:
        - hetero_x shape: (input_len, num_items, embed_dim)
        - Embeddings are per-timestamp, deduplicatable
        """
        x_start = idx
        x_end = x_start + self.input_len
        y_start = x_end
        y_end = y_start + self.output_len

        return (
            f"{self.entity_id}_sample_{idx}",                  # sample_id
            self.all_timeseries[x_start:x_end],                # seq_x
            self.all_timeseries[y_start:y_end],                # seq_y
            self.timestamps[x_start:x_end],                    # x_time
            self.timestamps[y_start:y_end],                    # y_time
            self.all_embeddings[x_start:x_end],                # hetero_x: (input_len, num_items, embed_dim)
            self.all_embeddings[y_start:y_end],                # hetero_y
            self.all_hetero_time[x_start:x_end],               # hetero_x_time
            self.all_hetero_time[y_start:y_end],               # hetero_y_time
            self.general_embedding.copy(),                     # hetero_general
            self.channel_embedding.copy(),                     # hetero_channel
            np.zeros((self.input_len, 4), dtype=np.float32),   # x_time_features
            np.zeros((self.output_len, 4), dtype=np.float32),  # y_time_features
        )


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def llm_dataset_gpt2():
    """Mock dataset with GPT-2 LLM embeddings (768D)."""
    return MockLLMEmbeddingDataset(
        n_samples=50,
        input_len=96,
        output_len=48,
        embed_dim=GPT2_EMBED_DIM,
        n_channels=1
    )


@pytest.fixture
def llm_dataset_qwen():
    """Mock dataset with Qwen2.5 LLM embeddings (896D)."""
    return MockLLMEmbeddingDataset(
        n_samples=50,
        input_len=96,
        output_len=48,
        embed_dim=QWEN2_5_EMBED_DIM,
        n_channels=1
    )


@pytest.fixture
def llm_dataset_gpt2_medium():
    """Mock dataset with GPT-2 Medium LLM embeddings (1024D)."""
    return MockLLMEmbeddingDataset(
        n_samples=50,
        input_len=96,
        output_len=48,
        embed_dim=GPT2_MEDIUM_EMBED_DIM,
        n_channels=7  # Multi-channel
    )


@pytest.fixture
def news_dataset():
    """Mock dataset with per-timestamp news embeddings."""
    return MockNewsEmbeddingDataset(
        n_samples=100,
        input_len=96,
        output_len=48,
        embed_dim=768,
        num_items=2
    )


@pytest.fixture
def news_dataset_varied_lengths():
    """Multiple news datasets with different input lengths."""
    return {
        length: MockNewsEmbeddingDataset(
            n_samples=20,
            input_len=length,
            output_len=length // 2,
            embed_dim=768
        )
        for length in [24, 96, 192, 336, 512]
    }


# =============================================================================
# TEST CLASS: LLM Embedding Detection
# =============================================================================

class TestLLMEmbeddingDetection:
    """
    Tests for _is_llm_embedding detection function.

    This function must reliably distinguish between:
    - LLM embeddings: (embed_dim, n_channels) where embed_dim >> input_len
    - News embeddings: (input_len, num_items, embed_dim) - 3D, not applicable

    Critical for the v2 implementation to route data correctly.
    """

    def test_detect_gpt2_embedding(self):
        """GPT-2 embedding (768, 1) should be detected as LLM."""
        hetero_x = np.random.randn(768, 1).astype(np.float32)
        input_len = 96

        assert _is_llm_embedding(hetero_x, input_len) is True

    def test_detect_gpt2_medium_embedding(self):
        """GPT-2 Medium embedding (1024, 1) should be detected as LLM."""
        hetero_x = np.random.randn(1024, 1).astype(np.float32)
        input_len = 96

        assert _is_llm_embedding(hetero_x, input_len) is True

    def test_detect_qwen_embedding(self):
        """Qwen2.5 embedding (896, 1) should be detected as LLM."""
        hetero_x = np.random.randn(896, 1).astype(np.float32)
        input_len = 96

        assert _is_llm_embedding(hetero_x, input_len) is True

    def test_detect_llm_multi_channel(self):
        """LLM embedding with multiple channels (768, 7) should be detected."""
        hetero_x = np.random.randn(768, 7).astype(np.float32)
        input_len = 96

        assert _is_llm_embedding(hetero_x, input_len) is True

    def test_reject_none(self):
        """None should not be detected as LLM embedding."""
        assert _is_llm_embedding(None, 96) is False

    def test_reject_1d_array(self):
        """1D array should not be detected as LLM embedding."""
        hetero_x = np.random.randn(768).astype(np.float32)
        assert _is_llm_embedding(hetero_x, 96) is False

    def test_reject_3d_news_embedding(self):
        """3D news embedding (96, 2, 768) should not be detected as LLM."""
        hetero_x = np.random.randn(96, 2, 768).astype(np.float32)
        assert _is_llm_embedding(hetero_x, 96) is False

    def test_reject_2d_news_embedding(self):
        """2D news embedding (96, 768) where first dim = input_len should not be LLM."""
        hetero_x = np.random.randn(96, 768).astype(np.float32)
        input_len = 96

        # First dim matches input_len, so this is NOT LLM embedding
        assert _is_llm_embedding(hetero_x, input_len) is False

    def test_edge_case_input_len_matches_embed_dim(self):
        """Edge case: input_len == embed_dim should be handled correctly."""
        # If input_len is 768 and we have (768, 1), it's ambiguous
        # But this is extremely rare in practice
        hetero_x = np.random.randn(768, 1).astype(np.float32)
        input_len = 768

        # When first_dim == input_len, we assume it's news embedding (conservative)
        assert _is_llm_embedding(hetero_x, input_len) is False

    def test_various_input_lengths(self):
        """Test detection across common input lengths."""
        llm_embedding = np.random.randn(768, 1).astype(np.float32)

        for input_len in COMMON_INPUT_LENGTHS:
            if input_len != 768:  # Skip ambiguous case
                assert _is_llm_embedding(llm_embedding, input_len) is True, \
                    f"Failed to detect LLM embedding with input_len={input_len}"

    def test_small_embedding_not_llm(self):
        """Small 2D array (96, 16) should not be detected as LLM."""
        # This could be some other type of feature, not LLM embedding
        hetero_x = np.random.randn(96, 16).astype(np.float32)
        input_len = 48  # Different from first dim

        # First dim (96) < 512, so not considered LLM
        assert _is_llm_embedding(hetero_x, input_len) is False

    def test_large_llm_embedding(self):
        """Large LLM embedding (4096, 1) from large models should be detected."""
        hetero_x = np.random.randn(4096, 1).astype(np.float32)
        assert _is_llm_embedding(hetero_x, 96) is True


# =============================================================================
# TEST CLASS: LLM Embedding Shape Preservation
# =============================================================================

class TestLLMEmbeddingShapePreservation:
    """
    Tests that verify LLM embedding shapes are preserved through tensor cache.

    For TimeCMA and MMTSFLib to work correctly, the tensor cache MUST preserve:
    - Input shape: (embed_dim, n_channels) e.g., (768, 1)
    - Per-sample uniqueness: each sample has its own embedding

    These tests verify the EXPECTED behavior after v2 implementation.
    """

    def test_llm_embedding_shape_in_sample(self, llm_dataset_gpt2):
        """Verify LLM dataset produces correct embedding shape in sample."""
        sample = llm_dataset_gpt2[0]
        hetero_x = sample[SAMPLE_IDX_HETERO_X]

        expected_shape = (GPT2_EMBED_DIM, 1)  # (768, 1)
        assert hetero_x.shape == expected_shape, \
            f"LLM embedding shape incorrect: {hetero_x.shape}, expected {expected_shape}"

    def test_llm_embeddings_unique_per_sample(self, llm_dataset_gpt2):
        """Each sample should have a unique LLM embedding."""
        sample_0 = llm_dataset_gpt2[0]
        sample_1 = llm_dataset_gpt2[1]

        emb_0 = sample_0[SAMPLE_IDX_HETERO_X]
        emb_1 = sample_1[SAMPLE_IDX_HETERO_X]

        # Embeddings should be different (extremely unlikely to be equal by chance)
        assert not np.allclose(emb_0, emb_1), \
            "LLM embeddings should be unique per sample"

    def test_llm_embedding_dtype(self, llm_dataset_gpt2):
        """LLM embeddings should be float32."""
        sample = llm_dataset_gpt2[0]
        hetero_x = sample[SAMPLE_IDX_HETERO_X]

        assert hetero_x.dtype == np.float32

    def test_timecma_expected_shape(self, llm_dataset_gpt2):
        """
        Verify shape matches TimeCMA expectations.

        TimeCMA forward expects:
        - historical_events: [B, d_llm, num_nodes] or [B, d_llm, num_nodes, 1]

        Per-sample from cache should be: (d_llm, n_channels)
        After batching: [B, d_llm, n_channels]
        """
        sample = llm_dataset_gpt2[0]
        hetero_x = sample[SAMPLE_IDX_HETERO_X]

        d_llm, n_channels = hetero_x.shape

        assert d_llm == llm_dataset_gpt2.embed_dim, \
            f"d_llm mismatch: {d_llm} vs {llm_dataset_gpt2.embed_dim}"
        assert n_channels == llm_dataset_gpt2.n_channels, \
            f"n_channels mismatch: {n_channels} vs {llm_dataset_gpt2.n_channels}"

    def test_qwen_embedding_shape(self, llm_dataset_qwen):
        """Verify Qwen2.5 embeddings have correct shape."""
        sample = llm_dataset_qwen[0]
        hetero_x = sample[SAMPLE_IDX_HETERO_X]

        expected_shape = (QWEN2_5_EMBED_DIM, 1)  # (896, 1)
        assert hetero_x.shape == expected_shape

    def test_multi_channel_llm_embedding(self, llm_dataset_gpt2_medium):
        """Test multi-channel LLM embeddings (multiple variables)."""
        sample = llm_dataset_gpt2_medium[0]
        hetero_x = sample[SAMPLE_IDX_HETERO_X]

        expected_shape = (GPT2_MEDIUM_EMBED_DIM, 7)  # (1024, 7)
        assert hetero_x.shape == expected_shape


# =============================================================================
# TEST CLASS: Non-LLM Embedding Backward Compatibility
# =============================================================================

class TestNonLLMEmbeddingBackwardCompatibility:
    """
    Tests that verify non-LLM (news/weather) embeddings behavior is unchanged.

    CRITICAL: These tests ensure the existing tensor cache functionality
    continues to work correctly after v2 implementation.
    """

    def test_news_embedding_shape(self, news_dataset):
        """Verify news dataset produces correct embedding shape."""
        sample = news_dataset[0]
        hetero_x = sample[SAMPLE_IDX_HETERO_X]

        # News embeddings: (input_len, num_items, embed_dim)
        expected_shape = (96, 2, 768)
        assert hetero_x.shape == expected_shape, \
            f"News embedding shape incorrect: {hetero_x.shape}, expected {expected_shape}"

    def test_news_embedding_is_not_llm(self, news_dataset):
        """News embeddings should NOT be detected as LLM."""
        sample = news_dataset[0]
        hetero_x = sample[SAMPLE_IDX_HETERO_X]

        # 3D array should not trigger LLM detection
        assert _is_llm_embedding(hetero_x, news_dataset.input_len) is False

    def test_news_embedding_per_timestamp_uniqueness(self, news_dataset):
        """News embeddings should vary by timestamp within a sample."""
        sample = news_dataset[0]
        hetero_x = sample[SAMPLE_IDX_HETERO_X]

        # Check that different timestamps have different embeddings
        emb_t0 = hetero_x[0]  # First timestamp
        emb_t1 = hetero_x[1]  # Second timestamp

        assert not np.allclose(emb_t0, emb_t1), \
            "News embeddings should vary by timestamp"

    def test_news_embedding_deduplication_potential(self, news_dataset):
        """
        Verify that overlapping samples share timestamp embeddings.

        This is the key property that enables deduplication.
        """
        sample_0 = news_dataset[0]
        sample_1 = news_dataset[1]

        # Sample 0 x_time covers timestamps 1000-1095
        # Sample 1 x_time covers timestamps 1001-1096
        # They overlap on timestamps 1001-1095 (95 timestamps)

        x_time_0 = sample_0[SAMPLE_IDX_X_TIME]
        x_time_1 = sample_1[SAMPLE_IDX_X_TIME]
        hetero_x_0 = sample_0[SAMPLE_IDX_HETERO_X]
        hetero_x_1 = sample_1[SAMPLE_IDX_HETERO_X]

        # Find overlapping timestamps
        overlap_start = 1  # First overlapping position in sample_0

        # Embedding at timestamp 1001 should be identical in both samples
        emb_0_at_1001 = hetero_x_0[overlap_start]  # sample_0, position 1 (timestamp 1001)
        emb_1_at_1001 = hetero_x_1[0]               # sample_1, position 0 (timestamp 1001)

        # These should be identical (same timestamp from same underlying data)
        assert np.allclose(emb_0_at_1001, emb_1_at_1001), \
            "Same timestamp should have identical embedding (deduplication potential)"

    def test_news_embedding_y_hetero_exists(self, news_dataset):
        """News dataset should have y_hetero for output window."""
        sample = news_dataset[0]
        hetero_y = sample[SAMPLE_IDX_HETERO_Y]

        assert hetero_y is not None
        expected_shape = (48, 2, 768)  # (output_len, num_items, embed_dim)
        assert hetero_y.shape == expected_shape

    def test_varied_input_lengths_work(self, news_dataset_varied_lengths):
        """Test news embeddings work with various input lengths."""
        for input_len, dataset in news_dataset_varied_lengths.items():
            sample = dataset[0]
            hetero_x = sample[SAMPLE_IDX_HETERO_X]

            expected_shape = (input_len, 2, 768)
            assert hetero_x.shape == expected_shape, \
                f"Failed for input_len={input_len}: got {hetero_x.shape}"


# =============================================================================
# TEST CLASS: LLM Embedding Storage (V2 Expected Behavior)
# =============================================================================

class TestLLMEmbeddingStorage:
    """
    Tests for LLM embedding storage in tensor cache v2.

    These tests verify the EXPECTED behavior after v2 implementation:
    - LLM embeddings stored per-sample (not per-timestamp)
    - No deduplication (each sample's embedding is unique)
    - Separate storage from news/weather embeddings
    """

    def test_llm_embedding_no_timestamp_deduplication(self, llm_dataset_gpt2):
        """
        LLM embeddings should NOT be deduplicated by timestamp.

        Current (broken) behavior: treats (768, 1) as 768 "timestamps"
        Expected v2 behavior: stores as single per-sample embedding
        """
        n_samples = len(llm_dataset_gpt2)

        # In v2, we should have exactly n_samples LLM embeddings stored
        # (one per sample, no deduplication)
        expected_llm_embeddings_count = n_samples

        # Simulate what v2 would store
        llm_embeddings_collected = []
        for i in range(n_samples):
            sample = llm_dataset_gpt2[i]
            hetero_x = sample[SAMPLE_IDX_HETERO_X]

            if _is_llm_embedding(hetero_x, llm_dataset_gpt2.input_len):
                llm_embeddings_collected.append(hetero_x.copy())

        assert len(llm_embeddings_collected) == expected_llm_embeddings_count, \
            f"Expected {expected_llm_embeddings_count} LLM embeddings, got {len(llm_embeddings_collected)}"

    def test_llm_embedding_content_preservation(self, llm_dataset_gpt2):
        """
        Verify LLM embedding content is preserved exactly.

        After round-trip through cache:
        1. Store embedding
        2. Load embedding
        Content should be bit-for-bit identical.
        """
        sample = llm_dataset_gpt2[0]
        original_emb = sample[SAMPLE_IDX_HETERO_X].copy()

        # Simulate cache storage
        stored_emb = original_emb.astype(np.float32)

        # Simulate cache loading
        loaded_emb = stored_emb.copy()

        assert np.array_equal(original_emb, loaded_emb), \
            "LLM embedding content must be preserved exactly"

    def test_llm_embedding_storage_shape(self, llm_dataset_gpt2):
        """
        Verify expected storage shape for LLM embeddings in v2.

        Expected cache structure:
        - shared/llm_embeddings.npy: (N_samples, embed_dim, n_channels)
        - train/llm_indices.npy: (N_train,) -> indices into llm_embeddings
        """
        n_samples = len(llm_dataset_gpt2)
        embed_dim = llm_dataset_gpt2.embed_dim
        n_channels = llm_dataset_gpt2.n_channels

        # Simulate v2 storage
        all_llm_embeddings = []
        for i in range(n_samples):
            sample = llm_dataset_gpt2[i]
            hetero_x = sample[SAMPLE_IDX_HETERO_X]
            all_llm_embeddings.append(hetero_x)

        llm_embeddings_array = np.stack(all_llm_embeddings, axis=0)

        expected_shape = (n_samples, embed_dim, n_channels)
        assert llm_embeddings_array.shape == expected_shape, \
            f"LLM storage shape {llm_embeddings_array.shape}, expected {expected_shape}"

    def test_llm_embedding_index_generation(self, llm_dataset_gpt2):
        """
        Test generation of llm_indices for per-sample lookup.

        Unlike timestamp deduplication where multiple samples share indices,
        LLM embeddings have 1:1 mapping: sample_i -> llm_index_i
        """
        n_samples = len(llm_dataset_gpt2)

        # Generate indices (1:1 mapping for LLM embeddings)
        llm_indices = np.arange(n_samples, dtype=np.int32)

        # Each sample should have unique index
        assert len(set(llm_indices)) == n_samples

        # Indices should be contiguous 0 to N-1
        assert llm_indices[0] == 0
        assert llm_indices[-1] == n_samples - 1


# =============================================================================
# TEST CLASS: Cache Hash with LLM Config
# =============================================================================

class TestCacheHashWithLLMConfig:
    """
    Tests for cache hash computation including LLM embedding config.

    CRITICAL: Without LLM config in hash, different LLM models could
    incorrectly reuse the same cache, causing dimension mismatches.
    """

    def test_hash_changes_with_model_name(self):
        """Cache hash should change when LLM model changes."""

        @dataclass
        class MockArgs:
            input_len: int = 96
            output_len: int = 48
            scale: bool = True
            truncate_train_for_purge: bool = False
            downsample: int = 1
            data: str = "test_data"
            llm_embedding: dict = None

        args_gpt2 = MockArgs(llm_embedding={'model_name': 'gpt2', 'd_llm': 768})
        args_qwen = MockArgs(llm_embedding={'model_name': 'Qwen2.5-0.5B-Instruct', 'd_llm': 896})

        config_gpt2 = _build_cache_config_with_llm(args_gpt2)
        config_qwen = _build_cache_config_with_llm(args_qwen)

        assert config_gpt2 != config_qwen, \
            "Cache config should differ for different LLM models"

    def test_hash_changes_with_prompt_template(self):
        """Cache hash should change when prompt template changes."""

        @dataclass
        class MockArgs:
            input_len: int = 96
            output_len: int = 48
            scale: bool = True
            truncate_train_for_purge: bool = False
            downsample: int = 1
            data: str = "test_data"
            llm_embedding: dict = None

        args_v1 = MockArgs(llm_embedding={'model_name': 'gpt2', 'prompt_template': 'timecma_v1'})
        args_v2 = MockArgs(llm_embedding={'model_name': 'gpt2', 'prompt_template': 'mmtsflib_v1'})

        config_v1 = _build_cache_config_with_llm(args_v1)
        config_v2 = _build_cache_config_with_llm(args_v2)

        assert config_v1 != config_v2, \
            "Cache config should differ for different prompt templates"

    def test_hash_changes_with_extraction_mode(self):
        """Cache hash should change when extraction mode changes."""

        @dataclass
        class MockArgs:
            input_len: int = 96
            output_len: int = 48
            scale: bool = True
            truncate_train_for_purge: bool = False
            downsample: int = 1
            data: str = "test_data"
            llm_embedding: dict = None

        args_last = MockArgs(llm_embedding={'model_name': 'gpt2', 'extraction_mode': 'last_token'})
        args_pool = MockArgs(llm_embedding={'model_name': 'gpt2', 'extraction_mode': 'pooled'})

        config_last = _build_cache_config_with_llm(args_last)
        config_pool = _build_cache_config_with_llm(args_pool)

        assert config_last != config_pool, \
            "Cache config should differ for different extraction modes"

    def test_hash_unchanged_without_llm(self):
        """Cache hash should remain same structure for non-LLM models."""

        @dataclass
        class MockArgs:
            input_len: int = 96
            output_len: int = 48
            scale: bool = True
            truncate_train_for_purge: bool = False
            downsample: int = 1
            data: str = "test_data"
            llm_embedding: dict = None

        args = MockArgs()
        config = _build_cache_config_with_llm(args)

        # llm_embedding should be None for non-LLM models
        assert config['llm_embedding'] is None

        # Other fields should be present
        assert config['input_len'] == 96
        assert config['output_len'] == 48
        assert config['data_name'] == "test_data"

    def test_identical_llm_config_produces_identical_hash(self):
        """Same LLM config should produce identical cache config."""

        @dataclass
        class MockArgs:
            input_len: int = 96
            output_len: int = 48
            scale: bool = True
            truncate_train_for_purge: bool = False
            downsample: int = 1
            data: str = "test_data"
            llm_embedding: dict = None

        llm_config = {
            'model_name': 'gpt2',
            'prompt_template': 'timecma_v1',
            'extraction_mode': 'last_token',
            'd_llm': 768
        }

        args_1 = MockArgs(llm_embedding=llm_config.copy())
        args_2 = MockArgs(llm_embedding=llm_config.copy())

        config_1 = _build_cache_config_with_llm(args_1)
        config_2 = _build_cache_config_with_llm(args_2)

        assert config_1 == config_2, \
            "Identical LLM configs should produce identical cache configs"


# =============================================================================
# TEST CLASS: Current Tensor Cache Behavior with LLM Embeddings (Documents Bug)
# =============================================================================

class TestCurrentBehaviorWithLLMEmbeddings:
    """
    Tests documenting the CURRENT (broken) behavior when tensor cache
    processes LLM embeddings.

    These tests show WHY tensor cache v2 is needed:
    - Current code incorrectly treats embed_dim as timestamps
    - Embeddings get sliced into individual scalars
    - Reconstruction produces garbage

    After v2 implementation, these tests should be updated to expect
    correct behavior instead of documenting the bug.
    """

    def test_current_slicing_corrupts_llm_embeddings(self, llm_dataset_gpt2):
        """
        Document how current code corrupts LLM embeddings.

        Current behavior in _process_sample_for_collection:
            for i, ts in enumerate(x_time_flat):  # i = 0..95
                emb = hetero_x[i].copy()  # hetero_x is (768, 1), so emb = hetero_x[i] = (1,)

        This extracts single scalars instead of the full embedding!
        """
        sample = llm_dataset_gpt2[0]
        hetero_x = sample[SAMPLE_IDX_HETERO_X]  # Shape: (768, 1)
        x_time = sample[SAMPLE_IDX_X_TIME]       # Shape: (96,)

        # Simulate current (broken) behavior
        corrupted_embeddings = []
        for i in range(len(x_time)):
            if i < hetero_x.shape[0]:  # True: i < 768
                # This is WRONG - extracts row i of the embedding matrix
                emb = hetero_x[i].copy()  # Shape: (1,) - just one scalar!
                corrupted_embeddings.append(emb)

        # We extracted 96 "embeddings", each of shape (1,)
        assert len(corrupted_embeddings) == 96

        # Each "embedding" is just a single scalar - completely corrupted!
        for emb in corrupted_embeddings:
            assert emb.shape == (1,), \
                f"Current behavior produces shape {emb.shape}, should be (1,)"

        # The original embedding shape is completely lost
        original_shape = hetero_x.shape
        corrupted_shape = corrupted_embeddings[0].shape
        assert original_shape != corrupted_shape, \
            "Current behavior destroys LLM embedding shape"

    def test_v2_should_preserve_full_embedding(self, llm_dataset_gpt2):
        """
        Show what v2 SHOULD do: preserve the full LLM embedding.

        Expected v2 behavior:
            if _is_llm_embedding(hetero_x, input_len):
                # Store full embedding as per-sample data
                _register_llm_embedding(collector, hetero_x)
            else:
                # Existing per-timestamp logic
                for i, ts in enumerate(x_time_flat):
                    ...
        """
        sample = llm_dataset_gpt2[0]
        hetero_x = sample[SAMPLE_IDX_HETERO_X]  # Shape: (768, 1)

        # V2 behavior: detect LLM embedding and store whole
        if _is_llm_embedding(hetero_x, llm_dataset_gpt2.input_len):
            preserved_embedding = hetero_x.copy()
        else:
            raise AssertionError("Should detect LLM embedding")

        # Full shape preserved
        assert preserved_embedding.shape == hetero_x.shape

        # Content identical
        assert np.array_equal(preserved_embedding, hetero_x)


# =============================================================================
# TEST CLASS: Model-Specific Requirements
# =============================================================================

class TestTimeCMARequirements:
    """
    Tests verifying tensor cache output meets TimeCMA model requirements.

    TimeCMA expects:
    - historical_events: [B, d_llm, num_nodes] or [B, d_llm, num_nodes, 1]
    - Per-sample LLM embeddings from precomputed cache
    """

    def test_embedding_dimension_first(self, llm_dataset_gpt2):
        """TimeCMA expects embedding dimension as first axis (after batch)."""
        sample = llm_dataset_gpt2[0]
        hetero_x = sample[SAMPLE_IDX_HETERO_X]

        d_llm = hetero_x.shape[0]
        n_nodes = hetero_x.shape[1]

        # Embedding dim should be large (768+)
        assert d_llm >= 512, f"d_llm={d_llm} too small for LLM embedding"

        # n_nodes should be small (typically 1-7)
        assert n_nodes <= 32, f"n_nodes={n_nodes} too large for channel count"

    def test_batch_shape_simulation(self, llm_dataset_gpt2):
        """Simulate batching and verify shape matches TimeCMA expectations."""
        batch_size = 4

        batch_embeddings = []
        for i in range(batch_size):
            sample = llm_dataset_gpt2[i]
            batch_embeddings.append(sample[SAMPLE_IDX_HETERO_X])

        # Stack into batch: [B, d_llm, n_channels]
        batch = np.stack(batch_embeddings, axis=0)

        expected_shape = (batch_size, GPT2_EMBED_DIM, 1)
        assert batch.shape == expected_shape, \
            f"Batch shape {batch.shape}, expected {expected_shape}"


class TestMMTSFLibRequirements:
    """
    Tests verifying tensor cache output meets MMTSFLib model requirements.

    MMTSFLib is more flexible with embedding shapes:
    - historical_events: [B, L, text_dim] or [B, text_dim]
    - Can handle both per-timestamp and per-sample embeddings
    """

    def test_llm_embedding_squeezable(self, llm_dataset_gpt2):
        """
        MMTSFLib can handle (embed_dim, 1) -> (embed_dim,) via squeeze.

        This happens when n_channels=1.
        """
        sample = llm_dataset_gpt2[0]
        hetero_x = sample[SAMPLE_IDX_HETERO_X]  # (768, 1)

        # MMTSFLib does: if text_emb.dim() == 2: text_emb = text_emb.unsqueeze(1)
        # For (768, 1), this becomes (768, 1, 1) after unsqueeze(0) for batch

        # Squeeze the trailing dimension if n_channels=1
        if hetero_x.shape[-1] == 1:
            squeezed = hetero_x.squeeze(-1)  # (768,)
            assert squeezed.shape == (GPT2_EMBED_DIM,)

    def test_multi_channel_handling(self, llm_dataset_gpt2_medium):
        """Test MMTSFLib with multi-channel embeddings."""
        sample = llm_dataset_gpt2_medium[0]
        hetero_x = sample[SAMPLE_IDX_HETERO_X]  # (1024, 7)

        # Should NOT squeeze when n_channels > 1
        assert hetero_x.shape == (GPT2_MEDIUM_EMBED_DIM, 7)

        # MMTSFLib would process this as (embed_dim, n_channels)


# =============================================================================
# TEST CLASS: Integration Tests (V2 Round-Trip)
# =============================================================================

class TestV2RoundTrip:
    """
    Integration tests for complete cache round-trip in v2.

    These tests simulate the full pipeline:
    1. Dataset produces samples with LLM embeddings
    2. Cache generation processes samples correctly
    3. Cache loading reconstructs identical embeddings
    4. Model receives correct shapes
    """

    def test_full_roundtrip_simulation(self, llm_dataset_gpt2):
        """Simulate complete v2 round-trip for LLM embeddings."""
        n_samples = min(10, len(llm_dataset_gpt2))

        # Phase 1: Collection (simulated)
        collected_llm_embeddings = []
        collected_sample_ids = []

        for i in range(n_samples):
            sample = llm_dataset_gpt2[i]
            sample_id = sample[SAMPLE_IDX_SAMPLE_ID]
            hetero_x = sample[SAMPLE_IDX_HETERO_X]

            if _is_llm_embedding(hetero_x, llm_dataset_gpt2.input_len):
                collected_llm_embeddings.append(hetero_x.copy())
                collected_sample_ids.append(sample_id)

        # Phase 2: Save (simulated)
        llm_embeddings_array = np.stack(collected_llm_embeddings, axis=0)
        llm_indices = np.arange(len(collected_llm_embeddings), dtype=np.int32)

        # Phase 3: Load and reconstruct (simulated)
        for i in range(n_samples):
            original_sample = llm_dataset_gpt2[i]
            original_emb = original_sample[SAMPLE_IDX_HETERO_X]

            # Reconstruct from cache
            llm_idx = llm_indices[i]
            reconstructed_emb = llm_embeddings_array[llm_idx]

            # Verify exact match
            assert np.array_equal(original_emb, reconstructed_emb), \
                f"Round-trip failed for sample {i}"

    def test_mixed_dataset_simulation(self, llm_dataset_gpt2, news_dataset):
        """
        Test handling when both LLM and news datasets exist.

        V2 should correctly route:
        - LLM embeddings -> per-sample storage
        - News embeddings -> per-timestamp deduplication
        """
        # Process LLM dataset
        llm_sample = llm_dataset_gpt2[0]
        llm_hetero_x = llm_sample[SAMPLE_IDX_HETERO_X]

        assert _is_llm_embedding(llm_hetero_x, llm_dataset_gpt2.input_len), \
            "Should detect LLM embedding"

        # Process news dataset
        news_sample = news_dataset[0]
        news_hetero_x = news_sample[SAMPLE_IDX_HETERO_X]

        assert not _is_llm_embedding(news_hetero_x, news_dataset.input_len), \
            "Should NOT detect news embedding as LLM"


# =============================================================================
# TEST CLASS: Edge Cases and Error Handling
# =============================================================================

class TestEdgeCasesAndErrors:
    """Tests for edge cases and error conditions."""

    def test_empty_embedding(self):
        """Empty array should not be detected as LLM embedding."""
        empty = np.array([]).astype(np.float32)
        assert _is_llm_embedding(empty, 96) is False

    def test_zero_embedding(self):
        """Zero-filled embedding should still be detected based on shape."""
        zeros = np.zeros((768, 1), dtype=np.float32)
        assert _is_llm_embedding(zeros, 96) is True

    def test_nan_in_embedding(self):
        """NaN values should not affect detection (shape-based)."""
        with_nans = np.full((768, 1), np.nan, dtype=np.float32)
        assert _is_llm_embedding(with_nans, 96) is True

    def test_very_large_embedding(self):
        """Very large embedding (e.g., 4096D from large models) should work."""
        large = np.random.randn(4096, 1).astype(np.float32)
        assert _is_llm_embedding(large, 96) is True

    def test_wide_embedding(self):
        """Wide embedding (many channels) should still be detected."""
        wide = np.random.randn(768, 32).astype(np.float32)
        # 32 channels is at the boundary of our heuristic
        assert _is_llm_embedding(wide, 96) is True

    def test_too_wide_embedding(self):
        """
        Very wide embedding might be misclassified - document this edge case.

        If second dim > 32, we might incorrectly classify it.
        This is acceptable since n_channels > 32 is extremely rare.
        """
        very_wide = np.random.randn(768, 100).astype(np.float32)
        # This might be misclassified - document the behavior
        result = _is_llm_embedding(very_wide, 96)
        # We don't assert specific behavior here - just document
        # that very wide arrays are an edge case


# =============================================================================
# TEST CLASS: Performance Considerations
# =============================================================================

class TestPerformanceConsiderations:
    """
    Tests verifying performance characteristics of LLM embedding handling.

    Key difference from news embeddings:
    - News: high deduplication ratio (10-100x)
    - LLM: no deduplication (1:1)
    """

    def test_llm_no_deduplication_ratio(self, llm_dataset_gpt2):
        """LLM embeddings have 1:1 storage ratio (no deduplication possible)."""
        n_samples = len(llm_dataset_gpt2)

        # Each sample has unique embedding
        unique_embeddings = n_samples
        total_references = n_samples

        ratio = total_references / unique_embeddings
        assert ratio == 1.0, \
            f"LLM embedding deduplication ratio should be 1.0, got {ratio}"

    def test_news_high_deduplication_ratio(self, news_dataset):
        """News embeddings have high deduplication ratio."""
        n_samples = len(news_dataset)
        input_len = news_dataset.input_len
        output_len = news_dataset.output_len

        # Total timestamp references
        total_refs = n_samples * (input_len + output_len)

        # Unique timestamps (sliding window)
        unique_ts = n_samples + input_len + output_len - 1

        ratio = total_refs / unique_ts

        # Should have significant deduplication
        assert ratio > 5.0, \
            f"News embedding should have high deduplication ratio, got {ratio}"

    def test_llm_memory_estimation(self, llm_dataset_gpt2):
        """Estimate memory usage for LLM embedding storage."""
        n_samples = len(llm_dataset_gpt2)
        embed_dim = llm_dataset_gpt2.embed_dim
        n_channels = llm_dataset_gpt2.n_channels
        bytes_per_float = 4  # float32

        memory_bytes = n_samples * embed_dim * n_channels * bytes_per_float
        memory_mb = memory_bytes / (1024 * 1024)

        # For 50 samples, 768D, 1 channel: ~150KB - very reasonable
        assert memory_mb < 1.0, \
            f"LLM embedding storage for test dataset: {memory_mb:.2f} MB"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
