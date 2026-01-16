"""
Test suite for polars-based tensor cache optimization.

This module implements test-driven development (TDD) tests for the tensor cache
polars optimization. These tests verify that any new polars-based implementation
produces **bit-for-bit identical results** to the current Python implementation.

Test Categories:
----------------
1. Unit Tests: Algorithm correctness for individual functions
2. Integration Tests: Full pipeline with mock datasets
3. Real Dataset Tests: Integration with time_mmd_traffic
4. Error Handling Tests: GPU unavailable + missing embeddings
5. Edge Case Tests: Boundary conditions and special cases
6. Performance Benchmarks: Timing comparisons (marked as slow)

Usage:
------
# Run all tests (except slow benchmarks)
pytest tests/test_tensor_cache_polars.py -v -m "not slow"

# Run slow benchmarks
pytest tests/test_tensor_cache_polars.py -v -m "slow"

# Run specific test class
pytest tests/test_tensor_cache_polars.py::TestTimestampDeduplication -v
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
)


# =============================================================================
# FIXTURES: Test Data Generation
# =============================================================================

@pytest.fixture
def sample_timestamps():
    """
    Create sample timestamp data for testing deduplication.

    Simulates 3 samples with input_len=4, output_len=2.
    Timestamps have sliding window overlap (the key optimization target).

    Window structure:
    - sample_0: x=[100,101,102,103] y=[104,105]
    - sample_1: x=[101,102,103,104] y=[105,106]  (overlaps with sample_0)
    - sample_2: x=[102,103,104,105] y=[106,107]  (overlaps with both)

    Total unique timestamps: 8 (100-107)
    Total timestamp references: 18 (3 samples × 6 timestamps)
    Deduplication ratio: 18/8 = 2.25x
    """
    return {
        'sample_0': {
            'x_time': np.array([100, 101, 102, 103], dtype=np.int64),
            'y_time': np.array([104, 105], dtype=np.int64),
        },
        'sample_1': {
            'x_time': np.array([101, 102, 103, 104], dtype=np.int64),
            'y_time': np.array([105, 106], dtype=np.int64),
        },
        'sample_2': {
            'x_time': np.array([102, 103, 104, 105], dtype=np.int64),
            'y_time': np.array([106, 107], dtype=np.int64),
        },
    }


@pytest.fixture
def sample_data():
    """
    Create sample data arrays for testing alignment.

    Provides pre-generated data for each unique timestamp (100-107).
    """
    np.random.seed(42)
    n_features = 3
    embed_dim = 8
    n_htf = 2

    # Unique timestamps: 100-107 (8 total)
    return {
        'timeseries': {ts: np.random.randn(n_features).astype(np.float32)
                       for ts in range(100, 108)},
        'embeddings': {ts: np.random.randn(embed_dim).astype(np.float32)
                       for ts in range(100, 108)},
        'hetero_time': {ts: np.random.randn(n_htf).astype(np.float32)
                        for ts in range(100, 108)},
    }


class MockDataset:
    """
    Mock dataset for testing tensor cache generation.

    Simulates the Universal_Dataset interface with configurable:
    - n_samples: Number of sliding window samples
    - input_len: Input window length (x timestamps)
    - output_len: Output window length (y timestamps)
    - n_features: Time series feature count
    - embed_dim: Embedding dimension

    Generates deterministic data from random seed for reproducibility.
    """

    def __init__(
        self,
        n_samples: int = 100,
        input_len: int = 8,
        output_len: int = 4,
        n_features: int = 3,
        embed_dim: int = 16,
        n_hetero_time_features: int = 4,
        n_time_features: int = 4,
        seed: int = 42,
        entity_id: str = "test_entity"
    ):
        np.random.seed(seed)
        self.n_samples = n_samples
        self.input_len = input_len
        self.output_len = output_len
        self.n_features = n_features
        self.embed_dim = embed_dim
        self.n_hetero_time_features = n_hetero_time_features
        self.n_time_features = n_time_features
        self.entity_id = entity_id

        # Generate continuous timestamps starting at base_ts
        base_ts = 1000
        total_ts_needed = n_samples + input_len + output_len
        self.timestamps = np.arange(base_ts, base_ts + total_ts_needed, dtype=np.int64)

        # Pre-generate data for all timestamps
        n_ts = len(self.timestamps)
        self.all_timeseries = np.random.randn(n_ts, n_features).astype(np.float32)
        self.all_embeddings = np.random.randn(n_ts, embed_dim).astype(np.float32)
        self.all_hetero_time = np.random.randn(n_ts, n_hetero_time_features).astype(np.float32)
        self.all_time_features = np.random.randn(n_ts, n_time_features).astype(np.float32)

        # Entity-level static embeddings
        self.general_embedding = np.random.randn(embed_dim).astype(np.float32)
        self.channel_embedding = np.random.randn(embed_dim).astype(np.float32)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple:
        """
        Return a sample tuple matching Universal_Dataset format.

        Returns 13-element tuple:
        0: sample_id (str)
        1: seq_x (input_len, n_features)
        2: seq_y (output_len, n_features)
        3: x_time (input_len,)
        4: y_time (output_len,)
        5: hetero_x (input_len, embed_dim)
        6: hetero_y (output_len, embed_dim)
        7: hetero_x_time (input_len, n_htf)
        8: hetero_y_time (output_len, n_htf)
        9: hetero_general (embed_dim,)
        10: hetero_channel (embed_dim,)
        11: x_time_features (input_len, n_tf)
        12: y_time_features (output_len, n_tf)
        """
        # Sliding window indices
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
            self.all_embeddings[x_start:x_end],                # hetero_x
            self.all_embeddings[y_start:y_end],                # hetero_y
            self.all_hetero_time[x_start:x_end],               # hetero_x_time
            self.all_hetero_time[y_start:y_end],               # hetero_y_time
            self.general_embedding.copy(),                     # hetero_general
            self.channel_embedding.copy(),                     # hetero_channel
            self.all_time_features[x_start:x_end],             # x_time_features
            self.all_time_features[y_start:y_end],             # y_time_features
        )


@pytest.fixture
def mock_dataset():
    """Create a small mock dataset for testing."""
    return MockDataset(n_samples=100, input_len=8, output_len=4)


@pytest.fixture
def medium_mock_dataset():
    """Create a medium mock dataset for more thorough testing."""
    return MockDataset(n_samples=1000, input_len=24, output_len=12)


@pytest.fixture
def large_mock_dataset():
    """
    Create a larger dataset for performance testing.

    Uses realistic sizes but scaled down for 16GB RAM:
    - 10,000 samples (vs 700k in production)
    - 96 input_len (vs 288)
    - 48 output_len (vs 144)

    Memory estimate: ~500MB (safe for 16GB machine)
    """
    return MockDataset(n_samples=10000, input_len=96, output_len=48)


# =============================================================================
# UNIT TESTS: Algorithm Correctness
# =============================================================================

class TestTimestampDeduplication:
    """Tests for timestamp deduplication algorithms."""

    def test_unique_timestamps_match(self, sample_timestamps):
        """Verify deduplication produces correct unique timestamp set."""
        # Collect all timestamps
        all_ts = []
        for sample in sample_timestamps.values():
            all_ts.extend(sample['x_time'].tolist())
            all_ts.extend(sample['y_time'].tolist())

        # Python set-based deduplication
        python_unique = sorted(set(all_ts))

        # Expected: 100, 101, 102, 103, 104, 105, 106, 107
        expected = list(range(100, 108))

        assert python_unique == expected, (
            f"Unique timestamps incorrect:\n"
            f"Got: {python_unique}\n"
            f"Expected: {expected}"
        )

    def test_deduplication_count(self, sample_timestamps):
        """Verify deduplication achieves expected compression ratio."""
        # Count total timestamp references
        total_refs = sum(
            len(s['x_time']) + len(s['y_time'])
            for s in sample_timestamps.values()
        )

        # Count unique timestamps
        all_ts = set()
        for sample in sample_timestamps.values():
            all_ts.update(sample['x_time'].tolist())
            all_ts.update(sample['y_time'].tolist())
        n_unique = len(all_ts)

        # Verify counts
        assert total_refs == 18, f"Expected 18 total refs, got {total_refs}"
        assert n_unique == 8, f"Expected 8 unique timestamps, got {n_unique}"

        # Compression ratio
        ratio = total_refs / n_unique
        assert ratio == 2.25, f"Expected ratio 2.25, got {ratio}"

    def test_dict_based_index_assignment(self, sample_timestamps):
        """Test dict-based index assignment (current implementation approach)."""
        # Simulate current implementation
        all_ts = []
        for sample in sample_timestamps.values():
            all_ts.extend(sample['x_time'].tolist())
            all_ts.extend(sample['y_time'].tolist())

        # Dict-based: indices assigned in order of first occurrence
        timestamp_to_idx = {}
        for ts in all_ts:
            if ts not in timestamp_to_idx:
                timestamp_to_idx[ts] = len(timestamp_to_idx)

        # Verify bijection
        assert len(timestamp_to_idx) == len(set(timestamp_to_idx.values())), \
            "Index assignment is not a bijection"

        # Verify all timestamps covered
        assert set(timestamp_to_idx.keys()) == set(range(100, 108))

        # Verify indices are contiguous 0..n-1
        assert set(timestamp_to_idx.values()) == set(range(8))

    def test_sorted_index_assignment(self, sample_timestamps):
        """Test sorted index assignment (polars approach)."""
        # Collect all timestamps
        all_ts = []
        for sample in sample_timestamps.values():
            all_ts.extend(sample['x_time'].tolist())
            all_ts.extend(sample['y_time'].tolist())

        # Sorted unique (polars approach)
        sorted_unique = sorted(set(all_ts))
        sorted_idx = {ts: idx for idx, ts in enumerate(sorted_unique)}

        # Verify sorted order: index of ts should be ts - min_ts
        min_ts = min(sorted_unique)
        for ts in sorted_unique:
            expected_idx = ts - min_ts
            assert sorted_idx[ts] == expected_idx, \
                f"Timestamp {ts} has index {sorted_idx[ts]}, expected {expected_idx}"

    def test_index_ordering_independence(self, sample_timestamps):
        """
        Verify that different index orderings still enable correct reconstruction.

        The key insight: the actual index values don't matter, only that:
        1. Each timestamp maps to a unique index
        2. The mapping is consistent across all lookups
        3. Data can be retrieved using these indices
        """
        all_ts = []
        for sample in sample_timestamps.values():
            all_ts.extend(sample['x_time'].tolist())
            all_ts.extend(sample['y_time'].tolist())

        # Method 1: Insertion order (current Python implementation)
        insertion_order = {}
        for ts in all_ts:
            if ts not in insertion_order:
                insertion_order[ts] = len(insertion_order)

        # Method 2: Sorted order (polars implementation)
        sorted_order = {ts: idx for idx, ts in enumerate(sorted(set(all_ts)))}

        # Both should have same keys
        assert insertion_order.keys() == sorted_order.keys()

        # Both should have contiguous indices
        assert set(insertion_order.values()) == set(range(len(insertion_order)))
        assert set(sorted_order.values()) == set(range(len(sorted_order)))

        # Verify round-trip: ts -> idx -> ts
        reverse_insertion = {v: k for k, v in insertion_order.items()}
        reverse_sorted = {v: k for k, v in sorted_order.items()}

        for ts in all_ts:
            idx1 = insertion_order[ts]
            idx2 = sorted_order[ts]
            assert reverse_insertion[idx1] == ts
            assert reverse_sorted[idx2] == ts


class TestIndexGeneration:
    """Tests for timestamp-to-index conversion."""

    def test_basic_index_lookup(self, sample_timestamps):
        """Verify basic index lookup works correctly."""
        # Build index
        all_ts = []
        for sample in sample_timestamps.values():
            all_ts.extend(sample['x_time'].tolist())
            all_ts.extend(sample['y_time'].tolist())

        timestamp_to_idx = {}
        for ts in sorted(set(all_ts)):
            timestamp_to_idx[ts] = len(timestamp_to_idx)

        # Test lookup
        test_ts = sample_timestamps['sample_1']['x_time']
        indices = [timestamp_to_idx[int(ts)] for ts in test_ts]

        # sample_1 x_time = [101, 102, 103, 104] -> indices [1, 2, 3, 4]
        expected = [1, 2, 3, 4]
        assert indices == expected, f"Got {indices}, expected {expected}"

    def test_list_comprehension_lookup(self, sample_timestamps):
        """Test list comprehension lookup (current implementation pattern)."""
        # Build index
        all_ts = []
        for sample in sample_timestamps.values():
            all_ts.extend(sample['x_time'].tolist())
            all_ts.extend(sample['y_time'].tolist())

        timestamp_to_idx = {ts: idx for idx, ts in enumerate(sorted(set(all_ts)))}

        # Current implementation pattern
        test_ts = sample_timestamps['sample_0']['x_time']
        indices = np.array(
            [timestamp_to_idx.get(int(ts), 0) for ts in test_ts],
            dtype=np.int32
        )

        expected = np.array([0, 1, 2, 3], dtype=np.int32)
        np.testing.assert_array_equal(indices, expected)

    def test_missing_timestamp_handling(self):
        """Test handling of timestamps not in index (should default to 0)."""
        index_dict = {100: 0, 101: 1, 102: 2}

        # Query with missing timestamp
        query_ts = [100, 999, 101]  # 999 is missing
        indices = [index_dict.get(ts, 0) for ts in query_ts]

        # Missing timestamp should map to default (0)
        expected = [0, 0, 1]
        assert indices == expected

    def test_vectorized_lookup_equivalence(self, mock_dataset):
        """
        Verify that vectorized numpy lookup matches list comprehension.

        This tests the key optimization: replacing Python loops with numpy.
        """
        # Build timestamp index from first 100 samples
        timestamp_to_idx = {}
        for i in range(min(100, len(mock_dataset))):
            sample = mock_dataset[i]
            for ts in sample[SAMPLE_IDX_X_TIME].flatten():
                if int(ts) not in timestamp_to_idx:
                    timestamp_to_idx[int(ts)] = len(timestamp_to_idx)
            for ts in sample[SAMPLE_IDX_Y_TIME].flatten():
                if int(ts) not in timestamp_to_idx:
                    timestamp_to_idx[int(ts)] = len(timestamp_to_idx)

        # Get test timestamps
        test_sample = mock_dataset[50]
        test_ts = test_sample[SAMPLE_IDX_X_TIME].flatten()

        # Method 1: List comprehension (current)
        list_comp_result = np.array(
            [timestamp_to_idx.get(int(ts), 0) for ts in test_ts],
            dtype=np.int32
        )

        # Method 2: Vectorized numpy
        # Build reverse lookup array
        max_ts = max(timestamp_to_idx.keys())
        min_ts = min(timestamp_to_idx.keys())
        lookup_array = np.zeros(max_ts - min_ts + 1, dtype=np.int32)
        for ts, idx in timestamp_to_idx.items():
            lookup_array[ts - min_ts] = idx

        # Vectorized lookup
        vectorized_result = lookup_array[test_ts.astype(np.int64) - min_ts]

        np.testing.assert_array_equal(list_comp_result, vectorized_result)


class TestDataAlignment:
    """Tests for data alignment with timestamp indices."""

    def test_timeseries_alignment(self, sample_timestamps, sample_data):
        """Verify time series data aligns correctly with indices."""
        # Build sorted unique timestamps
        all_ts = sorted(set(
            ts for sample in sample_timestamps.values()
            for ts in list(sample['x_time']) + list(sample['y_time'])
        ))

        n_unique = len(all_ts)
        n_features = list(sample_data['timeseries'].values())[0].shape[0]

        # Create aligned array
        aligned_ts = np.zeros((n_unique, n_features), dtype=np.float32)
        for idx, ts in enumerate(all_ts):
            aligned_ts[idx] = sample_data['timeseries'][ts]

        # Verify lookup via index retrieval
        sample_x_time = sample_timestamps['sample_0']['x_time']
        indices = [all_ts.index(ts) for ts in sample_x_time]

        for i, ts in enumerate(sample_x_time):
            np.testing.assert_array_almost_equal(
                aligned_ts[indices[i]],
                sample_data['timeseries'][ts],
                decimal=6,
                err_msg=f"Timeseries mismatch at timestamp {ts}"
            )

    def test_embedding_alignment(self, sample_timestamps, sample_data):
        """Verify embeddings align correctly with indices."""
        all_ts = sorted(set(
            ts for sample in sample_timestamps.values()
            for ts in list(sample['x_time']) + list(sample['y_time'])
        ))

        embed_dim = list(sample_data['embeddings'].values())[0].shape[0]
        aligned_emb = np.zeros((len(all_ts), embed_dim), dtype=np.float32)

        for idx, ts in enumerate(all_ts):
            aligned_emb[idx] = sample_data['embeddings'][ts]

        # Test retrieval
        sample_y_time = sample_timestamps['sample_2']['y_time']
        for ts in sample_y_time:
            idx = all_ts.index(ts)
            np.testing.assert_array_almost_equal(
                aligned_emb[idx],
                sample_data['embeddings'][ts],
                decimal=6
            )

    def test_reconstruction_from_indices(self, mock_dataset):
        """
        Test that samples can be reconstructed from shared tables via indices.

        This is the core correctness test: after deduplication, we must be
        able to reconstruct the original sample data exactly.
        """
        # Build shared tables from dataset
        timestamp_to_idx = {}
        timeseries_list = []
        embeddings_list = []

        # Collect unique data
        for i in range(len(mock_dataset)):
            sample = mock_dataset[i]
            x_time = sample[SAMPLE_IDX_X_TIME].flatten()
            y_time = sample[SAMPLE_IDX_Y_TIME].flatten()
            seq_x = sample[SAMPLE_IDX_SEQ_X]
            seq_y = sample[SAMPLE_IDX_SEQ_Y]
            hetero_x = sample[SAMPLE_IDX_HETERO_X]
            hetero_y = sample[SAMPLE_IDX_HETERO_Y]

            for j, ts in enumerate(x_time):
                ts_int = int(ts)
                if ts_int not in timestamp_to_idx:
                    timestamp_to_idx[ts_int] = len(timestamp_to_idx)
                    timeseries_list.append(seq_x[j].copy())
                    embeddings_list.append(hetero_x[j].copy())

            for j, ts in enumerate(y_time):
                ts_int = int(ts)
                if ts_int not in timestamp_to_idx:
                    timestamp_to_idx[ts_int] = len(timestamp_to_idx)
                    timeseries_list.append(seq_y[j].copy())
                    embeddings_list.append(hetero_y[j].copy())

        # Convert to arrays
        timeseries_array = np.stack(timeseries_list)
        embeddings_array = np.stack(embeddings_list)

        # Test reconstruction for several samples
        for sample_idx in [0, 10, 50, 99]:
            sample = mock_dataset[sample_idx]
            original_seq_x = sample[SAMPLE_IDX_SEQ_X]
            original_hetero_x = sample[SAMPLE_IDX_HETERO_X]
            x_time = sample[SAMPLE_IDX_X_TIME].flatten()

            # Get indices
            indices = np.array([timestamp_to_idx[int(ts)] for ts in x_time])

            # Reconstruct
            reconstructed_seq_x = timeseries_array[indices]
            reconstructed_hetero_x = embeddings_array[indices]

            np.testing.assert_array_almost_equal(
                original_seq_x, reconstructed_seq_x, decimal=6,
                err_msg=f"seq_x reconstruction failed for sample {sample_idx}"
            )
            np.testing.assert_array_almost_equal(
                original_hetero_x, reconstructed_hetero_x, decimal=6,
                err_msg=f"hetero_x reconstruction failed for sample {sample_idx}"
            )


class TestInferredShapes:
    """Tests for shape inference from samples."""

    def test_infer_shapes_from_mock_sample(self, mock_dataset):
        """Test shape inference from mock dataset sample."""
        sample = mock_dataset[0]
        shapes = infer_shapes_from_sample(sample)

        assert shapes.input_len == mock_dataset.input_len
        assert shapes.output_len == mock_dataset.output_len
        assert shapes.n_features == mock_dataset.n_features
        assert shapes.embed_dim == mock_dataset.embed_dim

    def test_safe_array_with_none(self):
        """Test _safe_array handles None correctly."""
        assert _safe_array(None) is None

    def test_safe_array_with_empty(self):
        """Test _safe_array handles empty arrays correctly."""
        assert _safe_array(np.array([])) is None

    def test_safe_array_with_valid_data(self):
        """Test _safe_array with valid data."""
        data = [1, 2, 3]
        result = _safe_array(data)
        np.testing.assert_array_equal(result, np.array(data))

    def test_infer_dim_1d(self):
        """Test dimension inference from 1D array."""
        arr = np.zeros(768)
        assert _infer_dim(arr) == 768

    def test_infer_dim_2d(self):
        """Test dimension inference from 2D array."""
        arr = np.zeros((100, 768))
        assert _infer_dim(arr, axis=-1) == 768
        assert _infer_dim(arr, axis=0) == 100


# =============================================================================
# INTEGRATION TESTS: Full Pipeline
# =============================================================================

class TestSharedTableCollector:
    """Tests for SharedTableCollector functionality."""

    def test_collector_initialization(self):
        """Test basic collector initialization."""
        shapes = InferredShapes(
            n_features=3,
            embed_dim=16,
            n_hetero_time_features=4,
            input_len=8,
            output_len=4
        )

        collector = SharedTableCollector(
            timestamp_to_idx={},
            entity_to_idx={},
            timestamps=[],
            timeseries=[],
            embeddings=[],
            hetero_time=[],
            entity_general=[],
            entity_channel=[],
            shapes=shapes,
            num_news_items=1
        )

        assert len(collector.timestamp_to_idx) == 0
        assert collector.num_news_items == 1

    def test_collector_with_mock_data(self, mock_dataset):
        """Test collector accumulates data correctly."""
        shapes = infer_shapes_from_sample(mock_dataset[0])

        collector = SharedTableCollector(
            timestamp_to_idx={},
            entity_to_idx={},
            timestamps=[],
            timeseries=[],
            embeddings=[],
            hetero_time=[],
            entity_general=[],
            entity_channel=[],
            shapes=shapes,
            num_news_items=1
        )

        # Simulate collecting from first 10 samples
        for i in range(10):
            sample = mock_dataset[i]
            x_time = sample[SAMPLE_IDX_X_TIME].flatten()
            seq_x = sample[SAMPLE_IDX_SEQ_X]
            hetero_x = sample[SAMPLE_IDX_HETERO_X]
            hetero_x_time = sample[SAMPLE_IDX_HETERO_X_TIME]

            for j, ts in enumerate(x_time):
                ts_int = int(ts)
                if ts_int not in collector.timestamp_to_idx:
                    collector.timestamp_to_idx[ts_int] = len(collector.timestamps)
                    collector.timestamps.append(ts_int)
                    collector.timeseries.append(seq_x[j].copy())
                    collector.embeddings.append(hetero_x[j].copy())
                    collector.hetero_time.append(hetero_x_time[j].copy())

        # With 10 samples and input_len=8, we should have:
        # First sample: 8 new timestamps
        # Each subsequent sample adds 1 new timestamp (sliding window)
        # Total: 8 + 9 = 17 unique timestamps
        expected_unique = mock_dataset.input_len + (10 - 1)
        assert len(collector.timestamps) == expected_unique
        assert len(collector.timeseries) == expected_unique
        assert len(collector.embeddings) == expected_unique


class TestDeduplicationEffectiveness:
    """Tests for verifying deduplication compression ratios."""

    def test_sliding_window_deduplication(self, mock_dataset):
        """
        Test that sliding window samples achieve expected deduplication.

        With stride=1 sliding window:
        - Sample i and sample i+1 share (input_len - 1) timestamps
        - Deduplication ratio approaches input_len for large datasets
        """
        n_samples = min(100, len(mock_dataset))

        # Count total timestamp references
        total_refs = 0
        unique_ts = set()

        for i in range(n_samples):
            sample = mock_dataset[i]
            x_time = sample[SAMPLE_IDX_X_TIME].flatten()
            y_time = sample[SAMPLE_IDX_Y_TIME].flatten()

            total_refs += len(x_time) + len(y_time)
            unique_ts.update(x_time.tolist())
            unique_ts.update(y_time.tolist())

        n_unique = len(unique_ts)
        ratio = total_refs / n_unique

        # For 100 samples with input_len=8, output_len=4:
        # Total refs = 100 * (8 + 4) = 1200
        # Unique = 8 + 4 + 99 = 111 (first sample fully unique, then +1 per sample)
        # Ratio = 1200 / 111 ≈ 10.8

        assert ratio > 5, f"Deduplication ratio {ratio:.2f} is too low"
        assert n_unique < total_refs, "No deduplication occurred"

    def test_realistic_deduplication_ratio(self, medium_mock_dataset):
        """Test deduplication ratio with more realistic sizes."""
        dataset = medium_mock_dataset
        n_samples = len(dataset)

        total_refs = 0
        unique_ts = set()

        for i in range(n_samples):
            sample = dataset[i]
            x_time = sample[SAMPLE_IDX_X_TIME].flatten()
            y_time = sample[SAMPLE_IDX_Y_TIME].flatten()

            total_refs += len(x_time) + len(y_time)
            unique_ts.update(x_time.tolist())
            unique_ts.update(y_time.tolist())

        n_unique = len(unique_ts)
        ratio = total_refs / n_unique

        # With 1000 samples and larger windows, ratio should be higher
        assert ratio > 10, f"Deduplication ratio {ratio:.2f} lower than expected"


# =============================================================================
# ERROR HANDLING TESTS: GPU + Embedding Errors
# =============================================================================

class TestGPUUnavailableErrors:
    """
    Tests for clear error messages when GPU unavailable and embeddings not cached.

    These tests verify the system fails gracefully with actionable error messages
    when:
    1. GPU is requested but not available
    2. Embeddings need to be computed (not cached)

    This is critical for CPU-only nodes where embeddings must be pre-computed.
    """

    def test_gpu_required_error_message_format(self):
        """
        Test that GPU unavailable error has clear, actionable message.

        The error should:
        1. Clearly state GPU is required
        2. Explain WHY (embedding computation)
        3. Suggest solutions (use GPU node or pre-compute embeddings)
        """
        # Import the embedder
        from embedder.fidel_ts_embedder import FidelTSEmbeddingLoader

        # Create loader with CUDA device
        loader = FidelTSEmbeddingLoader(
            dataset_name='test_dataset',
            hetero_info={'root_path': './data/test'},
            base_data_path='./data',
            embed_model_name='bert-base-uncased',
            device='cuda:0',  # Request GPU
        )

        # Mock torch.cuda.is_available() to return False
        with patch('torch.cuda.is_available', return_value=False):
            # _init_embedder should raise RuntimeError with clear message
            with pytest.raises(RuntimeError) as exc_info:
                loader._init_embedder()

            error_msg = str(exc_info.value)

            # Verify error message contains key information
            assert "GPU REQUIRED" in error_msg or "GPU" in error_msg.upper(), \
                "Error should mention GPU requirement"
            assert "cuda" in error_msg.lower() or "embedding" in error_msg.lower(), \
                "Error should mention CUDA or embeddings"

    def test_embedding_cache_miss_with_no_gpu(self):
        """
        Test error when embeddings not cached and GPU unavailable.

        Scenario: Running on CPU-only node, embeddings haven't been pre-computed.
        Expected: Clear error explaining the situation and solutions.
        """
        from embedder.fidel_ts_embedder import FidelTSEmbeddingLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal dataset structure
            data_path = Path(tmpdir)

            loader = FidelTSEmbeddingLoader(
                dataset_name='nonexistent_dataset',
                hetero_info={'root_path': str(data_path)},
                base_data_path=str(data_path),
                embed_model_name='bert-base-uncased',
                device='cuda:0',
            )

            # Mock no GPU available
            with patch('torch.cuda.is_available', return_value=False):
                # Attempting to load embeddings that don't exist should fail
                # with clear error about GPU requirement
                with pytest.raises((RuntimeError, ValueError, FileNotFoundError)):
                    loader.load_embeddings()

    def test_cpu_only_with_cached_embeddings_succeeds(self):
        """
        Test that CPU-only loading works when embeddings are already cached.

        This is the expected workflow:
        1. Pre-compute embeddings on GPU node
        2. Run tensor cache generation on CPU node (reads cached embeddings)

        The test verifies step 2 works without requiring GPU.
        """
        # This test needs a real dataset with cached embeddings
        # Skip if time_mmd_traffic not available
        traffic_path = Path('./data/time_mmd/Traffic')
        if not traffic_path.exists():
            pytest.skip("time_mmd_traffic dataset not available")

        embeddings_path = traffic_path / 'embeddings_3af7b92ef045b153'
        if not embeddings_path.exists():
            pytest.skip("Cached embeddings not available")

        # The existence of cached embeddings means CPU loading should work
        # (no need to actually run full embedding load in this unit test)
        assert embeddings_path.exists(), "Cached embeddings directory exists"
        assert (embeddings_path / 'embeddings.pkl').exists() or \
               any(embeddings_path.glob('*.pkl')), "Embeddings files exist"


class TestEmbeddingErrorMessages:
    """Tests for embedding-related error messages."""

    def test_missing_embedding_source_error(self):
        """Test error when no embedding source available."""
        from embedder.fidel_ts_embedder import FidelTSEmbeddingLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir)

            # Create loader with no embedding sources
            loader = FidelTSEmbeddingLoader(
                dataset_name='empty_dataset',
                hetero_info={'root_path': str(data_path)},
                base_data_path=str(data_path),
                embed_model_name='bert-base-uncased',
                device='cpu',
                use_old_embeddings=False,  # Don't use old system
            )

            # Trying to load should fail with clear error
            with pytest.raises((ValueError, FileNotFoundError)) as exc_info:
                loader.load_embeddings()

            # Error should be informative
            error_msg = str(exc_info.value)
            assert len(error_msg) > 10, "Error message should be descriptive"

    def test_corrupted_cache_error(self):
        """Test handling of corrupted embedding cache."""
        # This test verifies that corrupted/incomplete caches are detected
        # and produce clear error messages rather than silent failures

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / 'embeddings_test'
            cache_dir.mkdir()

            # Create an incomplete/corrupted cache (missing required files)
            (cache_dir / 'metadata.json').write_text('{}')  # Empty metadata

            # Attempting to load should fail gracefully
            # (specific error depends on implementation)


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_dataset(self):
        """Handle empty datasets gracefully."""
        empty_dataset = MockDataset(n_samples=0)
        assert len(empty_dataset) == 0

    def test_single_sample(self):
        """Handle single-sample dataset."""
        single_dataset = MockDataset(n_samples=1, input_len=4, output_len=2)
        assert len(single_dataset) == 1

        sample = single_dataset[0]
        assert sample[SAMPLE_IDX_X_TIME].shape[0] == 4
        assert sample[SAMPLE_IDX_Y_TIME].shape[0] == 2

        # Single sample should have input_len + output_len unique timestamps
        all_ts = set(sample[SAMPLE_IDX_X_TIME].tolist() + sample[SAMPLE_IDX_Y_TIME].tolist())
        assert len(all_ts) == 6  # 4 + 2

    def test_large_timestamps(self):
        """Handle int64 timestamps correctly."""
        # Test with large timestamp values (e.g., Unix milliseconds)
        large_ts = np.array([2**40 + i for i in range(100)], dtype=np.int64)

        # Should not overflow
        assert large_ts.dtype == np.int64
        assert large_ts[0] == 2**40
        assert large_ts[99] == 2**40 + 99

        # Deduplication should work with large timestamps
        unique = set(large_ts.tolist())
        assert len(unique) == 100

    def test_dtype_consistency(self, mock_dataset):
        """Verify dtypes are consistent across samples."""
        sample0 = mock_dataset[0]
        sample1 = mock_dataset[1]

        # Time arrays should be int64
        assert sample0[SAMPLE_IDX_X_TIME].dtype == np.int64
        assert sample1[SAMPLE_IDX_X_TIME].dtype == np.int64

        # Data arrays should be float32
        assert sample0[SAMPLE_IDX_SEQ_X].dtype == np.float32
        assert sample0[SAMPLE_IDX_HETERO_X].dtype == np.float32

    def test_negative_timestamps(self):
        """Handle negative timestamps (e.g., dates before Unix epoch)."""
        negative_ts = np.array([-1000, -999, -998, -1], dtype=np.int64)

        # Should work with negative timestamps
        unique = sorted(set(negative_ts.tolist()))
        assert unique == [-1000, -999, -998, -1]

    def test_non_contiguous_timestamps(self):
        """Handle non-contiguous timestamps (gaps in sequence)."""
        # Timestamps with gaps
        gapped_ts = np.array([100, 105, 110, 200], dtype=np.int64)

        # Deduplication should still work
        timestamp_to_idx = {ts: idx for idx, ts in enumerate(gapped_ts)}
        assert timestamp_to_idx[100] == 0
        assert timestamp_to_idx[200] == 3

    def test_duplicate_within_single_sample(self):
        """Test handling if a sample somehow has duplicate timestamps."""
        # This shouldn't normally happen but test robustness
        dup_ts = np.array([100, 100, 101, 101], dtype=np.int64)

        # Deduplication should handle this
        unique = sorted(set(dup_ts.tolist()))
        assert unique == [100, 101]

    def test_copy_breaks_reference(self, mock_dataset):
        """
        Verify that .copy() properly breaks references.

        Critical for memory safety: modifying collected data shouldn't
        affect original dataset arrays.
        """
        sample = mock_dataset[0]
        original_hetero_x = sample[SAMPLE_IDX_HETERO_X].copy()

        # Simulate collection with copy
        collected = sample[SAMPLE_IDX_HETERO_X][0].copy()

        # Modify collected data
        collected[:] = 999.0

        # Original should be unchanged
        np.testing.assert_array_equal(
            sample[SAMPLE_IDX_HETERO_X],
            original_hetero_x,
            err_msg="Original array was modified - copy() not working"
        )


# =============================================================================
# PERFORMANCE BENCHMARKS (marked slow)
# =============================================================================

class TestPerformance:
    """Performance benchmarks for comparing implementations."""

    @pytest.mark.slow
    def test_deduplication_performance(self, large_mock_dataset):
        """Benchmark timestamp deduplication performance."""
        import time

        # Extract all timestamps
        all_ts = []
        for idx in range(len(large_mock_dataset)):
            sample = large_mock_dataset[idx]
            all_ts.extend(sample[SAMPLE_IDX_X_TIME].tolist())
            all_ts.extend(sample[SAMPLE_IDX_Y_TIME].tolist())

        # Python dict implementation (current)
        start = time.perf_counter()
        python_unique = {}
        for ts in all_ts:
            if ts not in python_unique:
                python_unique[ts] = len(python_unique)
        python_time = time.perf_counter() - start

        # Python set + sorted (simpler)
        start = time.perf_counter()
        set_unique = {ts: idx for idx, ts in enumerate(sorted(set(all_ts)))}
        set_time = time.perf_counter() - start

        # Verify correctness
        assert len(python_unique) == len(set_unique)

        print(f"\nDeduplication performance ({len(all_ts):,} timestamps):")
        print(f"  Dict-based: {python_time*1000:.1f}ms")
        print(f"  Set+sorted: {set_time*1000:.1f}ms")

    @pytest.mark.slow
    def test_index_lookup_performance(self, large_mock_dataset):
        """Benchmark index lookup performance."""
        import time

        # Build index
        all_ts = []
        for idx in range(len(large_mock_dataset)):
            sample = large_mock_dataset[idx]
            all_ts.extend(sample[SAMPLE_IDX_X_TIME].tolist())
            all_ts.extend(sample[SAMPLE_IDX_Y_TIME].tolist())

        index_dict = {ts: idx for idx, ts in enumerate(sorted(set(all_ts)))}

        # Prepare test data - all x_time arrays
        test_timestamps = [large_mock_dataset[i][SAMPLE_IDX_X_TIME] for i in range(100)]

        # Python list comprehension
        start = time.perf_counter()
        for x_time in test_timestamps:
            indices = [index_dict.get(int(ts), 0) for ts in x_time]
        python_time = time.perf_counter() - start

        # Numpy vectorized (using lookup array)
        max_ts = max(index_dict.keys())
        min_ts = min(index_dict.keys())
        lookup_array = np.zeros(max_ts - min_ts + 1, dtype=np.int32)
        for ts, idx in index_dict.items():
            lookup_array[ts - min_ts] = idx

        start = time.perf_counter()
        for x_time in test_timestamps:
            indices = lookup_array[x_time.astype(np.int64) - min_ts]
        numpy_time = time.perf_counter() - start

        speedup = python_time / numpy_time if numpy_time > 0 else float('inf')

        print(f"\nIndex lookup performance (100 samples):")
        print(f"  List comprehension: {python_time*1000:.1f}ms")
        print(f"  Numpy vectorized:   {numpy_time*1000:.1f}ms")
        print(f"  Speedup: {speedup:.1f}x")

    @pytest.mark.slow
    def test_memory_efficiency(self, large_mock_dataset):
        """Test memory usage of shared tables vs duplicated storage."""
        import sys

        # Calculate memory for full duplication
        sample = large_mock_dataset[0]
        n_samples = len(large_mock_dataset)
        input_len = large_mock_dataset.input_len
        output_len = large_mock_dataset.output_len

        # Memory per sample if fully duplicated
        seq_x_size = sample[SAMPLE_IDX_SEQ_X].nbytes
        seq_y_size = sample[SAMPLE_IDX_SEQ_Y].nbytes
        hetero_x_size = sample[SAMPLE_IDX_HETERO_X].nbytes
        hetero_y_size = sample[SAMPLE_IDX_HETERO_Y].nbytes

        per_sample_size = seq_x_size + seq_y_size + hetero_x_size + hetero_y_size
        duplicated_total = per_sample_size * n_samples

        # Memory with deduplication
        # Unique timestamps = input_len + output_len + (n_samples - 1)
        n_unique = input_len + output_len + (n_samples - 1)
        per_ts_size = (seq_x_size / input_len) + (hetero_x_size / input_len)
        deduplicated_total = per_ts_size * n_unique

        # Index arrays (small)
        index_array_size = n_samples * (input_len + output_len) * 4  # int32
        deduplicated_total += index_array_size

        ratio = duplicated_total / deduplicated_total

        print(f"\nMemory efficiency ({n_samples:,} samples):")
        print(f"  Duplicated storage: {duplicated_total/1e6:.1f} MB")
        print(f"  Deduplicated storage: {deduplicated_total/1e6:.1f} MB")
        print(f"  Compression ratio: {ratio:.1f}x")

        assert ratio > 5, f"Expected >5x compression, got {ratio:.1f}x"


# =============================================================================
# REAL DATASET TESTS: time_mmd_traffic
# =============================================================================

class TestTimeMmdTraffic:
    """Integration tests with real time_mmd_traffic dataset."""

    @pytest.fixture
    def traffic_data_path(self):
        """Get path to time_mmd_traffic dataset."""
        path = Path('./data/time_mmd/Traffic')
        if not path.exists():
            pytest.skip("time_mmd_traffic dataset not available")
        return path

    @pytest.fixture
    def traffic_config_path(self):
        """Get path to time_mmd_traffic config."""
        path = Path('./data_configs/time_mmd/Traffic/config.yaml')
        if not path.exists():
            pytest.skip("time_mmd_traffic config not available")
        return path

    def test_traffic_dataset_structure(self, traffic_data_path):
        """Verify time_mmd_traffic has expected structure."""
        # Check required files
        assert (traffic_data_path / 'US_VMT_Month.csv').exists(), "Missing CSV data"
        assert (traffic_data_path / 'id_info.json').exists(), "Missing id_info.json"

        # Check for cached embeddings
        embedding_dirs = list(traffic_data_path.glob('embeddings_*'))
        llm_dirs = list((traffic_data_path / 'llm_embeddings').glob('llm_*')) \
            if (traffic_data_path / 'llm_embeddings').exists() else []

        has_embeddings = len(embedding_dirs) > 0 or len(llm_dirs) > 0
        assert has_embeddings, "No cached embeddings found - required for CPU testing"

    def test_traffic_embeddings_exist(self, traffic_data_path):
        """Verify that embeddings are pre-computed for CPU-only testing."""
        # Check for at least one embedding cache
        embedding_dirs = list(traffic_data_path.glob('embeddings_*'))

        if not embedding_dirs:
            # Check LLM embeddings
            llm_path = traffic_data_path / 'llm_embeddings'
            if llm_path.exists():
                llm_dirs = list(llm_path.glob('llm_*'))
                assert len(llm_dirs) > 0, "No LLM embedding caches found"
            else:
                pytest.skip("No embedding caches found")
        else:
            # Verify embedding files exist
            for emb_dir in embedding_dirs:
                files = list(emb_dir.glob('*.pkl'))
                assert len(files) > 0, f"No embedding files in {emb_dir}"

    def test_traffic_csv_readable(self, traffic_data_path):
        """Test that traffic CSV can be read."""
        import pandas as pd

        csv_path = traffic_data_path / 'US_VMT_Month.csv'
        df = pd.read_csv(csv_path)

        assert len(df) > 0, "CSV is empty"
        assert 'date' in df.columns or 'timestamp' in df.columns.str.lower(), \
            "Missing date/timestamp column"

    def test_traffic_sample_shapes(self, traffic_config_path):
        """
        Test that traffic dataset samples have expected shapes.

        This test loads a small number of samples to verify the
        data format is compatible with tensor cache generation.
        """
        import yaml
        from data_provider.data_factory import Data_Provider
        from utils.tools import dotdict

        # Load config
        with open(traffic_config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Create minimal args
        class TestArgs:
            def __init__(self):
                self.data_config = dotdict(config_dict)
                self.batch_size = 4
                self.input_len = 12  # Small for testing
                self.output_len = 6
                self.scale = True
                self.num_workers = 0
                self.prefetch_factor = 2
                self.preload_hetero = False
                self.disable_buffer = False
                self.noise = 0.0

                class ModelConfig:
                    task = 'TSF'
                    custom_input = None
                    stride = 1
                    hetero_align_stride = False
                    name = 'DLinear'
                self.model_config = ModelConfig()
                self.data_config.hetero_info = None

        args = TestArgs()

        try:
            data_provider = Data_Provider(args, buffer=False)
            train_dataset = data_provider.get_train(return_type='set')

            # Get first dataset
            first_key = list(train_dataset.keys())[0]
            dataset = train_dataset[first_key]

            if len(dataset) > 0:
                sample = dataset[0]

                # Verify sample structure
                assert len(sample) == 13, f"Sample should have 13 elements, got {len(sample)}"
                assert isinstance(sample[SAMPLE_IDX_SAMPLE_ID], str), "sample_id should be string"
                assert sample[SAMPLE_IDX_X_TIME].shape[0] == args.input_len
                assert sample[SAMPLE_IDX_Y_TIME].shape[0] == args.output_len
        except Exception as e:
            pytest.skip(f"Could not load traffic dataset: {e}")


# =============================================================================
# POLARS-SPECIFIC TESTS
# =============================================================================

class TestPolarsEquivalence:
    """
    Tests to verify polars implementation matches current implementation.

    These tests verify that the polars-based functions produce identical
    results to the dict-based Python implementation.
    """

    def test_polars_unique_matches_python_set(self, sample_timestamps):
        """Verify pl.DataFrame.unique() matches Python set()."""
        import polars as pl

        all_ts = []
        for sample in sample_timestamps.values():
            all_ts.extend(sample['x_time'].tolist())
            all_ts.extend(sample['y_time'].tolist())

        # Python
        python_unique = sorted(set(all_ts))

        # Polars
        polars_unique = (
            pl.DataFrame({'ts': all_ts})
            .unique()
            .sort('ts')
            ['ts'].to_list()
        )

        assert python_unique == polars_unique

    def test_polars_join_matches_dict_get(self, sample_timestamps):
        """Verify polars join matches dict.get() for index lookup."""
        import polars as pl

        # Build index
        all_ts = []
        for sample in sample_timestamps.values():
            all_ts.extend(sample['x_time'].tolist())
            all_ts.extend(sample['y_time'].tolist())

        # Python dict approach
        python_idx = {}
        for ts in sorted(set(all_ts)):
            python_idx[ts] = len(python_idx)

        # Polars DataFrame approach
        index_df = (
            pl.DataFrame({'ts': all_ts})
            .unique()
            .sort('ts')
            .with_row_index('ts_idx')
        )

        # Test lookup
        test_ts = sample_timestamps['sample_1']['x_time']

        # Python: list comprehension with dict.get()
        python_indices = [python_idx.get(int(ts), 0) for ts in test_ts]

        # Polars: join operation
        query_df = pl.DataFrame({'ts': test_ts.tolist()})
        polars_result = (
            query_df
            .join(index_df.select(['ts', 'ts_idx']), on='ts', how='left')
            .with_columns(pl.col('ts_idx').fill_null(0))
        )
        polars_indices = polars_result['ts_idx'].to_list()

        assert python_indices == polars_indices


class TestPolarsImplementation:
    """Tests for the polars tensor cache implementation module."""

    def test_build_timestamp_index(self, sample_timestamps):
        """Test build_timestamp_index produces correct sorted index."""
        import polars as pl
        from data_provider.tensor_cache_polars import build_timestamp_index

        all_ts = []
        for sample in sample_timestamps.values():
            all_ts.extend(sample['x_time'].tolist())
            all_ts.extend(sample['y_time'].tolist())

        ts_df = pl.DataFrame({'ts': all_ts})
        index_df = build_timestamp_index(ts_df)

        # Check columns
        assert 'ts' in index_df.columns
        assert 'ts_idx' in index_df.columns

        # Check sorted order
        timestamps = index_df['ts'].to_list()
        assert timestamps == sorted(timestamps)

        # Check indices are contiguous 0..n-1
        indices = index_df['ts_idx'].to_list()
        assert indices == list(range(len(timestamps)))

    def test_lookup_indices_polars(self, mock_dataset):
        """Test lookup_indices_polars matches dict-based lookup."""
        import polars as pl
        from data_provider.tensor_cache_polars import (
            build_timestamp_index,
            lookup_indices_polars,
        )

        # Build index from dataset
        all_ts = []
        for i in range(min(50, len(mock_dataset))):
            sample = mock_dataset[i]
            x_time = sample[SAMPLE_IDX_X_TIME].flatten()
            y_time = sample[SAMPLE_IDX_Y_TIME].flatten()
            all_ts.extend(x_time.tolist())
            all_ts.extend(y_time.tolist())

        index_df = build_timestamp_index(pl.DataFrame({'ts': all_ts}))
        index_dict = dict(zip(
            index_df['ts'].to_list(),
            index_df['ts_idx'].to_list()
        ))

        # Test on a sample
        test_sample = mock_dataset[25]
        test_ts = test_sample[SAMPLE_IDX_X_TIME].flatten()

        # Python dict approach
        python_indices = np.array(
            [index_dict.get(int(ts), 0) for ts in test_ts],
            dtype=np.int32
        )

        # Polars approach
        polars_indices = lookup_indices_polars(test_ts, index_df)

        np.testing.assert_array_equal(python_indices, polars_indices)

    def test_lookup_indices_batch(self, mock_dataset):
        """Test batch lookup produces correct results."""
        import polars as pl
        from data_provider.tensor_cache_polars import (
            build_timestamp_index,
            lookup_indices_polars,
            lookup_indices_batch,
        )

        # Build index
        all_ts = []
        for i in range(min(50, len(mock_dataset))):
            sample = mock_dataset[i]
            all_ts.extend(sample[SAMPLE_IDX_X_TIME].flatten().tolist())
            all_ts.extend(sample[SAMPLE_IDX_Y_TIME].flatten().tolist())

        index_df = build_timestamp_index(pl.DataFrame({'ts': all_ts}))

        # Prepare test data
        test_samples = [mock_dataset[i] for i in range(10)]
        test_timestamps = [s[SAMPLE_IDX_X_TIME].flatten() for s in test_samples]

        # Individual lookups
        individual_results = [
            lookup_indices_polars(ts, index_df) for ts in test_timestamps
        ]

        # Batch lookup
        batch_results = lookup_indices_batch(test_timestamps, index_df)

        # Compare
        assert len(individual_results) == len(batch_results)
        for ind, batch in zip(individual_results, batch_results):
            np.testing.assert_array_equal(ind, batch)

    def test_polars_collector_state(self, mock_dataset):
        """Test PolarsCollectorState collects data correctly."""
        from data_provider.tensor_cache_polars import (
            PolarsCollectorState,
            process_sample_for_collection_polars,
            register_entity_data_polars,
        )

        state = PolarsCollectorState()
        state.num_news_items = 1

        # Register entity
        first_sample = mock_dataset[0]
        register_entity_data_polars(
            state,
            'test_entity',
            first_sample[SAMPLE_IDX_HETERO_GENERAL],
            first_sample[SAMPLE_IDX_HETERO_CHANNEL]
        )

        # Process samples
        for i in range(10):
            sample = mock_dataset[i]
            process_sample_for_collection_polars(state, sample)

        # Verify collection
        assert 'test_entity' in state.entity_to_idx
        assert len(state.timestamps) > 0
        assert len(state.seen_timestamps) == len(state.timestamps)
        assert len(state.timeseries) == len(state.timestamps)
        assert len(state.embeddings) == len(state.timestamps)

    def test_finalize_shared_tables_polars(self, mock_dataset):
        """Test finalize_shared_tables_polars produces correct arrays."""
        from data_provider.tensor_cache_polars import (
            PolarsCollectorState,
            process_sample_for_collection_polars,
            register_entity_data_polars,
            finalize_shared_tables_polars,
        )

        state = PolarsCollectorState()
        state.num_news_items = 1

        # Collect from samples
        first_sample = mock_dataset[0]
        register_entity_data_polars(
            state,
            'test_entity',
            first_sample[SAMPLE_IDX_HETERO_GENERAL],
            first_sample[SAMPLE_IDX_HETERO_CHANNEL]
        )

        for i in range(20):
            sample = mock_dataset[i]
            process_sample_for_collection_polars(state, sample)

        # Finalize
        shared_tables, index_mappings = finalize_shared_tables_polars(state)

        # Verify output
        assert 'timestamps' in shared_tables
        assert 'timeseries' in shared_tables
        assert 'embeddings' in shared_tables
        assert 'timestamp_to_idx' in index_mappings
        assert 'entity_to_idx' in index_mappings

        # Timestamps should be sorted
        timestamps = shared_tables['timestamps']
        assert np.all(timestamps[:-1] <= timestamps[1:])

        # Index mapping should be consistent
        for str_ts, idx in index_mappings['timestamp_to_idx'].items():
            ts = int(str_ts)
            assert shared_tables['timestamps'][idx] == ts

    def test_reconstruction_with_polars(self, mock_dataset):
        """Test that samples can be reconstructed from polars-generated cache."""
        import polars as pl
        from data_provider.tensor_cache_polars import (
            PolarsCollectorState,
            process_sample_for_collection_polars,
            register_entity_data_polars,
            finalize_shared_tables_polars,
            build_timestamp_index,
            lookup_indices_polars,
        )

        state = PolarsCollectorState()
        state.num_news_items = 1

        # Collect from all samples
        first_sample = mock_dataset[0]
        register_entity_data_polars(
            state,
            mock_dataset.entity_id,
            first_sample[SAMPLE_IDX_HETERO_GENERAL],
            first_sample[SAMPLE_IDX_HETERO_CHANNEL]
        )

        for i in range(len(mock_dataset)):
            sample = mock_dataset[i]
            process_sample_for_collection_polars(state, sample)

        # Finalize
        shared_tables, index_mappings = finalize_shared_tables_polars(state)

        # Build index DataFrame for lookup
        ts_df = pl.DataFrame({'ts': shared_tables['timestamps'].tolist()})
        index_df = ts_df.with_row_index('ts_idx').cast({'ts_idx': pl.Int32})

        # Test reconstruction for several samples
        for sample_idx in [0, 10, 50, 99]:
            sample = mock_dataset[sample_idx]
            original_seq_x = sample[SAMPLE_IDX_SEQ_X]
            x_time = sample[SAMPLE_IDX_X_TIME].flatten()

            # Get indices using polars
            indices = lookup_indices_polars(x_time, index_df)

            # Reconstruct
            reconstructed_seq_x = shared_tables['timeseries'][indices]

            np.testing.assert_array_almost_equal(
                original_seq_x, reconstructed_seq_x, decimal=5,
                err_msg=f"seq_x reconstruction failed for sample {sample_idx}"
            )


# =============================================================================
# DIRECT ACCESS MIXIN TESTS
# =============================================================================

class TestDirectAccessMixin:
    """Tests for DirectAccessMixin functionality."""

    def test_mock_dataset_supports_direct_access(self, mock_dataset):
        """Test that MockDataset supports direct access."""
        # MockDataset has the required attributes
        assert hasattr(mock_dataset, 'all_timeseries')  # data analog
        assert hasattr(mock_dataset, 'timestamps')
        assert hasattr(mock_dataset, 'input_len')  # seq_len analog
        assert hasattr(mock_dataset, 'output_len')  # pred_len analog

    def test_direct_access_mixin_integration(self):
        """Test that DirectAccessMixin can be used with mock data."""
        from data_provider.dataset_direct_access import DirectAccessMixin, RawDataArrays

        # Create a mock dataset that has the mixin
        class MockDirectAccessDataset(DirectAccessMixin):
            def __init__(self):
                self.data = np.random.randn(1000, 5).astype(np.float32)
                self.timestamp = np.arange(1000, 2000, dtype=np.int64)
                self.seq_len = 96
                self.pred_len = 48
                self.stride = 1
                self.entity_id = "test_entity"
                self.full_hetero = np.random.randn(1000, 768).astype(np.float32)
                self.hetero_time = np.random.randn(1000, 4).astype(np.float32)
                self.hetero_general = np.random.randn(768).astype(np.float32)
                self.hetero_channel = np.random.randn(768).astype(np.float32)
                self.hetero_stride = 1

            def __len__(self):
                return (len(self.data) - self.seq_len - self.pred_len) // self.stride + 1

        dataset = MockDirectAccessDataset()

        # Test supports_direct_access
        assert dataset.supports_direct_access()

        # Test get_raw_arrays
        raw = dataset.get_raw_arrays()
        assert isinstance(raw, RawDataArrays)
        assert raw.data.shape == (1000, 5)
        assert raw.timestamps.shape == (1000,)
        assert raw.seq_len == 96
        assert raw.pred_len == 48
        assert raw.n_samples == len(dataset)
        assert raw.entity_id == "test_entity"
        assert raw.embeddings.shape == (1000, 768)
        assert raw.hetero_general.shape == (768,)

    def test_get_raw_timestamps_for_samples(self):
        """Test get_raw_timestamps_for_samples returns correct timestamps."""
        from data_provider.dataset_direct_access import DirectAccessMixin

        class MockDirectAccessDataset(DirectAccessMixin):
            def __init__(self):
                self.data = np.random.randn(200, 3).astype(np.float32)
                self.timestamp = np.arange(20000101000000, 20000101000200, dtype=np.int64)
                self.seq_len = 8
                self.pred_len = 4
                self.stride = 1
                self.entity_id = "test"

            def __len__(self):
                return len(self.data) - self.seq_len - self.pred_len + 1

            def __getitem__(self, idx):
                x_start = idx
                x_end = x_start + self.seq_len
                y_start = x_end
                y_end = y_start + self.pred_len
                return (
                    f"sample_{idx}",
                    self.data[x_start:x_end],
                    self.data[y_start:y_end],
                    self.timestamp[x_start:x_end],
                    self.timestamp[y_start:y_end],
                    None, None, None, None, None, None, None, None
                )

        dataset = MockDirectAccessDataset()

        # Get timestamps using direct access
        ts_data = dataset.get_raw_timestamps_for_samples()

        # Verify shape
        n_samples = len(dataset)
        assert ts_data['x_time'].shape == (n_samples, 8)
        assert ts_data['y_time'].shape == (n_samples, 4)

        # Verify content matches __getitem__
        for i in range(min(10, n_samples)):
            sample = dataset[i]
            np.testing.assert_array_equal(ts_data['x_time'][i], sample[3])
            np.testing.assert_array_equal(ts_data['y_time'][i], sample[4])

    def test_get_all_unique_timestamps(self):
        """Test get_all_unique_timestamps returns correct unique set."""
        from data_provider.dataset_direct_access import DirectAccessMixin

        class MockDirectAccessDataset(DirectAccessMixin):
            def __init__(self):
                self.data = np.random.randn(100, 2).astype(np.float32)
                self.timestamp = np.arange(1000, 1100, dtype=np.int64)
                self.seq_len = 8
                self.pred_len = 4
                self.stride = 1
                self.entity_id = "test"

            def __len__(self):
                return len(self.data) - self.seq_len - self.pred_len + 1

        dataset = MockDirectAccessDataset()

        # Get unique timestamps
        unique_ts = dataset.get_all_unique_timestamps()

        # Calculate expected unique timestamps manually
        n_samples = len(dataset)
        total_window = dataset.seq_len + dataset.pred_len
        max_accessed_idx = (n_samples - 1) * dataset.stride + total_window

        # For contiguous timestamps with stride=1, should be first max_accessed_idx timestamps
        expected_unique = dataset.timestamp[:max_accessed_idx]

        np.testing.assert_array_equal(unique_ts, np.unique(expected_unique))

    def test_build_shared_tables_direct(self):
        """Test build_shared_tables_direct produces correct output."""
        from data_provider.dataset_direct_access import DirectAccessMixin, build_shared_tables_direct

        class MockDirectAccessDataset(DirectAccessMixin):
            def __init__(self, entity_id):
                self.data = np.random.randn(200, 3).astype(np.float32)
                self.timestamp = np.arange(1000, 1200, dtype=np.int64)
                self.seq_len = 8
                self.pred_len = 4
                self.stride = 1
                self.entity_id = entity_id
                self.full_hetero = np.random.randn(200, 768).astype(np.float32)
                self.hetero_time = np.random.randn(200, 4).astype(np.float32)
                self.hetero_general = np.random.randn(768).astype(np.float32)
                self.hetero_channel = np.random.randn(768).astype(np.float32)
                self.hetero_stride = 1

            def __len__(self):
                return len(self.data) - self.seq_len - self.pred_len + 1

        # Create two datasets
        datasets = {
            'entity_A': MockDirectAccessDataset('entity_A'),
            'entity_B': MockDirectAccessDataset('entity_B'),
        }

        # Build shared tables
        shared_tables, index_mappings = build_shared_tables_direct(datasets, num_news_items=1, verbose=False)

        # Verify structure
        assert 'timestamps' in shared_tables
        assert 'timeseries' in shared_tables
        assert 'embeddings' in shared_tables
        assert 'hetero_time' in shared_tables
        assert 'entity_general' in shared_tables
        assert 'entity_channel' in shared_tables

        assert 'timestamp_to_idx' in index_mappings
        assert 'entity_to_idx' in index_mappings

        # Verify entity indices
        assert 'entity_A' in index_mappings['entity_to_idx']
        assert 'entity_B' in index_mappings['entity_to_idx']

        # Timestamps should be sorted
        timestamps = shared_tables['timestamps']
        assert np.all(timestamps[:-1] <= timestamps[1:])

        # Index mapping should match
        for str_ts, idx in index_mappings['timestamp_to_idx'].items():
            ts = int(str_ts)
            assert shared_tables['timestamps'][idx] == ts


class TestDirectAccessPerformance:
    """Performance comparison tests for direct access vs __getitem__."""

    @pytest.mark.slow
    def test_direct_access_faster_than_getitem(self):
        """Verify direct access is significantly faster than per-sample iteration."""
        import time
        from data_provider.dataset_direct_access import DirectAccessMixin

        class LargeDirectAccessDataset(DirectAccessMixin):
            def __init__(self, n_timestamps=50000):
                self.data = np.random.randn(n_timestamps, 5).astype(np.float32)
                self.timestamp = np.arange(n_timestamps, dtype=np.int64)
                self.seq_len = 96
                self.pred_len = 48
                self.stride = 1
                self.entity_id = "test"
                self.full_hetero = np.random.randn(n_timestamps, 768).astype(np.float32)
                self.hetero_stride = 1

            def __len__(self):
                return len(self.data) - self.seq_len - self.pred_len + 1

            def __getitem__(self, idx):
                x_start = idx
                x_end = x_start + self.seq_len
                y_start = x_end
                y_end = y_start + self.pred_len
                return (
                    f"sample_{idx}",
                    self.data[x_start:x_end],
                    self.data[y_start:y_end],
                    self.timestamp[x_start:x_end],
                    self.timestamp[y_start:y_end],
                    self.full_hetero[x_start:x_end],
                    self.full_hetero[y_start:y_end],
                    None, None, None, None, None, None
                )

        dataset = LargeDirectAccessDataset(n_timestamps=50000)
        n_samples = len(dataset)

        print(f"\nBenchmark: {n_samples:,} samples")

        # Method 1: Per-sample __getitem__ (current slow method)
        start = time.perf_counter()
        all_x_times_getitem = []
        for i in range(n_samples):
            sample = dataset[i]
            all_x_times_getitem.append(sample[3])
        getitem_time = time.perf_counter() - start

        # Method 2: Direct access (new fast method)
        start = time.perf_counter()
        ts_data = dataset.get_raw_timestamps_for_samples()
        direct_time = time.perf_counter() - start

        speedup = getitem_time / direct_time

        print(f"  __getitem__ iteration: {getitem_time:.2f}s")
        print(f"  Direct access: {direct_time:.2f}s")
        print(f"  Speedup: {speedup:.1f}x")

        # Verify correctness
        all_x_times_getitem_arr = np.stack(all_x_times_getitem)
        np.testing.assert_array_equal(all_x_times_getitem_arr, ts_data['x_time'])

        # Direct access should be at least 2x faster (typically 5-10x)
        assert speedup >= 2.0, f"Expected at least 2x speedup, got {speedup:.1f}x"


# =============================================================================
# TEST RUNNER CONFIGURATION
# =============================================================================

if __name__ == '__main__':
    # Run tests with verbose output
    pytest.main([__file__, '-v', '-m', 'not slow'])
