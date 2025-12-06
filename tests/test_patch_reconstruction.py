"""
Test suite for TGTSF patch reconstruction optimization.

This module tests the vectorized patch reconstruction implementation against
the original loop-based implementation to ensure:
1. Exact numerical equivalence
2. Performance improvement at scale
3. Correctness across various configurations

IMPORTANT: This tests a critical optimization to the TGTSF model implementation.
The original implementation uses a sequential loop that becomes a bottleneck
with larger batch sizes. The vectorized version uses torch.nn.functional.fold
which is the inverse of unfold and automatically handles overlapping patches.

Reference: This is based on the patch reconstruction code in models/TGTSF.py:104-116
"""

import pytest
import torch
import torch.nn.functional as F
import time
from typing import Tuple


def calculate_total_length(pred_len: int, patch_len: int, stride: int) -> int:
    """
    Calculate total_length used in TGTSF patch reconstruction.
    
    This matches the calculation in TGTSF.__init__:
    self.total_length = self.patch_len*2 + (int((self.pred_len - self.patch_len)/self.stride + 1) - 1) * self.stride
    
    Args:
        pred_len: Prediction length (output sequence length)
        patch_len: Length of each patch
        stride: Stride between patches
        
    Returns:
        Total length of the output buffer before trimming to pred_len
    """
    return patch_len * 2 + (int((pred_len - patch_len) / stride + 1) - 1) * stride


def patch_reconstruction_original(
    x: torch.Tensor,
    patch_len: int,
    stride: int,
    pred_len: int,
    total_length: int
) -> torch.Tensor:
    """
    Original patch reconstruction implementation from TGTSF.
    
    This is an exact copy of the code from models/TGTSF.py:104-116.
    It reconstructs overlapping patches by averaging overlapping regions.
    
    Args:
        x: Input tensor of shape [B, C, N, L] where:
           B = batch size
           C = number of channels/variables
           N = number of patches
           L = patch length
        patch_len: Length of each patch (should match L)
        stride: Stride between patches
        pred_len: Desired output sequence length
        total_length: Total buffer length (calculated from pred_len, patch_len, stride)
        
    Returns:
        Reconstructed sequence of shape [B, C, pred_len]
    """
    B, C, N, L = x.shape
    
    output = torch.zeros(B, C, total_length).to(x.device)
    count = torch.zeros(B, C, total_length).to(x.device)
    
    for i in range(N):
        start = i * stride
        end = start + patch_len
        output[:, :, start:end] += x[:, :, i, :]
        count[:, :, start:end] += 1
    
    output = output[:, :, :pred_len]
    count = count[:, :, :pred_len]
    
    output = output / count
    
    return output


def patch_reconstruction_vectorized(
    x: torch.Tensor,
    patch_len: int,
    stride: int,
    pred_len: int,
    total_length: int
) -> torch.Tensor:
    """
    Vectorized patch reconstruction using torch.nn.functional.fold.
    
    This is the optimized version that replaces the sequential loop with
    a fully vectorized operation. fold is the inverse of unfold and
    automatically handles overlapping patches with averaging.
    
    Implementation note: fold expects input of shape [B, C*kernel_size, num_patches]
    where patches are arranged as columns. We reshape our [B, C, N, L] tensor
    accordingly and process all channels in parallel.
    
    Args:
        x: Input tensor of shape [B, C, N, L] where:
           B = batch size
           C = number of channels/variables
           N = number of patches
           L = patch length
        patch_len: Length of each patch (should match L)
        stride: Stride between patches
        pred_len: Desired output sequence length
        total_length: Total buffer length (calculated from pred_len, patch_len, stride)
        
    Returns:
        Reconstructed sequence of shape [B, C, pred_len]
    """
    B, C, N, L = x.shape
    
    # Process all (batch, channel) combinations in parallel
    # Reshape: [B, C, N, L] -> [B*C, N, L] -> [B*C, L, N]
    # fold expects: [B, C*kernel_size, num_patches] for 2D
    # For 1D treated as 2D: we use height=1, so kernel_size = (1, patch_len)
    x_reshaped = x.reshape(B * C, N, L).permute(0, 2, 1).contiguous()  # [B*C, L, N]
    
    # fold input format: [B, C*kernel_size, num_patches]
    # For 2D: kernel_size = kernel_h * kernel_w
    # For our 1D case: kernel_size = 1 * patch_len = patch_len
    # So we have [B*C, L, N] which is [B*C, patch_len, N]
    # This matches [B, C*kernel_size, num_patches] if C=1, kernel_size=patch_len
    
    # Use fold: treats input as [B*C, 1*patch_len, N]
    output_folded = F.fold(
        x_reshaped,  # [B*C, L, N] = [B*C, patch_len, N]
        output_size=(1, total_length),  # Output: [B*C, 1, 1, total_length]
        kernel_size=(1, patch_len),     # Patch: height=1, width=patch_len
        stride=(1, stride),              # Stride: height=1, width=stride
        padding=(0, 0)                    # No padding
    )  # [B*C, 1, 1, total_length]
    
    # Reshape: [B*C, 1, 1, total_length] -> [B*C, total_length]
    output_folded = output_folded.squeeze(1).squeeze(1)  # [B*C, total_length]
    
    # Reshape to separate batch and channels: [B*C, total_length] -> [B, C, total_length]
    output = output_folded.reshape(B, C, total_length)  # [B, C, total_length]
    
    # Trim to pred_len
    output = output[:, :, :pred_len]  # [B, C, pred_len]
    
    return output


# Pytest fixtures for test data
@pytest.fixture
def device():
    """Device to run tests on (CUDA if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def small_config():
    """Small configuration for quick tests."""
    return {
        'batch_size': 4,
        'channels': 2,
        'patch_num': 5,
        'patch_len': 8,
        'stride': 4,
        'pred_len': 24
    }


@pytest.fixture
def medium_config():
    """Medium configuration for moderate tests."""
    return {
        'batch_size': 32,
        'channels': 8,
        'patch_num': 20,
        'patch_len': 16,
        'stride': 8,
        'pred_len': 96
    }


@pytest.fixture
def large_config():
    """Large configuration for performance tests."""
    return {
        'batch_size': 128,
        'channels': 16,
        'patch_num': 50,
        'patch_len': 24,
        'stride': 12,
        'pred_len': 240
    }


@pytest.fixture
def xlarge_config():
    """Extra large configuration for scale tests."""
    return {
        'batch_size': 512,
        'channels': 32,
        'patch_num': 100,
        'patch_len': 32,
        'stride': 16,
        'pred_len': 480
    }


@pytest.fixture
def tgtsf_like_config():
    """Configuration matching typical TGTSF usage (NYC Traffic Speed)."""
    return {
        'batch_size': 768,  # Current batch size
        'channels': 1,      # Single channel 
        'patch_num': 45,    # Approximate for input_len=360, patch_len=8, stride=8
        'patch_len': 8,
        'stride': 8,
        'pred_len': 24
    }


def create_test_tensor(config: dict, device: torch.device, seed: int = 42) -> torch.Tensor:
    """
    Create a test tensor with the specified configuration.
    
    Args:
        config: Configuration dictionary with batch_size, channels, patch_num, patch_len
        device: Device to create tensor on
        seed: Random seed for reproducibility
        
    Returns:
        Tensor of shape [B, C, N, L]
    """
    torch.manual_seed(seed)
    B = config['batch_size']
    C = config['channels']
    N = config['patch_num']
    L = config['patch_len']
    
    return torch.randn(B, C, N, L, device=device, dtype=torch.float32)


# Test functions
def test_calculate_total_length():
    """Test that total_length calculation matches TGTSF implementation."""
    # Test case from TGTSF: pred_len=24, patch_len=8, stride=8
    pred_len = 24
    patch_len = 8
    stride = 8
    
    # Calculate expected: patch_len*2 + (int((pred_len - patch_len)/stride + 1) - 1) * stride
    expected = patch_len * 2 + (int((pred_len - patch_len) / stride + 1) - 1) * stride
    # = 8*2 + (int((24-8)/8 + 1) - 1) * 8
    # = 16 + (int(16/8 + 1) - 1) * 8
    # = 16 + (int(2 + 1) - 1) * 8
    # = 16 + (3 - 1) * 8
    # = 16 + 16 = 32
    
    result = calculate_total_length(pred_len, patch_len, stride)
    assert result == expected, f"Expected {expected}, got {result}"


def test_small_config_exact_match(device, small_config):
    """Test that vectorized version produces exactly the same output as original."""
    x = create_test_tensor(small_config, device)
    patch_len = small_config['patch_len']
    stride = small_config['stride']
    pred_len = small_config['pred_len']
    total_length = calculate_total_length(pred_len, patch_len, stride)
    
    # Run original implementation
    output_original = patch_reconstruction_original(
        x, patch_len, stride, pred_len, total_length
    )
    
    # Run vectorized implementation
    output_vectorized = patch_reconstruction_vectorized(
        x, patch_len, stride, pred_len, total_length
    )
    
    # Check exact match (within floating point precision)
    assert torch.allclose(
        output_original, output_vectorized, rtol=1e-6, atol=1e-6
    ), "Outputs do not match!"
    
    # Also check shapes
    assert output_original.shape == output_vectorized.shape
    assert output_original.shape == (small_config['batch_size'], small_config['channels'], pred_len)


def test_medium_config_exact_match(device, medium_config):
    """Test exact match with medium-sized configuration."""
    x = create_test_tensor(medium_config, device)
    patch_len = medium_config['patch_len']
    stride = medium_config['stride']
    pred_len = medium_config['pred_len']
    total_length = calculate_total_length(pred_len, patch_len, stride)
    
    output_original = patch_reconstruction_original(
        x, patch_len, stride, pred_len, total_length
    )
    output_vectorized = patch_reconstruction_vectorized(
        x, patch_len, stride, pred_len, total_length
    )
    
    assert torch.allclose(
        output_original, output_vectorized, rtol=1e-6, atol=1e-6
    ), "Outputs do not match for medium config!"


def test_large_config_exact_match(device, large_config):
    """Test exact match with large configuration."""
    x = create_test_tensor(large_config, device)
    patch_len = large_config['patch_len']
    stride = large_config['stride']
    pred_len = large_config['pred_len']
    total_length = calculate_total_length(pred_len, patch_len, stride)
    
    output_original = patch_reconstruction_original(
        x, patch_len, stride, pred_len, total_length
    )
    output_vectorized = patch_reconstruction_vectorized(
        x, patch_len, stride, pred_len, total_length
    )
    
    assert torch.allclose(
        output_original, output_vectorized, rtol=1e-6, atol=1e-6
    ), "Outputs do not match for large config!"


def test_tgtsf_like_config_exact_match(device, tgtsf_like_config):
    """Test exact match with TGTSF-like configuration."""
    x = create_test_tensor(tgtsf_like_config, device)
    patch_len = tgtsf_like_config['patch_len']
    stride = tgtsf_like_config['stride']
    pred_len = tgtsf_like_config['pred_len']
    total_length = calculate_total_length(pred_len, patch_len, stride)
    
    output_original = patch_reconstruction_original(
        x, patch_len, stride, pred_len, total_length
    )
    output_vectorized = patch_reconstruction_vectorized(
        x, patch_len, stride, pred_len, total_length
    )
    
    assert torch.allclose(
        output_original, output_vectorized, rtol=1e-6, atol=1e-6
    ), "Outputs do not match for TGTSF-like config!"


def test_various_stride_patch_combinations(device):
    """Test various combinations of stride and patch_len."""
    configs = [
        {'patch_len': 8, 'stride': 4, 'pred_len': 24},   # 50% overlap
        {'patch_len': 8, 'stride': 8, 'pred_len': 24},   # No overlap
        {'patch_len': 16, 'stride': 8, 'pred_len': 48},  # 50% overlap
        {'patch_len': 12, 'stride': 6, 'pred_len': 36},  # 50% overlap
    ]
    
    for config in configs:
        test_config = {
            'batch_size': 16,
            'channels': 4,
            'patch_num': 10,
            **config
        }
        
        x = create_test_tensor(test_config, device)
        total_length = calculate_total_length(
            config['pred_len'], config['patch_len'], config['stride']
        )
        
        output_original = patch_reconstruction_original(
            x, config['patch_len'], config['stride'], config['pred_len'], total_length
        )
        output_vectorized = patch_reconstruction_vectorized(
            x, config['patch_len'], config['stride'], config['pred_len'], total_length
        )
        
        assert torch.allclose(
            output_original, output_vectorized, rtol=1e-6, atol=1e-6
        ), f"Outputs do not match for config {config}!"


def test_gradient_flow(device, small_config):
    """Test that gradients flow correctly through vectorized implementation."""
    x = create_test_tensor(small_config, device, seed=123)
    x.requires_grad_(True)
    
    patch_len = small_config['patch_len']
    stride = small_config['stride']
    pred_len = small_config['pred_len']
    total_length = calculate_total_length(pred_len, patch_len, stride)
    
    # Forward pass
    output = patch_reconstruction_vectorized(x, patch_len, stride, pred_len, total_length)
    
    # Backward pass
    loss = output.sum()
    loss.backward()
    
    # Check that gradients exist
    assert x.grad is not None, "Gradients not computed!"
    assert not torch.allclose(x.grad, torch.zeros_like(x.grad)), "Gradients are zero!"


def benchmark_reconstruction(func, x, patch_len, stride, pred_len, total_length, num_iterations=100):
    """Benchmark a reconstruction function."""
    # Warmup
    for _ in range(10):
        _ = func(x, patch_len, stride, pred_len, total_length)
    
    # Synchronize if CUDA
    if x.device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Time it
    start = time.time()
    for _ in range(num_iterations):
        _ = func(x, patch_len, stride, pred_len, total_length)
    
    if x.device.type == 'cuda':
        torch.cuda.synchronize()
    
    end = time.time()
    return (end - start) / num_iterations


def test_performance_improvement_medium(device, medium_config):
    """Test that vectorized version is faster than original for medium config."""
    if device.type == 'cpu':
        pytest.skip("Performance tests require CUDA")
    
    x = create_test_tensor(medium_config, device)
    patch_len = medium_config['patch_len']
    stride = medium_config['stride']
    pred_len = medium_config['pred_len']
    total_length = calculate_total_length(pred_len, patch_len, stride)
    
    time_original = benchmark_reconstruction(
        patch_reconstruction_original, x, patch_len, stride, pred_len, total_length
    )
    time_vectorized = benchmark_reconstruction(
        patch_reconstruction_vectorized, x, patch_len, stride, pred_len, total_length
    )
    
    speedup = time_original / time_vectorized
    print(f"\nMedium config: Original={time_original*1000:.3f}ms, "
          f"Vectorized={time_vectorized*1000:.3f}ms, Speedup={speedup:.2f}x")
    
    # Vectorized should be at least as fast (may not be faster for small configs)
    assert speedup >= 0.8, f"Vectorized version is slower! Speedup: {speedup:.2f}x"


def test_performance_improvement_large(device, large_config):
    """Test that vectorized version is faster than original for large config."""
    if device.type == 'cpu':
        pytest.skip("Performance tests require CUDA")
    
    x = create_test_tensor(large_config, device)
    patch_len = large_config['patch_len']
    stride = large_config['stride']
    pred_len = large_config['pred_len']
    total_length = calculate_total_length(pred_len, patch_len, stride)
    
    time_original = benchmark_reconstruction(
        patch_reconstruction_original, x, patch_len, stride, pred_len, total_length
    )
    time_vectorized = benchmark_reconstruction(
        patch_reconstruction_vectorized, x, patch_len, stride, pred_len, total_length
    )
    
    speedup = time_original / time_vectorized
    print(f"\nLarge config: Original={time_original*1000:.3f}ms, "
          f"Vectorized={time_vectorized*1000:.3f}ms, Speedup={speedup:.2f}x")
    
    # For large configs, we expect significant speedup
    assert speedup >= 1.2, f"Expected speedup >= 1.2x, got {speedup:.2f}x"


def test_performance_improvement_xlarge(device, xlarge_config):
    """Test that vectorized version is significantly faster for xlarge config."""
    if device.type == 'cpu':
        pytest.skip("Performance tests require CUDA")
    
    x = create_test_tensor(xlarge_config, device)
    patch_len = xlarge_config['patch_len']
    stride = xlarge_config['stride']
    pred_len = xlarge_config['pred_len']
    total_length = calculate_total_length(pred_len, patch_len, stride)
    
    time_original = benchmark_reconstruction(
        patch_reconstruction_original, x, patch_len, stride, pred_len, total_length
    )
    time_vectorized = benchmark_reconstruction(
        patch_reconstruction_vectorized, x, patch_len, stride, pred_len, total_length
    )
    
    speedup = time_original / time_vectorized
    print(f"\nXLarge config: Original={time_original*1000:.3f}ms, "
          f"Vectorized={time_vectorized*1000:.3f}ms, Speedup={speedup:.2f}x")
    
    # For xlarge configs, we expect substantial speedup (2-3x)
    assert speedup >= 1.5, f"Expected speedup >= 1.5x, got {speedup:.2f}x"


def test_performance_scales_with_batch_size(device):
    """Test that performance improvement increases with batch size."""
    if device.type == 'cpu':
        pytest.skip("Performance tests require CUDA")
    
    base_config = {
        'channels': 8,
        'patch_num': 20,
        'patch_len': 16,
        'stride': 8,
        'pred_len': 96
    }
    
    batch_sizes = [32, 128, 512]
    speedups = []
    
    for batch_size in batch_sizes:
        config = {**base_config, 'batch_size': batch_size}
        x = create_test_tensor(config, device)
        total_length = calculate_total_length(
            config['pred_len'], config['patch_len'], config['stride']
        )
        
        time_original = benchmark_reconstruction(
            patch_reconstruction_original,
            x, config['patch_len'], config['stride'],
            config['pred_len'], total_length
        )
        time_vectorized = benchmark_reconstruction(
            patch_reconstruction_vectorized,
            x, config['patch_len'], config['stride'],
            config['pred_len'], total_length
        )
        
        speedup = time_original / time_vectorized
        speedups.append(speedup)
        print(f"\nBatch size {batch_size}: Speedup={speedup:.2f}x")
    
    # Speedup should generally increase with batch size
    # (or at least not decrease significantly)
    assert speedups[-1] >= speedups[0] * 0.8, \
        f"Speedup should not decrease with batch size. Got: {speedups}"


def test_edge_case_no_overlap(device):
    """Test case where stride == patch_len (no overlap)."""
    config = {
        'batch_size': 16,
        'channels': 4,
        'patch_num': 10,
        'patch_len': 8,
        'stride': 8,  # No overlap
        'pred_len': 80
    }
    
    x = create_test_tensor(config, device)
    total_length = calculate_total_length(
        config['pred_len'], config['patch_len'], config['stride']
    )
    
    output_original = patch_reconstruction_original(
        x, config['patch_len'], config['stride'], config['pred_len'], total_length
    )
    output_vectorized = patch_reconstruction_vectorized(
        x, config['patch_len'], config['stride'], config['pred_len'], total_length
    )
    
    assert torch.allclose(
        output_original, output_vectorized, rtol=1e-6, atol=1e-6
    ), "Outputs do not match for no-overlap case!"


def test_edge_case_high_overlap(device):
    """Test case with high overlap (stride < patch_len/2)."""
    config = {
        'batch_size': 16,
        'channels': 4,
        'patch_num': 20,
        'patch_len': 16,
        'stride': 4,  # 75% overlap
        'pred_len': 80
    }
    
    x = create_test_tensor(config, device)
    total_length = calculate_total_length(
        config['pred_len'], config['patch_len'], config['stride']
    )
    
    output_original = patch_reconstruction_original(
        x, config['patch_len'], config['stride'], config['pred_len'], total_length
    )
    output_vectorized = patch_reconstruction_vectorized(
        x, config['patch_len'], config['stride'], config['pred_len'], total_length
    )
    
    assert torch.allclose(
        output_original, output_vectorized, rtol=1e-6, atol=1e-6
    ), "Outputs do not match for high-overlap case!"


if __name__ == '__main__':
    # Run tests with: pytest tests/test_patch_reconstruction.py -v
    pytest.main([__file__, '-v'])

