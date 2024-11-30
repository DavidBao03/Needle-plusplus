import sys
sys.path.append('./python')
sys.path.append('./apps')
import numpy as np
import pytest
import torch
import itertools

import needle as ndl
from needle import ops

import pytest
import numpy as np
import torch
import time

# Performance Test with PyTorch vs Custom CPU Attention
@pytest.mark.parametrize("batch_size, num_heads, seq_len, dims", [
    (12, 16, 32, 32),
    (12, 16, 64, 64),
    (12, 16, 96, 96)
])
@pytest.mark.parametrize("device", [ndl.cuda()], ids=["cuda"])
def test_attention_cpu_vs_torch(batch_size, num_heads, seq_len, dims, device):
    """
    Compare performance and output between PyTorch and custom CPU-based attention.
    """
    # Generate random input tensors for Q, K, V, and mask
    Q = np.random.rand(batch_size, num_heads, seq_len, dims).astype(np.float32)
    K = np.random.rand(batch_size, num_heads, seq_len, dims).astype(np.float32)
    V = np.random.rand(batch_size, num_heads, seq_len, dims).astype(np.float32)

    # Convert inputs to PyTorch tensors
    Q_torch = torch.tensor(Q, requires_grad=False)
    K_torch = torch.tensor(K, requires_grad=False)
    V_torch = torch.tensor(V, requires_grad=False)

    Q_ = ndl.Tensor(Q, device=device)
    K_ = ndl.Tensor(K, device=device)
    V_ = ndl.Tensor(V, device=device)

    Q_N = ndl.Tensor(Q, device=device)
    K_N = ndl.Tensor(K, device=device)
    V_N = ndl.Tensor(V, device=device)

    start_time = time.perf_counter()
    batch_size, num_head, queries_len, q_dim = Q_N.shape
    _, _, keys_values_len, k_dim = K_N.shape
    _, _, _, v_dim = V_N.shape

    assert q_dim == k_dim == v_dim

    result = None
    probs = None

    Z = Q_N @ K_N / np.sqrt(q_dim)

    max_val = ndl.Tensor(
        Z.realize_cached_data().max(axis=3),
        device=Z.device,
        dtype=Z.dtype,
        requires_grad=False
    )

    max_val = max_val.reshape((*Z.shape[:-1], 1))
    max_val = max_val.broadcast_to(Z.shape)

    probs = ops.exp(Z - max_val)

    denom = probs.sum(axes=3)
    denom = denom.reshape((*Z.shape[:-1], 1))
    denom = denom.broadcast_to(Z.shape)

    probs = probs / denom
    
    result = probs @ V_N.transpose((2, 3))
    navie_time = time.perf_counter() - start_time

    # Custom CPU Attention timing
    start_time = time.perf_counter()
    custom_out = ndl.flash_attention(Q_, K_, V_)
    custom_time = time.perf_counter() - start_time

    # PyTorch Attention timing
    start_time = time.perf_counter()
    torch_out = torch.nn.functional.scaled_dot_product_attention(Q_torch, K_torch, V_torch)
    torch_time = time.perf_counter() - start_time

    # Compare results
    assert np.allclose(custom_out.numpy(), torch_out.detach().numpy(), atol=1e-5), "Outputs do not match!"

    # Print performance
    print(f"Batch Size: {batch_size}, Num Heads: {num_heads}, Seq Len: {seq_len}, Dims: {dims}")
    print(f"Navie Attention Time (GPU): {navie_time:.6f} seconds")
    print(f"Custom Attention Time (GPU): {custom_time:.6f} seconds")
    print(f"PyTorch Attention Time (GPU): {torch_time:.6f} seconds")