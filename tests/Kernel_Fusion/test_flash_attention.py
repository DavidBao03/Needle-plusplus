import sys
sys.path.append('./python')
sys.path.append('./apps')
import numpy as np
import pytest
import torch
import itertools

import needle as ndl

_DEVICES = [pytest.param(ndl.cuda(),
    marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU"))]

import pytest
import numpy as np
import torch

# Parametrize for different devices
@pytest.mark.parametrize("device", _DEVICES, ids=["cuda"])
def test_flash_attention_forward_medium_seq(device):
    """Test flash attention with smaller sequence lengths."""
    Q, K, V = (
        np.random.rand(12, 16, 64, 64).astype(np.float32),
        np.random.rand(12, 16, 64, 64).astype(np.float32),
        np.random.rand(12, 16, 64, 64).astype(np.float32),
    )

    Q_ = ndl.Tensor(Q, device=ndl.cuda())
    K_ = ndl.Tensor(K, device=ndl.cuda())
    V_ = ndl.Tensor(V, device=ndl.cuda())

    T_Q = torch.tensor(Q, requires_grad=False)
    T_K = torch.tensor(K, requires_grad=False)
    T_V = torch.tensor(V, requires_grad=False)

    T_out = torch.nn.functional.scaled_dot_product_attention(T_Q, T_K, T_V)
    _out = ndl.flash_attention(Q_, K_, V_)

    assert np.allclose(T_out.detach().cpu().numpy(), _out.numpy(), rtol=0, atol=1e-02)


@pytest.mark.parametrize("device", _DEVICES, ids=["cuda"])
def test_flash_attention_forward_small_seq(device):
    """Test flash attention with larger feature dimensions."""
    Q, K, V = (
        np.random.rand(12, 16, 32, 32).astype(np.float32),
        np.random.rand(12, 16, 32, 32).astype(np.float32),
        np.random.rand(12, 16, 32, 32).astype(np.float32),
    )
    T_Q = torch.tensor(Q, requires_grad=False)
    T_K = torch.tensor(K, requires_grad=False)
    T_V = torch.tensor(V, requires_grad=False)

    Q_ = ndl.Tensor(Q, device=ndl.cuda())
    K_ = ndl.Tensor(K, device=ndl.cuda())
    V_ = ndl.Tensor(V, device=ndl.cuda())

    T_out = torch.nn.functional.scaled_dot_product_attention(T_Q, T_K, T_V)
    _out = ndl.flash_attention(Q_, K_, V_)

    assert np.allclose(T_out.detach().cpu().numpy(), _out.numpy(), rtol=0, atol=1e-02)

@pytest.mark.parametrize("device", _DEVICES, ids=["cuda"])
def test_flash_attention_forward_large_seq(device):
    """Test flash attention with larger feature dimensions."""
    Q, K, V = (
        np.random.rand(12, 16, 96, 96).astype(np.float32),
        np.random.rand(12, 16, 96, 96).astype(np.float32),
        np.random.rand(12, 16, 96, 96).astype(np.float32),
    )
    T_Q = torch.tensor(Q, requires_grad=False)
    T_K = torch.tensor(K, requires_grad=False)
    T_V = torch.tensor(V, requires_grad=False)

    Q_ = ndl.Tensor(Q, device=ndl.cuda())
    K_ = ndl.Tensor(K, device=ndl.cuda())
    V_ = ndl.Tensor(V, device=ndl.cuda())

    T_out = torch.nn.functional.scaled_dot_product_attention(T_Q, T_K, T_V)
    _out = ndl.flash_attention(Q_, K_, V_)

    assert np.allclose(T_out.detach().cpu().numpy(), _out.numpy(), rtol=0, atol=1e-02)

# @pytest.mark.parametrize("device", _DEVICES, ids=["cuda"])
# def test_flash_attention_forward_huge_seq(device):
#     """Test flash attention with larger feature dimensions."""
#     Q, K, V = (
#         np.random.rand(12, 16, 128, 128).astype(np.float32),
#         np.random.rand(12, 16, 128, 128).astype(np.float32),
#         np.random.rand(12, 16, 128, 128).astype(np.float32),
#     )
#     T_Q = torch.tensor(Q, requires_grad=False)
#     T_K = torch.tensor(K, requires_grad=False)
#     T_V = torch.tensor(V, requires_grad=False)

#     Q_ = ndl.Tensor(Q, device=ndl.cuda())
#     K_ = ndl.Tensor(K, device=ndl.cuda())
#     V_ = ndl.Tensor(V, device=ndl.cuda())

#     T_out = torch.nn.functional.scaled_dot_product_attention(T_Q, T_K, T_V)
#     _out = ndl.flash_attention(Q_, K_, V_)

#     assert np.allclose(T_out.detach().cpu().numpy(), _out.numpy(), rtol=0, atol=1e-02)


# @pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
# def test_flash_attention_forward_single_batch(device):
#     """Test flash attention with single batch and head."""
#     Q, K, V = (
#         np.random.rand(1, 1, 4, 8).astype(np.float32),
#         np.random.rand(1, 1, 4, 8).astype(np.float32),
#         np.random.rand(1, 1, 4, 8).astype(np.float32),
#     )
#     T_Q = torch.tensor(Q, requires_grad=False)
#     T_K = torch.tensor(K, requires_grad=False)
#     T_V = torch.tensor(V, requires_grad=False)

#     Q_ = ndl.Tensor(Q, device=device, dtype="float32")
#     K_ = ndl.Tensor(K, device=device)
#     V_ = ndl.Tensor(V, device=device)

#     T_out = torch.nn.functional.scaled_dot_product_attention(T_Q, T_K, T_V)
#     _out = ndl.flash_attention(Q_, K_, V_)

#     assert np.allclose(T_out.detach().cpu().numpy(), _out.numpy(), atol=1e-5)


# @pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
# def test_flash_attention_forward_randomized(device):
#     """Test flash attention with random tensor dimensions."""
#     batch_size = np.random.randint(1, 5)
#     num_heads = np.random.randint(1, 5)
#     seq_len = np.random.randint(1, 10)
#     dims = np.random.randint(1, 64)

#     Q = np.random.rand(batch_size, num_heads, seq_len, dims).astype(np.float32)
#     K = np.random.rand(batch_size, num_heads, seq_len, dims).astype(np.float32)
#     V = np.random.rand(batch_size, num_heads, seq_len, dims).astype(np.float32)

#     T_Q = torch.tensor(Q, requires_grad=False)
#     T_K = torch.tensor(K, requires_grad=False)
#     T_V = torch.tensor(V, requires_grad=False)

#     Q_ = ndl.Tensor(Q, device=device)
#     K_ = ndl.Tensor(K, device=device)
#     V_ = ndl.Tensor(V, device=device)

#     T_out = torch.nn.functional.scaled_dot_product_attention(T_Q, T_K, T_V)
#     _out = ndl.flash_attention(Q_, K_, V_)

#     assert np.allclose(T_out.detach().cpu().numpy(), _out.numpy(), atol=1e-5)


# @pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
# def test_flash_attention_forward_edge_case_zero(device):
#     """Test flash attention when inputs contain zeros."""
#     Q, K, V = (
#         np.zeros((2, 2, 4, 8), dtype=np.float32),
#         np.zeros((2, 2, 4, 8), dtype=np.float32),
#         np.zeros((2, 2, 4, 8), dtype=np.float32),
#     )
#     T_Q = torch.tensor(Q, requires_grad=False)
#     T_K = torch.tensor(K, requires_grad=False)
#     T_V = torch.tensor(V, requires_grad=False)

#     Q_ = ndl.Tensor(Q, device=device)
#     K_ = ndl.Tensor(K, device=device)
#     V_ = ndl.Tensor(V, device=device)

#     T_out = torch.nn.functional.scaled_dot_product_attention(T_Q, T_K, T_V)
#     _out = ndl.flash_attention(Q_, K_, V_)

#     assert np.allclose(T_out.detach().cpu().numpy(), _out.numpy(), atol=1e-5)

# @pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
# def test_flash_attention_with_mask(device):
#     """Test flash attention with masking."""
#     # Inputs
#     Q, K, V = (
#         np.random.rand(2, 2, 4, 8).astype(np.float32),
#         np.random.rand(2, 2, 4, 8).astype(np.float32),
#         np.random.rand(2, 2, 4, 8).astype(np.float32),
#     )

#     # Mask (random mask for valid and invalid positions)
#     mask = np.random.randint(0, 2, (2, 2, 4, 4)).astype(np.float32)
#     mask = np.where(mask == 0, -np.inf, 0.0)  # Convert to mask logits

#     # Torch tensors
#     T_Q = torch.tensor(Q, requires_grad=False)
#     T_K = torch.tensor(K, requires_grad=False)
#     T_V = torch.tensor(V, requires_grad=False)
#     T_mask = torch.tensor(mask, dtype=T_Q.dtype, requires_grad=False)

#     # Custom tensors
#     Q_ = ndl.Tensor(Q, device=device)
#     K_ = ndl.Tensor(K, device=device)
#     V_ = ndl.Tensor(V, device=device)
#     mask_ = ndl.Tensor(mask, device=device)

#     # Torch implementation with mask
#     T_out = torch.nn.functional.scaled_dot_product_attention(T_Q, T_K, T_V, attn_mask=T_mask)

#     # Custom implementation with mask
#     _out = ndl.flash_attention(Q_, K_, V_, mask=mask_)

#     assert np.allclose(T_out.detach().cpu().numpy(), _out.numpy(), atol=1e-5)


# @pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
# def test_flash_attention_causal_mask(device):
#     """Test flash attention with causal mask."""
#     Q, K, V = (
#         np.random.rand(2, 2, 4, 8).astype(np.float32),
#         np.random.rand(2, 2, 4, 8).astype(np.float32),
#         np.random.rand(2, 2, 4, 8).astype(np.float32),
#     )

#     # Causal mask
#     seq_len = Q.shape[2]
#     causal_mask = np.tril(np.ones((seq_len, seq_len), dtype=np.float32))  # Lower triangular
#     causal_mask = np.where(causal_mask == 0, -np.inf, 0.0)  # Convert to mask logits
#     causal_mask = causal_mask[None, None, :, :]  # Expand to [BATCH_SIZE, NUM_HEAD, SEQ_LEN, SEQ_LEN]
#     causal_mask = np.broadcast_to(causal_mask, (2, 2, seq_len, seq_len))

#     # Torch tensors
#     T_Q = torch.tensor(Q, requires_grad=False)
#     T_K = torch.tensor(K, requires_grad=False)
#     T_V = torch.tensor(V, requires_grad=False)
#     T_mask = torch.tensor(causal_mask, dtype=T_Q.dtype, requires_grad=False)

#     # Custom tensors
#     Q_ = ndl.Tensor(Q, device=device)
#     K_ = ndl.Tensor(K, device=device)
#     V_ = ndl.Tensor(V, device=device)
#     mask_ = ndl.Tensor(causal_mask, device=device)

#     # Torch implementation with causal mask
#     T_out = torch.nn.functional.scaled_dot_product_attention(T_Q, T_K, T_V, attn_mask=T_mask)
    
#     # Custom implementation with causal mask
#     _out = ndl.flash_attention(Q_, K_, V_, mask=mask_)

#     assert np.allclose(T_out.detach().cpu().numpy(), _out.numpy(), atol=1e-5)