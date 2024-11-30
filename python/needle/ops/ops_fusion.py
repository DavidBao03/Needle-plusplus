from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *
from .ops_logarithmic import *

import needle as ndl
import numpy as np

from ..backend_selection import array_api, BACKEND 


class FlashAttention(TensorOp):
    def __init__(self):
        pass

    def compute(self, Q, K ,V, max_, mask):
        return array_api.attention(Q, K, V, max_, mask)
    

    def gradient(self, out_grad, node):
        # TODO: Implement the gradient computation for the FlashAttention operation
        pass

def flash_attention(Q, K, V, mask=None):
    # Handle optional mask
    B, nH, S, D = Q.shape
    if mask is None:
        mask = ndl.Tensor(np.zeros((B, nH, S, S)), device=Q.device, dtype=Q.dtype)
    max_ = ndl.Tensor(np.full((B, nH, S), -np.inf), device=Q.device, dtype=Q.dtype)
    return FlashAttention()(Q, K, V, max_, mask)

