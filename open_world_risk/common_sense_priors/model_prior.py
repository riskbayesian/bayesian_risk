# Standard library imports
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal, Optional

# Third-party imports
import faiss
import faiss.contrib.torch_utils
import numpy as np
from scipy.spatial import KDTree
import torch
import torch.nn.functional as F
from torch import vmap
import torch.func as func
from tqdm import tqdm
import math

# Local imports
from open_world_risk.utils.bsplines import eval_cubic_bspline
from open_world_risk.utils.cbf import min_norm_solution, min_norm_solution_with_slack
from open_world_risk.utils.covariance_utils import compute_cov
from open_world_risk.utils.scalar_fields import eval_sigmoid_field

def monotone_ncsp(z, dim=-1, slack=1.0, delta=0.0, eps=1e-12):
    d = F.softplus(z) + delta           # >= delta
    c = torch.cumsum(d, dim=dim)
    denom = slack + c.select(dim, -1).unsqueeze(dim)  # slack + total
    y = c / (denom + eps)
    return y

# Multi-head self-attention block with pre-layernorm and residual
class MultiHeadAttentionBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Pre-LN for attention
        self.norm = torch.nn.LayerNorm(d_model)
        self.Wq = torch.nn.Linear(d_model, d_model)
        self.Wk = torch.nn.Linear(d_model, d_model)
        self.Wv = torch.nn.Linear(d_model, d_model)
        self.Wo = torch.nn.Linear(d_model, d_model)
        self.attn_dropout = torch.nn.Dropout(dropout_p)
        self.out_dropout = torch.nn.Dropout(dropout_p)

        # FFN (square MLP) with separate pre-LN and residual
        self.ff_norm = torch.nn.LayerNorm(d_model)
        self.ff1 = torch.nn.Linear(d_model, d_model)
        self.ff_act = torch.nn.GELU()
        self.ff2 = torch.nn.Linear(d_model, d_model)
        self.ff_dropout = torch.nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # x: (B, S, d_model) where B is batch size and S is sequence length
        # Attention sublayer (Pre-LN + MHA + residual)
        attn_residual = x
        x = self.norm(x)

        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        B, S, _ = q.shape
        H = self.num_heads
        D = self.head_dim

        # reshape to (B, H, S, D)
        q = q.view(B, S, H, D).transpose(1, 2)
        k = k.view(B, S, H, D).transpose(1, 2)
        v = v.view(B, S, H, D).transpose(1, 2)

        # scaled dot-product attention per head -> (B, H, S, S)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
        if mask is not None:
            # Expect mask of shape (B, S) or (S,) -> broadcast to (B, 1, 1, S)
            if mask.dim() == 2:
                key_mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 1:
                key_mask = mask.view(1, 1, 1, S)
            else:
                key_mask = mask
            scores = scores.masked_fill(key_mask == 0, -1e9)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # attention output (B, H, S, D)
        attn_out = torch.matmul(attn_weights, v)

        # concat heads back to (B, S, H*D=d_model)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, H * D)
        attn_out = self.Wo(attn_out)
        attn_out = self.out_dropout(attn_out)

        # Residual add
        x = attn_residual + attn_out

        # FFN sublayer (Pre-LN + square MLP + residual)
        ffn_residual = x
        x = self.ff_norm(x)
        x = self.ff1(x)
        x = self.ff_act(x)
        x = self.ff_dropout(x)
        x = self.ff2(x)
        x = self.ff_dropout(x)
        x = ffn_residual + x

        return x, attn_weights

# The risk field takes in a set of clip coordinates and outputs the parameters of a n-degree B-spline or sigmoid field, as well as optionally covariances
class PriorModel(torch.nn.Module):
    def __init__(self,
                 clip_dim: int,
                 num_layers: int,
                 num_neurons: int,
                 projection_dim: int,
                 alphas_dim: int,
                 monotonic: bool = True,
                 num_layers_trans: int = 3,
                 num_heads: int = 8,
                 attn_dropout: float = 0.1):
        super().__init__()
        self.clip_dim = clip_dim
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.alphas_dim = alphas_dim
        self.projection_dim = projection_dim
        self.num_layers_trans = num_layers_trans
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout

        self.output_dim = self.alphas_dim
        self.monotonic = monotonic

        modules = []

        # Attention stack
        self.attn_layers = torch.nn.ModuleList([
            MultiHeadAttentionBlock(clip_dim, self.num_heads, dropout_p=self.attn_dropout)
            for _ in range(self.num_layers_trans)
        ])

        self.input_proj = torch.nn.Linear(clip_dim, projection_dim)

        modules = [torch.nn.Linear(2*projection_dim, num_neurons),  # Start with Linear
            torch.nn.LayerNorm(num_neurons),  # LayerNorm after Linear
            torch.nn.ReLU()]
        modules.extend([torch.nn.Linear(num_neurons, self.output_dim)])
        self.mlp = torch.nn.Sequential(*modules)

        # Initialize all weights and biases to xavier normal 
        for module in self.mlp:
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_normal_(module.weight)
                torch.nn.init.constant_(module.bias, 0.01)

    def forward(self, clip_coord1: torch.Tensor, clip_coord2: torch.Tensor) -> torch.Tensor:
        # Expect (B, D) inputs; build a 2-length sequence per batch: (B, 2, D)
        assert clip_coord1.dim() == 2 and clip_coord2.dim() == 2, "clip_coord tensors must be (B, D)"
        assert clip_coord1.shape == clip_coord2.shape, "clip_coord1 and clip_coord2 must have same shape"
        B, D = clip_coord1.shape
        seq = torch.stack([clip_coord1, clip_coord2], dim=1)  # (B, 2, D)

        # Positional encoding for 2 tokens (broadcast over batch)
        pos = torch.tensor([0.0, 0.01], device=seq.device, dtype=seq.dtype).view(1, 2, 1)
        seq = seq + pos

        # Apply attention stack with residual connections in batch mode
        x = seq
        for layer in self.attn_layers:
            x_residual = x
            x, _ = layer(x)
            x = x + x_residual

        # Project to lower dimension after attention: (B, 2, projection_dim)
        attended = self.input_proj(x)

        # Concatenate the two tokens' features per batch: (B, 2P) → MLP
        mlp_input = torch.cat([attended[:, 0, :], attended[:, 1, :]], dim=-1)  # (B, 2*P)

        output = self.mlp(mlp_input)  # (B, output_dim)

        output = torch.sigmoid(output)

        # THIS IS FOR SIGMOID FIELDS
        alphas = output[..., :self.alphas_dim]

        if self.monotonic:
            alphas = monotone_ncsp(alphas, dim=-1)

        
        return alphas
