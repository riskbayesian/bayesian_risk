import torch
from torch import Tensor
from typing import Tuple

def eval_sigmoid_field(
        inputs: Tensor,
        coeffs: Tensor,
        ) -> Tuple[Tensor, Tensor]:
    """
    Evaluate sigmoid field at batch of inputs.
    Args:
        inputs: (N,) or (batch_size, num_points) - maintains dimensionality
        coeffs: (N, 3) or (batch_size, num_points, 3) - coefficients per point
    Returns:
        out: (N,) or (batch_size, num_points) - same shape as inputs
    """
    # Handle both 1D and 2D inputs while maintaining dimensionality
    if inputs.dim() == 2:
        # inputs is (batch_size, num_points) 
        # coeffs is (batch_size, num_points, 3) - each point has its own coefficients
        batch_size, num_points = inputs.shape
        
        # Reshape inputs and coeffs for vectorized computation
        inputs_flat = inputs.view(-1)  # (batch_size * num_points,)
        coeffs_flat = coeffs.view(-1, 3)  # (batch_size * num_points, 3)
        
        # Compute sigmoid field for flattened inputs
        a, b, c = coeffs_flat[..., 0], coeffs_flat[..., 1], coeffs_flat[..., 2]
        
        # Add numerical stability checks
        b = torch.clamp(b, min=1e-6)
        
        argument = (inputs_flat + c) / b
        argument = torch.clamp(argument, min=-20, max=20)
        
        eval_flat = a * torch.sigmoid(-argument)
        eval_flat = eval_flat + 1e-6
        eval_flat = torch.clamp(eval_flat, min=1e-6, max=1e6)
        
        deval_flat = -a * torch.sigmoid(-argument) * (1 - torch.sigmoid(-argument)) / b
        deval_flat = torch.clamp(deval_flat, min=-1e6, max=1e6)
        
        # Reshape back to original dimensions
        eval = eval_flat.view(batch_size, num_points)
        deval = deval_flat.view(batch_size, num_points)
        
    else:
        # Original 1D case
        N, M = coeffs.shape
        assert inputs.shape == (N,), f"inputs.shape must be (N,), but got {inputs.shape}"
        assert coeffs.shape[-1] == 3, f"coeffs.shape must be (N, 3), but got {coeffs.shape}"

        # Evaluate the sigmoid field (a / (1 + exp( (x + b) / c)))
        a, b, c = coeffs[..., 0], coeffs[..., 1], coeffs[..., 2]

        # Add numerical stability checks
        b = torch.clamp(b, min=1e-6)
        
        argument = (inputs + c) / b
        argument = torch.clamp(argument, min=-20, max=20)
        
        eval = a * torch.sigmoid(-argument)
        eval = eval + 1e-6
        eval = torch.clamp(eval, min=1e-6, max=1e6)
        
        deval = -a * torch.sigmoid(-argument) * (1 - torch.sigmoid(-argument)) / b
        deval = torch.clamp(deval, min=-1e6, max=1e6)

    return eval, deval
