# Standard library imports
from typing import Optional

# Third-party imports
import cvxpy as cp
import torch
from torch import Tensor

def min_norm_solution(
        u_desired: Tensor,
        h: Tensor, 
        lfh: Tensor, 
        lgh: Tensor, 
        alpha: Optional[float] = 1.0) -> Tensor:
    # Analytical solution for the minimum norm solution of the CBF problem

    '''
    u_desired: desired control input (N, udim)
    h: cbf (N)
    lfh: Lie derivative of h with respect to f (N)
    lgh: Lie derivative of h with respect to g (N, udim)
    alpha: regularization parameter (scalar)
    '''
    N, udim = u_desired.shape
    assert lgh.shape == (N, udim), f"lgh.shape must be (N, udim), but got {lgh.shape}"
    # assert lfh.shape == (N,), f"lfh.shape must be (N,), but got {lfh.shape}"
    # assert h.shape == (N,), f"h.shape must be (N,), but got {h.shape}"

    constraint = lfh + torch.einsum('ij,ij->i', lgh, u_desired) + alpha * h
    constraint_mask = constraint <= 0
    delta = - ( ( lfh + alpha * h + torch.einsum('ij,ij->i', lgh, u_desired) ) / (torch.norm(lgh, dim=-1) ** 2) ).unsqueeze(-1) * lgh
    u_out = u_desired + delta * constraint_mask.unsqueeze(-1)

    return u_out, constraint_mask

def min_norm_solution_cvx(
    u_desired: Tensor,
    h: Tensor, 
    lfh: Tensor, 
    lgh: Tensor, 
    alpha: Optional[float] = 1.0) -> Tensor:
    '''
    u_desired: desired control input (N, udim)
    h: cbf (N)
    lfh: Lie derivative of h with respect to f (N)
    lgh: Lie derivative of h with respect to g (N, udim)
    alpha: regularization parameter (scalar)
    '''
    N, udim = u_desired.shape
    device = u_desired.device
    dtype = u_desired.dtype

    assert lgh.shape == (N, udim), f"lgh.shape must be (N, udim), but got {lgh.shape}"
    assert lfh.shape == (N,), f"lfh.shape must be (N,), but got {lfh.shape}"
    assert h.shape == (N,), f"h.shape must be (N,), but got {h.shape}"
    
    u_desired = u_desired.detach().cpu().numpy()
    h = h.detach().cpu().numpy()
    lfh = lfh.detach().cpu().numpy()
    lgh = lgh.detach().cpu().numpy()

    u = cp.Variable((N, udim))
    objective = cp.Minimize(cp.sum_squares(u - u_desired))
    constraints = [
        lfh + cp.sum(cp.multiply(lgh, u), axis=1) >= -alpha * h,
    ]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    u_out = torch.from_numpy(u.value).to(device=device, dtype=dtype)
    return u_out

def min_norm_solution_with_slack(
        u_desired: Tensor,
        h: Tensor, 
        lfh: Tensor, 
        lgh: Tensor, 
        alpha: Optional[float] = 1.0,
        lam: Optional[float] = 1.0) -> Tensor:
    # Analytical solution for the minimum norm solution of the CBF problem

    '''
    u_desired: desired control input (N, udim) or (batch_size, num_points, udim)
    h: cbf (N) or (batch_size, num_points)
    lfh: Lie derivative of h with respect to f (N) or (batch_size, num_points)
    lgh: Lie derivative of h with respect to g (N, udim) or (batch_size, num_points, udim)
    alpha: regularization parameter (scalar)
    '''
    
    # Handle both 1D and 2D inputs
    if u_desired.dim() == 3:
        # Batched case: (batch_size, num_points, udim)
        batch_size, num_points, udim = u_desired.shape
        
        # Aggregate across points dimension
        # take mean across points, since all points belong to same pt cloud
        h_agg = torch.mean(h, dim=1)  # (batch_size,)
        lfh_agg = torch.mean(lfh, dim=1)  # (batch_size,)
        lgh_agg = torch.mean(lgh, dim=1)  # (batch_size, udim)
        u_desired_agg = torch.mean(u_desired, dim=1)  # (batch_size, udim)
        
        # Call the function with aggregated tensors
        u_out_agg, mask_agg, slack_agg = min_norm_solution_with_slack_1d(u_desired_agg, h_agg, lfh_agg, lgh_agg, alpha, lam)
        
        # Expand the output back to original shape
        u_out = u_out_agg.unsqueeze(1).expand(batch_size, num_points, udim)
        mask = mask_agg.unsqueeze(1).expand(batch_size, num_points)
        slack = slack_agg.unsqueeze(1).expand(batch_size, num_points)
        
        return u_out, mask, slack
        
    else:
        # Original 1D case
        return min_norm_solution_with_slack_1d(u_desired, h, lfh, lgh, alpha, lam)

def min_norm_solution_with_slack_1d(
        u_desired: Tensor,
        h: Tensor, 
        lfh: Tensor, 
        lgh: Tensor, 
        alpha: Optional[float] = 1.0,
        lam: Optional[float] = 1.0) -> Tensor:
    # Original implementation for 1D inputs
    '''
    u_desired: desired control input (N, udim)
    h: cbf (N)
    lfh: Lie derivative of h with respect to f (N)
    lgh: Lie derivative of h with respect to g (N, udim)
    alpha: regularization parameter (scalar)
    '''
    N, udim = u_desired.shape
    assert lgh.shape == (N, udim), f"lgh.shape must be (N, udim), but got {lgh.shape}"
    # assert lfh.shape == (N,), f"lfh.shape must be (N,), but got {lfh.shape}"
    # assert h.shape == (N,), f"h.shape must be (N,), but got {h.shape}"

    b = -lfh - alpha * h
    nominal = b - torch.sum(lgh * u_desired, dim=-1)

    # If nominal is negative, then we don't need slack since the nominal is already feasible
    # Otherwise, there must be slack
    mu = 2 * (nominal) / (torch.sum(lgh ** 2, dim=-1) + 1/lam + 1e-8)  # Add small epsilon
    slack = mu / (2 * lam)
    u_out = u_desired + ( mu.unsqueeze(-1)/2 * lgh ) * (nominal >= 0).unsqueeze(-1)
    slack = slack * (nominal >= 0).unsqueeze(-1)

    return u_out, (nominal >= 0), slack

def min_norm_solution_cvx_with_slack(
    u_desired: Tensor,
    h: Tensor, 
    lfh: Tensor, 
    lgh: Tensor, 
    alpha: Optional[float] = 1.0,
    lam: Optional[float] = 1.0) -> Tensor:
    '''
    u_desired: desired control input (N, udim)
    h: cbf (N)
    lfh: Lie derivative of h with respect to f (N)
    lgh: Lie derivative of h with respect to g (N, udim)
    alpha: regularization parameter (scalar)
    '''
    N, udim = u_desired.shape
    device = u_desired.device
    dtype = u_desired.dtype
    assert lgh.shape == (N, udim), f"lgh.shape must be (N, udim), but got {lgh.shape}"
    assert lfh.shape == (N,), f"lfh.shape must be (N,), but got {lfh.shape}"
    assert h.shape == (N,), f"h.shape must be (N,), but got {h.shape}"
    u_desired = u_desired.detach().cpu().numpy()
    h = h.detach().cpu().numpy()
    lfh = lfh.detach().cpu().numpy()
    lgh = lgh.detach().cpu().numpy()

    u = cp.Variable((N, udim))
    slack = cp.Variable((N,))
    objective = cp.Minimize(cp.sum_squares(u - u_desired) + lam*cp.sum_squares(slack))
    constraints = [
        lfh + cp.sum(cp.multiply(lgh, u), axis=1) >= -alpha * h - slack,
        slack >= 0,
    ]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    u_out = torch.from_numpy(u.value).to(device=device, dtype=dtype)
    return u_out

# Test the functions
# u_desired = torch.randn(10, 2)
# h = torch.randn(10)
# lfh = torch.randn(10)
# lgh = torch.randn(10, 2)

# u_out = min_norm_solution(u_desired, h, lfh, lgh)
# u_out_cvx = min_norm_solution_cvx(u_desired, h, lfh, lgh)

# assert torch.allclose(u_out, u_out_cvx)
# print("Test passed")

# Test the functions with slack
# u_desired = torch.randn(10, 2)
# h = torch.randn(10)
# lfh = torch.randn(10)
# lgh = torch.randn(10, 2)

# u_out, _ = min_norm_solution_with_slack(u_desired, h, lfh, lgh, lam=1.0)
# u_out_cvx = min_norm_solution_cvx_with_slack(u_desired, h, lfh, lgh, lam=1.0)

# assert torch.allclose(u_out, u_out_cvx)
# print("Test passed")

# %%
