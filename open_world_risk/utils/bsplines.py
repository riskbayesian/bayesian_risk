from typing import Optional, Tuple
import time
import torch
from torch import Tensor

# TODO: Make this generalizable to any degree
cubic_bspline_matrix = (1/6) * torch.tensor([
    [-1, 3, -3, 1],
    [3, -6, 3, 0],
    [-3, 0, 3, 0],
    [1, 4, 1, 0]
])

# def cubic_bspline_basis_and_derivative(t):
#     """
#     Input:
#         t: (N,) fractional position in [0, 1)
#     Returns:
#         basis: (N, 4) cubic B-spline basis functions
#         deriv: (N, 4) analytic derivatives w.r.t t
#     """
#     # Compute basis functions and derivatives in closed form
#     # Ref: https://en.wikipedia.org/wiki/B-spline#Basis_functions_for_uniform_B-splines
#     basis = torch.stack([
#         (1 - t) ** 3 / 6,
#         (3 * t ** 3 - 6 * t ** 2 + 4) / 6,
#         (-3 * t ** 3 + 3 * t ** 2 + 3 * t + 1) / 6,
#         t ** 3 / 6
#     ], dim=-1)  # (N, 4)
#     deriv = torch.stack([
#         -0.5 * (1 - t) ** 2,
#         1.5 * t ** 2 - 2 * t,
#         -1.5 * t ** 2 + t + 0.5,
#         0.5 * t ** 2
#     ], dim=-1)  # (N, 4)
#     return basis, deriv

def eval_cubic_bspline_normalized(
        t: Tensor,
        coeffs: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Evaluate cubic uniform B-splines at normalized input.
    Args:
        t: (N,)
        coeffs: (N, 4)
    Returns:
        out: (N,)
    """
    N, M = coeffs.shape
    # assert t.shape == (N,), f"t.shape must be (N,), but got {t.shape}"
    # assert coeffs.shape == (N, M), f"coeffs.shape must be (N, M), but got {coeffs.shape}"
    # assert M == 4, f"M must be 4, but got {M}"

    # Create a tensor [t^3, t^2, t, 1]
    powers = torch.flip(torch.arange(4, device=t.device, dtype=t.dtype), dims=[0])
    t_tensor = t[..., None] ** powers[None, :]      # (N, 4)

    # Create a tensor [3t^2, 2t, 1, 0]
    dt_tensor = powers[None, :] * ( t[..., None] ** torch.relu(powers - 1)[None, :] )

    print(t_tensor)
    print(dt_tensor)
    print(coeffs)

    # eval_ = torch.einsum('ni, ij, nj->n', t_tensor, cubic_bspline_matrix.to(t.device), coeffs)
    # deval_ = torch.einsum('ni, ij, nj->n', dt_tensor, cubic_bspline_matrix.to(t.device), coeffs)
    eval = torch.sum( ( t_tensor @  cubic_bspline_matrix.to(t.device) ) * ( coeffs ), dim=-1)
    print("eval", eval, eval.isnan().sum())
    print(coeffs @  cubic_bspline_matrix.to(t.device).T)
    # print("eval_", eval_)
    # print( (eval - eval_).abs().max())
    # assert torch.allclose(eval, eval_)
    deval = torch.sum( ( dt_tensor @  cubic_bspline_matrix.to(t.device) ) * ( coeffs ), dim=-1)
    # print("deval", deval, deval.isnan().sum())
    # assert torch.allclose(deval, deval_)
    return eval, deval

def eval_cubic_bspline(
        inputs: Tensor,
        coeffs: Tensor,
        knot_start: Optional[float] = 0.0,
        knot_spacing: Optional[float] = 1.0) -> Tuple[Tensor, Tensor]:
    """
    Evaluate cubic uniform B-splines at batch of inputs.
    Args:
        inputs: (N,)
        coeffs: (N, M)
    Returns:
        out: (N,)
    """
    N, M = coeffs.shape
    assert inputs.shape == (N,), f"inputs.shape must be (N,), but got {inputs.shape}"

    # Map each input to the left-most coefficient influencing it
    t = (inputs - knot_start) / knot_spacing  # (N,)
    i = torch.floor(t)  # (N,) left cell index
    f = t - i                  # (N,) fractional local position

    # Each output uses coeffs[..., i-1], i, i+1, i+2
    offset = torch.tensor([-1, 0, 1, 2], device=inputs.device)
    idxs = i[..., None] + offset[None, :]  # (N, 4)
    print("Indices", idxs)

    # Clamp indices to [0, M-1]
    idxs = idxs.clamp(0, M - 1).long()  # (N, 4)
    print("Indices", idxs)

    # Gather coefficients for each spline in the batch, shape (N, 4)
    # print(coeffs, idxs)
    coeffs_gathered = torch.gather(coeffs, 1, idxs)

    # Basis and derivative: (N, 4)
    eval, deval = eval_cubic_bspline_normalized(f, coeffs_gathered)
    return eval, deval / knot_spacing

# def eval_cubic_bspline(inputs, coeffs, knot_start=0.0, knot_spacing=1.0):
#     """
#     Evaluate cubic uniform B-splines at batch of inputs.
#     Args:
#         inputs: (N,)
#         coeffs: (N, M)
#     Returns:
#         out: (N,)
#     """
#     N, M = coeffs.shape
#     degree = 3

#     # Map each input to the left-most coefficient influencing it
#     t = (inputs - knot_start) / knot_spacing  # (N,)
#     i = torch.floor(t).long()  # (N,) left cell index
#     f = t - i                  # (N,) fractional local position

#     # Each output uses coeffs[..., i-1], i, i+1, i+2
#     idxs = torch.stack([i - 1, i, i + 1, i + 2], dim=-1)  # (N, 4)
#     # Clamp indices to [0, M-1]
#     idxs = idxs.clamp(0, M - 1)  # (N, 4)

#     # Gather coefficients for each spline in the batch, shape (N, 4)
#     coeffs_gathered = torch.gather(coeffs, 1, idxs)

#     # Basis and derivative: (N, 4)
#     basis, _ = cubic_bspline_basis_and_derivative(f)

#     # Weighted sum
#     return torch.sum(coeffs_gathered * basis, dim=-1)

# def eval_cubic_bspline_derivative(inputs, coeffs, knot_start=0.0, knot_spacing=1.0):
#     """
#     Compute derivative of cubic uniform B-splines w.r.t. input (analytic).
#     Args:
#         inputs: (N,)
#         coeffs: (N, M)
#     Returns:
#         out: (N,)
#     """
#     N, M = coeffs.shape
#     degree = 3

#     t = (inputs - knot_start) / knot_spacing  # (N,)
#     i = torch.floor(t).long()  # (N,)
#     f = t - i                  # (N,)

#     idxs = torch.stack([i - 1, i, i + 1, i + 2], dim=-1)  # (N, 4)
#     idxs = idxs.clamp(0, M - 1)  # (N, 4)

#     coeffs_gathered = torch.gather(coeffs, 1, idxs)

#     # Derivative of basis w.r.t. normalized t, then scale by 1/knot_spacing (chain rule)
#     _, dbasis = cubic_bspline_basis_and_derivative(f)
#     return torch.sum(coeffs_gathered * dbasis, dim=-1) / knot_spacing

# Test the functions
# inputs = torch.linspace(-5, 5, 1000000)
# coeffs = torch.tensor([0.0, 0.2, 0.66, 1.0])[None, :].repeat(1000000, 1)

# torch.cuda.synchronize()
# tnow = time.time()
# # eval = eval_cubic_bspline(inputs, coeffs)
# # deval = eval_cubic_bspline_derivative(inputs, coeffs)
# eval, deval = eval_cubic_bspline(inputs, coeffs)
# torch.cuda.synchronize()
# print(f"Time taken: {time.time() - tnow} seconds")

# from matplotlib import pyplot as plt
# fig, ax = plt.subplots(2, 1)
# # Plot the evaluated spline
# ax[0].plot(inputs, eval)

# # Draw the derivative of the curve
# ax[1].plot(inputs, deval, label='Analytic derivative')

# # Compute the finite differences from eval
# finite_diff = torch.diff(eval) / torch.diff(inputs)

# # Plot the finite differences
# ax[1].plot(inputs[1:], finite_diff, label='Finite differences')

# ax[1].legend()
# plt.show()
