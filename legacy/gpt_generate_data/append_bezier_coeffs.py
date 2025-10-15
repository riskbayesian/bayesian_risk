#!/usr/bin/env python3
"""
Fit monotone Bezier (Bernstein) coefficients to probability bins in a tensor and append them.

Input tensor shape: [N, M] where the last 100 dims are probability bins in [0, 1]
Output tensor shape: [N, M + K], with K Bezier control points appended per row.

Monotonicity: c[0] <= c[1] <= ... <= c[K-1]
(Optionally) Bounds: 0 <= c <= 1

t-grid: 100 uniform samples in [0, 1] so that t=0 maps to distance 0.0 and t=1 maps to distance 0.5.
"""

import argparse
import math
import numpy as np
import torch
import cvxpy as cp

def bernstein_basis(num_ctrl: int, t: np.ndarray) -> np.ndarray:
    """
    Build Bernstein basis matrix B of shape [len(t), num_ctrl]
    B[i, k] = C(n, k) * t_i^k * (1 - t_i)^(n - k), with n = num_ctrl - 1.
    """
    assert num_ctrl >= 2, "num_ctrl must be at least 2"
    n = num_ctrl - 1
    T = t.reshape(-1, 1)                     # [S,1]
    one_minus_T = 1.0 - T                    # [S,1]
    # Precompute binomial coefficients
    coeffs = np.array([math.comb(n, k) for k in range(num_ctrl)], dtype=np.float64)  # [K]
    # Powers
    Tk = np.stack([np.power(T, k).squeeze(1) for k in range(num_ctrl)], axis=1)      # [S,K]
    Tmk = np.stack([np.power(one_minus_T, n - k).squeeze(1) for k in range(num_ctrl)], axis=1)  # [S,K]
    B = (Tk * Tmk) * coeffs[None, :]                                              # [S,K]
    return B.astype(np.float64)

def fit_bezier_monotone(y: np.ndarray, num_ctrl: int, clamp01: bool = True, solver_order=("OSQP","ECOS","SCS")) -> np.ndarray:
    """
    Solve min_c ||B c - y||_2^2 s.t. c is nondecreasing (and optionally 0<=c<=1).
    y: [S] probability samples in [0,1]
    returns c: [num_ctrl]
    """
    S = y.shape[0]
    t = np.linspace(0.0, 1.0, S, dtype=np.float64)
    B = bernstein_basis(num_ctrl, t)          # [S, K]

    c = cp.Variable(num_ctrl)
    obj = cp.Minimize(cp.sum_squares(B @ c - y))
    cons = [c[1:] - c[:-1] >= 0]              # monotone nondecreasing
    if clamp01:
        cons += [c >= 0, c <= 1]

    prob = cp.Problem(obj, cons)

    # Try solvers in order
    last_err = None
    for s in solver_order:
        try:
            prob.solve(solver=getattr(cp, s), verbose=False, warm_start=True)
            if c.value is not None:
                return np.asarray(c.value, dtype=np.float64)
        except Exception as e:
            last_err = e
            continue

    # Fallback (no CVXPY solution): use sampled points as coarse controls and enforce monotone by cummax
    # Choose indices roughly evenly spaced across y
    idx = np.round(np.linspace(0, S-1, num_ctrl)).astype(int)
    c0 = y[idx].astype(np.float64)
    c0 = np.maximum.accumulate(c0)  # enforce nondecreasing
    if clamp01:
        c0 = np.clip(c0, 0.0, 1.0)
    return c0

def main():
    ap = argparse.ArgumentParser(description="Append monotone Bezier coefficients (via CVXPY QP) to a tensor.")
    ap.add_argument("--in_pt", required=True, help="Input .pt tensor (N x M) with last 100 dims = probability bins.")
    ap.add_argument("--out_pt", required=True, help="Output .pt path (N x (M+K)).")
    ap.add_argument("--num_ctrl", type=int, required=True, help="Number of Bezier control points K (>=2).")
    ap.add_argument("--bins", type=int, default=100, help="Number of probability bins at the end of each row (default: 100).")
    ap.add_argument("--no_clamp01", action="store_true", help="Do NOT clamp coefficients to [0,1].")
    ap.add_argument("--device", default="cpu", choices=["cpu","cuda"], help="Device to load/save tensor (computation happens on CPU).")
    args = ap.parse_args()

    if args.num_ctrl < 2:
        raise SystemExit("num_ctrl must be >= 2")

    # Load tensor
    tensor = torch.load(args.in_pt, map_location=args.device)
    if not isinstance(tensor, torch.Tensor):
        raise SystemExit("Loaded object is not a torch.Tensor.")
    if tensor.ndim != 2:
        raise SystemExit(f"Expected 2D tensor, got shape {tuple(tensor.shape)}")

    N, M = tensor.shape
    if M < args.bins:
        raise SystemExit(f"Tensor has {M} dims, which is less than bins={args.bins}.")
    print(f"Loaded tensor: shape={tuple(tensor.shape)}; appending K={args.num_ctrl} coefficients per row.")

    # Split features and probability bins
    feats = tensor[:, : M - args.bins]                 # [N, M-100]
    probs = tensor[:, M - args.bins : M]               # [N, 100]
    probs_np = probs.detach().cpu().numpy().astype(np.float64)

    # Fit per-row and collect coefficients
    coeffs = np.zeros((N, args.num_ctrl), dtype=np.float64)
    clamp01 = not args.no_clamp01

    for i in range(N):
        y = probs_np[i, :]
        c = fit_bezier_monotone(y, num_ctrl=args.num_ctrl, clamp01=clamp01)
        coeffs[i, :] = c

    coeffs_t = torch.from_numpy(coeffs.astype(np.float32))   # [N, K]
    out = torch.cat([tensor, coeffs_t.to(tensor.device)], dim=1)  # [N, M+K]

    torch.save(out, args.out_pt)
    print(f"Saved augmented tensor with coefficients: shape={tuple(out.shape)} -> {args.out_pt}")

if __name__ == "__main__":
    main()
