#!/usr/bin/env python3
"""
Visualize fitted Bezier curves per row of pair_tensor_with_bezier.pt

Each subplot shows:
- 100 discrete probability bins (distance in [0, 0.5])
- the fitted Bezier curve (Bernstein basis with K control points)

Subplot title: "A → B" taken from the CSV row order.
"""

import argparse
import math
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# --------- Bernstein / Bezier utilities ---------
def bernstein_basis(num_ctrl: int, t: np.ndarray) -> np.ndarray:
    """
    B[i,k] = C(n,k) * t_i^k * (1-t_i)^(n-k), with n = num_ctrl-1
    t: [S], returns B: [S, num_ctrl]
    """
    n = num_ctrl - 1
    S = t.shape[0]
    B = np.empty((S, num_ctrl), dtype=np.float64)
    # Precompute binomial coefficients
    binom = [math.comb(n, k) for k in range(num_ctrl)]
    for k in range(num_ctrl):
        B[:, k] = binom[k] * (t**k) * ((1.0 - t) ** (n - k))
    return B

def bezier_eval(coeffs: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Evaluate Bezier curve with given control points (coeffs: [K]) at t: [S]
    Returns y(t): [S]
    """
    B = bernstein_basis(len(coeffs), t)   # [S,K]
    return B @ coeffs

# --------- Plotting utility ---------
def plot_pages(pairs, bins_all, coeffs_all, out_pdf=None, out_dir=None,
               ncols=5, nrows=5, curve_samples=400):
    """
    Create multi-page plots with up to (nrows*ncols) subplots per page.
    pairs: list[str] titles for each subplot
    bins_all: np.ndarray shape [N, 100]
    coeffs_all: np.ndarray shape [N, K]
    """
    N = len(pairs)
    per_page = ncols * nrows
    num_pages = (N + per_page - 1) // per_page

    # x for discrete bins (distance 0..0.5 for 100 bins)
    x_bins = np.linspace(0.0, 0.5, bins_all.shape[1], dtype=np.float64)
    # t for Bezier; we map to same distance domain by dist = 0.5 * t
    t = np.linspace(0.0, 1.0, curve_samples, dtype=np.float64)
    x_curve = 0.5 * t

    def save_current(fig, page_idx):
        if out_pdf is not None:
            out_pdf.savefig(fig)
        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
            fig.savefig(os.path.join(out_dir, f"bezier_page_{page_idx+1:02d}.png"), dpi=200)

    page_idx = 0
    for page in range(num_pages):
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3.2*ncols, 2.6*nrows))
        axes = np.atleast_2d(axes).reshape(nrows, ncols)

        for k in range(per_page):
            i = page*per_page + k
            ax = axes[k // ncols, k % ncols]
            if i >= N:
                ax.axis("off")
                continue

            # Data
            y_bins = bins_all[i]
            coeffs = coeffs_all[i]
            y_curve = bezier_eval(coeffs, t)

            # Plot
            ax.plot(x_bins, y_bins, linestyle="none", marker="o", markersize=2)
            ax.plot(x_curve, y_curve, linewidth=1.5)

            # Styling
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlim(0.0, 0.5)
            ax.set_xlabel("distance (m)")
            ax.set_ylabel("P(nothing bad occurs)")
            ax.set_title(pairs[i], fontsize=9)

            # Light grid
            ax.grid(True, linestyle="--", alpha=0.3)

        plt.tight_layout()
        save_current(fig, page_idx)
        page_idx += 1
        plt.close(fig)

# --------- Main ---------
def main():
    ap = argparse.ArgumentParser("Visualize Bezier fits for each object pair row.")
    ap.add_argument("--tensor", required=True, help="Path to pair_tensor_with_bezier.pt")
    ap.add_argument("--csv", required=True, help="CSV used to build the tensor (for row names).")
    ap.add_argument("--featA", type=int, default=1024, help="Feature dim of object A (default: 1024).")
    ap.add_argument("--featB", type=int, default=1024, help="Feature dim of object B (default: 1024).")
    ap.add_argument("--bins", type=int, default=100, help="Number of probability bins (default: 100).")
    ap.add_argument("--num_ctrl", type=int, required=True, help="Number of Bezier control points K appended per row.")
    ap.add_argument("--out_pdf", default="bezier_plots.pdf", help="Output multi-page PDF path.")
    ap.add_argument("--out_dir", default=None, help="Optional: also save per-page PNGs to this directory.")
    ap.add_argument("--ncols", type=int, default=5, help="Subplots per row.")
    ap.add_argument("--nrows", type=int, default=5, help="Subplot rows per page.")
    args = ap.parse_args()

    # Load tensor
    T = torch.load(args.tensor, map_location="cpu")
    if not isinstance(T, torch.Tensor) or T.ndim != 2:
        raise SystemExit("Expected a 2D torch.Tensor in the input .pt file.")
    N, M = T.shape

    # Index layout: [featA | featB | bins | coeffs]
    feat_total = args.featA + args.featB
    expected_M = feat_total + args.bins + args.num_ctrl
    if M != expected_M:
        raise SystemExit(f"Tensor shape {tuple(T.shape)} does not match "
                         f"featA+featB={feat_total}, bins={args.bins}, num_ctrl={args.num_ctrl} "
                         f"-> expected M={expected_M}")

    idx_bins_start = feat_total
    idx_bins_end   = feat_total + args.bins
    idx_coef_start = idx_bins_end
    idx_coef_end   = idx_coef_start + args.num_ctrl

    # Extract
    bins_all = T[:, idx_bins_start:idx_bins_end].numpy()
    coeffs_all = T[:, idx_coef_start:idx_coef_end].numpy()

    # Read CSV for titles
    df = pd.read_csv(args.csv)
    if len(df) != N:
        raise SystemExit(f"CSV rows ({len(df)}) != tensor rows ({N}).")
    if "manipulated_object" not in df.columns or "around_object" not in df.columns:
        raise SystemExit("CSV must contain 'manipulated_object' and 'around_object' columns.")
    pairs = [f"{a} → {b}" for a, b in zip(df["manipulated_object"], df["around_object"])]

    # Plot to PDF (and optional PNGs)
    pdf = PdfPages(args.out_pdf) if args.out_pdf else None
    try:
        plot_pages(pairs, bins_all, coeffs_all, out_pdf=pdf, out_dir=args.out_dir,
                   ncols=args.ncols, nrows=args.nrows)
    finally:
        if pdf is not None:
            pdf.close()
    print(f"Saved plots to {args.out_pdf}")
    if args.out_dir:
        print(f"Also saved per-page PNGs to: {args.out_dir}")

if __name__ == "__main__":
    main()
