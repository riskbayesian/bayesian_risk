#!/usr/bin/env python3
"""
posterior_to_ply.py  (with control-point resizing to prior resolution)

- Loads control points (CP) as [K,H,W] or [H,W,K]
- Loads prior grayscale AS-IS (no normalization)
- If CP spatial size != prior size, resizes CP (bilinear) to match prior
- Multiplies CP per-pixel by prior
- Uses JSON (x,y,z,u,v) correspondences to pick CP vectors and compute radii
- Subsamples to <= max_points
- Saves:
    * points PLY (xyz + radius)
    * mesh PLY (union via voxel MC if scikit-image available, else stacked spheres)
"""

import os
import argparse
import json
import math
import random
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import numpy as np
from PIL import Image

def load_yaml(path: Path) -> dict:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_config(explicit: str | None = None) -> Tuple[dict, Path]:
    """
    Load YAML config. If explicit is provided, load that path.
    Else use CONFIG env or ./posterior_to_ply_config.yaml next to this script.
    Returns (cfg, cfg_dir).
    """
    if explicit:
        cfg_path = Path(explicit).expanduser().resolve()
    else:
        envp = os.environ.get("CONFIG")
        cfg_path = Path(envp).expanduser().resolve() if envp else (Path(__file__).parent / "posterior_to_ply_config.yaml").resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config YAML not found: {cfg_path}")
    return load_yaml(cfg_path), cfg_path.parent

def load_cp(cp_path: Path) -> np.ndarray:
    """Load CP array, coerce to [K,H,W] float32."""
    arr = np.load(cp_path)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D CP array; got {arr.shape}")
    cp = arr.astype(np.float32, copy=False)
    if cp.shape[0] >= max(cp.shape[1], cp.shape[2]):  # [K,H,W]
        return cp
    return np.transpose(cp, (2, 0, 1)).astype(np.float32, copy=False)  # [H,W,K] -> [K,H,W]

def load_prior_any(prior_path: Path) -> np.ndarray:
    """
    Load prior AS-IS (NO normalization).
    Recommended: .npy float32 in [0,1]. If an image is provided,
    it is read as float via 'F' mode (so uint8 becomes 0..255 float).
    """
    if prior_path.suffix.lower() == ".npy":
        prior = np.load(prior_path).astype(np.float32, copy=False)
    else:
        im = Image.open(prior_path).convert("F")
        prior = np.array(im, dtype=np.float32) / 255.
    return prior

def load_points_json(json_path: Path) -> List[dict]:
    with open(json_path, "r") as f:
        data = json.load(f)
    pts = data.get("points", [])
    if not isinstance(pts, list) or len(pts) == 0:
        raise ValueError("JSON must contain a non-empty 'points' list.")
    return pts

def uv_to_idx(u: float, v: float, mode: str = "nearest") -> Tuple[int, int]:
    if mode == "nearest":
        return int(round(u)), int(round(v))
    if mode == "floor":
        return int(math.floor(u)), int(math.floor(v))
    if mode == "ceil":
        return int(math.ceil(u)), int(math.ceil(v))
    raise ValueError(f"Unknown uv_round mode: {mode}")

def bernstein_basis(n: int, t: np.ndarray) -> np.ndarray:
    C = np.array([math.comb(n - 1, k) for k in range(n)], dtype=np.float64)
    t = t.astype(np.float64)
    ks = np.arange(n, dtype=np.float64)[None, :]
    t_col = t[:, None]
    M = C[None, :] * (t_col ** ks) * ((1.0 - t_col) ** (n - 1 - ks))
    return M.astype(np.float32)

def distance_at_threshold(cp_vec: np.ndarray,
                          t_samples: np.ndarray,
                          basis: np.ndarray,
                          threshold: float,
                          max_distance: float,
                          fallback_distance: float) -> float:
    if cp_vec[-1] < threshold:
        return float(fallback_distance)
    if cp_vec[0] > threshold:
        return 0.0
    F = basis @ cp_vec.astype(np.float32)  # [S]
    idx = np.argmin(np.abs(F - threshold))
    return float(t_samples[idx] * max_distance)
    # idx = np.argmax(F >= threshold)
    # if F[idx] < threshold:
    #     return float(fallback_distance)
    # if idx == 0:
    #     t_cross = t_samples[0]
    # else:
    #     t0, t1 = t_samples[idx - 1], t_samples[idx]
    #     f0, f1 = float(F[idx - 1]), float(F[idx])
    #     if f1 <= f0 + 1e-9:
    #         t_cross = t1
    #     else:
    #         w = (threshold - f0) / (f1 - f0)
    #         w = min(max(w, 0.0), 1.0)
    #         t_cross = t0 + w * (t1 - t0)
    # return float(t_cross * max_distance)

# ---- NEW: resize CP to match prior (bilinear) ----
def resize_cp_to_prior(cp: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    cp: [K,H,W] float32 → resize spatial dims to (target_h, target_w) with bilinear
    Tries torch (fast), then OpenCV, then PIL.
    """
    K, H, W = cp.shape
    if (H, W) == (target_h, target_w):
        return cp

    # Try PyTorch if available
    try:
        import torch
        import torch.nn.functional as F
        t = torch.from_numpy(cp).unsqueeze(0)            # [1,K,H,W]
        t = F.interpolate(t, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return t.squeeze(0).cpu().numpy().astype(np.float32, copy=False)
    except Exception:
        pass

    # Try OpenCV if available
    try:
        import cv2
        out = np.empty((K, target_h, target_w), dtype=np.float32)
        for k in range(K):
            out[k] = cv2.resize(cp[k], (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        return out
    except Exception:
        pass

    # Fallback: PIL per-channel
    out = np.empty((K, target_h, target_w), dtype=np.float32)
    for k in range(K):
        im = Image.fromarray(cp[k])
        out[k] = np.array(im.resize((target_w, target_h), resample=Image.BILINEAR), dtype=np.float32)
    return out

# ------------- PLY writers & mesh generators -------------

def save_ply_points(path: Path, xyz: np.ndarray, radius: np.ndarray) -> None:
    N = xyz.shape[0]
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {N}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float radius\n"
        "end_header\n"
    )
    with open(path, "w") as f:
        f.write(header)
        for i in range(N):
            x, y, z = xyz[i]
            r = radius[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r:.6f}\n")

def save_ply_mesh(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    V = verts.shape[0]
    F = faces.shape[0]
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {V}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        f"element face {F}\n"
        "property list uchar int vertex_index\n"
        "end_header\n"
    )
    with open(path, "w") as f:
        f.write(header)
        for i in range(V):
            x, y, z = verts[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
        for i in range(F):
            a, b, c = faces[i]
            f.write(f"3 {int(a)} {int(b)} {int(c)}\n")

def generate_uv_sphere(center: np.ndarray, radius: float,
                       n_lat: int = 12, n_lon: int = 24) -> Tuple[np.ndarray, np.ndarray]:
    cx, cy, cz = center
    verts = []
    for i in range(n_lat + 1):
        theta = math.pi * i / n_lat
        for j in range(n_lon):
            phi = 2 * math.pi * j / n_lon
            x = cx + radius * math.sin(theta) * math.cos(phi)
            y = cy + radius * math.sin(theta) * math.sin(phi)
            z = cz + radius * math.cos(theta)
            verts.append((x, y, z))
    verts = np.asarray(verts, dtype=np.float32)

    def idx(i, j):
        return i * n_lon + (j % n_lon)

    faces = []
    for i in range(n_lat):
        for j in range(n_lon):
            a = idx(i, j)
            b = idx(i, j + 1)
            c = idx(i + 1, j)
            d = idx(i + 1, j + 1)
            if i != 0:
                faces.append((a, c, b))
            if i != n_lat - 1:
                faces.append((b, c, d))
    faces = np.asarray(faces, dtype=np.int32)
    return verts, faces

def mesh_stacked_spheres(xyz: np.ndarray, radii: np.ndarray,
                         n_lat: int, n_lon: int,
                         max_spheres: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    if max_spheres is not None and xyz.shape[0] > max_spheres:
        idx = np.random.choice(xyz.shape[0], max_spheres, replace=False)
        xyz = xyz[idx]; radii = radii[idx]
    v_acc, f_acc, v_ofs = [], [], 0
    for i in range(xyz.shape[0]):
        v, f = generate_uv_sphere(xyz[i], float(radii[i]), n_lat=n_lat, n_lon=n_lon)
        f = f + v_ofs
        v_acc.append(v); f_acc.append(f)
        v_ofs += v.shape[0]
    if not v_acc:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32)
    return np.vstack(v_acc), np.vstack(f_acc)

def mesh_union_voxel_mc(xyz: np.ndarray, radii: np.ndarray,
                        voxel_size: float, margin: float,
                        iso_level: float = 0.0,
                        max_grid: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from skimage.measure import marching_cubes
    except Exception as e:
        raise RuntimeError("scikit-image is required for union_voxel_mc. Install with: pip install scikit-image") from e

    mins = xyz.min(axis=0) - (radii.max() + margin)
    maxs = xyz.max(axis=0) + (radii.max() + margin)
    span = maxs - mins

    dims = np.maximum((span / voxel_size).astype(int) + 1, 3)
    scale = 1.0
    if dims.max() > max_grid:
        scale = max_grid / dims.max()
        dims = np.maximum((dims.astype(float) * scale).astype(int), 3)

    nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])

    xs = np.linspace(mins[0], maxs[0], nx)
    ys = np.linspace(mins[1], maxs[1], ny)
    zs = np.linspace(mins[2], maxs[2], nz)

    cell_map: Dict[Tuple[int, int, int], List[int]] = {}

    def clamp_idx(a, lo, hi):
        return max(lo, min(hi, int(a)))

    for si in range(xyz.shape[0]):
        c = xyz[si]; r = float(radii[si])
        lo = ((c - r - mins) / (maxs - mins) * (np.array([nx - 1, ny - 1, nz - 1]))).astype(int)
        hi = ((c + r - mins) / (maxs - mins) * (np.array([nx - 1, ny - 1, nz - 1]))).astype(int)
        i0, j0, k0 = clamp_idx(lo[0], 0, nx - 1), clamp_idx(lo[1], 0, ny - 1), clamp_idx(lo[2], 0, nz - 1)
        i1, j1, k1 = clamp_idx(hi[0], 0, nx - 1), clamp_idx(hi[1], 0, ny - 1), clamp_idx(hi[2], 0, nz - 1)
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                for k in range(k0, k1 + 1):
                    cell_map.setdefault((i, j, k), []).append(si)

    sdf = np.empty((nx, ny, nz), dtype=np.float32)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                candidates = cell_map.get((i, j, k), [])
                if not candidates:
                    sdf[i, j, k] = 1e6
                    continue
                px = X[i, j, k]; py = Y[i, j, k]; pz = Z[i, j, k]
                dmin = 1e6
                for si in candidates:
                    dx = px - xyz[si, 0]; dy = py - xyz[si, 1]; dz = pz - xyz[si, 2]
                    d = math.sqrt(dx*dx + dy*dy + dz*dz) - float(radii[si])
                    if d < dmin:
                        dmin = d
                sdf[i, j, k] = dmin

    verts, faces, _, _ = marching_cubes(
        sdf.transpose(2, 1, 0), level=iso_level,
        spacing=(zs[1]-zs[0], ys[1]-ys[0], xs[1]-xs[0])
    )
    verts[:, 0] += mins[2]
    verts[:, 1] += mins[1]
    verts[:, 2] += mins[0]
    verts = verts[:, [2, 1, 0]].astype(np.float32)
    faces = faces.astype(np.int32)
    return verts, faces

# ------------------ Main ------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None, help="Path to YAML (or set CONFIG env).")
    args = ap.parse_args()
    cfg, _ = load_config(args.config)

    paths = cfg["paths"]
    cp_path = Path(paths["cp_npy"]).expanduser().resolve()
    prior_path = Path(paths["prior"]).expanduser().resolve()
    json_path = Path(paths["correspondences"]).expanduser().resolve()
    out_points_ply = Path(paths["out_points_ply"]).expanduser().resolve()
    out_mesh_ply = Path(paths["out_mesh_ply"]).expanduser().resolve()

    params = cfg.get("params", {})
    threshold = float(params.get("threshold", 0.1))
    fallback_distance = float(params.get("fallback_distance", 0.5))
    max_distance = float(params.get("max_distance", 1.0))
    max_points = int(params.get("max_points", 100000))
    samples = int(params.get("samples", 512))
    uv_round = str(params.get("uv_round", "nearest"))
    seed = int(params.get("seed", 42))

    mesh_cfg = cfg.get("mesh", {})
    method = str(mesh_cfg.get("method", "union_voxel_mc"))
    voxel_size = float(mesh_cfg.get("voxel_size", 0.01))
    margin = float(mesh_cfg.get("margin", 0.05))
    iso_level = float(mesh_cfg.get("iso_level", 0.0))
    max_grid = int(mesh_cfg.get("max_grid", 256))
    n_lat = int(mesh_cfg.get("sphere_n_lat", 12))
    n_lon = int(mesh_cfg.get("sphere_n_lon", 24))
    stacked_max_spheres = mesh_cfg.get("stacked_max_spheres", None)
    if stacked_max_spheres is not None:
        stacked_max_spheres = int(stacked_max_spheres)

    # Load arrays
    cp = load_cp(cp_path)                # [K,Hc,Wc]
    prior = load_prior_any(prior_path)   # [Hp,Wp]  (AS-IS)

    # Resize CP spatially if needed
    K, Hc, Wc = cp.shape
    Hp, Wp = prior.shape
    if (Hc, Wc) != (Hp, Wp):
        cp = resize_cp_to_prior(cp, Hp, Wp)

    # Multiply CP by prior per-pixel
    cp_scaled = cp * prior[None, :, :]

    # Precompute Bernstein basis
    t_samples = np.linspace(0.0, 1.0, num=max(2, samples), dtype=np.float32)
    basis = bernstein_basis(cp_scaled.shape[0], t_samples)  # [S,K]

    # Load points with (u,v)
    pts = load_points_json(json_path)

    # Compute per-point radii
    xyz_list, rad_list = [], []
    H, W = cp_scaled.shape[1], cp_scaled.shape[2]
    oob = 0
    for p in pts:
        try:
            x, y, z = float(p["x"]), float(p["y"]), float(p["z"])
            u, v = float(p["u"]), float(p["v"])
        except KeyError as e:
            raise ValueError(f"Missing key in a point entry: {e}")

        ix, iy = uv_to_idx(u, v, uv_round)
        if ix < 0 or iy < 0 or ix >= W or iy >= H:
            oob += 1
            continue

        cp_vec = cp_scaled[:, iy, ix]  # [K]
        dist = distance_at_threshold(cp_vec, t_samples, basis,
                                     threshold=threshold,
                                     max_distance=max_distance,
                                     fallback_distance=fallback_distance)
        xyz_list.append((x, y, z))
        rad_list.append(dist)

    if oob:
        print(f"[INFO] Skipped {oob} points outside image bounds (post-resize).")
    if not xyz_list:
        raise RuntimeError("No valid points after bounds check.")

    xyz = np.asarray(xyz_list, dtype=np.float32)
    radii = np.asarray(rad_list, dtype=np.float32)

    # Subsample
    N = xyz.shape[0]
    if N > max_points:
        random.seed(seed)
        idx = np.array(random.sample(range(N), max_points), dtype=np.int64)
        xyz = xyz[idx]; radii = radii[idx]

    # Save PLYs
    out_points_ply.parent.mkdir(parents=True, exist_ok=True)
    save_ply_points(out_points_ply, xyz, radii)
    print(f"[OK] Wrote points PLY: {out_points_ply}  (N={xyz.shape[0]})")

    if method == "union_voxel_mc":
        try:
            verts, faces = mesh_union_voxel_mc(
                xyz, radii, voxel_size=voxel_size, margin=margin,
                iso_level=iso_level, max_grid=max_grid
            )
            if verts.size == 0 or faces.size == 0:
                print("[WARN] Marching cubes returned empty mesh; falling back to stacked_spheres.")
                verts, faces = mesh_stacked_spheres(
                    xyz, radii, n_lat=n_lat, n_lon=n_lon, max_spheres=stacked_max_spheres
                )
        except Exception as e:
            print(f"[WARN] union_voxel_mc failed ({e}); using stacked_spheres fallback.")
            verts, faces = mesh_stacked_spheres(
                xyz, radii, n_lat=n_lat, n_lon=n_lon, max_spheres=stacked_max_spheres
            )
    else:
        verts, faces = mesh_stacked_spheres(
            xyz, radii, n_lat=n_lat, n_lon=n_lon, max_spheres=stacked_max_spheres
        )

    if verts.size == 0 or faces.size == 0:
        raise RuntimeError("Mesh generation produced an empty result.")
    save_ply_mesh(out_mesh_ply, verts, faces)
    print(f"[OK] Wrote mesh PLY:   {out_mesh_ply}  (V={verts.shape[0]}, F={faces.shape[0]})")

if __name__ == "__main__":
    main()
