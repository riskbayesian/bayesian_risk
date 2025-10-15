#!/usr/bin/env python3
"""
make_videos_from_folders.py

Given a root directory, create one video per immediate sub-folder that contains images.
- Frames are sorted "naturally" (e.g., f2.png before f10.png).
- Frame size is inferred from the first image; later images are resized to match.
- Outputs MP4 videos with configurable FPS and codec.
- Skips sub-folders that contain no images.

Usage:
    python make_videos_from_folders.py \
        --root /path/to/root_dir \
        --fps 12 \
        --exts .png .jpg .jpeg \
        --codec mp4v \
        --out-dir /path/to/output_dir \
        --overwrite

Notes:
- Only immediate sub-folders are scanned by default (no recursion into nested sub-folders-of-sub-folders).
- If you want the root dir itself (without subfolders) to also become a video, pass --include-root.
"""
import argparse
import os
from pathlib import Path
import re
import sys
from typing import List, Tuple

import cv2
import numpy as np


def natural_key(s: str):
    """Sort helper: 'f10.png' > 'f2.png' in natural order."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def list_image_files(folder: Path, exts: Tuple[str, ...]) -> List[Path]:
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort(key=lambda p: natural_key(p.name))
    return files


def ensure_even_dims(w: int, h: int) -> Tuple[int, int]:
    # Some codecs prefer even dimensions; adjust by trimming a pixel if odd.
    if w % 2 != 0:
        w -= 1
    if h % 2 != 0:
        h -= 1
    return max(w, 2), max(h, 2)


def write_video_from_images(
    images: List[Path],
    out_path: Path,
    fps: int = 12,
    codec: str = "mp4v",
    verbose: bool = True,
):
    if not images:
        if verbose:
            print(f"[SKIP] No images to write: {out_path.parent.name}")
        return

    # Read first frame to get target size
    first = cv2.imread(str(images[0]))
    if first is None:
        raise RuntimeError(f"Failed to read first image: {images[0]}")
    h, w = first.shape[:2]
    w_even, h_even = ensure_even_dims(w, h)
    if (w_even, h_even) != (w, h):
        first = cv2.resize(first, (w_even, h_even), interpolation=cv2.INTER_AREA)
        w, h = w_even, h_even

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Set up writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(
            f"Failed to open VideoWriter for {out_path} with codec '{codec}'. "
            f"Try a different codec (e.g., 'avc1', 'mp4v', 'XVID')."
        )

    # Write frames
    count = 0
    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            if verbose:
                print(f"[WARN] Could not read image: {img_path}")
            continue
        if frame.shape[1] != w or frame.shape[0] != h:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        writer.write(frame)
        count += 1

    writer.release()
    if verbose:
        print(f"[OK] Wrote {count} frames → {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Create one video per image sub-folder.")
    ap.add_argument("--root", type=str, required=True, help="Root directory containing sub-folders of images.")
    ap.add_argument("--exts", type=str, nargs="+", default=[".png", ".jpg", ".jpeg"], help="Image extensions to include.")
    ap.add_argument("--fps", type=int, default=12, help="Frames per second for output videos.")
    ap.add_argument("--codec", type=str, default="mp4v", help="FourCC codec (e.g., mp4v, avc1, XVID).")
    ap.add_argument("--out-dir", type=str, default="", help="Directory to place the videos (default: alongside sub-folders).")
    ap.add_argument("--suffix", type=str, default="_out", help="Suffix for video filename (subfoldername + suffix + .mp4).")
    ap.add_argument("--include-root", action="store_true", help="Also make a video from images directly under --root.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing video files.")
    ap.add_argument("--quiet", action="store_true", help="Reduce logging.")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists() or not root.is_dir():
        print(f"[ERR] Root not found or not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    exts = tuple(e.lower() if e.startswith(".") else f".{e.lower()}" for e in args.exts)
    out_dir = Path(args.out_dir).resolve() if args.out_dir else None
    verbose = not args.quiet

    # Optionally include images directly under root
    if args.include_root:
        imgs = list_image_files(root, exts)
        if imgs:
            vid_name = f"{root.name}{args.suffix}.mp4"
            out_path = (out_dir / vid_name) if out_dir else (root.parent / vid_name)
            if out_path.exists() and not args.overwrite:
                if verbose:
                    print(f"[SKIP] Exists (use --overwrite): {out_path}")
            else:
                write_video_from_images(imgs, out_path, fps=args.fps, codec=args.codec, verbose=verbose)

    # Process each immediate sub-folder
    for sub in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
        imgs = list_image_files(sub, exts)
        if not imgs:
            if verbose:
                print(f"[SKIP] No images in {sub}")
            continue
        vid_name = f"{sub.name}{args.suffix}.mp4"
        if out_dir:
            out_path = out_dir / vid_name
        else:
            # default: place video at same level as sub-folder
            out_path = sub.parent / vid_name

        if out_path.exists() and not args.overwrite:
            if verbose:
                print(f"[SKIP] Exists (use --overwrite): {out_path}")
            continue

        try:
            write_video_from_images(imgs, out_path, fps=args.fps, codec=args.codec, verbose=verbose)
        except Exception as e:
            print(f"[ERR] Failed on {sub}: {e}", file=sys.stderr)

    if verbose:
        print("[DONE]")


if __name__ == "__main__":
    main()
