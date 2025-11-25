#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys

import pandas as pd


def load_table(path: str) -> pd.DataFrame:
    """Load CSV or Excel spreadsheet."""
    ext = os.path.splitext(path)[1].lower()
    if ext in [".csv"]:
        return pd.read_csv(path)
    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported spreadsheet type: {ext}")


def main():
    parser = argparse.ArgumentParser(
        description="Clip a batch of videos based on a spreadsheet of start/end times."
    )
    parser.add_argument(
        "spreadsheet",
        help="Path to spreadsheet (CSV / XLSX) with columns for trial id, method, start, end.",
    )
    parser.add_argument(
        "video_dir",
        help="Directory containing the original mp4 files (720p_{trial}_{method}_sanitized.mp4).",
    )
    parser.add_argument(
        "output_dir",
        help="Directory to write the clipped videos. Filenames are kept the same.",
    )

    # Allow you to adjust column names if your sheet differs
    parser.add_argument("--trial-col", default="Trial Number", help="Column name for trial id.")
    parser.add_argument("--method-col", default="Method", help="Column name for method name.")
    parser.add_argument("--start-col", default="Start", help="Column name for start time (sec).")
    parser.add_argument("--end-col", default="End", help="Column name for end time (sec).")

    args = parser.parse_args()

    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)

    # Load spreadsheet
    try:
        df = load_table(args.spreadsheet)
    except Exception as e:
        print(f"Error loading spreadsheet: {e}", file=sys.stderr)
        sys.exit(1)

    # Make sure required columns exist
    for col in [args.trial_col, args.method_col, args.start_col, args.end_col]:
        if col not in df.columns:
            print(f"Missing required column '{col}' in spreadsheet.", file=sys.stderr)
            print(f"Available columns: {list(df.columns)}", file=sys.stderr)
            sys.exit(1)

    for idx, row in df.iterrows():
        trial_val = row[args.trial_col]
        method_val = row[args.method_col]
        start_val = row[args.start_col]
        end_val = row[args.end_col]

        # ---- NEW: skip any row with missing trial/method/start/end ----
        if any(pd.isna(v) for v in [trial_val, method_val, start_val, end_val]):
            print(f"[Row {idx}] Skipping because one of trial/method/start/end is NaN.")
            continue

        # Convert to proper types
        trial_id = str(trial_val)
        method = str(method_val)

        try:
            start = float(start_val)
            end = float(end_val)
        except ValueError:
            print(f"[Row {idx}] Invalid start/end times: {start_val}, {end_val}. Skipping.")
            continue

        if end <= start:
            print(f"[Row {idx}] end <= start ({end} <= {start}). Skipping.")
            continue

        input_filename = f"720p_{trial_id}_{method}_sanitized.mp4"
        input_path = os.path.join(args.video_dir, input_filename)
        output_path = os.path.join(args.output_dir, input_filename)  # same name, new folder

        if not os.path.exists(input_path):
            print(f"[Row {idx}] Input file not found: {input_path}. Skipping.")
            continue

        duration = end - start

        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(start),
            "-i", input_path,
            "-t", str(duration),
            "-c:v", "libx264",  # re-encode video
            "-preset", "veryfast",
            "-crf", "18",       # adjust quality if you want (lower = better)
            "-c:a", "copy",     # copy audio without re-encoding
            output_path,
        ]

        print(f"[Row {idx}] Clipping {input_filename}: {start}s -> {end}s")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[Row {idx}] ffmpeg failed for {input_filename}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
