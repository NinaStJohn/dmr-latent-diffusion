#!/usr/bin/env python3
"""
Porosity analysis for a single sweep directory.

Assumes your generated images are organized like:
  ROOT/
    poro_1.0000/
      *.png
    poro_0.9900/
      *.png
    ...

- Recursively scans subfolders under --dir
- Computes per-image porosity via luminance + threshold (percent)
- Extracts the *input* porosity value from the folder name (e.g., "poro_0.9800" -> 0.9800)
- Writes:
    - porosity_per_image.csv
    - porosity_by_bin.csv (mean+median per input porosity)
- Plots:
    - scatter: measured porosity (%) vs input porosity (fraction)
    - median line per input porosity

Example
-------
python porosity_bin.py --dir /path/to/sweep --outdir /path/to/out --threshold 0.5
"""
import argparse
from pathlib import Path
from typing import List, Optional
import re
import numpy as np
from PIL import Image
import pandas as pd
import os
import matplotlib

if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def calculate_porosity(img_path: Path, threshold: float = 0.5) -> float:
    """Return porosity (%) using luminance threshold in [0,1]."""
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        arr = np.asarray(im, dtype=np.float32) / 255.0
    Y = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    pores = (Y < threshold)
    return float(100.0 * pores.mean())


def list_images(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def infer_bin_folder(img_path: Path, root: Path) -> str:
    """
    Infer "bin" as the immediate subfolder under root.
    If image is directly under root, bin="root".
    """
    try:
        rel = img_path.relative_to(root)
    except ValueError:
        return img_path.parent.name
    parts = rel.parts
    return parts[0] if len(parts) > 1 else "root"


def extract_input_porosity(bin_name: str) -> Optional[float]:
    """
    Extract the FIRST float-like number from a folder/bin name.

    Examples:
      "poro_0.9800" -> 0.98
      "0.97"        -> 0.97
      "poro1.0"     -> 1.0

    Returns None if no float found.
    """
    s = str(bin_name)
    m = re.search(r"(\d+\.\d+|\d+)", s)
    if not m:
        return None
    return float(m.group(0))


def scan_folder(root: Path, threshold: float) -> pd.DataFrame:
    records = []
    for img in list_images(root):
        bin_name = infer_bin_folder(img, root)
        in_p = extract_input_porosity(bin_name)
        por = calculate_porosity(img, threshold=threshold)
        records.append(
            {
                "bin": str(bin_name),
                "input_porosity_frac": in_p,
                "path": str(img),
                "porosity_pct": por,
            }
        )
    if not records:
        return pd.DataFrame(columns=["bin", "input_porosity_frac", "path", "porosity_pct"])
    df = pd.DataFrame.from_records(records)
    return df


def plot_scatter(df: pd.DataFrame, out_png: Path) -> None:
    """
    Scatter: measured porosity (%) vs input porosity (fraction).
    Also draws median measured porosity per input porosity.
    """
    if df.empty:
        print("[warn] No data to plot; df is empty.")
        return

    df = df.copy()
    df = df.dropna(subset=["input_porosity_frac"])
    if df.empty:
        print("[warn] No bins with parseable input porosity values.")
        return

    # sort by input porosity
    df = df.sort_values("input_porosity_frac")

    # jitter for visibility
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.0008, 0.0008, size=len(df))

    x = df["input_porosity_frac"].to_numpy()
    y = df["porosity_pct"].to_numpy()

    plt.figure()
    plt.scatter(x + jitter, y, s=12, alpha=0.4)

    # median line per x
    med = df.groupby("input_porosity_frac")["porosity_pct"].median().reset_index()
    plt.plot(med["input_porosity_frac"], med["porosity_pct"], marker="o", linewidth=1.5, label="median")

    plt.xlabel("Input porosity (fraction)")
    plt.ylabel("Measured porosity (%)")
    plt.title("Measured porosity vs input porosity (scatter + median)")
    plt.legend()

    plt.gca().invert_xaxis()

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Porosity bin analysis for a single sweep directory.")
    ap.add_argument("--dir", required=True, type=Path, help="Root folder of generated images (subfolders = porosity bins).")
    ap.add_argument("--outdir", required=True, type=Path, help="Output directory for CSVs and plot.")
    ap.add_argument("--threshold", type=float, default=0.5, help="Luminance threshold in [0,1] (default: 0.5).")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    df = scan_folder(args.dir, threshold=args.threshold)

    per_image_csv = args.outdir / "porosity_per_image.csv"
    df.to_csv(per_image_csv, index=False)

    # stats per bin (input porosity)
    df_stats = (
        df.dropna(subset=["input_porosity_frac"])
          .groupby(["bin", "input_porosity_frac"])["porosity_pct"]
          .agg(["mean", "median", "count"])
          .reset_index()
          .rename(columns={"mean": "mean_porosity_pct", "median": "median_porosity_pct", "count": "n"})
          .sort_values("input_porosity_frac")
    )

    stats_csv = args.outdir / "porosity_by_bin.csv"
    df_stats.to_csv(stats_csv, index=False)

    plot_path = args.outdir / "porosity_by_input_scatter.png"
    plot_scatter(df, plot_path)

    print(f"[ok] Wrote per-image CSV: {per_image_csv}")
    print(f"[ok] Wrote per-bin CSV:   {stats_csv}")
    print(f"[ok] Wrote plot:          {plot_path}")


if __name__ == "__main__":
    main()
