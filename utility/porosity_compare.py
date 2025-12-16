
#!/usr/bin/env python3
"""
Porosity analysis for two folders (real vs. samples).

- Recursively scans class subfolders under each root.
- Computes per-image porosity via luminance + threshold.
- Writes two CSVs: per-image porosity, and per-class averages.
- Plots a line chart of average porosity by class for real vs. samples.

Example
-------
python porosity_compare.py     --real_dir /path/to/real     --sample_dir /path/to/samples     --outdir /path/to/out     --threshold 0.5
"""
import argparse
from pathlib import Path
from typing import List
import numpy as np
from PIL import Image
import pandas as pd
import os
import matplotlib
if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def calculate_porosity(img_path: Path, threshold: float = 0.5) -> float:
    """
    Porosity (percent) using luminance + fixed threshold:
      - load image
      - convert to RGB
      - luminance Y = 0.299 R + 0.587 G + 0.114 B (scaled to [0,1])
      - pores = (Y < threshold)
      - porosity = mean(pores) * 100
    """
    with Image.open(img_path) as im:
        im = im.convert('RGB')  # ensure 3 channels
        arr = np.asarray(im, dtype=np.float32) / 255.0  # [H,W,3] in [0,1]
    Y = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]  # [H,W]
    pores = (Y < threshold)
    porosity = 100.0 * (pores.sum() / pores.size)
    return float(porosity)

def list_images(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]

def infer_class(img_path: Path, root: Path) -> str:
    """
    Infer class as the immediate subfolder under root.
    If the image is directly under root, use 'root' as the class.
    """
    try:
        rel = img_path.relative_to(root)
    except ValueError:
        # Fallback in odd path cases
        return img_path.parent.name
    parts = rel.parts
    return parts[0] if len(parts) > 1 else "root"

def scan_folder(root: Path, split_name: str, threshold: float) -> pd.DataFrame:
    records = []
    for img in list_images(root):
        cls = infer_class(img, root)
        por = calculate_porosity(img, threshold=threshold)
        records.append({
            "split": split_name,
            "class": str(cls),
            "path": str(img),
            "porosity": por
        })
    if not records:
        return pd.DataFrame(columns=["split","class","path","porosity"])
    return pd.DataFrame.from_records(records)

from typing import Optional
import re

def _normalize_class_to_int(cls_str: str) -> Optional[int]:
    """
    Extract the first integer from a class-like string.
    Examples:
      'class_0'  -> 0
      '0_0%'     -> 0
      '12%'      -> 12
      '7'        -> 7
    Returns None if no integer is found.
    """
    m = re.search(r"\d+", str(cls_str))
    return int(m.group(0)) if m else None


def plot_by_class(df: pd.DataFrame, out_png: Path) -> None:
    """
    df columns: split, class, path, porosity

    Produces, for each split:
      - a line plot of median porosity vs. class
      - a scatter of individual-image porosities per class (with slight jitter)
    so you can see spread and potential outliers per class.
    """
    if df.empty:
        print("[warn] No data to plot; df is empty.")
        return

    df = df.copy()
    df["class_num"] = df["class"].apply(_normalize_class_to_int)

    # Drop rows we couldn't parse
    n_before = len(df)
    df = df.dropna(subset=["class_num"])
    if df.empty:
        print("[warn] No plottable classes after normalization.")
        return
    if len(df) < n_before:
        print(f"[info] Dropped {n_before - len(df)} rows with unparseable class labels.")

    df["class_num"] = df["class_num"].astype(int)

    classes = sorted(df["class_num"].unique().tolist())
    splits = list(df["split"].unique())

    # For reproducible jitter
    rng = np.random.default_rng(0)

    plt.figure()

    for split in splits:
        df_s = df[df["split"] == split]
        if df_s.empty:
            continue

        # Median line per class
        medians = (
            df_s.groupby("class_num")["porosity"]
                .median()
        )
        y = [medians.get(c, np.nan) for c in classes]

        if np.all(np.isnan(y)):
            continue

        # Plot median line and remember its color
        (line,) = plt.plot(classes, y, marker="o", label=f"{split} median")
        color = line.get_color()

        # Scatter per-image porosities with slight horizontal jitter
        for c in classes:
            vals = df_s[df_s["class_num"] == c]["porosity"].values
            if len(vals) == 0:
                continue
            jitter = rng.uniform(-0.15, 0.15, size=len(vals))
            xs = c + jitter
            plt.scatter(xs, vals, s=12, alpha=0.4, color=color, edgecolors="none")

    plt.xlabel("Class")
    plt.ylabel("Porosity (%)")
    plt.title("Porosity by Class (Median + Per-Image Scatter)")
    plt.xticks(classes, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()




def main():
    ap = argparse.ArgumentParser(description="Compare porosity by class for two folders (real vs. samples).")
    ap.add_argument("--real_dir", required=True, type=Path, help="Root folder of real images (with class subfolders).")
    ap.add_argument("--sample_dir", required=True, type=Path, help="Root folder of sample images (with class subfolders).")
    ap.add_argument("--outdir", required=True, type=Path, help="Output directory for CSVs and plot.")
    ap.add_argument("--threshold", type=float, default=0.5, help="Luminance threshold in [0,1] (default: 0.5).")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    df_real = scan_folder(args.real_dir, "real", threshold=args.threshold)
    df_samp = scan_folder(args.sample_dir, "samples", threshold=args.threshold)
    df = pd.concat([df_real, df_samp], ignore_index=True)

    # Save per-image CSV
    per_image_csv = args.outdir / "porosity_per_image.csv"
    df.to_csv(per_image_csv, index=False)

    # Compute per-class stats (mean + median) in a pandas-version-safe way
    grouped = df.groupby(["split", "class"])["porosity"]

    df_stats = (
        grouped
        .agg(["mean", "median"])
        .reset_index()
        .rename(columns={"mean": "mean_porosity", "median": "median_porosity"})
    )

    avg_csv = args.outdir / "porosity_by_class.csv"
    df_stats.to_csv(avg_csv, index=False)

    # Plot using per-image data so we can show spread + median per class
    plot_path = args.outdir / "porosity_by_class.png"
    plot_by_class(df, plot_path)



    print(f"[ok] Wrote per-image CSV: {per_image_csv}")
    print(f"[ok] Wrote per-class CSV: {avg_csv}")
    print(f"[ok] Wrote plot: {plot_path}")

if __name__ == "__main__":
    main()
