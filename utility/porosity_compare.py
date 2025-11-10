
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

def plot_by_class(df_avg: pd.DataFrame, out_png: Path) -> None:
    """
    df_avg columns: class, split, mean_porosity
    Produces a single line plot (one line per split) of mean porosity vs. class.
    X-axis spans all classes from 0 to 20 (numeric, no % sign).
    """
    # Remove trailing % if present and cast to int
    df_avg = df_avg.copy()
    df_avg["class"] = df_avg["class"].astype(str).str.rstrip("%")

    # Force classes 0–20 in numeric order
    classes = [str(c) for c in range(0, 21)]
    splits = list(df_avg["split"].unique())

    plt.figure()
    for split in splits:
        y = []
        for c in classes:
            vals = df_avg[(df_avg["class"] == c) & (df_avg["split"] == split)]["mean_porosity"].values
            y.append(vals[0] if len(vals) else np.nan)
        plt.plot(classes, y, marker="o", label=split)

    plt.xlabel("Class")
    plt.ylabel("Average Porosity (%)")
    plt.title("Average Porosity by Class")
    plt.xticks(rotation=45)
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

    # Compute per-class averages
    df_avg = (df.groupby(["split","class"], as_index=False)["porosity"]
                .mean()
                .rename(columns={"porosity":"mean_porosity"}))
    avg_csv = args.outdir / "porosity_by_class.csv"
    df_avg.to_csv(avg_csv, index=False)

    # Plot
    plot_path = args.outdir / "porosity_by_class.png"
    plot_by_class(df_avg, plot_path)

    print(f"[ok] Wrote per-image CSV: {per_image_csv}")
    print(f"[ok] Wrote per-class CSV: {avg_csv}")
    print(f"[ok] Wrote plot: {plot_path}")

if __name__ == "__main__":
    main()
