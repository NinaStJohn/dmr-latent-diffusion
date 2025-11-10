import os
import re
import subprocess
import argparse
from pathlib import Path

def normalize_class_name(name: str):
    """
    Normalize a class folder name like 'class_0', '0_0%', '1_4%' -> '0', '1', etc.
    Returns numeric string if found, else original name.
    """
    # remove prefix/suffix noise
    name = name.strip().lower()
    name = name.replace("class_", "").replace("%", "")
    # extract first integer in name
    match = re.search(r"\d+", name)
    return match.group(0) if match else name

def main():
    parser = argparse.ArgumentParser(description="Batch metric calculator for class folders.")
    parser.add_argument("--train_dir", required=True, help="Path to the root training (real) images directory.")
    parser.add_argument("--test_dir", required=True, help="Path to the root test (generated) images directory.")
    parser.add_argument("--output_dir", required=True, help="Folder to save metrics results.")
    parser.add_argument("--metrics_script", default="metrics.py",
                        help="Path to metrics.py script (default: metrics.py in current dir).")
    parser.add_argument("--fid_only", action="store_true", help="Only compute FID (skip pore metrics).")
    parser.add_argument("--thresh_method", default="otsu", help="Thresholding method (e.g. otsu, sauvola).")
    parser.add_argument("--pores_are_bright", action="store_true", help="If pores are bright.")
    parser.add_argument("--target_porosity", type=float, default=None, help="Target porosity for match_porosity.")
    parser.add_argument("--sauvola_window", type=int, default=101, help="Window size for Sauvola.")
    parser.add_argument("--sauvola_k", type=float, default=0.2, help="k param for Sauvola.")
    args = parser.parse_args()

    train_root = Path(args.train_dir)
    test_root = Path(args.test_dir)
    output_root = Path(args.output_dir)
    metrics_script = Path(args.metrics_script)

    if not train_root.exists():
        raise FileNotFoundError(f"Train directory not found: {train_root}")
    if not test_root.exists():
        raise FileNotFoundError(f"Test directory not found: {test_root}")
    if not metrics_script.exists():
        raise FileNotFoundError(f"metrics.py not found: {metrics_script}")

    output_root.mkdir(parents=True, exist_ok=True)
    print(f"Running metrics for all classes in: {train_root}")

    # map normalized name -> real/test folder
    train_classes = {normalize_class_name(d.name): d for d in train_root.iterdir() if d.is_dir()}
    test_classes = {normalize_class_name(d.name): d for d in test_root.iterdir() if d.is_dir()}

    print(f"Found {len(train_classes)} train classes and {len(test_classes)} test classes.")

    matched = []
    for key, train_path in sorted(train_classes.items()):
        if key in test_classes:
            test_path = test_classes[key]
            matched.append((key, train_path, test_path))
        else:
            print(f"⚠️  No matching test class for train/{train_path.name}")

    if not matched:
        raise SystemExit("No matching class folders found!")

    for key, train_path, test_path in matched:
        class_out = output_root / f"class_{key}"
        class_out.mkdir(parents=True, exist_ok=True)
        print(f"\n🔹 Running metrics for class '{key}'")
        cmd = [
            "python", str(metrics_script),
            "--train_folder", str(train_path),
            "--test_folder", str(test_path),
            "--metrics_folder", str(class_out)
        ]
        if args.fid_only:
            cmd.append("--fid_only")
        if args.pores_are_bright:
            cmd.append("--pores_are_bright")
        if args.target_porosity is not None:
            cmd += ["--target_porosity", str(args.target_porosity)]

        print(" ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ metrics.py failed for {key}: {e}")
        else:
            print(f"✅ Finished {key} → {class_out}")

if __name__ == "__main__":
    main()
