#!/usr/bin/env python3

# call with python org_testFile.py input_path output_path

import os
import re
import shutil
from pathlib import Path
import argparse

ALLOWED_EXTS = {".tif", ".tiff", ".jpeg"}

PCT_TO_CLASS = {
    0:  "class0_00%",
    2:  "class1_02%",
    4:  "class2_04%",
    6:  "class3_06%",
    8:  "class4_08%",
    10: "class5_10%",
    12: "class6_12%",
}

PCT_RE = re.compile(r"(\d+)\s*%(?=\.[^.]+$)")


def reorganize(input_root: Path, output_root: Path, move: bool, add_in_situ: bool):
    for root, _, files in os.walk(input_root):
        for fname in files:
            src = Path(root) / fname
            ext = src.suffix.lower()

            if ext not in ALLOWED_EXTS:
                continue

            m = PCT_RE.search(fname)
            if not m:
                continue

            pct = int(m.group(1))
            if pct not in PCT_TO_CLASS:
                continue

            class_dir = PCT_TO_CLASS[pct]

            dst_dir = output_root
            if add_in_situ:
                dst_dir = dst_dir / "in-situ"
            dst_dir = dst_dir / class_dir
            dst_dir.mkdir(parents=True, exist_ok=True)

            dst = dst_dir / src.name
            if dst.exists():
                stem = src.stem
                i = 1
                while dst.exists():
                    dst = dst_dir / f"{stem}__{i}{src.suffix}"
                    i += 1

            if move:
                shutil.move(src, dst)
            else:
                shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(
        description="Recursively reorganize images into percent-based class folders"
    )
    parser.add_argument("input_root", type=Path, help="Root folder to scan")
    parser.add_argument("output_root", type=Path, help="Where to create class folders")
    parser.add_argument(
        "--copy", action="store_true", help="Copy instead of move"
    )
    parser.add_argument(
        "--no-in-situ", action="store_true", help="Do not create in-situ/ folder"
    )

    args = parser.parse_args()

    reorganize(
        input_root=args.input_root,
        output_root=args.output_root,
        move=not args.copy,
        add_in_situ=not args.no_in_situ,
    )


if __name__ == "__main__":
    main()
