#!/usr/bin/env python3
"""
crop_images.py

Recursively crops images under an input directory and writes them to an output
directory while mirroring the input folder structure.

Default behavior crops from the TOP-LEFT corner (x=0,y=0), which will remove a
bottom-right label if you crop smaller than the original.

Example (your case):
  python crop_images.py -i /path/in -o /path/out --width 1952 --height 1968

Optional offsets:
  python crop_images.py -i in -o out --width 1952 --height 1968 --x 10 --y 20
"""

import argparse
import os
from pathlib import Path

from PIL import Image


EXTS = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}


def is_image(path: Path) -> bool:
    return path.suffix.lower() in EXTS


def crop_one(src: Path, dst: Path, width: int, height: int, x: int, y: int) -> None:
    # Ensure output folder exists
    dst.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(src) as im:
        # Some TIFFs load lazily; convert to something Pillow can crop reliably
        im.load()

        W, H = im.size
        if x < 0 or y < 0:
            raise ValueError(f"Negative crop offsets not allowed: x={x}, y={y}")

        if x + width > W or y + height > H:
            raise ValueError(
                f"Crop {width}x{height}+{x}+{y} exceeds image bounds {W}x{H} for: {src}"
            )

        cropped = im.crop((x, y, x + width, y + height))

        # Preserve format based on original when possible
        save_kwargs = {}
        fmt = (im.format or "").upper()

        # If JPEG, ensure RGB (some TIFF/PNG can be RGBA/LA)
        if fmt == "JPEG" and cropped.mode not in ("RGB", "L"):
            cropped = cropped.convert("RGB")

        # For TIFFs, keep compression reasonable (optional)
        if src.suffix.lower() in (".tif", ".tiff"):
            # LZW is common; if it fails on your cluster, remove this line.
            save_kwargs["compression"] = "tiff_lzw"

        cropped.save(dst, **save_kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description="Recursively crop images and mirror directory structure.")
    parser.add_argument("-i", "--input_dir", required=True, help="Root input folder containing subfolders/images.")
    parser.add_argument("-o", "--output_dir", required=True, help="Root output folder to write cropped images.")
    parser.add_argument("--width", type=int, required=True, help="Crop width in pixels.")
    parser.add_argument("--height", type=int, required=True, help="Crop height in pixels.")
    parser.add_argument("--x", type=int, default=0, help="Crop x-offset (default: 0).")
    parser.add_argument("--y", type=int, default=0, help="Crop y-offset (default: 0).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument("--dry_run", action="store_true", help="Print planned actions without writing files.")
    args = parser.parse_args()

    in_root = Path(args.input_dir).expanduser().resolve()
    out_root = Path(args.output_dir).expanduser().resolve()

    if not in_root.is_dir():
        print(f"ERROR: input_dir is not a directory: {in_root}")
        return 2

    out_root.mkdir(parents=True, exist_ok=True)

    total = 0
    ok = 0
    skipped = 0
    failed = 0

    for src in in_root.rglob("*"):
        if not src.is_file() or not is_image(src):
            continue

        total += 1
        rel = src.relative_to(in_root)
        dst = out_root / rel

        if dst.exists() and not args.overwrite:
            skipped += 1
            continue

        if args.dry_run:
            print(f"[DRY] {src} -> {dst}  crop={args.width}x{args.height}+{args.x}+{args.y}")
            ok += 1
            continue

        try:
            crop_one(src, dst, args.width, args.height, args.x, args.y)
            print(f"Cropped: {rel}")
            ok += 1
        except Exception as e:
            failed += 1
            print(f"FAILED: {rel}  ({e})")

    print("\nDone.")
    print(f"  total found : {total}")
    print(f"  cropped     : {ok}")
    print(f"  skipped     : {skipped}")
    print(f"  failed      : {failed}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
