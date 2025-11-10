#!/usr/bin/env python3
"""
Tile images into 8 patches (4x2 grid) of 256x256.
Designed for ~1034x512 inputs that visually contain a 4x2 mosaic.
We crop a centered 1024x512 area (or left-aligned if requested) before tiling.
Usage:
  python tile_images.py --indir <folder> --outdir <folder> [--align center|left] [--ext png,jpg,jpeg,tif,tiff]
"""
import argparse
from pathlib import Path
from PIL import Image

def tile_image(img: Image.Image, align: str = "center"):
    TILE = 256
    cols, rows = 4, 2
    need_w, need_h = TILE * cols, TILE * rows  # 1024 x 512

    w, h = img.size
    if h < need_h or w < need_w:
        raise ValueError(f"Image {w}x{h} is smaller than required {need_w}x{need_h}.")

    # Determine crop box to get exactly 1024x512
    if align == "center":
        left = (w - need_w) // 2
    elif align == "left":
        left = 0
    else:
        raise ValueError("align must be 'center' or 'left'")
    top = (h - need_h) // 2 if h > need_h else 0

    box = (left, top, left + need_w, top + need_h)
    img_c = img.crop(box)

    tiles = []
    for r in range(rows):
        for c in range(cols):
            x0 = c * TILE
            y0 = r * TILE
            tiles.append(img_c.crop((x0, y0, x0 + TILE, y0 + TILE)))
    return tiles

def main():
    p = argparse.ArgumentParser(description="Tile images into 8x 256x256 patches (4x2).")
    p.add_argument("--indir", required=True, type=Path, help="Input folder of images")
    p.add_argument("--outdir", required=True, type=Path, help="Output folder for tiles")
    p.add_argument("--align", default="center", choices=["center", "left"], help="Alignment when cropping to 1024 width")
    p.add_argument("--ext", default="png,jpg,jpeg,tif,tiff", help="Comma-separated list of extensions to include")
    args = p.parse_args()

    exts = {e.lower().strip().lstrip('.') for e in args.ext.split(',') if e.strip()}
    args.outdir.mkdir(parents=True, exist_ok=True)

    files = [f for f in args.indir.iterdir() if f.is_file() and f.suffix.lower().lstrip('.') in exts]
    if not files:
        print("No matching images found.")
        return

    count = 0
    for f in sorted(files):
        try:
            with Image.open(f) as im:
                im = im.convert("RGB")
                tiles = tile_image(im, align=args.align)
            for i, t in enumerate(tiles):
                r = i // 4
                c = i % 4
                out_name = f"{f.stem}_r{r}_c{c}.png"
                t.save(args.outdir / out_name, format="PNG")
                count += 1
        except Exception as e:
            print(f"Skipping {f.name}: {e}")
    print(f"Saved {count} tiles to {args.outdir}")

if __name__ == "__main__":
    main()

# C:\Users\seali\calpoly\internship\train\samples
# C:\Users\seali\calpoly\internship\train\samples_indivi

# C:\Users\seali\calpoly\internship\train\inputs
# C:\Users\seali\calpoly\internship\train\inputs_indivi

# python C:\Users\seali\calpoly\dmr-latent-diffusion\utility\tile_images.py --indir C:\Users\seali\calpoly\internship\train\samples --outdir C:\Users\seali\calpoly\internship\train\samples_indivi
# python C:\Users\seali\calpoly\dmr-latent-diffusion\utility\tile_images.py --indir C:\Users\seali\calpoly\internship\train\samples --outdir C:\Users\seali\calpoly\internship\train\inputs_indivi