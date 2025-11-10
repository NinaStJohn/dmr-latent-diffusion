#!/usr/bin/env python3
import os, argparse, random
from pathlib import Path
from PIL import Image

ALLOWED = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

def list_images(folder: Path):
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in ALLOWED])

def random_native_patch(im: Image.Image, size: int) -> Image.Image:
    w, h = im.size
    if w < size or h < size:
        # minimally upscale to fit
        scale = max(size / w, size / h)
        im = im.resize((int(round(w*scale)), int(round(h*scale))), Image.BILINEAR)
        w, h = im.size
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    return im.crop((x, y, x + size, y + size))

def resize_then_random_crop(im: Image.Image, size: int, short_side="min") -> Image.Image:
    # 1) resize so the short side == size (keeps aspect), like many training loaders
    w, h = im.size
    if short_side == "min":
        if min(w, h) != size:
            if w < h:
                new_w, new_h = size, int(round(h * (size / w)))
            else:
                new_w, new_h = int(round(w * (size / h))), size
            im = im.resize((new_w, new_h), Image.LANCZOS)
    else:
        # alternative: ensure short side >= size (no downscale if already smaller)
        if min(w, h) < size:
            scale = size / min(w, h)
            im = im.resize((int(round(w*scale)), int(round(h*scale))), Image.LANCZOS)

    # 2) random 256×256 crop from the resized image
    w, h = im.size
    if w < size or h < size:
        pad_w = max(0, size - w)
        pad_h = max(0, size - h)
        im = im.crop((0, 0, w, h)).resize((max(size, w), max(size, h)), Image.BILINEAR)
        w, h = im.size
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    return im.crop((x, y, x + size, y + size))

def maybe_flip(im: Image.Image, p=0.5) -> Image.Image:
    return im.transpose(Image.FLIP_LEFT_RIGHT) if random.random() < p else im

def main(args):
    random.seed(args.seed)
    in_root = Path(args.input_dir)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    class_dirs = [d for d in in_root.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if not class_dirs:
        raise SystemExit(f"No class subfolders in {in_root}")

    print(f"Found {len(class_dirs)} classes.")
    for cdir in sorted(class_dirs):
        images = list_images(cdir)
        if not images:
            print(f"[WARN] No images in {cdir}, skipping.")
            continue

        out_class = out_root / cdir.name
        out_class.mkdir(parents=True, exist_ok=True)

        print(f"[{cdir.name}] {len(images)} src imgs -> {args.per_class} crops ({args.mode})")
        order = list(range(len(images)))
        random.shuffle(order)
        idx, made = 0, 0

        while made < args.per_class:
            if idx >= len(order):
                random.shuffle(order)
                idx = 0
            src = images[order[idx]]; idx += 1

            try:
                with Image.open(src) as im:
                    im = im.convert("RGB")
                    if args.mode == "match":
                        crop = resize_then_random_crop(im, args.crop_size)
                    else:
                        crop = random_native_patch(im, args.crop_size)
                    if args.hflip:
                        crop = maybe_flip(crop, p=0.5)
            except Exception as e:
                print(f"[WARN] {src.name}: {e}"); continue

            crop.save(out_class / f"{cdir.name}_crop_{made:04d}.png")
            made += 1

        print(f"[{cdir.name}] done -> {out_class}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Make 256×256 crops per class.")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--per_class", type=int, default=40)
    ap.add_argument("--crop_size", type=int, default=256)
    ap.add_argument("--mode", choices=["match","patch"], default="match",
                    help="'match' ≈ training: resize short side to crop_size, then random crop; 'patch' = native 256 tiles")
    ap.add_argument("--hflip", action="store_true", help="random horizontal flip like common training aug")
    ap.add_argument("--seed", type=int, default=23)
    args = ap.parse_args()
    main(args)
