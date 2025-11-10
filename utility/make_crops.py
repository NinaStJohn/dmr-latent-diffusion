#!/usr/bin/env python3
import os, argparse, random
from pathlib import Path
from PIL import Image

ALLOWED = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

def list_images(folder: Path):
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in ALLOWED])

def _ensure_min_size(im: Image.Image, target_w: int, target_h: int, resample=Image.LANCZOS) -> Image.Image:
    w, h = im.size
    if w >= target_w and h >= target_h:
        return im
    scale = max(target_w / w, target_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return im.resize((new_w, new_h), resample)

def random_native_patch(im: Image.Image, target_w: int, target_h: int) -> Image.Image:
    im = _ensure_min_size(im, target_w, target_h, resample=Image.BILINEAR)
    w, h = im.size
    x = random.randint(0, w - target_w)
    y = random.randint(0, h - target_h)
    return im.crop((x, y, x + target_w, y + target_h))

def resize_then_random_crop(im: Image.Image, target_w: int, target_h: int) -> Image.Image:
    # like common training pipelines: keep aspect, first ensure both dims >= target, then random crop
    im = _ensure_min_size(im, target_w, target_h, resample=Image.LANCZOS)
    w, h = im.size
    x = random.randint(0, w - target_w)
    y = random.randint(0, h - target_h)
    return im.crop((x, y, x + target_w, y + target_h))

def _resolve_target_size(args):
    # priority: explicit w/h > square size
    if args.crop_w and args.crop_h:
        return int(args.crop_w), int(args.crop_h)
    # allow "WxH" in crop_size as a convenience
    if isinstance(args.crop_size, str) and "x" in args.crop_size.lower():
        w_str, h_str = args.crop_size.lower().split("x")
        return int(w_str), int(h_str)
    # fallback: square
    return int(args.crop_size), int(args.crop_size)

def _resolve_resize_arg(resize_arg):
    """Parse --resize flag into (width, height) tuple or None."""
    if resize_arg is None:
        return None
    if isinstance(resize_arg, int):
        return (resize_arg, resize_arg)
    if isinstance(resize_arg, str) and "x" in resize_arg.lower():
        w_str, h_str = resize_arg.lower().split("x")
        return int(w_str), int(h_str)
    # fallback: assume square
    return (int(resize_arg), int(resize_arg))


def maybe_flip(im: Image.Image, p=0.5) -> Image.Image:
    return im.transpose(Image.FLIP_LEFT_RIGHT) if random.random() < p else im

def main(args):
    random.seed(args.seed)
    in_root = Path(args.input_dir)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    target_w, target_h = _resolve_target_size(args)

    if args.resize_only:
        resize_to = _resolve_resize_arg(args.resize)
        if resize_to is None:
            raise ValueError("--resize_only requires --resize WxH")
        os.makedirs(args.output_dir, exist_ok=True)

        for cls_dir in os.listdir(args.input_dir):
            in_path = os.path.join(args.input_dir, cls_dir)
            if os.path.isdir(in_path):
                out_path = os.path.join(args.output_dir, cls_dir)
                os.makedirs(out_path, exist_ok=True)
                for fname in os.listdir(in_path):
                    if fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                        im = Image.open(os.path.join(in_path, fname))
                        im = im.resize(resize_to, Image.LANCZOS)
                        im.save(os.path.join(out_path, fname))
        print("Finished resize-only mode.")
        return


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
                        crop = resize_then_random_crop(im, target_w, target_h)
                    else:
                        crop = random_native_patch(im, target_w, target_h)
                    if args.hflip:
                        crop = maybe_flip(crop, p=0.5)
            except Exception as e:
                print(f"[WARN] {src.name}: {e}"); continue

            crop.save(out_class / f"{cdir.name}_crop_{target_w}x{target_h}_{made:04d}.png")
            made += 1

        print(f"[{cdir.name}] done -> {out_class}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Make 256×256 crops per class.")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--per_class", type=int, default=40)
    ap.add_argument("--crop_size", default=256,
                    help="Square side or 'WxH' (e.g., 256 or 320x192)")
    ap.add_argument("--crop_w", type=int, default=None,
                    help="Override width (used with --crop_h)")
    ap.add_argument("--crop_h", type=int, default=None,
                    help="Override height (used with --crop_w)")

    ap.add_argument("--mode", choices=["match","patch"], default="match",
                    help="'match' ≈ training: resize short side to crop_size, then random crop; 'patch' = native 256 tiles")
    ap.add_argument("--hflip", action="store_true", help="random horizontal flip like common training aug")
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--resize", default=None,
                help="Resize output crops to this size (e.g. 256 or 512x384). "
                     "If not set, crops are saved at their native cropped size.")
    ap.add_argument("--resize_only", action="store_true",
                    help="If set, skip cropping and just resize all images to --resize.")

    args = ap.parse_args()
    main(args)
