"""
Prepare real (ground-truth) images for fair comparison against two different
generative models, by mirroring each model's own training-time preprocessing:

  GAN pipeline:  center crop to square (shorter side) -> downsample to 512x512
                 (matches: 3072x2048 real -> 2048x2048 center crop -> 512x512 GAN training)

  LDM pipeline:  uniform downsample straight to 576x384, no crop
                 (matches: 3072x2048 real -> 576x384 LDM training, aspect ratio preserved)

Handles a nested-by-class input layout:

    real_images/0_0%/*.tif
    real_images/5_0%/*.tif
    ... (one subfolder per strain class, images directly inside each)

and mirrors that same class-folder structure in the output directories, so
output looks like gan_output_dir/0_0%/*.png, gan_output_dir/5_0%/*.png, etc.

Usage:
    python prepare_real_images.py --input_dir /path/to/real_images \
        --gan_output_dir /path/to/output/gan_real \
        --ldm_output_dir /path/to/output/ldm_real

Only one of --gan_output_dir / --ldm_output_dir needs to be given if you only
want one pipeline run. If --input_dir contains image files directly (no class
subfolders), it's treated as a single class and processed as-is.
"""

import argparse
import os
from PIL import Image

VALID_EXT = (".png", ".jpeg", ".jpg", ".tif", ".tiff")

# Pillow renamed the resampling enum location in v9.1+; this works either way.
try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE = Image.LANCZOS


def list_images(folder):
    return [f for f in sorted(os.listdir(folder)) if f.lower().endswith(VALID_EXT)]


def center_crop_square(img):
    """Crop to a square using the shorter side, centered on the longer side.
    No distortion: every kept pixel is untouched, we're just narrowing the
    field of view. This matches a standard center-crop-to-square preprocessing
    step (e.g. 3072x2048 -> 2048x2048)."""
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def prepare_for_gan(img, crop_to_square=True, final_size=(512, 512)):
    """Center crop to square (if needed), then uniformly downsample.
    No distortion introduced at either step."""
    if crop_to_square:
        img = center_crop_square(img)
    return img.resize(final_size, RESAMPLE)


def prepare_for_ldm(img, final_size=(576, 384)):
    """Uniform downsample only, no crop. Only valid if final_size preserves
    the original aspect ratio -- this function does NOT check that for you,
    since it depends on your source image dimensions."""
    return img.resize(final_size, RESAMPLE)


def find_class_dirs(input_dir):
    """Return sorted subfolder names under input_dir that themselves contain
    at least one valid image -- i.e. the strain-class folders. Returns an
    empty list if input_dir has no such subfolders (flat, single-class layout)."""
    class_dirs = []
    for entry in sorted(os.listdir(input_dir)):
        full = os.path.join(input_dir, entry)
        if os.path.isdir(full) and list_images(full):
            class_dirs.append(entry)
    return class_dirs


def run_pipeline_single(input_dir, output_dir, prepare_fn, label):
    """Process one flat folder of images."""
    os.makedirs(output_dir, exist_ok=True)
    filenames = list_images(input_dir)
    if not filenames:
        print(f"[{label}] No images found in {input_dir} (checked {VALID_EXT})")
        return

    print(f"[{label}] Processing {len(filenames)} images from {input_dir} -> {output_dir}")
    for i, filename in enumerate(filenames, 1):
        src_path = os.path.join(input_dir, filename)
        with Image.open(src_path) as img:
            out = prepare_fn(img.convert("RGB"))
            # keep original filename/extension so pairing real<->synthetic stays obvious
            out.save(os.path.join(output_dir, filename))
        if i % 25 == 0 or i == len(filenames):
            print(f"[{label}] {i}/{len(filenames)}")


def run_pipeline(input_dir, output_dir, prepare_fn, label):
    """Process input_dir, auto-detecting a nested class-folder layout
    (real_images/<class>/<image files>) vs. a flat single-class layout
    (real_images/<image files>). Mirrors class subfolders into output_dir."""
    class_dirs = find_class_dirs(input_dir)

    if not class_dirs:
        # flat layout: no class subfolders detected, treat input_dir itself
        # as the one set of images to process
        run_pipeline_single(input_dir, output_dir, prepare_fn, label)
        return

    print(f"[{label}] Found {len(class_dirs)} class folders: {class_dirs}")
    for class_name in class_dirs:
        run_pipeline_single(
            os.path.join(input_dir, class_name),
            os.path.join(output_dir, class_name),
            prepare_fn,
            label=f"{label}/{class_name}",
        )


def main():
    parser = argparse.ArgumentParser(
        description="Crop/resize real images to match GAN and/or LDM training preprocessing."
    )
    parser.add_argument("--input_dir", required=True, help="Folder of original real images")
    parser.add_argument("--gan_output_dir", default=None, help="Output folder for GAN-matched real images")
    parser.add_argument("--ldm_output_dir", default=None, help="Output folder for LDM-matched real images")
    parser.add_argument("--gan_final_size", type=int, nargs=2, default=[512, 512], metavar=("W", "H"))
    parser.add_argument("--ldm_final_size", type=int, nargs=2, default=[576, 384], metavar=("W", "H"))
    args = parser.parse_args()

    if not args.gan_output_dir and not args.ldm_output_dir:
        parser.error("Provide at least one of --gan_output_dir or --ldm_output_dir")

    if args.gan_output_dir:
        run_pipeline(
            args.input_dir,
            args.gan_output_dir,
            lambda img: prepare_for_gan(img, final_size=tuple(args.gan_final_size)),
            label="GAN",
        )

    if args.ldm_output_dir:
        # sanity check: warn (don't fail) if the requested size doesn't preserve
        # the aspect ratio of a sample image, since a mismatch here silently
        # reintroduces the exact distortion issue this script exists to avoid.
        # Works whether input_dir is flat or nested by class.
        class_dirs = find_class_dirs(args.input_dir)
        sample_dir = os.path.join(args.input_dir, class_dirs[0]) if class_dirs else args.input_dir
        filenames = list_images(sample_dir)
        if filenames:
            with Image.open(os.path.join(sample_dir, filenames[0])) as sample:
                sw, sh = sample.size
            tw, th = args.ldm_final_size
            src_ratio = sw / sh
            tgt_ratio = tw / th
            if abs(src_ratio - tgt_ratio) > 0.01:
                print(
                    f"[LDM] WARNING: source aspect ratio {sw}x{sh} ({src_ratio:.3f}) "
                    f"does not match target {tw}x{th} ({tgt_ratio:.3f}). "
                    f"This resize will distort images -- double check --ldm_final_size."
                )

        run_pipeline(
            args.input_dir,
            args.ldm_output_dir,
            lambda img: prepare_for_ldm(img, final_size=tuple(args.ldm_final_size)),
            label="LDM",
        )


if __name__ == "__main__":
    main()