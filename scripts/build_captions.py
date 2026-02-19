#!/usr/bin/env python3
import argparse, csv, json, re, sys
from pathlib import Path

from PIL import Image
import numpy as np

# Keep TIFFs
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

def find_images(root: Path):
    for p in root.rglob('*'):
        if p.suffix.lower() in IMG_EXTS:
            yield p

def parse_strain_from_folder(folder_name: str, default="0%"):
    """
    Look for things like '0%', '5 %', '10percent' in the immediate parent folder.
    Returns a string like '0%'. Falls back to default if none found.
    """
    m = re.search(r'(\d+)\s*%|\b(\d+)\s*percent\b', folder_name, flags=re.IGNORECASE)
    if m:
        val = m.group(1) or m.group(2)
        return f"{val}%"
    return default

def parse_class_id_from_folder(folder_name: str) -> int:
    """
    Infer class id from common folder naming schemes.

    Supports:
      - "class0_00%" -> 0
      - "class6_12%" -> 6
      - "0_0%"       -> 0
      - "3_sample"   -> 3
    Falls back to 0 if nothing found.
    """
    s = folder_name.strip()

    # 1) Preferred: "class<id>_..." or "class<id>-..."
    m = re.search(r'(?i)\bclass\s*[_-]?\s*(\d+)\b', s)
    if m:
        return int(m.group(1))

    # 2) If it starts with digits: "3_foo", "12bar"
    m = re.match(r'\s*(\d+)', s)
    if m:
        return int(m.group(1))

    # 3) Any integer anywhere (last resort)
    m = re.search(r'(\d+)', s)
    return int(m.group(1)) if m else 0


def guess_material_from_folder(folder_name: str, class_id: int, class_to_name: dict) -> str:
    """
    Resolve a material name:
      1) If class_id exists in class_to_name, use that.
      2) Else, strip leading digits/underscores/dashes from folder name and use the remainder.
      3) Else, fall back to 'class_<id>'.
    """
    if class_id in class_to_name:
        return str(class_to_name[class_id])

    # strip leading digits + separators, e.g. "3_CNTA_10%" -> "CNTA_10%"
    rest = re.sub(r'^\d+[\s_\-]*', '', folder_name).strip()
    # keep only letters, numbers, +, -, _ (drop trailing strain hints if any)
    rest = re.split(r'[/\\]', rest)[0]
    rest = rest if rest else f"class_{class_id}"
    return rest

def calculate_porosity(img_path: Path) -> float:
    """
    Porosity (percent) using luminance + fixed threshold:
      - load image
      - convert to RGB
      - luminance Y = 0.299 R + 0.587 G + 0.114 B (scaled to [0,1])
      - pores = (Y < 0.5)
      - porosity = mean(pores) * 100
    """
    with Image.open(img_path) as im:
        im = im.convert('RGB')  # ensure 3 channels
        arr = np.asarray(im, dtype=np.float32) / 255.0  # [H,W,3] in [0,1]
    Y = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]  # [H,W]
    pores = (Y < 0.5)
    porosity = 100.0 * (pores.sum() / pores.size)
    return float(porosity)

def main():
    ap = argparse.ArgumentParser(description="Build image→text manifest with prompts (caption); optionally write filelist.txt and conds.npy for ADM conditioning.")
    ap.add_argument("--image-root", required=True, help="Absolute path to image folder (will recurse).")
    ap.add_argument("--out", default=None, help="Output path. Defaults to <image-root>/manifest.jsonl")
    ap.add_argument("--format", choices=["jsonl","csv"], default="jsonl", help="Manifest format.")

    # Prompt generation options
    ap.add_argument("--caption-prefix", default="CNT microstructure", help="Prefix used in generated caption.")
    ap.add_argument("--default-strain", default="0%", help="Fallback strain if none found in parent folder name.")
    ap.add_argument("--material-map", default=None,
                    help="Optional JSON mapping of class_id (int or str) → material name.")

    # Legacy description options (kept for backwards compatibility)
    ap.add_argument("--desc-prefix", default="CNT microstructure under", help="Prefix for legacy description.")
    ap.add_argument("--use-folder-name", action="store_true",
                    help="Include parent folder name in legacy description.")

    # Porosity + ADM outputs (unchanged behavior)
    ap.add_argument("--porosity", action="store_true", help="Compute porosity (luminance threshold @ 0.5).")
    ap.add_argument("--filelist-out", default=None, help="If set, write absolute image paths here (one per line).")
    ap.add_argument("--conds-out", default=None, help="If set, write numeric conditioning matrix here as .npy")
    ap.add_argument("--num-classes", type=int, default=None,
                    help="If given, one-hot will use this K. If omitted, K = max(class_id)+1 from folders.")
    args = ap.parse_args()

    # Load optional class_id → material name mapping
    class_to_name = {}
    if args.material_map:
        mp = Path(args.material_map)
        with open(mp, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # normalize keys to int
        for k, v in raw.items():
            try:
                class_to_name[int(k)] = v
            except Exception:
                continue

    root = Path(args.image_root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        print(f"[error] image-root not found or not a directory: {root}", file=sys.stderr)
        sys.exit(2)

    out_path = Path(args.out) if args.out else (root / f"manifest.{args.format}")
    items = []
    paths_for_files = []
    triples_for_conds = []  # (porosity_frac, strain_frac, class_id)

    # Stable order
    images = sorted(find_images(root), key=lambda p: str(p).lower())

    for img in images:
        parent = img.parent.name
        strain_str = parse_strain_from_folder(parent, default=args.default_strain)
        class_id = parse_class_id_from_folder(parent)
        material_name = guess_material_from_folder(parent, class_id, class_to_name)

        # --- Build prompt (caption) for txt2img ---
        # e.g., "CNT microstructure (material=MatA) under 10% strain. Porosity: 0.623%."
        caption = f"{args.caption_prefix} (material={material_name}) under {strain_str} strain"

        # --- Legacy description (kept, not used by txt2img pipeline) ---
        if args.use_folder_name:
            desc = f"{args.desc_prefix} {strain_str} {parent} strain"
        else:
            desc = f"{args.desc_prefix} {strain_str} strain"

        rec = {
            "path": str(img.resolve()),
            "description": desc,   # legacy
            "caption": caption     # new prompt for txt2img
        }

        # Compute porosity if requested or needed for ADM
        need_porosity = args.porosity or (args.conds_out is not None)
        porosity_val = None
        if need_porosity:
            try:
                porosity_val = calculate_porosity(img)  # percent
                if args.porosity:
                    rec["porosity"] = round(porosity_val, 3)
            except Exception:
                porosity_val = None
                if args.porosity:
                    rec["porosity"] = None

        # Append porosity to the text fields only if we computed it
        if porosity_val is not None:
            por_str = f"{porosity_val:.3f}%"
            rec["caption"] = f"{rec['caption']}. Porosity: {por_str}."
            rec["description"] = f"{rec['description']}. Porosity: {por_str}"

        # Add machine-friendly numerics (unchanged)
        s_clean = strain_str.strip().replace('%', '')
        strain_frac = float(s_clean) / 100.0 if s_clean else 0.0
        rec["strain_frac"] = round(strain_frac, 6)
        rec["class_id"] = int(class_id)
        if porosity_val is not None:
            rec["porosity_frac"] = round(float(porosity_val) / 100.0, 6)

        items.append(rec)
        paths_for_files.append(rec["path"])

        # If we will write conds.npy, queue the triple (use 0.0 if porosity missing)
        if args.conds_out is not None:
            por_frac = (float(porosity_val) / 100.0) if (porosity_val is not None) else 0.0
            triples_for_conds.append((por_frac, strain_frac, class_id))

    # Write manifest
    if args.format == "jsonl":
        with open(out_path, "w", encoding="utf-8") as f:
            for r in items:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    else:
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            header = ["path", "description", "caption"]
            if args.porosity:
                header.append("porosity")
            header += ["porosity_frac", "strain_frac", "class_id"]
            writer.writerow(header)
            for r in items:
                row = [r["path"], r["description"], r["caption"]]
                if args.porosity:
                    row.append("" if r.get("porosity") is None else r["porosity"])
                row += [
                    "" if r.get("porosity_frac") is None else r["porosity_frac"],
                    r["strain_frac"],
                    r["class_id"],
                ]
                writer.writerow(row)

    print(f"Wrote {len(items)} items → {out_path}")

    # Optional outputs for ADM conditioning (unchanged)
    if args.filelist_out:
        with open(args.filelist_out, "w", encoding="utf-8") as f:
            for p in paths_for_files:
                f.write(p + "\n")
        print(f"Wrote filelist with {len(paths_for_files)} paths → {args.filelist_out}")

    if args.conds_out:
        if not triples_for_conds:
            print("[warn] no cond triples captured; nothing to write for conds.npy", file=sys.stderr)
        else:
            if args.num_classes is not None:
                K = int(args.num_classes)
            else:
                max_cid = max(cid for _, _, cid in triples_for_conds)
                K = max_cid + 1
            N = len(triples_for_conds)
            D = 2 + K  # porosity_frac, strain_frac, one_hot(class_id, K)
            conds = np.zeros((N, D), dtype=np.float32)
            for i, (por, st, cid) in enumerate(triples_for_conds):
                conds[i, 0] = float(por)
                conds[i, 1] = float(st)
                if 0 <= cid < K:
                    conds[i, 2 + cid] = 1.0
            np.save(args.conds_out, conds)
            print(f"Wrote conds {conds.shape} → {args.conds_out}")

if __name__ == "__main__":
    main()
