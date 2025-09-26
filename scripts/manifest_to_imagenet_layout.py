#!/usr/bin/env python3
"""
Build ImageNet-style class folders from a manifest.jsonl and write class mapping files.

- Reads JSONL lines like:
  {"path": ".../t1_0%_21r19.tif", "description": "... 0% strain ...", "class_id": 0, ...}

- Creates (by default) symlinks into folders:
    <out_root>/<split>/<class_folder>/*.tif
  where class_folder is one of:
    s000_unstrained, s004, s008, s012, s016, s020

- Writes mapping files:
    <maps_out>/index_strain.yaml               # idx -> class_folder
    <maps_out>/strain_clsidx_to_label.txt      # idx -> human label

You can toggle copy vs symlink via --copy. Optionally split into train/val with --val-split.
"""

import argparse, json, os, sys, shutil, random
from pathlib import Path

# Fixed mappings (class_id -> folder name / human label)
CLASS_FOLDER = {
    0: "s000_unstrained",
    1: "s004",
    2: "s008",
    3: "s012",
    4: "s016",
    5: "s020",
}
CLASS_LABEL = {
    0: "CNT microstructure, 0% strain",
    1: "CNT microstructure, 4% strain",
    2: "CNT microstructure, 8% strain",
    3: "CNT microstructure, 12% strain",
    4: "CNT microstructure, 16% strain",
    5: "CNT microstructure, 20% strain",
}

def write_mappings(maps_out: Path):
    maps_out.mkdir(parents=True, exist_ok=True)
    # index_strain.yaml
    yaml_path = maps_out / "index_strain.yaml"
    with yaml_path.open("w", encoding="utf-8") as f:
        for k in sorted(CLASS_FOLDER.keys()):
            f.write(f"{k}: {CLASS_FOLDER[k]}\n")
    # strain_clsidx_to_label.txt
    txt_path = maps_out / "strain_clsidx_to_label.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        for k in sorted(CLASS_LABEL.keys()):
            f.write(f"{k}: '{CLASS_LABEL[k]}'\n")
    print(f"[mappings] Wrote:\n  {yaml_path}\n  {txt_path}")

def safe_link_or_copy(src: Path, dst: Path, do_copy: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "skipped"  # already there
    if do_copy:
        shutil.copy2(src, dst)
        return "copied"
    else:
        # Create a relative symlink when possible
        try:
            rel = os.path.relpath(src, start=dst.parent)
            os.symlink(rel, dst)
        except OSError:
            # Fallback to absolute symlink
            os.symlink(str(src), dst)
        return "linked"

def derive_class_id(rec: dict):
    """Prefer explicit class_id. Fallback to strain_frac or description if present."""
    if "class_id" in rec and rec["class_id"] is not None:
        return int(rec["class_id"])
    # Try strain_frac -> id mapping
    sf = rec.get("strain_frac", None)
    if isinstance(sf, (int, float)):
        mapping = {0.00:0, 0.04:1, 0.08:2, 0.12:3, 0.16:4, 0.20:5}
        if sf in mapping:
            return mapping[sf]
    # Last resort: parse description for "0%"/"4%" etc.
    desc = (rec.get("description") or "").lower()
    for pct, cid in [("0%",0),("4%",1),("8%",2),("12%",3),("16%",4),("20%",5)]:
        if pct in desc:
            return cid
    return None

def load_manifest(manifest_path: Path):
    records = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[warn] Skipping malformed JSON on line {i}: {e}", file=sys.stderr)
                continue
            p = rec.get("path")
            if not p:
                print(f"[warn] Skipping line {i}: missing 'path'", file=sys.stderr)
                continue
            cid = derive_class_id(rec)
            if cid is None or cid not in CLASS_FOLDER:
                print(f"[warn] Skipping line {i}: bad or unknown class for {p}", file=sys.stderr)
                continue
            rec["_class_id"] = cid
            records.append(rec)
    return records

def split_by_class(records, val_split: float, seed: int):
    """Return (train, val) lists, preserving per-class proportions."""
    if val_split <= 0:
        return records, []

    random.seed(seed)
    by_cls = {}
    for r in records:
        by_cls.setdefault(r["_class_id"], []).append(r)

    train, val = [], []
    for cid, items in by_cls.items():
        random.shuffle(items)
        n_val = int(round(len(items) * val_split))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    return train, val

def export_split(recs, out_dir: Path, copy_files: bool):
    n_ok = n_missing = 0
    for r in recs:
        src = Path(r["path"])
        if not src.exists():
            n_missing += 1
            print(f"[missing] {src}", file=sys.stderr)
            continue
        cls_id = r["_class_id"]
        cls_folder = CLASS_FOLDER[cls_id]
        dst = out_dir / cls_folder / src.name
        status = safe_link_or_copy(src, dst, do_copy=copy_files)
        n_ok += 1
    return n_ok, n_missing

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="Path to manifest.jsonl")
    ap.add_argument("--out-root", required=True, help="Output dataset root (will create class folders)")
    ap.add_argument("--maps-out", required=True, help="Where to write index_strain.yaml and strain_clsidx_to_label.txt")
    ap.add_argument("--val-split", type=float, default=0.0, help="Fraction for validation split (0.0 = no val)")
    ap.add_argument("--seed", type=int, default=23, help="Random seed for splitting")
    ap.add_argument("--copy", action="store_true", help="Copy files instead of making symlinks")
    ap.add_argument("--split-names", nargs=2, default=["train","val"], help="Names for train/val dirs")
    args = ap.parse_args()

    manifest_path = Path(args.manifest).expanduser()
    out_root = Path(args.out_root).expanduser()
    maps_out = Path(args.maps_out).expanduser()

    print(f"[info] Loading manifest: {manifest_path}")
    recs = load_manifest(manifest_path)
    if not recs:
        print("[error] No usable records found.", file=sys.stderr)
        sys.exit(2)

    # Write mappings
    write_mappings(maps_out)

    # Prepare splits
    train_name, val_name = args.split_names
    train_recs, val_recs = split_by_class(recs, args.val_split, args.seed)

    # Export train
    train_dir = out_root / train_name
    print(f"[export] {train_name} → {train_dir}")
    ok, miss = export_split(train_recs, train_dir, copy_files=args.copy)
    print(f"[done] {train_name}: linked/copied {ok}, missing {miss}")

    # Export val (optional)
    if val_recs:
        val_dir = out_root / val_name
        print(f"[export] {val_name} → {val_dir}")
        ok, miss = export_split(val_recs, val_dir, copy_files=args.copy)
        print(f"[done] {val_name}: linked/copied {ok}, missing {miss}")

    print("\n[success] ImageNet-style layout is ready.")

if __name__ == "__main__":
    main()
