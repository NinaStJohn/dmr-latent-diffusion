import json
import random
import argparse
import re
from pathlib import Path

def infer_class_id(entry):
    # Try to use existing class_id
    if "class_id" in entry and entry["class_id"] is not None:
        return entry["class_id"]
    
    # Try to extract from path, e.g. "/u/nstjohn/images/4_16%/..."
    match = re.search(r"/(\d+)_", entry["path"])
    if match:
        return int(match.group(1))
    
    # As fallback, try from strain_frac if present (scaled like 0.16 → class 4)
    if "strain_frac" in entry:
        return int(round(entry["strain_frac"] * 25))  # since 0.04 → 1, 0.16 → 4
    
    raise ValueError(f"Could not infer class_id for entry: {entry}")

def split_manifest(input_path, train_output, val_output, val_ratio=0.2, seed=42):
    with open(input_path, "r") as f:
        lines = f.readlines()
    
    entries = [json.loads(line) for line in lines if line.strip()]
    
    # Ensure every entry has class_id
    for e in entries:
        e["class_id"] = infer_class_id(e)
    
    random.seed(seed)
    random.shuffle(entries)
    
    split_idx = int(len(entries) * (1 - val_ratio))
    train_entries = entries[:split_idx]
    val_entries = entries[split_idx:]
    
    with open(train_output, "w") as f:
        for entry in train_entries:
            f.write(json.dumps(entry) + "\n")
    
    with open(val_output, "w") as f:
        for entry in val_entries:
            f.write(json.dumps(entry) + "\n")
    
    print(f"Total entries: {len(entries)}")
    print(f"Training entries: {len(train_entries)} -> {train_output}")
    print(f"Validation entries: {len(val_entries)} -> {val_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split manifest.json into train/val sets with class_id check")
    parser.add_argument("input", help="Path to input manifest.jsonl")
    parser.add_argument("--train", default="train_manifest.jsonl", help="Output training manifest")
    parser.add_argument("--val", default="val_manifest.jsonl", help="Output validation manifest")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Fraction for validation set")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    split_manifest(args.input, args.train, args.val, args.val_ratio, args.seed)
