import json, os, random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class CNTManifest(Dataset):
    def __init__(self, manifest, size, random_crop=false,
                 resize_mode="keep_aspect", # "keep_aspect" | "letterbox" | "square_crop" 
                 return_meta=False, class_key="class_id", porosity_key="porosity"):
        self._bad_files = set()
        self.manifest = manifest
        self.w, self.h = size if isinstance(size, (list, tuple)) else (size, size)
        self.random_crop = random_crop
        self.return_meta = return_meta
        self.class_key = class_key
        self.porosity_key = porosity_key

        # Load JSONL lines into self.records
        self.records = []
        with open(self.manifest, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if os.path.exists(rec["path"]):
                    self.records.append(rec)

        if len(self.records) == 0:
            raise RuntimeError(f"No valid records found in {self.manifest}")

    def __len__(self):
        return len(self.records)

    # allows for multiple aspect ratios & resizing
    def _load_resize_crop(self, path):
        img = Image.open(path).convert("RGB")
        tw, th = self.w, self.h

        if self.resize_mode == "keep_aspect":
            # direct 3:2 → 3:2 scaling (no crop, no pad)
            img = img.resize((tw, th), Image.LANCZOS)

        elif self.resize_mode == "letterbox":
            # preserve content; pad to (tw, th) without cropping
            iw, ih = img.size
            scale = min(tw / iw, th / ih)
            nw, nh = int(round(iw * scale)), int(round(ih * scale))
            img = img.resize((nw, nh), Image.LANCZOS)
            pad_w, pad_h = tw - nw, th - nh
            img = ImageOps.expand(img,
                                  border=(pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2),
                                  fill=0)

        else:  # "square_crop" (legacy behavior)
            side = min(img.width, img.height)
            if self.random_crop:
                x0 = 0 if img.width == side else random.randint(0, img.width - side)
                y0 = 0 if img.height == side else random.randint(0, img.height - side)
            else:
                x0 = (img.width - side) // 2
                y0 = (img.height - side) // 2
            img = img.crop((x0, y0, x0 + side, y0 + side))
            img = img.resize((tw, th), Image.LANCZOS)

        arr = np.asarray(img, dtype=np.float32)
        arr = (arr / 127.5) - 1.0
        return arr


    def __getitem__(self, idx):
        rec = self.records[idx]

        # Robust load with one retry on the next record
        for _ in range(2):
            try:
                image = self._load_resize_crop(rec["path"])  # H, W, C
                break
            except Exception as e:
                bad = rec.get("path", f"<idx:{idx}>")
                self._bad_files.add(bad)
                print(f"[SKIP BAD] {bad} :: {type(e).__name__}: {e}", flush=True)
                # advance to next record and try once more
                idx = (idx + 1) % len(self.records)
                rec = self.records[idx]
        else:
            raise RuntimeError("Multiple unreadable files detected; dataset likely corrupted.")

        example = {"image": image}

        if self.class_key in rec:
            example["class_id"] = int(rec[self.class_key]) 
        if self.porosity_key in rec:
            # allow None -> NaN if present
            por = rec[self.porosity_key]
            example["porosity"] = float('nan') if por is None else float(por)
        if self.return_meta:
            example["file_path_"] = rec["path"]
        return example


