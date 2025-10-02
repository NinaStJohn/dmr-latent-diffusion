import json, os, random, numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset

VALID_RESIZE = {"keep_aspect", "letterbox", "square_crop"}

class CNTManifest(Dataset):
    def __init__(self, manifest, size,
                 random_crop=False,
                 resize_mode="keep_aspect",   # "keep_aspect" | "letterbox" | "square_crop"
                 return_meta=False, class_key="class_id", porosity_key="porosity"):
        self._bad_files = set()
        self.manifest = manifest

        # Interpret size robustly: allow int, (H,W), or (W,H)
        if isinstance(size, int):
            target_h, target_w = size, size
        else:
            try:
                s = list(size)          # works for ListConfig, tuple, list
                assert len(s) == 2
                target_h, target_w = int(s[0]), int(s[1])
            except Exception:
                raise ValueError("size must be int or (H,W)")
        self.tH, self.tW = target_h, target_w
        self.WH = (self.tW, self.tH)

        self.random_crop = bool(random_crop)
        self.resize_mode = str(resize_mode)
        if self.resize_mode not in VALID_RESIZE:
            raise ValueError(f"resize_mode must be one of {VALID_RESIZE}, got {self.resize_mode}")

        self.return_meta = return_meta
        self.class_key = class_key
        self.porosity_key = porosity_key

        # Load JSONL lines into self.records
        self.records = []
        with open(self.manifest, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                p = rec.get("path")
                if p and os.path.exists(p):
                    self.records.append(rec)
                else:
                    # optional: print missing paths once
                    # print(f"[MISSING] {p}")
                    pass

        if len(self.records) == 0:
            raise RuntimeError(f"No valid records found in {self.manifest}")

    def __len__(self):
        return len(self.records)

    def _load_resize_crop(self, path):
        img = Image.open(path).convert("RGB")
        iw, ih = img.size
        TW, TH = self.WH  # (W,H)

        if self.resize_mode == "keep_aspect":
            # Direct scale to target (assumes source is already 3:2-ish and target matches)
            img = img.resize((TW, TH), Image.LANCZOS)

        elif self.resize_mode == "letterbox":
            scale = min(TW / iw, TH / ih)
            nw, nh = int(round(iw * scale)), int(round(ih * scale))
            img = img.resize((nw, nh), Image.LANCZOS)
            pad_w, pad_h = TW - nw, TH - nh
            img = ImageOps.expand(
                img,
                border=(pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2),
                fill=0,
            )

        else:  # "square_crop"
            side = min(iw, ih)
            if self.random_crop:
                x0 = 0 if iw == side else random.randint(0, iw - side)
                y0 = 0 if ih == side else random.randint(0, ih - side)
            else:
                x0 = (iw - side) // 2
                y0 = (ih - side) // 2
            img = img.crop((x0, y0, x0 + side, y0 + side))
            img = img.resize((TW, TH), Image.LANCZOS)

        arr = np.asarray(img, dtype=np.float32)  # H, W, C
        arr = (arr / 127.5) - 1.0
        return arr

    def __getitem__(self, idx):
        rec = self.records[idx]

        # Try current record, then one fallback
        for attempt in range(2):
            try:
                image = self._load_resize_crop(rec["path"])
                example = {"image": image}
                if self.class_key in rec:
                    example["class_id"] = int(rec[self.class_key])
                if self.porosity_key in rec:
                    por = rec[self.porosity_key]
                    example["porosity"] = float('nan') if por is None else float(por)
                if self.return_meta:
                    example["file_path_"] = rec["path"]
                return example
            except Exception as e:
                bad = rec.get("path", f"<idx:{idx}>")
                if bad not in self._bad_files:
                    # Print exact error once to help debugging
                    print(f"[DATA ERROR] {bad} :: {type(e).__name__}: {e}", flush=True)
                self._bad_files.add(bad)
                # advance to next record
                idx = (idx + 1) % len(self.records)
                rec = self.records[idx]

        raise RuntimeError("Multiple unreadable files detected; dataset likely misconfigured (see [DATA ERROR] above).")
