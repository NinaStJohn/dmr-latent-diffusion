import os, json, random, numpy as np
from pathlib import Path
from PIL import Image, ImageOps
from torch.utils.data import Dataset

VALID_RESIZE = {"keep_aspect", "letterbox", "square_crop"}

class CNTManifest(Dataset):
    def __init__(self, manifest, size,
                 paths=None,
                 random_crop=False,
                 edge_prob=0.35,               # <— new: chance a crop touches an edge
                 resize_mode="keep_aspect",
                 random_flip=False,
                 hflip_prob=0.5,               # adding in random h/v flips
                 vflip_prob=0.5,
                 return_meta=False, class_key="class_id", porosity_key="porosity"):
        self._bad_files = set()
        self.manifest = manifest
        self.edge_prob = float(edge_prob)
        self.random_flip = bool(random_flip)
        self.hflip_prob = float(hflip_prob)
        self.vflip_prob = float(vflip_prob)

        # --- compatibility: allow configs to pass `paths` instead of `manifest`
        if manifest is None and paths is not None:
            manifest = paths
        if manifest is None:
            raise TypeError("CNTManifest requires `manifest` (or legacy alias `paths`).")
        if size is None:
            raise TypeError("CNTManifest requires `size`.")

        # Interpret size
        if isinstance(size, int):
            target_h, target_w = size, size
        else:
            s = list(size)
            assert len(s) == 2
            target_h, target_w = int(s[0]), int(s[1])
        self.tH, self.tW = target_h, target_w
        self.WH = (self.tW, self.tH)

        self.random_crop = bool(random_crop)
        self.resize_mode = str(resize_mode)
        if self.resize_mode not in VALID_RESIZE:
            raise ValueError(f"resize_mode must be one of {VALID_RESIZE}, got {self.resize_mode}")

        self.return_meta = return_meta
        self.class_key = class_key
        self.porosity_key = porosity_key

        # --- Load and normalize paths ---
        man_dir = Path(self.manifest).resolve().parent
        self.records = []
        missing = 0
        with open(self.manifest, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                p = rec.get("path")
                if not p:
                    continue
                p = Path(p)
                if not p.is_absolute():
                    p = (man_dir / p).resolve()      # <— resolve relative to manifest folder
                if p.exists():
                    rec = dict(rec)
                    rec["path"] = str(p)
                    self.records.append(rec)
                else:
                    missing += 1
        if missing:
            print(f"[DATASET] Skipped {missing} missing file(s) while loading {self.manifest}", flush=True)
        if len(self.records) == 0:
            raise RuntimeError(f"No valid records found in {self.manifest}")

    def __len__(self):
        return len(self.records)



    def _edge_aware_crop_hwc(self, arr, out_h, out_w, edge_prob=0.35):
        H, W, C = arr.shape
        if H < out_h or W < out_w:
            # safety pad so we never short-slice; reflect so edges remain natural
            pad_h = max(0, out_h - H)
            pad_w = max(0, out_w - W)
            if pad_h or pad_w:
                # pad as (top, bottom), (left, right)
                top = pad_h // 2
                bottom = pad_h - top
                left = pad_w // 2
                right = pad_w - left
                arr = np.pad(arr, ((top, bottom), (left, right), (0, 0)), mode="reflect")
                H, W, _ = arr.shape

        # choose crop origin
        if random.random() < edge_prob:
            # pick a side: 0=top,1=bottom,2=left,3=right
            side = random.randint(0, 3)
            if side in (0, 1):  # top/bottom -> y fixed to edge; x random
                y0 = 0 if side == 0 else (H - out_h)
                x0 = 0 if W == out_w else random.randint(0, W - out_w)
            else:               # left/right -> x fixed to edge; y random
                x0 = 0 if side == 2 else (W - out_w)
                y0 = 0 if H == out_h else random.randint(0, H - out_h)
        else:
            y0 = 0 if H == out_h else random.randint(0, H - out_h)
            x0 = 0 if W == out_w else random.randint(0, W - out_w)

        return arr[y0:y0+out_h, x0:x0+out_w, :]


    def _load_resize_crop(self, path):
        img = Image.open(path).convert("RGB")
        iw, ih = img.size
        TW, TH = self.WH  # (W, H)

        if self.resize_mode in ("keep_aspect", "letterbox"):
            # scale so shorter side >= target, then crop to exact (TH,TW)
            scale = max(TW / iw, TH / ih)  # overscale
            nw, nh = int(round(iw * scale)), int(round(ih * scale))
            img = img.resize((nw, nh), Image.LANCZOS)

            arr = np.asarray(img, dtype=np.float32)  # H, W, C
            if self.random_crop:
                arr = self._edge_aware_crop_hwc(arr, TH, TW, edge_prob=self.edge_prob)
            else:
                # center crop fallback
                y0 = max((arr.shape[0] - TH) // 2, 0)
                x0 = max((arr.shape[1] - TW) // 2, 0)
                arr = arr[y0:y0+TH, x0:x0+TW, :]

        else:  # "square_crop"
            side = min(iw, ih)
            if self.random_crop:
                # choose square position; allow snapping to edges by biasing randint with edge_prob
                if random.random() < self.edge_prob:
                    # horizontally snap (left/right) or vertically snap (top/bottom)
                    if iw > ih:  # wider -> slide in x
                        x0 = 0 if random.random() < 0.5 else (iw - side)
                        y0 = 0 if side == ih else random.randint(0, ih - side)
                    else:        # taller -> slide in y
                        y0 = 0 if random.random() < 0.5 else (ih - side)
                        x0 = 0 if side == iw else random.randint(0, iw - side)
                else:
                    x0 = 0 if iw == side else random.randint(0, iw - side)
                    y0 = 0 if ih == side else random.randint(0, ih - side)
            else:
                x0 = (iw - side) // 2
                y0 = (ih - side) // 2

            img = img.crop((x0, y0, x0 + side, y0 + side))
            img = img.resize((TW, TH), Image.LANCZOS)
            arr = np.asarray(img, dtype=np.float32)

            # optional: one more edge-aware crop within the resized square
            if self.random_crop and (TH < arr.shape[0] or TW < arr.shape[1]):
                arr = self._edge_aware_crop_hwc(arr, TH, TW, edge_prob=self.edge_prob)

        # random flips (on final resized crop), before normalization
        if self.random_flip:
            if self.hflip_prob > 0.0 and random.random() < self.hflip_prob:
                arr = np.flip(arr, axis=1).copy()  # horizontal
            if self.vflip_prob > 0.0 and random.random() < self.vflip_prob:
                arr = np.flip(arr, axis=0).copy()  # vertical

        # normalize to [-1,1]
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
                    example["porosity_frac"] = np.nan if por is None else np.float32(por)
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