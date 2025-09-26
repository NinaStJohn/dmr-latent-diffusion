import json
from pathlib import Path
from typing import Dict, Any, List

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


def _load_rgb(path: str) -> Image.Image:
    """
    Robustly load an image and return a PIL RGB image.
    """
    p = Path(path)
    with Image.open(p) as im:
        # Convert anything (L/LA/CMYK/float TIFF) to RGB
        return im.convert("RGB")


class ManifestClassImageDataset(Dataset):
    """
    JSONL manifest with at least:
      {"path": ".../image.tif", "class_id": <int>}

    Returns a dict with:
      - "image":  FloatTensor [3, H, W] normalized to [-1, 1]
      - "class_label": LongTensor [] with the class index
    """
    def __init__(
        self,
        manifest: str,
        size: int = 256,
        label_key: str = "class_id",
        out_label_key: str = "class_label",
        random_crop: bool = True,
    ):
        self.entries: List[Dict[str, Any]] = []
        with open(manifest, "r") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                self.entries.append(json.loads(ln))

        self.label_key = label_key
        self.out_label_key = out_label_key

        # Transform: resize -> crop -> to tensor -> normalize to [-1, 1]
        crop_tf = T.RandomCrop(size) if random_crop else T.CenterCrop(size)
        self.tf = T.Compose([
            T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
            crop_tf,
            T.ToTensor(),                               # [0,1]
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # -> [-1,1]
        ])

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.entries[idx]
        img = _load_rgb(rec["path"])
        img = self.tf(img)  # FloatTensor [3,H,W] in [-1,1]

        # Class label as LongTensor (required by nn.Embedding)
        label = int(rec[self.label_key])
        label_t = torch.tensor(label, dtype=torch.long)

        return {"image": img, self.out_label_key: label_t}
