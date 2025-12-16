import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image  # <- fixed

def calculate_image_porosity(img_tensor):
    # Convert [-1,1] -> [0,1]
    img = (img_tensor + 1) / 2
    # Grayscale
    if img.size(0) == 3:
        img_gray = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
    else:
        img_gray = img.squeeze(0)
    # Threshold (pores dark)
    binary = (img_gray < 0.5).float()
    porosity = binary.sum() / (binary.size(0) * binary.size(1))
    return porosity

class PorosityDataset(Dataset):
    def __init__(self, dataset, batch_size, porosity_tolerance=.5):
        self.dataset = dataset
        self.batch_size = batch_size
        self.tolerance = porosity_tolerance
        self.porosities = self._calculate_porosities()
        self.indices_by_porosity = self._group_by_porosity()

    def _calculate_porosities(self):
        porosities = []
        for i in range(len(self.dataset)):
            item = self.dataset[i]
            img = item[0] if isinstance(item, tuple) else item
            porosity = calculate_image_porosity(img)
            porosities.append(float(porosity.item()))
        return porosities

    def _group_by_porosity(self):
        indices_by_porosity = {}
        for i, por in enumerate(self.porosities):
            rounded = round(por / self.tolerance) * self.tolerance
            indices_by_porosity.setdefault(rounded, []).append(i)
        return indices_by_porosity

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # simple passthrough so index access returns a sample
        item = self.dataset[idx]
        img = item[0] if isinstance(item, tuple) else item
        return img

    def __iter__(self):
        # NOTE: This yields indices, not samples — only use with a custom Sampler.
        batches = []
        porosity_groups = list(self.indices_by_porosity.keys())
        np.random.shuffle(porosity_groups)
        for porosity in porosity_groups:
            indices = self.indices_by_porosity[porosity]
            np.random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                if len(batch_indices) < self.batch_size:
                    remaining = self.batch_size - len(batch_indices)
                    all_other = []
                    for p in porosity_groups:
                        if p != porosity:
                            all_other.extend(self.indices_by_porosity[p])
                    if len(all_other) >= remaining:
                        np.random.shuffle(all_other)
                        batch_indices.extend(all_other[:remaining])
                    else:
                        while len(batch_indices) < self.batch_size:
                            batch_indices.append(np.random.choice(batch_indices))
                batches.append(batch_indices)
        np.random.shuffle(batches)
        for batch in batches:
            for idx in batch:
                yield idx
