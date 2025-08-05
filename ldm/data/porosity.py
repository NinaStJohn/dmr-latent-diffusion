import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import image


def calculate_image_porosity(img_tensor):
    """
    Calculate porosity from an image tensor
    
    Args:
        img_tensor: Tensor of shape [C, H, W] with values in [-1, 1]
        
    Returns:
        Porosity value (scalar) between 0 and 1
    """
    # Convert to [0, 1] range
    img = (img_tensor + 1) / 2
    
    # Convert to grayscale if RGB
    if img.size(0) == 3:
        # Use luminance formula
        img_gray = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
    else:
        img_gray = img.squeeze(0)
    
    # Apply threshold to create binary image (pores are assumed to be dark)
    # Adjust threshold if needed for your specific microstructure images
    threshold = 0.5
    binary = (img_gray < threshold).float()
    
    # Calculate porosity as ratio of pore pixels to total pixels
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
        """Calculate porosity for all images in the dataset"""
        porosities = []
        for i in range(len(self.dataset)):
            if isinstance(self.dataset[i], tuple):
                img = self.dataset[i][0]
            else:
                img = self.dataset[i]
            porosity = calculate_image_porosity(img)
            porosities.append(porosity.item())
        return porosities

    def _group_by_porosity(self):
        """Group indices by similar porosity values"""
        indices_by_porosity = {}
        
        # Round porosities to group similar values
        for i, porosity in enumerate(self.porosities):
            rounded = round(porosity / self.tolerance) * self.tolerance
            if rounded not in indices_by_porosity:
                indices_by_porosity[rounded] = []
            indices_by_porosity[rounded].append(i)
            
        return indices_by_porosity
    
    def __iter__(self):
        # Create batches with similar porosity
        batches = []
        
        # Get list of porosity groups
        porosity_groups = list(self.indices_by_porosity.keys())
        
        # Shuffle the order of porosity groups
        np.random.shuffle(porosity_groups)
        
        for porosity in porosity_groups:
            indices = self.indices_by_porosity[porosity]
            # Shuffle indices within each porosity group
            np.random.shuffle(indices)
            
            # Create batches from this porosity group
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                # If we don't have enough samples to form a complete batch
                if len(batch_indices) < self.batch_size:
                    # Fill remaining spots with random samples from other groups
                    remaining = self.batch_size - len(batch_indices)
                    all_other_indices = []
                    for p in porosity_groups:
                        if p != porosity:
                            all_other_indices.extend(self.indices_by_porosity[p])
                    
                    if len(all_other_indices) >= remaining:
                        np.random.shuffle(all_other_indices)
                        batch_indices.extend(all_other_indices[:remaining])
                    else:
                        # Not enough samples, just duplicate some
                        while len(batch_indices) < self.batch_size:
                            batch_indices.append(np.random.choice(batch_indices))
                
                batches.append(batch_indices)
        
        # Shuffle the order of batches
        np.random.shuffle(batches)
        
        # Flatten batches to yield indices
        for batch in batches:
            for idx in batch:
                yield idx
    
    def __len__(self):
        return len(self.dataset)