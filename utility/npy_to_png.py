import os
import numpy as np
from PIL import Image

def npy_to_pil(npy_array):
    npy_array = (npy_array * 255).astype(np.uint8)
    return Image.fromarray(npy_array, mode='L')

def convert_npy_folder_to_images(source_folder, target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for file_name in os.listdir(source_folder):
        if file_name.endswith('.npy'):
            npy_path = os.path.join(source_folder, file_name)
            npy_array = np.load(npy_path)
            image = npy_to_pil(npy_array)
            output_path = os.path.join(target_folder, file_name.replace('.npy', '.png'))
            image.save(output_path)

source_folder = 'MCRpy/mcrpy/results_1536_1024_Corr'
target_folder = 'MCRpy/mcrpy/results_1536_1024_Corr/img'
convert_npy_folder_to_images(source_folder, target_folder)
