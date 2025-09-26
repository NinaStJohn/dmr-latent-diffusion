from PIL import Image
import numpy as np
import os

def binarize(source_dir, destination_dir, threshold):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    for filename in os.listdir(source_dir):
        image = Image.open(os.path.join(source_dir, filename))
        gray_image = image.convert('L')
        gray_array = np.array(gray_image)
        normalized_array = gray_array / 255.0
        binary_array = (normalized_array > threshold).astype(np.uint8) * 255
        binary_image = Image.fromarray(binary_array)
        binary_image.save(os.path.join(destination_dir, "binarized_" + filename))


source_dir = r"..\data\images"
destination_dir = r"..\data\binarized_images"

binarize(source_dir, destination_dir, 0.65)