from PIL import Image
import numpy as np
import os

def jpeg_to_npy(image_path, threshold=0.65*256):
    image = Image.open(image_path).convert('L')
    image = image.resize((1536, 1024))
    image_array = np.array(image)
    print(image_array.max())
    binarized_array = (image_array > threshold).astype(np.uint8)
    return binarized_array

directory_path = 'data/img_together'
save_path = 'MCRpy/microstructure_data_1536_1024'
os.makedirs(save_path, exist_ok=True)

for filename in os.listdir(directory_path):
    npy_array = jpeg_to_npy(os.path.join(directory_path, filename))
    np.save('{}.npy'.format(os.path.join(save_path, filename.split('.')[0])), npy_array)

print(npy_array)
