import os
from PIL import Image

def convert_images(source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for root, dirs, files in os.walk(source_dir):
        relative_path = os.path.relpath(root, source_dir)
        target_root = os.path.join(target_dir, relative_path)
        
        if not os.path.exists(target_root):
            os.makedirs(target_root)
        
        for file in files:
            if file.endswith('.tif') or file.endswith('.tiff'):
                source_file = os.path.join(root, file)
                target_file = os.path.join(target_root, os.path.splitext(file)[0] + '.jpeg')
                
                try:
                    with Image.open(source_file) as img:
                        img.convert('RGB').save(target_file, 'JPEG')
                    print(f"Converted {source_file} to {target_file}")
                except Exception as e:
                    print(f"Failed to convert {source_file}: {e}")

source_dir = r'..\data\binarized_images'
target_dir = r'..\data\binarized_images_jpg'

convert_images(source_dir, target_dir)