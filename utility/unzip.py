import os
import zipfile

def unzip_files(source_dir, destination_dir):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    for filename in os.listdir(source_dir):
        if filename.endswith('.zip'):
            zip_path = os.path.join(source_dir, filename)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(destination_dir)

source_dir = r"..\data\zip_2"
destination_dir = r"..\data\images_2"

unzip_files(source_dir, destination_dir)

