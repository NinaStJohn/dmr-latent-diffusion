import os
import shutil

def make_folders(source_dir):
    for filename in os.listdir(source_dir):
        if not filename.endswith('.tif'):
            continue
        
        underscore_index = filename.rfind('_', 0, filename.rfind('_')-1)
        percent_index = filename.index('%')
        percentage = filename[underscore_index + 1:percent_index + 1]
        
        target_dir = os.path.join(source_dir, percentage)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        
        source_path = os.path.join(source_dir, filename)
        target_path = os.path.join(target_dir, filename)
        shutil.move(source_path, target_path)

source_dir = r"..\data\images_2"
make_folders(source_dir)
