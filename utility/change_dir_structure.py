import os
import shutil

def change_directory_structure(source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for dir_name in os.listdir(source_dir):
        for filename in os.listdir(os.path.join(source_dir, dir_name)):
            shutil.move(os.path.join(source_dir, dir_name, filename), target_dir)

change_directory_structure('data/images_jpg', 'data/img_together')