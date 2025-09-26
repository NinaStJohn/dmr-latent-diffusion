import os
import shutil

def combine(dir1, dir2):
    for class_name in os.listdir(dir2):
        for img_file in os.listdir(os.path.join(dir2, class_name)):
            if os.path.isfile(os.path.join(dir2, class_name, img_file)):
                shutil.move(os.path.join(dir2, class_name, img_file), os.path.join(dir1, class_name))

dir1 = 'data/images'
dir2 = 'data/images_2'

combine(dir1, dir2)