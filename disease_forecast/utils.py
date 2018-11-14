import os
import nibabel as nib

def load_img(path, view='axial'):
    return nib.load(path)

def create_dirs(paths):
    for path in paths:
        if os.path.exists(path)==False:
            os.mkdir(path)

