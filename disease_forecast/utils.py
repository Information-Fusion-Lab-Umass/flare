import os
import nibabel as nib
import numpy as np
import torch

def load_img(path, view='axial'):
    return nib.load(path)

def create_dirs(paths):
    for path in paths:
        if os.path.exists(path)==False:
            os.mkdir(path)

def one_hot(labels, C=5):
    N = len(labels)
    labels_onehot = torch.zeros([N, C], dtype=torch.float)
    labels = torch.squeeze(labels) - 1
    labels_onehot[torch.arange(N), labels.type(torch.long)] = 1
    return labels_onehot
