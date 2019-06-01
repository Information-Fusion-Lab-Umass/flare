import os
import nibabel as nib
import numpy as np
import torch
import pickle

def load_img(path, view='axial'):
    if(path[-3:] == 'nii'):
        return nib.load(path)
    else:
        try:
            with open(path,'rb') as f:
                image = pickle.load(f)
                return image
        except IOError:
            print("image does not exist in path")

def create_dirs(paths):
    for path in paths:
        if os.path.exists(path)==False:
            os.mkdir(path)

def one_hot(labels, device, C = 5):
    N = len(labels)
    labels_onehot = torch.zeros([N, C], dtype=torch.float)
    labels = torch.squeeze(labels) - 1
    labels_onehot[torch.arange(N), labels.type(torch.long)] = 1
    return labels_onehot.to(device)

# Data utils
def is_consec(seq):
    '''
    Checks if the iterable 'seq' has consecutive entries
    '''
    return sorted(seq) == list(range(min(seq),max(seq)+1))

def return_consec(listseq):
    '''
    Given a list of sequences listseq, return only the sequences seq such
    that seq[:-1] is consecutive.
    '''
    consec = []
    for seq in listseq:
        if(is_consec(seq[:-1])):
            consec.append(seq)
    return consec

def get_classWeights(data, ids_path):
    train_ids = np.loadtxt(ids_path, dtype = str)
    num_nl = 0; num_mci = 0; num_ad = 0
    for pid in train_ids:
        if pid in data:
            visits = data[pid].visits
            labels = [visits[key].data['labels'][0] for key in visits]
            num_nl += labels.count(0)
            num_mci += labels.count(1)
            num_ad += labels.count(2)
    max_count = max(num_nl, num_mci, num_ad)
    print(num_nl, num_mci, num_ad, max_count)
    return [max_count/num_nl, max_count/num_mci, max_count/num_ad]
            






