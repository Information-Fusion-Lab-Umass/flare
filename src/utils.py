import os
import nibabel as nib
import numpy as np
import torch
import pickle
from src import evaluate

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

# Aggregate results utils

def filter_None(mat_list):
    '''
    Filters out None Type elements in a list of matrices using
    list comprehension

    Input:
        mat_list (list): list of arrays where some entries
                may be None

    Returns:
        list of matrices with 'None' elements replaced with 0

    '''            
    return [item if np.any(item) else 0 for item in mat_list]

def calculate_averages(confmats):
    '''
    Take a list of confusion matrices from multiple experiments and returns average f1 scores and accuracies

    Input:
        confmats (list): list of confusion matrices

    Returns:
        agg_metrics (dict): dictionary of average metrics: accuracy, f1 score, precision, recall, and counts.
    '''
    agg_metrics = {}

    agg_metrics['precision'] = np.mean([mat.precision for mat in confmats],axis=0)
    agg_metrics['recall'] = np.mean([mat.recall for mat in confmats],axis=0)
    agg_metrics['f1'] = (2*agg_metrics['precision']*agg_metrics['recall'])/(agg_metrics['precision']+agg_metrics['recall'])
    
    accuracies = [[filter_None(item) for item in mats_T.probs] for mats_T in confmats]

    counts = [[filter_None(item) for item in mats_T.probs] for mats_T in confmats]

    agg_metrics['accuracy'] = np.mean(accuracies,axis=0)
    agg_metrics['counts'] = np.mean(counts,axis=0)

    return agg_metrics

