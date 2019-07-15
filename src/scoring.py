from sklearn.metrics import fbeta_score, precision_recall_fscore_support
import torch
import numpy as np

'''
Some modified scoring functions in order to make skorch compatible with some quirks of pytorch
'''

def drop_last_f1_score(y_true, y_pred, labels=None, pos_label=1, average='binary',sample_weight=None):
    y_true = y_true[:len(y_pred)]
    return fbeta_score(y_true, y_pred, 1, labels=labels,
                       pos_label=pos_label, average=average,
                       sample_weight=sample_weight)


