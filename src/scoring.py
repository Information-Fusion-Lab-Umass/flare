from sklearn.metrics import fbeta_score, precision_recall_fscore_support
import torch
import numpy as np

'''
Some modified scoring functions in order to make skorch compatible with some quirks of pytorch
'''

def f1_score(y_true, y_pred, labels=None, pos_label=1, average='binary',sample_weight=None):
    y_true = y_true[:len(y_pred)]
    return fbeta_score(y_true, y_pred, 1, labels=labels,
                       pos_label=pos_label, average=average,
                       sample_weight=sample_weight)

def aux_loss_train(net, X=None, y=None):
    return net.history[-1,'batches',-1,'aux_loss_train']

def clf_loss_train(net, X=None, y=None):
    return net.history[-1,'batches',-1,'clf_loss_train']

def aux_loss_valid(net, X=None, y=None):
    return net.history[-1,'batches',-1,'aux_loss_valid']

def clf_loss_valid(net, X=None, y=None):
    return net.history[-1,'batches',-1,'clf_loss_valid']
