import numpy as np
import torch 
from sklearn.metrics import confusion_matrix

def confmatrix_dx(ypred, y):
    (N, C) = ypred.shape
    _, idx = torch.max(ypred, 1)
    
    y_pred = idx.data.numpy()
    y = y.data.numpy()

    cmat = confusion_matrix(y, y_pred, labels=[0,1,2])
    cmat = cmat/np.sum(cmat, axis=1)[:,np.newaxis]
    return cmat

