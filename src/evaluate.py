import numpy as np
import torch 
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import savemat

class cmatCell:
    def __init__(self, cmat):
        self.cmat = cmat

def get_output(cmat, exp_dir, data_name, task = 'dx', num_classes = 3):
    (num_T, num_gap) = cmat.shape
    if task == 'dx':
        cmat_out = np.zeros((num_T*num_classes, num_gap*num_classes))
        for t in range(num_T):
            for tau in range(num_gap):
                if cmat[t, tau] != None:
                    cmat_out[t*num_classes:(t+1)*num_classes, \
                            tau*num_classes:(tau+1)*num_classes] = cmat[t, tau].cmat
        savemat(exp_dir+'/results/confmatrix_'+data_name+'.mat', {'cmat':cmat_out})

        fig, ax = plt.subplots()
        im = ax.imshow(cmat_out)
        for i in range(num_T*num_classes):
            for j in range(num_gap*num_classes):
                text = ax.text(j, i, round(cmat_out[i, j]*100, 1), \
                        ha="center", va="center", color="w")
        ax.set_title("Confusion Matrix")
        fig.tight_layout()
        plt.savefig(exp_dir+'/results/confmatrix_'+data_name+'.png', dpi=300)

def confmatrix_dx(ypred, y, num_classes):
    (N, C) = ypred.shape
    _, idx = torch.max(ypred, 1)
    
    y_pred = idx.data.numpy()
    y = y.data.numpy()

    cmat = confusion_matrix(y, y_pred, labels=list(range(num_classes)))
    cmat = cmat/(np.sum(cmat, axis=1)[:,np.newaxis]+0.001)
    return cmat

