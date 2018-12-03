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

def get_output(cmat, exp_dir, task='dx'):
    (num_T, num_gap) = cmat.shape
    if task=='dx':
        cmat_out = np.zeros((num_T*3, num_gap*3))
        for t in range(num_T):
            for tau in range(num_gap):
                if cmat[t, tau] != None:
                    cmat_out[t*3:(t+1)*3, tau*3:(tau+1)*3] = cmat[t, tau].cmat
        savemat(exp_dir+'/output.mat', {'cmat':cmat_out})

        fig, ax = plt.subplots()
        im = ax.imshow(cmat_out)
        ax.set_xticks(np.arange(num_gap)+1)
        ax.set_yticks(np.arange(num_T)+1)
        for i in range(num_T):
            for j in range(num_gap):
                text = ax.text(j, i, cmat_out[i, j],\
                        ha="center", va="center", color="w")
                ax.set_title("Confusion Matrix")
                fig.tight_layout()
        plt.savefig(exp_dir+'/output.png', dpi=300)

def confmatrix_dx(ypred, y):
    (N, C) = ypred.shape
    _, idx = torch.max(ypred, 1)
    
    y_pred = idx.data.numpy()
    y = y.data.numpy()

    cmat = confusion_matrix(y, y_pred, labels=[0,1,2])
    cmat = cmat/np.sum(cmat, axis=1)[:,np.newaxis]
    return cmat

