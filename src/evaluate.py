import numpy as np
import torch 
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
import matplotlib
matplotlib.use('Agg')
import os
import matplotlib.pyplot as plt
import pylab
from scipy.io import savemat

class ConfMatrix:
    def __init__(self, numT, num_classes):
        self.numT = numT
        self.num_classes = num_classes
        self.probs = np.empty((numT, numT), dtype=object)
        self.counts = np.empty((numT, numT), dtype=object)
        self.f1 = np.zeros((numT, numT))
        self.precision = np.zeros((numT, numT))
        self.recall = np.zeros((numT, numT))

    def set_vals(self, agg_metrics):
        self.probs = agg_metrics['accuracy']
        self.counts = agg_metrics['counts']
        self.f1 = agg_metrics['f1']
        self.precision = agg_metrics['precision']
        self.recall = agg_metrics['recall']

    def update(self, T, tau, ypred, y):
        (N, C) = ypred.shape
        _, idx = torch.max(ypred, 1)
        
        y_pred = idx.data.numpy()
        y = y.data.numpy()

        counts = confusion_matrix(y, y_pred, labels = list(range(self.num_classes)))
        probs = counts/(np.sum(counts, axis=1)[:,np.newaxis]+0.001)
        f1 = f1_score(y, y_pred, labels = list(range(self.num_classes)), 
                average = 'macro')
        precision, recall, _, _ = precision_recall_fscore_support(y, y_pred, labels = list(range(self.num_classes)), average = 'macro')
        self.counts[T, tau] = counts
        self.probs[T, tau] = probs
        self.f1[T, tau] = f1
        self.precision[T,tau] = precision
        self.recall[T,tau] = recall
        
    def save(self, exp_dir, filename):
        T, C, Tau = self.numT, self.num_classes, self.numT
        probs = np.zeros((T*C, Tau*C))
        counts = np.zeros((T*C, Tau*C))

        for t in range(T):
            for tau in range(Tau):
                if isinstance(self.probs[t, tau], np.ndarray):
                    probs[t*C:(t+1)*C, tau*C:(tau+1)*C] = self.probs[t, tau]
                    counts[t*C:(t+1)*C, tau*C:(tau+1)*C] = self.counts[t, tau]
                    #print(t, tau, self.f1[t, tau], self.precision[t,tau], self.recall[t,tau])
        savemat(
            os.path.join(exp_dir, 'results', 'confmatrix_' + filename + '.mat'), 
            {'probs' : probs, 'counts' : counts}
            )

        # Probability image
        fig, ax = plt.subplots()
        im = ax.imshow(probs)

        for i in range(T*C):
          for j in range(Tau*C):
              text = ax.text(j, i, round(probs[i, j]*100, 1), \
                      ha="center", va="center", color="w", fontsize = 8)
        ax.set_title("Confusion Matrix (Probabilities)")
        fig.tight_layout()
        plt.savefig(os.path.join(exp_dir, 'results', 'confmatrix_' \
                + filename + '.png'), dpi=300)
        plt.close(fig)
        
        # Counts image
        fig, ax = plt.subplots()
        im = ax.imshow(probs)
        #  for i in range(T*C):
        #      for j in range(Tau*C):
        #          text = ax.text(j, i, int(counts[i, j]), \
        #                  ha="center", va="center", color="w", fontsize = 8)
        ax.set_title("Confusion Matrix (Frequencies)")
        fig.tight_layout()
        plt.savefig(os.path.join(exp_dir, 'results', 'confmatrix_counts_' \
                + filename + '.png'), dpi=300)
        plt.close(fig)

        # Save F1 scores
        fig, ax = plt.subplots()
        col_labels = ['tau = ' + str(i+1) for i in range(T)]
        row_labels = ['T = ' + str(i+1) for i in range(T)]
        ax.set_title('F1 scores')
        fig.tight_layout()
        ax.axis('off')
        ax.table(cellText = np.round(self.f1, 4), colLabels = col_labels, \
                rowLabels = row_labels, loc = 'center')
        plt.savefig(os.path.join(exp_dir, 'results', 'f1_' \
                + filename + '.png'), dpi=300)
        plt.close(fig)

class LossVals:
    def __init__(self, num_epochs, validation_period, num_T):
        self.num_epochs = int(num_epochs)
        self.validation_period = validation_period
        self.num_T = int(num_T)

        # Train Loss
        self.train_loss_T = {}
        for loss_type in ['clfLoss', 'auxLoss', 'totalLoss']:
            self.train_loss_T[loss_type] = np.zeros((num_epochs, num_T))
        self.train_loss = {}
        for loss_type in ['clfLoss', 'auxLoss', 'totalLoss']:
            self.train_loss[loss_type] = np.zeros((num_epochs))

        # Validation Loss
        self.num_val = int(num_epochs / validation_period)
        self.val_loss_T = {}
        for loss_type in ['clfLoss', 'auxLoss', 'totalLoss']:
            self.val_loss_T[loss_type] = np.zeros((self.num_val, num_T))
        self.val_loss = {}
        for loss_type in ['clfLoss', 'auxLoss', 'totalLoss']:
            self.val_loss[loss_type] = np.zeros((self.num_val))

        self.iterations = {
                'train': np.zeros((num_T)),
                'val': np.zeros((num_T))
                }
        self.loss_T = {
                'train': self.train_loss_T,
                'val': self.val_loss_T
                }
        self.loss = {
                'train': self.train_loss,
                'val': self.val_loss
                }

    def update_T(self, data_type, loss, epoch, idx, step):
        clfLoss = float(loss[0])/step
        auxLoss = float(loss[1])/step
        totalLoss = clfLoss + auxLoss
        self.iterations[data_type][idx] = step
        self.loss_T[data_type]['clfLoss'][epoch, idx] = clfLoss
        self.loss_T[data_type]['auxLoss'][epoch, idx] = auxLoss
        self.loss_T[data_type]['totalLoss'][epoch, idx] = totalLoss

    def update(self, data_type, epoch):
        iterations = self.iterations[data_type]
        sum_iterations = np.sum(iterations)

        for loss_type in ['clfLoss', 'auxLoss', 'totalLoss']:
            lossval = self.loss_T[data_type][loss_type][epoch, :]
            self.loss[data_type][loss_type][epoch] = \
                    iterations.dot(lossval)/sum_iterations

    def plot_graphs(self, path, num_graphs = 6):
        xaxis = np.arange(self.num_epochs)
        # Train Graph : comparison of aux, clf and total losses
        plt.figure()
        plt.plot(xaxis, self.train_loss['clfLoss'], c = 'b', label = 'Classifier loss')
        plt.plot(xaxis, self.train_loss['auxLoss'], c = 'r', label = 'Auxiliary loss')
        plt.plot(xaxis, self.train_loss['totalLoss'], c = 'g', label = 'Total loss')
        plt.legend()
        plt.title('Train Loss : Classifier, Auxiliary and Total')
        plt.savefig(os.path.join(path, 'train_loss_1.png'), dpi = 300)
        plt.close()

        # Train Graph : comparison of datagen losses
        plt.figure()
        cm = plt.get_cmap('nipy_spectral')
        colors = [cm(1.*i/num_graphs) for i in range(num_graphs)]
        for T in range(self.num_T):
            plt.plot(xaxis, self.train_loss_T['totalLoss'][:, T], \
                    c = colors[T], label = 'T = '+str(T+1))
        plt.legend()
        plt.title('Train Loss : Total Loss of datagens')
        plt.savefig(os.path.join(path, 'train_loss_2.png'), dpi = 300)
        plt.close()

        # Val Graph : comparison of aux, clf and total losses
        xaxis = np.arange(len(self.val_loss['clfLoss']))
        plt.figure()
        plt.plot(xaxis, self.val_loss['clfLoss'], c = 'b', label = 'Classifier loss')
        plt.plot(xaxis, self.val_loss['auxLoss'], c = 'r', label = 'Auxiliary loss')
        plt.plot(xaxis, self.val_loss['totalLoss'], c = 'g', label = 'Total loss')
        plt.legend()
        plt.title('Test Loss : Classifier, Auxiliary and Total')
        plt.savefig(os.path.join(path, 'test_loss_1.png'), dpi = 300)
        plt.close()

        # Val Graph : comparison of datagen losses
        plt.figure()
        for T in range(self.num_T):
            plt.plot(xaxis, self.val_loss_T['totalLoss'][:, T], \
                    c = colors[T], label = 'T = '+str(T+1))
        plt.legend()
        plt.title('Test Loss : Total Loss of datagens')
        plt.savefig(os.path.join(path, 'test_loss_2.png'), dpi = 300)
        plt.close()       

