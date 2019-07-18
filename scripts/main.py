import sys
import unittest
from unittest import TestCase as test
sys.path.append('..')
import os
import yaml
import argparse
from shutil import copyfile
#import ipdb
from time import time
import pickle
import numpy as np
from src import datagen, utils, engine, evaluate, scoring
from src import forecastNet
import copy
from skorch import NeuralNetClassifier
from skorch.utils import noop
from skorch.callbacks import EpochScoring, BatchScoring
from sklearn.metrics import f1_score, roc_auc_score,make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import torch
import torch.nn as nn
from scipy.stats import expon

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ipdb

def main(config_file,debug,numT,n_iter,exp_id):
    # Parser config file
    with open(config_file) as f:
        config = yaml.load(f,Loader=yaml.FullLoader)

    main_exp_dir = os.path.join(config['output_dir'], exp_id)
    if(not os.path.exists(main_exp_dir)):
        os.mkdir(main_exp_dir)

    max_T = config['datagen']['max_T'] # Load data and get image paths
    num_classes = config['model']['module__task']['params']['num_classes'] 

    model_config = copy.deepcopy(config['model'])
    t = time()
    path_load = config['data'].pop('path_load')
    if os.path.exists(path_load):
        with open(path_load, 'rb') as f:
            data = pickle.load(f)
    else:
        data = datagen.get_data(**config['data'])
        with open(path_load, 'wb') as f:
            pickle.dump(data, f)
    print('Data Loaded : ', time()-t)
    print('Basic Data Stats:')
    print('Number of patients = ', len(data) - 2)

    # Datagens
    t = time()

    # Catch dataset_size % batch_size == 1 issue with batchnorm 1d:

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    datagen_all, datasets_all, data_train_size = \
            datagen.get_datagen(data, **config['datagen'])
    print('Datagens Loaded : ', time()-t)

    print('Dataset Length', datasets_all[0].__len__())
    class_wt = utils.get_classWeights(data, config['data']['train_ids_path'])
    print(class_wt)

    # Define sklearn wrapper and scoring function

    f1_scorer = make_scorer(scoring.f1_score, average = 'macro')
    auc = EpochScoring(scoring='roc_auc', lower_is_better=False)
    f1 = EpochScoring(scoring=f1_scorer,lower_is_better=False)

    clf_loss_train = BatchScoring(scoring.clf_loss_train, on_train=True, target_extractor=noop)
    aux_loss_train = BatchScoring(scoring.aux_loss_train, on_train=True, target_extractor=noop)

    clf_loss_valid = BatchScoring(scoring.clf_loss_valid, on_train=False, target_extractor=noop)
    aux_loss_valid = BatchScoring(scoring.aux_loss_valid, on_train=False, target_extractor=noop)

    if(debug):
        epochs = 1
    else:
        epochs = 60
    net = forecastNet.forecastNet(
            engine.Model, 
            device = device,
            callbacks = [f1, clf_loss_train, aux_loss_train, clf_loss_valid, aux_loss_valid],
            max_epochs = epochs,
            batch_size = 64,
            optimizer=torch.optim.Adam,
            criterion=nn.CrossEntropyLoss,
            optimizer__lr= .00120892,
            optimizer__weight_decay= .00130934,
            **model_config,
            module__device=device,
            module__class_wt=class_wt
        )

    X,Y = datasets_all[numT-1].return_all()
    print('X length: ', X.__len__())
    print('Y length: ', Y.__len__())
    
    net.fit(X,Y)

    create_loss_graphs(net,main_exp_dir,debug,T=numT)

def create_loss_graphs(net, main_exp_dir, debug, T):
    clf_loss_train = net.history[:,'clf_loss_train']
    aux_loss_train = net.history[:,'aux_loss_train']
    total_loss_train = net.history[:,'train_loss'] 

    clf_loss_valid = net.history[:,'clf_loss_valid']
    aux_loss_valid = net.history[:,'aux_loss_valid']
    total_loss_valid = net.history[:,'valid_loss'] 
    epochs = [i for i in range(len(net.history))]

    if(not debug):
        txtstr = '\n'.join((
            r'$\mathrm{lr}=%.8f$' % (net.optimizer__lr, ),
            r'$\mathrm{wd}=%.8f$' % (net.optimizer__weight_decay, ),
#            r'$\mathrm{epochs}=%d$' % (net.best_params_['max_epochs'], ),
            r'$\mathrm{bsize}=%d$' % (net.batch_size, ),
            r'$\mathrm{T}=%d$' % (T,), ))
#            r'$\mathrm{f1}=%.8f$' % (net.best_score_,) ))

    else:
        txtstr = '\n'.join((
            r'$\mathrm{lr}=%.8f$' % (net.optimizer__lr, ),
            r'$\mathrm{wd}=%.8f$' % (net.optimizer__weight_decay, ),
            r'$\mathrm{T}=%d$' % (T,) ))
          #  r'$\mathrm{f1}=%.8f$' % (net.best_score_,) ))


    # Set up bounding box parameters
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    f = plt.figure()
    axes = f.add_subplot(1,1,1)

    plt.plot(epochs, total_loss_train, c='g', label = 'Train Loss')
    plt.plot(epochs, total_loss_valid, c='r', label = 'Validation Loss')
    plt.title('Train and Test Loss Curves')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()

    # Place text box with best parameters
    plt.text(0.85,0.70,txtstr, ha = 'center', va = 'center', transform=axes.transAxes, bbox=props)
    plt.savefig(main_exp_dir + '/train_v_train.png', dpi = 300)
    plt.close()

    plt.figure()
    plt.plot(epochs, clf_loss_train, c='g', label = 'Train Loss Clf')
    plt.plot(epochs, aux_loss_train, c='b', label = 'Train Loss Aux')
    plt.title('Train Loss Curves')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(main_exp_dir + '/train_loss.png', dpi = 300)
    plt.close()


    plt.figure()
    plt.plot(epochs, clf_loss_valid, c='g', label = 'Valid Loss Clf')
    plt.plot(epochs, aux_loss_valid, c='b', label = 'Valid Loss Aux')
    plt.title('Validation Loss Curves')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(main_exp_dir + '/valid_loss.png', dpi = 300)
    plt.close()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../configs/config.yaml')
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--numT', type=int, default=1)
    parser.add_argument('--n_iter', type=int, default=30)
    parser.add_argument('--exp_id', type=str, default='debug')
    args = parser.parse_args()
    main(args.config,args.debug,args.numT,args.n_iter,args.exp_id)
   

