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
from skorch.callbacks import EpochScoring
from sklearn.metrics import f1_score, roc_auc_score,make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import torch
import torch.nn as nn
from scipy.stats import expon

import ipdb

def main(config_file,debug,numT):
    # Parser config file
    with open(config_file) as f:
        config = yaml.load(f,Loader=yaml.FullLoader)

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

    f1_scorer = make_scorer(scoring.drop_last_f1_score, average = 'macro')
    auc = EpochScoring(scoring='roc_auc', lower_is_better=True)
    f1 = EpochScoring(scoring=f1_scorer,lower_is_better=False)

    if(debug):
        net = forecastNet.forecastNet(
                engine.Model, 
                max_epochs=1,
                batch_size=128,
                device = device,
                callbacks = [f1],
                optimizer=torch.optim.Adam,
                criterion=nn.CrossEntropyLoss,
                **model_config,
                module__device=device,
                module__class_wt=class_wt
            )

        params = {
                    'optimizer__lr': expon(scale = 0.01),
                    'optimizer__weight_decay': expon(scale=0.01)
                 }
    else:
        print('Test false')
        net = forecastNet.forecastNet(
                engine.Model, 
                device = device,
                callbacks = [f1],
                optimizer=torch.optim.Adam,
                criterion=nn.CrossEntropyLoss,
                **model_config,
                module__device=device,
                module__class_wt=class_wt
            )

        params = {
                    'max_epochs': [15,20,25,30,35,40,50],
                    'batch_size': [32,64,128],
                    'optimizer__lr': expon(scale = 0.01),
                    'optimizer__weight_decay': expon(scale=0.01)
                 }

    X,Y = datasets_all[numT-1].return_all()
    print('X length: ', X.__len__())
    print('Y length: ', Y.__len__())

    rs = RandomizedSearchCV(net, params, refit=True, cv=3, scoring=f1_scorer, verbose=1, n_iter=30)

    print('Randomized Search!')
    rs.fit(X,Y)

    print(rs.best_score_, rs.best_params_)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../configs/config.yaml')
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--numT', type=int, default=1)
    args = parser.parse_args()
    main(args.config,args.debug,args.numT)

