import sys
import unittest
from unittest import TestCase as test
sys.path.append('..')
import os
import yaml
import argparse
from shutil import copyfile
import ipdb
from time import time
import pickle
import numpy as np
from src import datagen_tadpole as datagen, utils, engine_cae as engine

def main(config_file):
    
    # Parser config file
    with open(config_file) as f:
        config = yaml.load(f)

    # Load data and get image paths
    t = time()
    path_load = config['data'].pop('path_load')
    if os.path.exists(path_load):
        with open(path_load, 'rb') as f:
            data = pickle.load(f)
    else:
        data = datagen.get_data_tadpole(**config['data'])
        with open(path_load, 'wb') as f:
            pickle.dump(data, f)
    print('Data Loaded : ', time()-t)
    #  ipdb.set_trace()

    # Datagens
    t = time()
    (datagen_train, data_train), (datagen_val, data_val) =\
            datagen.get_datagen(data, **config['datagen'])
    print('Datagens Loaded : ', time()-t)
    
    data_t = datagen.get_timeBatch(data_val,0,'tadpole')
    y_dx = datagen.get_labels(data_t, task='dx')
    
    print("number of labels of class no AD",len(np.where(y_dx==0)[0]))

    print("number of labels of class MCI",len(np.where(y_dx==1)[0]))

    print("number of labels of class AD",len(np.where(y_dx==2)[0]))
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config_grab.yaml')
    args = parser.parse_args()
    main(args.config)

