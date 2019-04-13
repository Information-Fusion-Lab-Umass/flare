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
from disease_forecast import datagen, utils, engine
import ipdb

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
        data = datagen.get_data(**config['data'])
        with open(path_load, 'wb') as f:
            pickle.dump(data, f)
    print('Data Loaded : ', time()-t)

    # Datagens
    t = time()
    (datagen_train, data_train), (datagen_val, data_val) =\
            datagen.get_datagen(data, **config['datagen'])
    print('Datagens Loaded : ', time()-t)

    t = time()
    x_train_batch = next(datagen_train)
    (B, T) = x_train_batch.shape
    
    x_img_data = datagen.get_img_batch(x_train_batch, as_tensor=True) 
    print('Image batch Loaded: ', time()-t)
    
    ipdb.set_trace()
    
    #return data

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml')
    args = parser.parse_args()
    main(args.config)

