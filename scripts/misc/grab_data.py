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
from src import datagen_tadpole as datagen, utils, engine

def main(config_file):
    
    # Parser config file
    with open(config_file) as f:
        config = yaml.load(f)

    # Load data and get image paths
    t = time()
    path_load = config['data'].pop('path_load')
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
    print('the jawn was grabbed')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config_grab.yaml')
    args = parser.parse_args()
    main(args.config)

