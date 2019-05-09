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
from src import datagen, utils, engine
os.environ['CUDA_VISIBLE_DEVICES']=''

def main(config_file):
    
    # Parser config file
    with open(config_file) as f:
        config = yaml.load(f)

    # Create experiment output directories
    exp_dir = os.path.join(config['output_dir'], config['exp_id'])
    utils.create_dirs([config['output_dir'], exp_dir,
            os.path.join(exp_dir, 'checkpoints'),
            os.path.join(exp_dir, 'results'),
            os.path.join(exp_dir, 'logs')])

    # Copy config file
    copyfile(config_file, os.path.join(exp_dir, config['exp_id']+'.yaml'))
    
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
    print('Basic Data Stats:')
    print('Number of patients = ', len(data) - 2)

    # Datagens
    t = time()
    datagen_train, datagen_val = \
            datagen.get_datagen(data, **config['datagen'])
    print('Datagens Loaded : ', time()-t)

    # Define Classification model
    model = engine.Engine(config['model'])

    # Train the model
    if config['train_model']:
        print('Training the model ...')
        model.train(datagen_train, datagen_val, exp_dir, **config['train'])

    # Test the model
    if config['test_model']:
        print('Testing the model ...')
        #  print('Train data : ')
        #  model.test(datagen_train, exp_dir, 'train')
        print('Val data : ')
        model.test(datagen_val, exp_dir, 'val')
        #  stats = model.test_stats(datagen_val)

    #  with open('stats.pickle', 'wb') as f:
        #  pickle.dump(stats, f)

    return datagen_train, datagen_val

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../configs/config.yaml')
    args = parser.parse_args()
    dgt, dgv = main(args.config)


    '''
    for datagen in dgt:
        for x, y in datagen:
            print(x.keys())
            print('y : ', y)
            print('tau : ', x['tau'])
            print('pid : ', x['pid'])
            print('traj_id : ', x['trajectory_id'])
            print('flag_ad : ', x['flag_ad'])
            print('first_occurance_ad : ', x['first_occurance_ad'])
            break
        break
    '''
