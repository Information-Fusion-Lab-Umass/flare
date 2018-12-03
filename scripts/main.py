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
from disease_forecast import datagen, utils, engine

def main(config_file):
    
    # Parser config file
    with open(config_file) as f:
        config = yaml.load(f)

    # Create experiment output directories
    exp_dir = os.path.join(config['output_dir'], config['exp_id'])
    utils.create_dirs([exp_dir,
            os.path.join(exp_dir, 'checkpoints'),
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

    # Datagens
    t = time()
    (datagen_train, data_train), (datagen_val, data_val) =\
            datagen.get_datagen(data, **config['datagen'])
    print('Datagens Loaded : ', time()-t)

    # Define Classification model
    model = engine.Model(**config['model'])

    # Train the model
    #  model.train(datagen_train, datagen_val, exp_dir, **config['train'])

    # Test the model
    print('Train data : ')
    model.test(data_train, exp_dir, **config['test'])
    #  print('Val data : ')
    #  model.test(data_val, **config['test'])

    return data, datagen_train, datagen_val

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml')
    args = parser.parse_args()
    main(args.config)
    #  data, dgt, dgv = main(args.config)
    #  print(len(data))
    #  for key in data:
    #      for vis in data[key].cogtests:
    #          if np.isnan(data[key].cogtests[vis]).any():
    #              print(key, vis, data[key].cogtests[vis])
    #  ipdb.set_trace()
    #
    #  #tests
    #  t = test()
    #  for batch_size in range(1,4):
    #      traj_type = 1
    #      ret = datagen.get_Batch(list(data.values()),batch_size,traj_type,'image')
    #      t.assertEqual(ret.shape,tuple((batch_size,traj_type+1))) #simple shape test
    #
    #      traj_type = 2
    #      if(batch_size > 1): #Testing the case where batch size B > number of trajectories
    #          t.assertRaises(ValueError, datagen.get_Batch,list(data.values()),batch_size,traj_type,'image')
    #
    #  ret = datagen.get_Batch(list(data.values()),3,1,'image')
    #  for row in ret:
    #      check = row[0].pid
    #      for item in row:
    #          t.assertEqual(check,item.pid) #make sure pid is the same throughout each timestep in a trajectory
    #
    #
    
