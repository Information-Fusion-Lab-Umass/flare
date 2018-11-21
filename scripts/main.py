import sys
import unittest
from unittest import TestCase as test
sys.path.append('..')
import os
import yaml
import argparse
from shutil import copyfile

from disease_forecast import datagen, utils#, engine

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
    data = datagen.get_data(**config['data'])
    return data

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml')
    args = parser.parse_args()
    data = main(args.config)

   
    #test
    t = test()
    for batch_size in range(1,4):
        traj_type = 1   
        ret = datagen.get_Batch(list(data.values()),batch_size,traj_type,'image')
        t.assertEqual(ret.shape,tuple((batch_size,traj_type+1))) #simple shape test
        
        traj_type = 2
        if(batch_size > 1): #Testing the case where batch size B > number of trajectories
            t.assertRaises(ValueError, datagen.get_Batch,list(data.values()),batch_size,traj_type,'image')
        
    ret = datagen.get_Batch(list(data.values()),3,1,'image')
    
    for row in ret:
        check = row[0].pid
        for item in row:
            t.assertEqual(check,item.pid) #make sure pid is the same throughout each timestep in a trajectory
        

    
