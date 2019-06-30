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
from src import datagen, utils, engine, evaluate
import copy
#  os.environ['CUDA_VISIBLE_DEVICES']=''

def main(config_file):
    # Parser config file
    with open(config_file) as f:
        config = yaml.load(f)

    main_exp_dir = os.path.join(config['output_dir'], config['exp_id'])


    utils.create_dirs([config['output_dir'], main_exp_dir, os.path.join(main_exp_dir, 'results'), os.path.join(main_exp_dir, 'logs')])


    max_T = config['datagen']['max_T'] # Load data and get image paths
    num_classes = config['model']['module_task']['num_classes'] 

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
    class_wt = utils.get_classWeights(data, config['data']['train_ids_path'])
    print(class_wt)

    num_iter = config['num_iter']

    model_list = [None]*num_iter
    val_cnfmats = [None]*num_iter
    train_cnfmats = [None]*num_iter 

    for iteration in range(num_iter):
        model_config = copy.deepcopy(config['model'])

        print('Iteration: {}'.format(iteration))
        # Create experiment output directories
        exp_dir = os.path.join(main_exp_dir, config['exp_id'] + '_' + str(iteration))
        utils.create_dirs([config['output_dir'], exp_dir,
                os.path.join(exp_dir, 'checkpoints'),
                os.path.join(exp_dir, 'results'),
                os.path.join(exp_dir, 'logs')])

        # Copy config file
#        copyfile(config_file, os.path.join(exp_dir, config['exp_id']+'_' + str(iteration) + '.yaml'))
        with open(os.path.join(exp_dir, config['exp_id']+'_' + str(iteration) + '.yaml'),'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        # Define Classification model
        model = engine.Engine(class_wt, model_config)
        model_list[iteration] = model

        # Train the model
        if config['train_model']:
            print('Training the model ...')
            model_list[iteration].train(datagen_train, datagen_val, exp_dir, **config['train'])

        # Test the model
        if config['test_model']:
            print('Testing the model ...')
            print('Train data : ')
            train_cnfmats[iteration] = model_list[iteration].test(datagen_train, exp_dir, 'train')
            print('Val data : ')
            val_cnfmats[iteration] = model_list[iteration].test(datagen_val, exp_dir, 'val')
            print('Generating and saving the stats ...')
            stats = model.test_stats(datagen_val)
            stats_dir = os.path.join(main_exp_dir, config['exp_id'] + '_' + str(iteration), 'stats.pickle')

            with open(stats_dir, 'wb') as f:
                pickle.dump(stats, f)

    if(num_iter > 1):
        print('Calculating aggregate results...')
        agg_metrics = utils.calculate_averages(val_cnfmats)
        agg_mat = evaluate.ConfMatrix(max_T-1,num_classes)
        agg_mat.set_vals(agg_metrics)
        agg_mat.save(main_exp_dir,'agg')
        with open(os.path.join(main_exp_dir,'results/agg_metrics_dict.pickle'),'wb') as f:
            pickle.dump(agg_metrics,f)

    return datagen_train, datagen_val

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../configs/config.yaml')
    args = parser.parse_args()
    dgt, dgv = main(args.config)
    
    # with open('../data/datagen_val.pickle','wb') as f:
    #    pickle.dump(dgv,f)
    
#    for i, datagen in enumerate(dgv):
#        num_traj = 0
#        minval = np.inf; maxval = np.NINF
#        for k, (x, y) in enumerate(datagen):
#            num_traj += x['img_features'].size()[0]
#            traj_id = x['trajectory_id'].data.numpy()
#            img_features = x['img_features'].data.numpy()
#            minval = min(minval, np.min(img_features))
#            maxval = max(maxval, np.max(img_features))
#        print('T = {}, traj = {}, min = {}, max = {}'.\
#              format(i, num_traj, minval, maxval))

            #  print(x.keys())
            #  print('y : ', y)
            #  print('tau : ', x['tau'])
            #  print('pid : ', x['pid'])
            #  print('traj_id : ', x['trajectory_id'])
            #  print('flag_ad : ', x['flag_ad'])
            #  print('first_occurance_ad : ', x['first_occurance_ad'])
