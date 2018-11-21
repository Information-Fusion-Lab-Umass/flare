import sys
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

    # Datagens
    #  datagen_train, datagen_val = datagen.get_datagen(**config['datagen'])

    # Define Classification model
    model = engine.Model(**config['model'])

    # Train the model
    #  model.train(datagen_train, datagen_val, exp_dir, **config['train'])

    # Test the model
    #  model.test(**config['test'])

    return data

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml')
    args = parser.parse_args()
    data = main(args.config)
