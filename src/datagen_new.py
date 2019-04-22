import numpy as np
import pickle
import itertools
import pandas as pd
import torch
from torch.utils import data

from src import patient

def get_data(path, min_visits = 1, only_consecutive = True):
    data_feat = pd.read_csv(path, dtype = object)
    id_list = list(set(data_feat.PTID.values))
    data = {}
    for pid in id_list:
        data_pid = data_feat[data_feat.PTID==pid]
        data_pid = patient.Patient(pid, data_pid, only_consecutive)
        if data_pid.num_visits >= min_visits:
            data[pid] = data_pid
    return data

def get_datagen(src_data, data_split, batch_size, max_visits):
    # Split data into train and validation sets
    N = len(src_data)
    num_train = int(data_split * N)
    data_train = dict(list(src_data.items())[:num_train])
    data_val = dict(list(src_data.items())[num_train:])

    # Get train datagenerators
    datagen_train = []
    for T in range(2, max_visits + 1):
        dataset = Dataset(data_train, T)
        dataloader = data.DataLoader(dataset, batch_size, shuffle = True)
        datagen_train.append(dataloader)

    # Get validation datagenerators
    datagen_val = []
    for T in range(2, max_visits + 1):
        dataset = Dataset(data_val, T)
        dataloader = data.DataLoader(dataset, batch_size, shuffle = True)
        datagen_train.append(dataloader)

    return datagen_train, datagen_val

class Dataset(data.Dataset):
    def __init__(self, data, T):
        self.T = T
        self.data = data
        self.trajectories = [self.data[pid].trajectories[T] \
                for pid in self.data]     

    def __len__(self):
        """ 
        Returns the number of unique patient ids in the directory
        """
        return len(self.trajectories)

    def __getitem__(self, index):
        x = {}
        x['tau'] = self.trajectories[index].tau
        x['img'] = self.get_data(self.trajectories[index], 'img_features')
        x['cov'] = self.get_data(self.trajectories[index], 'covariates')
        x['long'] = self.get_data(self.trajectories[index], 'test_scores')
        y = self.get_data(self.trajectories[index], 'labels')
        return x, y

    def get_data(self, trajectory, key):
        x = [visit.data[key] for visit in trajectory.visits]
        x = np.vstack(x)
        return torch.from_numpy(x).float()



