import numpy as np
import pickle
import itertools
import pandas as pd
import torch
from torch.utils import data

def get_data(path, min_visits = 1, only_consecutive = True):
    data_feat = pd.read_csv(path, dtype = object)
    id_list = list(set(data_feat.PTID.values))
    data = {}
    for pid in tqdm(id_list):
        data_pid = data_feat[data_feat.PTID==pid]
        data_pid = Patient(pid, data_pid, only_consecutive)
        if data_pid.num_visits >= min_visits:
            data[pid] = data_pid
    return data

class DataIterator(data.Dataset):
    def __init__(self, path, T):
        self.T = T
        with open(path, 'rb') as f:
            self.data = pickle.load(f)
        self.trajectories = [self.data[pid].trajectories[T] \
                for pid in self.data]     

    def __len__(self):
        """ 
        Returns the number of unique patient ids in the directory
        """
        return len(self.trajectories)

    def __getitem__(self, index):
        x = {}
        x['time'] = self.get_time_data(self.trajectories[index])
        x['img'] = self.get_img_data(self.trajectories[index])
        x['cov'] = self.get_cov_data(self.trajectories[index])
        x['long'] = self.get_long_data(self.trajectories[index])
        y =
        return x, y

    def get_img_data(self, trajectory):
        x = [visit.data['img_features'] for visit in trajectory.visits]
        x = np.vstack(x)
        return torch.from_numpy(x).float()



       




