import numpy as np
import pickle
import itertools
import pandas as pd
import torch
# import skorch
from torch.utils import data
from tqdm import tqdm
import sys
import os
import random
import json

from src import patient


def get_split(tadpole_path, splits):
    train_split, val_split, test_split = splits

    assert train_split + val_split + test_split == 1

    tad_df = pd.read_csv(tadpole_path)
    ptids = list(tad_df['PTID'].unique())

    random.shuffle(ptids)

    total_sum = len(ptids)

    train_ids = ptids[: int(total_sum * train_split)]
    val_ids = ptids[int(total_sum * train_split): int(total_sum * (train_split + val_split))]
    test_ids = ptids[int(total_sum * (train_split + val_split)):]

    np.savetxt('../data/patientID_train.txt', np.array(train_ids), delimiter='\n', fmt="%s")
    np.savetxt('../data/patientID_val.txt', np.array(val_ids), delimiter='\n', fmt="%s")
    np.savetxt('../data/patientID_test.txt', np.array(test_ids), delimiter='\n', fmt="%s")

    print("num train: {}".format(len(train_ids)))
    print("num val: {}".format(len(val_ids)))
    print("num test: {}".format(len(test_ids)))


def get_split_from_liu():
    root_path = "/mnt/nfs/work1/mfiterau/sidongzhang/liunet-autoencoder/splits"
    df = pd.read_csv("/mnt/nfs/work1/mfiterau/ADNI_IMG/mriimg_meta_v4.csv")

    train_ids = []
    for i in range(8):
        split_fname = os.path.join(root_path, f'split{i}.json')
        with open(split_fname, 'r', encoding='utf-8') as split_file:
            split_obj = json.load(split_file)
            assert split_obj['number'] == i
            split = split_obj['indices']
            for idx in split:
                row = df.iloc[idx]
                ptid = row["PTID"].replace('sub-ADNI', '')
                train_ids.append("{}_S_{}".format(ptid[:3], ptid[4:]))

    test_ids = []
    for i in range(8, 10):
        split_fname = os.path.join(root_path, f'split{i}.json')
        with open(split_fname, 'r', encoding='utf-8') as split_file:
            split_obj = json.load(split_file)
            assert split_obj['number'] == i
            split = split_obj['indices']
            for idx in split:
                row = df.iloc[idx]
                ptid = row["PTID"].replace('sub-ADNI', '')
                test_ids.append("{}_S_{}".format(ptid[:3], ptid[4:]))

    np.savetxt('../data/patientID_train_liu.txt', np.array(train_ids), delimiter='\n', fmt="%s")
    np.savetxt('../data/patientID_test_liu.txt', np.array(test_ids), delimiter='\n', fmt="%s")

    print("num train: {}".format(len(train_ids)))
    print("num test: {}".format(len(test_ids)))


# def get_tadpole(tadpole_path, train_ids_path, val_ids_path, test_ids_path,
#                 min_visits=1, only_consecutive=True):
#     tad_df = pd.read_csv(tadpole_path)
#     ptids = list(tad_df['PTID'].unique())
#
#     data = {}
#     for pid in tqdm(ptids):
#         query = tad_df.loc[tad_df["PTID"] == pid]
#         data_pid = patient.Patient(pid, query, only_consecutive)
#         if data_pid.num_visits >= min_visits:
#             data[pid] = data_pid
#         sys.stdout.flush()
#
#     data['train_ids'] = np.loadtxt(train_ids_path, dtype=str)
#     data['val_ids'] = np.loadtxt(val_ids_path, dtype=str)
#     data['test_ids'] = np.loadtxt(test_ids_path, dtype=str)
#     return data

def get_tadpole(tadpole_path, train_ids_path, test_ids_path,
                min_visits=1, only_consecutive=True):
    tad_df = pd.read_csv(tadpole_path)
    ptids = list(tad_df['PTID'].unique())

    data = {}
    for pid in tqdm(ptids):
        query = tad_df.loc[tad_df["PTID"] == pid]
        data_pid = patient.Patient(pid, query, only_consecutive)
        if data_pid.num_visits >= min_visits:
            data[pid] = data_pid
        sys.stdout.flush()

    data['train_ids'] = np.loadtxt(train_ids_path, dtype=str)
    data['test_ids'] = np.loadtxt(test_ids_path, dtype=str)
    return data


if __name__ == '__main__':
    # get_split("../data/TADPOLE_D1_D2_m48.csv", (0.8, 0.1, 0.1))
    get_split_from_liu()
    data_dict = get_tadpole("../data/TADPOLE_D1_D2_proc_norm_test.csv",
                            "../data/patientID_train_liu.txt", "../data/patientID_test_liu.txt")
    pickle.dump(data_dict, open("../data/tadpole_data_liu.pkl", "wb"))
