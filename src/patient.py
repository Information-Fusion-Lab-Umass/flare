import pickle
import pandas as pd
import numpy as np
from itertools import combinations as comb

from src import utils

class Patient:
    """
    Stores patient information from all available visits
    pid : patient ID
    visits_id : IDs of available visits of the patient 
    trajectories_id :
    visits : Stores data of each visits
    trajectories : Stores data of each trajectory
    
    Parameters
    ----------
    data 
    pid
    only_consecutive

    Returns
    -------
    None
    """
    def __init__(self, pid, df, only_consecutive = True):
        self.patient_id = pid
        self.visits_id = []
        self.trajectories_id = {}
        self.visits = {}
        self.trajectories = {}

        # Extract data of the visits
        for idx, row in df.iterrows():
            visit_id = row['visit_number']
            self.visits_id.append(visit_id)
            self.visits[visit_id] = Visit(row)          
        self.num_visits = len(self.visits_id)

        # Obtain trajectory data 
        for i in range(2, num_visits + 1):
            self.trajectories_id[i] = list(comb(self.visits_id, i))
            if only_consecutive: 
                self.trajectories_id[i] = utils.return_consec(\
                        self.trajectories_id[i])
            self.trajectories[i] = [
                    Trajectory([self.visits[tt] for tt in t]) 
                    for t in self.trajectories_id[i]
                    ]
    
class Trajectory:
    '''
    A Trajectory is a series of visits

    Input:
        init_vis (list): A list of tuples. Each tuple is a visit entry.
    '''
    def __init__(self, visits):
        self.visits = {}
        for visit in visits:
            self.visits[visit.visit_number] = visit
        self.length = len(self.visits)

        # calculate tau and T values for the trajectory
        self.T = self.length - 1
        keys = sorted(list(self.visits.keys()))
        self.tau = keys[-1] - keys[-2]

class Visit:
    '''
    Each visit has query functions that allow you to access data from a
    visit that has already been partitioned into different categories fo
    easy model access i.e image features, structured covariates, 
    test scores, and of course the label.

    Input:
    df (DataFrame): Row of main dataframe corresponding to one patient's
    visit.
    '''
    def __init__(self, df):
        self.visit_code = df['visit_code']
        self.visit_number = df['visit_number']

        self.data = {}
        self.data['covariates'] = self.get_covariates(df)
        self.data['labels'] = self.get_labels(df)
        self.data['test_scores'] = self.get_cogtest(df)
        self.data['img_features'] = self.get_img_features(df)
        #  self.data['image_path'] = df['misc']

    def get_cogtest(self, df):
        return list([
                float(df['ADAS11'].values[0]),
                float(df['CDRSB'].values[0]),
                float(df['MMSE'].values[0]),
                float(df['RAVLT_immediate'].values[0])
                ])
    
    # Extract image features. len = 692. 
    def get_img_features(self, df):
        df_names = df.columns.values
        img_df = []
        for name in df.columns.values:
            if('UCSFFSX' in name or 'UCSFFSL' in name):
                if(name.startswith('ST') and 'STATUS' not in name):
                    if df[name].values[0] != ' ':
                        img_df.append(float(df[name].values[0]))
                    else:
                        img_df.append(0.0)
        return img_df

    def get_labels(self, df):
        dict_dx = {'NL':0,
                   'MCI to NL':0,
                   'NL to MCI':0,
                   'Dementia to MCI':0,
                   'MCI':0,
                   'MCI to Dementia':1,
                   'Dementia':1,
                   'NL to Dementia':1
                   }
        dx = df['DX'].values[0]
        if dx!=dx:
            dx = 'NL'
        return [dict_dx[dx], 0, 0]
        #  return [dict_dx[dx],
        #          float(feat['ADAS13'].values[0]),
        #          float(feat['Ventricles'].values[0])
                #  ]

    def get_covariates(self, df):
        dict_gender = {'male':0, 'female':1}
        return [
                float(df['AGE'].values[0]),
                dict_gender[df['PTGENDER'].values[0].lower()],
                #  df['PTEDUCAT'].values[0],
                #  df['PTETHCAT'].values[0],
                #  df['PTRACCAT'].values[0],
                #  df['PTMARRY'].values[0],
                int(df['APOE4'].values[0])
                ]


