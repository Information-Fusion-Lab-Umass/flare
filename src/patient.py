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
        self.flag_ad = False
        self.first_occurance_ad = -1

        self.visits_id = []
        self.trajectories_id = {}
        self.visits = {}
        self.trajectories = {}

        # Extract data of the visits
        for idx, row in df.iterrows():
            visit_id = int(row['VISNUM'])
            self.visits_id.append(visit_id)
            self.visits[visit_id] = Visit(row)

        # Check if the patient develops AD
        for visit_id in self.visits_id:
            if self.visits[visit_id].data['labels'][0] == 2:
                self.flag_ad = True
                self.first_occurance_ad = visit_id
                break

        self.num_visits = len(self.visits_id)
        self.visits_id = sorted(self.visits_id)
        patient_info = {
                'pid' : self.patient_id,
                'flag_ad' : self.flag_ad,
                'first_occurance_ad' : self.first_occurance_ad
                }

        # Obtain trajectory data 
        for i in range(2, self.num_visits + 1):
            self.trajectories_id[i] = list(comb(self.visits_id, i))
            if only_consecutive: 
                self.trajectories_id[i] = utils.return_consec(\
                        self.trajectories_id[i])
            self.trajectories[i] = [
                    Trajectory(
                        [self.visits[tt] for tt in t], 
                        t, patient_info
                        ) 
                    for t in self.trajectories_id[i]
                    ]
        
class Trajectory:
    '''
    A Trajectory is a series of visits

    Input:
        init_vis (list): A list of tuples. Each tuple is a visit entry.
    '''
    def __init__(self, visits, trajectory_id, patient_info):
        self.visits = {}
        for visit in visits:
            self.visits[visit.visit_id] = visit
        self.length = len(self.visits)

        # calculate tau and T values for the trajectory
        self.T = self.length - 1
        keys = sorted(list(self.visits.keys()))
        self.tau = keys[-1] - keys[-2]

        # Additional information
        self.pid = patient_info['pid']
        self.flag_ad = patient_info['flag_ad'] 
        self.first_occurance_ad = patient_info['first_occurance_ad']
        self.trajectory_id = trajectory_id

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
        self.visit_code = df['VISCODE']
        self.visit_id = int(df['VISNUM'])

        self.data = {}
        self.data['covariates'] = self.get_covariates(df)
        self.data['labels'] = self.get_labels(df)
        self.data['test_scores'] = self.get_cogtest(df)
        self.data['img_features'] = self.get_img_features(df)
        #  self.data['image_path'] = df['misc']

    def get_cogtest(self, df):
        return list([
                float(df['ADAS11']),
                float(df['CDRSB']),
                float(df['MMSE']),
                float(df['RAVLT_immediate'])
                ])
    
    # Extract image features. len = 692. 
    def get_img_features(self, df):
        img_df = []
        for name in list(df.index):
            if('UCSFFSX' in name or 'UCSFFSL' in name):
                if(name.startswith('ST') and 'STATUS' not in name):
                    if df[name] != ' ':
                        img_df.append(float(df[name]))
                    else:
                        img_df.append(0.0)
        return img_df

    def get_labels(self, df):
        dict_dx = {'NL':0,
                   'MCI to NL':0,
                   'NL to MCI':1,
                   'Dementia to MCI':1,
                   'MCI':1,
                   'MCI to Dementia':2,
                   'Dementia':2,
                   'NL to Dementia':2
                   }
        dx = df['DX']
        if dx != dx:
            dx = 'NL'
        return [dict_dx[dx]]
        #  return [dict_dx[dx],
        #          float(feat['ADAS13'].values[0]),
        #          float(feat['Ventricles'].values[0])
                #  ]

    def get_covariates(self, df):
        dict_gender = {'male':0, 'female':1}
        return [
                float(df['AGE']),
                dict_gender[df['PTGENDER'].lower()],
                #  df['PTEDUCAT'].values[0],
                #  df['PTETHCAT'].values[0],
                #  df['PTRACCAT'].values[0],
                #  df['PTMARRY'].values[0],
                float(df['APOE4'])
                ]


