import numpy as np
import pandas as pd
import os
from glob import glob
import xml.etree.ElementTree as ET 
from itertools import combinations as comb
from itertools import chain

class Data_Batch: #BxT Data_Batch objects equals one batch
    def __init__(self,time_step,img_path,cogtests,covariates):
        self.time_step = time_step
        self.img_path = img_path #Path of image
        self.cogtests = cogtests #Cognitive tests score
        self.covariates = covariates
        self.metrics = [] #1x3 output of multitask prediction, for time_step = T

class Data:
    def __init__(self, pid, paths, feat):
        self.pid = pid
        self.num_visits = len(paths)
        self.max_visits = 5
        self.which_visits = []

        self.path_imgs = {}
        self.cogtests = {}
        self.metrics = {}
        self.covariates = {}
        
        self.trajectories = [] #list of [traj_1, traj_2,....]
        
        flag_get_covariates = 0
        
        temp_visits = []
        for fmeta, fimg in paths:
            # Extract visit code from meta file         
            viscode = self.get_viscode(fmeta)
            
            #append viscode to list of visits
            temp_visits.append(viscode)
 
            # Store image path with viscode as key
            self.path_imgs[viscode] = fimg

            # Store cognitive test data with viscode as key
            feat_viscode = feat.loc[(feat.PTID==pid) & (feat.VISCODE==viscode)]
            self.cogtests[viscode] = self.get_cogtest(feat_viscode)

            # Store evaluation metrics with viscode as key
            self.metrics[viscode] = self.get_metrics(feat_viscode)

            # Store covariate values 
            if flag_get_covariates==0:
                self.covariates = self.get_covariates(feat_viscode)
                flag_get_covariates = 1
                
        #Store visit values in sorted, integer form
        self.which_visits = self.get_which_visits(temp_visits) 
        
        #Store trajectory values
        self.trajectories = self.get_trajectories()
                    
    
    def get_trajectories(self):
        """
        JW: Returns (traj_1,traj_2,...., traj_{max_visits-1}).
        If some traj_i does not exist, the entry for it will be an empty list.
        Written to support an arbitrary amount of trajectories
        """
        trajectories = [None]*(self.max_visits - 1)
        for i in range(self.max_visits-1):
            if(i+1 < self.num_visits):
                trajectories[i] = list(comb(self.which_visits,i+2))
        return tuple(trajectories)
        
    def get_which_visits(self, visits):
        """
        JW: Returns list of visits in integer form where bl -> 0, m06 ->1, ...
        """
        which_visits = []
        dict_visit2int = {'bl':0,
              'm06':1,
              'm12':2,
              'm18':3,
              'm24':4,
              'm36':5,
              'none':-1}
        for key in visits:
            which_visits.append(dict_visit2int[key])
            
        which_visits = [x for x in which_visits if x != -1]
        return sorted(which_visits)         
    
    def get_cogtest(self, feat):
        return [
                float(feat['ADAS11'].values[0]),
                float(feat['CDRSB'].values[0]),
                float(feat['MMSE'].values[0]),
                float(feat['RAVLT_immediate'].values[0])
                ]

    def get_metrics(self, feat):
        #  dict_dx =
        return [
                feat['DX'].values[0],
                float(feat['ADAS13'].values[0]),
                float(feat['Ventricles'].values[0])
                ]

    def get_covariates(self, feat):
        dict_gender = {'male':0, 'female':1}
        return [
                float(feat['AGE'].values[0]),
                dict_gender[feat['PTGENDER'].values[0].lower()],
                #  feat['PTEDUCAT'].values[0],
                #  feat['PTETHCAT'].values[0],
                #  feat['PTRACCAT'].values[0],
                #  feat['PTMARRY'].values[0],
                int(feat['APOE4'].values[0])
                ]

    def get_viscode(self, fmeta):
        dict_visit = {'ADNI Screening':'bl',
              'ADNI1/GO Month 6':'m06',
              'ADNI1/GO Month 12': 'm12',
              'ADNI1/GO Month 18' : 'm18',
              'ADNI1/GO Month 24': 'm24',
              'ADNI1/GO Month 36': 'm36',
             'No Visit Defined': 'none'}
        tree = ET.parse(fmeta)
        root = tree.getroot()
        vals = [x for x in root.iter('visit')]
        key = [x for x in vals[0].iter('visitIdentifier')][0].text
        return dict_visit[key]

def get_data(path_meta, path_images, path_feat, min_visits=1): 

    data_feat = pd.read_csv(path_feat, dtype=object)
    
    id_list = next(os.walk(path_meta))[1]

    data = {}
    # Iterate over patient folders
    for pid in id_list:
        # List of identifier files of patient
        pmeta = glob(os.path.join(path_meta, \
                pid+'/**/*.xml'), recursive=True)
        p_paths = []
        # Iterate over identifier files
        for f in pmeta:
            # Get image path
            f_img = glob(os.path.dirname(f).\
                    replace(path_meta, path_images)+'/*.nii')
            if len(f_img)==1:
                f_img = f_img[0]
                # Get metadata path
                f_basename = os.path.basename(f_img)[:-3]
                f_meta = f_basename.split('_')
                idx_mr, idx_br = f_meta.index('MR'), f_meta.index('Br')
                f_meta = f_meta[:idx_mr] + \
                        f_meta[idx_mr+1:idx_br] + f_meta[idx_br+2:]
                f_meta = os.path.join(path_meta, '_'.join(f_meta)+'xml')
                p_paths.append((f_meta, f_img))
        # Get all data for the pid
        if len(p_paths)>=min_visits:
            data[pid] = Data(pid, p_paths, data_feat[data_feat.PTID==pid])

    return data 


def get_Batch(patients,B,n_t):
    """
    JW:
    Arguments:
        'patients': is a list of 'Data objects, one for each patient. Size P x 1.
        'B': an integer value which represents the batch size
        'n_t': some integer between 1 and the number of trajectories. 
               used to select which trajectory we want to sample from
    
    Returns:
        'ret': a BxTxP matrix of Data_Batch objects 
    """
    P = len(patients) #number of patients
    T = n_t+1 #number of visits in total trajectory. 
    
    ret = np.empty((B,T,P),dtype=object)
    dict_int2visit = {0:'bl', #reverse dictionary
               1:'m06',
               2:'m12',
               3:'m18',
               4:'m24',
               5:'m36',
              -1:'none'}
    
    selections = []

    for p in patients:
        selections = list(chain(selections,p.trajectories[n_t-1])) #concatenate trajectories
   
    samples_idx = np.random.choice(len(selections),B,replace=False)
    samples = [selections[i] for i in samples_idx]

        
    def one_batch_one_patient(p,sample):
        """
        JW
        Description: subtask. produces Batch for one patient for each timestep in a sample.
       
        returns an iterable of Data_Batch objects, which has T entries when cast to a list
        """
        for time_step in sample:
            key = dict_int2visit[time_step]
            if key not in p.which_visits: #check if it's missing
                yield Data_Batch(time_step,'M','M','M')
            else:
                yield Data_Batch(time_step,
                             p.path_imgs[key],
                             p.cogtests[key],
                             p.covariates)
                
                
    for outer, p in enumerate(patients):
       mat = np.empty((B,T),dtype=object) #BxT matrix of Data_Batches
       for inner,sample in enumerate(samples):
            temp = []
            temp = list(chain(temp,one_batch_one_patient(p,sample)))
            mat[inner,:] = temp
       ret[:,:,outer] = mat 
    return ret

