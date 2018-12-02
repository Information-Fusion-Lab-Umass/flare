import numpy as np
import pandas as pd
import os
from glob import glob
import torch
import xml.etree.ElementTree as ET 
from itertools import combinations as comb
from itertools import chain
#import ipdb
#from tqdm import tqdm

class Data_Batch: #BxT matrix Data_Batch objects is one whole batch
    def __init__(self, time_step, feat_flag, pid, img_path, cogtests, 
            covariates, metrics, img_features):
        self.time_step = time_step
        self.image_type = feat_flag #'tadpole' or 'cnn3d' 
        self.pid = pid #pid of patient that features are taken from
        self.img_path = img_path #Path of image
        self.cogtests = cogtests #Cognitive tests score
        self.covariates = covariates
        self.metrics = metrics #1x3 output of multitask prediction, for time_step = T
        self.img_features = img_features

class Data:
    def __init__(self, pid, paths, feat):
        self.pid = pid
        self.num_visits = 0
        self.max_visits = 5
        self.which_visits = []

        self.path_imgs = {}
        self.cogtests = {}
        self.metrics = {}
        self.covariates = {}
        self.img_features = {}
        
        self.trajectories = [] #list of [traj_1, traj_2,....]
        
        flag_get_covariates = 0
        
        temp_visits = []
        temp_viscodes = ['none']
        for fmeta, fimg in paths:
            # Extract visit code from meta file         
            viscode = self.get_viscode(fmeta)           
            if viscode not in temp_viscodes:
                temp_viscodes.append(viscode)
                self.num_visits += 1
                #append viscode to list of visits
                temp_visits.append(viscode)
      
                # Store image path with viscode as key
                self.path_imgs[viscode] = fimg

                # Store cognitive test data with viscode as key
                feat_viscode = feat.loc[(feat.PTID==pid) & (feat.VISCODE==viscode)]
                self.cogtests[viscode] = self.get_cogtest(feat_viscode)

                # Store image features
                self.img_features[viscode] = self.get_img_features(feat_viscode)

                # Store evaluation metrics with viscode as key
                self.metrics[viscode] = self.get_metrics(feat_viscode)

                # Store covariate values 
                if flag_get_covariates==0:
                    self.covariates = self.get_covariates(feat_viscode)
                    flag_get_covariates = 1
                
        #Store visit values in sorted, integer form
        self.which_visits = self.get_which_visits(temp_visits) 
        
        #Store trajectory values. Set to get_trajectories_cont() until we find a way to impute missing data
        self.trajectories = self.get_trajectories_cont()

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
    
    def get_trajectories_cont(self):
        """
        JW: Returns continuous trajectories (traj_1,traj_2,...., traj_{max_visits-1}).
        If some traj_i does not exist, the entry for it will be an empty list.
        Written to support an arbitrary amount of trajectories
        """ 
        def check_consec(t):
            """
            checks if the first len(t)-1 tuples are consecutive,
            unless len(t) = 2, then t[0] and t[1] must be consecutive.
            """                
            for i in range(len(t)-2):
                if t[i+1]-t[i] > 1:
                    return False
            return True
        
        trajectories = [None]*(self.max_visits - 1)
        for i in range(self.max_visits-1):
            if(i+1 < self.num_visits):
                cont_trajecs = [] #list of continuous trajectories. EX: for T=3, (1,2,5) is continuous (1,3,5) is NOT. 
                temp = list(comb(self.which_visits,i+2))
                for t in temp:
                    if check_consec(t):
                        cont_trajecs.append(t)            
                trajectories[i] = cont_trajecs
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
        return list(np.nan_to_num([
                float(feat['ADAS11'].values[0]),
                float(feat['CDRSB'].values[0]),
                float(feat['MMSE'].values[0]),
                float(feat['RAVLT_immediate'].values[0])
                ]))
    
    # Extract image features. len = 692. 
    # Several values are missing (' '). Replaced with 0
    # TODO: normalize features
    def get_img_features(self, feat):
        feat_names = feat.columns.values
        img_feat = []
        for name in feat.columns.values:
            if('UCSFFSX' in name or 'UCSFFSL' in name):
                if(name.startswith('ST') and 'STATUS' not in name):
                    if feat[name].values[0] != ' ':
                        img_feat.append(float(feat[name].values[0]))
                    else:
                        img_feat.append(0.0)
        return img_feat

    def get_metrics(self, feat):
        dict_dx = {'NL':0,
                   'MCI to NL':0,
                   'NL to MCI':1,
                   'Dementia to MCI':1,
                   'MCI':1,
                   'MCI to Dementia':2,
                   'Dementia':2,
                   'NL to Dementia':2
                   }
#       dict_dx = {'NL':1,
#                   'MCI to NL':2,
#                   'NL to MCI':3,
#                   'Dementia to MCI':4,
#                   'MCI':5,
#                   'MCI to Dementia':6,
#                   'Dementia':7,
#                   'NL to Dementia':8}
        dx = feat['DX'].values[0]
        if dx!=dx:
            dx = 'NL'
        return [dict_dx[dx],
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
            for idx_imgdir in range(1, 11):        
                f_img = glob(os.path.dirname(f).\
                        replace(path_meta, path_images+\
                        '/'+str(idx_imgdir)+'/ADNI')+'/*.nii')
                if len(f_img)==1:
                    break
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

def get_datagen(data, data_split, batch_size, num_visits, feat_flag):
    # Get Train and Test PIDs
    data_items = list(data.values())
    data_train = data_items[:int(data_split*len(data_items))]
    data_val = data_items[len(data_train):]
    print('Train = {}, Val = {}'.format(len(data_train), len(data_val)))

    # Get data Generator
    n_t = np.random.randint(1, 5) if num_visits==-1 else num_visits
    datagen_train = get_Batch(data_train, batch_size, n_t, feat_flag)
    datagen_val = get_Batch(data_val, batch_size, n_t, feat_flag)

    return datagen_train, datagen_val

def get_Batch(patients,B,n_t,feat_flag):
    """
    JW:
    Arguments:
        'patients': List of 'Data objects, one for each patient. Size P x 1.
        'B': Integer value which represents the batch size
        'n_t': Integer between 1 and the number of trajectory types traj_{n_t}. 
               used to select which trajectory type we want to sample from
        'feat_flag': String that is set to 'tadpole' or 'image' 
                    depending what kind of image features we want to train with.
    Returns:
        'ret': a BxT matrix of Data_Batch objects 
    """
    def one_batch_one_patient(p,sample):
            """
            JW:
            arguments:
                'p':  Data object corresponding to a patient.
                'sample': List of integers. It is the trajectory we want to 
                          create the batch entry for.
            Description: Produces batch entry for given patient for each 
                        timestep in a sample.
            returns:
                'ret': a generator of Data_Batch objects which has T entries.
            """
            batch = []
            for time_step in sample:
                key = dict_int2visit[time_step]
                batch.append(Data_Batch(time_step,feat_flag,
                                p.pid,
                                p.path_imgs[key],
                                p.cogtests[key],
                                p.covariates,
                                p.metrics[key],
                                p.img_features[key]))
            return batch

    while 1:
        T = n_t+1 #number of visits in traj_{n_t}. 
        
        ret = np.empty((B,T),dtype=object)
        dict_int2visit = {0:'bl', #reverse dictionary
                      1:'m06',
                      2:'m12',
                      3:'m18',
                      4:'m24',
                      5:'m36',
                    -1:'none'}
        
        selections = []
        patient_idx = []
        
        for idx,p in enumerate(patients):
            item = p.trajectories[n_t-1]
            #Check if trajectory exists. If it doesn't, don't concat it.
            if item is not None: 
                traj_len = len(item)
                selections.append(item)
                patient_idx.append([idx]*traj_len)
            
        selections = list(chain.from_iterable(selections))
        #print(selections)
        patient_idx = list(chain.from_iterable(patient_idx))
        #print(patient_idx)
        num_trajs = len(selections)
        #  print(n_t, num_trajs)
        if(B > num_trajs): 
            raise ValueError("Batch size: '{}' is larger than number \
                    of trajectories: '{}'".format(B,num_trajs))
          
        samples_idx = np.random.choice(len(selections),B,replace=False)
        #list of B trajectories chosen for batch.
        samples = [selections[i] for i in samples_idx] 
        #list of B patient Data objects corresponding to ones chosen for Batch
        samples_p = [patients[patient_idx[i]] for i in samples_idx] 
        #print(samples)
        #print([patient_idx[i] for i in samples_idx])
        
        for idx in range(B):
            temp = one_batch_one_patient(samples_p[idx],samples[idx])
            ret[idx,:] = temp
        yield ret

# Given a (BxT) array of DataBatch objects, return the image paths/features
# as a (BxT-1) paths (CNN) or (BxT-1xF) features (tadpole)
def get_img_batch(x, as_tensor=False):
    (B, T) = x.shape
    img_type = x[0, 0].image_type
    if img_type =='tadpole':
        num_feat = len(x[0, 0].img_features)
        feat = np.zeros((B, T-1, num_feat))
        for b in range(B):
            for t in range(T-1):
                feat[b, t, :] = x[b, t].img_features
    elif img_type == 'cnn3d':
        feat = np.zeros((B, T-1))
        for b in range(B):
            for t in range(T-1):                
                feat[b, t] = x[b, t].img_path
    if as_tensor==True:
        feat = torch.from_numpy(feat).float()
    return feat

# Given a (BxT) array of DataBatch objects, return the time indices
# as (BxT) matrix
def get_time_batch(x, as_tensor=False):
    (B, T) = x.shape
    timeidx = np.zeros((B, T))
    for b in range(B):
        for t in range(T):
            timeidx[b, t] = x[b, t].time_step
    if as_tensor==True:
        timeidx = torch.from_numpy(timeidx).float()
    return timeidx

# Given a (BxT) array of DataBatch objects, return the cognitive tests
# as (BxT-1xF) features
def get_long_batch(x, as_tensor=False):
    (B, T) = x.shape
    num_feat = len(x[0, 0].cogtests)
    feat = np.zeros((B, T-1, num_feat))
    for b in range(B):
        for t in range(T-1):
            feat[b, t, :] = x[b, t].cogtests
    if as_tensor==True:
        feat = torch.from_numpy(feat).float()           
    return feat

# Given a (BxT) array of DataBatch objects, return the covariates
# as (BxT-1xF) features
def get_cov_batch(x, as_tensor=False):
    (B, T) = x.shape
    num_feat = len(x[0, 0].covariates)
    feat = np.zeros((B, T-1, num_feat))
    for b in range(B):
        for t in range(T-1):
            feat[b, t, :] = x[b, t].covariates
    if as_tensor==True:
        feat = torch.from_numpy(feat).float()           
    return feat

# Given a (BxT) array of DataBatch objects and the task name, 
# return the labels as (BxF) tensor
def get_labels(x, task='dx', as_tensor=False):
    (B, T) = x.shape
    if task=='dx':
        labels = np.array([x[b, -1].metrics[0] for b in range(B)])
        if as_tensor==True:
            labels = torch.from_numpy(labels).long() 
    return labels









