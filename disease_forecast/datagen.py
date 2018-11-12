import numpy as np
import pandas as pd
import os
from glob import glob
import xml.etree.ElementTree as ET 

class Data:
    def __init__(self, pid, paths, feat):
        self.pid = pid
        self.num_visits = len(paths)

        self.path_imgs = {}
        self.cogtests = {}
        self.metrics = {}
        self.covariates = {}
        
        flag_get_covariates = 0
        for fmeta, fimg in paths:
            # Extract visit code from meta file         
            viscode = self.get_viscode(fmeta)

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

def get_data(path_meta, path_images, path_feat): 

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
        if len(p_paths)!=0:
            data[pid] = Data(pid, p_paths, data_feat[data_feat.PTID==pid])
    return data 



