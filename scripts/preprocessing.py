import pandas as pd
import numpy as np
from tqdm import tqdm
#import ipdb

def preprocess_adni(input_path, output_path, pid_train):
    data = pd.read_csv(input_path, dtype = object)
    train_ids = np.loadtxt(pid_train, dtype = str)
    
    # Add VISNUM column
    visit_id = {
            'bl': 0,
            'm06': 1,
            'm12': 2,
            'm18': 3,
            'm24': 4,
            'm36': 5,
            'm48': 6
            }
#    visit_codes = ['bl', 'm06', 'm12', 'm18', 'm24', 'm30', 'm36', \
#            'm42', 'm48', 'm54', 'm60', 'm66', 'm72', 'm78', 'm84', 'm90', \
#            'm96', 'm102', 'm108', 'm114', 'm120']
#    visit_id = {key: i for i, key in enumerate(visit_codes)}
    data['VISNUM'] = data['VISCODE'].apply(lambda x: visit_id[x] \
            if x in visit_id else -1)

    # Retain only rows with required visit_id
    data = data.loc[data['VISNUM'] != -1]

    # Impute missing image feature data  
    data.sort_values(by = ['PTID', 'VISNUM'], inplace = True)
    data = data.groupby('PTID').ffill()
    all_nan_cols = []
    for name in tqdm(data.columns.values):
        if('UCSFFSX' in name or 'UCSFFSL' in name):
            if(name.startswith('ST') and 'STATUS' not in name):
                data[name] = data[name].apply(pd.to_numeric, errors = 'coerce')
                data[name].fillna(data[name].mean(), inplace=True)                
                if np.sum(np.isnan(data[name].values)) > 0: 
                    all_nan_cols.append(name)
    data = data.drop(all_nan_cols, axis = 1)    
    
    # Fill Nan values of features with mean
    cols = ['ADAS13', 'MMSE', 'ADAS11', 'RAVLT_immediate', \
            'RAVLT_forgetting', 'AGE', 'CDRSB']
    for col in cols:
        data[col] = data[col].apply(pd.to_numeric, errors = 'coerce')
        data[col].fillna(data[col].mean(), inplace=True)

        featcol = data[data['PTID'].isin(train_ids)][col].values
        mean, std = np.mean(featcol), np.std(featcol)
        data[col] = (data[col] - mean)/(std + 1e-4)
        print(len(featcol), mean, std)
        
    # Fill Nan values of APOE4 gene with 0
    data['APOE4'] = data['APOE4'].apply(pd.to_numeric, errors = 'coerce')
    data['APOE4'].fillna(0, inplace=True)

    # Normalize the image feature columns

    for name in tqdm(data.columns.values):
        if('UCSFFSX' in name or 'UCSFFSL' in name):
            if(name.startswith('ST') and 'STATUS' not in name):
                featcol = data[data['PTID'].isin(train_ids)][name].values
                mean, std = np.mean(featcol), np.std(featcol)
                data[name] = (data[name] - mean)/(std + 1e-4)
                print(len(featcol), mean, std)
       
    # Save processed Dataframe to output_path
    data.to_csv(output_path)

def preprocess_imfeat(input_path,output_path,list_path):
    '''
    Routine to preprocess csv file for flare with
    image features experiment. Input should be a 
    cleaned/preprocessed ADNIMERGE file

    Input:
    input_path (string): path to the input csv
    output_path (string): path to the output csv

    id_vis_list (list): list of PTIDs and their 
    corresponding viscodes to take from the csv
    '''
    
    data = pd.read_csv(input_path, dtype=object) 
    id_vis_list = np.load(list_path)
    l = []
    for (ptid,viscode) in id_vis_list:
       ptid_c = ptid[:3]+'_'+ ptid[3] + '_' + ptid[4:]
       if viscode == 'm00':
            viscode_c = 'bl'
       else:
            viscode_c = viscode
       check = data[(data['PTID'] == ptid_c) & (data['VISCODE'] == str(viscode_c))]
       feat_path = '/mnt/nfs/work1/mfiterau/yeahuaywu/ad_clf/feature_outputs'
       check.insert(loc=len(check.columns),column='img_feat_path',value=feat_path+'/'+ptid+'/'+viscode+'/'+'features.pt')
       if not check.empty:
            l.append(check)
       else:
            print(viscode)
    imfeat_data = pd.concat(l)
    imfeat_data.to_csv(output_path,index=False)

if __name__ == '__main__':
    input_path = '../data/TADPOLE_D1_D2.csv'
    output_path = '../data/TADPOLE_D1_D2_m48.csv'
    pid_train = '../data/patientID_train_all.txt'
    list_path = '../data/ptid_viscode_list.npy'
#    preprocess_adni(input_path, output_path, pid_train)
    preprocess_imfeat(output_path,'../data/TADPOLE_images_only.csv',list_path)

"""
bl      1737
m06     1618
m12     1485
m18     1293
m24     1326
m36      853

m03      793
m30      750
m48      659
m60      354
m42      307
m72      255
m66      217
m78      213
m54      200
m84      199
m96      155
m90      129
m108     119
m120      71
m102       7
m114       1
"""
