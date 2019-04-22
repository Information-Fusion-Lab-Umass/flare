import pandas as pd
import numpy as np
from tqdm import tqdm
import ipdb

def preprocess_adni(input_path, output_path):
    data = pd.read_csv(input_path, dtype = object)
    
    # Add VISNUM column
    visit_id = {
            'bl': 0,
            'm06': 1,
            'm12': 2,
            'm18': 3,
            'm24': 4,
            'm36': 5
            }
    data['VISNUM'] = data['VISCODE'].apply(lambda x: visit_id[x] \
            if x in visit_id else -1)

    # Retain only rows with required visit_id
    data = data.loc[data['VISNUM'] != -1]

    # Impute missing image feature data  
    data.sort_values(by = ['PTID', 'VISNUM'], inplace = True)
    data = data.groupby('PTID').ffill()
    for name in tqdm(data.columns.values):
        if('UCSFFSX' in name or 'UCSFFSL' in name):
            if(name.startswith('ST') and 'STATUS' not in name):
                data[name] = data[name].apply(pd.to_numeric, errors = 'coerce')
                data[name].fillna(data[name].mean(), inplace=True)
    
    # Fill Nan values of APOE4 gene with 0
    data['APOE4'] = data['APOE4'].apply(pd.to_numeric, errors = 'coerce')
    data['APOE4'].fillna(0, inplace=True)
       
    # Save processed Dataframe to output_path
    data.to_csv(output_path)

if __name__ == '__main__':
    input_path = '../data/TADPOLE_D1_D2.csv'
    output_path = '../data/TADPOLE_D1_D2_proc.csv'
    preprocess_adni(input_path, output_path)

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
