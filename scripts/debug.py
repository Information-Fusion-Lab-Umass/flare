import pandas as pd
from tqdm import tqdm

data = pd.read_csv('../data/TADPOLE_D1_D2_proc.csv')

minvals = []; maxvals = []
for name in tqdm(data.columns.values):
    if('UCSFFSX' in name or 'UCSFFSL' in name):
        if(name.startswith('ST') and 'STATUS' not in name):
            minvals.append(data[name].min())
            maxvals.append(data[name].max())

print(min(minvals), max(minvals))
print(min(maxvals), max(maxvals))
print(minvals)
print(maxvals)

