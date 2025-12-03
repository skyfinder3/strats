from tqdm import tqdm
import os
import pandas as pd
import pickle
import numpy as np


RAW_DATA_PATH = r'C:\Users\Skyfinder\Projects\STraTS\data\unprocessed\ml_health'


def read_ts(raw_data_path, set_name):
    ts = []
    pbar = tqdm(os.listdir(raw_data_path+'/set-'+set_name), 
                desc='Reading time series set '+set_name)
    for f in pbar:
        data = pd.read_csv(raw_data_path+'/set-'+set_name+'/'+f).iloc[1:]
        data = data.loc[data.Parameter.notna()]
        if len(data)<=5:
            continue
        data = data.loc[data.Value>=0] # neg Value indicates missingness.
        data['RecordID'] = f[:-4]
        ts.append(data)
    ts = pd.concat(ts)
    ts.Time = ts.Time.apply(lambda x:int(x[:2])*60
                            +int(x[3:])) # No. of minutes since admission.
    ts.rename(columns={'Time':'minute', 'Parameter':'variable', 
                       'Value':'value', 'RecordID':'ts_id'}, inplace=True)
    return ts


def read_outcomes(raw_data_path, set_name):
    oc = pd.read_csv(raw_data_path+'/Outcomes-'+set_name+'.txt', 
                     usecols=['RecordID', 'Onset_time', 'Sepsis3']) # TODO maybe adjust??
    oc['subset'] = set_name
    oc.RecordID = oc.RecordID.astype(str)
    oc.rename(columns={'RecordID':'ts_id', 'Onset_time':'onset_time', 
                       'Sepsis3':'sepsis_3'}, inplace=True)
    return oc


ts = pd.concat([read_ts(RAW_DATA_PATH, set_name) 
                for set_name in ['a']])
oc = pd.concat([read_outcomes(RAW_DATA_PATH, set_name) 
                for set_name in ['a']])
# TODO If we want other sets, do that here, dont think we need to though

ts_ids = sorted(list(ts.ts_id.unique()))
oc = oc.loc[oc.ts_id.isin(ts_ids)]

# Drop duplicates.
ts = ts.drop_duplicates()

# Convert categorical to numeric.
ii = (ts.variable=='ICUType')
for val in [4,3,2,1]:
    kk = ii&(ts.value==val)
    ts.loc[kk, 'variable'] = 'ICUType_'+str(val)
ts.loc[ii, 'value'] = 1
    

# make test train val split
all_ids = list(oc.ts_id)
np.random.seed(123)
np.random.shuffle(all_ids)
bp1 = int(0.7 * len(all_ids))   # 70% train
bp2 = int(0.85 * len(all_ids))  # next 15% val
train_ids = all_ids[:bp1]
valid_ids = all_ids[bp1:bp2]
test_ids = all_ids[bp2:]

# Store data.
os.makedirs('../data/processed', exist_ok=True)
pickle.dump([ts, oc, train_ids, valid_ids, test_ids], 
            open('../data/processed/aumc.pkl','wb'))