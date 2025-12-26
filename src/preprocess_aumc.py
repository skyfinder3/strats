from tqdm import tqdm
import os
import pandas as pd
import pickle
import numpy as np


RAW_DATA_PATH = r'C:\Users\Skyfinder\Projects\STraTS\data\unprocessed\ml_health'


def read_ts(raw_data_path, set_name):
    import os
    import pandas as pd
    from tqdm import tqdm

    ts_list = []
    base_path = os.path.join(raw_data_path, f"set-{set_name}")

    for fname in tqdm(os.listdir(base_path), desc=f"Reading time series set {set_name}"):
        fpath = os.path.join(base_path, fname)

        df = pd.read_csv(fpath)

        # drop header row duplication + invalid params
        df = df.iloc[1:]
        df = df[df["Parameter"].notna()]

        # skip tiny admissions
        if len(df) <= 5:
            continue

        # missingness encoded as negative values
        df = df[df["Value"] >= 0]

        # extract admission id
        df["ts_id"] = fname.replace(".txt", "")

        ts_list.append(df)

    if not ts_list:
        return pd.DataFrame()

    ts = pd.concat(ts_list, ignore_index=True)

    # --- Robust time parsing ---
    time_split = ts["Time"].str.split(":", expand=True).astype(int)
    ts["minute"] = time_split[0] * 60 + time_split[1]

    # rename columns
    ts.rename(
        columns={
            "Parameter": "variable",
            "Value": "value"
        },
        inplace=True
    )

    return ts[["ts_id", "minute", "variable", "value"]]



def read_outcomes(raw_data_path, set_name):
    oc = pd.read_csv(raw_data_path+'/Outcomes-'+set_name+'.txt', 
                     usecols=['AdmissionID', 'Sepsis3']) # TODO maybe adjust??
    oc['subset'] = set_name
    oc.AdmissionID = oc.AdmissionID.astype(str)
    oc.rename(columns={'AdmissionID':'ts_id',
                       'Sepsis3':'Sepsis3'}, inplace=True)
    return oc

# get time series
ts = pd.concat([read_ts(RAW_DATA_PATH, set_name) 
                for set_name in ['a']])
# get outcomes
oc = pd.concat([read_outcomes(RAW_DATA_PATH, set_name) 
                for set_name in ['a']])

# Keep only ts_ids present in outcomes.
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
print(f"Number of training samples: {len(train_ids)}")
valid_ids = all_ids[bp1:bp2]
print(f"Number of validation samples: {len(valid_ids)}")
test_ids = all_ids[bp2:]
print(f"Number of test samples: {len(test_ids)}")

# Store data.
os.makedirs('../data/processed', exist_ok=True)
pickle.dump([ts, oc, train_ids, valid_ids, test_ids], 
            open('../data/processed/aumc.pkl','wb'))