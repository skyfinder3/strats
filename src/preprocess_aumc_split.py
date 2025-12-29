from tqdm import tqdm
import os
import pandas as pd
import pickle
import numpy as np
import argparse

#### Alternative way for generating splits, not used right now.
def read_ts(raw_data_path, set_name):
    import os
    import pandas as pd
    from tqdm import tqdm

    ts_list = []
    base_path = os.path.join(raw_data_path, f"set-{set_name}")
    n_files = 0
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

        n_files +1

        if n_files >1000:
            break

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


def run_all(args):

    # get time series
    ts = pd.concat([read_ts(args.raw_data_path, set_name) 
                    for set_name in ['a_new','b_new','c_new']])
    # get outcomes
    oc_org = pd.concat([read_outcomes(args.raw_data_path, set_name) 
                    for set_name in ['a','b','c']])

    # Keep only ts_ids present in outcomes.
    ts_ids = set(ts.ts_id)
    oc = oc_org[oc_org.ts_id.isin(ts_ids)]


    print(f"Dropped {len(ts_ids) - len(oc)} admissions due to tiny time series")

    # Drop duplicates.
    ts = ts.drop_duplicates()
        

    # make test train val split
    # Generate split.
    train_valid_ids = list(oc.loc[oc.subset!='a'].ts_id)
    np.random.seed(123)
    np.random.shuffle(train_valid_ids)
    bp = int(0.8*len(train_valid_ids))
    train_ids = train_valid_ids[:bp]
    valid_ids = train_valid_ids[bp:]
    test_ids = np.array(oc.loc[oc.subset=='a'].ts_id)
    infer_ids = test_ids
    oc.drop(columns='subset', inplace=True)

    # Store data.
    os.makedirs(args.output_dir, exist_ok=True)
    pickle.dump([ts, oc, train_ids, valid_ids, test_ids, infer_ids], 
                open(os.path.join(args.output_dir, args.dataset_name + '.pkl'),'wb'))


def parse_args() -> argparse.Namespace:
    """Function to parse arguments."""
    parser = argparse.ArgumentParser()

    # dataset related arguments
    parser.add_argument('--raw_data_path', type=str, default=r'..\data\unprocessed\ml_health')
    parser.add_argument('--output_dir', type=str, default=r"..\data\processed")
    parser.add_argument('--dataset_name', type=str, default=r"aumc-test")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # Preliminary setup.
    args = parse_args()
    run_all(args)