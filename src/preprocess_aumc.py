from tqdm import tqdm
import os
import pandas as pd
import pickle
import numpy as np
import argparse


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


def run_all(args):

    # get time series
    ts = pd.concat([read_ts(args.raw_data_path, set_name) 
                    for set_name in ['a']])
    # get outcomes
    oc = pd.concat([read_outcomes(args.raw_data_path, set_name) 
                    for set_name in ['a']])

    # Keep only ts_ids present in outcomes.
    ts_ids = sorted(list(ts.ts_id.unique()))
    oc = oc.loc[oc.ts_id.isin(ts_ids)]

    # Drop duplicates.
    ts = ts.drop_duplicates()
        

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
    infer_ids = all_ids[bp2:]  # using test as infer for now

    # Store data.
    os.makedirs(args.output_dir, exist_ok=True)
    pickle.dump([ts, oc, train_ids, valid_ids, test_ids, infer_ids], 
                open(os.path.join(args.output_dir, args.dataset_name, '.pkl'),'wb'))


def parse_args() -> argparse.Namespace:
    """Function to parse arguments."""
    parser = argparse.ArgumentParser()

    # dataset related arguments
    parser.add_argument('--raw_data_path', type=str, default=r'..\data\unprocessed\ml_health')
    parser.add_argument('--output_dir', type=str, default=r"..\data\processed")
    parser.add_argument('--dataset_name', type=str, default=r"aumc")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # Preliminary setup.
    args = parse_args()
    run_all(args)