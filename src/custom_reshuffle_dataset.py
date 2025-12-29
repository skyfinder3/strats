import os
import shutil
import numpy as np
import pandas as pd



###### Script used for debugging purpuses to track down label leakage ####

BASE_DIR = r"C:\Users\Skyfinder\Projects\STraTS\data\unprocessed\ml_health-8"
SRC_SET = "set-a"

np.random.seed(123)

# Load outcomes
outcomes_path = os.path.join(BASE_DIR, "Outcomes-a.txt")
oc = pd.read_csv(outcomes_path)  # change sep if needed

all_ids = oc.AdmissionID.unique().tolist()
np.random.shuffle(all_ids)

# Reference-style split
n = len(all_ids)
split1 = n // 3
split2 = 2 * n // 3

set_a_ids = set(all_ids[:split1])
set_b_ids = set(all_ids[split1:split2])
set_c_ids = set(all_ids[split2:])

for s in ["set-a_new", "set-b_new", "set-c_new"]:
    os.makedirs(os.path.join(BASE_DIR, s), exist_ok=True)

def extract_ts_id(filename):
    # EXAMPLES:
    # "12345.txt" → 12345
    # "ts_12345.csv" → 12345
    return int("".join(filter(str.isdigit, filename)))

src_ts_dir = os.path.join(BASE_DIR, SRC_SET)


for fname in os.listdir(src_ts_dir):
    if not fname.endswith((".txt", ".csv")):
        continue
    if fname.startswith("Outcomes"):
        continue

    ts_id = extract_ts_id(fname)
    src = os.path.join(src_ts_dir, fname)

    if ts_id in set_a_ids:
        dst_dir = "set-a_new"
    elif ts_id in set_b_ids:
        dst_dir = "set-c_new"
    elif ts_id in set_c_ids:
        dst_dir = "set-b_new"
    else: 
        continue

    shutil.copy(src, os.path.join(BASE_DIR, dst_dir, fname))

def write_outcomes(folder, ids, set):
    out = oc[oc.AdmissionID.isin(ids)].copy()
    out.to_csv(
        os.path.join(BASE_DIR, f"Outcomes-{set}.txt"),
        index=False
    )

write_outcomes("set-a_new", set_a_ids, "a_new")
write_outcomes("set-b_new", set_b_ids, "b_new")
write_outcomes("set-c_new", set_c_ids, "c_new")