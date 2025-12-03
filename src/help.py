import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import re

# specify where the custom data is found
data_path = r"..\data\custom\values"
out_path = r"..\data\unprocessed\ml_health\set-a"
cohort_path = r"..\data\custom\cohort"

'''
filename = os.path.join(r"..\data\custom\Matias", "output.csv")
df = pd.read_csv(filename)

print(df.head())

'''

filename = os.path.join(r"..\data\custom\Matias", "Dataset.csv")
df = pd.read_csv(filename)

# Extract column names
cols = df.columns.tolist()

# Convert to dataframe with one column
df_ids = pd.DataFrame({"admission_id": cols})
print(df_ids.head())

out_filename = os.path.join(r"..\data\custom\Matias", "Dataset_corrected.csv")
df_ids.to_csv(out_filename, index=False)


