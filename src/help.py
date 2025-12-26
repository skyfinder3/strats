import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import pickle

# specify where the custom data is found
data_path = r"..\data\custom\values"
out_path = r"..\data\unprocessed\ml_health\set-a"
cohort_path = r"..\data\custom\cohort"

df_values = pd.read_csv(os.path.join(r"..\data\custom\Matias", "output.csv"))

summary = pd.DataFrame({
    "dtype": df_values.dtypes,
    "nan_count": df_values.isna().sum(),
    "nan_pct": df_values.isna().mean() * 100,
    "non_null_count": df_values.notna().sum(),
})

print(summary)

print