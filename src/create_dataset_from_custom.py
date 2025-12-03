import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import re

# specify where the custom data is found
data_path = r"..\data\custom\values"
out_path = r"..\data\unprocessed\ml_health\set-a"
cohort_path = r"..\data\custom\cohort"

def format_time(minutes):
    """Convert minutes since admission to HH:MM string."""
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours:02d}:{mins:02d}"

def clean_value(value):
    """Round numeric values to 2 decimals, keep others as-is."""
    if pd.isna(value):
        return -1
    try:
        return round(float(value), 2)
    except:
        return value


def clean_static_value(param, value):
    """Clean static values to match reference format."""
    
    if pd.isna(value):
        return -1
    
    # Gender
    if param == "Gender":
        if str(value).lower() in ["man", "male"]:
            return 1
        elif str(value).lower() in ["vrouw", "female"]:
            return 0
        else:
            return -1

    # RecordID & ICUType → int
    if param in ["RecordID", "ICUType"]:
        try:
            return int(float(value))
        except:
            return -1

    # Age, Height, Weight → float or int
    if param in ["Age", "Height", "Weight"]:
        value_str = str(value)
        # Range: "60-69" → take midpoint
        match = re.match(r"(\d+)-(\d+)", value_str)
        if match:
            low, high = map(float, match.groups())
            val = (low + high) / 2
            # Age as int, height/weight as float with 1 decimal
            return int(round(val)) if param=="Age" else round(val, 1)
        # Greater than: "190+" → add 0.5 as estimate
        match = re.match(r"(\d+)\+", value_str)
        if match:
            val = float(match.group(1))
            return int(val) if param=="Age" else round(val + 0.5, 1)
        # Single numeric value
        try:
            val = float(value)
            return int(val) if param=="Age" else round(val, 1)
        except:
            return -1

    # Default: keep as is
    return value

    
def write_measurements(admission_id, group_df):
    """
    admission_id : patient ID
    group_df     : DataFrame with columns ["admissionid", "time", "item", "value"]
    """

    filename = os.path.join(out_path, f"{admission_id}.txt")

    new_df = group_df[["time", "item", "value"]].copy()
    # Convert time from minutes to HH:MM
    new_df["time"] = new_df["time"].apply(format_time)
    # Round numeric values
    new_df["value"] = new_df["value"].apply(clean_value)

    new_df.columns = ["Time", "Parameter", "Value"]
    new_df = new_df.astype({"Time": "string", "Parameter": "string", "Value": "string"})

    # If file exists, load it
    if os.path.exists(filename):
        existing = pd.read_csv(filename).astype({"Time": "string", "Parameter": "string", "Value": "string"})
    else:
        existing = pd.DataFrame(columns=["Time", "Parameter", "Value"]).astype({
            "Time": "string",
            "Parameter": "string",
            "Value": "string"
        })

    ## Concatenate only non-empty frames to avoid FutureWarning
    frames = [df for df in [existing, new_df] if not df.empty]
    out_df = pd.concat(frames, ignore_index=True)

    # Sort and save
    out_df = out_df.sort_values(by=["Time", "Parameter"]).reset_index(drop=True)
    out_df.to_csv(filename, index=False)


def write_static_measurements(df, out_path):
    """
    df: DataFrame with patient/admission info
    txt_folder: folder where <admissionid>.txt will be stored
    """
    # Columns to extract and map to triplets
    mapping = {
        "patientid": "RecordID",
        "agegroup": "Age",
        "gender": "Gender",
        "heightgroup": "Height",
        "ICUType": "ICUType",
        "weightgroup": "Weight"
    }
    
    # Loop over all admissions
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Static values"):
        admission_id = row["admissionid"]
        triplets = []

        for col, param in mapping.items():
            value = row.get(col, -1)
            value = clean_static_value(param, value)
            triplets.append(("00:00", param, value))

        # Write to file
        filename = os.path.join(out_path, f"{admission_id}.txt")
        
        # New static DataFrame
        new_df = pd.DataFrame(triplets, columns=["Time", "Parameter", "Value"])
        new_df = new_df.astype({"Time": "string", "Parameter": "string", "Value": "string"})
        
        # Load existing file if present
        if os.path.exists(filename):
            existing = pd.read_csv(filename).astype({"Time": "string", "Parameter": "string", "Value": "string"})
        else:
            existing = pd.DataFrame(columns=["Time", "Parameter", "Value"]).astype({
                "Time": "string",
                "Parameter": "string",
                "Value": "string"
            })
        
        # Concatenate only non-empty frames
        frames = [df for df in [existing, new_df] if not df.empty]
        out_df = pd.concat(frames, ignore_index=True)

        # Sort and save
        out_df = out_df.sort_values(by=["Time", "Parameter"]).reset_index(drop=True)
        out_df.to_csv(filename, index=False)


def process_values():
    # loop through files in the directory WITH progress bar
    csv_files = [f for f in os.listdir(data_path) if f.endswith(".csv")]

    for filename in tqdm(csv_files, desc="Processing files"):
        full_path = os.path.join(data_path, filename)
        print("\nLoading:", full_path)

        df = pd.read_csv(full_path)
        triplets_df = df[["admissionid", "time", "item", "value"]]

        # group by admissionid
        df_by_admission = {
            admission_id: group.reset_index(drop=True)
            for admission_id, group in triplets_df.groupby("admissionid")
        }

        # progress bar for admissions in this file
        for admission_id, group_df in tqdm(df_by_admission.items(),
                                        desc=f"Admissions in {filename}",
                                        leave=False):
            write_measurements(admission_id, group_df)

def process_static_values():
    filename = os.path.join(cohort_path, "sepsis_patients_legacy.csv")
    df = pd.read_csv(filename)

    write_static_measurements(df, out_path)

def create_outcomes():
    filename = os.path.join(cohort_path, "sepsis_patients_legacy.csv")
    df = pd.read_csv(filename)

    # --- Impute length of stay ---
    mean_stay = df["lengthofstay"].mean()
    sd_stay = df["lengthofstay"].std()
    mask = df["lengthofstay"].isna()
    df.loc[mask, "lengthofstay"] = np.random.normal(mean_stay, sd_stay, mask.sum())

    df["RecordID"] = df["admissionid"].astype(str)

    # --- Create random Sepsis3 column (0 or 1) ---
    df["Sepsis3"] = np.random.randint(0, 2, size=len(df))

    # --- Create onset_time ---
    # For sepsis cases (1): draw from normal distribution
    # For non-sepsis cases (0): use unmodified lengthofstay
    onset_time = np.where(
        df["Sepsis3"] == 1,
        np.random.normal(mean_stay, sd_stay, len(df)).astype(int),
        df["lengthofstay"]
    )
    # Clip negative values to zero
    onset_time = np.clip(onset_time, a_min=0, a_max=None)
    df["Onset_time"] = onset_time

    # --- Select output columns ---
    df_out = df[["RecordID", "Onset_time", "Sepsis3"]]

    # --- Write to txt file (CSV-style) ---
    out = os.path.join(r"C:\Users\Skyfinder\Projects\STraTS\data\unprocessed\ml_health", "Outcomes-a.txt")
    df_out.to_csv(out, index=False)

    



def prep_dirs():
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ## clear the data completely
    for f in os.listdir(out_path):
        os.remove(os.path.join(out_path, f))

# clear old output and make sure the directories are set up
# prep_dirs()

## run
# process_static_values()
# process_values()
# create_outcomes()


