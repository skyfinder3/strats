import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
from datetime import timedelta

# specify where the custom data is found
data_path = r"..\data\custom\values"
out_path = r"..\data\unprocessed\ml_health\set-a"
cohort_path = r"..\data\custom\cohort"
matias_path = r"..\data\custom\Matias"

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


def process_values(in_path, out_path):
    # read csv
    df_values = pd.read_csv(in_path)
    
    # parse datetimes (allow mixed formats)
    df_values["event_datetime"] = pd.to_datetime(
        df_values["event_datetime"],
        errors="coerce",
        utc=True
    )

    # group by visit_occurrence_id
    grouped = list(df_values.groupby("visit_occurrence_id"))  # convert to list to know total length
    
    # --- wrap with tqdm for progress bar ---
    for visit_id, g in tqdm(grouped, desc="Processing admissions"):
        # sort by datetime, NaT last
        g = g.sort_values("event_datetime", na_position="last")

        # first valid datetime as t=0
        if g["event_datetime"].notna().any():
            t0 = g["event_datetime"].min()
        else:
            t0 = pd.Timestamp(0)

        rows = []
        
        # --- Required static vars ---
        required_vars = ["AdmissionID", "Age", "Gender", "Height"]
        required_values = {
            "AdmissionID": visit_id,
            "Age": -1,
            "Gender": -1,
            "Height": -1
        }

        # fill static values from dataframe
        for _, row in g.iterrows():
            var = row["variable_name"]
            val = row["value"]
            if pd.isna(val):
                val = -1
            else:
                val = round(val, 2)

            if var == "female_static":
                required_values["Gender"] = 1 if val == 1.0 else 0
            elif var in required_vars and var != "AdmissionID":
                required_values[var] = val

        # append required vars in order at t=00:00
        for var in required_vars:
            rows.append(["00:00", var, required_values[var]])

        # append remaining events
        for _, row in g.iterrows():
            var = row["variable_name"]
            if var in required_vars or var == "female_static":
                continue

            dt = row["event_datetime"]
            if pd.isna(dt):
                time_str = "00:00"
            else:
                delta = dt - t0
                total_minutes = int(delta.total_seconds() // 60)
                hours = total_minutes // 60
                minutes = total_minutes % 60
                time_str = f"{hours:02d}:{minutes:02d}"

            val = row["value"]
            if pd.isna(val):
                val = -1
            else:
                val = round(val, 2)
            # only look at first 48 hours
            if hours <= 47:
                rows.append([time_str, var, val])

        df_out = pd.DataFrame(rows, columns=["Time", "Parameter", "Value"])

        # save
        filename = os.path.join(out_path, f"{visit_id}.txt")
        df_out.to_csv(filename, index=False)

def create_outcomes(path, out_path):
    # get admission ids of the dataset
    
    df_admissions = pd.read_csv(path)
    
    # get sepsis cohort to produce labels
    sepsis_cohort = os.path.join(cohort_path, "sepsis_patients_legacy.csv")
    # get cohort that has sepsis for inital label config
    df_sepsis_cohort = pd.read_csv(sepsis_cohort)

    outcome_df = pd.DataFrame()

    outcome_df["AdmissionID"] = df_admissions["admission_id"]
    outcome_df["Sepsis3"] = outcome_df["AdmissionID"].isin(df_sepsis_cohort["admissionid"]).astype(int)
    
    # --- Select output columns ---
    outcome_df[["AdmissionID", "Sepsis3"]]

    # --- Write to txt file (CSV-style) ---
    out = os.path.join(out_path, "Outcomes-a.txt")
    outcome_df.to_csv(out, index=False)

    



def prep_dirs():
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ## clear the data completely
    for f in os.listdir(out_path):
        os.remove(os.path.join(out_path, f))


os.chdir(r"C:\Users\Skyfinder\Projects\STraTS\src")
# clear old output and make sure the directories are set up
# prep_dirs()

## run

process_values(os.path.join(r"..\data\custom\Matias", "output.csv"), r"..\data\unprocessed\ml_health\set-a")
create_outcomes(os.path.join(r"..\data\custom\Matias", "Dataset_corrected.csv"), r"..\data\unprocessed\ml_health")


