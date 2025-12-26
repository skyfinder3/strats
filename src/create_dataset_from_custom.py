import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
from datetime import timedelta
from datetime import datetime

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


def process_values(
    cohort_path,
    labels_path,
    input_path,
    out_path, 
    debug=False,
):
    os.makedirs(out_path, exist_ok=True)

    # Load cohort
    df_cohort = pd.read_csv(cohort_path)

    # --- Load values ---
    df_values = pd.read_csv(input_path)

    # only keep values in cohort
    df_values = df_values[df_values["visit_occurrence_id"].isin(df_cohort["admission_id"].values)]

    # --- Clean datetime ---
    df_values["event_datetime"] = pd.to_datetime(
        df_values["event_datetime"],
        errors="coerce",
        utc=True
    )

    # --- Load sepsis labels ---
    df_sepsis = pd.read_csv(labels_path)
    df_sepsis = df_sepsis.dropna(subset=["time"]).copy()

    # filter only positive time values
    df_sepsis = df_sepsis[df_sepsis["time"] >= 0]

    if debug: print(f"Total sepsis episodes after time filter: {len(df_sepsis)}")

    # keep only earliest valid episode per admission
    df_sepsis = (
        df_sepsis
        .sort_values("time")
        .groupby("admission_id", as_index=False)
        .first()
    )
    
    # update sepsis time to 1 if time == 0
    df_sepsis.loc[df_sepsis["time"] == 0, "time"] = 1
    df_sepsis["sepsis_time_hours"] = df_sepsis["time"] * 24

    visits = list(df_values.groupby("visit_occurrence_id"))

    # process one patient as a time
    for visit_id, g in tqdm(visits, desc="Processing admissions"):

        # sort events
        g = g.sort_values("event_datetime", na_position="last")

        # get earliest datetime as admission time
        t0 = g["event_datetime"].min()

        # get last possible datetime depending on sepsis time
        if visit_id in df_sepsis["admission_id"].values:
            sepsis_time = df_sepsis.loc[df_sepsis["admission_id"] == visit_id, "sepsis_time_hours"].values[0]
            t_end = t0 + timedelta(hours=sepsis_time)
        else:
            t_end = t0 + timedelta(hours=14*48)  # set observation
        
        rows = []

        # --- Static variables ---
        required_vars = ["AdmissionID", "Age", "Gender", "Height"]
        required_values = {
            "AdmissionID": visit_id,
            "Age": 18,     # TODO
            "Gender": -1,
            "Height": 165  # TODO
        }

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

        # append static values first with 00:00 time
        for var in required_vars:
            rows.append(["00:00", var, required_values[var]])

        # --- Time-dependent variables ---
        for _, row in g.iterrows():
            var = row["variable_name"]
            if var in required_vars or var == "female_static":
                continue
            
            # get date time of event
            dt = row["event_datetime"]
            # if no datetime, skip
            if pd.isna(dt):
                continue

            # if datetime is after t_end, skip
            if dt > t_end:
                continue
            
            # if visit_id in sepsis admission id and tieme after sepsis time on set, skip
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

            rows.append([time_str, var, val])

        # skip tiny admissions
        if len(rows) <= len(required_vars):
            continue

        df_out = pd.DataFrame(rows, columns=["Time", "Parameter", "Value"])
        df_out.to_csv(os.path.join(out_path, f"{visit_id}.txt"), index=False)



def create_outcomes(
    cohort_path,
    labels_path,
    output_path,
    debug = False
):
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    df_cohort = pd.read_csv(cohort_path)
    df_sepsis = pd.read_csv(labels_path)

    if debug: print(f"Total sepsis episodes before restriction: {len(df_sepsis)}")

    # keep earliest sepsis episode
    df_sepsis = (
        df_sepsis
        .dropna(subset=["time"])
        .sort_values("time")
        .groupby("admission_id", as_index=False)
        .first()
    )

    # get only episodes vith times after admission
    valid = df_sepsis["time"] >= 0
    df_sepsis = df_sepsis.loc[valid]

    plt.hist(df_sepsis["time"])
    plt.show()


    if debug: print(f"Total unique sepsis episodes with valid times: {len(df_sepsis)}")

    if debug: print(f"Total cohort size: {len(df_cohort)}")
    if debug: print(f"Total sepsis episodes in cohort: {len(df_sepsis)}")

    # create outcome file. all cohort admissions, mark sepsis=1 if in sepsis list
    outcome_df = pd.DataFrame({
        "AdmissionID": df_cohort["admission_id"],
        "Sepsis3": df_cohort["admission_id"].isin(
            df_sepsis["admission_id"]
        ).astype(int)
    })

    if debug: print(f"Total admissions in outcome file: {len(outcome_df)}")

    out = os.path.join(output_path, "Outcomes-a.txt")
    outcome_df.to_csv(out, index=False)



def write_cohort_log(
    experiment_name: str,
    paths: dict,
    admission_id_col: str = "admission_id",
    sepsis_time_col: str = "time",
    n_time_bins: int = 10,
):
    """
    Create a log.txt file summarizing a cohort-based sepsis prediction experiment.
    Ensures outcomes are restricted to the cohort and sepsis onsets are unique
    per admission (earliest onset only).
    """

    os.makedirs(paths["output_dir"], exist_ok=True)
    log_path = os.path.join(paths["output_dir"], "log.txt")

    # --- Load data ---
    df_cohort = pd.read_csv(paths["cohort_file"])
    df_outcomes = pd.read_csv(paths["outcomes_file"])
    df_variables = pd.read_csv(paths["features_file"])

    # --- Restrict outcomes to cohort admissions only ---
    cohort_ids = df_cohort[admission_id_col].unique()

    df_outcomes = df_outcomes[
        df_outcomes[admission_id_col].isin(cohort_ids)
    ]

    n_total = len(df_outcomes)
    # --- Keep earliest sepsis onset per admission ---
    df_outcomes = df_outcomes.dropna(subset=["time"]).copy()

    df_outcomes["sepsis_time_hours"] = df_outcomes["time"] * 24

    # keep earliest valid episode per admission
    df_outcomes = (
        df_outcomes
        .sort_values("sepsis_time_hours")
        .groupby("admission_id", as_index=False)
        .first()
    )


    n_predictable = len(df_outcomes)

    # --- Counts ---
    total_admissions = df_cohort[admission_id_col].nunique()
    septic_admissions = df_outcomes[admission_id_col].nunique()
    n_variables = df_variables["variable_name"].nunique()

    # --- Histogram ---
    sepsis_times = df_outcomes[sepsis_time_col]

    if len(sepsis_times) > 0:
        counts, bin_edges = np.histogram(sepsis_times, bins=n_time_bins)
    else:
        counts, bin_edges = [], []

    # --- Write log ---
    with open(log_path, "w") as f:
        # Header
        f.write("=" * 60 + "\n")
        f.write(f"EXPERIMENT: {experiment_name}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Run timestamp: {datetime.now().isoformat()}\n\n")


        # Paths
        f.write("INPUT PATHS / FILES\n")
        f.write("-" * 60 + "\n")
        for name, path in paths.items():
            f.write(f"{name}: {path}\n")
        f.write("\n")

        # Cohort summary
        f.write("COHORT SUMMARY\n")
        f.write("-" * 60 + "\n")
        f.write(f"Total number of admissions: {total_admissions}\n")
        f.write(f"Total number of septic admissions: {septic_admissions}\n")
        f.write(f"Number of sepsis episodes (total): {n_total}\n")
        f.write(f"Number of predictable sepsis episodes: {n_predictable}\n")
        f.write(
            f"Sepsis prevalence: "
            f"{septic_admissions / total_admissions:.4f}\n\n"
        )

        # Histogram
        f.write("SEPSIS ONSET TIME HISTOGRAM\n")
        f.write("-" * 60 + "\n")
        if len(sepsis_times) == 0:
            f.write("No sepsis onset times available.\n\n")
        else:
            for i in range(len(counts)):
                f.write(
                    f"Bin {i + 1}: "
                    f"{bin_edges[i]:.2f} – {bin_edges[i + 1]:.2f} "
                    f"-> {counts[i]} admissions\n"
                )
            f.write("\n")

        # Variables
        f.write("PREDICTION VARIABLES\n")
        f.write("-" * 60 + "\n")
        f.write(f"Number of variables used for prediction: {n_variables}\n")

    return log_path

def prep_dirs():
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ## clear the data completely
    for f in os.listdir(out_path):
        os.remove(os.path.join(out_path, f))


os.chdir(r"C:\Users\Skyfinder\Projects\STraTS\src")
# clear old output and make sure the directories are set up
# prep_dirs()

## run
paths = {
    "cohort_file": os.path.join(r"..\data\custom\Matias", "Dataset_corrected.csv"),
    "outcomes_file": os.path.join(r"..\data\custom\labels", "sepsis3_septic_episodes.csv"),
    "features_file": os.path.join(r"..\data\custom\Matias", "output.csv"),
    "output_dir": r"..\data\unprocessed\ml_health"
}

create_outcomes(cohort_path = paths["cohort_file"], labels_path = paths["outcomes_file"], output_path = paths["output_dir"], debug = True)

process_values(cohort_path=paths["cohort_file"], labels_path = paths["outcomes_file"], input_path = paths["features_file"], out_path = r"..\data\unprocessed\ml_health\set-a", debug = True)


log_file = write_cohort_log(
    experiment_name=f"Sepsis-3 Prediction run 1 only positive times, episode within first day is processed as within the 2 day of admission",
    paths=paths,
)