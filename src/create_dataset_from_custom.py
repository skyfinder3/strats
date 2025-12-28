import os
from pdb import run
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
from datetime import timedelta
from datetime import datetime
import argparse


def process_values(
    input_path,
    out_path, 
    debug=False,
    pred_window_hours=0
):
    # append set-a to out_path
    out_path = os.path.join(out_path, "set-a")
    os.makedirs(out_path, exist_ok=True)

    # --- Load values ---
    df_values = pd.read_csv(input_path)

    # load cohort
    df_cohort = pd.DataFrame({
        "visit_occurrence_id": df_values["visit_occurrence_id"].unique()
    })

    # only keep values in cohort
    df_values = df_values[df_values["visit_occurrence_id"].isin(df_cohort["visit_occurrence_id"].values)]

    # --- Clean datetimes ---
    df_values["event_datetime"] = pd.to_datetime(
        df_values["event_datetime"],
        errors="coerce",
        utc=True
    )

    # extract admission time
    df_admission_start_time = df_values[df_values['variable_name'] == 'admission_start']
    # get a time map for admission start
    admission_time_map = dict(
        zip(
            df_admission_start_time["visit_occurrence_id"],
            df_admission_start_time["event_datetime"]
        )
    )

    # --- Get sepsis labels ---
    df_sepsis = df_values[df_values['variable_name'] == 'sepsis']


    # remove harmful variables death and death in visit
    df_values = df_values[~df_values['variable_name'].isin(['death', 'death_in_visit', 'sepsis'])]

    if debug: print(f"Total sepsis episodes after time filter: {len(df_sepsis)}")

    # get a time map for sepsis onset times
    sepsis_time_map = dict(
        zip(
            df_sepsis["visit_occurrence_id"],
            df_sepsis["time_since_admission_hours"]
        )
    )
    
    visits = list(df_values.groupby("visit_occurrence_id"))

    # process one patient as a time
    for visit_id, g in tqdm(visits, desc="Processing admissions"):

        g = g.sort_values(by='event_datetime', ascending=True)
        
        t0 = admission_time_map[visit_id]

        if visit_id in sepsis_time_map:
            t_end = t0 + timedelta(hours=sepsis_time_map[visit_id] - pred_window_hours)
        else:
            t_end = t0 + timedelta(hours=14 * 24) # 14 day max
        
        rows = []

        # --- Static variables --- first
        required_vars = ["AdmissionID", "Age", "Gender", "Height"]
        required_values = {
            "AdmissionID": visit_id,
            "Age": 18,     # TODO update once in the data
            "Gender": -1,
            "Height": 165,  # TODO update once in the data
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
            
            if var == "age":
                required_values["Age"] = val
            elif var in required_vars and var != "AdmissionID":
                required_values[var] = val
            
            if var == "height_static":
                required_values["Height"] = 165
            elif var in required_vars and var != "AdmissionID":
                required_values[var] = val


        # append static values first with 00:00 time
        for var in required_vars:
            rows.append(["00:00", var, required_values[var]])

        # --- Time-dependent variables ---
        for _, row in g.iterrows():
            # skip static variables
            var = row["variable_name"]
            if var in required_vars or var == "female_static" or var == "height_static" or var == "age":
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
    input_path,
    output_path,
    debug = False
):
    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    df_values = pd.read_csv(input_path)

    # read cohort
    df_cohort =pd.DataFrame({
        "visit_occurrence_id": df_values["visit_occurrence_id"].unique()
    })

    # read lables
    df_sepsis = df_values[df_values['variable_name'] == 'sepsis']

    if debug: print("Total sepsis episodes: "+str(len(df_sepsis)))

    if debug: print(f"Total cohort size: {len(df_cohort)}")

    # create outcome file. all cohort admissions, mark sepsis=1 if in sepsis list
    outcome_df = pd.DataFrame({
        "AdmissionID": df_cohort["visit_occurrence_id"],
        "Sepsis3": df_cohort["visit_occurrence_id"].isin(
            df_sepsis["visit_occurrence_id"]
        ).astype(int)
    })

    if debug: print(f"Total sepsis episodes in cohort: {len(outcome_df[outcome_df['Sepsis3']==1])}") 
    if debug: print(f"Total admissions in outcome file: {len(outcome_df)}")

    out = os.path.join(output_path, "Outcomes-a.txt")
    outcome_df.to_csv(out, index=False)



def write_cohort_log(
    experiment_name: str,
    args
):
    """
    Create a log.txt file summarizing a cohort-based sepsis prediction experiment.
    Ensures outcomes are restricted to the cohort and sepsis onsets are unique
    per admission (earliest onset only).
    """

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "log.txt")

    # --- Load data ---
    df_values = pd.read_csv(args.features_file)

    df_outcomes = df_values[df_values['variable_name'] == 'sepsis']

    df_values = df_values[~df_values['variable_name'].isin(['death', 'death_in_visit', 'sepsis'])]

    # --- Restrict outcomes to cohort admissions only ---
    df_cohort = pd.DataFrame({'visit_occurrence_id': df_values['visit_occurrence_id'].unique()})

    n_total_values = len(df_values)
   
    # --- Counts ---
    total_admissions = df_cohort["visit_occurrence_id"].nunique()
    septic_admissions = len(df_outcomes)
    n_variables = df_values["variable_name"].nunique()

    # --- Histogram ---
    sepsis_times = df_outcomes["time_since_admission_hours"].tolist()

    if len(sepsis_times) > 0:
        counts, bin_edges = np.histogram(sepsis_times, bins=df_outcomes["time_since_admission_hours"].nunique())
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
        for name, path in args.items():
            f.write(f"{name}: {path}\n")
        f.write("\n")

        # Cohort summary
        f.write("COHORT SUMMARY\n")
        f.write("-" * 60 + "\n")
        f.write(f"Total number of admissions: {total_admissions}\n")
        f.write(f"Total number of septic admissions: {septic_admissions}\n")
        f.write(f"Number of sepsis episodes (total): {n_total_values}\n")
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

def prep_dirs(args):
    # output dir
    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)

    # values dir
    os.makedirs(os.path.join(args.output_dir, "set-a"), exist_ok=True)

    ## clear the data completely
    ##for f in os.listdir(args.output_dir):
    ##    os.remove(os.path.join(args.output_dir, f))

def run_all(args):
    os.chdir(r"C:\Users\Skyfinder\Projects\STraTS\src")
    # clear old output and make sure the directories are set up
    prep_dirs(args)

    # def pred window hours
    pred_window_hours = 2

    create_outcomes(input_path = args.features_file, output_path = args.output_dir, debug = True)

    process_values(input_path = args.features_file, out_path = args.output_dir, pred_window_hours=pred_window_hours, debug = True)

'''
log_file = write_cohort_log(
    experiment_name=f"Sepsis-3 Prediction matias_data output_1.1.csv",
    paths=paths,
)
'''

def parse_args() -> argparse.Namespace:
    """Function to parse arguments."""
    parser = argparse.ArgumentParser()

    # dataset related arguments
    parser.add_argument('--features_file', type=str, default=r'..\data\custom\Matias\output_1.1.csv')
    parser.add_argument('--output_dir', type=str, default=r"..\data\unprocessed\ml_health")
    parser.add_argument('--pred_window_hours', type=int, default=0)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # Preliminary setup.
    args = parse_args()
    run_all(args)