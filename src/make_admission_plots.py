import csv
from datetime import timedelta
import os
import pandas as pd
import matplotlib.pyplot as plt

df_values = pd.read_csv(r'C:\Users\Skyfinder\Projects\STraTS\data\custom\Matias\df_long.csv')

# person_id,visit_occurrence_id,variable_name,concept_id,event_datetime,value,time_since_admission_hours,hour

# Filter for visit_occurence_id 20
visit_occurence_id = 20
df_values = df_values[df_values["visit_occurrence_id"] == visit_occurence_id]

variables = df_values["variable_name"].unique()
print(variables)

# --- Clean datetimes ---
df_values["event_datetime"] = pd.to_datetime(
    df_values["event_datetime"],
    errors="coerce",
    utc=True
)

# sepsis
df_sepsis = df_values[df_values['variable_name'] == 'sepsis']

# get a time map for sepsis onset times
sepsis_time_map = dict(
    zip(
        df_sepsis["visit_occurrence_id"],
        df_sepsis["time_since_admission_hours"]
    )
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

admission_time = admission_time_map[visit_occurence_id]

if visit_occurence_id in sepsis_time_map:
    t_end = admission_time + timedelta(hours=sepsis_time_map[visit_occurence_id])
else:
    t_end = admission_time + timedelta(hours=14 * 24) # 14 day max

rows = []
for _, row in df_values.iterrows():
    dt = row["event_datetime"]
    # if no datetime, skip

    # if visit_id in sepsis admission id and tieme after sepsis time on set, skip
    delta = dt - admission_time

    if dt > t_end:
        continue

    rows.append([delta, row["variable_name"], row["value"]])

df_rows = pd.DataFrame(
    rows,
    columns=["delta", "variable_name", "value"]
)

df_rows["delta_minutes"] = (
    df_rows["delta"].dt.total_seconds() / 60
)

plt.figure(figsize=(12, 6))

for var, group in df_rows.groupby("variable_name"):
    group = group.sort_values("delta_minutes")

    plt.step(
        group["delta_minutes"],
        group["value"],
        where="post",
        label=var,
        alpha=0.8
    )

# ── Vertical reference lines ───────────────────────────

# Admission (t = 0)
plt.axvline(
    x=0,
    color="black",
    linestyle="--",
    linewidth=2,
    label="Admission"
)

# Sepsis onset (if exists)
if visit_occurence_id in sepsis_time_map:
    sepsis_minutes = sepsis_time_map[visit_occurence_id] * 60
    plt.axvline(
        x=sepsis_minutes,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Sepsis onset"
    )

# ── Labels & layout ───────────────────────────────────

plt.xlabel("Time since admission (minutes)")
plt.ylabel("Value")
plt.title("Staircase Plot of Variables Over Time")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()