
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# -----------------------------------
# CAREFLOW - CLINICAL PATHWAY ANALYSIS
# Member 2
# -----------------------------------

INPUT_FILE = "data/sample_patient_events.csv"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load dataset
print("Loading patient event dataset...")

df = pd.read_csv(INPUT_FILE)

print("\nOriginal Dataset:")
print(df.head())

# 2. Standardize column names
df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

# Recognize common column names
aliases = {
    "case_id": ["case_id", "patient_id", "patientid"],
    "activity": ["activity", "activity_name", "event", "event_name"],
    "timestamp": ["timestamp", "time", "event_time"],
    "department": ["department", "dept"]
}

def find_column(possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    return None

case_col = find_column(aliases["case_id"])
activity_col = find_column(aliases["activity"])
time_col = find_column(aliases["timestamp"])
dept_col = find_column(aliases["department"])

required = [case_col, activity_col, time_col]

if any(col is None for col in required):
    raise ValueError(
        "Required columns missing. Expected patient/case ID, "
        "activity name and timestamp. Found: "
        + str(df.columns.tolist())
    )

# 3. Data preprocessing
print("\nCleaning dataset...")

df = df.drop_duplicates()

df = df.dropna(subset=required)

df[time_col] = pd.to_datetime(
    df[time_col], errors="coerce"
)

df = df.dropna(subset=[time_col])

df = df.sort_values([case_col, time_col])

# 4. Save cleaned dataset
clean_file = os.path.join(
    OUTPUT_DIR, "cleaned_patient_events.csv"
)

df.to_csv(clean_file, index=False)

print("Cleaned dataset saved:", clean_file)

# 5. Calculate patient journey duration
journeys = df.groupby(case_col)[time_col].agg(
    start_time="min",
    end_time="max"
)

journeys["journey_duration_minutes"] = (
    journeys["end_time"] - journeys["start_time"]
).dt.total_seconds() / 60

journeys.to_csv(
    os.path.join(OUTPUT_DIR, "patient_journey_duration.csv")
)

print("\nPatient Journey Duration:")
print(journeys.head())

# 6. Calculate waiting time between activities
df["next_timestamp"] = df.groupby(case_col)[time_col].shift(-1)

df["next_activity"] = df.groupby(case_col)[activity_col].shift(-1)

df["waiting_time_minutes"] = (
    df["next_timestamp"] - df[time_col]
).dt.total_seconds() / 60

transitions = df.dropna(
    subset=["next_activity", "waiting_time_minutes"]
).copy()

# 7. Identify bottlenecks
bottlenecks = transitions.groupby(activity_col)[
    "waiting_time_minutes"
].agg(
    average_wait="mean",
    maximum_wait="max",
    total_cases="count"
).sort_values(
    "average_wait", ascending=False
)

bottlenecks.to_csv(
    os.path.join(OUTPUT_DIR, "bottleneck_analysis.csv")
)

print("\nBottleneck Analysis:")
print(bottlenecks)

# 8. Generate waiting-time visualization
plt.figure(figsize=(11, 6))

bottlenecks["average_wait"].plot(
    kind="bar", rot=45
)

plt.title("CareFlow - Average Waiting Time by Activity")
plt.xlabel("Hospital Activity")
plt.ylabel("Waiting Time (Minutes)")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "waiting_time_analysis.png")
)

plt.close()

# 9. Generate patient journey visualization
activity_counts = df[activity_col].value_counts()

plt.figure(figsize=(10, 6))

activity_counts.plot(kind="bar", rot=45)

plt.title("Patient Activity Distribution")
plt.xlabel("Hospital Activity")
plt.ylabel("Number of Events")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "patient_activity_distribution.png")
)

plt.close()

# 10. Process mining using PM4Py
try:
    import pm4py

    event_log = df.rename(columns={
        case_col: "case:concept:name",
        activity_col: "concept:name",
        time_col: "time:timestamp"
    })

    event_log = pm4py.format_dataframe(
        event_log,
        case_id="case:concept:name",
        activity_key="concept:name",
        timestamp_key="time:timestamp"
    )

    process_model = pm4py.discover_bpmn_inductive(
        event_log
    )

    pm4py.write_bpmn(
        process_model,
        os.path.join(OUTPUT_DIR, "careflow_process.bpmn")
    )

    print("\nPM4Py process model generated!")

except ImportError:
    print("\nPM4Py not installed. Skipping process mining.")

except Exception as error:
    print("\nPM4Py process mining error:", error)

print("\n--------------------------------")
print("CAREFLOW ANALYSIS COMPLETED!")
print("Check the outputs folder.")
print("--------------------------------")
