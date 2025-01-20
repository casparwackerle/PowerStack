import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Define directories
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ANALYSIS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_analysis", "diagrams")
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# KEPLER metrics to analyze
METRICS = [
    "kepler_container_package_joules_total"
]

EXPERIMENT_TYPES = ["cpu", "mem"]

# Find the latest log file
def find_latest_log(experiment_type):
    log_files = sorted(
        [f for f in os.listdir(DATA_DIR) if f.startswith(f"stress-test-{experiment_type}") and f.endswith(".log")],
        key=lambda x: re.search(r"(\d{8}-\d{6})", x).group(1),
        reverse=True
    )
    return os.path.join(DATA_DIR, log_files[0]) if log_files else None

# Extract load phases and container under test from log file
def parse_log(log_file):
    test_phases = []
    container_under_test = None

    with open(log_file, "r") as f:
        for line in f:
            # Detect the container under test
            pod_match = re.search(r"Selected pod ([A-Za-z0-9]+(-[A-Za-z0-9]+)+) on node", line)
            if pod_match:
                container_under_test = pod_match.group(1)

            # Match test start times, load, and duration
            match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(Starting stress-ng at (\d+)% .+? for (\d+) seconds on pod .+?),(idle_cluster|busy_cluster)", line)
            if match:
                start_time = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
                load = int(match.group(3))
                duration = int(match.group(4))
                phase = match.group(5)

                test_phases.append({
                    "start": start_time,
                    "end": start_time + timedelta(seconds=duration),
                    "load": load,
                    "phase": phase
                })

    return test_phases, container_under_test

# Load KEPLER data from CSV and filter by container under test
def load_kepler_data(experiment_type, metric, container_under_test):
    file_path = os.path.join(DATA_DIR, experiment_type, metric, f"{metric}_full_experiment.csv")
    if not os.path.exists(file_path):
        print(f"Warning: Missing KEPLER data file: {file_path}")
        return None

    df = pd.read_csv(file_path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])  # Assume timestamps are consistent

    # Filter for only the container under test
    df = df[df["pod"] == container_under_test]

    # Convert Joules to Watts using time differences
    df["power_watts"] = df["value"].diff() / df["timestamp"].diff().dt.total_seconds()

    return df

# Generate a time-series plot
def generate_plot(experiment_type, metric, test_phases, kepler_data):
    plt.figure(figsize=(10, 6))

    # Adjust time to start from zero
    first_test_time = test_phases[0]["start"]
    
    # Plot the test load curve
    time_points = []
    load_points = []
    
    for phase in test_phases:
        time_points.append((phase["start"] - first_test_time).total_seconds())
        load_points.append(phase["load"])
        
        time_points.append((phase["end"] - first_test_time).total_seconds())
        load_points.append(phase["load"])

    plt.plot(time_points, load_points, color="black", linestyle="--", label="Applied Load (%)", linewidth=2)

    # Plot KEPLER energy data (converted to Watts)
    if kepler_data is not None:
        plt.plot(
            (kepler_data["timestamp"].dt.tz_localize(None) - first_test_time).dt.total_seconds(),
            kepler_data["power_watts"],
            label=f"{metric} (Watts)",
            color="blue"
        )

    # Labels and formatting
    plt.xlabel("Time (seconds)")
    plt.ylabel("Load (%)", color="black")
    plt.legend(loc="upper left")
    plt.title(f"{experiment_type.capitalize()} - {metric}")
    

    ax2 = plt.gca().twinx()
    ax2.set_ylabel("Power Consumption (Watts)", color="black")

    # Save figure
    output_dir = os.path.join(ANALYSIS_DIR, experiment_type, metric)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{experiment_type}_{metric}.png")
    
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved plot: {output_path}")

# Main function
def main():
    for experiment_type in EXPERIMENT_TYPES:
        log_file = find_latest_log(experiment_type)
        if not log_file:
            print(f"Warning: No log file found for {experiment_type}. Skipping...")
            continue

        print(f"Processing log file: {log_file}")
        test_phases, container_under_test = parse_log(log_file)

        if not container_under_test:
            print(f"Error: Could not determine the container under test for {experiment_type}. Skipping...")
            continue

        for metric in METRICS:
            kepler_data = load_kepler_data(experiment_type, metric, container_under_test)
            
            if kepler_data is None:
                print(f"Warning: No data available for {metric}. Skipping...")
                continue

            generate_plot(experiment_type, metric, test_phases, kepler_data)

if __name__ == "__main__":
    main()
