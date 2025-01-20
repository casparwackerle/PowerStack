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
    # Kepler Metrics for Container Energy Consumption
    "kepler_container_joules_total",
    "kepler_container_core_joules_total",
    "kepler_container_dram_joules_total",
    "kepler_container_package_joules_total",
    "kepler_container_other_joules_total",
    
    # Kepler Metrics for Container Resource Utilization
    "kepler_container_cpu_cycles_total",
    "kepler_container_cpu_instructions_total",
    "kepler_container_cache_miss_total",
    "kepler_container_bpf_net_tx_irq_total",
    "kepler_container_bpf_net_rx_irq_total",
    "kepler_container_bpf_block_irq_total",
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

# Extract load phases from the log file
def parse_log(log_file):
    test_phases = []
    test_pod = None  # Store the specific test pod being stressed

    with open(log_file, "r") as f:
        for line in f:
            # Detect the container under test
            pod_match = re.search(r"Selected pod ([A-Za-z0-9]+(-[A-Za-z0-9]+)+) on node", line)
            if pod_match:
                test_pod = pod_match.group(1)

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

    return test_phases, test_pod

# Load KEPLER data from CSV and filter to relevant pod
def load_kepler_data(experiment_type, metric, test_pod):
    file_path = os.path.join(DATA_DIR, experiment_type, metric, f"{metric}_full_experiment.csv")
    if not os.path.exists(file_path):
        print(f"Warning: Missing KEPLER data file: {file_path}")
        return None

    df = pd.read_csv(file_path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Filter only the pod under test
    df = df[df["pod"] == test_pod]

    # Convert Joules to Watts (Δ Joules / Δ Time)
    df["power_watts"] = df["value"].diff() / df["timestamp"].diff().dt.total_seconds()

    return df

# Generate a time-series plot
def generate_plot(experiment_type, metric, test_phases, kepler_data):
    plt.figure(figsize=(10, 6))

    # Adjust time to start from zero
    first_test_time = test_phases[0]["start"]

    # Create the primary axis (KEPLER energy data)
    ax1 = plt.gca()

    # Plot KEPLER energy data (Watts) on the PRIMARY Y-Axis
    ax1.plot(
        (kepler_data["timestamp"].dt.tz_localize(None) - first_test_time).dt.total_seconds(),
        kepler_data["power_watts"],
        label=f"{metric} (Watts)",
        color="blue"
    )

    # **Primary Y-Axis Label (KEPLER Power Consumption)**
    ax1.set_ylabel("Power Consumption (Watts)", color="blue")

    # Create secondary y-axis for the workload
    ax2 = ax1.twinx()

    # Plot the workload profile
    time_points = []
    load_points = []

    for phase in test_phases:
        time_points.append((phase["start"] - first_test_time).total_seconds())
        load_points.append(phase["load"])

        time_points.append((phase["end"] - first_test_time).total_seconds())
        load_points.append(phase["load"])

    ax2.plot(time_points, load_points, color="black", linestyle="--", label="Applied Load (%)", linewidth=2)


    # Determine transition time automatically
    experiment_start = test_phases[0]["start"]
    experiment_end = test_phases[-1]["end"]
    transition_time = experiment_start + (experiment_end - experiment_start) / 2
    transition_time_seconds = (transition_time - first_test_time).total_seconds()

    # Add Background Shading for Idle and Busy Cluster Phases
    ax1.axvspan(0, transition_time_seconds, facecolor="lightblue", alpha=0.3, label="Idle Cluster")
    ax1.axvspan(transition_time_seconds, (experiment_end - first_test_time).total_seconds(), 
                facecolor="lightcoral", alpha=0.3, label="Busy Cluster")

    # Labels and formatting
    ax1.set_xlabel("Time (seconds)")
    
    ax2.set_ylabel("Load (%)", color="black")

    # Legends
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.title(f"{experiment_type.capitalize()} - {metric}")

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
        test_phases, test_pod = parse_log(log_file)

        for metric in METRICS:
            kepler_data = load_kepler_data(experiment_type, metric, test_pod)
            
            if kepler_data is None or kepler_data.empty:
                print(f"Warning: No data available for {metric}. Skipping...")
                continue

            generate_plot(experiment_type, metric, test_phases, kepler_data)

if __name__ == "__main__":
    main()
