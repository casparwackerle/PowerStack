import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import re

# Define directories
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ANALYSIS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagrams")
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# Configuration for moving average
MOVING_AVERAGE_WINDOW = 10  # Configurable window size for smoothing

# Node-level KEPLER Metrics (Joule-based)
NODE_METRICS = {
    "kepler_node_dram_joules_total",
    "kepler_node_package_joules_total",
    "kepler_node_platform_joules_total",
    "kepler_node_other_joules_total",
}

# Experiment types and nodes
EXPERIMENT_TYPES = ["cpu", "mem", "diskIO", "netIO"]
NODES = ["ho1", "ho2", "ho3"]

# Find the latest log file
def find_latest_log(experiment_type):
    log_files = sorted(
        [f for f in os.listdir(DATA_DIR) if f.startswith(f"stress-test-{experiment_type}") and f.endswith(".log")],
        key=lambda x: re.search(r"(\d{8}-\d{6})", x).group(1),
        reverse=True
    )
    print(f"Found log files for {experiment_type}: {log_files}")
    return os.path.join(DATA_DIR, log_files[0]) if log_files else None

# Extract load phases from the log file
def parse_log(log_file):
    test_phases = []
    print(f"Parsing log file: {log_file}")
    with open(log_file, "r") as f:
        for line in f:
            print(f"Processing log line: {line.strip()}")
            match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(Starting (stress-ng|fio|iperf3) at (\d+)% .+? for (\d+) seconds on pod .+?),(idle_node|busy_node)", line)
            if match:
                start_time = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
                load = int(match.group(4))
                duration = int(match.group(5))
                phase = match.group(6)
                # print(f"Matched phase: start={start_time}, load={load}, duration={duration}, phase={phase}")
                test_phases.append({
                    "start": start_time,
                    "end": start_time + timedelta(seconds=duration),
                    "load": load,
                    "phase": phase
                })
    return test_phases

# Convert Joules to Watts
def convert_to_watts(df):
    df["value_converted"] = df["value"].diff() / df["timestamp"].diff().dt.total_seconds()
    df["value_converted"] = df["value_converted"].clip(lower=0)  # Ensure no negative values
    return df

# Apply moving average smoothing
def smooth_data(data, window=MOVING_AVERAGE_WINDOW):
    return data.rolling(window=window, min_periods=window, center=True).mean()

# Load KEPLER data from CSV and filter relevant node
def load_kepler_data(experiment_type, metric, node):
    file_path = os.path.join(DATA_DIR, experiment_type, metric, f"{metric}_full_experiment.csv")
    print(f"Loading KEPLER data from: {file_path}")
    if not os.path.exists(file_path):
        print(f"Warning: Missing KEPLER data file: {file_path}")
        return None

    df = pd.read_csv(file_path, parse_dates=["timestamp"])
    df.columns = ["timestamp", "value", "node", "mode", "source"]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # print(f"Loaded data preview:\n{df.head()}")
    
    # Filter for the relevant node
    df = df[df["node"] == node]
    
    # Separate dynamic and idle modes
    dynamic_df = df[df["mode"] == "dynamic"].copy()
    idle_df = df[df["mode"] == "idle"].copy()
    
    # Convert Joules to Watts
    dynamic_df = convert_to_watts(dynamic_df)
    idle_df = convert_to_watts(idle_df)
    
    return dynamic_df, idle_df

# Generate time-series plot
def generate_plot(experiment_type, metric, node, dynamic_data, idle_data, test_phases):
    print(f"Generating plot for {experiment_type} - {metric} - {node}")
    plt.figure(figsize=(12, 6))
    
    # Adjust time to start from zero
    first_test_time = test_phases[0]["start"]
    dynamic_data["time_seconds"] = (dynamic_data["timestamp"].dt.tz_localize(None) - first_test_time).dt.total_seconds()
    idle_data["time_seconds"] = (idle_data["timestamp"].dt.tz_localize(None) - first_test_time).dt.total_seconds()
    
    ax1 = plt.gca()
    
    # Plot dynamic power consumption
    ax1.plot(dynamic_data["time_seconds"], dynamic_data["value_converted"], label=f"{metric} (Dynamic, raw)", color="fuchsia", alpha=0.5)
    ax1.plot(idle_data["time_seconds"], idle_data["value_converted"], label=f"{metric} (Idle, raw)", color="green", alpha=0.5)
    
    # Smoothed data
    ax1.plot(dynamic_data["time_seconds"], smooth_data(dynamic_data["value_converted"]), label=f"{metric} (Dynamic, smoothed)", color="teal")
    ax1.plot(idle_data["time_seconds"], smooth_data(idle_data["value_converted"]), label=f"{metric} (Idle, smoothed)", color="orange")
    ax1.set_ylim(bottom=0)
    ax1.set_ylabel("Node-Level Power Consumption (Watts)", color="black")
    ax1.set_xlabel("Time (seconds)")
    ax1.legend(loc="upper left")

    # Compute transition time
    experiment_start = test_phases[0]["start"]
    experiment_end = test_phases[-1]["end"]
    transition_time_seconds = (experiment_start + (experiment_end - experiment_start) / 2 - first_test_time).total_seconds()

    # Add shading to indicate idle vs busy node
    if experiment_type in ["cpu", "mem"]:
        ax1.axvspan(0, transition_time_seconds, facecolor="lightblue", alpha=0.3, label="Idle Node")
        ax1.axvspan(transition_time_seconds, dynamic_data["time_seconds"].max(), facecolor="lightcoral", alpha=0.3, label="Busy Node")

    # Create secondary y-axis for workload profile
    ax2 = ax1.twinx()
    time_points, load_points = [], []
    
    for phase in test_phases:
        time_points.append((phase["start"] - first_test_time).total_seconds())
        load_points.append(phase["load"])

        time_points.append((phase["end"] - first_test_time).total_seconds())
        load_points.append(phase["load"])
    
    ax2.set_ylim(bottom=0)
    ax2.set_ylim(top=100)
    ax2.set_ylabel("Load (%)", color="black")
    ax2.plot(time_points, load_points, color="black", linestyle="--", label="Applied Load (%)", linewidth=2)
    ax2.legend(loc="upper right")

    plt.title(f"{experiment_type.upper()} Stress Test: metric ({metric}) (converted to Watts)")

    output_dir = os.path.join(ANALYSIS_DIR, experiment_type, metric)
    output_path = os.path.join(output_dir, f"{experiment_type}_{metric}_{node}.png")
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved plot for {experiment_type} - {metric} - {node}")

# Main function
def main():
    for experiment_type in EXPERIMENT_TYPES:
        log_file = find_latest_log(experiment_type)
        test_phases = parse_log(log_file) if log_file else []
        for metric in NODE_METRICS:
            for node in NODES:
                dynamic_data, idle_data = load_kepler_data(experiment_type, metric, node)
                if dynamic_data is None or idle_data is None:
                    continue
                generate_plot(experiment_type, metric, node, dynamic_data, idle_data, test_phases)

if __name__ == "__main__":
    main()
