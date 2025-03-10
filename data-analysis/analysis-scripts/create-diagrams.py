import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Define directories
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ANALYSIS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagrams")
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# Configuration for moving average
MOVING_AVERAGE_WINDOW = 10  # Configurable window size for smoothing

# Joule-based KEPLER Metrics (to be converted to Watts)
JOULE_METRICS = {
    "kepler_container_joules_total",
    # "kepler_container_core_joules_total",         # Not supported by CPU
    "kepler_container_dram_joules_total",
    # "kepler_container_uncore_joules_total"        # Not supported by hardware
    "kepler_container_package_joules_total",
    "kepler_container_other_joules_total",
    # "kepler_container_gpu_joules_total"           # deactivated due to gpu absence
}

# KEPLER metrics to analyze (includes both Joules-based and raw metrics)
METRICS = list(JOULE_METRICS) + [
    # "kepler_container_bpf_cpu_time_us_total",      # not working
    "kepler_container_cpu_cycles_total",
    "kepler_container_cpu_instructions_total",
    "kepler_container_cache_miss_total",
    "kepler_container_bpf_net_tx_irq_total",
    "kepler_container_bpf_net_rx_irq_total",
    "kepler_container_bpf_block_irq_total",
]

NODE_METRICS = {
    # "kepler_node_core_joules_total",               # not working
    "kepler_node_dram_joules_total",
    # "kepler_node_uncore_joules_total",             # not supported by hardware
    "kepler_node_package_joules_total",
    "kepler_node_other_joules_total",
    # "kepler_node_gpu_joules_total",                # deactivated due to hardware absence
    "kepler_node_platform_joules_total",
}

EXPERIMENT_TYPES = ["cpu", "mem", "diskIO", "netIO"]

# Find the latest log file
def find_latest_log(experiment_type):
    log_files = sorted(
        [f for f in os.listdir(DATA_DIR) if f.startswith(f"stress-test-{experiment_type}") and f.endswith(".log")],
        key=lambda x: re.search(r"(\d{8}-\d{6})", x).group(1),
        reverse=True
    )
    return os.path.join(DATA_DIR, log_files[0]) if log_files else None

# Extract load phases and test pod from the log file
def parse_log(log_file):
    test_phases = []
    test_pod = None

    with open(log_file, "r") as f:
        for line in f:
            # Identify the pod under test
            pod_match = re.search(r"Selected pod ([A-Za-z0-9]+(-[A-Za-z0-9]+)+) on node \S+ for the main stress test\.", line)
            
            if pod_match:
                test_pod = pod_match.group(1)

            # Extract stress test details
            match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(Starting (stress-ng|fio|iperf3) at (\d+)% .+? for (\d+) seconds on pod .+?),(idle_node|busy_node)", line)
            if match:
                start_time = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
                load = int(match.group(4))
                duration = int(match.group(5))
                phase = match.group(6)

                test_phases.append({
                    "start": start_time,
                    "end": start_time + timedelta(seconds=duration),
                    "load": load,
                    "phase": phase
                })

    return test_phases, test_pod

# Load KEPLER data from CSV and filter relevant pod
def load_kepler_data(experiment_type, metric, test_pod):
    file_path = os.path.join(DATA_DIR, experiment_type, metric, f"{metric}_full_experiment.csv")
    if not os.path.exists(file_path):
        print(f"Warning: Missing KEPLER data file: {file_path}")
        return None

    df = pd.read_csv(file_path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Filter for the relevant test pod
    df = df[df["pod"] == test_pod]

    # Convert Joules to Watts or calculate operations per second
    if metric in JOULE_METRICS:
        df["value_converted"] = df["value"].diff() / df["timestamp"].diff().dt.total_seconds()
    else:
        df["value_converted"] = df["value"].diff() / df["timestamp"].diff().dt.total_seconds()  # Convert to operations per second
    df["value_converted"] = df["value_converted"].clip(lower=0)  # Ensure no negative values
    return df

# Apply moving average smoothing
def smooth_data(data, window=MOVING_AVERAGE_WINDOW):
    return data.rolling(window=window, min_periods=window, center=True).mean()

# Generate time-series plot
def generate_plot(experiment_type, metric, test_phases, kepler_data, is_joule_based, show_smoothed):
    plt.figure(figsize=(12, 6))
    first_test_time = test_phases[0]["start"]
    kepler_data["time_seconds"] = (kepler_data["timestamp"].dt.tz_localize(None) - first_test_time).dt.total_seconds()
    # Create primary axis (KEPLER data)
    ax1 = plt.gca()
    ax1.plot(
        kepler_data["time_seconds"],
        kepler_data["value_converted"],
        label=f"{metric} (raw)",
        color="blue",
        alpha=0.5
    )
        
    if show_smoothed:
        kepler_data["smoothed_value"] = smooth_data(kepler_data["value_converted"])
        ax1.plot(
            kepler_data["time_seconds"],
            kepler_data["smoothed_value"],
            label=f"{metric} (smoothed)",
            color="red"
        )

    
    ax1.set_ylim(bottom=0)
    ax1.set_ylabel("Container-Level Power Consumption (Watts)" if is_joule_based else "Container-Level Operations per Second", color="black")
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
    ax2.plot(time_points, load_points, color="black", linestyle="--", label="Applied Load (%)", linewidth=2)

    # Compute transition time
    experiment_start = test_phases[0]["start"]
    experiment_end = test_phases[-1]["end"]
    transition_time_seconds = (experiment_start + (experiment_end - experiment_start) / 2 - first_test_time).total_seconds()

    # Add shading to indicate idle vs busy node
    if experiment_type in ["cpu", "mem"]:
        ax1.axvspan(0, transition_time_seconds, facecolor="lightblue", alpha=0.3, label="Idle Node")
        ax1.axvspan(transition_time_seconds, kepler_data["time_seconds"].max(), facecolor="lightcoral", alpha=0.3, label="Busy Node")

    # Labels and formatting
    ax1.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Load (%)", color="black")

    # Legends
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.title(f"{experiment_type.upper()} Stress Test: metric ({metric}) (converted to Watts)" if is_joule_based else f"{experiment_type.upper()} Stress Test: metric ({metric})")

    # Save figure
    output_dir = os.path.join(ANALYSIS_DIR, experiment_type, metric)
    os.makedirs(output_dir, exist_ok=True)
    if show_smoothed:
        output_path = os.path.join(output_dir, f"{experiment_type}_{metric}_smoothed.png")
    else:
        output_path = os.path.join(output_dir, f"{experiment_type}_{metric}_raw.png")

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
            print(f"Processing metric: {metric}")

            # Load the relevant KEPLER data
            kepler_data = load_kepler_data(experiment_type, metric, test_pod)

            if kepler_data is None or kepler_data.empty:
                print(f"Warning: No data available for {metric}. Skipping...")
                continue

            # Determine if the metric is a Joule-based metric
            is_joule_based = metric in JOULE_METRICS

            # Save raw-only diagram
            generate_plot(experiment_type, metric, test_phases, kepler_data, is_joule_based, show_smoothed=False)

            # Save raw + smoothed diagram
            generate_plot(experiment_type, metric, test_phases, kepler_data, is_joule_based, show_smoothed=True)


if __name__ == "__main__":
    main()