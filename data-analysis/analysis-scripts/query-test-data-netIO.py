import os
import re
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta

PROMETHEUS_URL = "http://160.85.30.104:30002/api/v1/query_range"
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
MAX_POINTS = 11000
SCRAPING_INTERVAL = 10      # set to 0 for automatic granularity (use if Prometheus scraping interval is unknown)

# Locate Latest Log File
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def find_latest_log():
    log_files = sorted(
        [f for f in os.listdir(DATA_DIR) if f.startswith("stress-test-netIO") and f.endswith(".log")],
        key=lambda x: re.search(r"(\d{8}-\d{6})", x).group(1),
        reverse=True
    )
    return os.path.join(DATA_DIR, log_files[0]) if log_files else None

# Extract Start/End Times from Log
def parse_log(log_file):
    test_phases = {}
    with open(log_file, "r") as f:
        for line in f:
            match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(Starting iperf3 at (\d+)% netIO for (\d+) seconds on pod .+?),(idle_node|busy_node)", line)
            if match:
                start_time = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                netIO_load = match.group(3) + "%"
                duration = int(match.group(4))
                end_time = start_time + timedelta(seconds=duration)
                end_time_str = end_time.isoformat(timespec='seconds') + "Z"
                start_time_str = start_time.isoformat(timespec='seconds') + "Z"
                phase = match.group(5)  # Extracts 'idle_node' or 'busy_node'
                test_phases[f"{netIO_load}_{phase}"] = {
                    "start": start_time_str,
                    "end": end_time_str
                }
    return test_phases

# Query Prometheus
def query_prometheus(metric, start, end):
    start_unix = int(datetime.fromisoformat(start.replace("Z", "")).timestamp())
    end_unix = int(datetime.fromisoformat(end.replace("Z", "")).timestamp())
    step = SCRAPING_INTERVAL
    min_step = max(1, (end_unix - start_unix) // MAX_POINTS)
    if SCRAPING_INTERVAL < min_step:
        step = max(1, (end_unix - start_unix) // MAX_POINTS)

    params = {
        "query": f"sum by (pod_name, container_name, source) ({metric}{{container_name=~'testing-high-mem'}})",
        "start": start_unix,
        "end": end_unix,
        "step": f"{step}s"
    }
    response = requests.get(PROMETHEUS_URL, params=params).json()
    return response.get("data", {}).get("result", [])

# Save Data to CSV
def save_to_csv(data, metric, phase):
    extracted_data = []
    for d in data:
        for v in d["values"]:
            timestamp_utc = datetime.utcfromtimestamp(float(v[0])).isoformat() + "Z"
            extracted_data.append({
                "timestamp": timestamp_utc,
                "value": round(float(v[1]), 3),
                "pod": d["metric"].get("pod_name", ""),
                "container_name": d["metric"].get("container_name", ""),
                "source": d["metric"].get("source", "")
            })
    
    # Define directory structure
    experiment_type = "netIO"  # This will be dynamic in future for other tests
    metric_dir = os.path.join(DATA_DIR, experiment_type, metric)
    os.makedirs(metric_dir, exist_ok=True)
    
    csv_file = os.path.join(metric_dir, f"{metric}_{phase}.csv")
    df = pd.DataFrame(extracted_data)
    df.to_csv(csv_file, index=False)
    print(f"Saved {csv_file}")

# Main Function
def main():
    latest_log = find_latest_log()
    if not latest_log:
        print("No log file found.")
        return

    print(f"Found latest log file: {latest_log}")
    test_phases = parse_log(latest_log)

    # Determine the overall experiment start and end time
    overall_start = min(times["start"] for times in test_phases.values())
    overall_end = max(times["end"] for times in test_phases.values())

    for metric in METRICS:
        # Query and save data for each test phase (individual load levels)
        for phase, times in test_phases.items():
            print(f"Querying {metric} for {phase} (from {times['start']} to {times['end']})")
            data = query_prometheus(metric, times["start"], times["end"])
            save_to_csv(data, metric, phase)

        # Query and save aggregated data for the full experiment duration
        print(f"Querying {metric} for full experiment duration (from {overall_start} to {overall_end})")
        full_data = query_prometheus(metric, overall_start, overall_end)
        save_to_csv(full_data, metric, "full_experiment")

if __name__ == "__main__":
    main()
