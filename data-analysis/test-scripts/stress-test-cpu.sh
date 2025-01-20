#!/bin/bash

# Configuration
NAMESPACE="testing"
TEST_DURATION=1200
SLEEP_DURATION=120
TOTAL_vCPU=4
TOTAL_DRAM=1
TIMESTAMP=$(date '+%Y%m%d-%H%M%S')
LOG_FILE="/opt/testing/stress-test-cpu-${TOTAL_vCPU}vCPU-${TOTAL_DRAM}Gi-${TIMESTAMP}.log"

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S'),$1" | tee -a "$LOG_FILE"
}

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

log "CPU Stress test script started."

# Get pod names and associate them with nodes (only select 'testing-high-cpu' pods)
log "Retrieving high-CPU pod-to-node mapping..."
declare -A pod_map
while read pod node; do
    pod_map[$node]=$pod
done < <(kubectl get pods -n $NAMESPACE --no-headers -o=custom-columns=":metadata.name,:spec.nodeName" | grep "testing-high-cpu")

# Ensure at least one high-CPU pod was found
if [[ ${#pod_map[@]} -eq 0 ]]; then
    log "Error: No high-CPU pods found in namespace $NAMESPACE. Exiting..."
    exit 1
fi

# Select the pod to stress (pick from first node)
TARGET_NODE=$(echo "${!pod_map[@]}" | awk '{print $1}')  # First node found
TARGET_POD="${pod_map[$TARGET_NODE]}"
log "Selected pod $TARGET_POD on node $TARGET_NODE for initial stress."

# Select the other two nodes
OTHER_NODES=()
for node in "${!pod_map[@]}"; do
    if [[ "$node" != "$TARGET_NODE" ]]; then
        OTHER_NODES+=("$node")
    fi
done

# Function to run stress-ng with different CPU loads
run_cpu_stress_test() {
    local cpu_load=$1
    local cluster_state=$2
    log "Starting stress-ng at ${cpu_load}% CPU for $TEST_DURATION seconds on pod $TARGET_POD (node: $TARGET_NODE),$cluster_state"
    start_time=$(date '+%s')
    kubectl exec -n $NAMESPACE $TARGET_POD -- bash -c "stress-ng --cpu $TOTAL_vCPU --cpu-load $cpu_load --timeout ${TEST_DURATION}s"
    end_time=$(date '+%s')
    log "Completed stress-ng at ${cpu_load}% CPU. Duration: $((end_time - start_time)) seconds,$cluster_state"
}

# Running tests with increasing CPU loads on an idle cluster
log "Starting CPU stress tests on an idle cluster"
for load in 10 30 50 70 90; do
    log "Waiting for $SLEEP_DURATION seconds before next stress test..."
    sleep "$SLEEP_DURATION"
    run_cpu_stress_test "$load" "idle_cluster"
done
log "CPU Stress test script finished on an idle cluster."

# Apply background stress on the other nodes
for node in "${OTHER_NODES[@]}"; do
    pod=${pod_map[$node]}
    log "Applying stress on pod $pod running on node $node with adjusted load."
    kubectl exec -n $NAMESPACE $pod -- bash -c "stress-ng --cpu $TOTAL_vCPU --cpu-load 90 --timeout $((6 * (TEST_DURATION + SLEEP_DURATION)))s" &
done

# Running tests with increasing CPU loads on a busy cluster
log "Starting CPU stress tests on a busy cluster"
for load in 10 30 50 70 90; do
    log "Waiting for $SLEEP_DURATION seconds before next stress test..."
    sleep "$SLEEP_DURATION"
    run_cpu_stress_test "$load" "busy_cluster"
done
wait  # Ensure all stress commands finish
log "CPU Stress test script finished on a busy cluster."

# Cleanup: Stop all stress-ng processes on all nodes
log "Cleaning up stress-ng processes to ensure no workload is running..."
for node in "${!pod_map[@]}"; do
    pod=${pod_map[$node]}
    log "Stopping stress-ng on pod $pod (node: $node)"
    kubectl exec -n $NAMESPACE $pod -- bash -c "pkill -f stress-ng" &
done
wait
log "All stress-ng processes stopped."
