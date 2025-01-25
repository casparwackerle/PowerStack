#!/bin/bash

# Configuration
NAMESPACE="testing"
TEST_DURATION=1800
SLEEP_DURATION=1
TOTAL_vCPU=150m
TOTAL_DRAM=25Gi
NODE="ho3"  # Ensure tests run on the same node
TIMESTAMP=$(date '+%Y%m%d-%H%M%S')
LOG_FILE="/opt/testing/stress-test-mem-${TOTAL_vCPU}vCPU-${TOTAL_DRAM}-${TIMESTAMP}.log"

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S'),$1" | tee -a "$LOG_FILE"
}

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

log "CPU Stress test script started on node $NODE."

# Retrieve the pods running on the selected node
log "Retrieving pod mapping on node $NODE..."
TESTING_POD=$(kubectl get pods -n $NAMESPACE -o=jsonpath="{.items[?(@.spec.nodeName=='$NODE')].metadata.name}" | tr ' ' '\n' | grep "testing-high-mem" | head -n 1)
LOAD_POD=$(kubectl get pods -n $NAMESPACE -o=jsonpath="{.items[?(@.spec.nodeName=='$NODE')].metadata.name}" | tr ' ' '\n' | grep "load-high-mem" | head -n 1)


# Ensure both pods were found
if [[ -z "$TESTING_POD" ]] || [[ -z "$LOAD_POD" ]]; then
    log "Error: Could not find the necessary pods on node $NODE. Exiting..."
    exit 1
fi

log "Selected pod $TESTING_POD on node $NODE for the main stress test."
log "Selected pod $LOAD_POD on node $NODE for background load."

# Function to run stress-ng with different Mem loads
run_mem_stress_test() {
    local mem_load=$1
    local cluster_state=$2
    log "Starting stress-ng at ${mem_load}% Mem for $TEST_DURATION seconds on pod $TESTING_POD (node: $NODE),$cluster_state"
    start_time=$(date '+%s')
    kubectl exec -n $NAMESPACE $TESTING_POD -- bash -c "stress-ng --vm 1 --vm-bytes $(((TOTAL_DRAM * mem_load) / 100))G --vm-keep --timeout ${TEST_DURATION}s"
    end_time=$(date '+%s')
    log "Completed stress-ng at ${mem_load}% Mem. Duration: $((end_time - start_time)) seconds,$cluster_state"
}

# Running tests with increasing Mem loads on an idle cluster
log "Starting Mem stress tests on an idle cluster"
for load in 10 30 50 70 90; do
    log "Waiting for $SLEEP_DURATION seconds before next stress test..."
    sleep "$SLEEP_DURATION"
    run_mem_stress_test "$load" "idle_cluster"
done
log "Mem Stress test script finished on an idle cluster."

# Apply background stress on the SAME NODE (on `load-high-mem`)
log "Applying stress on pod $LOAD_POD running on node $NODE with adjusted load."
kubectl exec -n $NAMESPACE $LOAD_POD -- bash -c "stress-ng --vm 1 --vm-bytes $(((TOTAL_DRAM * 9) / 10))G --vm-keep --timeout $((6 * (TEST_DURATION + SLEEP_DURATION)))s" &

# Running tests with increasing Mem loads on a busy cluster
log "Starting Mem stress tests on a busy cluster"
for load in 10 30 50 70 90; do
    log "Waiting for $SLEEP_DURATION seconds before next stress test..."
    sleep "$SLEEP_DURATION"
    run_mem_stress_test "$load" "busy_cluster"
done
wait  # Ensure all stress commands finish
log "Mem Stress test script finished on a busy cluster."

# Cleanup: Stop all stress-ng processes on the test node
log "Cleaning up stress-ng processes on node $NODE..."
kubectl exec -n $NAMESPACE $TESTING_POD -- bash -c "pkill -f stress-ng" &
kubectl exec -n $NAMESPACE $LOAD_POD -- bash -c "pkill -f stress-ng" &
log "All stress-ng processes stopped on $NODE."
