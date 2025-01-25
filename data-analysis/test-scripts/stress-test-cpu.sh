#!/bin/bash

# Configuration
NAMESPACE="testing"
TEST_DURATION=1800
SLEEP_DURATION=1
TOTAL_vCPU=2.5
TOTAL_DRAM=1Gi
NODE="ho3"  # Ensure tests run on the same node
TIMESTAMP=$(date '+%Y%m%d-%H%M%S')
LOG_FILE="/opt/testing/stress-test-cpu-${TOTAL_vCPU}vCPU-${TOTAL_DRAM}-${TIMESTAMP}.log"
ROUNDED_VCPU=$(echo "$TOTAL_vCPU" | awk '{print ($1 == int($1) ? $1 : int($1) + 1)}')

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S'),$1" | tee -a "$LOG_FILE"
}

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

log "CPU Stress test script started on node $NODE."

# Retrieve the pods running on the selected node
log "Retrieving pod mapping on node $NODE..."
TESTING_POD=$(kubectl get pods -n $NAMESPACE -o=jsonpath="{.items[?(@.spec.nodeName=='$NODE')].metadata.name}" | tr ' ' '\n' | grep "testing-high-cpu" | head -n 1)
LOAD_POD=$(kubectl get pods -n $NAMESPACE -o=jsonpath="{.items[?(@.spec.nodeName=='$NODE')].metadata.name}" | tr ' ' '\n' | grep "load-high-cpu" | head -n 1)


# Ensure both pods were found
if [[ -z "$TESTING_POD" ]] || [[ -z "$LOAD_POD" ]]; then
    log "Error: Could not find the necessary pods on node $NODE. Exiting..."
    exit 1
fi

log "Selected pod $TESTING_POD on node $NODE for the main stress test."
log "Selected pod $LOAD_POD on node $NODE for background load."

# Function to run stress-ng with different CPU loads
run_cpu_stress_test() {
    local cpu_load=$1
    local cluster_state=$2
    log "Starting stress-ng at ${cpu_load}% CPU for $TEST_DURATION seconds on pod $TESTING_POD (node: $NODE),$cluster_state"
    start_time=$(date '+%s')
    kubectl exec -n $NAMESPACE $TESTING_POD -- bash -c "stress-ng --cpu $ROUNDED_VCPU --cpu-load $cpu_load --timeout ${TEST_DURATION}s"
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

# Apply background stress on the SAME NODE (on `load-high-mem`)
log "Applying stress on pod $LOAD_POD running on node $NODE with adjusted load."
kubectl exec -n $NAMESPACE $LOAD_POD -- bash -c "stress-ng --cpu $ROUNDED_VCPU --cpu-load 90 --timeout $((6 * (TEST_DURATION + SLEEP_DURATION)))s" &

# Running tests with increasing CPU loads on a busy cluster
log "Starting CPU stress tests on a busy cluster"
for load in 10 30 50 70 90; do
    log "Waiting for $SLEEP_DURATION seconds before next stress test..."
    sleep "$SLEEP_DURATION"
    run_cpu_stress_test "$load" "busy_cluster"
    done
wait  # Ensure all stress commands finish
log "CPU Stress test script finished on a busy cluster."

# Cleanup: Stop all stress-ng processes on the test node
log "Cleaning up stress-ng processes on node $NODE..."
kubectl exec -n $NAMESPACE $TESTING_POD -- bash -c "pkill -f stress-ng" &
kubectl exec -n $NAMESPACE $LOAD_POD -- bash -c "pkill -f stress-ng" &
log "All stress-ng processes stopped on $NODE."
