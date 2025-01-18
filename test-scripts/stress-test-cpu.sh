#!/bin/bash

# Configuration
POD_NAME="testing"
NAMESPACE="testing"
TEST_DURACTION=1200
SLEEP_DURACTION=120
TOTAL_vCPU=4
TOTAL_DRAM=1
LOG_FILE="/opt/testing/stress-test-cpu-${TOTAL_vCPU}vCPU-${TOTAL_DRAM}Gi.log"

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a "$LOG_FILE"
}

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

log "CPU Stress test script started."

# Function to run stress-ng with different CPU loads
run_cpu_stress_test() {
    local cpu_load=$1
    log "Starting stress-ng at ${cpu_load}% CPU for $TEST_DURACTION seconds..."
    kubectl exec -n "$NAMESPACE" "$POD_NAME" -- bash -c "stress-ng --cpu $TOTAL_vCPU --cpu-load $cpu_load --timeout ${TEST_DURACTION}s"
    log "Completed stress-ng at ${cpu_load}% CPU."
}

# Running tests with increasing CPU loads
log "Starting CPU stress tests"
for load in 10 30 50 70 90; do
    log "Waiting for $SLEEP_DURACTION seconds before next stress test..."
    sleep "$SLEEP_DURACTION"
    run_cpu_stress_test "$load"
done
log "CPU Stress test script finished."

