#!/bin/bash

# Configuration
NAMESPACE="testing"
TEST_DURATION=1800
SLEEP_DURATION=1
NODE="ho3"
DEVICE="/dev/sdb1"
MOUNT_POINT="/mnt/testing"
TEST_FILE="$MOUNT_POINT/testfile"
TIMESTAMP=$(date '+%Y%m%d-%H%M%S')
LOG_FILE="/opt/testing/stress-test-diskIO-${TIMESTAMP}.log"

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S'),$1" | tee -a "$LOG_FILE"
}

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

# Ensure test file exists
log "Write test file at $TEST_FILE"
fio --name=writefile --filename=$TEST_FILE --size=5G --rw=write --bs=4k --numjobs=1 --direct=1 --iodepth=32 --ioengine=libaio --group_reporting
sync

sleep 600

log "Disk I/O Stress test script started on node $NODE."

# Retrieve the pods running on the selected node
log "Retrieving pod mapping on node $NODE..."
TESTING_POD=$(kubectl get pods -n $NAMESPACE -o=jsonpath="{.items[?(@.spec.nodeName=='$NODE')].metadata.name}" | tr ' ' '\n' | grep "testing-high-mem" | head -n 1)

# Ensure pod is found
if [[ -z "$TESTING_POD" ]]; then
    log "Error: Could not find the necessary pods on node $NODE. Exiting..."
    exit 1
fi

log "Selected pod $TESTING_POD on node $NODE for the main stress test."

# Measure max IOPS using fio
log "Measuring max IOPS using fio..."
MAX_IOPS_RAW=$(kubectl exec -n $NAMESPACE $TESTING_POD -- bash -c \
    "fio --name=randread --direct=1 --filename=$TEST_FILE --size=1G --ioengine=libaio \
    --rw=randread --bs=4k --numjobs=1 --iodepth=32 --time_based --group_reporting \
    --runtime=60s" | grep -oP 'IOPS=\K[\d.]+[kM]?')
# Convert shorthand notation to full numbers
if [[ "$MAX_IOPS_RAW" == *k ]]; then
    MAX_IOPS=$(echo "$MAX_IOPS_RAW" | sed 's/k//' | awk '{print $1 * 1000}')
elif [[ "$MAX_IOPS_RAW" == *M ]]; then
    MAX_IOPS=$(echo "$MAX_IOPS_RAW" | sed 's/M//' | awk '{print $1 * 1000000}')
else
    MAX_IOPS=$MAX_IOPS_RAW
fi

if [[ -z "$MAX_IOPS" ]]; then
    log "Failed to determine max IOPS. Exiting..."
    exit 1
fi

log "Max IOPS detected: $MAX_IOPS"

# Function to run fio with different diskIOloads
run_diskIO_stress_test() {
    local diskIO_load=$1
    local node_state=$2
    ROUNDED_IOPS=$(( (MAX_IOPS * diskIO_load) / 100 ))
    log "Starting fio at ${diskIO_load}% diskIO for $TEST_DURATION seconds on pod $TESTING_POD (node: $NODE),$node_state"
    start_time=$(date '+%s')
    kubectl exec -n $NAMESPACE $TESTING_POD -- bash -c "fio --name=randread --direct=1 --filename=$TEST_FILE --size=1G --ioengine=libaio --rw=randread --bs=4k --numjobs=1 --iodepth=32 --time_based --group_reporting --rate_iops=$ROUNDED_IOPS --runtime=${TEST_DURATION}"
    end_time=$(date '+%s')
    log "Completed fio at ${diskIO_load}% diskIO. Duration: $((end_time - start_time)) seconds,$node_state"
}

# Running tests with increasing diskIO loads on an idle node
log "Starting diskIO stress tests on an idle node"
for load in 10 30 50 70 90; do
    log "Waiting for $SLEEP_DURATION seconds before next stress test..."
    sleep "$SLEEP_DURATION"
    run_diskIO_stress_test "$load" "idle_node"
done
log "diskIO Stress test script finished on an idle node."
wait  # Ensure all stress commands finish

log "Cleaning up disk test files..."
kubectl exec -n $NAMESPACE $TESTING_POD -- bash -c "rm -rf $TEST_FILE"
log "Disk test files removed."
