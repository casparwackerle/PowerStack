#!/bin/bash

# Configuration
NAMESPACE="testing"
TEST_DURATION=1800
SLEEP_DURATION=1
CLIENT_NODE="ho3"
SERVER_NODE="ho2"
TIMESTAMP=$(date '+%Y%m%d-%H%M%S')
LOG_FILE="/opt/testing/stress-test-netIO-${TIMESTAMP}.log"

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S'),$1" | tee -a "$LOG_FILE"
}

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

log "Network Stress test script started. Client: $CLIENT_NODE, Server: $SERVER_NODE."

# Retrieve the pods running on the selected nodes
log "Retrieving pod mapping on client node $CLIENT_NODE and server node $SERVER_NODE..."
SERVER_POD=$(kubectl get pods -n $NAMESPACE -o=jsonpath="{.items[?(@.spec.nodeName=='$SERVER_NODE')].metadata.name}" | tr ' ' '\n' | grep "testing-high-mem" | head -n 1)
CLIENT_POD=$(kubectl get pods -n $NAMESPACE -o=jsonpath="{.items[?(@.spec.nodeName=='$CLIENT_NODE')].metadata.name}" | tr ' ' '\n' | grep "testing-high-mem" | head -n 1)
SERVER_POD_IP=$(kubectl get pod -n $NAMESPACE $SERVER_POD -o jsonpath='{.status.podIP}')
CLIENT_POD_IP=$(kubectl get pod -n $NAMESPACE $CLIENT_POD -o jsonpath='{.status.podIP}')

# Ensure both pods were found
if [[ -z "$SERVER_POD" || -z "$CLIENT_POD" ]]; then
    log "Error: Could not find the necessary pods on both nodes. Exiting..."
    exit 1
fi

log "Selected server pod: $SERVER_POD on node $SERVER_NODE with IP $SERVER_POD_IP"
log "Selected client pod: $CLIENT_POD on node $CLIENT_NODE with IP $CLIENT_POD_IP"
log "Selected pod $CLIENT_POD on node $CLIENT_NODE for the main stress test."       # Leave this line, it is important for data processing

# Start iperf3 server
log "Starting iperf3 server on pod $SERVER_POD..."
kubectl exec -n $NAMESPACE $SERVER_POD -- bash -c "nohup iperf3 -s > /dev/null 2>&1 &"
sleep 2  # Give the server some time to start

# Measure max network throughput
log "Measuring maximum network throughput..."
MAX_BW=$(kubectl exec -n $NAMESPACE $CLIENT_POD -- bash -c "iperf3 -c $SERVER_POD_IP -t 30 -f m" | grep -o '[0-9.]\+ Mbits/sec' | awk '{print $1}' | sort -nr | head -n 1)

if [[ -z "$MAX_BW" ]]; then
    log "Failed to determine max bandwidth. Exiting..."
    exit 1
fi

log "Max bandwidth detected: $MAX_BW Mbps"

sleep 600

# Function to run iperf3 with different loads
run_network_stress_test() {
    local net_load=$1
    local node_state=$2
    ROUNDED_BW=$(( (MAX_BW * net_load) / 100 ))
    log "Starting iperf3 at ${net_load}% netIO for $TEST_DURATION seconds on pod $CLIENT_POD (node: $CLIENT_NODE),$node_state"
    start_time=$(date '+%s')
    kubectl exec -n $NAMESPACE $CLIENT_POD -- bash -c "iperf3 -c $SERVER_POD_IP -t $TEST_DURATION -b ${ROUNDED_BW}M"
    end_time=$(date '+%s')
    log "Completed iperf3 at ${net_load}% netIO. Duration: $((end_time - start_time)) seconds,$node_state"
}

# Running tests with increasing network loads on an idle node
log "Starting network stress tests on an idle node"
for load in 10 30 50 70 90; do
    log "Waiting for $SLEEP_DURATION seconds before next stress test..."
    sleep "$SLEEP_DURATION"
    run_network_stress_test "$load" "idle_node"
done
log "Network stress test script finished on an idle node."
wait  # Ensure all stress commands finish

# Cleanup: Stop iperf3 server
log "Stopping iperf3 server on pod $SERVER_POD..."
kubectl exec -n $NAMESPACE $SERVER_POD -- pkill iperf3
log "iperf3 server stopped."

log "Network Stress Test completed successfully."
