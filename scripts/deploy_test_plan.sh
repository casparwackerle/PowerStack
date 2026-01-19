#!/bin/bash

export ANSIBLE_FORCE_COLOR=true

# Enable strict error handling
set -e  # Exit immediately if a command exits with a non-zero status
set -u  # Treat unset variables as an error
set -o pipefail  # Return the exit status of the last command in a pipeline that failed

# Store the original working directory
ORIGINAL_DIR=$(pwd)

# Detect the repository path as one level above the current script's directory
REPO_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# Ensure REPO_PATH is set and valid
if [ -z "$REPO_PATH" ] || [ ! -d "$REPO_PATH" ]; then
    echo "Error: Unable to determine the repository path. Please check your setup."
    exit 1
fi

# Ensure the logs directory exists
mkdir -p "$REPO_PATH/logs"

# Add a timestamp to the log file name
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="$REPO_PATH/logs/testing_start_$TIMESTAMP.log"

# Redirect script output to a timestamped log file
exec > >(tee -i "$LOG_FILE") 2>&1

echo "Log file created: $LOG_FILE"

# -------------------------
# Args / Plan selection
# -------------------------

# Optional flags
RESET=false
if [ "${1:-}" = "--reset" ]; then
  RESET=true
fi

# -------------------------
# Available test plans
# (activate exactly ONE)
# -------------------------

# PLAN_ID="cpu-short-lived-v1"
# PLAN_ID="cpu-burst-train-v1"
# PLAN_ID="cpu-warmup-ramp-test-v1"
# PLAN_ID="cpu-jitter-train-v1"
# PLAN_ID="cpu-burst-baseline-v1" # CPU burst train with baseline stress (1 core) and bursts (10 cores)
# PLAN_ID="cpu-burst-hard-off-v1." # off_workers=0 means no stress in off-phase (sleep only), while keeping off_mode=stress to prove the branch logic is correct
# PLAN_ID="cpu-slice4-acceptance-v1"  # This is the “single run validates everything CPU” plan.
# PLAN_ID="cpu-noop-v1"
# PLAN_ID="cpu-stress-workload-set"
# PLAN_ID="gpu-burn-steady-v1"
# PLAN_ID="gpu-burst-train-3s-1-v1"
# PLAN_ID="gpu-burn-steady-set"
# PLAN_ID="all-tests-v1"
# PLAN_ID="tycho-testing"
PLAN_ID="tycho-testing-short"


# -------------------------
# Safety check
# -------------------------
if [ -z "$PLAN_ID" ]; then
  echo "Error: PLAN_ID is empty. Activate exactly one plan."
  exit 1
fi

echo "Starting Tycho testing with plan: $PLAN_ID"
echo "Reset enabled: $RESET"

# -------------------------
# Step 0: NFS prep (real cluster inventory)
# -------------------------
echo
echo "Step 0: Preparing NFS directory (/mnt/data/tycho-testing) on control-plane (NFS server)..."

cd "$REPO_PATH/ansible/testing-ansible" || { echo "Error: 'ansible/testing-ansible' directory not found in $REPO_PATH"; exit 1; }

ansible-playbook playbooks/nfs-prep.yml \
  -i ../../configs/inventory.yml \
  --ask-vault-pass

echo "NFS prep complete."

# -------------------------
# Step 0.5: Optional reset (kubectl-only, localhost inventory)
# -------------------------
if [ "$RESET" = true ]; then
  echo
  echo "Step 0.5: Resetting Tycho testing PV/PVC (and prior jobs) before launch..."
  ansible-playbook playbooks/reset-storage.yml \
    -i inventory.yml
  echo "Reset complete."
fi

# -------------------------
# Step 1: Apply plan + start job (localhost inventory)
# -------------------------
echo
echo "Step 1: Applying test plan ConfigMap/PV/PVC and launching in-cluster executor job..."

ansible-playbook playbooks/start.yml \
  -i inventory.yml \
  -e tycho_test_plan_id="$PLAN_ID"

echo "Start complete (job launched)."

# Return to the original directory
cd "$ORIGINAL_DIR" || { echo "Error: Unable to return to the original directory: $ORIGINAL_DIR"; exit 1; }
