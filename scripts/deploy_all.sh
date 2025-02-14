#!/bin/bash

export ANSIBLE_FORCE_COLOR=true

# Enable error handling for unset variables and pipelines
set -u  # Treat unset variables as an error
set -o pipefail  # Return the exit status of the last command in a pipeline that failed

# Store the original working directory
ORIGINAL_DIR=$(pwd)

# Detect the repository path as one level above the current script's directory
REPO_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# Ensure REPO_PATH is valid
if [ -z "$REPO_PATH" ] || [ ! -d "$REPO_PATH" ]; then
    echo "Error: Unable to determine the repository path. Please check your setup."
    exit 1
fi

# Notify the user about multiple password prompts (wait 3 seconds)
echo "======================================================================"
echo "NOTE: You will be prompted to enter the Ansible Vault password"
echo "multiple times during this installation process."
echo "======================================================================"

# Dynamic countdown from 5 seconds to 0
for i in {2..1}; do
    echo -ne "$i...\r"
    sleep 1
done
echo "0... Continuing installation."


# Define the list of scripts to execute in order
SCRIPTS=(
    "$REPO_PATH/scripts/deploy_k3s.sh"
    "$REPO_PATH/scripts/deploy_rancher.sh"
    "$REPO_PATH/scripts/deploy_nfs.sh"
    "$REPO_PATH/scripts/deploy_monitoring-stack.sh"
    "$REPO_PATH/scripts/deploy_kepler.sh"
    "$REPO_PATH/scripts/deploy_ubuntu-kubectl.sh"
)

# Execute each script in order
for SCRIPT in "${SCRIPTS[@]}"; do
    if [ ! -f "$SCRIPT" ]; then
        echo "Error: Script $SCRIPT not found. Aborting."
        exit 1
    fi

    echo "Executing $SCRIPT..."
    
    # Capture and display errors without exiting immediately
    if ! bash "$SCRIPT"; then
        echo "======================================================================"
        echo "Error: $SCRIPT failed. Stopping further execution."
        echo "Check the output above for details about the failure."
        echo "======================================================================"
        exit 1  # Stop execution, but keep the terminal open
    fi

    echo "$SCRIPT executed successfully."
done

# Return to the original directory
cd "$ORIGINAL_DIR" || { echo "Error: Unable to return to the original directory: $ORIGINAL_DIR"; exit 1; }

echo "======================================================================"
echo ""
echo "All scripts executed successully"
echo ""
echo "======================================================================"
