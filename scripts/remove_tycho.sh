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
LOG_FILE="$REPO_PATH/logs/removal_tycho_$TIMESTAMP.log"

# Redirect script output to a timestamped log file
exec > >(tee -i "$LOG_FILE") 2>&1

echo "Log file created: $LOG_FILE"

# Navigate to the tycho-ansible directory
cd "$REPO_PATH/ansible/tycho-ansible" || { echo "Error: 'tycho-ansible' directory not found in $REPO_PATH"; exit 1; }

# Run the Ansible playbook with the inventory file and ask for vault password
ansible-playbook playbooks/remove-tycho.yml -i inventory.yml -i ../../configs/inventory.yml
echo "tycho removal complete!"

# Return to the original directory
cd "$ORIGINAL_DIR" || { echo "Error: Unable to return to the original directory: $ORIGINAL_DIR"; exit 1; }