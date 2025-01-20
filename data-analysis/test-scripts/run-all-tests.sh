#!/bin/bash

# Ensure the script is running from the correct directory
SCRIPT_DIR="$(dirname "$0")"
SCRIPT_NAME="$(basename "$0")"

# Find all executable scripts in the directory, excluding this script
SCRIPTS=($(find "$SCRIPT_DIR" -maxdepth 1 -type f -name "*.sh" -executable ! -name "$SCRIPT_NAME" | sort))

# Run each script sequentially with a 20-minute sleep in between
for script in "${SCRIPTS[@]}"; do
    echo "Running $script"
    "$script"
    echo "Sleeping for 20 minutes before running the next script..."
    sleep 1200
done

echo "All scripts completed."