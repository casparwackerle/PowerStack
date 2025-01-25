#!/bin/bash

# Ensure the script is running from the correct directory
SCRIPT_DIR="$(dirname "$0")"
SCRIPT_NAME="$(basename "$0")"

# Find all shell scripts in the directory, excluding this script
SCRIPTS=($(find "$SCRIPT_DIR" -maxdepth 1 -type f -name "*.sh" ! -name "$SCRIPT_NAME" | sort))

# Ensure all scripts are executable
for script in "${SCRIPTS[@]}"; do
    if [[ ! -x "$script" ]]; then
        echo "Making $script executable..."
        chmod +x "$script"
    fi
done

# Run each script sequentially with a 20-minute sleep in between
for script in "${SCRIPTS[@]}"; do
    echo "Running $script"
    "$script"
    echo "Sleeping for 20 minutes before running the next script..."
    sleep 1200
done

echo "All scripts completed."
