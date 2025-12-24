#!/bin/bash
# filepath: gearboxAssembly/scripts/rule_based_unstop.sh

# 1. Activate Conda environment
# Note: To activate conda in a script, you need to source conda.sh first
# Please modify the path below according to your actual installation, typically ~/anaconda3/etc/profile.d/conda.sh or ~/miniconda3/...
source ~/miniconda3/etc/profile.d/conda.sh
conda activate roco

# Define the command to run
CMD="python scripts/rule_based_agent.py --task=Template-Galaxea-Lab-External-Direct-v0 --enable_cameras"

# Define runtime for each iteration (seconds)
RUNTIME=600  # 10 minutes = 600 seconds

echo "Starting task loop..."
echo "Press Ctrl+C to terminate the script (may need to hold for a few seconds or press multiple times)"

while true; do
    # Get current time as log prefix
    TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
    echo "[$TIMESTAMP] Starting new task iteration..."

    # Run command in background
    $CMD &
    
    # Get the PID of the background process
    PID=$!
    echo "[$TIMESTAMP] Process started, PID: $PID"

    # Wait for specified time
    sleep $RUNTIME

    # Check if process is still running
    if ps -p $PID > /dev/null; then
        echo "[$TIMESTAMP] Time's up, terminating process $PID ..."
        
        # Send SIGINT (equivalent to Ctrl+C), allowing Python script to execute finally block and save data
        kill -SIGINT $PID
        
        # Wait a moment for it to save data
        sleep 10
        
        # If still running, force kill
        if ps -p $PID > /dev/null; then
            echo "[$TIMESTAMP] Process not responding, force killing..."
            kill -9 $PID
        fi
    else
        echo "[$TIMESTAMP] Process ended early."
    fi

    echo "[$TIMESTAMP] Preparing for next iteration..."
    echo "----------------------------------------"
    sleep 2 # Brief pause before restart
done