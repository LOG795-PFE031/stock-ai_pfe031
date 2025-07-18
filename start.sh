#!/bin/bash

# Log: Starting Prefect server
echo "[start.sh] Starting Prefect server..."
prefect server start --host 0.0.0.0 &

# Wait a few seconds to ensure Prefect server starts
sleep 5

# Log: Starting main app
echo "[start.sh] Starting main app (python main.py)..."
python main.py

# Log: If this line is reached, main.py has exited
echo "[start.sh] main.py has exited. Script completed."
