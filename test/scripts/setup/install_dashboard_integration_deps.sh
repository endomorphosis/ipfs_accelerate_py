#!/bin/bash
# Script to install dependencies for the monitoring dashboard integration

# Exit on error
set -e

# Print commands
set -x

# Create a virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python -m venv venv
fi

# Activate the virtual environment
source venv/bin/activate

# Install required packages
pip install duckdb pandas fastapi uvicorn pydantic plotly requests websocket-client

# Install additional packages for visualization
pip install matplotlib seaborn

# Print success message
echo "Dependencies installed successfully!"
echo "You can now run:"
echo "  source venv/bin/activate"
echo "  python run_monitoring_dashboard_integration.py --mode=run-and-sync"