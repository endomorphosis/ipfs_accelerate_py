#!/bin/bash
# Example script to demonstrate the performance trend analyzer functionality
# This shows how to generate performance reports and visualizations

# Create directory for the script if it doesn't exist
mkdir -p $(dirname "$0")

# Ensure the database directory exists
mkdir -p ./database

# Create visualization directory
mkdir -p ./visualizations

# Generate basic performance report
echo "Generating performance report..."
python3 ../coordinator.py \
  --db-path ./database/benchmark_results.duckdb \
  --performance-analyzer \
  --visualization-path ./visualizations \
  --report \
  --report-output ./visualizations/performance_report.html

# Start coordinator with performance analyzer
echo "Starting coordinator with performance analyzer..."
python3 ../coordinator.py \
  --host localhost \
  --port 8080 \
  --db-path ./database/benchmark_results.duckdb \
  --performance-analyzer \
  --visualization-path ./visualizations &

# Store coordinator PID
COORDINATOR_PID=$!

# Wait for the coordinator to start
sleep 2

echo "Coordinator started with performance analyzer."
echo "Performance visualizations will be stored in ./visualizations/"
echo "Press Enter to stop the coordinator."
read

# Stop the coordinator
echo "Stopping coordinator..."
kill $COORDINATOR_PID

echo "Generating comprehensive performance report..."
python3 ../coordinator.py \
  --db-path ./database/benchmark_results.duckdb \
  --performance-analyzer \
  --visualization-path ./visualizations \
  --report \
  --report-output ./visualizations/comprehensive_report.html

echo "Performance analysis completed. Check ./visualizations/ for reports and visualizations."