#!/bin/bash
# Example script to start a high-availability coordinator cluster
# This demonstrates the auto recovery system for coordinator redundancy and failover

# Create directory for the script if it doesn't exist
mkdir -p $(dirname "$0")

# Ensure the database directory exists
mkdir -p ./database

# Create visualization directory
mkdir -p ./visualizations

# Start first coordinator (leader)
echo "Starting primary coordinator..."
python3 ../coordinator.py \
  --host localhost \
  --port 8080 \
  --db-path ./database/coordinator.duckdb \
  --auto-recovery \
  --coordinator-id coordinator-1 \
  --performance-analyzer \
  --visualization-path ./visualizations &

# Store the PID of the first coordinator
COORDINATOR1_PID=$!

# Wait for the first coordinator to start
sleep 2

# Start second coordinator (follower)
echo "Starting secondary coordinator..."
python3 ../coordinator.py \
  --host localhost \
  --port 8081 \
  --db-path ./database/coordinator.duckdb \
  --auto-recovery \
  --coordinator-id coordinator-2 \
  --coordinator-addresses localhost:8080 \
  --performance-analyzer \
  --visualization-path ./visualizations &

# Store the PID of the second coordinator
COORDINATOR2_PID=$!

# Wait for the second coordinator to start
sleep 2

# Start third coordinator (follower)
echo "Starting third coordinator..."
python3 ../coordinator.py \
  --host localhost \
  --port 8082 \
  --db-path ./database/coordinator.duckdb \
  --auto-recovery \
  --coordinator-id coordinator-3 \
  --coordinator-addresses localhost:8080,localhost:8081 \
  --performance-analyzer \
  --visualization-path ./visualizations &

# Store the PID of the third coordinator
COORDINATOR3_PID=$!

echo "Cluster started with 3 coordinators. Press Enter to simulate leader failure."
read

# Simulate leader failure by killing the first coordinator
echo "Simulating leader failure by stopping coordinator-1..."
kill $COORDINATOR1_PID

echo "Leader failed. The cluster should automatically elect a new leader."
echo "Press Enter to restart the failed coordinator."
read

# Restart the first coordinator (should join as follower)
echo "Restarting coordinator-1..."
python3 ../coordinator.py \
  --host localhost \
  --port 8080 \
  --db-path ./database/coordinator.duckdb \
  --auto-recovery \
  --coordinator-id coordinator-1 \
  --coordinator-addresses localhost:8081,localhost:8082 \
  --performance-analyzer \
  --visualization-path ./visualizations &

# Store the PID of the restarted coordinator
COORDINATOR1_PID=$!

echo "Cluster is now fully operational again."
echo "Press Enter to shutdown the cluster."
read

# Shutdown all coordinators
echo "Shutting down the cluster..."
kill $COORDINATOR1_PID $COORDINATOR2_PID $COORDINATOR3_PID

echo "Cluster shutdown completed."