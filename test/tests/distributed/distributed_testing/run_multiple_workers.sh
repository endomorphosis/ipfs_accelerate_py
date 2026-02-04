#!/bin/bash
# Script to run multiple worker nodes with different configurations
# This demonstrates how to deploy worker nodes in various configurations

# Common settings
COORDINATOR_URL="http://localhost:8080"
API_KEY="YOUR_API_KEY"
LOG_DIR="./logs"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Function to start a worker with specific settings
start_worker() {
    local worker_name=$1
    local tags=$2
    local db_path=$3
    local log_file="${LOG_DIR}/${worker_name}_$(date +%Y%m%d_%H%M%S).log"
    
    echo "Starting worker: $worker_name with tags: $tags"
    
    # Start worker in background with specified configuration
    python run_worker_example.py \
        --coordinator "$COORDINATOR_URL" \
        --api-key "$API_KEY" \
        --name "$worker_name" \
        --tags "$tags" \
        --db-path "$db_path" \
        > "$log_file" 2>&1 &
    
    # Store PID for later management
    echo $! > "${LOG_DIR}/${worker_name}.pid"
    echo "Worker $worker_name started (PID: $!), logs at $log_file"
}

# Function to stop all workers
stop_all_workers() {
    echo "Stopping all workers..."
    for pid_file in ${LOG_DIR}/*.pid; do
        if [ -f "$pid_file" ]; then
            pid=$(cat "$pid_file")
            worker_name=$(basename "$pid_file" .pid)
            echo "Stopping worker $worker_name (PID: $pid)"
            kill -SIGINT $pid
            rm "$pid_file"
        fi
    done
    echo "All workers stopped"
}

# Handle Ctrl+C to gracefully shut down workers
trap stop_all_workers EXIT

# Start different types of workers

# 1. CPU Worker
start_worker "cpu-worker-01" "cpu" "./db/cpu_worker.duckdb"

# 2. GPU Worker
# Only start if CUDA is available
if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    start_worker "gpu-worker-01" "gpu,cuda,transformers" "./db/gpu_worker.duckdb"
else
    echo "GPU not available, skipping GPU worker"
fi

# 3. Multi-model Worker
start_worker "model-worker-01" "transformers,bert,vit,clip" "./db/model_worker.duckdb"

# 4. OpenVINO Worker
# Only start if OpenVINO is available
if python -c "import importlib.util; print(importlib.util.find_spec('openvino') is not None)" | grep -q "True"; then
    start_worker "openvino-worker-01" "openvino,cpu" "./db/openvino_worker.duckdb"
else
    echo "OpenVINO not available, skipping OpenVINO worker"
fi

# 5. Benchmark Worker (optimized for benchmark tasks)
start_worker "benchmark-worker-01" "benchmark,performance" "./db/benchmark_worker.duckdb"

echo "All workers started. Press Ctrl+C to stop all workers."

# Wait for Ctrl+C
while true; do
    sleep 1
done