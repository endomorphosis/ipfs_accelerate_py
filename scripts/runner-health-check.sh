#!/bin/bash

# GitHub Actions Runner Health Check Script
# Monitors runner health and automatically restarts if needed

HEALTH_LOG="/var/log/github-actions/health-check.log"

# Function to log with timestamp
health_log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$HEALTH_LOG"
}

# Function to check runner health
check_runner_health() {
    local service_name=$1
    local runner_name=$2
    
    if systemctl is-active --quiet "$service_name"; then
        health_log "OK: $runner_name service is running"
        
        # Check if runner is actually listening/responsive
        # This is a basic check - you might want to enhance this
        if systemctl status "$service_name" | grep -q "Listening for Jobs"; then
            health_log "OK: $runner_name is listening for jobs"
            return 0
        else
            health_log "WARNING: $runner_name service running but not listening for jobs"
            return 1
        fi
    else
        health_log "ERROR: $runner_name service is not running"
        return 1
    fi
}

# Function to restart runner
restart_runner() {
    local service_name=$1
    local runner_name=$2
    
    health_log "Attempting to restart $runner_name..."
    sudo systemctl restart "$service_name"
    sleep 10
    
    if systemctl is-active --quiet "$service_name"; then
        health_log "SUCCESS: $runner_name restarted successfully"
    else
        health_log "ERROR: Failed to restart $runner_name"
    fi
}

health_log "=== Runner Health Check Started ==="

# Check main runner
if ! check_runner_health "github-actions-runner" "main runner"; then
    restart_runner "github-actions-runner" "main runner"
fi

# Check datasets runner (if it exists)
if systemctl list-unit-files | grep -q "github-actions-runner-datasets"; then
    if ! check_runner_health "github-actions-runner-datasets" "datasets runner"; then
        restart_runner "github-actions-runner-datasets" "datasets runner"
    fi
fi

health_log "=== Runner Health Check Completed ==="