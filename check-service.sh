#!/bin/bash
#
# IPFS Accelerate Service Monitor
# This script checks if the service is running and starts it if not
#

SERVICE_NAME="ipfs-accelerate"
LOG_FILE="/tmp/ipfs-accelerate-monitor.log"

# Function to log messages
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
}

# Check if service exists
if ! systemctl list-unit-files | grep -q "^${SERVICE_NAME}.service"; then
    log_message "ERROR: Service $SERVICE_NAME is not installed"
    exit 1
fi

# Check if service is active
if ! systemctl is-active --quiet "$SERVICE_NAME"; then
    log_message "WARNING: Service $SERVICE_NAME is not running, attempting to start..."
    
    # Try to start the service
    if sudo systemctl start "$SERVICE_NAME"; then
        log_message "SUCCESS: Service $SERVICE_NAME started successfully"
    else
        log_message "ERROR: Failed to start service $SERVICE_NAME"
        exit 1
    fi
else
    # Service is running, check if it's actually responding
    if timeout 5s curl -s http://localhost:9000/health > /dev/null 2>&1; then
        log_message "INFO: Service $SERVICE_NAME is running and responding"
    else
        log_message "WARNING: Service $SERVICE_NAME is running but not responding, restarting..."
        if sudo systemctl restart "$SERVICE_NAME"; then
            log_message "SUCCESS: Service $SERVICE_NAME restarted successfully"
        else
            log_message "ERROR: Failed to restart service $SERVICE_NAME"
        fi
    fi
fi

# Clean up old log entries (keep last 100 lines)
if [ -f "$LOG_FILE" ]; then
    tail -n 100 "$LOG_FILE" > "${LOG_FILE}.tmp" && mv "${LOG_FILE}.tmp" "$LOG_FILE"
fi
