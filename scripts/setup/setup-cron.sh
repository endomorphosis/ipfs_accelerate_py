#!/bin/bash
#
# IPFS Accelerate Cron Job Setup
# This script sets up a cron job to ensure the service is always running
#

set -e

USER="barberb"
PROJECT_DIR="/home/barberb/ipfs_accelerate_py"
CRON_SCRIPT="$PROJECT_DIR/check-service.sh"

echo "Setting up IPFS Accelerate monitoring cron job..."

# Create the service check script
cat > "$CRON_SCRIPT" << 'EOF'
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
EOF

# Make the script executable
chmod +x "$CRON_SCRIPT"
chown "$USER:$USER" "$CRON_SCRIPT"

# Add cron job (runs every 5 minutes)
CRON_JOB="*/5 * * * * $CRON_SCRIPT"

# Check if cron job already exists
if ! crontab -u "$USER" -l 2>/dev/null | grep -q "$CRON_SCRIPT"; then
    # Add the new cron job
    (crontab -u "$USER" -l 2>/dev/null; echo "$CRON_JOB") | crontab -u "$USER" -
    echo "✓ Cron job added successfully"
else
    echo "✓ Cron job already exists"
fi

echo ""
echo "Cron job setup complete!"
echo "  Script location: $CRON_SCRIPT"
echo "  Runs every: 5 minutes"
echo "  Log file: /tmp/ipfs-accelerate-monitor.log"
echo ""
echo "To view current cron jobs: crontab -u $USER -l"
echo "To remove the cron job: crontab -u $USER -e (then delete the line)"