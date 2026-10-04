#!/bin/bash

# GitHub Actions Runner Monitoring Script
# Checks runner status, system resources, and logs alerts

LOG_FILE="/var/log/github-actions/monitor.log"
ALERT_THRESHOLD_CPU=80
ALERT_THRESHOLD_MEM=80
ALERT_THRESHOLD_DISK=90

# Function to log with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Check CPU usage
cpu_usage=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')
cpu_usage_int=${cpu_usage%.*}

if [ "$cpu_usage_int" -gt "$ALERT_THRESHOLD_CPU" ]; then
    log_message "WARNING: High CPU usage detected: ${cpu_usage}%"
fi

# Check memory usage
mem_usage=$(free | grep Mem | awk '{printf("%.1f", ($3/$2) * 100.0)}')
mem_usage_int=${mem_usage%.*}

if [ "$mem_usage_int" -gt "$ALERT_THRESHOLD_MEM" ]; then
    log_message "WARNING: High memory usage detected: ${mem_usage}%"
fi

# Check disk usage
disk_usage=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')

if [ "$disk_usage" -gt "$ALERT_THRESHOLD_DISK" ]; then
    log_message "WARNING: High disk usage detected: ${disk_usage}%"
fi

# Check runner services
for runner_service in github-actions-runner; do
    if systemctl is-active --quiet "$runner_service"; then
        log_message "OK: $runner_service is running"
    else
        log_message "ERROR: $runner_service is not running"
    fi
done

# Check for additional dataset runner if it exists
if systemctl list-unit-files | grep -q "github-actions-runner-datasets"; then
    if systemctl is-active --quiet "github-actions-runner-datasets"; then
        log_message "OK: github-actions-runner-datasets is running"
    else
        log_message "ERROR: github-actions-runner-datasets is not running"
    fi
fi

# Check for recent errors in runner logs
for runner_dir in /home/barberb/actions-runner-* /home/barberb/actions-runner-datasets; do
    if [ -d "$runner_dir" ]; then
        error_count=$(find "$runner_dir/_diag" -name "*.log" -newer /tmp/last_check 2>/dev/null | xargs grep -i "error\|exception\|failed" 2>/dev/null | wc -l)
        if [ "$error_count" -gt 0 ]; then
            log_message "WARNING: Found $error_count errors in $runner_dir logs since last check"
        fi
    fi
done

# Update timestamp for next check
touch /tmp/last_check