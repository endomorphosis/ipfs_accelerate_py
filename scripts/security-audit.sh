#!/bin/bash

# GitHub Actions Runner Security Audit Script
# Performs regular security checks and generates reports

AUDIT_LOG="/var/log/github-actions/security-audit.log"

# Function to log with timestamp
audit_log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$AUDIT_LOG"
}

audit_log "=== Security Audit Started ==="

# Check for unauthorized SSH keys
audit_log "Checking SSH authorized keys..."
if [ -f ~/.ssh/authorized_keys ]; then
    key_count=$(wc -l < ~/.ssh/authorized_keys)
    audit_log "Found $key_count authorized SSH keys"
    if [ "$key_count" -gt 5 ]; then
        audit_log "WARNING: Large number of SSH keys detected"
    fi
else
    audit_log "No SSH authorized_keys file found"
fi

# Check for unusual network connections
audit_log "Checking network connections..."
netstat_count=$(netstat -tuln | wc -l)
audit_log "Active network connections: $netstat_count"

# Check for running Docker containers
audit_log "Checking Docker containers..."
if command -v docker &> /dev/null; then
    container_count=$(docker ps -q | wc -l)
    audit_log "Running Docker containers: $container_count"
fi

# Check system updates
audit_log "Checking for system updates..."
update_count=$(apt list --upgradable 2>/dev/null | grep -c "upgradable")
audit_log "Available system updates: $update_count"

if [ "$update_count" -gt 20 ]; then
    audit_log "WARNING: Many system updates available ($update_count)"
fi

# Check fail2ban status
audit_log "Checking fail2ban status..."
if systemctl is-active --quiet fail2ban; then
    banned_ips=$(sudo fail2ban-client status sshd | grep "Banned IP list" | cut -d: -f2 | wc -w)
    audit_log "fail2ban is active, banned IPs: $banned_ips"
else
    audit_log "WARNING: fail2ban is not active"
fi

# Check for suspicious processes
audit_log "Checking for suspicious processes..."
suspicious_processes=$(ps aux | grep -E "(nc|netcat|ncat|socat|python.*socket|perl.*socket|ruby.*socket)" | grep -v grep | wc -l)
audit_log "Potentially suspicious processes: $suspicious_processes"

audit_log "=== Security Audit Completed ==="