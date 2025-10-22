#!/bin/bash

# GitHub Actions Runner Security Configuration Script
# This script sets up firewall rules, log monitoring, and security best practices

set -e

echo "=== GitHub Actions Runner Security Setup ==="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root for security reasons"
   exit 1
fi

# 1. Configure UFW Firewall
print_status "Configuring UFW firewall..."

# Reset UFW to default state
sudo ufw --force reset

# Set default policies
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow SSH (be careful not to lock yourself out)
sudo ufw allow ssh

# Allow HTTP and HTTPS for package downloads and GitHub communication
sudo ufw allow out 80/tcp
sudo ufw allow out 443/tcp

# Allow DNS
sudo ufw allow out 53

# Allow NTP for time synchronization
sudo ufw allow out 123/udp

# Allow GitHub Actions runner communication
# GitHub uses various IPs, so we'll allow HTTPS outbound which covers this
print_status "Allowing GitHub Actions communication (HTTPS outbound already configured)"

# Allow Docker daemon (if using Docker in workflows)
if command -v docker &> /dev/null; then
    print_status "Docker detected, configuring Docker-related firewall rules..."
    # Allow Docker bridge network
    sudo ufw allow in on docker0
    sudo ufw allow out on docker0
fi

# Allow local services that might be used in CI/CD
# MCP servers, dashboards, etc.
sudo ufw allow from 127.0.0.1 to any port 8000:9999
sudo ufw allow from ::1 to any port 8000:9999

# Enable firewall
sudo ufw --force enable

print_status "Firewall configuration completed"

# 2. Configure log monitoring
print_status "Setting up log monitoring..."

# Create log directory for runner logs
sudo mkdir -p /var/log/github-actions
sudo chown barberb:barberb /var/log/github-actions

# Create logrotate configuration for GitHub Actions logs
sudo tee /etc/logrotate.d/github-actions > /dev/null << 'EOF'
/var/log/github-actions/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    copytruncate
    su barberb barberb
}
EOF

# 3. Set up fail2ban for SSH protection
print_status "Installing and configuring fail2ban..."

if ! command -v fail2ban-server &> /dev/null; then
    sudo apt update
    sudo apt install -y fail2ban
fi

# Create fail2ban configuration for SSH
sudo tee /etc/fail2ban/jail.local > /dev/null << 'EOF'
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 5
backend = systemd

[sshd]
enabled = true
port = ssh
logpath = %(sshd_log)s
maxretry = 3
bantime = 7200
EOF

# Enable and start fail2ban
sudo systemctl enable fail2ban
sudo systemctl restart fail2ban

# 4. Configure system resource limits
print_status "Configuring system resource limits..."

# Create limits for the barberb user (runner user)
sudo tee /etc/security/limits.d/github-actions.conf > /dev/null << 'EOF'
# GitHub Actions runner resource limits
barberb soft nproc 4096
barberb hard nproc 8192
barberb soft nofile 65536
barberb hard nofile 65536
barberb soft memlock unlimited
barberb hard memlock unlimited
EOF

# 5. Set up monitoring script
print_status "Creating monitoring script..."

cat > /home/barberb/ipfs_accelerate_py/scripts/monitor-runners.sh << 'EOF'
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
for runner_service in github-actions-runner-ipfs-accelerate github-actions-runner-datasets; do
    if systemctl is-active --quiet "$runner_service"; then
        log_message "OK: $runner_service is running"
    else
        log_message "ERROR: $runner_service is not running"
    fi
done

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
EOF

chmod +x /home/barberb/ipfs_accelerate_py/scripts/monitor-runners.sh

# 6. Set up cron job for monitoring
print_status "Setting up monitoring cron job..."

# Add monitoring job to crontab (run every 5 minutes)
(crontab -l 2>/dev/null || echo "") | grep -v "monitor-runners.sh" > /tmp/current_cron
echo "*/5 * * * * /home/barberb/ipfs_accelerate_py/scripts/monitor-runners.sh" >> /tmp/current_cron
crontab /tmp/current_cron
rm /tmp/current_cron

# 7. Create security audit script
print_status "Creating security audit script..."

cat > /home/barberb/ipfs_accelerate_py/scripts/security-audit.sh << 'EOF'
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
EOF

chmod +x /home/barberb/ipfs_accelerate_py/scripts/security-audit.sh

# 8. Set up weekly security audit
print_status "Setting up weekly security audit..."

# Add security audit job to crontab (run weekly on Sunday at 2 AM)
(crontab -l 2>/dev/null || echo "") | grep -v "security-audit.sh" > /tmp/current_cron
echo "0 2 * * 0 /home/barberb/ipfs_accelerate_py/scripts/security-audit.sh" >> /tmp/current_cron
crontab /tmp/current_cron
rm /tmp/current_cron

# 9. Create runner health check script
print_status "Creating runner health check script..."

cat > /home/barberb/ipfs_accelerate_py/scripts/runner-health-check.sh << 'EOF'
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

# Check ipfs-accelerate runner
if ! check_runner_health "github-actions-runner-ipfs-accelerate" "ipfs-accelerate runner"; then
    restart_runner "github-actions-runner-ipfs-accelerate" "ipfs-accelerate runner"
fi

# Check datasets runner (if it exists)
if systemctl list-unit-files | grep -q "github-actions-runner-datasets"; then
    if ! check_runner_health "github-actions-runner-datasets" "datasets runner"; then
        restart_runner "github-actions-runner-datasets" "datasets runner"
    fi
fi

health_log "=== Runner Health Check Completed ==="
EOF

chmod +x /home/barberb/ipfs_accelerate_py/scripts/runner-health-check.sh

# 10. Set up hourly health checks
print_status "Setting up hourly health checks..."

# Add health check job to crontab (run every hour)
(crontab -l 2>/dev/null || echo "") | grep -v "runner-health-check.sh" > /tmp/current_cron
echo "0 * * * * /home/barberb/ipfs_accelerate_py/scripts/runner-health-check.sh" >> /tmp/current_cron
crontab /tmp/current_cron
rm /tmp/current_cron

print_status "Security and monitoring setup completed!"

echo ""
echo "=== Summary ==="
print_status "✅ UFW firewall configured with secure rules"
print_status "✅ fail2ban installed and configured for SSH protection"
print_status "✅ System resource limits configured"
print_status "✅ Log rotation configured"
print_status "✅ Monitoring scripts created and scheduled"
print_status "✅ Security audit script created and scheduled"
print_status "✅ Health check script created and scheduled"

echo ""
print_warning "Please review the firewall rules with: sudo ufw status verbose"
print_warning "Monitor logs in: /var/log/github-actions/"
print_warning "Check cron jobs with: crontab -l"

echo ""
print_status "Security setup is complete!"