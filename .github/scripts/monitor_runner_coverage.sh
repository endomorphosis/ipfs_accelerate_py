#!/bin/bash

# Continuous Runner Coverage Monitor
# Ensures repositories updated in the last 24 hours have at least one active runner
# Works cooperatively with github_autoscaler.py
# Run this as a systemd service or cron job

set -e

# Resolve repo root so this script works from cron/systemd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
GH_API_CACHED=(python3 "$REPO_ROOT/scripts/utils/gh_api_cached.py")

LOG_FILE="${LOG_FILE:-/tmp/runner-coverage-monitor.log}"
CHECK_INTERVAL="${CHECK_INTERVAL:-300}" # 5 minutes default
AUTOSCALER_LOCKFILE="/tmp/github-autoscaler.lock"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Get list of all configured runner directories
get_runner_dirs() {
    local dirs=()
    for dir in /home/barberb/actions-runner* /home/barberb/*/actions-runner /home/barberb/actions-runners/*; do
        if [ -f "$dir/.runner" ]; then
            dirs+=("$dir")
        fi
    done
    echo "${dirs[@]}"
}

# Get repository URL for a runner
get_repo_url() {
    local runner_dir="$1"
    if [ -f "$runner_dir/.runner" ]; then
        cat "$runner_dir/.runner" 2>/dev/null | jq -r '.gitHubUrl // empty' 2>/dev/null
    fi
}

# Check if runner is running
is_runner_running() {
    local runner_dir="$1"
    pgrep -f "$runner_dir/bin/Runner.Listener" > /dev/null 2>&1
}

# Check if autoscaler is running and active
is_autoscaler_active() {
    # Check if github-autoscaler service is running
    if systemctl --user is-active --quiet github-autoscaler 2>/dev/null; then
        return 0
    fi
    
    # Check if autoscaler process is running
    if pgrep -f "github_autoscaler.py" > /dev/null 2>&1; then
        return 0
    fi
    
    return 1
}

# Start a runner
start_runner() {
    local runner_dir="$1"
    
    # If autoscaler is active, let it handle runner provisioning
    if is_autoscaler_active; then
        log "INFO: Autoscaler is active - it will handle runner provisioning for $runner_dir"
        return 0
    fi
    
    log "Starting runner at $runner_dir (autoscaler not active)"
    
    if [ -f "$runner_dir/svc.sh" ]; then
        cd "$runner_dir" && sudo ./svc.sh start >> "$LOG_FILE" 2>&1
        return $?
    elif [ -f "$runner_dir/runsvc.sh" ]; then
        cd "$runner_dir" && nohup ./runsvc.sh >> "$LOG_FILE" 2>&1 &
        return $?
    else
        log "ERROR: No service script found for $runner_dir"
        return 1
    fi
}

# Check repository activity (last commit time)
check_repo_activity() {
    local repo_url="$1"
    local repo_name=$(echo "$repo_url" | sed 's|https://github.com/||')
    
    # Try to get last commit time via GitHub CLI (if available and not rate-limited)
    if command -v gh &> /dev/null; then
        local last_push=$(${GH_API_CACHED[@]} "repos/$repo_name/commits" --jq '.[0].commit.committer.date' 2>/dev/null || echo "")
        
        if [ -n "$last_push" ]; then
            local last_push_timestamp=$(date -d "$last_push" +%s 2>/dev/null || echo "0")
            local now=$(date +%s)
            local age=$((now - last_push_timestamp))
            
            # Return 0 if updated within last 24 hours (86400 seconds)
            if [ $age -lt 86400 ]; then
                return 0
            fi
        fi
    fi
    
    # Fallback: check local repository if it exists
    local local_repos=(
        "/home/barberb/ipfs_accelerate_py"
        "/home/barberb/ipfs_datasets_py"
        "/home/barberb/ipfs_kit_py"
        "/home/barberb/swissknife"
        "/home/barberb/motion"
    )
    
    for local_repo in "${local_repos[@]}"; do
        if [ -d "$local_repo/.git" ]; then
            local remote=$(cd "$local_repo" && git config --get remote.origin.url 2>/dev/null | sed 's|\.git$||')
            if [[ "$remote" == *"$repo_name"* ]] || [[ "$repo_url" == *"$(basename $local_repo)"* ]]; then
                # Check last commit time in local repo
                local last_commit=$(cd "$local_repo" && git log -1 --format=%ct 2>/dev/null || echo "0")
                local now=$(date +%s)
                local age=$((now - last_commit))
                
                if [ $age -lt 86400 ]; then
                    return 0
                fi
            fi
        fi
    done
    
    return 1
}

# Main monitoring function
monitor_runners() {
    log "Starting runner coverage check..."
    
    # Check if autoscaler is active
    local autoscaler_active=false
    if is_autoscaler_active; then
        autoscaler_active=true
        log "INFO: GitHub autoscaler is active - working cooperatively"
    else
        log "INFO: GitHub autoscaler not detected - will start runners if needed"
    fi
    
    # Build map of repo -> runners
    declare -A repo_runners
    declare -A repo_running_count
    
    local runner_dirs=($(get_runner_dirs))
    
    for runner_dir in "${runner_dirs[@]}"; do
        local repo_url=$(get_repo_url "$runner_dir")
        if [ -n "$repo_url" ]; then
            # Initialize counters
            if [ -z "${repo_running_count[$repo_url]}" ]; then
                repo_running_count[$repo_url]=0
            fi
            
            # Track runner
            if [ -z "${repo_runners[$repo_url]}" ]; then
                repo_runners[$repo_url]="$runner_dir"
            else
                repo_runners[$repo_url]="${repo_runners[$repo_url]} $runner_dir"
            fi
            
            # Check if running
            if is_runner_running "$runner_dir"; then
                repo_running_count[$repo_url]=$((${repo_running_count[$repo_url]} + 1))
            fi
        fi
    done
    
    # Check each repository
    local issues_found=0
    for repo_url in "${!repo_runners[@]}"; do
        # Check if repository was updated in last 24 hours
        if check_repo_activity "$repo_url"; then
            local running=${repo_running_count[$repo_url]:-0}
            
            if [ $running -eq 0 ]; then
                log "WARNING: Repository $repo_url was recently updated but has no running runners!"
                
                if [ "$autoscaler_active" = true ]; then
                    log "INFO: Autoscaler is active - it will provision runners as needed"
                    log "INFO: No action taken - allowing autoscaler to manage $repo_url"
                else
                    # Try to start the first configured runner for this repo
                    local first_runner=$(echo "${repo_runners[$repo_url]}" | awk '{print $1}')
                    log "Attempting to start runner: $first_runner"
                    
                    if start_runner "$first_runner"; then
                        log "SUCCESS: Started runner for $repo_url"
                    else
                        log "ERROR: Failed to start runner for $repo_url"
                        ((issues_found++))
                    fi
                fi
            else
                log "OK: Repository $repo_url has $running running runner(s)"
            fi
        fi
    done
    
    if [ $issues_found -gt 0 ]; then
        log "ALERT: $issues_found repositories have issues"
        return 1
    else
        log "All recently-active repositories have adequate runner coverage"
        return 0
    fi
}

# Run once or continuously
if [ "$1" = "--once" ]; then
    log "Running single check..."
    monitor_runners
    exit $?
elif [ "$1" = "--daemon" ]; then
    log "Starting continuous monitoring (check interval: ${CHECK_INTERVAL}s)"
    while true; do
        monitor_runners || true
        log "Next check in ${CHECK_INTERVAL} seconds..."
        sleep "$CHECK_INTERVAL"
    done
else
    echo "Usage: $0 [--once|--daemon]"
    echo ""
    echo "Options:"
    echo "  --once    Run a single check and exit"
    echo "  --daemon  Run continuously with periodic checks"
    echo ""
    echo "Environment variables:"
    echo "  CHECK_INTERVAL  Seconds between checks (default: 300)"
    echo "  LOG_FILE        Log file path (default: /tmp/runner-coverage-monitor.log)"
    exit 1
fi
