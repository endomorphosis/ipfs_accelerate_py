#!/bin/bash

# GitHub Actions Runner Permission Fix Script
# Fixes EACCES permission denied errors when removing files in runner workspace
# Error: EACCES: permission denied, unlink '/home/actions-runner/_work/...'

# Don't exit on error - we want to continue through all runners
set +e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to fix permissions for a specific runner
fix_runner_permissions() {
    local runner_path="$1"
    local runner_name=$(basename "$runner_path")
    
    log_info "Fixing permissions for runner: $runner_name"
    
    if [ ! -d "$runner_path" ]; then
        log_error "Runner directory not found: $runner_path"
        return 1
    fi
    
    # Navigate to runner workspace
    local work_dir="$runner_path/_work"
    
    if [ ! -d "$work_dir" ]; then
        log_warn "No _work directory found for $runner_name"
        return 0
    fi
    
    # Save current directory
    local original_dir=$(pwd)
    
    cd "$work_dir" || return 1
    
    log_info "Cleaning workspace: $work_dir"
    
    # Remove all git lock files
    log_info "Removing stale git lock files..."
    find . -name "*.lock" -type f -delete 2>/dev/null || true
    find . -name "index.lock" -type f -delete 2>/dev/null || true
    
    # Fix git directory permissions
    log_info "Fixing git directory permissions..."
    find . -name ".git" -type d -exec chmod -R u+rwX {} \; 2>/dev/null || true
    find . -name ".git" -type d -exec find {} -type f -exec chmod u+rw {} \; \; 2>/dev/null || true
    
    # Fix git logs directory permissions specifically (common issue in Actions)
    log_info "Fixing git logs permissions..."
    find . -path "*/.git/logs/*" -type f -exec chmod u+rw {} \; 2>/dev/null || true
    find . -path "*/.git/logs" -type d -exec chmod u+rwx {} \; 2>/dev/null || true
    find . -path "*/.git/logs/refs" -type d -exec chmod -R u+rwx {} \; 2>/dev/null || true
    
    # Fix .github directory permissions
    log_info "Fixing .github directory permissions..."
    find . -path "*/.github/*" -type f -exec chmod u+rw {} \; 2>/dev/null || true
    find . -path "*/.github" -type d -exec chmod u+rwx {} \; 2>/dev/null || true
    
    # Remove Python cache files
    log_info "Cleaning Python cache files..."
    find . -name "*.pyc" -delete 2>/dev/null || true
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    
    # Remove node_modules if needed (optional)
    # Uncomment if you have Node.js projects
    # log_info "Cleaning node_modules..."
    # find . -name "node_modules" -type d -exec rm -rf {} + 2>/dev/null || true
    
    # Fix ownership to current user
    local current_user=$(whoami)
    log_info "Fixing ownership to $current_user..."
    
    # First try without sudo (for files already owned by user)
    find "$work_dir" -type f ! -user "$current_user" -path "*/.git/*" 2>/dev/null | head -1 > /dev/null
    if [ $? -eq 0 ]; then
        # There are files not owned by current user, try to fix with sudo
        if sudo -n chown -R "$current_user:$current_user" "$work_dir" 2>/dev/null; then
            log_info "✓ Fixed ownership with sudo"
        else
            log_warn "Some files owned by other users - may need manual sudo"
        fi
    else
        log_info "✓ All files already owned by $current_user"
    fi
    
    # Return to original directory
    cd "$original_dir" || true
    
    log_info "✓ Completed fixing permissions for $runner_name"
    return 0
}

# Main execution
main() {
    log_info "GitHub Actions Runner Permission Fix Script"
    log_info "==========================================="
    
    # Default runner locations (discovered from ps aux)
    RUNNER_LOCATIONS=(
        "/home/barberb/actions-runner"
        "/home/barberb/actions-runner-ipfs_datasets_py"
        "/home/barberb/actions-runners/endomorphosis-ipfs_kit_py"
        "/home/barberb/swissknife/actions-runner"
        "/home/barberb/motion/actions-runner"
    )
    
    # Check if specific runner path provided
    if [ -n "$1" ]; then
        RUNNER_LOCATIONS=("$1")
    fi
    
    local fixed_count=0
    local error_count=0
    
    for runner_path in "${RUNNER_LOCATIONS[@]}"; do
        if [ -d "$runner_path" ]; then
            if fix_runner_permissions "$runner_path"; then
                ((fixed_count++))
            else
                ((error_count++))
            fi
        else
            log_warn "Runner not found: $runner_path"
        fi
    done
    
    echo ""
    log_info "==========================================="
    log_info "Summary:"
    log_info "  Runners fixed: $fixed_count"
    if [ $error_count -gt 0 ]; then
        log_warn "  Runners with errors: $error_count"
    fi
    log_info "==========================================="
    
    if [ $fixed_count -eq 0 ]; then
        log_error "No runners were fixed. Check runner locations."
        exit 1
    fi
}

# Show usage
if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    echo "Usage: $0 [runner_path]"
    echo ""
    echo "Fix GitHub Actions runner permission issues"
    echo ""
    echo "Options:"
    echo "  runner_path    Optional path to specific runner directory"
    echo "                 Default: searches common runner locations"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Fix all runners"
    echo "  $0 /home/barberb/actions-runner      # Fix specific runner"
    exit 0
fi

main "$@"
