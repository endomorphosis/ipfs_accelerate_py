#!/bin/bash
# Cleanup script for GitHub Actions runner workspace permissions
# This should be run as a cron job or systemd timer

set -e

RUNNER_DIRS=(
    "/home/barberb/actions-runner/_work"
    "/home/barberb/actions-runner-ipfs_datasets_py/_work"
    "/home/devel/actions-runner/_work"
    "/home/actions-runner/_work"
    "/tmp/_work"
)

echo "🧹 Starting runner workspace cleanup..."
echo "Timestamp: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"

for dir in "${RUNNER_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "Processing: $dir"
        
        # Remove lock files
        find "$dir" -name "*.lock" -type f -delete 2>/dev/null || true
        find "$dir" -name "index.lock" -type f -delete 2>/dev/null || true
        
        # Fix .git directory permissions
        find "$dir" -name ".git" -type d -exec chmod -R u+rwX {} \; 2>/dev/null || true
        
        # Fix specific problematic files
        find "$dir" -name "FETCH_HEAD" -type f -exec chmod u+rw {} \; 2>/dev/null || true
        find "$dir" -name ".gitignore" -type f -exec chmod u+rw {} \; 2>/dev/null || true
        
        # Fix .pytest_cache directories (common source of permission issues)
        find "$dir" -name ".pytest_cache" -type d -exec chmod -R u+rwX {} \; 2>/dev/null || true
        
        # Fix ownership if running as root/sudo
        if [ "$EUID" -eq 0 ]; then
            # Detect directory owner and fix ownership
            DIR_OWNER=$(stat -c '%U' "$dir" 2>/dev/null || echo "unknown")
            if [ "$DIR_OWNER" != "unknown" ] && [ "$DIR_OWNER" != "root" ]; then
                echo "  Fixing ownership to $DIR_OWNER"
                chown -R "$DIR_OWNER:$DIR_OWNER" "$dir" 2>/dev/null || true
            fi
        fi
        
        echo "  ✓ Cleaned $dir"
    fi
done

echo "✅ Cleanup complete"
