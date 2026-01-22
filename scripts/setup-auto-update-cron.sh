#!/bin/bash
# Setup cron job for auto-updating ipfs_accelerate_py

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPDATE_SCRIPT="${SCRIPT_DIR}/auto-update.sh"

echo "Setting up auto-update cron job..."

# Check if the update script exists
if [ ! -f "${UPDATE_SCRIPT}" ]; then
    echo "ERROR: Update script not found at ${UPDATE_SCRIPT}"
    exit 1
fi

# Make sure the update script is executable
chmod +x "${UPDATE_SCRIPT}"

# Remove any existing auto-update cron jobs
(crontab -l 2>/dev/null || echo "") | grep -v "auto-update.sh" > /tmp/current_cron || true

# Add new cron job to run every 6 hours
echo "# Auto-update ipfs_accelerate_py from main branch every 6 hours" >> /tmp/current_cron
echo "0 */6 * * * ${UPDATE_SCRIPT} >> /home/barberb/ipfs_accelerate_py/logs/auto-update-cron.log 2>&1" >> /tmp/current_cron

# Install the new crontab
crontab /tmp/current_cron
rm /tmp/current_cron

echo "Cron job installed successfully!"
echo "The auto-update script will run every 6 hours."
echo ""
echo "Current crontab:"
crontab -l

exit 0
