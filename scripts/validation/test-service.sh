#!/bin/bash
#
# IPFS Accelerate Service Test
# This script tests the installed service
#

echo "=========================================="
echo "IPFS Accelerate Service Test"
echo "=========================================="
echo

# Test 1: Check if service is enabled
echo "1. Checking if service is enabled for startup..."
if sudo systemctl is-enabled ipfs-accelerate &>/dev/null; then
    echo "   ✓ Service is enabled for startup"
else
    echo "   ✗ Service is NOT enabled for startup"
fi

# Test 2: Check if service is running
echo
echo "2. Checking if service is running..."
if sudo systemctl is-active ipfs-accelerate &>/dev/null; then
    echo "   ✓ Service is running"
else
    echo "   ✗ Service is NOT running"
fi

# Test 3: Check if cron job exists
echo
echo "3. Checking if monitoring cron job exists..."
if crontab -l 2>/dev/null | grep -q "check-service.sh"; then
    echo "   ✓ Monitoring cron job is installed"
else
    echo "   ✗ Monitoring cron job is NOT installed"
fi

# Test 4: Check if health endpoint responds
echo
echo "4. Testing health endpoint..."
if timeout 10s curl -s http://localhost:9000/health >/dev/null 2>&1; then
    echo "   ✓ Health endpoint is responding"
    
    # Show health status
    echo "   Health status:"
    curl -s http://localhost:9000/health | python -m json.tool | sed 's/^/   /'
else
    echo "   ✗ Health endpoint is NOT responding"
fi

# Test 5: Check dashboard accessibility
echo
echo "5. Testing dashboard endpoint..."
if timeout 10s curl -s http://localhost:9000/dashboard >/dev/null 2>&1; then
    echo "   ✓ Dashboard endpoint is responding"
else
    echo "   ✗ Dashboard endpoint is NOT responding"
fi

# Show service status
echo
echo "6. Current service status:"
sudo systemctl status ipfs-accelerate --no-pager -l | sed 's/^/   /'

echo
echo "=========================================="
echo "Test Complete!"
echo "=========================================="
echo
echo "Access points:"
echo "  Dashboard: http://localhost:9000/dashboard"
echo "  Health:    http://localhost:9000/health"
echo "  API:       http://localhost:9000/api/"
echo
echo "Management commands:"
echo "  sudo systemctl status ipfs-accelerate"
echo "  sudo systemctl stop ipfs-accelerate"
echo "  sudo systemctl start ipfs-accelerate"
echo "  sudo systemctl restart ipfs-accelerate"
echo "  sudo journalctl -u ipfs-accelerate -f"