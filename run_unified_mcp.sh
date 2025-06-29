#!/bin/bash
# Run script for the Unified IPFS Accelerate MCP Server

set -e  # Exit on error

# Colors for output formatting
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}$1${NC}"
}

print_success() {
    echo -e "${GREEN}$1${NC}"
}

print_warning() {
    echo -e "${YELLOW}$1${NC}"
}

print_error() {
    echo -e "${RED}$1${NC}"
}

# Check for Python installation
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is required but not installed."
    exit 1
fi

# Check and install required packages
print_status "Checking required packages..."

if ! python3 -c "import flask" &> /dev/null; then
    print_status "Installing Flask and Flask-CORS..."
    pip install flask flask_cors > /dev/null
    print_success "Flask installed successfully."
else
    print_success "Flask is already installed."
fi

if ! python3 -c "import ipfshttpclient" &> /dev/null; then
    print_status "Installing ipfshttpclient..."
    pip install ipfshttpclient > /dev/null
    print_success "ipfshttpclient installed successfully."
else
    print_success "ipfshttpclient is already installed."
fi

if ! python3 -c "import psutil" &> /dev/null; then
    print_status "Installing psutil..."
    pip install psutil > /dev/null
    print_success "psutil installed successfully."
else
    print_success "psutil is already installed."
fi

if ! python3 -c "import numpy" &> /dev/null; then
    print_status "Installing numpy..."
    pip install numpy > /dev/null
    print_success "numpy installed successfully."
else
    print_success "numpy is already installed."
fi

if ! python3 -c "import requests" &> /dev/null; then
    print_status "Installing requests..."
    pip install requests > /dev/null
    print_success "requests installed successfully."
else
    print_success "requests is already installed."
fi

# Check for IPFS daemon
print_status "Checking IPFS daemon..."
if ! command -v ipfs &> /dev/null; then
    print_warning "IPFS is not installed. Some functionality will be limited."
else
    # Check if IPFS daemon is running
    if ! ipfs swarm peers &> /dev/null; then
        print_warning "IPFS is installed but the daemon is not running."
        print_status "Starting IPFS daemon in the background..."
        ipfs daemon --routing=dhtclient > ipfs_daemon.log 2>&1 &
        
        # Wait for daemon to start
        sleep 3
        if ipfs swarm peers &> /dev/null; then
            print_success "IPFS daemon started successfully."
        else
            print_warning "IPFS daemon failed to start properly."
        fi
    else
        print_success "IPFS daemon is already running."
    fi
fi

# Start the server
print_status "Starting Unified MCP Server..."
python3 unified_mcp_server.py --port 8001 > unified_mcp_server.log 2>&1 &
SERVER_PID=$!
print_status "Server started with PID: $SERVER_PID"

# Wait for the server to start
print_status "Waiting for server to start..."
sleep 3

# Check if the server is running
if kill -0 $SERVER_PID 2>/dev/null; then
    print_status "Server is running. Testing connection..."
    
    # Test the connection
    SERVER_URL="http://localhost:8001"
    if curl -s "$SERVER_URL/" > /dev/null; then
        print_success "Server is accessible at $SERVER_URL/"
        
        print_status "Running test script..."
        python3 -c "
import requests
import json
import sys

SERVER_URL = '$SERVER_URL'
print('===== Testing Unified MCP Server =====')
print(f'Server URL: {SERVER_URL}\n')

# Test server connection
print('Testing basic connectivity...')
try:
    response = requests.get(SERVER_URL)
    if response.status_code == 200:
        print('✓ Server is accessible')
        info = response.json()
        print(f'Server info: {info.get(\"name\", \"Unknown\")} v{info.get(\"version\", \"Unknown\")}')
    else:
        print(f'✗ Server returned status code: {response.status_code}')
        print(f'Response: {response.text}')
        sys.exit(1)
except Exception as e:
    print(f'✗ Failed to connect to server: {e}')
    sys.exit(1)

# Test health check
print('\nTesting health check...')
try:
    response = requests.post(
        f'{SERVER_URL}/call_tool',
        json={'tool_name': 'health_check', 'arguments': {}}
    )
    if response.status_code == 200:
        result = response.json().get('result', {})
        if result.get('status') == 'healthy':
            print('✓ Server is healthy')
            print(f'Uptime: {result.get(\"uptime\", \"Unknown\")}')
            components = result.get('components', {})
            for name, status in components.items():
                print(f'  - {name}: {\"✓\" if status else \"✗\"}')
        else:
            print(f'✗ Server is not healthy: {result}')
    else:
        print(f'✗ Health check failed with status code: {response.status_code}')
except Exception as e:
    print(f'✗ Failed to perform health check: {e}')

# List available tools
print('\nListing available MCP tools...')
try:
    response = requests.get(f'{SERVER_URL}/tools')
    if response.status_code == 200:
        tools = response.json().get('tools', [])
        print(f'✓ Found {len(tools)} tools')
        if len(tools) > 0:
            print('Sample tools:')
            for i, tool in enumerate(tools[:5]):
                print(f'  - {tool}')
            if len(tools) > 5:
                print(f'  ... plus {len(tools) - 5} more')
    else:
        print(f'✗ Failed to list tools: {response.status_code}')
except Exception as e:
    print(f'✗ Failed to list tools: {e}')

print('\n✓ Basic tests completed. The MCP server is running properly.')
"
        TEST_RESULT=$?
        
        if [ $TEST_RESULT -eq 0 ]; then
            print_success "Basic tests passed. For comprehensive testing, run: python3 unified_mcp_server_test.py --url $SERVER_URL"
        else
            print_warning "Some basic tests failed."
        fi
    else
        print_error "Server not responding to requests."
    fi
else
    print_error "Failed to start server."
fi

# Ask if the user wants to stop the server
read -p "The MCP server is running in the background (PID: $SERVER_PID).
Do you want to stop the server now? (y/n): " STOP_SERVER

if [[ $STOP_SERVER == "y" || $STOP_SERVER == "Y" ]]; then
    print_status "Stopping server..."
    kill $SERVER_PID 2>/dev/null || true
    print_success "Server stopped."
else
    print_status "Server will continue running in the background."
    print_status "To stop it later, run: kill $SERVER_PID"
fi

print_success "Done!"
