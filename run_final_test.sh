#!/bin/bash

# Color definitions
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=========================================================${NC}"
echo -e "${BLUE}    IPFS Accelerate MCP - Final Solution Test Script     ${NC}"
echo -e "${BLUE}=========================================================${NC}"

cd /home/barberb/ipfs_accelerate_py

# Stop any existing processes
pkill -f final_mcp_server.py || true
sleep 2

# Activate virtual environment and start server
echo -e "${YELLOW}Starting MCP server with virtual environment...${NC}"
source ipfs_env/bin/activate

# Start server in background
python3 final_mcp_server.py --host 127.0.0.1 --port 8002 &> server_startup.log &
SERVER_PID=$!
echo -e "${GREEN}Server started with PID ${SERVER_PID}${NC}"

# Wait for server to start
echo -e "${YELLOW}Waiting for server to start...${NC}"
sleep 15

# Check if server is running
if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo -e "${RED}Server failed to start. Checking logs...${NC}"
    cat server_startup.log
    exit 1
fi

# Test server is responding
echo -e "${YELLOW}Testing server is responding...${NC}"
if curl -s http://127.0.0.1:8002/health > /dev/null; then
    echo -e "${GREEN}Server is responding!${NC}"
else
    echo -e "${RED}Server is not responding${NC}"
fi

# Run a quick test
echo -e "${YELLOW}Running quick test...${NC}"
python3 -c "
import requests
import json

base_url = 'http://127.0.0.1:8002'

# Test health endpoint
print('Testing health endpoint...')
try:
    response = requests.get(f'{base_url}/health')
    print(f'Health: {response.status_code} - {response.text}')
except Exception as e:
    print(f'Health test failed: {e}')

# Test tools endpoint
print('Testing tools endpoint...')
try:
    response = requests.get(f'{base_url}/tools')
    print(f'Tools: {response.status_code}')
    if response.status_code == 200:
        tools = response.json()
        print(f'Found {len(tools)} tools: {tools[:5]}...')
except Exception as e:
    print(f'Tools test failed: {e}')

# Test a specific tool
print('Testing ipfs_node_info tool...')
try:
    response = requests.post(f'{base_url}/tools/ipfs_node_info', json={})
    print(f'ipfs_node_info: {response.status_code} - {response.text[:200]}...')
except Exception as e:
    print(f'Tool test failed: {e}')
"

# Stop server
echo -e "${YELLOW}Stopping server...${NC}"
kill $SERVER_PID || true
sleep 2

echo -e "${GREEN}Test completed!${NC}"
