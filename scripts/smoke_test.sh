#!/usr/bin/env bash
set -euo pipefail

HOST=${HOST:-127.0.0.1}
API_PORT=${API_PORT:-8000}
DASH_PORT=${DASH_PORT:-8080}

echo "Testing MCP API on http://${HOST}:${API_PORT}"
curl -sf "http://${HOST}:${API_PORT}/mcp/resource/ipfs_accelerate/supported_models" | head -c 200 || { echo "API resource check failed"; exit 1; }
curl -sf -X POST "http://${HOST}:${API_PORT}/mcp/tool/get_hardware_info" -H 'Content-Type: application/json' -d '{}' | head -c 200 || { echo "API tool check failed"; exit 1; }

echo "Testing Dashboard on http://${HOST}:${DASH_PORT}"
curl -sf "http://${HOST}:${DASH_PORT}/api/status" | head -c 200 || { echo "Dashboard status check failed"; exit 1; }
curl -sf "http://${HOST}:${DASH_PORT}/api/hardware_info" | head -c 200 || { echo "Dashboard hardware check failed"; exit 1; }

echo "Smoke test passed"
