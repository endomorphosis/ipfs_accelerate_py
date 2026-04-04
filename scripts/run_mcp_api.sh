#!/usr/bin/env bash
set -euo pipefail

# Defaults
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
LOG=${LOG:-mcp_server.out}

# Activate venv if present
if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# Environment guards to avoid heavy imports by default; set FULL_IPFS=1 to enable full features
if [ "${FULL_IPFS:-0}" = "1" ]; then
  export IPFS_ACCEL_SKIP_CORE=${IPFS_ACCEL_SKIP_CORE:-0}
  export MCP_DISABLE_IPFS=${MCP_DISABLE_IPFS:-0}
else
  export IPFS_ACCEL_SKIP_CORE=${IPFS_ACCEL_SKIP_CORE:-1}
  export MCP_DISABLE_IPFS=${MCP_DISABLE_IPFS:-1}
fi
export MCP_CORS_ORIGINS=${MCP_CORS_ORIGINS:-*}

echo "Starting MCP API on ${HOST}:${PORT} (CORS: ${MCP_CORS_ORIGINS})..."
nohup python -m ipfs_accelerate_py.mcp.server --host "${HOST}" --port "${PORT}" --debug >"${LOG}" 2>&1 &
echo $! > mcp_server.pid
sleep 1
echo "Logs: ${LOG} | PID: $(cat mcp_server.pid)"
