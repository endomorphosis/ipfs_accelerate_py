#!/bin/bash
# P2P Peer Bootstrap Script for GitHub Actions
# This script helps runners discover each other for P2P cache sharing

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PEER_INFO_DIR="${RUNNER_TEMP:-/tmp}/p2p_peers"
REPO="${GITHUB_REPOSITORY:-unknown/repo}"
RUN_ID="${GITHUB_RUN_ID:-unknown}"

# Create peer info directory
mkdir -p "$PEER_INFO_DIR"

# Function to register this peer
register_peer() {
    local peer_id="$1"
    local listen_port="$2"
    local public_ip="$3"
    
    if [ -z "$peer_id" ] || [ -z "$listen_port" ]; then
        echo "❌ Missing required parameters: peer_id or listen_port"
        return 1
    fi
    
    # Create peer info file
    local peer_file="$PEER_INFO_DIR/${RUNNER_NAME:-runner}.json"
    
    cat > "$peer_file" <<EOF
{
  "peer_id": "$peer_id",
  "runner_name": "${RUNNER_NAME:-unknown}",
  "public_ip": "$public_ip",
  "listen_port": $listen_port,
  "multiaddr": "/ip4/$public_ip/tcp/$listen_port/p2p/$peer_id",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "run_id": "$RUN_ID"
}
EOF
    
    echo "✓ Created peer info: $peer_file"
    
    # Upload as artifact if gh CLI is available
    if command -v gh &> /dev/null; then
        # Use GitHub Actions artifact upload action instead
        echo "peer_file=$peer_file" >> $GITHUB_OUTPUT
        echo "peer_multiaddr=/ip4/$public_ip/tcp/$listen_port/p2p/$peer_id" >> $GITHUB_OUTPUT
    fi
}

# Function to discover peers from artifacts
discover_peers() {
    local max_peers="${1:-10}"
    
    echo "🔍 Discovering P2P peers..."
    
    # List peer files in the directory
    if [ -d "$PEER_INFO_DIR" ]; then
        local count=0
        local bootstrap_peers=""
        
        for peer_file in "$PEER_INFO_DIR"/*.json; do
            if [ -f "$peer_file" ] && [ "$count" -lt "$max_peers" ]; then
                # Extract multiaddr from peer file
                if command -v jq &> /dev/null; then
                    local multiaddr=$(jq -r '.multiaddr' "$peer_file" 2>/dev/null || echo "")
                    if [ -n "$multiaddr" ] && [ "$multiaddr" != "null" ]; then
                        if [ -n "$bootstrap_peers" ]; then
                            bootstrap_peers="$bootstrap_peers,$multiaddr"
                        else
                            bootstrap_peers="$multiaddr"
                        fi
                        count=$((count + 1))
                        echo "  ✓ Found peer: $multiaddr"
                    fi
                fi
            fi
        done
        
        if [ -n "$bootstrap_peers" ]; then
            echo "✓ Discovered $count peer(s)"
            echo "bootstrap_peers=$bootstrap_peers" >> ${GITHUB_OUTPUT:-/dev/null}
            export CACHE_BOOTSTRAP_PEERS="$bootstrap_peers"
            echo "CACHE_BOOTSTRAP_PEERS=$bootstrap_peers" >> $GITHUB_ENV
        else
            echo "⚠ No peers discovered"
        fi
    else
        echo "⚠ Peer directory not found: $PEER_INFO_DIR"
    fi
}

# Function to get public IP
get_public_ip() {
    local ip=""
    
    # Try multiple services
    for service in "https://api.ipify.org" "https://ifconfig.me/ip" "https://icanhazip.com"; do
        ip=$(curl -s --max-time 5 "$service" 2>/dev/null | tr -d '[:space:]')
        if [ -n "$ip" ]; then
            echo "$ip"
            return 0
        fi
    done
    
    # Fallback to local IP
    ip=$(hostname -I 2>/dev/null | awk '{print $1}')
    if [ -n "$ip" ]; then
        echo "$ip"
        return 0
    fi
    
    # Last resort
    echo "127.0.0.1"
}

# Function to initialize P2P cache
init_p2p_cache() {
    local listen_port="${CACHE_LISTEN_PORT:-9000}"
    
    echo "🚀 Initializing P2P cache..."
    
    # Set P2P environment variables
    export CACHE_ENABLE_P2P=true
    export CACHE_LISTEN_PORT="$listen_port"
    
    echo "CACHE_ENABLE_P2P=true" >> $GITHUB_ENV
    echo "CACHE_LISTEN_PORT=$listen_port" >> $GITHUB_ENV
    
    # Get public IP
    local public_ip=$(get_public_ip)
    echo "  Public IP: $public_ip"
    echo "  Listen port: $listen_port"
    
    # Try to get peer ID from Python if available
    if command -v python3 &> /dev/null; then
        echo "  Attempting to get peer ID from cache initialization..."
        
        # This will initialize the cache and print peer info
        python3 -c "
import sys
import os
sys.path.insert(0, '${GITHUB_WORKSPACE:-/workspace}')
try:
    os.environ['CACHE_ENABLE_P2P'] = 'true'
    os.environ['CACHE_LISTEN_PORT'] = '${listen_port}'
    from ipfs_accelerate_py.github_cli.cache import get_global_cache
    cache = get_global_cache()
    # Give it time to initialize
    import time
    time.sleep(2)
    stats = cache.get_stats()
    if 'peer_id' in stats:
        print(f'PEER_ID={stats[\"peer_id\"]}')
        print(f'MULTIADDR=/ip4/${public_ip}/tcp/${listen_port}/p2p/{stats[\"peer_id\"]}')
except Exception as e:
    print(f'Warning: Could not initialize P2P cache: {e}', file=sys.stderr)
" 2>&1 | tee /tmp/p2p_init.log
        
        # Parse output
        if [ -f /tmp/p2p_init.log ]; then
            local peer_id=$(grep "PEER_ID=" /tmp/p2p_init.log | cut -d= -f2)
            if [ -n "$peer_id" ]; then
                echo "  ✓ Peer ID: $peer_id"
                register_peer "$peer_id" "$listen_port" "$public_ip"
            fi
        fi
    fi
    
    echo "✓ P2P cache initialized"
}

# Main command dispatcher
case "${1:-help}" in
    register)
        register_peer "$2" "$3" "$4"
        ;;
    discover)
        discover_peers "$2"
        ;;
    init)
        init_p2p_cache
        ;;
    help|*)
        cat <<EOF
P2P Peer Bootstrap Script

Usage:
  $0 init                              Initialize P2P cache on this runner
  $0 register <peer_id> <port> <ip>   Register this runner as a peer
  $0 discover [max_peers]              Discover available peers
  $0 help                              Show this help

Environment Variables:
  CACHE_LISTEN_PORT       Port for P2P cache (default: 9000)
  CACHE_BOOTSTRAP_PEERS   Comma-separated list of peer multiaddrs
  GITHUB_REPOSITORY       Repository name (auto-detected)
  RUNNER_NAME             Runner name (auto-detected)

Examples:
  # Initialize P2P cache
  $0 init
  
  # Discover peers
  $0 discover 5
  
  # Register manually
  $0 register QmPeer123... 9000 192.168.1.100
EOF
        ;;
esac
