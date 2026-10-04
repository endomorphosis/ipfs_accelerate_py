#!/usr/bin/env bash
#
# High-Availability Cluster Setup Script
# This script sets up and demonstrates a 3-node coordinator cluster with redundancy
# for the Distributed Testing Framework.
#
# Usage: ./high_availability_cluster.sh [start|stop|status|test|cleanup]
#

set -e

# Configuration
BASE_DIR=$(pwd)/ha_cluster
NODE_COUNT=3
BASE_PORT=8080
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Display banner
function show_banner() {
    echo -e "${BLUE}======================================================${NC}"
    echo -e "${BLUE}  Distributed Testing Framework - HA Cluster Example  ${NC}"
    echo -e "${BLUE}======================================================${NC}"
    echo ""
}

# Create directory structure
function create_directories() {
    echo -e "${YELLOW}Creating directory structure...${NC}"
    
    mkdir -p "$BASE_DIR"
    
    for i in $(seq 1 $NODE_COUNT); do
        mkdir -p "$BASE_DIR/node$i"
        echo -e "${GREEN}Created directory for node $i: $BASE_DIR/node$i${NC}"
    done
    
    echo ""
}

# Create configuration files
function create_configs() {
    echo -e "${YELLOW}Creating configuration files...${NC}"
    
    for i in $(seq 1 $NODE_COUNT); do
        NODE_ID="node-$i"
        PORT=$((BASE_PORT + i - 1))
        
        # Build peers list
        PEERS="["
        for j in $(seq 1 $NODE_COUNT); do
            if [ "$j" -ne "$i" ]; then
                PEER_PORT=$((BASE_PORT + j - 1))
                if [ "$PEERS" != "[" ]; then
                    PEERS="$PEERS,"
                fi
                PEERS="$PEERS{\"id\":\"node-$j\",\"host\":\"localhost\",\"port\":$PEER_PORT}"
            fi
        done
        PEERS="$PEERS]"
        
        # Create config file
        cat > "$BASE_DIR/node$i/config.json" <<EOF
{
    "node_id": "$NODE_ID",
    "host": "localhost",
    "port": $PORT,
    "data_dir": "$BASE_DIR/node$i",
    "db_path": "$BASE_DIR/node$i/coordinator.duckdb",
    "enable_redundancy": true,
    "peers": $PEERS,
    "election_timeout_min": 150,
    "election_timeout_max": 300,
    "heartbeat_interval": 50,
    "log_level": "INFO"
}
EOF
        
        echo -e "${GREEN}Created configuration for node $i${NC}"
    done
    
    echo ""
}

# Start the cluster
function start_cluster() {
    echo -e "${YELLOW}Starting coordinator cluster...${NC}"
    
    for i in $(seq 1 $NODE_COUNT); do
        NODE_LOG="$BASE_DIR/node$i/coordinator.log"
        CONFIG="$BASE_DIR/node$i/config.json"
        
        echo -e "${BLUE}Starting node $i...${NC}"
        
        # Start the coordinator as a background process
        python -m distributed_testing.coordinator --config "$CONFIG" > "$NODE_LOG" 2>&1 &
        
        # Save the PID
        echo $! > "$BASE_DIR/node$i/coordinator.pid"
        
        echo -e "${GREEN}Node $i started with PID $(cat "$BASE_DIR/node$i/coordinator.pid")${NC}"
        echo -e "${GREEN}Log file: $NODE_LOG${NC}"
    done
    
    # Wait for cluster to stabilize
    echo -e "${YELLOW}Waiting for cluster to stabilize...${NC}"
    sleep 5
    
    echo -e "${GREEN}Coordinator cluster started successfully${NC}"
    echo ""
}

# Stop the cluster
function stop_cluster() {
    echo -e "${YELLOW}Stopping coordinator cluster...${NC}"
    
    for i in $(seq 1 $NODE_COUNT); do
        PID_FILE="$BASE_DIR/node$i/coordinator.pid"
        
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            
            echo -e "${BLUE}Stopping node $i (PID: $PID)...${NC}"
            
            # Check if process exists
            if kill -0 $PID 2>/dev/null; then
                # Stop the process
                kill $PID
                
                # Wait for it to exit
                for j in $(seq 1 10); do
                    if ! kill -0 $PID 2>/dev/null; then
                        break
                    fi
                    sleep 1
                done
                
                # Force kill if necessary
                if kill -0 $PID 2>/dev/null; then
                    echo -e "${RED}Node $i did not exit gracefully, force killing...${NC}"
                    kill -9 $PID
                fi
            else
                echo -e "${YELLOW}Node $i (PID: $PID) is not running${NC}"
            fi
            
            # Remove PID file
            rm "$PID_FILE"
            
            echo -e "${GREEN}Node $i stopped${NC}"
        else
            echo -e "${YELLOW}Node $i is not running (no PID file)${NC}"
        fi
    done
    
    echo -e "${GREEN}Coordinator cluster stopped successfully${NC}"
    echo ""
}

# Get cluster status
function cluster_status() {
    echo -e "${YELLOW}Checking cluster status...${NC}"
    
    # Check if nodes are running
    for i in $(seq 1 $NODE_COUNT); do
        PID_FILE="$BASE_DIR/node$i/coordinator.pid"
        
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            
            if kill -0 $PID 2>/dev/null; then
                echo -e "${GREEN}Node $i is running (PID: $PID)${NC}"
                
                # Get node role
                PORT=$((BASE_PORT + i - 1))
                ROLE=$(curl -s http://localhost:$PORT/api/status | grep -o '"role":"[^"]*"' | cut -d'"' -f4)
                TERM=$(curl -s http://localhost:$PORT/api/status | grep -o '"term":[0-9]*' | cut -d':' -f2)
                LEADER=$(curl -s http://localhost:$PORT/api/status | grep -o '"current_leader":"[^"]*"' | cut -d'"' -f4)
                
                echo -e "  Role: ${BLUE}$ROLE${NC}, Term: ${BLUE}$TERM${NC}, Leader: ${BLUE}$LEADER${NC}"
            else
                echo -e "${RED}Node $i (PID: $PID) is not running${NC}"
            fi
        else
            echo -e "${RED}Node $i is not running (no PID file)${NC}"
        fi
    done
    
    echo ""
}

# Test failover
function test_failover() {
    echo -e "${YELLOW}Testing leader failover...${NC}"
    
    # Find the current leader
    LEADER_NODE=""
    LEADER_PID=""
    
    for i in $(seq 1 $NODE_COUNT); do
        PORT=$((BASE_PORT + i - 1))
        ROLE=$(curl -s http://localhost:$PORT/api/status | grep -o '"role":"[^"]*"' | cut -d'"' -f4)
        
        if [ "$ROLE" == "LEADER" ]; then
            LEADER_NODE=$i
            LEADER_PID=$(cat "$BASE_DIR/node$i/coordinator.pid")
            break
        fi
    done
    
    if [ -z "$LEADER_NODE" ]; then
        echo -e "${RED}No leader found in the cluster${NC}"
        return 1
    fi
    
    echo -e "${BLUE}Current leader is node $LEADER_NODE (PID: $LEADER_PID)${NC}"
    
    # Register a test worker
    echo -e "${YELLOW}Registering test worker...${NC}"
    
    PORT=$((BASE_PORT + LEADER_NODE - 1))
    curl -s -X POST http://localhost:$PORT/api/workers/register \
        -H "Content-Type: application/json" \
        -d '{"worker_id":"test-worker","host":"test-host","port":9000}' > /dev/null
    
    echo -e "${GREEN}Test worker registered through leader${NC}"
    
    # Kill the leader
    echo -e "${YELLOW}Killing leader (node $LEADER_NODE)...${NC}"
    kill $LEADER_PID
    rm "$BASE_DIR/node$LEADER_NODE/coordinator.pid"
    
    echo -e "${GREEN}Leader killed, waiting for new leader election...${NC}"
    sleep 5
    
    # Find the new leader
    NEW_LEADER_NODE=""
    
    for i in $(seq 1 $NODE_COUNT); do
        if [ "$i" -ne "$LEADER_NODE" ]; then
            PORT=$((BASE_PORT + i - 1))
            ROLE=$(curl -s http://localhost:$PORT/api/status | grep -o '"role":"[^"]*"' | cut -d'"' -f4)
            
            if [ "$ROLE" == "LEADER" ]; then
                NEW_LEADER_NODE=$i
                break
            fi
        fi
    done
    
    if [ -z "$NEW_LEADER_NODE" ]; then
        echo -e "${RED}No new leader elected after failure${NC}"
        return 1
    fi
    
    echo -e "${GREEN}New leader elected: node $NEW_LEADER_NODE${NC}"
    
    # Verify state replication
    echo -e "${YELLOW}Verifying state replication...${NC}"
    
    PORT=$((BASE_PORT + NEW_LEADER_NODE - 1))
    WORKER_COUNT=$(curl -s http://localhost:$PORT/api/workers | grep -o '"test-worker"' | wc -l)
    
    if [ "$WORKER_COUNT" -eq "1" ]; then
        echo -e "${GREEN}State successfully replicated - test worker found in new leader's state${NC}"
    else
        echo -e "${RED}State replication failed - test worker not found in new leader's state${NC}"
        return 1
    fi
    
    # Restart the killed node
    echo -e "${YELLOW}Restarting killed node $LEADER_NODE...${NC}"
    
    NODE_LOG="$BASE_DIR/node$LEADER_NODE/coordinator.log"
    CONFIG="$BASE_DIR/node$LEADER_NODE/config.json"
    
    python -m distributed_testing.coordinator --config "$CONFIG" >> "$NODE_LOG" 2>&1 &
    echo $! > "$BASE_DIR/node$LEADER_NODE/coordinator.pid"
    
    echo -e "${GREEN}Node $LEADER_NODE restarted with PID $(cat "$BASE_DIR/node$LEADER_NODE/coordinator.pid")${NC}"
    
    # Wait for the node to sync
    echo -e "${YELLOW}Waiting for restarted node to sync...${NC}"
    sleep 5
    
    # Verify the restarted node has the worker
    PORT=$((BASE_PORT + LEADER_NODE - 1))
    WORKER_COUNT=$(curl -s http://localhost:$PORT/api/workers | grep -o '"test-worker"' | wc -l)
    
    if [ "$WORKER_COUNT" -eq "1" ]; then
        echo -e "${GREEN}Restarted node successfully synced state - test worker found${NC}"
    else
        echo -e "${RED}Restarted node failed to sync state - test worker not found${NC}"
        return 1
    fi
    
    echo -e "${GREEN}Failover test completed successfully${NC}"
    echo ""
}

# Cleanup files
function cleanup() {
    echo -e "${YELLOW}Cleaning up...${NC}"
    
    # Stop cluster if running
    stop_cluster
    
    # Remove the base directory
    rm -rf "$BASE_DIR"
    
    echo -e "${GREEN}Cleanup completed successfully${NC}"
    echo ""
}

# Main function
function main() {
    show_banner
    
    ACTION="${1:-start}"
    
    case "$ACTION" in
        start)
            create_directories
            create_configs
            start_cluster
            cluster_status
            ;;
        stop)
            stop_cluster
            ;;
        status)
            cluster_status
            ;;
        test)
            test_failover
            ;;
        cleanup)
            cleanup
            ;;
        *)
            echo -e "${RED}Unknown action: $ACTION${NC}"
            echo -e "Usage: $0 [start|stop|status|test|cleanup]"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"