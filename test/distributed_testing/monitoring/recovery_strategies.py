#!/usr/bin/env python3
"""
Advanced Recovery Strategies for Distributed Testing Framework.
Implements recovery mechanisms for various edge case failures in coordinator redundancy.
"""

import asyncio
import os
import sys
import time
import json
import argparse
import logging
import aiohttp
import subprocess
import signal
import psutil
import shutil
import datetime
import hashlib
from enum import Enum, auto

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Types of failures that can occur in the coordinator cluster."""
    PROCESS_CRASH = auto()
    NETWORK_PARTITION = auto()
    DATABASE_CORRUPTION = auto()
    LOG_CORRUPTION = auto()
    SPLIT_BRAIN = auto()
    TERM_DIVERGENCE = auto()
    STATE_DIVERGENCE = auto()
    DEADLOCK = auto()
    RESOURCE_EXHAUSTION = auto()
    SLOW_FOLLOWER = auto()


class RecoveryStrategy:
    """Base class for recovery strategies."""
    
    def __init__(self, cluster_config, data_dir="/tmp/distributed_testing_recovery"):
        """Initialize the recovery strategy."""
        self.cluster_config = cluster_config
        self.data_dir = data_dir
        self.recovery_log = []
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
    async def detect_failures(self):
        """Detect failures in the cluster."""
        raise NotImplementedError("Subclasses must implement detect_failures")
        
    async def recover(self, failure_type, affected_nodes):
        """Recover from a failure."""
        raise NotImplementedError("Subclasses must implement recover")
        
    def log_recovery_action(self, action, details):
        """Log a recovery action."""
        timestamp = datetime.datetime.now().isoformat()
        self.recovery_log.append({
            "timestamp": timestamp,
            "action": action,
            "details": details
        })
        
        logger.info(f"Recovery action: {action} - {details}")
        
    async def save_recovery_log(self):
        """Save the recovery log to a file."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.data_dir, f"recovery_log_{timestamp}.json")
        
        with open(filename, "w") as f:
            json.dump(self.recovery_log, f, indent=2)
            
        logger.info(f"Recovery log saved to {filename}")
        return filename


class CoordinatorRecoveryStrategy(RecoveryStrategy):
    """Recovery strategy for coordinator redundancy failures."""
    
    def __init__(self, cluster_config, data_dir="/tmp/distributed_testing_recovery"):
        """Initialize the recovery strategy."""
        super().__init__(cluster_config, data_dir)
        
        # Map of process IDs to nodes
        self.processes = {}
        
        # Map of node IDs to status
        self.node_status = {}
        
        # Cache of log entries for verification
        self.log_cache = {}
        
    async def detect_failures(self):
        """Detect failures in the cluster."""
        logger.info("Detecting failures in the coordinator cluster")
        
        failures = []
        
        # Check each node's status
        for node in self.cluster_config["nodes"]:
            node_id = node["id"]
            host = node["host"]
            port = node["port"]
            
            # Check if the process is running
            if node_id in self.processes:
                process = self.processes[node_id]
                if not psutil.pid_exists(process.pid):
                    failures.append({
                        "type": FailureType.PROCESS_CRASH,
                        "node_id": node_id,
                        "details": f"Process with PID {process.pid} is not running"
                    })
                    
            # Check if the node is responding to API requests
            try:
                url = f"http://{host}:{port}/api/status"
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=2) as response:
                        if response.status == 200:
                            status = await response.json()
                            self.node_status[node_id] = status
                        else:
                            failures.append({
                                "type": FailureType.NETWORK_PARTITION,
                                "node_id": node_id,
                                "details": f"Node returned HTTP {response.status}"
                            })
            except asyncio.TimeoutError:
                failures.append({
                    "type": FailureType.NETWORK_PARTITION,
                    "node_id": node_id,
                    "details": "Request timed out"
                })
            except Exception as e:
                failures.append({
                    "type": FailureType.NETWORK_PARTITION,
                    "node_id": node_id,
                    "details": f"Error connecting to node: {e}"
                })
                
        # Check for database corruption
        await self._detect_database_corruption(failures)
        
        # Check for log corruption
        await self._detect_log_corruption(failures)
        
        # Check for split brain
        await self._detect_split_brain(failures)
        
        # Check for term divergence
        await self._detect_term_divergence(failures)
        
        # Check for state divergence
        await self._detect_state_divergence(failures)
        
        # Check for deadlock
        await self._detect_deadlock(failures)
        
        # Check for resource exhaustion
        await self._detect_resource_exhaustion(failures)
        
        # Check for slow followers
        await self._detect_slow_followers(failures)
        
        return failures
    
    async def _detect_database_corruption(self, failures):
        """Detect database corruption."""
        for node in self.cluster_config["nodes"]:
            node_id = node["id"]
            host = node["host"]
            port = node["port"]
            
            try:
                url = f"http://{host}:{port}/api/health/db"
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=5) as response:
                        if response.status == 200:
                            health = await response.json()
                            if not health.get("healthy", True):
                                failures.append({
                                    "type": FailureType.DATABASE_CORRUPTION,
                                    "node_id": node_id,
                                    "details": health.get("details", "Database health check failed")
                                })
            except Exception:
                # Skip if we can't connect - this will be caught elsewhere
                pass
                
    async def _detect_log_corruption(self, failures):
        """Detect log corruption."""
        for node in self.cluster_config["nodes"]:
            node_id = node["id"]
            host = node["host"]
            port = node["port"]
            
            try:
                url = f"http://{host}:{port}/api/health/log"
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=5) as response:
                        if response.status == 200:
                            health = await response.json()
                            if not health.get("healthy", True):
                                failures.append({
                                    "type": FailureType.LOG_CORRUPTION,
                                    "node_id": node_id,
                                    "details": health.get("details", "Log health check failed")
                                })
            except Exception:
                # Skip if we can't connect - this will be caught elsewhere
                pass
                
    async def _detect_split_brain(self, failures):
        """Detect split brain condition (multiple leaders)."""
        # Find all nodes that think they are the leader
        leaders = []
        for node_id, status in self.node_status.items():
            if status.get("role") == "LEADER":
                leaders.append({
                    "node_id": node_id,
                    "term": status.get("term", 0)
                })
                
        # If we have multiple leaders with the same term, we have a split brain
        if len(leaders) > 1:
            terms = set(leader["term"] for leader in leaders)
            if len(terms) == 1:
                failures.append({
                    "type": FailureType.SPLIT_BRAIN,
                    "node_id": [leader["node_id"] for leader in leaders],
                    "details": f"Multiple leaders detected with term {list(terms)[0]}: {[leader['node_id'] for leader in leaders]}"
                })
                
    async def _detect_term_divergence(self, failures):
        """Detect term divergence."""
        # Get all unique terms
        terms = set()
        for node_id, status in self.node_status.items():
            terms.add(status.get("term", 0))
            
        # If we have more than 2 unique terms, we might have term divergence
        # It's normal to have at most 2 terms during a leader election
        if len(terms) > 2:
            failures.append({
                "type": FailureType.TERM_DIVERGENCE,
                "node_id": list(self.node_status.keys()),
                "details": f"Term divergence detected: {list(terms)}"
            })
            
    async def _detect_state_divergence(self, failures):
        """Detect state divergence."""
        # Get state checksums from all nodes
        state_checksums = {}
        for node in self.cluster_config["nodes"]:
            node_id = node["id"]
            host = node["host"]
            port = node["port"]
            
            try:
                url = f"http://{host}:{port}/api/health/state_checksum"
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=5) as response:
                        if response.status == 200:
                            checksum = await response.json()
                            state_checksums[node_id] = checksum.get("checksum")
            except Exception:
                # Skip if we can't connect - this will be caught elsewhere
                pass
                
        # If we have different checksums, we have state divergence
        if len(set(state_checksums.values())) > 1:
            failures.append({
                "type": FailureType.STATE_DIVERGENCE,
                "node_id": list(state_checksums.keys()),
                "details": f"State divergence detected: {state_checksums}"
            })
            
    async def _detect_deadlock(self, failures):
        """Detect deadlock conditions."""
        # Check if any operations are processing for too long
        for node in self.cluster_config["nodes"]:
            node_id = node["id"]
            host = node["host"]
            port = node["port"]
            
            try:
                url = f"http://{host}:{port}/api/health/operations"
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=5) as response:
                        if response.status == 200:
                            operations = await response.json()
                            
                            # Check for long-running operations
                            for op in operations.get("operations", []):
                                duration = op.get("duration", 0)
                                if duration > 60:  # More than 60 seconds
                                    failures.append({
                                        "type": FailureType.DEADLOCK,
                                        "node_id": node_id,
                                        "details": f"Long-running operation detected: {op}"
                                    })
            except Exception:
                # Skip if we can't connect - this will be caught elsewhere
                pass
                
    async def _detect_resource_exhaustion(self, failures):
        """Detect resource exhaustion."""
        for node in self.cluster_config["nodes"]:
            node_id = node["id"]
            host = node["host"]
            port = node["port"]
            
            try:
                url = f"http://{host}:{port}/api/health/resources"
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=5) as response:
                        if response.status == 200:
                            resources = await response.json()
                            
                            # Check CPU usage
                            cpu_usage = resources.get("cpu_usage", 0)
                            if cpu_usage > 90:  # More than 90%
                                failures.append({
                                    "type": FailureType.RESOURCE_EXHAUSTION,
                                    "node_id": node_id,
                                    "details": f"High CPU usage: {cpu_usage}%"
                                })
                                
                            # Check memory usage
                            memory_usage = resources.get("memory_usage", 0)
                            if memory_usage > 90:  # More than 90%
                                failures.append({
                                    "type": FailureType.RESOURCE_EXHAUSTION,
                                    "node_id": node_id,
                                    "details": f"High memory usage: {memory_usage}%"
                                })
                                
                            # Check disk usage
                            disk_usage = resources.get("disk_usage", 0)
                            if disk_usage > 90:  # More than 90%
                                failures.append({
                                    "type": FailureType.RESOURCE_EXHAUSTION,
                                    "node_id": node_id,
                                    "details": f"High disk usage: {disk_usage}%"
                                })
            except Exception:
                # Skip if we can't connect - this will be caught elsewhere
                pass
                
    async def _detect_slow_followers(self, failures):
        """Detect slow followers."""
        # Get leader
        leader_id = None
        leader_term = 0
        leader_commit_index = 0
        
        for node_id, status in self.node_status.items():
            if status.get("role") == "LEADER":
                leader_id = node_id
                leader_term = status.get("term", 0)
                leader_commit_index = status.get("commit_index", 0)
                break
                
        if leader_id is None:
            return  # No leader detected
            
        # Check followers
        for node_id, status in self.node_status.items():
            if node_id != leader_id:
                # Check if follower is significantly behind
                follower_commit_index = status.get("commit_index", 0)
                lag = leader_commit_index - follower_commit_index
                
                # If lag is more than 100 entries, consider it a slow follower
                if lag > 100:
                    failures.append({
                        "type": FailureType.SLOW_FOLLOWER,
                        "node_id": node_id,
                        "details": f"Slow follower detected: {lag} entries behind leader"
                    })
                    
    async def recover(self, failures):
        """Recover from detected failures."""
        logger.info(f"Recovering from {len(failures)} failures")
        
        # Group failures by type
        failures_by_type = {}
        for failure in failures:
            failure_type = failure["type"]
            if failure_type not in failures_by_type:
                failures_by_type[failure_type] = []
                
            failures_by_type[failure_type].append(failure)
            
        # Process each failure type
        for failure_type, type_failures in failures_by_type.items():
            if failure_type == FailureType.PROCESS_CRASH:
                await self._recover_from_process_crash(type_failures)
            elif failure_type == FailureType.NETWORK_PARTITION:
                await self._recover_from_network_partition(type_failures)
            elif failure_type == FailureType.DATABASE_CORRUPTION:
                await self._recover_from_database_corruption(type_failures)
            elif failure_type == FailureType.LOG_CORRUPTION:
                await self._recover_from_log_corruption(type_failures)
            elif failure_type == FailureType.SPLIT_BRAIN:
                await self._recover_from_split_brain(type_failures)
            elif failure_type == FailureType.TERM_DIVERGENCE:
                await self._recover_from_term_divergence(type_failures)
            elif failure_type == FailureType.STATE_DIVERGENCE:
                await self._recover_from_state_divergence(type_failures)
            elif failure_type == FailureType.DEADLOCK:
                await self._recover_from_deadlock(type_failures)
            elif failure_type == FailureType.RESOURCE_EXHAUSTION:
                await self._recover_from_resource_exhaustion(type_failures)
            elif failure_type == FailureType.SLOW_FOLLOWER:
                await self._recover_from_slow_follower(type_failures)
                
        # Return updated status
        return await self.detect_failures()
    
    async def _recover_from_process_crash(self, failures):
        """Recover from process crashes."""
        for failure in failures:
            node_id = failure["node_id"]
            
            # Find the node configuration
            node_config = None
            for node in self.cluster_config["nodes"]:
                if node["id"] == node_id:
                    node_config = node
                    break
                    
            if node_config is None:
                logger.error(f"Node {node_id} not found in configuration")
                continue
                
            # Restart the node
            await self._restart_node(node_config)
            
    async def _restart_node(self, node_config):
        """Restart a coordinator node."""
        node_id = node_config["id"]
        host = node_config["host"]
        port = node_config["port"]
        data_dir = node_config.get("data_dir", os.path.join(self.data_dir, node_id))
        db_path = node_config.get("db_path", os.path.join(data_dir, "coordinator.duckdb"))
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Build peers list
        peers = []
        for node in self.cluster_config["nodes"]:
            if node["id"] != node_id:
                peers.append(f"{node['host']}:{node['port']}")
                
        peers_arg = ",".join(peers)
        
        # Create command for starting the node
        cmd = [
            sys.executable,
            "-m", "distributed_testing.coordinator",
            "--id", node_id,
            "--host", host,
            "--port", str(port),
            "--db-path", db_path,
            "--data-dir", data_dir,
            "--enable-redundancy",
            "--peers", peers_arg,
            "--log-level", "INFO"
        ]
        
        # Add any extra args from config
        if "extra_args" in node_config:
            cmd.extend(node_config["extra_args"])
            
        # Start the process
        logger.info(f"Restarting node {node_id}")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Store the process
        self.processes[node_id] = process
        
        # Log recovery action
        self.log_recovery_action(
            "restart_node",
            {
                "node_id": node_id,
                "host": host,
                "port": port,
                "data_dir": data_dir,
                "db_path": db_path,
                "pid": process.pid
            }
        )
        
        # Wait for the node to start responding
        max_attempts = 10
        for i in range(max_attempts):
            try:
                url = f"http://{host}:{port}/api/status"
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=1) as response:
                        if response.status == 200:
                            logger.info(f"Node {node_id} successfully restarted")
                            return True
            except Exception:
                pass
                
            await asyncio.sleep(1)
            
        logger.warning(f"Node {node_id} did not respond after restart")
        return False
        
    async def _recover_from_network_partition(self, failures):
        """Recover from network partitions."""
        # For network partitions, we simply log the issue and wait for connectivity to be restored
        for failure in failures:
            node_id = failure["node_id"]
            details = failure["details"]
            
            logger.warning(f"Network partition detected for node {node_id}: {details}")
            
            # Log recovery action
            self.log_recovery_action(
                "network_partition_detected",
                {
                    "node_id": node_id,
                    "details": details
                }
            )
            
        # Wait for partition to heal naturally
        logger.info("Waiting for network partitions to heal naturally")
        await asyncio.sleep(5)
        
    async def _recover_from_database_corruption(self, failures):
        """Recover from database corruption."""
        for failure in failures:
            node_id = failure["node_id"]
            
            # Find the node configuration
            node_config = None
            for node in self.cluster_config["nodes"]:
                if node["id"] == node_id:
                    node_config = node
                    break
                    
            if node_config is None:
                logger.error(f"Node {node_id} not found in configuration")
                continue
                
            # Stop the node if it's running
            if node_id in self.processes:
                process = self.processes[node_id]
                try:
                    process.send_signal(signal.SIGTERM)
                    process.wait(timeout=5)
                except Exception as e:
                    logger.warning(f"Error stopping node {node_id}: {e}")
                    
                del self.processes[node_id]
                
            # Backup the corrupted database
            data_dir = node_config.get("data_dir", os.path.join(self.data_dir, node_id))
            db_path = node_config.get("db_path", os.path.join(data_dir, "coordinator.duckdb"))
            
            if os.path.exists(db_path):
                backup_path = f"{db_path}.corrupted.{int(time.time())}"
                try:
                    shutil.copy2(db_path, backup_path)
                    logger.info(f"Backed up corrupted database to {backup_path}")
                except Exception as e:
                    logger.warning(f"Error backing up database: {e}")
                    
                # Remove the corrupted database
                try:
                    os.remove(db_path)
                    logger.info(f"Removed corrupted database {db_path}")
                except Exception as e:
                    logger.warning(f"Error removing database: {e}")
                    
            # Restart the node
            await self._restart_node(node_config)
            
            # Log recovery action
            self.log_recovery_action(
                "recover_database_corruption",
                {
                    "node_id": node_id,
                    "db_path": db_path,
                    "backup_path": backup_path if os.path.exists(db_path) else None
                }
            )
            
    async def _recover_from_log_corruption(self, failures):
        """Recover from log corruption."""
        for failure in failures:
            node_id = failure["node_id"]
            
            # Find the node configuration
            node_config = None
            for node in self.cluster_config["nodes"]:
                if node["id"] == node_id:
                    node_config = node
                    break
                    
            if node_config is None:
                logger.error(f"Node {node_id} not found in configuration")
                continue
                
            # Stop the node if it's running
            if node_id in self.processes:
                process = self.processes[node_id]
                try:
                    process.send_signal(signal.SIGTERM)
                    process.wait(timeout=5)
                except Exception as e:
                    logger.warning(f"Error stopping node {node_id}: {e}")
                    
                del self.processes[node_id]
                
            # Backup and remove the corrupted log
            data_dir = node_config.get("data_dir", os.path.join(self.data_dir, node_id))
            log_path = os.path.join(data_dir, "raft_log.json")
            
            if os.path.exists(log_path):
                backup_path = f"{log_path}.corrupted.{int(time.time())}"
                try:
                    shutil.copy2(log_path, backup_path)
                    logger.info(f"Backed up corrupted log to {backup_path}")
                except Exception as e:
                    logger.warning(f"Error backing up log: {e}")
                    
                # Remove the corrupted log
                try:
                    os.remove(log_path)
                    logger.info(f"Removed corrupted log {log_path}")
                except Exception as e:
                    logger.warning(f"Error removing log: {e}")
                    
            # Restart the node
            await self._restart_node(node_config)
            
            # Log recovery action
            self.log_recovery_action(
                "recover_log_corruption",
                {
                    "node_id": node_id,
                    "log_path": log_path,
                    "backup_path": backup_path if os.path.exists(log_path) else None
                }
            )
            
    async def _recover_from_split_brain(self, failures):
        """Recover from split brain condition."""
        for failure in failures:
            node_ids = failure["node_id"]
            
            logger.warning(f"Split brain detected among nodes: {node_ids}")
            
            # Strategy: Stop all nodes and restart them one by one with a delay
            # This allows a single leader to be elected cleanly
            
            # Stop all nodes
            for node_id in node_ids:
                if node_id in self.processes:
                    process = self.processes[node_id]
                    try:
                        process.send_signal(signal.SIGTERM)
                        process.wait(timeout=5)
                    except Exception as e:
                        logger.warning(f"Error stopping node {node_id}: {e}")
                        
                    del self.processes[node_id]
                    
            # Wait a moment
            await asyncio.sleep(5)
            
            # Restart nodes one by one with a delay
            for node_id in node_ids:
                # Find the node configuration
                node_config = None
                for node in self.cluster_config["nodes"]:
                    if node["id"] == node_id:
                        node_config = node
                        break
                        
                if node_config is None:
                    logger.error(f"Node {node_id} not found in configuration")
                    continue
                    
                # Restart the node
                await self._restart_node(node_config)
                
                # Wait before starting the next node
                await asyncio.sleep(5)
                
            # Log recovery action
            self.log_recovery_action(
                "recover_split_brain",
                {
                    "node_ids": node_ids
                }
            )
            
    async def _recover_from_term_divergence(self, failures):
        """Recover from term divergence."""
        # Similar to split brain recovery
        await self._recover_from_split_brain(failures)
        
    async def _recover_from_state_divergence(self, failures):
        """Recover from state divergence."""
        for failure in failures:
            node_ids = failure["node_id"]
            
            logger.warning(f"State divergence detected among nodes: {node_ids}")
            
            # Strategy:
            # 1. Identify the leader
            # 2. For each follower with divergent state:
            #    a. Stop the node
            #    b. Backup its data
            #    c. Restart with --force-sync flag to perform full state sync
            
            # Find the leader
            leader_id = None
            for node_id, status in self.node_status.items():
                if status.get("role") == "LEADER":
                    leader_id = node_id
                    break
                    
            if leader_id is None:
                logger.warning("No leader found - cannot recover from state divergence")
                continue
                
            # Get leader's state checksum
            leader_checksum = None
            for node in self.cluster_config["nodes"]:
                if node["id"] == leader_id:
                    host = node["host"]
                    port = node["port"]
                    
                    try:
                        url = f"http://{host}:{port}/api/health/state_checksum"
                        async with aiohttp.ClientSession() as session:
                            async with session.get(url, timeout=5) as response:
                                if response.status == 200:
                                    result = await response.json()
                                    leader_checksum = result.get("checksum")
                    except Exception as e:
                        logger.warning(f"Error getting leader checksum: {e}")
                        
            if leader_checksum is None:
                logger.warning("Could not get leader's state checksum - cannot recover from state divergence")
                continue
                
            # Process each node
            for node_id in node_ids:
                if node_id == leader_id:
                    continue  # Skip the leader
                    
                # Find the node configuration
                node_config = None
                for node in self.cluster_config["nodes"]:
                    if node["id"] == node_id:
                        node_config = node
                        break
                        
                if node_config is None:
                    logger.error(f"Node {node_id} not found in configuration")
                    continue
                    
                # Get node's state checksum
                node_checksum = None
                host = node_config["host"]
                port = node_config["port"]
                
                try:
                    url = f"http://{host}:{port}/api/health/state_checksum"
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url, timeout=5) as response:
                            if response.status == 200:
                                result = await response.json()
                                node_checksum = result.get("checksum")
                except Exception as e:
                    logger.warning(f"Error getting node checksum: {e}")
                    
                # Skip if checksums match
                if node_checksum == leader_checksum:
                    logger.info(f"Node {node_id} state matches leader - skipping")
                    continue
                    
                # Stop the node if it's running
                if node_id in self.processes:
                    process = self.processes[node_id]
                    try:
                        process.send_signal(signal.SIGTERM)
                        process.wait(timeout=5)
                    except Exception as e:
                        logger.warning(f"Error stopping node {node_id}: {e}")
                        
                    del self.processes[node_id]
                    
                # Backup the node's data
                data_dir = node_config.get("data_dir", os.path.join(self.data_dir, node_id))
                backup_dir = f"{data_dir}.divergent.{int(time.time())}"
                
                if os.path.exists(data_dir):
                    try:
                        shutil.copytree(data_dir, backup_dir)
                        logger.info(f"Backed up divergent state to {backup_dir}")
                    except Exception as e:
                        logger.warning(f"Error backing up state: {e}")
                        
                # Add force-sync flag
                if "extra_args" not in node_config:
                    node_config["extra_args"] = []
                    
                if "--force-sync" not in node_config["extra_args"]:
                    node_config["extra_args"].append("--force-sync")
                    
                # Restart the node
                await self._restart_node(node_config)
                
                # Log recovery action
                self.log_recovery_action(
                    "recover_state_divergence",
                    {
                        "node_id": node_id,
                        "leader_id": leader_id,
                        "data_dir": data_dir,
                        "backup_dir": backup_dir
                    }
                )
                
    async def _recover_from_deadlock(self, failures):
        """Recover from deadlock conditions."""
        for failure in failures:
            node_id = failure["node_id"]
            
            # Find the node configuration
            node_config = None
            for node in self.cluster_config["nodes"]:
                if node["id"] == node_id:
                    node_config = node
                    break
                    
            if node_config is None:
                logger.error(f"Node {node_id} not found in configuration")
                continue
                
            # Try to cancel the operation first
            host = node_config["host"]
            port = node_config["port"]
            
            cancel_succeeded = False
            try:
                url = f"http://{host}:{port}/api/health/cancel_operations"
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, timeout=5) as response:
                        if response.status == 200:
                            result = await response.json()
                            cancel_succeeded = result.get("success", False)
            except Exception as e:
                logger.warning(f"Error cancelling operations: {e}")
                
            if cancel_succeeded:
                logger.info(f"Successfully cancelled deadlocked operations on node {node_id}")
                
                # Log recovery action
                self.log_recovery_action(
                    "cancel_deadlocked_operations",
                    {
                        "node_id": node_id
                    }
                )
            else:
                # If cancellation failed, restart the node
                logger.warning(f"Failed to cancel deadlocked operations - restarting node {node_id}")
                
                # Stop the node if it's running
                if node_id in self.processes:
                    process = self.processes[node_id]
                    try:
                        process.send_signal(signal.SIGTERM)
                        process.wait(timeout=5)
                    except Exception as e:
                        logger.warning(f"Error stopping node {node_id}: {e}")
                        
                    del self.processes[node_id]
                    
                # Restart the node
                await self._restart_node(node_config)
                
                # Log recovery action
                self.log_recovery_action(
                    "restart_deadlocked_node",
                    {
                        "node_id": node_id
                    }
                )
                
    async def _recover_from_resource_exhaustion(self, failures):
        """Recover from resource exhaustion."""
        for failure in failures:
            node_id = failure["node_id"]
            details = failure["details"]
            
            logger.warning(f"Resource exhaustion detected on node {node_id}: {details}")
            
            # Find the node configuration
            node_config = None
            for node in self.cluster_config["nodes"]:
                if node["id"] == node_id:
                    node_config = node
                    break
                    
            if node_config is None:
                logger.error(f"Node {node_id} not found in configuration")
                continue
                
            # Try to free resources first
            host = node_config["host"]
            port = node_config["port"]
            
            cleanup_succeeded = False
            try:
                url = f"http://{host}:{port}/api/health/cleanup_resources"
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, timeout=5) as response:
                        if response.status == 200:
                            result = await response.json()
                            cleanup_succeeded = result.get("success", False)
            except Exception as e:
                logger.warning(f"Error cleaning up resources: {e}")
                
            if cleanup_succeeded:
                logger.info(f"Successfully cleaned up resources on node {node_id}")
                
                # Log recovery action
                self.log_recovery_action(
                    "cleanup_resources",
                    {
                        "node_id": node_id,
                        "details": details
                    }
                )
            else:
                # If cleanup failed, restart the node
                logger.warning(f"Failed to clean up resources - restarting node {node_id}")
                
                # Stop the node if it's running
                if node_id in self.processes:
                    process = self.processes[node_id]
                    try:
                        process.send_signal(signal.SIGTERM)
                        process.wait(timeout=5)
                    except Exception as e:
                        logger.warning(f"Error stopping node {node_id}: {e}")
                        
                    del self.processes[node_id]
                    
                # Add resource limits
                if "extra_args" not in node_config:
                    node_config["extra_args"] = []
                    
                if "--memory-limit" not in " ".join(node_config["extra_args"]):
                    node_config["extra_args"].extend(["--memory-limit", "1024"])
                    
                # Restart the node
                await self._restart_node(node_config)
                
                # Log recovery action
                self.log_recovery_action(
                    "restart_resource_exhausted_node",
                    {
                        "node_id": node_id,
                        "details": details
                    }
                )
                
    async def _recover_from_slow_follower(self, failures):
        """Recover from slow follower condition."""
        for failure in failures:
            node_id = failure["node_id"]
            details = failure["details"]
            
            logger.warning(f"Slow follower detected: {node_id} - {details}")
            
            # Find the node configuration
            node_config = None
            for node in self.cluster_config["nodes"]:
                if node["id"] == node_id:
                    node_config = node
                    break
                    
            if node_config is None:
                logger.error(f"Node {node_id} not found in configuration")
                continue
                
            # Try to trigger a snapshot sync first
            host = node_config["host"]
            port = node_config["port"]
            
            sync_succeeded = False
            try:
                url = f"http://{host}:{port}/api/health/trigger_sync"
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, timeout=5) as response:
                        if response.status == 200:
                            result = await response.json()
                            sync_succeeded = result.get("success", False)
            except Exception as e:
                logger.warning(f"Error triggering sync: {e}")
                
            if sync_succeeded:
                logger.info(f"Successfully triggered sync on slow follower {node_id}")
                
                # Log recovery action
                self.log_recovery_action(
                    "trigger_sync_on_slow_follower",
                    {
                        "node_id": node_id,
                        "details": details
                    }
                )
            else:
                # If sync failed, restart the node with force-sync
                logger.warning(f"Failed to trigger sync - restarting node {node_id} with force-sync")
                
                # Stop the node if it's running
                if node_id in self.processes:
                    process = self.processes[node_id]
                    try:
                        process.send_signal(signal.SIGTERM)
                        process.wait(timeout=5)
                    except Exception as e:
                        logger.warning(f"Error stopping node {node_id}: {e}")
                        
                    del self.processes[node_id]
                    
                # Add force-sync flag
                if "extra_args" not in node_config:
                    node_config["extra_args"] = []
                    
                if "--force-sync" not in node_config["extra_args"]:
                    node_config["extra_args"].append("--force-sync")
                    
                # Restart the node
                await self._restart_node(node_config)
                
                # Log recovery action
                self.log_recovery_action(
                    "restart_slow_follower",
                    {
                        "node_id": node_id,
                        "details": details
                    }
                )
                

async def main():
    """Main function to run recovery strategies."""
    parser = argparse.ArgumentParser(description="Run coordinator recovery strategies")
    parser.add_argument("--config", type=str, required=True,
                      help="Path to cluster configuration file")
    parser.add_argument("--data-dir", type=str, default="/tmp/distributed_testing_recovery",
                      help="Directory for recovery data")
    parser.add_argument("--interval", type=float, default=30.0,
                      help="Monitoring interval in seconds")
    parser.add_argument("--daemon", action="store_true",
                      help="Run in daemon mode (continuous monitoring)")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, "r") as f:
        config = json.load(f)
        
    # Create recovery strategy
    recovery = CoordinatorRecoveryStrategy(config, data_dir=args.data_dir)
    
    if args.daemon:
        # Run in daemon mode
        logger.info("Starting recovery daemon")
        
        try:
            while True:
                # Detect failures
                failures = await recovery.detect_failures()
                
                if failures:
                    logger.info(f"Detected {len(failures)} failures - initiating recovery")
                    await recovery.recover(failures)
                else:
                    logger.info("No failures detected")
                    
                # Wait for next interval
                await asyncio.sleep(args.interval)
                
        except KeyboardInterrupt:
            logger.info("Recovery daemon stopped by user")
            
        finally:
            # Save recovery log
            await recovery.save_recovery_log()
            
    else:
        # Run once
        logger.info("Detecting failures")
        failures = await recovery.detect_failures()
        
        if failures:
            logger.info(f"Detected {len(failures)} failures - initiating recovery")
            await recovery.recover(failures)
        else:
            logger.info("No failures detected")
            
        # Save recovery log
        log_file = await recovery.save_recovery_log()
        print(f"Recovery log saved to {log_file}")
        

if __name__ == "__main__":
    asyncio.run(main())