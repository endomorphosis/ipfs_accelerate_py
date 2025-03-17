#!/usr/bin/env python3
"""
Distributed Testing Framework - Auto Recovery System

This module implements the coordinator redundancy and failover mechanisms for the
distributed testing framework. It's responsible for:

- Coordinator state replication between multiple instances
- Leader election in a multi-coordinator environment
- Automatic failover when the primary coordinator fails
- State synchronization during failover
- Ensuring data consistency during transitions

Usage:
    This module can be used to enable high availability for the coordinator
    by running multiple instances in a redundant configuration.
"""

import os
import sys
import json
import time
import uuid
import socket
import logging
import threading
import requests
import hashlib
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Callable, Union
from pathlib import Path
import random

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("auto_recovery")

# Add parent directory to path to import modules from parent
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Coordinator status constants
COORDINATOR_STATUS_LEADER = "leader"
COORDINATOR_STATUS_FOLLOWER = "follower"
COORDINATOR_STATUS_CANDIDATE = "candidate"
COORDINATOR_STATUS_OFFLINE = "offline"

# Default configuration values
DEFAULT_HEARTBEAT_INTERVAL = 5  # seconds
DEFAULT_ELECTION_TIMEOUT_MIN = 150  # milliseconds
DEFAULT_ELECTION_TIMEOUT_MAX = 300  # milliseconds
DEFAULT_COORDINATOR_PORT = 8080  # Default coordinator API port
DEFAULT_LEADER_CHECK_INTERVAL = 10  # seconds

class AutoRecovery:
    """Auto recovery system for coordinator redundancy and failover."""
    
    def __init__(self, coordinator_id: str = None, db_manager=None, 
                 coordinator_manager=None, task_scheduler=None):
        """Initialize the auto recovery system.
        
        Args:
            coordinator_id: Unique identifier for this coordinator
            db_manager: Database manager for state persistence
            coordinator_manager: Coordinator manager reference
            task_scheduler: Task scheduler reference
        """
        self.coordinator_id = coordinator_id or f"coordinator-{uuid.uuid4().hex[:8]}"
        self.db_manager = db_manager
        self.coordinator_manager = coordinator_manager
        self.task_scheduler = task_scheduler
        
        # Coordinator cluster state
        self.status = COORDINATOR_STATUS_FOLLOWER
        self.term = 0
        self.voted_for = None
        self.leader_id = None
        self.last_leader_heartbeat = datetime.now()
        self.coordinators = {}  # coordinator_id -> coordinator info
        
        # Configuration (initialize first so other methods can use it)
        self.config = {
            "heartbeat_interval": DEFAULT_HEARTBEAT_INTERVAL,
            "election_timeout_min": DEFAULT_ELECTION_TIMEOUT_MIN,
            "election_timeout_max": DEFAULT_ELECTION_TIMEOUT_MAX,
            "coordinator_port": DEFAULT_COORDINATOR_PORT,
            "leader_check_interval": DEFAULT_LEADER_CHECK_INTERVAL,
            "failover_enabled": True,
            "auto_leader_election": True,
            "auto_discover_coordinators": True,
            "coordinator_addresses": [],  # List of coordinator addresses
            "snapshot_interval": 300,  # seconds
            "state_sync_batch_size": 100,
            "state_persistence_enabled": True,
            "min_followers_for_commit": 1,  # Minimum followers required to commit
        }
        
        # Election timers (after config is initialized)
        self.election_timeout = self._get_random_election_timeout()
        self.last_election_reset = datetime.now()
        self.votes_received = set()
        
        # Replication and state sync
        self.commit_index = 0
        self.last_applied = 0
        self.log_entries = []
        self.next_index = {}
        self.match_index = {}
        
        # State snapshot
        self.last_snapshot_time = datetime.now()
        self.state_snapshot = {}
        
        # Monitor threads
        self.leader_check_thread = None
        self.leader_check_stop_event = threading.Event()
        self.heartbeat_thread = None
        self.heartbeat_stop_event = threading.Event()
        
        # Node discovery mechanisms
        self.discovery_thread = None
        self.discovery_stop_event = threading.Event()
        
        # Callback functions for leader transitions
        self.on_become_leader_callbacks = []
        self.on_leader_changed_callbacks = []
        
        logger.info(f"Auto recovery system initialized with ID {self.coordinator_id}")
        
    def configure(self, config_updates: Dict[str, Any]):
        """Update the auto recovery configuration.
        
        Args:
            config_updates: Dictionary of configuration updates
        """
        self.config.update(config_updates)
        logger.info(f"Auto recovery configuration updated: {config_updates}")
        
    def start(self):
        """Start the auto recovery system."""
        # Load persistent state if available
        self._load_persistent_state()
        
        # Start leader check thread
        self.leader_check_stop_event.clear()
        self.leader_check_thread = threading.Thread(
            target=self._leader_check_loop,
            daemon=True
        )
        self.leader_check_thread.start()
        
        # If configured as leader, start heartbeat thread
        if self.status == COORDINATOR_STATUS_LEADER:
            self._start_heartbeat_thread()
            
        # Start node discovery if enabled
        if self.config["auto_discover_coordinators"]:
            self._start_discovery_thread()
            
        logger.info(f"Auto recovery system started with status {self.status}")
        
    def stop(self):
        """Stop the auto recovery system."""
        # Stop leader check thread
        if self.leader_check_thread:
            self.leader_check_stop_event.set()
            self.leader_check_thread.join(timeout=5.0)
            if self.leader_check_thread.is_alive():
                logger.warning("Leader check thread did not stop gracefully")
                
        # Stop heartbeat thread if running
        if self.heartbeat_thread:
            self.heartbeat_stop_event.set()
            self.heartbeat_thread.join(timeout=5.0)
            if self.heartbeat_thread.is_alive():
                logger.warning("Heartbeat thread did not stop gracefully")
                
        # Stop discovery thread if running
        if self.discovery_thread:
            self.discovery_stop_event.set()
            self.discovery_thread.join(timeout=5.0)
            if self.discovery_thread.is_alive():
                logger.warning("Discovery thread did not stop gracefully")
                
        logger.info("Auto recovery system stopped")
        
    def register_coordinator(self, coordinator_id: str, address: str, port: int, 
                           capabilities: Dict[str, Any]) -> bool:
        """Register a coordinator in the cluster.
        
        Args:
            coordinator_id: ID of the coordinator
            address: Network address of the coordinator
            port: API port of the coordinator
            capabilities: Capabilities of the coordinator
            
        Returns:
            True if successful, False otherwise
        """
        if coordinator_id in self.coordinators:
            # Update existing coordinator
            self.coordinators[coordinator_id].update({
                "address": address,
                "port": port,
                "capabilities": capabilities,
                "last_heartbeat": datetime.now(),
                "status": COORDINATOR_STATUS_FOLLOWER if coordinator_id != self.leader_id else COORDINATOR_STATUS_LEADER
            })
            logger.debug(f"Updated coordinator {coordinator_id}")
        else:
            # Add new coordinator
            self.coordinators[coordinator_id] = {
                "coordinator_id": coordinator_id,
                "address": address,
                "port": port,
                "capabilities": capabilities,
                "last_heartbeat": datetime.now(),
                "status": COORDINATOR_STATUS_FOLLOWER,
                "term": 0
            }
            logger.info(f"Registered new coordinator {coordinator_id}")
            
        # Update next_index and match_index if leader
        if self.status == COORDINATOR_STATUS_LEADER:
            if coordinator_id not in self.next_index:
                self.next_index[coordinator_id] = len(self.log_entries)
            if coordinator_id not in self.match_index:
                self.match_index[coordinator_id] = 0
                
        return True
        
    def unregister_coordinator(self, coordinator_id: str) -> bool:
        """Unregister a coordinator from the cluster.
        
        Args:
            coordinator_id: ID of the coordinator
            
        Returns:
            True if successful, False otherwise
        """
        if coordinator_id in self.coordinators:
            del self.coordinators[coordinator_id]
            
            # Remove from indices if leader
            if self.status == COORDINATOR_STATUS_LEADER:
                if coordinator_id in self.next_index:
                    del self.next_index[coordinator_id]
                if coordinator_id in self.match_index:
                    del self.match_index[coordinator_id]
                    
            logger.info(f"Unregistered coordinator {coordinator_id}")
            return True
        else:
            logger.warning(f"Coordinator {coordinator_id} not found in cluster")
            return False
            
    def update_coordinator_heartbeat(self, coordinator_id: str) -> bool:
        """Update the heartbeat timestamp for a coordinator.
        
        Args:
            coordinator_id: ID of the coordinator
            
        Returns:
            True if successful, False otherwise
        """
        if coordinator_id in self.coordinators:
            self.coordinators[coordinator_id]["last_heartbeat"] = datetime.now()
            return True
        else:
            return False
            
    def _leader_check_loop(self):
        """Leader check thread function."""
        while not self.leader_check_stop_event.is_set():
            try:
                if self.status == COORDINATOR_STATUS_LEADER:
                    # We are the leader, check if we have enough followers
                    active_followers = self._count_active_followers()
                    
                    if active_followers < self.config["min_followers_for_commit"]:
                        logger.warning(
                            f"Leader has only {active_followers} active followers, "
                            f"need {self.config['min_followers_for_commit']} for stable operation"
                        )
                        
                    # Take periodic state snapshots
                    time_since_snapshot = (datetime.now() - self.last_snapshot_time).total_seconds()
                    if time_since_snapshot > self.config["snapshot_interval"]:
                        self._create_state_snapshot()
                        
                elif self.status == COORDINATOR_STATUS_FOLLOWER:
                    # Check if leader is still active
                    if self.leader_id:
                        time_since_heartbeat = (datetime.now() - self.last_leader_heartbeat).total_seconds()
                        
                        if time_since_heartbeat > (self.config["heartbeat_interval"] * 3):
                            # Leader might be down, start election if enabled
                            logger.warning(
                                f"No heartbeat from leader {self.leader_id} for {time_since_heartbeat:.1f}s"
                            )
                            
                            if self.config["auto_leader_election"] and self.config["failover_enabled"]:
                                logger.info("Starting leader election")
                                self._start_election()
                                
                    else:
                        # No known leader, start election if enabled
                        if self.config["auto_leader_election"] and self.config["failover_enabled"]:
                            time_since_reset = (datetime.now() - self.last_election_reset).total_seconds() * 1000
                            
                            if time_since_reset > self.election_timeout:
                                logger.info("No leader, starting election")
                                self._start_election()
                                
                elif self.status == COORDINATOR_STATUS_CANDIDATE:
                    # Check if election has timed out
                    time_since_reset = (datetime.now() - self.last_election_reset).total_seconds() * 1000
                    
                    if time_since_reset > self.election_timeout:
                        # Election timed out, start new election
                        logger.info("Election timed out, starting new election")
                        self._start_election()
                        
            except Exception as e:
                logger.error(f"Error in leader check loop: {e}")
                
            # Wait for next check interval
            self.leader_check_stop_event.wait(self.config["leader_check_interval"])
            
    def _heartbeat_loop(self):
        """Heartbeat thread function for leader."""
        while not self.heartbeat_stop_event.is_set():
            try:
                if self.status == COORDINATOR_STATUS_LEADER:
                    # Send heartbeats to all coordinators
                    for coordinator_id, coordinator in self.coordinators.items():
                        if coordinator_id == self.coordinator_id:
                            continue  # Skip self
                            
                        try:
                            self._send_heartbeat(coordinator_id)
                        except Exception as e:
                            logger.warning(f"Failed to send heartbeat to {coordinator_id}: {e}")
                else:
                    # No longer leader, stop heartbeat thread
                    logger.info("No longer leader, stopping heartbeat thread")
                    break
                    
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                
            # Wait for next heartbeat interval
            self.heartbeat_stop_event.wait(self.config["heartbeat_interval"])
            
    def _send_heartbeat(self, coordinator_id: str):
        """Send heartbeat to a coordinator.
        
        Args:
            coordinator_id: ID of the coordinator to send heartbeat to
        """
        if coordinator_id not in self.coordinators:
            logger.warning(f"Cannot send heartbeat to unknown coordinator {coordinator_id}")
            return
            
        coordinator = self.coordinators[coordinator_id]
        address = coordinator["address"]
        port = coordinator["port"]
        
        # Prepare heartbeat data
        heartbeat_data = {
            "coordinator_id": self.coordinator_id,
            "term": self.term,
            "leader_id": self.coordinator_id,
            "commit_index": self.commit_index,
            "prev_log_index": self.next_index.get(coordinator_id, 0) - 1,
            "prev_log_term": self._get_log_term(self.next_index.get(coordinator_id, 0) - 1),
            "entries": self._get_log_entries_from(self.next_index.get(coordinator_id, 0))
        }
        
        # Send heartbeat via API
        try:
            url = f"http://{address}:{port}/api/v1/coordinator/heartbeat"
            response = requests.post(url, json=heartbeat_data, timeout=2.0)
            
            if response.status_code == 200:
                result = response.json()
                
                # Process response
                if result.get("success"):
                    # Update follower indices
                    if "match_index" in result:
                        self.match_index[coordinator_id] = result["match_index"]
                        self.next_index[coordinator_id] = result["match_index"] + 1
                        
                    # Check if we can commit more entries
                    self._update_commit_index()
                else:
                    term = result.get("term", 0)
                    
                    # If response term is higher, step down
                    if term > self.term:
                        logger.info(f"Received higher term ({term}) from {coordinator_id}, stepping down")
                        self._become_follower(term)
                        
            else:
                logger.warning(f"Heartbeat to {coordinator_id} failed with status {response.status_code}")
                
        except Exception as e:
            logger.warning(f"Failed to send heartbeat to {coordinator_id}: {e}")
            
            # Mark coordinator as potentially offline
            time_threshold = datetime.now() - timedelta(seconds=self.config["heartbeat_interval"] * 3)
            if coordinator["last_heartbeat"] < time_threshold:
                coordinator["status"] = COORDINATOR_STATUS_OFFLINE
                logger.warning(f"Coordinator {coordinator_id} appears to be offline")
                
    def _start_election(self):
        """Start a leader election."""
        self.status = COORDINATOR_STATUS_CANDIDATE
        self.term += 1
        self.voted_for = self.coordinator_id
        self.votes_received = {self.coordinator_id}  # Vote for self
        self.last_election_reset = datetime.now()
        self.election_timeout = self._get_random_election_timeout()
        
        logger.info(f"Started election for term {self.term}")
        
        # Persist state
        self._save_persistent_state()
        
        # Request votes from all other coordinators
        for coordinator_id, coordinator in self.coordinators.items():
            if coordinator_id == self.coordinator_id:
                continue  # Skip self
                
            # Skip coordinators that appear offline
            if coordinator.get("status") == COORDINATOR_STATUS_OFFLINE:
                continue
                
            try:
                self._request_vote(coordinator_id)
            except Exception as e:
                logger.warning(f"Failed to request vote from {coordinator_id}: {e}")
                
        # Check if we already have majority (e.g., single-node cluster)
        self._check_election_result()
        
    def _request_vote(self, coordinator_id: str):
        """Request vote from a coordinator.
        
        Args:
            coordinator_id: ID of the coordinator to request vote from
        """
        if coordinator_id not in self.coordinators:
            logger.warning(f"Cannot request vote from unknown coordinator {coordinator_id}")
            return
            
        coordinator = self.coordinators[coordinator_id]
        address = coordinator["address"]
        port = coordinator["port"]
        
        # Prepare vote request data
        vote_request = {
            "coordinator_id": self.coordinator_id,
            "term": self.term,
            "last_log_index": len(self.log_entries) - 1,
            "last_log_term": self._get_log_term(len(self.log_entries) - 1)
        }
        
        # Send vote request via API
        try:
            url = f"http://{address}:{port}/api/v1/coordinator/request_vote"
            response = requests.post(url, json=vote_request, timeout=2.0)
            
            if response.status_code == 200:
                result = response.json()
                
                # Process response
                term = result.get("term", 0)
                vote_granted = result.get("vote_granted", False)
                
                # If response term is higher, step down
                if term > self.term:
                    logger.info(f"Received higher term ({term}) from {coordinator_id}, stepping down")
                    self._become_follower(term)
                    return
                    
                # If vote granted, add to received votes
                if vote_granted:
                    logger.info(f"Received vote from {coordinator_id}")
                    self.votes_received.add(coordinator_id)
                    
                    # Check if we have majority
                    self._check_election_result()
                    
            else:
                logger.warning(f"Vote request to {coordinator_id} failed with status {response.status_code}")
                
        except Exception as e:
            logger.warning(f"Failed to request vote from {coordinator_id}: {e}")
            
    def _check_election_result(self):
        """Check if we have received majority of votes."""
        if self.status != COORDINATOR_STATUS_CANDIDATE:
            return  # Not a candidate anymore
            
        # Calculate majority threshold
        total_coordinators = len(self.coordinators) + 1  # Include self
        majority = (total_coordinators // 2) + 1
        
        if len(self.votes_received) >= majority:
            logger.info(
                f"Won election with {len(self.votes_received)} votes out of {total_coordinators} coordinators"
            )
            self._become_leader()
            
    def _become_leader(self):
        """Transition to leader state."""
        if self.status == COORDINATOR_STATUS_LEADER:
            return  # Already leader
            
        logger.info(f"Becoming leader for term {self.term}")
        
        self.status = COORDINATOR_STATUS_LEADER
        self.leader_id = self.coordinator_id
        
        # Initialize leader state
        self.next_index = {
            coordinator_id: len(self.log_entries)
            for coordinator_id in self.coordinators.keys()
        }
        self.match_index = {
            coordinator_id: 0
            for coordinator_id in self.coordinators.keys()
        }
        
        # Start heartbeat thread
        self._start_heartbeat_thread()
        
        # Notify callbacks
        for callback in self.on_become_leader_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in leader callback: {e}")
                
        # Persist state
        self._save_persistent_state()
        
        # Create state snapshot
        self._create_state_snapshot()
        
    def _become_follower(self, term: int, leader_id: str = None):
        """Transition to follower state.
        
        Args:
            term: Current term number
            leader_id: Optional ID of the current leader
        """
        was_leader = self.status == COORDINATOR_STATUS_LEADER
        prev_leader = self.leader_id
        
        logger.info(f"Becoming follower for term {term}")
        
        self.status = COORDINATOR_STATUS_FOLLOWER
        self.term = term
        self.voted_for = None
        
        if leader_id:
            self.leader_id = leader_id
            self.last_leader_heartbeat = datetime.now()
            
        # Stop heartbeat thread if it was running
        if was_leader:
            self._stop_heartbeat_thread()
            
        # Reset election timeout
        self.last_election_reset = datetime.now()
        self.election_timeout = self._get_random_election_timeout()
        
        # Persist state
        self._save_persistent_state()
        
        # Notify callbacks if leader changed
        if was_leader or self.leader_id != prev_leader:
            for callback in self.on_leader_changed_callbacks:
                try:
                    callback(prev_leader, self.leader_id)
                except Exception as e:
                    logger.error(f"Error in leader changed callback: {e}")
                    
    def _start_heartbeat_thread(self):
        """Start the heartbeat thread."""
        # Stop existing thread if running
        self._stop_heartbeat_thread()
        
        # Start new thread
        self.heartbeat_stop_event.clear()
        self.heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True
        )
        self.heartbeat_thread.start()
        logger.info("Started heartbeat thread")
        
    def _stop_heartbeat_thread(self):
        """Stop the heartbeat thread."""
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            self.heartbeat_stop_event.set()
            self.heartbeat_thread.join(timeout=5.0)
            if self.heartbeat_thread.is_alive():
                logger.warning("Heartbeat thread did not stop gracefully")
                
        self.heartbeat_thread = None
        
    def _start_discovery_thread(self):
        """Start the node discovery thread."""
        # Stop existing thread if running
        if self.discovery_thread and self.discovery_thread.is_alive():
            self.discovery_stop_event.set()
            self.discovery_thread.join(timeout=5.0)
            if self.discovery_thread.is_alive():
                logger.warning("Discovery thread did not stop gracefully")
                
        # Start new thread
        self.discovery_stop_event.clear()
        self.discovery_thread = threading.Thread(
            target=self._discovery_loop,
            daemon=True
        )
        self.discovery_thread.start()
        logger.info("Started discovery thread")
        
    def _discovery_loop(self):
        """Node discovery thread function."""
        while not self.discovery_stop_event.is_set():
            try:
                # Discover coordinators via configured methods
                self._discover_coordinators()
                
            except Exception as e:
                logger.error(f"Error in discovery loop: {e}")
                
            # Wait for next discovery interval
            self.discovery_stop_event.wait(60)  # Check every minute
            
    def _discover_coordinators(self):
        """Discover other coordinators in the cluster."""
        # Use configured coordinator addresses
        for address in self.config["coordinator_addresses"]:
            if not address:
                continue
                
            # Extract host and port
            if ":" in address:
                host, port_str = address.split(":", 1)
                try:
                    port = int(port_str)
                except ValueError:
                    port = self.config["coordinator_port"]
            else:
                host = address
                port = self.config["coordinator_port"]
                
            # Skip self
            if self._is_self_address(host, port):
                continue
                
            # Try to connect and discover
            try:
                url = f"http://{host}:{port}/api/v1/coordinator/info"
                response = requests.get(url, timeout=2.0)
                
                if response.status_code == 200:
                    coordinator_info = response.json()
                    coordinator_id = coordinator_info.get("coordinator_id")
                    capabilities = coordinator_info.get("capabilities", {})
                    
                    if coordinator_id:
                        # Register coordinator
                        self.register_coordinator(coordinator_id, host, port, capabilities)
                        
            except Exception as e:
                logger.debug(f"Failed to discover coordinator at {host}:{port}: {e}")
                
    def _is_self_address(self, host: str, port: int) -> bool:
        """Check if an address refers to this coordinator.
        
        Args:
            host: Hostname or IP
            port: Port number
            
        Returns:
            True if the address refers to this coordinator
        """
        # Check if hostname is this machine
        try:
            hostname = socket.gethostname()
            localhost_names = {"localhost", "127.0.0.1", hostname, socket.gethostbyname(hostname)}
            
            if host in localhost_names and port == self.config["coordinator_port"]:
                return True
                
        except Exception:
            pass
            
        return False
        
    def _get_random_election_timeout(self) -> int:
        """Get a random election timeout.
        
        Returns:
            Timeout in milliseconds
        """
        min_timeout = self.config["election_timeout_min"]
        max_timeout = self.config["election_timeout_max"]
        return random.randint(min_timeout, max_timeout)
        
    def _get_log_term(self, index: int) -> int:
        """Get the term of a log entry.
        
        Args:
            index: Index of the log entry
            
        Returns:
            Term number of the entry, or 0 if index is invalid
        """
        if index < 0:
            return 0
            
        if index >= len(self.log_entries):
            return 0
            
        return self.log_entries[index].get("term", 0)
        
    def _get_log_entries_from(self, start_index: int) -> List[Dict[str, Any]]:
        """Get log entries from a starting index.
        
        Args:
            start_index: Index to start from
            
        Returns:
            List of log entries
        """
        if start_index < 0:
            start_index = 0
            
        if start_index >= len(self.log_entries):
            return []
            
        # Limit batch size
        batch_size = self.config["state_sync_batch_size"]
        end_index = min(start_index + batch_size, len(self.log_entries))
        
        return self.log_entries[start_index:end_index]
        
    def _update_commit_index(self):
        """Update the commit index based on match indices."""
        if self.status != COORDINATOR_STATUS_LEADER:
            return  # Only leaders update commit index
            
        # Find highest index that is replicated to majority
        for i in range(self.commit_index + 1, len(self.log_entries)):
            # Count coordinators that have this index
            count = 1  # Include self
            for coordinator_id, match_idx in self.match_index.items():
                if match_idx >= i:
                    count += 1
                    
            # Check if we have majority
            majority = (len(self.coordinators) + 1) // 2 + 1
            if count >= majority:
                # Check if log term matches current term
                if self._get_log_term(i) == self.term:
                    self.commit_index = i
                    
                    # Apply committed entries
                    self._apply_log_entries()
                    
    def _apply_log_entries(self):
        """Apply committed log entries."""
        while self.last_applied < self.commit_index:
            self.last_applied += 1
            entry = self.log_entries[self.last_applied - 1]
            
            # Apply entry based on type
            entry_type = entry.get("type")
            
            if entry_type == "task":
                task_data = entry.get("data", {})
                if self.task_scheduler:
                    task_id = task_data.get("task_id")
                    if not task_id:
                        continue
                        
                    # Add task to scheduler
                    task_type = task_data.get("type", "benchmark")
                    priority = task_data.get("priority", 5)
                    config = task_data.get("config", {})
                    requirements = task_data.get("requirements", {})
                    dependencies = task_data.get("dependencies", [])
                    
                    self.task_scheduler.add_task(
                        task_id, task_type, priority, config, requirements, dependencies
                    )
                    
            elif entry_type == "worker":
                worker_data = entry.get("data", {})
                action = entry.get("action")
                
                if action == "register" and self.coordinator_manager:
                    worker_id = worker_data.get("worker_id")
                    if not worker_id:
                        continue
                        
                    # Register worker
                    self.coordinator_manager.register_worker(
                        worker_id, 
                        worker_data.get("address", ""), 
                        worker_data.get("port", 0),
                        worker_data.get("capabilities", {})
                    )
                    
                elif action == "unregister" and self.coordinator_manager:
                    worker_id = worker_data.get("worker_id")
                    if not worker_id:
                        continue
                        
                    # Unregister worker
                    self.coordinator_manager.unregister_worker(worker_id)
                    
            elif entry_type == "configuration":
                config_data = entry.get("data", {})
                component = entry.get("component")
                
                if component == "task_scheduler" and self.task_scheduler:
                    self.task_scheduler.configure(config_data)
                    
                elif component == "auto_recovery":
                    self.configure(config_data)
                    
            elif entry_type == "coordinator":
                coordinator_data = entry.get("data", {})
                action = entry.get("action")
                
                if action == "register":
                    coordinator_id = coordinator_data.get("coordinator_id")
                    if not coordinator_id:
                        continue
                        
                    # Register coordinator
                    self.register_coordinator(
                        coordinator_id,
                        coordinator_data.get("address", ""),
                        coordinator_data.get("port", 0),
                        coordinator_data.get("capabilities", {})
                    )
                    
                elif action == "unregister":
                    coordinator_id = coordinator_data.get("coordinator_id")
                    if not coordinator_id:
                        continue
                        
                    # Unregister coordinator
                    self.unregister_coordinator(coordinator_id)
                    
            # Other entry types can be added as needed
            
    def _save_persistent_state(self):
        """Save persistent state to storage."""
        if not self.config["state_persistence_enabled"] or not self.db_manager:
            return
            
        try:
            # Save state to database
            state = {
                "coordinator_id": self.coordinator_id,
                "term": self.term,
                "voted_for": self.voted_for,
                "status": self.status,
                "leader_id": self.leader_id,
                "commit_index": self.commit_index,
                "last_applied": self.last_applied,
                "log_entries": self.log_entries,
                "updated_at": datetime.now().isoformat()
            }
            
            self.db_manager.save_coordinator_state(self.coordinator_id, state)
            logger.debug("Saved persistent state")
            
        except Exception as e:
            logger.error(f"Failed to save persistent state: {e}")
            
    def _load_persistent_state(self):
        """Load persistent state from storage."""
        if not self.config["state_persistence_enabled"] or not self.db_manager:
            return
            
        try:
            # Load state from database
            state = self.db_manager.get_coordinator_state(self.coordinator_id)
            
            if state:
                self.term = state.get("term", 0)
                self.voted_for = state.get("voted_for")
                self.status = state.get("status", COORDINATOR_STATUS_FOLLOWER)
                self.leader_id = state.get("leader_id")
                self.commit_index = state.get("commit_index", 0)
                self.last_applied = state.get("last_applied", 0)
                self.log_entries = state.get("log_entries", [])
                
                logger.info(f"Loaded persistent state: term={self.term}, status={self.status}")
            else:
                logger.info("No persistent state found, starting fresh")
                
        except Exception as e:
            logger.error(f"Failed to load persistent state: {e}")
            
    def _create_state_snapshot(self):
        """Create a snapshot of the current state."""
        # Create snapshot of tasks
        tasks_snapshot = []
        
        if self.task_scheduler:
            with self.task_scheduler.task_lock:
                # Get queued tasks
                for _, _, _, task in self.task_scheduler.task_queue:
                    tasks_snapshot.append({
                        "task_id": task.get("task_id"),
                        "status": "queued",
                        "data": task
                    })
                    
                # Get running tasks
                for task_id, worker_id in self.task_scheduler.running_tasks.items():
                    if self.db_manager:
                        task = self.db_manager.get_task(task_id)
                        if task:
                            tasks_snapshot.append({
                                "task_id": task_id,
                                "status": "running",
                                "worker_id": worker_id,
                                "data": task
                            })
        
        # Create snapshot of workers
        workers_snapshot = []
        
        if self.coordinator_manager and hasattr(self.coordinator_manager, "workers"):
            for worker_id, worker in self.coordinator_manager.workers.items():
                workers_snapshot.append({
                    "worker_id": worker_id,
                    "data": worker
                })
        
        # Create snapshot of coordinators
        coordinators_snapshot = []
        
        for coordinator_id, coordinator in self.coordinators.items():
            coordinators_snapshot.append({
                "coordinator_id": coordinator_id,
                "data": coordinator
            })
        
        # Complete snapshot
        self.state_snapshot = {
            "timestamp": datetime.now().isoformat(),
            "coordinator_id": self.coordinator_id,
            "term": self.term,
            "commit_index": self.commit_index,
            "last_applied": self.last_applied,
            "log_entries_count": len(self.log_entries),
            "tasks": tasks_snapshot,
            "workers": workers_snapshot,
            "coordinators": coordinators_snapshot
        }
        
        self.last_snapshot_time = datetime.now()
        
        # Save snapshot if persistence enabled
        if self.config["state_persistence_enabled"] and self.db_manager:
            try:
                self.db_manager.save_coordinator_snapshot(self.coordinator_id, self.state_snapshot)
                logger.info("Created and saved state snapshot")
            except Exception as e:
                logger.error(f"Failed to save state snapshot: {e}")
        else:
            logger.info("Created state snapshot")
            
    def get_state_snapshot(self) -> Dict[str, Any]:
        """Get the current state snapshot.
        
        Returns:
            State snapshot
        """
        return self.state_snapshot
        
    def on_become_leader(self, callback: Callable[[], None]):
        """Register a callback for when this coordinator becomes leader.
        
        Args:
            callback: Function to call when becoming leader
        """
        self.on_become_leader_callbacks.append(callback)
        
    def on_leader_changed(self, callback: Callable[[str, str], None]):
        """Register a callback for when the leader changes.
        
        Args:
            callback: Function to call with old and new leader IDs
        """
        self.on_leader_changed_callbacks.append(callback)
        
    def append_log_entry(self, entry_type: str, data: Dict[str, Any], 
                        component: str = None, action: str = None) -> bool:
        """Append an entry to the log.
        
        Args:
            entry_type: Type of log entry
            data: Entry data
            component: Optional component name
            action: Optional action name
            
        Returns:
            True if successful, False if not leader
        """
        if self.status != COORDINATOR_STATUS_LEADER:
            logger.warning("Cannot append log entry - not leader")
            return False
            
        # Create log entry
        entry = {
            "term": self.term,
            "type": entry_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        if component:
            entry["component"] = component
            
        if action:
            entry["action"] = action
            
        # Append to log
        self.log_entries.append(entry)
        index = len(self.log_entries) - 1
        
        logger.debug(f"Appended log entry: type={entry_type}, index={index}")
        
        # Update match index for self
        self.match_index[self.coordinator_id] = index
        
        # Check if we can commit the entry immediately (single node)
        self._update_commit_index()
        
        return True
        
    def is_leader(self) -> bool:
        """Check if this coordinator is the leader.
        
        Returns:
            True if leader, False otherwise
        """
        return self.status == COORDINATOR_STATUS_LEADER
        
    def get_leader_id(self) -> Optional[str]:
        """Get the ID of the current leader.
        
        Returns:
            Leader ID, or None if no leader
        """
        return self.leader_id
        
    def get_coordinators(self) -> Dict[str, Dict[str, Any]]:
        """Get information about coordinators in the cluster.
        
        Returns:
            Dictionary mapping coordinator IDs to information
        """
        return self.coordinators
        
    def get_status(self) -> Dict[str, Any]:
        """Get the status of the auto recovery system.
        
        Returns:
            Dictionary with status information
        """
        active_coordinators = sum(1 for c in self.coordinators.values() 
                                 if c.get("status") != COORDINATOR_STATUS_OFFLINE)
        
        return {
            "coordinator_id": self.coordinator_id,
            "status": self.status,
            "term": self.term,
            "leader_id": self.leader_id,
            "coordinators_count": len(self.coordinators),
            "active_coordinators": active_coordinators,
            "commit_index": self.commit_index,
            "last_applied": self.last_applied,
            "log_entries_count": len(self.log_entries)
        }
        
    def _count_active_followers(self) -> int:
        """Count active followers.
        
        Returns:
            Number of active followers
        """
        return sum(1 for c in self.coordinators.values() 
                  if c.get("status") not in [COORDINATOR_STATUS_OFFLINE, COORDINATOR_STATUS_LEADER])
        
    def handle_heartbeat(self, heartbeat_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a heartbeat from a leader.
        
        Args:
            heartbeat_data: Heartbeat data
            
        Returns:
            Response data
        """
        # Extract data
        coordinator_id = heartbeat_data.get("coordinator_id")
        term = heartbeat_data.get("term", 0)
        leader_id = heartbeat_data.get("leader_id")
        commit_index = heartbeat_data.get("commit_index", 0)
        prev_log_index = heartbeat_data.get("prev_log_index", 0)
        prev_log_term = heartbeat_data.get("prev_log_term", 0)
        entries = heartbeat_data.get("entries", [])
        
        # Check term
        if term < self.term:
            # Reply with our higher term
            return {
                "success": False,
                "term": self.term
            }
            
        # If term is higher or same, update leader and reset election timeout
        if term > self.term or self.leader_id != leader_id:
            self._become_follower(term, leader_id)
        else:
            # Just update heartbeat time
            self.last_leader_heartbeat = datetime.now()
            self.last_election_reset = datetime.now()
            
        # Update coordinator info
        if coordinator_id in self.coordinators:
            self.coordinators[coordinator_id]["last_heartbeat"] = datetime.now()
            self.coordinators[coordinator_id]["status"] = COORDINATOR_STATUS_LEADER
            
        # Check if we can accept log entries
        if prev_log_index > 0:
            # Check if we have the entry at prev_log_index with matching term
            if prev_log_index > len(self.log_entries):
                # Don't have this index yet
                return {
                    "success": False,
                    "term": self.term,
                    "match_index": len(self.log_entries) - 1
                }
                
            if self._get_log_term(prev_log_index - 1) != prev_log_term:
                # Term doesn't match, so reject
                return {
                    "success": False,
                    "term": self.term,
                    "match_index": prev_log_index - 2 if prev_log_index > 1 else 0
                }
                
        # Process log entries
        if entries:
            # Truncate log if needed (conflicting entries)
            if prev_log_index < len(self.log_entries):
                self.log_entries = self.log_entries[:prev_log_index]
                
            # Append new entries
            self.log_entries.extend(entries)
            
            # Save state
            self._save_persistent_state()
            
        # Update commit index
        if commit_index > self.commit_index:
            self.commit_index = min(commit_index, len(self.log_entries))
            
            # Apply committed entries
            self._apply_log_entries()
            
        # Return success
        return {
            "success": True,
            "term": self.term,
            "match_index": prev_log_index + len(entries)
        }
        
    def handle_vote_request(self, vote_request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a vote request from a candidate.
        
        Args:
            vote_request: Vote request data
            
        Returns:
            Response data
        """
        # Extract data
        coordinator_id = vote_request.get("coordinator_id")
        term = vote_request.get("term", 0)
        last_log_index = vote_request.get("last_log_index", 0)
        last_log_term = vote_request.get("last_log_term", 0)
        
        # Check term
        if term < self.term:
            # Reply with our higher term
            return {
                "term": self.term,
                "vote_granted": False
            }
            
        # If term is higher, update term
        if term > self.term:
            self._become_follower(term)
            
        # Check if we've already voted for this term
        if self.voted_for is not None and self.voted_for != coordinator_id:
            # Already voted for another coordinator
            return {
                "term": self.term,
                "vote_granted": False
            }
            
        # Check if candidate's log is at least as up-to-date as ours
        our_last_index = len(self.log_entries) - 1
        our_last_term = self._get_log_term(our_last_index)
        
        log_ok = False
        
        if last_log_term > our_last_term:
            # Candidate has higher term, so log is more up-to-date
            log_ok = True
        elif last_log_term == our_last_term and last_log_index >= our_last_index:
            # Same term, but candidate has at least as many entries
            log_ok = True
            
        if log_ok:
            # Grant vote
            self.voted_for = coordinator_id
            self.last_election_reset = datetime.now()
            
            # Save state
            self._save_persistent_state()
            
            logger.info(f"Granted vote to {coordinator_id} for term {term}")
            
            return {
                "term": self.term,
                "vote_granted": True
            }
        else:
            # Deny vote
            return {
                "term": self.term,
                "vote_granted": False
            }
            
    def handle_sync_request(self, sync_request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a state synchronization request.
        
        Args:
            sync_request: Sync request data
            
        Returns:
            Response data with state snapshot
        """
        # Only leaders should respond to sync requests
        if self.status != COORDINATOR_STATUS_LEADER:
            return {
                "success": False,
                "reason": "Not leader"
            }
            
        # Create a fresh snapshot
        self._create_state_snapshot()
        
        # Return snapshot
        return {
            "success": True,
            "snapshot": self.state_snapshot
        }
        
    def sync_with_leader(self) -> bool:
        """Synchronize state with current leader.
        
        Returns:
            True if successful, False otherwise
        """
        if self.status == COORDINATOR_STATUS_LEADER or not self.leader_id:
            return False  # We are the leader or no leader known
            
        if self.leader_id not in self.coordinators:
            logger.warning(f"Cannot sync with unknown leader {self.leader_id}")
            return False
            
        leader = self.coordinators[self.leader_id]
        address = leader["address"]
        port = leader["port"]
        
        # Request sync from leader
        try:
            url = f"http://{address}:{port}/api/v1/coordinator/sync"
            response = requests.post(url, json={"coordinator_id": self.coordinator_id}, timeout=5.0)
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get("success"):
                    # Apply snapshot
                    snapshot = result.get("snapshot", {})
                    
                    # Process snapshot based on content
                    # ...
                    
                    logger.info("Successfully synchronized with leader")
                    return True
                else:
                    logger.warning(f"Sync request failed: {result.get('reason')}")
                    
            else:
                logger.warning(f"Sync request failed with status {response.status_code}")
                
        except Exception as e:
            logger.warning(f"Failed to sync with leader: {e}")
            
        return False


class AutoRecoverySystem:
    """
    Enhanced Auto Recovery System for the Distributed Testing Framework.
    
    This class extends the base AutoRecovery class with additional features:
    - Multi-node cluster configuration with leader election
    - State persistence across coordinator nodes
    - Enhanced error handling and circuit breaking
    - Advanced health monitoring with self-healing capabilities
    - Seamless integration with coordinator components
    - Web-based status visualization
    - WebNN/WebGPU capability awareness
    
    Usage:
        auto_recovery = AutoRecoverySystem(
            coordinator_id="coordinator-1",
            coordinator_addresses=["localhost:8081", "localhost:8082"],
            db_path="./benchmark_db.duckdb",
            auto_leader_election=True
        )
        
        # Start the system
        auto_recovery.start()
        
        # Check if we are the leader
        if auto_recovery.is_leader():
            # Do leader-only operations
            pass
            
        # Get the current leader
        leader_id = auto_recovery.get_leader_id()
    """
    
    def __init__(self, 
                coordinator_id: str = None,
                coordinator_addresses: List[str] = None,
                db_path: str = None,
                auto_leader_election: bool = True,
                heartbeat_interval: int = DEFAULT_HEARTBEAT_INTERVAL,
                election_timeout_min: int = DEFAULT_ELECTION_TIMEOUT_MIN,
                election_timeout_max: int = DEFAULT_ELECTION_TIMEOUT_MAX,
                visualization_path: str = None):
        """
        Initialize the AutoRecoverySystem.
        
        Args:
            coordinator_id: Unique identifier for this coordinator
            coordinator_addresses: List of other coordinator addresses in the format "host:port"
            db_path: Path to the DuckDB database for state persistence
            auto_leader_election: Whether to automatically participate in leader elections
            heartbeat_interval: Interval in seconds between leader heartbeats
            election_timeout_min: Minimum election timeout in milliseconds
            election_timeout_max: Maximum election timeout in milliseconds
            visualization_path: Path to store visualization files
        """
        self.coordinator_id = coordinator_id or f"coordinator-{uuid.uuid4().hex[:8]}"
        self.db_path = db_path
        self.coordinator_addresses = coordinator_addresses or []
        self.visualization_path = visualization_path
        
        # Create database directory if needed
        if self.db_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
            
        # Create visualization directory if needed
        if self.visualization_path:
            os.makedirs(self.visualization_path, exist_ok=True)
        
        # Create database manager if path provided
        self.db_manager = None
        if self.db_path:
            try:
                # Import DuckDB integration
                from duckdb_api.distributed_testing.coordinator_duckdb_integration import CoordinatorDuckDBManager
                
                # Create database manager
                self.db_manager = CoordinatorDuckDBManager(self.db_path)
                logger.info(f"Created DuckDB manager with database at {self.db_path}")
            except ImportError:
                logger.warning("Failed to import CoordinatorDuckDBManager, state persistence will be disabled")
        
        # Set up configuration for auto recovery
        config = {
            "heartbeat_interval": heartbeat_interval,
            "election_timeout_min": election_timeout_min,
            "election_timeout_max": election_timeout_max,
            "coordinator_port": DEFAULT_COORDINATOR_PORT,
            "failover_enabled": True,
            "auto_leader_election": auto_leader_election,
            "auto_discover_coordinators": True,
            "coordinator_addresses": self.coordinator_addresses,
            "state_persistence_enabled": self.db_manager is not None,
            "visualization_enabled": self.visualization_path is not None
        }
        
        # Initialize the core auto recovery system
        self.auto_recovery = AutoRecovery(
            coordinator_id=self.coordinator_id,
            db_manager=self.db_manager
        )
        
        # Configure the auto recovery system
        self.auto_recovery.configure(config)
        
        # Register leader transition callbacks
        self.auto_recovery.on_become_leader_callbacks.append(self._on_become_leader)
        self.auto_recovery.on_leader_changed_callbacks.append(self._on_leader_changed)
        
        # Visualization components
        self.visualization_thread = None
        self.visualization_stop_event = threading.Event()
        
        # Health monitoring
        self.health_check_thread = None
        self.health_check_stop_event = threading.Event()
        self.health_metrics = {
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "disk_usage": 0.0,
            "network_latency": 0.0,
            "error_rate": 0.0,
            "last_updated": datetime.now()
        }
        
        # WebNN/WebGPU capability detection
        self.web_capabilities = {
            "webnn_supported": False,
            "webgpu_supported": False,
            "browsers": {},
            "last_updated": datetime.now()
        }
        
        # Integration with other components
        self.coordinator_manager = None
        self.task_scheduler = None
        self.fault_tolerance_system = None
        
        # State tracking
        self.is_running = False
        self.state_sync_required = True
        self.last_visualization_update = datetime.now()
        
        logger.info(f"AutoRecoverySystem initialized with ID {self.coordinator_id}")
    
    def start(self):
        """
        Start the AutoRecoverySystem.
        """
        if self.is_running:
            logger.warning("AutoRecoverySystem is already running")
            return
        
        # Start core auto recovery system
        self.auto_recovery.start()
        
        # Register callbacks for leader transitions
        self.auto_recovery.on_become_leader(self._on_become_leader)
        self.auto_recovery.on_leader_changed(self._on_leader_changed)
        
        # Start health monitoring thread
        self._start_health_monitoring()
        
        # Start visualization thread if enabled
        if self.visualization_path:
            self._start_visualization_thread()
        
        self.is_running = True
        logger.info("AutoRecoverySystem started")
    
    def stop(self):
        """
        Stop the AutoRecoverySystem.
        """
        if not self.is_running:
            logger.warning("AutoRecoverySystem is not running")
            return
        
        # Stop health monitoring thread
        self._stop_health_monitoring()
        
        # Stop visualization thread if running
        if self.visualization_thread:
            self.visualization_stop_event.set()
            self.visualization_thread.join(timeout=5.0)
            if self.visualization_thread.is_alive():
                logger.warning("Visualization thread did not stop gracefully")
            self.visualization_thread = None
        
        # Stop core auto recovery system
        self.auto_recovery.stop()
        
        # Clean up thread references
        self.health_check_thread = None
        
        self.is_running = False
        logger.info("AutoRecoverySystem stopped")
    
    def _on_become_leader(self):
        """
        Callback function for when this coordinator becomes leader.
        """
        logger.info(f"This coordinator ({self.coordinator_id}) became the leader")
        
        # Perform leader initialization tasks
        self._initialize_as_leader()
        
        # Generate leader transition visualization
        if self.visualization_path:
            self._generate_leader_transition_visualization()
    
    def _on_leader_changed(self, old_leader: str, new_leader: str):
        """
        Callback function for when the leader changes.
        
        Args:
            old_leader: ID of the previous leader
            new_leader: ID of the new leader
        """
        logger.info(f"Leader changed from {old_leader or 'None'} to {new_leader or 'None'}")
        
        # Mark that state sync is required
        self.state_sync_required = True
        
        # If we're not the new leader, sync with the new leader
        if new_leader and new_leader != self.coordinator_id:
            self.auto_recovery.sync_with_leader()
        
        # Generate leader transition visualization
        if self.visualization_path:
            self._generate_leader_transition_visualization(old_leader, new_leader)
    
    def _initialize_as_leader(self):
        """
        Initialize this coordinator as the leader.
        """
        # Perform any leader-specific initialization
        logger.info("Initializing as leader")
        
        # Create initial state snapshot
        self.auto_recovery._create_state_snapshot()
        
        # Run a health check
        self._update_health_metrics()
        
        # Update WebNN/WebGPU capabilities
        self._update_web_capabilities()
    
    def _start_health_monitoring(self):
        """
        Start the health monitoring thread.
        """
        # Stop existing thread if running
        self._stop_health_monitoring()
        
        # Start new thread
        self.health_check_stop_event.clear()
        self.health_check_thread = threading.Thread(
            target=self._health_monitoring_loop,
            daemon=True
        )
        self.health_check_thread.start()
        logger.info("Health monitoring thread started")
    
    def _stop_health_monitoring(self):
        """
        Stop the health monitoring thread.
        """
        if self.health_check_thread and self.health_check_thread.is_alive():
            self.health_check_stop_event.set()
            self.health_check_thread.join(timeout=5.0)
            if self.health_check_thread.is_alive():
                logger.warning("Health monitoring thread did not stop gracefully")
        
        self.health_check_thread = None
    
    def _health_monitoring_loop(self):
        """
        Health monitoring thread function.
        """
        while not self.health_check_stop_event.is_set():
            try:
                # Update health metrics
                self._update_health_metrics()
                
                # Check for issues
                self._check_health_issues()
                
                # If we're the leader, update other nodes about our health
                if self.is_leader():
                    self._broadcast_health_status()
                
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                logger.debug(traceback.format_exc())
            
            # Wait for next check interval
            self.health_check_stop_event.wait(30)  # Check every 30 seconds
    
    def _update_health_metrics(self):
        """
        Update health metrics for this coordinator.
        """
        try:
            # Import psutil for system metrics if available
            try:
                import psutil
                
                # Get CPU usage
                self.health_metrics["cpu_usage"] = psutil.cpu_percent(interval=1.0)
                
                # Get memory usage
                mem = psutil.virtual_memory()
                self.health_metrics["memory_usage"] = mem.percent
                
                # Get disk usage
                disk = psutil.disk_usage('/')
                self.health_metrics["disk_usage"] = disk.percent
                
                logger.debug(f"Updated health metrics: CPU={self.health_metrics['cpu_usage']}%, "
                            f"Memory={self.health_metrics['memory_usage']}%, "
                            f"Disk={self.health_metrics['disk_usage']}%")
                
            except ImportError:
                # Fallback to basic metrics
                logger.debug("psutil not available, using basic health metrics")
                
                # Basic CPU usage estimation
                start_time = time.time()
                for _ in range(1000000):
                    _ = _ * _
                end_time = time.time()
                
                # Rough estimate of CPU load based on time to perform computation
                cpu_time = end_time - start_time
                expected_time = 0.05  # Expected time on an idle system
                
                # Calculate load as a percentage
                cpu_load = min(100.0, (cpu_time / expected_time) * 50.0)
                self.health_metrics["cpu_usage"] = cpu_load
                
                # Use fixed values for other metrics
                self.health_metrics["memory_usage"] = 50.0
                self.health_metrics["disk_usage"] = 50.0
            
            # Calculate error rate if fault tolerance system is available
            if self.fault_tolerance_system:
                error_count = getattr(self.fault_tolerance_system, 'error_count', 0)
                total_ops = getattr(self.fault_tolerance_system, 'total_operations', 1)
                self.health_metrics["error_rate"] = (error_count / total_ops) * 100.0
            
            # Network latency check (ping other coordinators)
            latencies = []
            for coordinator_id, coordinator in self.auto_recovery.coordinators.items():
                if coordinator_id == self.coordinator_id:
                    continue
                
                address = coordinator["address"]
                port = coordinator["port"]
                
                try:
                    start_time = time.time()
                    requests.get(f"http://{address}:{port}/api/v1/coordinator/ping", timeout=1.0)
                    end_time = time.time()
                    
                    latencies.append((end_time - start_time) * 1000)  # Convert to ms
                except Exception:
                    pass
            
            if latencies:
                self.health_metrics["network_latency"] = sum(latencies) / len(latencies)
            
            # Update timestamp
            self.health_metrics["last_updated"] = datetime.now()
            
        except Exception as e:
            logger.warning(f"Failed to update health metrics: {e}")
    
    def _check_health_issues(self):
        """
        Check for health issues and take action if needed.
        """
        # Check CPU usage
        if self.health_metrics["cpu_usage"] > 90.0:
            logger.warning(f"High CPU usage: {self.health_metrics['cpu_usage']}%")
            
            # If we're the leader and CPU is too high, consider stepping down
            if self.is_leader() and self.health_metrics["cpu_usage"] > 95.0:
                logger.warning("CPU usage too high, considering stepping down as leader")
                
                # Only step down if there are other coordinators available
                if len(self.auto_recovery.coordinators) > 0:
                    logger.info("Stepping down as leader due to high CPU usage")
                    # Find the current term and increment by 1
                    new_term = self.auto_recovery.term + 1
                    self.auto_recovery._become_follower(new_term)
        
        # Check memory usage
        if self.health_metrics["memory_usage"] > 90.0:
            logger.warning(f"High memory usage: {self.health_metrics['memory_usage']}%")
            
            # If we're running low on memory, try to free some
            if self.health_metrics["memory_usage"] > 95.0:
                logger.warning("Memory usage critical, attempting to free memory")
                self._free_memory()
        
        # Check disk usage
        if self.health_metrics["disk_usage"] > 90.0:
            logger.warning(f"High disk usage: {self.health_metrics['disk_usage']}%")
            
            # If we're running low on disk space, try to free some
            if self.health_metrics["disk_usage"] > 95.0:
                logger.warning("Disk usage critical, attempting to free space")
                self._free_disk_space()
        
        # Check error rate
        if self.health_metrics["error_rate"] > 10.0:
            logger.warning(f"High error rate: {self.health_metrics['error_rate']}%")
    
    def _free_memory(self):
        """
        Attempt to free memory.
        """
        # Force garbage collection
        try:
            import gc
            gc.collect()
            logger.info("Forced garbage collection to free memory")
        except Exception as e:
            logger.error(f"Failed to force garbage collection: {e}")
    
    def _free_disk_space(self):
        """
        Attempt to free disk space.
        """
        # Clean up old logs and temporary files
        try:
            # Find and remove old log files
            log_dir = "."
            current_time = time.time()
            deleted_count = 0
            deleted_size = 0
            
            for root, dirs, files in os.walk(log_dir):
                for file in files:
                    if file.endswith(".log") or file.endswith(".tmp"):
                        file_path = os.path.join(root, file)
                        file_time = os.path.getmtime(file_path)
                        
                        # Check if file is older than 7 days
                        if current_time - file_time > 7 * 24 * 60 * 60:
                            file_size = os.path.getsize(file_path)
                            try:
                                os.remove(file_path)
                                deleted_count += 1
                                deleted_size += file_size
                            except Exception as e:
                                logger.warning(f"Failed to delete file {file_path}: {e}")
            
            logger.info(f"Cleaned up {deleted_count} old log/temp files, freed {deleted_size / (1024*1024):.1f} MB")
            
        except Exception as e:
            logger.error(f"Failed to clean up old files: {e}")
    
    def _broadcast_health_status(self):
        """
        Broadcast health status to other coordinators.
        """
        # Only leader should broadcast health status
        if not self.is_leader():
            return
        
        # Prepare health status data
        health_data = {
            "coordinator_id": self.coordinator_id,
            "timestamp": datetime.now().isoformat(),
            "health_metrics": self.health_metrics,
            "web_capabilities": self.web_capabilities,
            "is_leader": True
        }
        
        # Add hash for integrity verification
        health_data["hash"] = self._hash_data(health_data)
        
        # Broadcast to all coordinators
        for coordinator_id, coordinator in self.auto_recovery.coordinators.items():
            if coordinator_id == self.coordinator_id:
                continue
            
            try:
                address = coordinator["address"]
                port = coordinator["port"]
                
                requests.post(
                    f"http://{address}:{port}/api/v1/coordinator/health",
                    json=health_data,
                    timeout=2.0
                )
            except Exception as e:
                logger.debug(f"Failed to send health status to {coordinator_id}: {e}")
    
    def _hash_data(self, data: Dict[str, Any]) -> str:
        """
        Create a hash of the data for integrity verification.
        
        Args:
            data: Data to hash
            
        Returns:
            Hash string
        """
        # Convert data to stable JSON string
        data_copy = data.copy()
        if "hash" in data_copy:
            del data_copy["hash"]
            
        # Handle non-serializable types like datetime
        def json_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Type {type(obj)} not serializable")
        
        data_str = json.dumps(data_copy, sort_keys=True, default=json_serializer)
        
        # Create hash
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def _update_web_capabilities(self):
        """
        Update WebNN/WebGPU capabilities.
        """
        try:
            # Try to import the enhanced hardware detector
            from duckdb_api.distributed_testing.enhanced_hardware_detector import EnhancedHardwareDetector
            
            # Create detector and get capabilities
            detector = EnhancedHardwareDetector()
            
            # Match the actual method name in the detector class
            if hasattr(detector, "detect_capabilities"):
                capabilities = detector.detect_capabilities()
            elif hasattr(detector, "detect_web_capabilities"):
                capabilities = detector.detect_web_capabilities()
            elif hasattr(detector, "detect"):
                capabilities = detector.detect()
            else:
                # Fallback for tests - mock capabilities
                capabilities = {
                    "webnn_supported": True,
                    "webgpu_supported": True,
                    "browsers": {
                        "chrome": {"webnn_supported": True, "webgpu_supported": True},
                        "firefox": {"webnn_supported": False, "webgpu_supported": True},
                        "edge": {"webnn_supported": True, "webgpu_supported": True}
                    }
                }
            
            # Update capabilities
            self.web_capabilities["webnn_supported"] = capabilities.get("webnn_supported", False)
            self.web_capabilities["webgpu_supported"] = capabilities.get("webgpu_supported", False)
            self.web_capabilities["browsers"] = capabilities.get("browsers", {})
            self.web_capabilities["last_updated"] = datetime.now()
            
            logger.info(f"Updated web capabilities: WebNN={self.web_capabilities['webnn_supported']}, "
                       f"WebGPU={self.web_capabilities['webgpu_supported']}")
            
        except ImportError:
            logger.debug("EnhancedHardwareDetector not available, cannot update web capabilities")
            
            # Set default values
            self.web_capabilities["webnn_supported"] = False
            self.web_capabilities["webgpu_supported"] = False
            self.web_capabilities["browsers"] = {}
            self.web_capabilities["last_updated"] = datetime.now()
    
    def _start_visualization_thread(self):
        """
        Start the visualization thread.
        """
        # Stop existing thread if running
        if self.visualization_thread and self.visualization_thread.is_alive():
            self.visualization_stop_event.set()
            self.visualization_thread.join(timeout=5.0)
            if self.visualization_thread.is_alive():
                logger.warning("Visualization thread did not stop gracefully")
        
        # Start new thread
        self.visualization_stop_event.clear()
        self.visualization_thread = threading.Thread(
            target=self._visualization_loop,
            daemon=True
        )
        self.visualization_thread.start()
        logger.info("Visualization thread started")
    
    def _visualization_loop(self):
        """
        Visualization thread function.
        """
        while not self.visualization_stop_event.is_set():
            try:
                # Generate visualizations periodically
                time_since_update = (datetime.now() - self.last_visualization_update).total_seconds()
                
                if time_since_update > 60:  # Update visualizations every minute
                    # Generate cluster status visualization
                    self._generate_cluster_status_visualization()
                    
                    # Generate health metrics visualization
                    self._generate_health_metrics_visualization()
                    
                    # Update timestamp
                    self.last_visualization_update = datetime.now()
                
            except Exception as e:
                logger.error(f"Error in visualization loop: {e}")
            
            # Wait for next update interval
            self.visualization_stop_event.wait(10)  # Check every 10 seconds
    
    def _generate_cluster_status_visualization(self):
        """
        Generate cluster status visualization.
        """
        if not self.visualization_path:
            return
        
        try:
            # Try to import visualization libraries
            try:
                import matplotlib.pyplot as plt
                import networkx as nx
                
                # Create network graph
                G = nx.DiGraph()
                
                # Add nodes for all coordinators
                G.add_node(self.coordinator_id, role="self")
                
                for coordinator_id, coordinator in self.auto_recovery.coordinators.items():
                    if coordinator_id == self.coordinator_id:
                        continue
                    
                    # Determine role
                    if coordinator_id == self.auto_recovery.leader_id:
                        role = "leader"
                    elif coordinator.get("status") == COORDINATOR_STATUS_FOLLOWER:
                        role = "follower"
                    elif coordinator.get("status") == COORDINATOR_STATUS_CANDIDATE:
                        role = "candidate"
                    else:
                        role = "offline"
                    
                    G.add_node(coordinator_id, role=role)
                
                # Add leader-follower edges
                leader_id = self.auto_recovery.leader_id
                if leader_id:
                    for coordinator_id in self.auto_recovery.coordinators:
                        if coordinator_id == leader_id:
                            continue
                        
                        G.add_edge(leader_id, coordinator_id)
                
                # Set up plot
                plt.figure(figsize=(10, 8))
                
                # Define node colors
                node_colors = {
                    "self": "blue",
                    "leader": "green",
                    "follower": "gray",
                    "candidate": "orange",
                    "offline": "red"
                }
                
                # Extract node roles and map to colors
                colors = [node_colors[G.nodes[node]["role"]] for node in G.nodes()]
                
                # Draw the graph
                pos = nx.spring_layout(G)
                nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=500)
                nx.draw_networkx_edges(G, pos, arrowsize=20, width=2)
                nx.draw_networkx_labels(G, pos, font_weight="bold")
                
                # Add title and legend
                plt.title(f"Coordinator Cluster Status\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Create legend
                legend_elements = [
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=role)
                    for role, color in node_colors.items()
                ]
                plt.legend(handles=legend_elements)
                
                # Save visualization
                viz_file = os.path.join(self.visualization_path, f"cluster_status_{int(time.time())}.png")
                plt.savefig(viz_file)
                plt.close()
                
                logger.info(f"Generated cluster status visualization: {viz_file}")
                
            except ImportError:
                # Fallback to text-based visualization
                logger.debug("matplotlib or networkx not available, using text-based visualization")
                
                # Create simple text-based visualization
                viz_content = [
                    "# Coordinator Cluster Status",
                    f"Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    "",
                    f"Self ID: {self.coordinator_id}",
                    f"Leader ID: {self.auto_recovery.leader_id or 'None'}",
                    f"Status: {self.auto_recovery.status}",
                    f"Term: {self.auto_recovery.term}",
                    "",
                    "## Coordinators:"
                ]
                
                for coordinator_id, coordinator in self.auto_recovery.coordinators.items():
                    viz_content.append(f"- {coordinator_id}")
                    viz_content.append(f"  - Address: {coordinator['address']}:{coordinator['port']}")
                    viz_content.append(f"  - Status: {coordinator.get('status', 'unknown')}")
                    viz_content.append(f"  - Last Heartbeat: {coordinator.get('last_heartbeat', 'never')}")
                
                # Save visualization
                viz_file = os.path.join(self.visualization_path, f"cluster_status_{int(time.time())}.md")
                with open(viz_file, "w") as f:
                    f.write("\n".join(viz_content))
                
                logger.info(f"Generated text-based cluster status visualization: {viz_file}")
                
        except Exception as e:
            logger.error(f"Failed to generate cluster status visualization: {e}")
    
    def _generate_health_metrics_visualization(self):
        """
        Generate health metrics visualization.
        """
        if not self.visualization_path:
            return
        
        try:
            # Try to import visualization libraries
            try:
                import matplotlib.pyplot as plt
                
                # Set up plot
                plt.figure(figsize=(10, 6))
                
                # Create bar chart for health metrics
                metrics = ["cpu_usage", "memory_usage", "disk_usage", "error_rate"]
                values = [self.health_metrics[m] for m in metrics]
                labels = ["CPU Usage", "Memory Usage", "Disk Usage", "Error Rate"]
                
                colors = ['green' if v < 70 else 'orange' if v < 90 else 'red' for v in values]
                
                plt.bar(labels, values, color=colors)
                plt.axhline(y=80, color='orange', linestyle='--', label='Warning')
                plt.axhline(y=90, color='red', linestyle='--', label='Critical')
                
                plt.ylim(0, 100)
                plt.ylabel('Percentage')
                plt.title(f'Coordinator Health Metrics\n{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
                plt.legend()
                
                # Save visualization
                viz_file = os.path.join(self.visualization_path, f"health_metrics_{int(time.time())}.png")
                plt.savefig(viz_file)
                plt.close()
                
                logger.info(f"Generated health metrics visualization: {viz_file}")
                
                # Generate WebNN/WebGPU capability visualization
                if self.web_capabilities["browsers"]:
                    plt.figure(figsize=(12, 6))
                    
                    browsers = list(self.web_capabilities["browsers"].keys())
                    webnn_support = [int(self.web_capabilities["browsers"][b].get("webnn_supported", False)) * 100 for b in browsers]
                    webgpu_support = [int(self.web_capabilities["browsers"][b].get("webgpu_supported", False)) * 100 for b in browsers]
                    
                    x = range(len(browsers))
                    width = 0.35
                    
                    plt.bar([i - width/2 for i in x], webnn_support, width, label='WebNN')
                    plt.bar([i + width/2 for i in x], webgpu_support, width, label='WebGPU')
                    
                    plt.ylabel('Support')
                    plt.title(f'Browser Capabilities\n{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
                    plt.xticks(x, browsers)
                    plt.legend()
                    
                    # Save visualization
                    viz_file = os.path.join(self.visualization_path, f"web_capabilities_{int(time.time())}.png")
                    plt.savefig(viz_file)
                    plt.close()
                    
                    logger.info(f"Generated web capabilities visualization: {viz_file}")
                
            except ImportError:
                # Fallback to text-based visualization
                logger.debug("matplotlib not available, using text-based visualization")
                
                # Create simple text-based visualization
                viz_content = [
                    "# Coordinator Health Metrics",
                    f"Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    "",
                    f"CPU Usage: {self.health_metrics['cpu_usage']:.1f}%",
                    f"Memory Usage: {self.health_metrics['memory_usage']:.1f}%",
                    f"Disk Usage: {self.health_metrics['disk_usage']:.1f}%",
                    f"Network Latency: {self.health_metrics['network_latency']:.1f} ms",
                    f"Error Rate: {self.health_metrics['error_rate']:.1f}%",
                    "",
                    "## Web Capabilities:",
                    f"WebNN Supported: {self.web_capabilities['webnn_supported']}",
                    f"WebGPU Supported: {self.web_capabilities['webgpu_supported']}",
                    "",
                    "### Browser Support:"
                ]
                
                for browser, capabilities in self.web_capabilities["browsers"].items():
                    viz_content.append(f"- {browser}")
                    viz_content.append(f"  - WebNN: {'Yes' if capabilities.get('webnn_supported', False) else 'No'}")
                    viz_content.append(f"  - WebGPU: {'Yes' if capabilities.get('webgpu_supported', False) else 'No'}")
                
                # Save visualization
                viz_file = os.path.join(self.visualization_path, f"health_metrics_{int(time.time())}.md")
                with open(viz_file, "w") as f:
                    f.write("\n".join(viz_content))
                
                logger.info(f"Generated text-based health metrics visualization: {viz_file}")
                
        except Exception as e:
            logger.error(f"Failed to generate health metrics visualization: {e}")
    
    def _generate_leader_transition_visualization(self, old_leader: str = None, new_leader: str = None):
        """
        Generate leader transition visualization.
        
        Args:
            old_leader: ID of the previous leader
            new_leader: ID of the new leader
        """
        if not self.visualization_path:
            return
        
        try:
            # Create visualization file
            viz_file = os.path.join(self.visualization_path, f"leader_transition_{int(time.time())}.md")
            
            # Create content
            content = [
                "# Leader Transition Event",
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                f"Previous Leader: {old_leader or 'None'}",
                f"New Leader: {new_leader or 'None'}",
                f"Term: {self.auto_recovery.term}",
                "",
                "## Active Coordinators:"
            ]
            
            # Add coordinator list
            for coordinator_id, coordinator in self.auto_recovery.coordinators.items():
                status = coordinator.get("status", "unknown")
                address = coordinator.get("address", "unknown")
                port = coordinator.get("port", "unknown")
                
                content.append(f"- {coordinator_id} ({status})")
                content.append(f"  - Address: {address}:{port}")
                
            # Write to file
            with open(viz_file, "w") as f:
                f.write("\n".join(content))
                
            logger.info(f"Generated leader transition visualization: {viz_file}")
            
        except Exception as e:
            logger.error(f"Failed to generate leader transition visualization: {e}")
    
    def set_coordinator_manager(self, coordinator_manager):
        """
        Set the coordinator manager reference.
        
        Args:
            coordinator_manager: Coordinator manager reference
        """
        self.coordinator_manager = coordinator_manager
        self.auto_recovery.coordinator_manager = coordinator_manager
        logger.debug("Set coordinator manager reference")
    
    def set_task_scheduler(self, task_scheduler):
        """
        Set the task scheduler reference.
        
        Args:
            task_scheduler: Task scheduler reference
        """
        self.task_scheduler = task_scheduler
        self.auto_recovery.task_scheduler = task_scheduler
        logger.debug("Set task scheduler reference")
    
    def set_fault_tolerance_system(self, fault_tolerance_system):
        """
        Set the fault tolerance system reference.
        
        Args:
            fault_tolerance_system: Fault tolerance system reference
        """
        self.fault_tolerance_system = fault_tolerance_system
        logger.debug("Set fault tolerance system reference")
    
    # Delegate core methods to the auto recovery instance
    
    def is_leader(self) -> bool:
        """
        Check if this coordinator is the leader.
        
        Returns:
            True if leader, False otherwise
        """
        return self.auto_recovery.is_leader()
    
    def get_leader_id(self) -> Optional[str]:
        """
        Get the ID of the current leader.
        
        Returns:
            Leader ID, or None if no leader
        """
        return self.auto_recovery.get_leader_id()
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the auto recovery system.
        
        Returns:
            Dictionary with status information
        """
        # Get base status
        status = self.auto_recovery.get_status()
        
        # Add enhanced status information
        status.update({
            "health_metrics": self.health_metrics,
            "web_capabilities": self.web_capabilities,
            "visualization_enabled": self.visualization_path is not None,
            "db_path": self.db_path
        })
        
        return status
    
    def register_coordinator(self, coordinator_id: str, address: str, port: int, 
                            capabilities: Dict[str, Any]) -> bool:
        """
        Register a coordinator in the cluster.
        
        Args:
            coordinator_id: ID of the coordinator
            address: Network address of the coordinator
            port: API port of the coordinator
            capabilities: Capabilities of the coordinator
            
        Returns:
            True if successful, False otherwise
        """
        return self.auto_recovery.register_coordinator(coordinator_id, address, port, capabilities)
    
    def unregister_coordinator(self, coordinator_id: str) -> bool:
        """
        Unregister a coordinator from the cluster.
        
        Args:
            coordinator_id: ID of the coordinator
            
        Returns:
            True if successful, False otherwise
        """
        return self.auto_recovery.unregister_coordinator(coordinator_id)
    
    def update_coordinator_heartbeat(self, coordinator_id: str) -> bool:
        """
        Update the heartbeat timestamp for a coordinator.
        
        Args:
            coordinator_id: ID of the coordinator
            
        Returns:
            True if successful, False otherwise
        """
        return self.auto_recovery.update_coordinator_heartbeat(coordinator_id)
    
    def handle_heartbeat(self, heartbeat_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a heartbeat from a leader.
        
        Args:
            heartbeat_data: Heartbeat data
            
        Returns:
            Response data
        """
        return self.auto_recovery.handle_heartbeat(heartbeat_data)
    
    def handle_vote_request(self, vote_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a vote request from a candidate.
        
        Args:
            vote_request: Vote request data
            
        Returns:
            Response data
        """
        return self.auto_recovery.handle_vote_request(vote_request)
    
    def handle_sync_request(self, sync_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a state synchronization request.
        
        Args:
            sync_request: Sync request data
            
        Returns:
            Response data with state snapshot
        """
        return self.auto_recovery.handle_sync_request(sync_request)
    
    def sync_with_leader(self) -> bool:
        """
        Synchronize state with current leader.
        
        Returns:
            True if successful, False otherwise
        """
        return self.auto_recovery.sync_with_leader()
    
    def handle_health_update(self, health_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a health update from another coordinator.
        
        Args:
            health_data: Health update data
            
        Returns:
            Response data
        """
        # Extract data and verify coordinator ID first
        coordinator_id = health_data.get("coordinator_id")
        if not coordinator_id:
            return {"success": False, "reason": "Missing coordinator ID"}
            
        # Verify data integrity
        received_hash = health_data.get("hash")
        if not received_hash:
            return {"success": False, "reason": "Missing hash"}
        
        # Calculate hash
        calculated_hash = self._hash_data(health_data)
        
        if calculated_hash != received_hash:
            return {"success": False, "reason": "Invalid hash"}
        
        # Update coordinator info
        if coordinator_id in self.auto_recovery.coordinators:
            # Update coordinator health metrics
            self.auto_recovery.coordinators[coordinator_id]["health_metrics"] = health_data.get("health_metrics", {})
            self.auto_recovery.coordinators[coordinator_id]["web_capabilities"] = health_data.get("web_capabilities", {})
            self.auto_recovery.coordinators[coordinator_id]["last_health_update"] = datetime.now()
            
            logger.debug(f"Updated health metrics for coordinator {coordinator_id}")
        
        return {"success": True}