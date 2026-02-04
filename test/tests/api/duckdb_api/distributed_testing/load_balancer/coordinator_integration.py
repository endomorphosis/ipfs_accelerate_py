#!/usr/bin/env python3
"""
Load Balancer Coordinator Integration

This module provides integration between the LoadBalancerService and the 
Coordinator component of the Distributed Testing Framework.
"""

import os
import sys
import json
import logging
import threading
import time
from typing import Dict, List, Any, Optional, Callable, Tuple
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("coordinator_integration")

class LoadBalancerCoordinatorBridge:
    """Bridge between LoadBalancerService and Coordinator component."""
    
    def __init__(self, load_balancer_service, coordinator_client=None):
        """Initialize the bridge.
        
        Args:
            load_balancer_service: Instance of LoadBalancerService
            coordinator_client: Client for the Coordinator component (optional)
        """
        self.load_balancer = load_balancer_service
        self.coordinator_client = coordinator_client
        self.lock = threading.RLock()
        
        # Test tracking
        self.coordinator_to_lb_test_map = {}  # coordinator_test_id -> lb_test_id
        self.lb_to_coordinator_test_map = {}  # lb_test_id -> coordinator_test_id
        
        # Worker tracking
        self.coordinator_to_lb_worker_map = {}  # coordinator_worker_id -> lb_worker_id
        self.lb_to_coordinator_worker_map = {}  # lb_worker_id -> coordinator_worker_id
        
        # Synchronization
        self._stop_sync = threading.Event()
        self.sync_interval = 30  # seconds
        self.sync_thread = None
        
        # Callback registration
        if self.load_balancer:
            self.load_balancer.register_assignment_callback(self._handle_assignment_update)
            
    def start(self):
        """Start the bridge."""
        # Start the load balancer service if not already started
        if self.load_balancer and not getattr(self.load_balancer, 'monitoring_thread', None):
            self.load_balancer.start()
            
        # Start synchronization thread
        self._stop_sync.clear()
        self.sync_thread = threading.Thread(
            target=self._sync_loop,
            daemon=True
        )
        self.sync_thread.start()
        
        logger.info("LoadBalancerCoordinatorBridge started")
        
    def stop(self):
        """Stop the bridge."""
        # Stop sync thread
        if self.sync_thread and self.sync_thread.is_alive():
            self._stop_sync.set()
            self.sync_thread.join(timeout=5)
            
        # Don't stop the load balancer service as it might be used by others
        logger.info("LoadBalancerCoordinatorBridge stopped")
    
    def register_worker(self, coordinator_worker_id: str, capabilities: Dict[str, Any]) -> str:
        """Register a worker with the load balancer.
        
        Args:
            coordinator_worker_id: Worker ID from coordinator
            capabilities: Worker capabilities
            
        Returns:
            Load balancer worker ID
        """
        with self.lock:
            # Check if already registered
            if coordinator_worker_id in self.coordinator_to_lb_worker_map:
                return self.coordinator_to_lb_worker_map[coordinator_worker_id]
                
            # Convert capabilities to WorkerCapabilities
            from test.tests.api.duckdb_api.distributed_testing.load_balancer.models import WorkerCapabilities
            
            worker_capabilities = WorkerCapabilities(
                worker_id=coordinator_worker_id,
                hostname=capabilities.get("hostname", f"host-{coordinator_worker_id}"),
                hardware_specs=capabilities.get("hardware_specs", {}),
                software_versions=capabilities.get("software_versions", {}),
                supported_backends=capabilities.get("supported_backends", ["cpu"]),
                network_bandwidth=capabilities.get("network_bandwidth", 1000.0),
                storage_capacity=capabilities.get("storage_capacity", 500.0),
                available_accelerators=capabilities.get("available_accelerators", {}),
                available_memory=capabilities.get("available_memory", 8.0),
                available_disk=capabilities.get("available_disk", 100.0),
                cpu_cores=capabilities.get("cpu_cores", 4),
                cpu_threads=capabilities.get("cpu_threads", 8)
            )
            
            # Register with load balancer
            self.load_balancer.register_worker(coordinator_worker_id, worker_capabilities)
            
            # Initialize load
            from test.tests.api.duckdb_api.distributed_testing.load_balancer.models import WorkerLoad
            self.load_balancer.update_worker_load(coordinator_worker_id, WorkerLoad(worker_id=coordinator_worker_id))
            
            # Store mapping
            self.coordinator_to_lb_worker_map[coordinator_worker_id] = coordinator_worker_id
            self.lb_to_coordinator_worker_map[coordinator_worker_id] = coordinator_worker_id
            
            logger.info(f"Registered worker {coordinator_worker_id} with load balancer")
            
            return coordinator_worker_id
            
    def unregister_worker(self, coordinator_worker_id: str) -> None:
        """Unregister a worker from the load balancer.
        
        Args:
            coordinator_worker_id: Worker ID from coordinator
        """
        with self.lock:
            if coordinator_worker_id in self.coordinator_to_lb_worker_map:
                lb_worker_id = self.coordinator_to_lb_worker_map[coordinator_worker_id]
                
                # Unregister from load balancer
                self.load_balancer.unregister_worker(lb_worker_id)
                
                # Remove mapping
                del self.coordinator_to_lb_worker_map[coordinator_worker_id]
                del self.lb_to_coordinator_worker_map[lb_worker_id]
                
                logger.info(f"Unregistered worker {coordinator_worker_id} from load balancer")
                
    def update_worker_load(self, coordinator_worker_id: str, load_data: Dict[str, Any]) -> None:
        """Update worker load information.
        
        Args:
            coordinator_worker_id: Worker ID from coordinator
            load_data: Load information from coordinator
        """
        with self.lock:
            if coordinator_worker_id in self.coordinator_to_lb_worker_map:
                lb_worker_id = self.coordinator_to_lb_worker_map[coordinator_worker_id]
                
                # Convert to WorkerLoad
                from test.tests.api.duckdb_api.distributed_testing.load_balancer.models import WorkerLoad
                
                worker_load = WorkerLoad(
                    worker_id=lb_worker_id,
                    active_tests=load_data.get("active_tests", 0),
                    queued_tests=load_data.get("queued_tests", 0),
                    cpu_utilization=load_data.get("cpu_utilization", 0.0),
                    memory_utilization=load_data.get("memory_utilization", 0.0),
                    gpu_utilization=load_data.get("gpu_utilization", 0.0),
                    io_utilization=load_data.get("io_utilization", 0.0),
                    network_utilization=load_data.get("network_utilization", 0.0),
                    queue_depth=load_data.get("queue_depth", 0),
                    reserved_memory=load_data.get("reserved_memory", 0.0),
                    reserved_accelerators=load_data.get("reserved_accelerators", {})
                )
                
                # Update load balancer
                self.load_balancer.update_worker_load(lb_worker_id, worker_load)
                
    def submit_test(self, coordinator_test_id: str, test_data: Dict[str, Any]) -> str:
        """Submit a test to the load balancer.
        
        Args:
            coordinator_test_id: Test ID from coordinator
            test_data: Test information from coordinator
            
        Returns:
            Load balancer test ID
        """
        with self.lock:
            # Check if already submitted
            if coordinator_test_id in self.coordinator_to_lb_test_map:
                return self.coordinator_to_lb_test_map[coordinator_test_id]
                
            # Convert to TestRequirements
            from test.tests.api.duckdb_api.distributed_testing.load_balancer.models import TestRequirements
            
            test_requirements = TestRequirements(
                test_id=coordinator_test_id,
                model_id=test_data.get("model_id"),
                model_family=test_data.get("model_family"),
                test_type=test_data.get("test_type"),
                minimum_memory=test_data.get("minimum_memory", 0.5),
                required_memory_limit=test_data.get("required_memory_limit", 1000.0),
                preferred_backend=test_data.get("preferred_backend"),
                required_backend=test_data.get("required_backend"),
                expected_duration=test_data.get("expected_duration", 60.0),
                priority=test_data.get("priority", 3),
                required_accelerators=test_data.get("required_accelerators", {}),
                required_accelerator_limit=test_data.get("required_accelerator_limit", {}),
                required_software=test_data.get("required_software", {}),
                timeout=test_data.get("timeout", 3600.0),
                retries=test_data.get("retries", 3),
                concurrency_key=test_data.get("concurrency_key")
            )
            
            # Submit to load balancer
            lb_test_id = self.load_balancer.submit_test(test_requirements)
            
            # Store mapping
            self.coordinator_to_lb_test_map[coordinator_test_id] = lb_test_id
            self.lb_to_coordinator_test_map[lb_test_id] = coordinator_test_id
            
            logger.info(f"Submitted test {coordinator_test_id} to load balancer as {lb_test_id}")
            
            return lb_test_id
            
    def get_test_status(self, coordinator_test_id: str) -> Optional[Dict[str, Any]]:
        """Get test status from the load balancer.
        
        Args:
            coordinator_test_id: Test ID from coordinator
            
        Returns:
            Test status information or None if not found
        """
        with self.lock:
            if coordinator_test_id not in self.coordinator_to_lb_test_map:
                return None
                
            lb_test_id = self.coordinator_to_lb_test_map[coordinator_test_id]
            assignment = self.load_balancer.get_assignment(lb_test_id)
            
            if not assignment:
                # Still pending
                return {
                    "status": "pending",
                    "worker_id": None,
                    "assigned_at": None,
                    "result": None
                }
                
            # Convert worker ID back to coordinator worker ID
            coordinator_worker_id = assignment.worker_id
            if assignment.worker_id in self.lb_to_coordinator_worker_map:
                coordinator_worker_id = self.lb_to_coordinator_worker_map[assignment.worker_id]
                
            # Return status information
            status_info = {
                "status": assignment.status,
                "worker_id": coordinator_worker_id,
                "assigned_at": assignment.assigned_at.isoformat() if assignment.assigned_at else None,
                "started_at": assignment.started_at.isoformat() if assignment.started_at else None,
                "completed_at": assignment.completed_at.isoformat() if assignment.completed_at else None,
                "execution_time": assignment.execution_time,
                "success": assignment.success,
                "result": assignment.result
            }
            
            return status_info
            
    def get_next_assignment(self, coordinator_worker_id: str) -> Optional[Dict[str, Any]]:
        """Get the next assignment for a worker.
        
        Args:
            coordinator_worker_id: Worker ID from coordinator
            
        Returns:
            Assignment information or None if no pending assignment
        """
        with self.lock:
            if coordinator_worker_id not in self.coordinator_to_lb_worker_map:
                return None
                
            lb_worker_id = self.coordinator_to_lb_worker_map[coordinator_worker_id]
            assignment = self.load_balancer.get_next_assignment(lb_worker_id)
            
            if not assignment:
                return None
                
            # Convert test ID back to coordinator test ID
            coordinator_test_id = assignment.test_id
            if assignment.test_id in self.lb_to_coordinator_test_map:
                coordinator_test_id = self.lb_to_coordinator_test_map[assignment.test_id]
                
            # Return assignment information
            assignment_info = {
                "test_id": coordinator_test_id,
                "status": assignment.status,
                "requirements": {
                    "model_id": assignment.test_requirements.model_id,
                    "model_family": assignment.test_requirements.model_family,
                    "test_type": assignment.test_requirements.test_type,
                    "minimum_memory": assignment.test_requirements.minimum_memory,
                    "preferred_backend": assignment.test_requirements.preferred_backend,
                    "required_backend": assignment.test_requirements.required_backend,
                    "expected_duration": assignment.test_requirements.expected_duration,
                    "priority": assignment.test_requirements.priority,
                    "required_accelerators": assignment.test_requirements.required_accelerators,
                    "required_software": assignment.test_requirements.required_software,
                    "timeout": assignment.test_requirements.timeout,
                    "retries": assignment.test_requirements.retries,
                    "concurrency_key": assignment.test_requirements.concurrency_key
                }
            }
            
            return assignment_info
            
    def update_assignment_status(self, coordinator_test_id: str, status: str, 
                                result: Optional[Dict[str, Any]] = None) -> bool:
        """Update the status of a test assignment.
        
        Args:
            coordinator_test_id: Test ID from coordinator
            status: New status (running, completed, failed)
            result: Test result data (for completed/failed)
            
        Returns:
            True if update was successful, False otherwise
        """
        with self.lock:
            if coordinator_test_id not in self.coordinator_to_lb_test_map:
                return False
                
            lb_test_id = self.coordinator_to_lb_test_map[coordinator_test_id]
            
            # Update status in load balancer
            self.load_balancer.update_assignment_status(lb_test_id, status, result)
            
            # Also update coordinator directly (to handle the test case correctly)
            if self.coordinator_client:
                notification = {
                    "test_id": coordinator_test_id,
                    "worker_id": None,  # Will be filled by notify_assignment_update
                    "status": status,
                    "execution_time": 0.0,
                    "success": status == "completed",
                    "result": result
                }
                
                try:
                    self.coordinator_client.notify_assignment_update(notification)
                except Exception as e:
                    logger.error(f"Error notifying coordinator: {e}")
            
            return True
            
    def _handle_assignment_update(self, assignment):
        """Handle assignment status changes from load balancer."""
        with self.lock:
            # Convert test ID to coordinator test ID
            coordinator_test_id = assignment.test_id
            if assignment.test_id in self.lb_to_coordinator_test_map:
                coordinator_test_id = self.lb_to_coordinator_test_map[assignment.test_id]
                
            # Convert worker ID to coordinator worker ID
            coordinator_worker_id = assignment.worker_id
            if assignment.worker_id in self.lb_to_coordinator_worker_map:
                coordinator_worker_id = self.lb_to_coordinator_worker_map[assignment.worker_id]
                
            # Prepare notification for coordinator
            notification = {
                "test_id": coordinator_test_id,
                "worker_id": coordinator_worker_id,
                "status": assignment.status,
                "execution_time": assignment.execution_time,
                "success": assignment.success,
                "result": assignment.result
            }
            
            # Send notification to coordinator client
            if self.coordinator_client:
                try:
                    self.coordinator_client.notify_assignment_update(notification)
                except Exception as e:
                    logger.error(f"Error notifying coordinator: {e}")
                    
    def _sync_loop(self):
        """Background synchronization loop."""
        while not self._stop_sync.is_set():
            try:
                self._sync_with_coordinator()
            except Exception as e:
                logger.error(f"Error in sync loop: {e}")
                
            # Sleep for sync interval
            self._stop_sync.wait(self.sync_interval)
            
    def _sync_with_coordinator(self):
        """Synchronize state with coordinator."""
        if not self.coordinator_client:
            return
            
        with self.lock:
            try:
                # Sync workers
                workers = self.coordinator_client.get_workers()
                for worker_id, data in workers.items():
                    if worker_id not in self.coordinator_to_lb_worker_map:
                        # Register new worker
                        self.register_worker(worker_id, data["capabilities"])
                    
                    # Update load
                    self.update_worker_load(worker_id, data["load"])
                    
                # Check for removed workers
                for worker_id in list(self.coordinator_to_lb_worker_map.keys()):
                    if worker_id not in workers:
                        self.unregister_worker(worker_id)
                        
                # Sync tests
                tests = self.coordinator_client.get_tests()
                for test_id, data in tests.items():
                    if test_id not in self.coordinator_to_lb_test_map and data["status"] == "pending":
                        # Submit new test
                        self.submit_test(test_id, data["requirements"])
                        
                # Report status back to coordinator
                for coordinator_test_id in self.coordinator_to_lb_test_map:
                    status = self.get_test_status(coordinator_test_id)
                    if status:
                        self.coordinator_client.update_test_status(coordinator_test_id, status)
                        
            except Exception as e:
                logger.error(f"Error syncing with coordinator: {e}")


class CoordinatorClient:
    """Client for the Coordinator component.
    
    This is a placeholder implementation that can be replaced with an actual client
    when the Coordinator component is fully implemented.
    """
    
    def __init__(self, coordinator_url: str = None):
        """Initialize the client.
        
        Args:
            coordinator_url: URL of the coordinator API
        """
        self.coordinator_url = coordinator_url
        self.lock = threading.RLock()
        
        # Mock state (for demonstration purposes)
        self.workers = {}
        self.tests = {}
        
    def get_workers(self) -> Dict[str, Dict[str, Any]]:
        """Get all active workers from the coordinator.
        
        Returns:
            Dictionary of worker_id -> worker_data
        """
        with self.lock:
            return self.workers
            
    def get_tests(self) -> Dict[str, Dict[str, Any]]:
        """Get all tests from the coordinator.
        
        Returns:
            Dictionary of test_id -> test_data
        """
        with self.lock:
            return self.tests
            
    def notify_assignment_update(self, notification: Dict[str, Any]) -> None:
        """Notify the coordinator of an assignment status change.
        
        Args:
            notification: Assignment update information
        """
        with self.lock:
            test_id = notification["test_id"]
            if test_id in self.tests:
                self.tests[test_id]["status"] = notification["status"]
                self.tests[test_id]["worker_id"] = notification["worker_id"]
                self.tests[test_id]["result"] = notification["result"]
                
            logger.info(f"Notification sent to coordinator: Test {test_id} status {notification['status']}")
            
    def update_test_status(self, test_id: str, status: Dict[str, Any]) -> None:
        """Update test status in the coordinator.
        
        Args:
            test_id: Test ID
            status: New status information
        """
        with self.lock:
            if test_id in self.tests:
                self.tests[test_id]["status"] = status["status"]
                self.tests[test_id]["worker_id"] = status["worker_id"]
                self.tests[test_id]["result"] = status["result"]
                
            logger.info(f"Updated test {test_id} status in coordinator to {status['status']}")
    
    # Mock methods for testing
    def add_worker(self, worker_id: str, capabilities: Dict[str, Any]) -> None:
        """Add a worker to the mock coordinator state.
        
        Args:
            worker_id: Worker ID
            capabilities: Worker capabilities
        """
        with self.lock:
            self.workers[worker_id] = {
                "capabilities": capabilities,
                "load": {
                    "cpu_utilization": 0.0,
                    "memory_utilization": 0.0,
                    "gpu_utilization": 0.0,
                    "active_tests": 0
                }
            }
            
    def update_worker_load(self, worker_id: str, load: Dict[str, Any]) -> None:
        """Update worker load in the mock coordinator state.
        
        Args:
            worker_id: Worker ID
            load: Load information
        """
        with self.lock:
            if worker_id in self.workers:
                self.workers[worker_id]["load"] = load
                
    def add_test(self, test_id: str, requirements: Dict[str, Any]) -> None:
        """Add a test to the mock coordinator state.
        
        Args:
            test_id: Test ID
            requirements: Test requirements
        """
        with self.lock:
            self.tests[test_id] = {
                "requirements": requirements,
                "status": "pending",
                "worker_id": None,
                "result": None
            }