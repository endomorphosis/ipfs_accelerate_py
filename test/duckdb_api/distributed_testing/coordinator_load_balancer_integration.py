#!/usr/bin/env python3
"""
Coordinator Load Balancer Integration

This module integrates the Coordinator with the Load Balancer service to provide
intelligent task distribution based on worker capabilities and current load.
"""

import os
import sys
import time
import json
import logging
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import modules from parent
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import from load balancer
from data.duckdb.distributed_testing.load_balancer.service import LoadBalancerService, create_load_balancer
from data.duckdb.distributed_testing.load_balancer.coordinator_integration import LoadBalancerCoordinatorBridge
from data.duckdb.distributed_testing.load_balancer.models import (
    WorkerCapabilities,
    TestRequirements,
    WorkerLoad
)
from data.duckdb.distributed_testing.load_balancer.task_analyzer import TaskRequirementsAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("coordinator_load_balancer")

class CoordinatorLoadBalancerIntegration:
    """
    Integrates the Coordinator with the LoadBalancerService to provide intelligent
    task distribution based on worker capabilities and current load.
    """
    
    def __init__(self, 
                 coordinator, 
                 load_balancer_config: Optional[Dict[str, Any]] = None,
                 db_path: Optional[str] = None):
        """
        Initialize the integration.
        
        Args:
            coordinator: Instance of the Coordinator
            load_balancer_config: Configuration for the load balancer (optional)
            db_path: Path to the database for persistence (optional)
        """
        self.coordinator = coordinator
        self.db_path = db_path
        
        # Initialize task analyzer for determining test requirements
        self.task_analyzer = TaskRequirementsAnalyzer()
        
        # Initialize load balancer service
        if load_balancer_config:
            self.load_balancer = create_load_balancer(load_balancer_config)
        else:
            self.load_balancer = LoadBalancerService(db_path=db_path)
            
        # Create bridge for integration
        self.bridge = LoadBalancerCoordinatorBridge(
            load_balancer_service=self.load_balancer,
            coordinator_client=self._create_coordinator_client()
        )
        
        # Setup client methods
        self._setup_coordinator_methods()
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        # Synchronization state
        self._last_sync = {}
        self._sync_interval = 15  # seconds
        self._stop_sync = threading.Event()
        self.sync_thread = None
        
        # Event handlers
        self.coordinator.register_event_handler("worker_registered", self._on_worker_registered)
        self.coordinator.register_event_handler("worker_deregistered", self._on_worker_deregistered)
        self.coordinator.register_event_handler("worker_status_changed", self._on_worker_status_changed)
        self.coordinator.register_event_handler("task_created", self._on_task_created)
        self.coordinator.register_event_handler("task_status_changed", self._on_task_status_changed)
        
        logger.info("CoordinatorLoadBalancerIntegration initialized")
    
    def start(self):
        """Start the integration service."""
        # Start load balancer
        if not getattr(self.load_balancer, 'monitoring_thread', None):
            self.load_balancer.start()
            
        # Start bridge
        self.bridge.start()
        
        # Start synchronization thread
        self._stop_sync.clear()
        self.sync_thread = threading.Thread(
            target=self._sync_loop,
            daemon=True,
            name="load-balancer-sync"
        )
        self.sync_thread.start()
        
        # Register existing workers and tasks
        self._synchronize_existing_state()
        
        logger.info("CoordinatorLoadBalancerIntegration started")
    
    def stop(self):
        """Stop the integration service."""
        # Stop synchronization thread
        if self.sync_thread and self.sync_thread.is_alive():
            self._stop_sync.set()
            self.sync_thread.join(timeout=5)
        
        # Stop bridge and load balancer
        self.bridge.stop()
        self.load_balancer.stop()
        
        logger.info("CoordinatorLoadBalancerIntegration stopped")
    
    def _create_coordinator_client(self):
        """
        Create a coordinator client that interfaces with the actual coordinator.
        
        Returns:
            A CoordinatorClient instance that bridges to the actual coordinator
        """
        # Create a client that can communicate with the coordinator
        class CoordinatorClientImpl:
            def __init__(self, parent):
                self.parent = parent
            
            def get_workers(self):
                """Get all active workers from the coordinator."""
                return self.parent._get_worker_data()
            
            def get_tests(self):
                """Get all tests from the coordinator."""
                return self.parent._get_task_data()
            
            def notify_assignment_update(self, notification):
                """Notify the coordinator of an assignment status change."""
                self.parent._handle_assignment_update(notification)
            
            def update_test_status(self, test_id, status):
                """Update test status in the coordinator."""
                self.parent._update_task_status(test_id, status)
        
        return CoordinatorClientImpl(self)
    
    def _setup_coordinator_methods(self):
        """
        Inject methods into the coordinator for direct access to load balancing.
        """
        # Add load balancer methods to coordinator
        self.coordinator.get_next_worker_assignment = self.get_next_worker_assignment
        self.coordinator.suggest_worker_for_task = self.suggest_worker_for_task
        self.coordinator.get_worker_capabilities_score = self.get_worker_capabilities_score
    
    def _synchronize_existing_state(self):
        """Synchronize existing coordinator state with load balancer."""
        with self.lock:
            # Register existing workers
            for worker_id, worker in self.coordinator.workers.items():
                if worker['status'] in ['registered', 'active']:
                    self._register_worker_with_load_balancer(worker_id, worker)
            
            # Register existing tasks
            for task_id, task in self.coordinator.tasks.items():
                if task['status'] == 'queued':
                    self._register_task_with_load_balancer(task_id, task)
    
    def _sync_loop(self):
        """Background synchronization loop."""
        while not self._stop_sync.is_set():
            try:
                self._sync_with_coordinator()
            except Exception as e:
                logger.error(f"Error in sync loop: {e}")
            
            # Sleep for sync interval
            self._stop_sync.wait(self._sync_interval)
    
    def _sync_with_coordinator(self):
        """Synchronize state between coordinator and load balancer."""
        with self.lock:
            # Sync workers - update loads
            for worker_id, worker in self.coordinator.workers.items():
                if worker['status'] in ['registered', 'active']:
                    # Update worker load
                    self._update_worker_load(worker_id, worker)
            
            # Check for tasks that need updates
            for task_id, task in self.coordinator.tasks.items():
                # Check if we should resubmit a failed task
                if task['status'] == 'queued' and task_id not in self._last_sync:
                    self._register_task_with_load_balancer(task_id, task)
            
            # Check statuses of all assigned tasks
            for task_id in list(self._last_sync.keys()):
                try:
                    # Get status from load balancer
                    status = self.bridge.get_test_status(task_id)
                    if status and status['status'] != 'pending':
                        coordinator_task = self.coordinator.tasks.get(task_id)
                        if coordinator_task and coordinator_task['status'] != status['status']:
                            # Update coordinator task status
                            self._update_task_status(task_id, status)
                except Exception as e:
                    logger.error(f"Error checking task {task_id} status: {e}")
    
    # Event handlers
    
    def _on_worker_registered(self, worker_id, worker_data):
        """Handle worker registration event."""
        self._register_worker_with_load_balancer(worker_id, worker_data)
    
    def _on_worker_deregistered(self, worker_id):
        """Handle worker deregistration event."""
        try:
            self.bridge.unregister_worker(worker_id)
            logger.info(f"Unregistered worker {worker_id} from load balancer")
        except Exception as e:
            logger.error(f"Error unregistering worker {worker_id}: {e}")
    
    def _on_worker_status_changed(self, worker_id, status, worker_data):
        """Handle worker status change event."""
        try:
            # Update load information
            self._update_worker_load(worker_id, worker_data)
            
            # If worker is no longer available, update load balancer
            if status in ['unavailable', 'disconnected']:
                worker_load = WorkerLoad(
                    worker_id=worker_id,
                    active_tests=0,
                    cpu_utilization=100.0,  # Mark as fully utilized to prevent new assignments
                    memory_utilization=100.0,
                    gpu_utilization=100.0,
                    # Make load extremely high so worker isn't chosen
                    queue_depth=999
                )
                self.bridge.update_worker_load(worker_id, worker_load.__dict__)
                logger.info(f"Marked worker {worker_id} as unavailable in load balancer")
        except Exception as e:
            logger.error(f"Error updating worker {worker_id} status: {e}")
    
    def _on_task_created(self, task_id, task_data):
        """Handle task creation event."""
        if task_data['status'] == 'queued':
            self._register_task_with_load_balancer(task_id, task_data)
    
    def _on_task_status_changed(self, task_id, status, task_data):
        """Handle task status change event."""
        try:
            # Update load balancer with the new status
            if status in ['running', 'completed', 'failed', 'canceled']:
                result = task_data.get('result', None)
                self.bridge.update_assignment_status(task_id, status, result)
                logger.debug(f"Updated task {task_id} status to {status} in load balancer")
                
                # Remove from sync tracking if completed or failed
                if status in ['completed', 'failed', 'canceled'] and task_id in self._last_sync:
                    del self._last_sync[task_id]
        except Exception as e:
            logger.error(f"Error updating task {task_id} status: {e}")
    
    # Implementation methods
    
    def _register_worker_with_load_balancer(self, worker_id, worker_data):
        """Register a worker with the load balancer."""
        try:
            capabilities = self._convert_to_worker_capabilities(worker_id, worker_data)
            self.bridge.register_worker(worker_id, capabilities.__dict__)
            
            # Update load information
            self._update_worker_load(worker_id, worker_data)
            
            logger.info(f"Registered worker {worker_id} with load balancer")
        except Exception as e:
            logger.error(f"Error registering worker {worker_id}: {e}")
    
    def _update_worker_load(self, worker_id, worker_data):
        """Update worker load information in the load balancer."""
        try:
            # Extract load information from worker data
            metrics = worker_data.get('metrics', {})
            tasks = worker_data.get('tasks', [])
            
            worker_load = WorkerLoad(
                worker_id=worker_id,
                active_tests=len(tasks),
                cpu_utilization=metrics.get('cpu_utilization', 0.0),
                memory_utilization=metrics.get('memory_utilization', 0.0),
                gpu_utilization=metrics.get('gpu_utilization', 0.0),
                io_utilization=metrics.get('io_utilization', 0.0),
                network_utilization=metrics.get('network_utilization', 0.0),
                queue_depth=len([t for t in tasks if self.coordinator.tasks.get(t, {}).get('status') == 'queued']),
                reserved_memory=metrics.get('memory_used', 0.0),
                reserved_accelerators={
                    'gpu': metrics.get('gpu_used', 0)
                }
            )
            
            # Update load balancer
            self.bridge.update_worker_load(worker_id, worker_load.__dict__)
            logger.debug(f"Updated worker {worker_id} load in load balancer")
        except Exception as e:
            logger.error(f"Error updating worker {worker_id} load: {e}")
    
    def _register_task_with_load_balancer(self, task_id, task_data):
        """Register a task with the load balancer."""
        try:
            # Analyze task to determine requirements
            requirements = self.task_analyzer.analyze_task(task_data)
            
            # Convert to test requirements object
            test_requirements = self._convert_to_test_requirements(task_id, task_data, requirements)
            
            # Submit to load balancer
            self.bridge.submit_test(task_id, test_requirements.__dict__)
            
            # Track for synchronization
            self._last_sync[task_id] = {
                'time': datetime.now(),
                'status': 'pending'
            }
            
            logger.info(f"Registered task {task_id} with load balancer")
        except Exception as e:
            logger.error(f"Error registering task {task_id}: {e}")
    
    def _handle_assignment_update(self, notification):
        """Handle assignment status updates from the load balancer."""
        try:
            task_id = notification['test_id']
            worker_id = notification['worker_id']
            status = notification['status']
            result = notification['result']
            
            # Update last sync tracking
            if task_id in self._last_sync:
                self._last_sync[task_id] = {
                    'time': datetime.now(),
                    'status': status
                }
            
            # Check for status changes
            if task_id in self.coordinator.tasks:
                task = self.coordinator.tasks[task_id]
                current_status = task['status']
                
                # Only process if this is a valid transition
                valid_transitions = {
                    'queued': ['assigned'],
                    'assigned': ['running'],
                    'running': ['completed', 'failed']
                }
                
                if current_status in valid_transitions and status in valid_transitions[current_status]:
                    # Map load balancer status to coordinator status
                    status_map = {
                        'assigned': 'assigned',
                        'running': 'running',
                        'completed': 'completed',
                        'failed': 'failed'
                    }
                    
                    # Update task in coordinator
                    if status in status_map:
                        coordinator_status = status_map[status]
                        self.coordinator.update_task_status(
                            task_id, 
                            coordinator_status, 
                            worker_id=worker_id,
                            result=result
                        )
                        
                        logger.info(f"Updated task {task_id} status to {coordinator_status} in coordinator")
        except Exception as e:
            logger.error(f"Error handling assignment update: {e}")
    
    def _update_task_status(self, task_id, status_info):
        """Update task status in the coordinator based on load balancer status."""
        try:
            if task_id not in self.coordinator.tasks:
                return
                
            task = self.coordinator.tasks[task_id]
            current_status = task['status']
            new_status = status_info['status']
            
            # Map load balancer status to coordinator status
            status_map = {
                'assigned': 'assigned',
                'running': 'running',
                'completed': 'completed',
                'failed': 'failed'
            }
            
            # Only process if this is a new status
            if new_status in status_map and current_status != status_map[new_status]:
                coordinator_status = status_map[new_status]
                
                # Update task in coordinator
                self.coordinator.update_task_status(
                    task_id, 
                    coordinator_status, 
                    worker_id=status_info['worker_id'],
                    result=status_info['result']
                )
                
                logger.info(f"Updated task {task_id} status to {coordinator_status} in coordinator")
        except Exception as e:
            logger.error(f"Error updating task status: {e}")
    
    # Helper methods for data conversion
    
    def _convert_to_worker_capabilities(self, worker_id, worker_data):
        """Convert coordinator worker data to load balancer WorkerCapabilities."""
        # Extract capabilities from worker data
        capabilities = worker_data.get('capabilities', {})
        metrics = worker_data.get('metrics', {})
        tags = worker_data.get('tags', {})
        
        # Map hardware accelerators
        available_accelerators = {}
        hardware = capabilities.get('hardware', {})
        
        if hardware.get('gpu', {}).get('available', False):
            available_accelerators['gpu'] = hardware.get('gpu', {}).get('count', 0)
            
        if hardware.get('tpu', {}).get('available', False):
            available_accelerators['tpu'] = hardware.get('tpu', {}).get('count', 0)
            
        # Determine supported backends
        supported_backends = ['cpu']
        
        if hardware.get('gpu', {}).get('available', False):
            cuda_available = hardware.get('gpu', {}).get('cuda_available', False)
            if cuda_available:
                supported_backends.append('cuda')
                
            rocm_available = hardware.get('gpu', {}).get('rocm_available', False)
            if rocm_available:
                supported_backends.append('rocm')
                
        if hardware.get('webgpu', {}).get('available', False):
            supported_backends.append('webgpu')
            
        if hardware.get('webnn', {}).get('available', False):
            supported_backends.append('webnn')
            
        # Software capabilities
        software_versions = capabilities.get('software', {})
        
        # Create capabilities object
        worker_capabilities = WorkerCapabilities(
            worker_id=worker_id,
            hostname=capabilities.get('hostname', f"host-{worker_id}"),
            hardware_specs={
                'platform': capabilities.get('platform', 'unknown'),
                'cpu': hardware.get('cpu', {}),
                'memory': hardware.get('memory', {}),
                'gpu': hardware.get('gpu', {})
            },
            software_versions=software_versions,
            supported_backends=supported_backends,
            network_bandwidth=hardware.get('network', {}).get('bandwidth', 1000.0),
            storage_capacity=hardware.get('storage', {}).get('capacity', 100.0),
            available_accelerators=available_accelerators,
            available_memory=hardware.get('memory', {}).get('available_gb', 8.0),
            available_disk=hardware.get('storage', {}).get('available_gb', 100.0),
            cpu_cores=hardware.get('cpu', {}).get('cores', 4),
            cpu_threads=hardware.get('cpu', {}).get('threads', 8)
        )
        
        return worker_capabilities
    
    def _convert_to_test_requirements(self, task_id, task_data, requirements):
        """Convert coordinator task data to load balancer TestRequirements."""
        # Extract config and metadata
        config = task_data.get('config', {})
        model_info = config.get('model', {})
        hardware_reqs = config.get('hardware_requirements', {})
        
        # Determine test type
        test_type = config.get('test_type', 'default')
        
        # Extract requirements from task analyzer
        model_family = requirements.get('model_family', 'unknown')
        preferred_backend = requirements.get('preferred_backend', None)
        required_backend = hardware_reqs.get('required_backend', None)
        
        # Create requirements object
        test_requirements = TestRequirements(
            test_id=task_id,
            model_id=model_info.get('model_id', ''),
            model_family=model_family,
            test_type=test_type,
            minimum_memory=hardware_reqs.get('minimum_memory', 0.5),
            required_memory_limit=hardware_reqs.get('maximum_memory', 16.0),
            preferred_backend=preferred_backend,
            required_backend=required_backend,
            expected_duration=config.get('expected_duration', 60.0),
            priority=config.get('priority', 3),
            required_accelerators=hardware_reqs.get('required_accelerators', {}),
            required_accelerator_limit=hardware_reqs.get('accelerator_limits', {}),
            required_software=hardware_reqs.get('required_software', {}),
            timeout=config.get('timeout', 3600.0),
            retries=config.get('retries', 3),
            concurrency_key=config.get('concurrency_key', None)
        )
        
        return test_requirements
    
    # Worker data access
    
    def _get_worker_data(self):
        """Get data for all active workers from the coordinator."""
        worker_data = {}
        
        for worker_id, worker in self.coordinator.workers.items():
            if worker['status'] in ['registered', 'active']:
                # Extract capabilities and metrics for load balancer
                worker_data[worker_id] = {
                    'capabilities': worker.get('capabilities', {}),
                    'load': {
                        'cpu_utilization': worker.get('metrics', {}).get('cpu_utilization', 0.0),
                        'memory_utilization': worker.get('metrics', {}).get('memory_utilization', 0.0),
                        'gpu_utilization': worker.get('metrics', {}).get('gpu_utilization', 0.0),
                        'active_tests': len(worker.get('tasks', []))
                    }
                }
                
        return worker_data
    
    # Task data access
    
    def _get_task_data(self):
        """Get data for all active tasks from the coordinator."""
        task_data = {}
        
        for task_id, task in self.coordinator.tasks.items():
            if task['status'] in ['queued', 'assigned', 'running']:
                # Extract requirements for load balancer
                config = task.get('config', {})
                model_info = config.get('model', {})
                hardware_reqs = config.get('hardware_requirements', {})
                
                # Analyze task to determine requirements if not already analyzed
                if not hasattr(task, '_analyzed_requirements'):
                    requirements = self.task_analyzer.analyze_task(task)
                    task['_analyzed_requirements'] = requirements
                else:
                    requirements = task['_analyzed_requirements']
                
                task_data[task_id] = {
                    'status': task['status'],
                    'worker_id': task.get('worker_id'),
                    'requirements': {
                        'model_id': model_info.get('model_id', ''),
                        'model_family': requirements.get('model_family', 'unknown'),
                        'test_type': config.get('test_type', 'default'),
                        'minimum_memory': hardware_reqs.get('minimum_memory', 0.5),
                        'preferred_backend': requirements.get('preferred_backend'),
                        'required_backend': hardware_reqs.get('required_backend'),
                        'expected_duration': config.get('expected_duration', 60.0),
                        'priority': config.get('priority', 3)
                    },
                    'result': task.get('result')
                }
                
        return task_data
    
    # Public API methods
    
    def get_next_worker_assignment(self, worker_id):
        """
        Get the next task assignment for a specific worker.
        
        Args:
            worker_id: ID of the worker requesting an assignment
            
        Returns:
            Task assignment information or None if no tasks available
        """
        try:
            # Get next assignment from load balancer
            assignment = self.bridge.get_next_assignment(worker_id)
            
            if assignment:
                # Convert to coordinator task format
                return {
                    'task_id': assignment['test_id'],
                    'config': assignment['requirements']
                }
            
            return None
        except Exception as e:
            logger.error(f"Error getting next assignment for worker {worker_id}: {e}")
            return None
    
    def suggest_worker_for_task(self, task_id):
        """
        Suggest the best worker for a specific task based on capabilities and load.
        
        Args:
            task_id: ID of the task to find a worker for
            
        Returns:
            Suggested worker ID or None if no suitable worker found
        """
        try:
            if task_id not in self.coordinator.tasks:
                return None
                
            task = self.coordinator.tasks[task_id]
            
            # Make sure task is registered with load balancer
            if task_id not in self._last_sync:
                self._register_task_with_load_balancer(task_id, task)
                
            # Get status from load balancer
            status = self.bridge.get_test_status(task_id)
            
            if status and status['worker_id']:
                return status['worker_id']
                
            # No assignment yet, use coordinator's default method
            return None
        except Exception as e:
            logger.error(f"Error suggesting worker for task {task_id}: {e}")
            return None
    
    def get_worker_capabilities_score(self, worker_id, task_id):
        """
        Calculate a capability score for a worker-task pair.
        
        Args:
            worker_id: ID of the worker to evaluate
            task_id: ID of the task to evaluate for
            
        Returns:
            Score from 0-100 indicating suitability, or -1 if incompatible
        """
        try:
            # Look for existing assignment
            status = self.bridge.get_test_status(task_id)
            
            if status and status['worker_id'] == worker_id:
                # Already assigned to this worker
                return 100
                
            if worker_id not in self.coordinator.workers:
                return -1
                
            if task_id not in self.coordinator.tasks:
                return -1
                
            # Get worker and task data
            worker = self.coordinator.workers[worker_id]
            task = self.coordinator.tasks[task_id]
            
            # Analyze task to determine requirements
            requirements = self.task_analyzer.analyze_task(task)
            
            # Get worker capabilities
            capabilities = self._convert_to_worker_capabilities(worker_id, worker)
            
            # Check basic compatibility
            if requirements.get('required_backend') and requirements.get('required_backend') not in capabilities.supported_backends:
                return -1
                
            # Score based on preferred hardware match
            score = 50  # Base score
            
            # Preferred backend match
            if requirements.get('preferred_backend') in capabilities.supported_backends:
                score += 20
                
            # Model family specialization
            model_family = requirements.get('model_family', 'unknown')
            if model_family == 'vision' and 'gpu' in capabilities.available_accelerators:
                score += 15
            elif model_family == 'audio' and worker.get('tags', {}).get('audio_optimized', False):
                score += 15
                
            # Adjust for current load
            metrics = worker.get('metrics', {})
            cpu_util = metrics.get('cpu_utilization', 0.0)
            gpu_util = metrics.get('gpu_utilization', 0.0)
            
            # Penalize heavily loaded workers
            if cpu_util > 80.0 or gpu_util > 80.0:
                score -= 30
            elif cpu_util > 50.0 or gpu_util > 50.0:
                score -= 15
                
            # Ensure score is within bounds
            score = max(0, min(score, 100))
            
            return score
        except Exception as e:
            logger.error(f"Error calculating capability score: {e}")
            return -1