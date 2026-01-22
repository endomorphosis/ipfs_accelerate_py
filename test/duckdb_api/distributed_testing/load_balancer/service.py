#!/usr/bin/env python3
"""
Distributed Testing Framework - Load Balancer Service

This module implements the core load balancing service for the distributed testing framework.
"""

import os
import json
import logging
import threading
import time
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from datetime import datetime, timedelta
import queue
import uuid
from dataclasses import asdict

from .models import (
    WorkerCapabilities, 
    WorkerLoad, 
    WorkerPerformance,
    TestRequirements,
    WorkerAssignment
)
from .capability_detector import WorkerCapabilityDetector
from .performance_tracker import PerformanceTracker
from .scheduling_algorithms import (
    SchedulingAlgorithm,
    RoundRobinScheduler,
    WeightedRoundRobinScheduler,
    PerformanceBasedScheduler,
    PriorityBasedScheduler,
    CompositeScheduler,
    AffinityBasedScheduler,
    AdaptiveScheduler
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("load_balancer_service")


class LoadBalancerService:
    """Core load balancing service for distributed testing."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize the load balancer service.
        
        Args:
            db_path: Path to SQLite database for performance tracking
        """
        self.db_path = db_path
        self.lock = threading.RLock()
        
        # Performance tracker
        self.performance_tracker = PerformanceTracker(db_path=db_path)
        
        # Worker management
        self.workers: Dict[str, WorkerCapabilities] = {}
        self.worker_loads: Dict[str, WorkerLoad] = {}
        self.worker_status: Dict[str, str] = {}  # worker_id -> status (active, offline, etc.)
        self.active_assignments: Dict[str, Dict[str, WorkerAssignment]] = {}  # worker_id -> test_id -> assignment
        
        # Test management
        self.test_queue = queue.PriorityQueue()  # Priority queue of (priority, test_requirements)
        self.pending_tests: Dict[str, TestRequirements] = {}  # test_id -> requirements
        self.test_assignments: Dict[str, WorkerAssignment] = {}  # test_id -> assignment
        self.test_requeue_count: Dict[str, int] = {}  # test_id -> requeue count
        self.max_requeue_attempts = 5  # Maximum number of requeue attempts for a test
        
        # Concurrency control
        self.concurrency_locks: Dict[str, threading.Lock] = {}  # concurrency_key -> lock
        
        # Change tracking
        self.worker_changes = threading.Event()  # Set when workers change
        self.last_rebalance_time = datetime.now()
        self.last_work_steal_time = datetime.now()
        
        # Scheduling
        self.default_scheduler = AdaptiveScheduler()
        self.test_type_schedulers: Dict[str, SchedulingAlgorithm] = {}  # test_type -> scheduler
        
        # Monitoring
        self.monitoring_interval = 10  # seconds
        self.rebalance_interval = 60  # seconds
        self.work_steal_interval = 30  # seconds
        self.idle_threshold = 0.3  # Load score below this is considered idle
        self.busy_threshold = 0.7  # Load score above this is considered busy
        self._stop_monitoring = threading.Event()
        self.monitoring_thread = None
        
        # Reporting
        self.assignment_callbacks: List[Callable[[WorkerAssignment], None]] = []
        
    def start(self) -> None:
        """Start the load balancer service."""
        # Start monitoring thread
        self._stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info("Load balancer service started")
        
    def stop(self) -> None:
        """Stop the load balancer service."""
        # Stop monitoring thread
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self._stop_monitoring.set()
            self.monitoring_thread.join(timeout=5)
            
        logger.info("Load balancer service stopped")
        
    def register_worker(self, worker_id: str, capabilities: WorkerCapabilities) -> None:
        """Register a worker with the load balancer.
        
        Args:
            worker_id: Unique identifier for the worker
            capabilities: Worker capabilities
        """
        with self.lock:
            self.workers[worker_id] = capabilities
            
            # Initialize worker load if not exists
            if worker_id not in self.worker_loads:
                self.worker_loads[worker_id] = WorkerLoad(worker_id=worker_id)
                
            # Initialize active assignments if not exists
            if worker_id not in self.active_assignments:
                self.active_assignments[worker_id] = {}
                
            # Mark worker as active
            self.worker_status[worker_id] = "active"
            
            # Signal worker changes
            self.worker_changes.set()
            
            logger.info(f"Registered worker {worker_id} with {len(capabilities.supported_backends)} backends")
            
    def unregister_worker(self, worker_id: str) -> None:
        """Unregister a worker from the load balancer.
        
        Args:
            worker_id: Unique identifier for the worker
        """
        with self.lock:
            if worker_id in self.workers:
                # Mark worker as offline
                self.worker_status[worker_id] = "offline"
                
                # Signal worker changes
                self.worker_changes.set()
                
                logger.info(f"Unregistered worker {worker_id}")
                
    def update_worker_capabilities(self, worker_id: str, capabilities: WorkerCapabilities) -> None:
        """Update capabilities for a registered worker.
        
        Args:
            worker_id: Unique identifier for the worker
            capabilities: Updated worker capabilities
        """
        with self.lock:
            if worker_id in self.workers:
                self.workers[worker_id] = capabilities
                
                # Signal worker changes
                self.worker_changes.set()
                
                logger.info(f"Updated capabilities for worker {worker_id}")
                
    def update_worker_load(self, worker_id: str, load: WorkerLoad) -> None:
        """Update load information for a registered worker.
        
        Args:
            worker_id: Unique identifier for the worker
            load: Updated worker load
        """
        with self.lock:
            if worker_id in self.workers:
                self.worker_loads[worker_id] = load
                
                # Check for rebalancing
                if load.calculate_load_score() > 0.9:  # High load threshold
                    self.worker_changes.set()
                    
                logger.debug(f"Updated load for worker {worker_id}: {load.calculate_load_score():.2f}")
                
    def submit_test(self, test_requirements: TestRequirements) -> str:
        """Submit a test for scheduling.
        
        Args:
            test_requirements: Requirements for the test
            
        Returns:
            Assigned test ID
        """
        with self.lock:
            # Generate test ID if not provided
            if not test_requirements.test_id:
                test_requirements.test_id = str(uuid.uuid4())
                
            # Store test requirements
            self.pending_tests[test_requirements.test_id] = test_requirements
            
            # Add to priority queue
            self.test_queue.put((test_requirements.priority, test_requirements.test_id))
            
            logger.info(f"Submitted test {test_requirements.test_id} with priority {test_requirements.priority}")
            
            # Trigger scheduling
            self._schedule_pending_tests()
            
            return test_requirements.test_id
            
    def get_assignment(self, test_id: str) -> Optional[WorkerAssignment]:
        """Get the assignment for a test.
        
        Args:
            test_id: Test ID
            
        Returns:
            Assignment or None if not assigned
        """
        with self.lock:
            return self.test_assignments.get(test_id)
            
    def get_worker_assignments(self, worker_id: str) -> List[WorkerAssignment]:
        """Get all assignments for a worker.
        
        Args:
            worker_id: Worker ID
            
        Returns:
            List of assignments
        """
        with self.lock:
            if worker_id in self.active_assignments:
                return list(self.active_assignments[worker_id].values())
            return []
            
    def update_assignment_status(self, test_id: str, status: str, 
                              result: Optional[Dict[str, Any]] = None) -> None:
        """Update the status of a test assignment.
        
        Args:
            test_id: Test ID
            status: New status (running, completed, failed)
            result: Test result data (for completed/failed)
        """
        with self.lock:
            if test_id in self.test_assignments:
                assignment = self.test_assignments[test_id]
                
                if status == "running":
                    assignment.mark_started()
                elif status in ["completed", "failed"]:
                    success = status == "completed"
                    assignment.mark_completed(success, result)
                    
                    # Record test execution
                    self.performance_tracker.record_test_execution(assignment)
                    
                    # Release resources
                    worker_id = assignment.worker_id
                    if worker_id in self.worker_loads and worker_id in self.active_assignments:
                        if test_id in self.active_assignments[worker_id]:
                            self.worker_loads[worker_id].release_resources(
                                test_id, assignment.test_requirements
                            )
                            del self.active_assignments[worker_id][test_id]
                            
                    # Remove from assignments
                    if assignment.test_requirements.concurrency_key:
                        # Release concurrency lock
                        key = assignment.test_requirements.concurrency_key
                        if key in self.concurrency_locks:
                            try:
                                self.concurrency_locks[key].release()
                            except:
                                pass
                                
                    # Notify callbacks
                    for callback in self.assignment_callbacks:
                        try:
                            callback(assignment)
                        except Exception as e:
                            logger.error(f"Error in assignment callback: {e}")
                            
                logger.info(f"Updated test {test_id} status to {status}")
                
                # Schedule more tests if possible
                self._schedule_pending_tests()
                
    def get_next_assignment(self, worker_id: str) -> Optional[WorkerAssignment]:
        """Get the next assignment for a worker.
        
        Args:
            worker_id: Worker ID
            
        Returns:
            Next assignment or None if no pending assignment
        """
        with self.lock:
            # Check if worker is registered
            if worker_id not in self.workers or self.worker_status.get(worker_id) != "active":
                return None
                
            # Check active assignments
            if worker_id in self.active_assignments:
                for assignment in self.active_assignments[worker_id].values():
                    if assignment.status == "assigned":
                        return assignment
                        
            return None
            
    def register_assignment_callback(self, callback: Callable[[WorkerAssignment], None]) -> None:
        """Register a callback for assignment status changes.
        
        Args:
            callback: Function to call with updated assignment
        """
        with self.lock:
            self.assignment_callbacks.append(callback)
            
    def set_scheduler_for_test_type(self, test_type: str, scheduler: SchedulingAlgorithm) -> None:
        """Set a specific scheduler for a test type.
        
        Args:
            test_type: Test type
            scheduler: Scheduler to use for this test type
        """
        with self.lock:
            self.test_type_schedulers[test_type] = scheduler
            logger.info(f"Set custom scheduler for test type {test_type}")
            
    def rebalance(self) -> None:
        """Rebalance assignments across workers."""
        with self.lock:
            self.last_rebalance_time = datetime.now()
            
            # Get all workers and their loads
            active_workers = {
                worker_id: capabilities
                for worker_id, capabilities in self.workers.items()
                if self.worker_status.get(worker_id) == "active"
            }
            
            if not active_workers:
                return
                
            logger.info(f"Rebalancing assignments across {len(active_workers)} workers")
            
            # Collect performance data for all workers
            performance_data = {}
            for worker_id in active_workers:
                worker_data = {}
                for test_type in self.test_type_schedulers:
                    perf = self.performance_tracker.get_worker_performance(
                        worker_id=worker_id, test_type=test_type
                    )
                    if perf:
                        worker_data[test_type] = perf
                performance_data[worker_id] = worker_data
                
            # Check if any worker is overloaded
            overloaded_workers = []
            for worker_id, load in self.worker_loads.items():
                if worker_id in active_workers and load.calculate_load_score() > 0.8:
                    overloaded_workers.append(worker_id)
                    
            if not overloaded_workers:
                logger.info("No overloaded workers, skipping rebalance")
                return
                
            # Find assignments to rebalance from overloaded workers
            assignments_to_rebalance = []
            for worker_id in overloaded_workers:
                if worker_id in self.active_assignments:
                    for assignment in self.active_assignments[worker_id].values():
                        if assignment.status == "assigned":
                            assignments_to_rebalance.append(assignment)
                            
            if not assignments_to_rebalance:
                logger.info("No assignments to rebalance")
                return
                
            # Sort by priority (lowest first, since they're less critical)
            assignments_to_rebalance.sort(
                key=lambda a: a.test_requirements.priority, reverse=True
            )
            
            # Try to rebalance each assignment
            rebalanced_count = 0
            for assignment in assignments_to_rebalance:
                # Skip if already started
                if assignment.status != "assigned":
                    continue
                    
                # Find a better worker
                current_worker = assignment.worker_id
                test_requirements = assignment.test_requirements
                test_id = assignment.test_id
                
                # Get scheduler for this test type
                scheduler = self._get_scheduler_for_test_type(test_requirements.test_type)
                
                # Exclude current worker
                available_workers = {
                    worker_id: capabilities
                    for worker_id, capabilities in active_workers.items()
                    if worker_id != current_worker
                }
                
                # Find best worker
                new_worker = scheduler.select_worker(
                    test_requirements, available_workers, self.worker_loads, performance_data
                )
                
                if new_worker:
                    # Transfer assignment
                    self._transfer_assignment(test_id, current_worker, new_worker)
                    rebalanced_count += 1
                    
                    # Stop if we've rebalanced enough
                    if rebalanced_count >= 3:  # Limit per rebalance cycle
                        break
                        
            logger.info(f"Rebalanced {rebalanced_count} assignments")
            
    def _transfer_assignment(self, test_id: str, from_worker: str, to_worker: str) -> None:
        """Transfer an assignment from one worker to another.
        
        Args:
            test_id: Test ID
            from_worker: Source worker ID
            to_worker: Destination worker ID
        """
        if test_id not in self.test_assignments:
            return
            
        assignment = self.test_assignments[test_id]
        requirements = assignment.test_requirements
        
        # Release resources from source worker
        if from_worker in self.worker_loads:
            self.worker_loads[from_worker].release_resources(test_id, requirements)
            
        # Update assignment
        assignment.worker_id = to_worker
        
        # Reserve resources on destination worker
        if to_worker in self.worker_loads:
            worker_capabilities = self.workers.get(to_worker)
            self.worker_loads[to_worker].reserve_resources(test_id, requirements, worker_capabilities)
            
        # Update active assignments
        if from_worker in self.active_assignments and test_id in self.active_assignments[from_worker]:
            del self.active_assignments[from_worker][test_id]
            
        if to_worker not in self.active_assignments:
            self.active_assignments[to_worker] = {}
            
        self.active_assignments[to_worker][test_id] = assignment
        
        logger.info(f"Transferred test {test_id} from {from_worker} to {to_worker}")
        
    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while not self._stop_monitoring.is_set():
            try:
                # Check for worker changes
                if self.worker_changes.is_set():
                    self.worker_changes.clear()
                    
                    # Schedule pending tests when workers change
                    self._schedule_pending_tests()
                    
                # Periodic rebalancing
                time_since_rebalance = (datetime.now() - self.last_rebalance_time).total_seconds()
                if time_since_rebalance >= self.rebalance_interval:
                    self.rebalance()
                
                # Periodic work stealing
                time_since_work_steal = (datetime.now() - self.last_work_steal_time).total_seconds()
                if time_since_work_steal >= self.work_steal_interval:
                    self._perform_work_stealing()
                
                # Update worker thermal states
                self._manage_worker_thermal_states()
                    
                # Clean up completed assignments
                self._cleanup_completed_assignments()
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                
            # Sleep for monitoring interval
            self._stop_monitoring.wait(self.monitoring_interval)
            
    def _manage_worker_thermal_states(self) -> None:
        """Manage worker thermal states (warming/cooling) based on load patterns."""
        with self.lock:
            # Get active workers
            active_workers = {
                worker_id: capabilities
                for worker_id, capabilities in self.workers.items()
                if self.worker_status.get(worker_id) == "active"
            }
            
            for worker_id, capabilities in active_workers.items():
                if worker_id not in self.worker_loads:
                    continue
                    
                load = self.worker_loads[worker_id]
                
                # Update existing thermal state if worker is warming or cooling
                if load.warming_state or load.cooling_state:
                    load.update_thermal_state()
                    continue
                    
                # Check load patterns to determine if warming/cooling is needed
                current_load = load.calculate_load_score()
                
                # If worker was idle and now receiving work, start warming
                if current_load < 0.2 and load.active_tests == 0 and len(self.pending_tests) > 0:
                    # Worker is idle but we have pending tests - warm it up
                    logger.info(f"Starting warm-up for idle worker {worker_id}")
                    load.start_warming()
                    
                # If worker had very high load, start cooling down
                elif current_load > 0.9 and load.active_tests > 3:
                    # Worker was working hard - needs cooling
                    logger.info(f"Starting cool-down for overloaded worker {worker_id}")
                    load.start_cooling()
    
    def _cleanup_completed_assignments(self) -> None:
        """Clean up completed assignments."""
        with self.lock:
            now = datetime.now()
            to_remove = []
            
            for test_id, assignment in self.test_assignments.items():
                if assignment.status in ["completed", "failed"]:
                    if assignment.completed_at:
                        time_since_completion = (now - assignment.completed_at).total_seconds()
                        if time_since_completion > 3600:  # 1 hour
                            to_remove.append(test_id)
                            
            for test_id in to_remove:
                del self.test_assignments[test_id]
                
    def _schedule_pending_tests(self) -> None:
        """Schedule pending tests to available workers."""
        with self.lock:
            # Get active workers
            active_workers = {
                worker_id: capabilities
                for worker_id, capabilities in self.workers.items()
                if self.worker_status.get(worker_id) == "active"
            }
            
            if not active_workers:
                logger.warning("No active workers available for scheduling")
                return
                
            # Collect performance data for all workers
            performance_data = {}
            for worker_id in active_workers:
                worker_data = {}
                for test_type in self.test_type_schedulers:
                    perf = self.performance_tracker.get_worker_performance(
                        worker_id=worker_id, test_type=test_type
                    )
                    if perf:
                        worker_data[test_type] = perf
                performance_data[worker_id] = worker_data
                
            # Process tests in priority order
            scheduled_count = 0
            while not self.test_queue.empty():
                try:
                    # Get next test
                    priority, test_id = self.test_queue.get_nowait()
                    
                    # Skip if already assigned
                    if test_id in self.test_assignments:
                        self.test_queue.task_done()
                        continue
                        
                    # Skip if test not found in pending tests
                    if test_id not in self.pending_tests:
                        self.test_queue.task_done()
                        continue
                        
                    # Get test requirements
                    requirements = self.pending_tests[test_id]
                    
                    # Check concurrency key
                    if requirements.concurrency_key:
                        # If key already has a lock, skip this test
                        if requirements.concurrency_key in self.concurrency_locks:
                            lock = self.concurrency_locks[requirements.concurrency_key]
                            if not lock.acquire(blocking=False):
                                # Requeue test
                                self.test_queue.put((priority, test_id))
                                self.test_queue.task_done()
                                continue
                        else:
                            # Create lock
                            lock = threading.Lock()
                            self.concurrency_locks[requirements.concurrency_key] = lock
                            # Acquire lock
                            lock.acquire()
                            
                    # Get scheduler for this test type
                    scheduler = self._get_scheduler_for_test_type(requirements.test_type)
                    
                    # Select worker
                    worker_id = scheduler.select_worker(
                        requirements, active_workers, self.worker_loads, performance_data
                    )
                    
                    if worker_id:
                        # Create assignment
                        assignment = WorkerAssignment(
                            worker_id=worker_id,
                            test_id=test_id,
                            test_requirements=requirements
                        )
                        
                        # Store assignment
                        self.test_assignments[test_id] = assignment
                        
                        # Reserve resources
                        if worker_id in self.worker_loads:
                            worker_capabilities = self.workers.get(worker_id)
                            self.worker_loads[worker_id].reserve_resources(test_id, requirements, worker_capabilities)
                            
                        # Add to active assignments
                        if worker_id not in self.active_assignments:
                            self.active_assignments[worker_id] = {}
                        self.active_assignments[worker_id][test_id] = assignment
                        
                        # Remove from pending tests
                        del self.pending_tests[test_id]
                        
                        # Clean up requeue count
                        if test_id in self.test_requeue_count:
                            del self.test_requeue_count[test_id]
                        
                        # Mark as scheduled
                        scheduled_count += 1
                        
                        logger.info(f"Assigned test {test_id} to worker {worker_id}")
                    else:
                        # Check requeue count
                        requeue_count = self.test_requeue_count.get(test_id, 0) + 1
                        self.test_requeue_count[test_id] = requeue_count
                        
                        if requeue_count < self.max_requeue_attempts:
                            # No suitable worker, requeue with lower priority
                            new_priority = priority + 1
                            self.test_queue.put((new_priority, test_id))
                            
                            # If we have concurrency lock, release it
                            if requirements.concurrency_key and requirements.concurrency_key in self.concurrency_locks:
                                try:
                                    self.concurrency_locks[requirements.concurrency_key].release()
                                except:
                                    pass
                                    
                            logger.warning(f"No suitable worker for test {test_id}, requeued with priority {new_priority} (attempt {requeue_count}/{self.max_requeue_attempts})")
                        else:
                            # Max requeue attempts reached, mark as failed
                            assignment = WorkerAssignment(
                                worker_id="none",
                                test_id=test_id,
                                test_requirements=requirements,
                                status="failed"
                            )
                            assignment.mark_completed(False, {"error": "Failed to find suitable worker after maximum attempts"})
                            
                            # Store assignment
                            self.test_assignments[test_id] = assignment
                            
                            # Remove from pending tests
                            del self.pending_tests[test_id]
                            
                            # Notify callbacks
                            for callback in self.assignment_callbacks:
                                try:
                                    callback(assignment)
                                except Exception as e:
                                    logger.error(f"Error in assignment callback: {e}")
                                    
                            # Clean up requeue count
                            del self.test_requeue_count[test_id]
                            
                            logger.error(f"Test {test_id} failed after {requeue_count} scheduling attempts, no suitable worker found")
                    
                    self.test_queue.task_done()
                    
                    # Adaptively adjust batch size based on worker availability and performance
                    max_batch_size = self._calculate_adaptive_batch_size()
                    
                    # Limit number of tests scheduled per cycle
                    if scheduled_count >= max_batch_size:
                        break
                        
                except queue.Empty:
                    break
                except Exception as e:
                    logger.error(f"Error scheduling test: {e}")
                    
            if scheduled_count > 0:
                logger.info(f"Scheduled {scheduled_count} tests")
                
    def _perform_work_stealing(self) -> None:
        """
        Perform work stealing from busy workers to idle workers,
        with browser-aware capabilities enhancement.
        """
        with self.lock:
            self.last_work_steal_time = datetime.now()
            
            # Get active workers
            active_workers = {
                worker_id: capabilities
                for worker_id, capabilities in self.workers.items()
                if self.worker_status.get(worker_id) == "active"
            }
            
            if len(active_workers) < 2:  # Need at least 2 workers for stealing
                return
                
            # Classify workers by load
            idle_workers = []
            busy_workers = []
            
            # Track browser-specific metrics per worker for browser-aware work stealing
            worker_browser_metrics = {}
            
            for worker_id, capabilities in active_workers.items():
                if worker_id not in self.worker_loads:
                    continue
                    
                load_score = self.worker_loads[worker_id].calculate_load_score()
                worker_load = self.worker_loads[worker_id]
                
                if load_score < self.idle_threshold:
                    idle_workers.append(worker_id)
                elif load_score > self.busy_threshold:
                    busy_workers.append(worker_id)
                
                # Check for browser-related properties in worker load
                if hasattr(worker_load, 'browser_metrics'):
                    worker_browser_metrics[worker_id] = getattr(worker_load, 'browser_metrics', {})
                elif hasattr(worker_load, 'browser_capacities'):
                    worker_browser_metrics[worker_id] = getattr(worker_load, 'browser_capacities', {})
                elif hasattr(worker_load, 'custom_properties'):
                    browser_metrics = worker_load.custom_properties.get('browser_metrics', {})
                    browser_capacities = worker_load.custom_properties.get('browser_capacities', {})
                    if browser_metrics:
                        worker_browser_metrics[worker_id] = browser_metrics
                    elif browser_capacities:
                        # Convert capacities to metrics format
                        metrics = {}
                        for browser, capacity in browser_capacities.items():
                            metrics[browser] = {'utilization': 1.0 - capacity}
                        worker_browser_metrics[worker_id] = metrics
                    
            if not idle_workers or not busy_workers:
                logger.debug("No work stealing needed - no idle workers or no busy workers")
                return
                
            logger.info(f"Work stealing: {len(idle_workers)} idle workers, {len(busy_workers)} busy workers")
            
            # Enable browser-aware work stealing if browser metrics are available
            browser_aware_stealing = len(worker_browser_metrics) > 0
            
            if browser_aware_stealing:
                # Calculate browser utilization across all workers
                total_browser_utilization = {'chrome': 0.0, 'firefox': 0.0, 'edge': 0.0}
                browser_worker_count = {'chrome': 0, 'firefox': 0, 'edge': 0}
                
                # Calculate average utilization by browser type
                for worker_id, browser_metrics in worker_browser_metrics.items():
                    for browser_type, metrics in browser_metrics.items():
                        if isinstance(metrics, dict) and 'utilization' in metrics:
                            total_browser_utilization[browser_type] += metrics['utilization']
                            browser_worker_count[browser_type] += 1
                        elif isinstance(metrics, (int, float)):
                            # Direct utilization value
                            total_browser_utilization[browser_type] += metrics
                            browser_worker_count[browser_type] += 1
                
                # Calculate average utilization for each browser type
                avg_browser_utilization = {}
                for browser_type, total in total_browser_utilization.items():
                    count = browser_worker_count.get(browser_type, 0)
                    if count > 0:
                        avg_browser_utilization[browser_type] = total / count
                    else:
                        avg_browser_utilization[browser_type] = 0.0
                
                # Log browser utilization for debugging
                logger.debug(f"Browser utilization: {avg_browser_utilization}")
                
                # Identify overloaded browser types (for targeted stealing)
                overloaded_browsers = [browser for browser, util in avg_browser_utilization.items()
                                      if util > 0.7 and browser_worker_count.get(browser, 0) > 0]
                
                # Identify underutilized browser types (potential targets)
                underutilized_browsers = [browser for browser, util in avg_browser_utilization.items()
                                         if util < 0.3 and browser_worker_count.get(browser, 0) > 0]
                
                # Browser-aware work stealing
                if overloaded_browsers and underutilized_browsers:
                    logger.info(f"Browser-aware work stealing: overloaded={overloaded_browsers}, "
                              f"underutilized={underutilized_browsers}")
                    
                    # Match model types with appropriate browsers
                    model_browser_affinity = {
                        'audio': 'firefox',
                        'vision': 'chrome',
                        'text_embedding': 'edge',
                        'large_language_model': 'chrome'
                    }
                    
                    # Enhance worker priority for stealing based on browser capabilities
                    enhanced_busy_workers = []
                    for busy_worker in busy_workers:
                        priority_score = 10  # Base priority
                        
                        # Check if worker has overloaded browsers
                        if busy_worker in worker_browser_metrics:
                            metrics = worker_browser_metrics[busy_worker]
                            for browser in overloaded_browsers:
                                if browser in metrics:
                                    if isinstance(metrics[browser], dict) and 'utilization' in metrics[browser]:
                                        util = metrics[browser]['utilization']
                                    else:
                                        util = metrics[browser]
                                    
                                    # Higher utilization = higher priority for stealing
                                    if util > 0.8:
                                        priority_score += 20
                                    elif util > 0.7:
                                        priority_score += 10
                        
                        enhanced_busy_workers.append((busy_worker, priority_score))
                    
                    # Sort by priority score
                    enhanced_busy_workers.sort(key=lambda x: x[1], reverse=True)
                    busy_workers = [worker for worker, _ in enhanced_busy_workers]
                    
                    # Enhance idle worker priority based on browser capabilities
                    enhanced_idle_workers = []
                    for idle_worker in idle_workers:
                        priority_score = 10  # Base priority
                        
                        # Check if worker has underutilized browsers
                        if idle_worker in worker_browser_metrics:
                            metrics = worker_browser_metrics[idle_worker]
                            for browser in underutilized_browsers:
                                if browser in metrics:
                                    if isinstance(metrics[browser], dict) and 'utilization' in metrics[browser]:
                                        util = metrics[browser]['utilization']
                                    else:
                                        util = metrics[browser]
                                    
                                    # Lower utilization = higher priority as target
                                    if util < 0.2:
                                        priority_score += 20
                                    elif util < 0.3:
                                        priority_score += 10
                        
                        enhanced_idle_workers.append((idle_worker, priority_score))
                    
                    # Sort by priority score
                    enhanced_idle_workers.sort(key=lambda x: x[1], reverse=True)
                    idle_workers = [worker for worker, _ in enhanced_idle_workers]
            else:
                # Sort busy workers by load (highest first) when browser metrics not available
                busy_workers.sort(
                    key=lambda wid: self.worker_loads[wid].calculate_load_score(), 
                    reverse=True
                )
                
                # Sort idle workers by load (lowest first) when browser metrics not available
                idle_workers.sort(
                    key=lambda wid: self.worker_loads[wid].calculate_load_score()
                )
            
            # Steal work
            stolen_count = 0
            max_steals = min(len(idle_workers), 5)  # Limit steals per cycle
            
            for busy_worker in busy_workers:
                if stolen_count >= max_steals:
                    break
                    
                # Get assigned but not yet running tests from busy worker
                if busy_worker not in self.active_assignments:
                    continue
                    
                stealable_tests = [
                    (test_id, assignment)
                    for test_id, assignment in self.active_assignments[busy_worker].items()
                    if assignment.status == "assigned"
                ]
                
                if not stealable_tests:
                    continue
                
                # Sort tests by priority and browser affinity for stealing
                if browser_aware_stealing:
                    # Enhanced prioritization based on browser affinity
                    model_browser_affinity = {
                        'audio': 'firefox',
                        'vision': 'chrome',
                        'text_embedding': 'edge',
                        'large_language_model': 'chrome'
                    }
                    
                    # Calculate stealing priority for each test
                    enhanced_stealable_tests = []
                    for test_id, assignment in stealable_tests:
                        steal_priority = 10  # Base priority (higher value = higher priority)
                        
                        # Lower priority for high priority tasks (less likely to steal)
                        test_req = assignment.test_requirements
                        if test_req.priority <= 2:  # High priority (1-2)
                            steal_priority -= 5
                        elif test_req.priority >= 4:  # Low priority (4-5)
                            steal_priority += 5
                        
                        # Check model type affinity with browsers
                        model_type = test_req.model_type if hasattr(test_req, 'model_type') else None
                        
                        if model_type and model_type in model_browser_affinity:
                            # Check if preferred browser for this model type is overloaded
                            preferred_browser = model_browser_affinity[model_type]
                            
                            # Higher priority to steal tasks whose preferred browser is overloaded
                            if preferred_browser in overloaded_browsers:
                                steal_priority += 10
                                
                            # Higher priority if there's an underutilized worker with right browser
                            for idle_worker in idle_workers:
                                if (idle_worker in worker_browser_metrics and 
                                    preferred_browser in worker_browser_metrics[idle_worker]):
                                    # Add bonus for matching browser
                                    steal_priority += 5
                                    break
                        
                        enhanced_stealable_tests.append((test_id, assignment, steal_priority))
                    
                    # Sort by stealing priority (highest first)
                    enhanced_stealable_tests.sort(key=lambda x: x[2], reverse=True)
                    stealable_tests = [(test_id, assignment) for test_id, assignment, _ in enhanced_stealable_tests]
                else:
                    # Default sorting by priority (lowest priority first for stealing)
                    stealable_tests.sort(
                        key=lambda x: x[1].test_requirements.priority,
                        reverse=True
                    )
                
                # Try to steal tests
                for test_id, assignment in stealable_tests:
                    test_req = assignment.test_requirements
                    model_type = test_req.model_type if hasattr(test_req, 'model_type') else None
                    
                    # Find an idle worker that can handle this test
                    for idle_worker in idle_workers:
                        # Skip if worker doesn't have required capabilities
                        if idle_worker not in self.workers:
                            continue
                            
                        # Check compatibility
                        worker_capabilities = self.workers[idle_worker]
                        if not worker_capabilities.is_compatible_with(test_req):
                            continue
                            
                        # Check capacity
                        if idle_worker not in self.worker_loads:
                            continue
                            
                        # Check if worker can handle this test
                        worker_load = self.worker_loads[idle_worker]
                        if not worker_load.has_capacity_for(test_req, worker_capabilities):
                            continue
                        
                        # Browser-aware selection - check if the idle worker has a better browser
                        if browser_aware_stealing and model_type and model_type in model_browser_affinity:
                            preferred_browser = model_browser_affinity[model_type]
                            
                            # Check if idle worker has the preferred browser underutilized
                            if (idle_worker in worker_browser_metrics and 
                                preferred_browser in worker_browser_metrics[idle_worker]):
                                
                                # Check browser utilization
                                metrics = worker_browser_metrics[idle_worker]
                                browser_util = (metrics[preferred_browser]['utilization'] 
                                               if isinstance(metrics[preferred_browser], dict) 
                                               else metrics[preferred_browser])
                                
                                # If browser is overutilized on idle worker too, may not be worth stealing
                                if browser_util > 0.7 and preferred_browser not in underutilized_browsers:
                                    # Skip to next worker if browser already heavily loaded
                                    continue
                        
                        # Transfer test
                        self._transfer_assignment(test_id, busy_worker, idle_worker)
                        stolen_count += 1
                        
                        if browser_aware_stealing and model_type:
                            logger.info(f"Browser-aware work stealing: transferred {model_type} test {test_id} "
                                      f"from {busy_worker} to {idle_worker}")
                        else:
                            logger.info(f"Work stealing: transferred test {test_id} from {busy_worker} to {idle_worker}")
                        
                        # Move to next idle worker
                        idle_workers.remove(idle_worker)
                        if not idle_workers:
                            break
                            
                    if stolen_count >= max_steals or not idle_workers:
                        break
                        
            if stolen_count > 0:
                logger.info(f"Work stealing: successfully stole {stolen_count} tests")
                
    def _calculate_adaptive_batch_size(self) -> int:
        """Calculate an adaptive batch size based on worker availability and system load.
        
        Returns:
            Batch size for scheduling tests
        """
        # Base batch size
        base_batch_size = 10
        
        # Count active workers
        active_worker_count = sum(1 for worker_id, status in self.worker_status.items()
                                if status == "active")
        
        # If no active workers, return minimum batch size
        if active_worker_count == 0:
            return 1
            
        # Calculate average load across workers
        total_load = 0.0
        loaded_worker_count = 0
        
        for worker_id, load in self.worker_loads.items():
            if self.worker_status.get(worker_id) == "active":
                total_load += load.calculate_load_score()
                loaded_worker_count += 1
                
        # If no load information, use base batch size
        if loaded_worker_count == 0:
            return base_batch_size
            
        average_load = total_load / loaded_worker_count
        
        # Adjust batch size based on average load and worker count
        # When load is low, use larger batch size
        # When load is high, use smaller batch size
        load_factor = max(0.5, 1.5 - average_load)  # 0.5 to 1.5
        
        # Scale with worker count (more workers = larger batches)
        worker_factor = min(2.0, 0.5 + (active_worker_count / 10.0))  # 0.5 to 2.0
        
        # Queue size factor (larger queue = larger batches)
        queue_size = self.test_queue.qsize()
        queue_factor = min(2.0, 0.5 + (queue_size / 20.0))  # 0.5 to 2.0
        
        # Calculate adaptive batch size
        batch_size = int(base_batch_size * load_factor * worker_factor * queue_factor)
        
        # Ensure minimum and maximum batch sizes
        min_batch_size = max(1, active_worker_count // 2)
        max_batch_size = max(20, active_worker_count * 5)
        
        batch_size = max(min_batch_size, min(batch_size, max_batch_size))
        
        logger.debug(f"Adaptive batch size: {batch_size} (load: {average_load:.2f}, workers: {active_worker_count})")
        
        return batch_size
    
    def _get_scheduler_for_test_type(self, test_type: Optional[str]) -> SchedulingAlgorithm:
        """Get the appropriate scheduler for a test type.
        
        Args:
            test_type: Test type or None
            
        Returns:
            Scheduler instance
        """
        if test_type and test_type in self.test_type_schedulers:
            return self.test_type_schedulers[test_type]
        return self.default_scheduler


# Factory function for creating scheduler instances
def create_scheduler(scheduler_type: str, **kwargs) -> SchedulingAlgorithm:
    """Create a scheduler instance of the specified type.
    
    Args:
        scheduler_type: Type of scheduler to create
        **kwargs: Additional parameters for the scheduler
        
    Returns:
        Scheduler instance
    """
    if scheduler_type == "round_robin":
        return RoundRobinScheduler()
    elif scheduler_type == "weighted_round_robin":
        return WeightedRoundRobinScheduler()
    elif scheduler_type == "performance_based":
        return PerformanceBasedScheduler()
    elif scheduler_type == "priority_based":
        return PriorityBasedScheduler()
    elif scheduler_type == "affinity_based":
        return AffinityBasedScheduler()
    elif scheduler_type == "adaptive":
        return AdaptiveScheduler()
    elif scheduler_type == "composite":
        algorithms = kwargs.get("algorithms", [])
        scheduler_configs = []
        for config in algorithms:
            algorithm_type = config.get("type")
            weight = config.get("weight", 1.0)
            algorithm = create_scheduler(algorithm_type)
            scheduler_configs.append((algorithm, weight))
        return CompositeScheduler(scheduler_configs)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


# Factory function for creating load balancer service
def create_load_balancer(config: Dict[str, Any]) -> LoadBalancerService:
    """Create a load balancer service with the specified configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        LoadBalancerService instance
    """
    # Create load balancer
    db_path = config.get("db_path")
    load_balancer = LoadBalancerService(db_path=db_path)
    
    # Configure monitoring intervals
    if "monitoring_interval" in config:
        load_balancer.monitoring_interval = config["monitoring_interval"]
    if "rebalance_interval" in config:
        load_balancer.rebalance_interval = config["rebalance_interval"]
        
    # Configure default scheduler
    default_scheduler_config = config.get("default_scheduler", {"type": "adaptive"})
    default_scheduler = create_scheduler(**default_scheduler_config)
    load_balancer.default_scheduler = default_scheduler
    
    # Configure test type schedulers
    test_type_schedulers = config.get("test_type_schedulers", {})
    for test_type, scheduler_config in test_type_schedulers.items():
        scheduler = create_scheduler(**scheduler_config)
        load_balancer.set_scheduler_for_test_type(test_type, scheduler)
        
    return load_balancer