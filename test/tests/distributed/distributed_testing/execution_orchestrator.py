#!/usr/bin/env python3
"""
Parallel Test Execution Orchestrator for Distributed Testing Framework

This module provides a comprehensive system for orchestrating the parallel execution
of tests across distributed resources. It leverages the test dependency manager to
ensure tests are executed in the correct order while maximizing parallel execution
where possible.

Key features:
- Multiple parallel execution strategies
- Execution group management
- Flow control mechanisms
- Progress tracking
- Integration with test dependencies
"""

import logging
import time
import anyio
import uuid
import json
from enum import Enum
from typing import Dict, List, Set, Optional, Tuple, Any, Union, DefaultDict, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import concurrent.futures

# Import test dependency manager
from test_dependency_manager import (
    TestDependencyManager, DependencyStatus
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("execution_orchestrator")


class ExecutionStrategy(Enum):
    """Available strategies for parallel test execution."""
    SIMPLE = "simple"  # Basic parallel execution with fixed workers
    MAX_PARALLEL = "max_parallel"  # Maximum possible parallelism
    RESOURCE_AWARE = "resource_aware"  # Hardware-aware parallel execution
    DEADLINE_BASED = "deadline_based"  # Prioritize tests with deadlines
    ADAPTIVE = "adaptive"  # Adapt parallelism based on system load


class ExecutionStatus(Enum):
    """Status of test execution."""
    PENDING = "pending"  # Test is waiting to run
    READY = "ready"  # Test is ready to run
    RUNNING = "running"  # Test is currently running
    COMPLETED = "completed"  # Test has completed successfully
    FAILED = "failed"  # Test has failed
    SKIPPED = "skipped"  # Test was skipped due to dependencies


@dataclass
class ExecutionContext:
    """Context for a test execution."""
    test_id: str
    worker_id: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    status: ExecutionStatus = ExecutionStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    progress: float = 0.0  # 0.0 to 1.0
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionGroup:
    """Group of tests that can be executed in parallel."""
    group_id: str
    test_ids: List[str]
    dependencies: Set[str] = field(default_factory=set)  # Other group IDs this group depends on
    is_ready: bool = False
    is_completed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExecutionOrchestrator:
    """
    Orchestrates the parallel execution of tests.
    
    This class is responsible for:
    - Creating execution groups based on dependencies
    - Scheduling tests for execution
    - Tracking test execution progress
    - Managing test execution flow
    """
    
    def __init__(
            self,
            dependency_manager: TestDependencyManager,
            max_workers: int = 0,  # 0 for auto-detection
            strategy: ExecutionStrategy = ExecutionStrategy.RESOURCE_AWARE,
            timeout_seconds: Optional[int] = None
        ):
        """
        Initialize the execution orchestrator.
        
        Args:
            dependency_manager: Test dependency manager
            max_workers: Maximum number of workers (0 for auto-detection)
            strategy: Execution strategy to use
            timeout_seconds: Optional timeout for execution
        """
        self.dependency_manager = dependency_manager
        self.strategy = strategy
        self.timeout_seconds = timeout_seconds
        
        # Determine max workers
        self.max_workers = max_workers
        if self.max_workers <= 0:
            import multiprocessing
            self.max_workers = multiprocessing.cpu_count()
        
        # Execution state
        self.execution_groups: Dict[str, ExecutionGroup] = {}
        self.execution_contexts: Dict[str, ExecutionContext] = {}
        self.active_workers: Dict[str, str] = {}  # worker_id -> test_id
        self.execution_queue: List[str] = []  # List of test_ids ready to execute
        
        # Execution metrics
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.metrics: Dict[str, Any] = {
            "total_tests": 0,
            "completed_tests": 0,
            "failed_tests": 0,
            "skipped_tests": 0,
            "execution_time": 0.0,
            "max_parallelism": 0,
            "avg_parallelism": 0.0,
            "avg_waiting_time": 0.0,
            "avg_execution_time": 0.0
        }
        
        # Execution hooks
        self.pre_execution_hook: Optional[Callable] = None
        self.post_execution_hook: Optional[Callable] = None
        self.test_start_hook: Optional[Callable] = None
        self.test_end_hook: Optional[Callable] = None
        
        logger.info(f"Execution Orchestrator initialized with strategy: {strategy.name}, "
                   f"max workers: {self.max_workers}")
    
    def set_pre_execution_hook(self, hook: Callable) -> None:
        """Set a hook to be called before execution starts."""
        self.pre_execution_hook = hook
    
    def set_post_execution_hook(self, hook: Callable) -> None:
        """Set a hook to be called after execution completes."""
        self.post_execution_hook = hook
    
    def set_test_start_hook(self, hook: Callable) -> None:
        """Set a hook to be called when a test starts."""
        self.test_start_hook = hook
    
    def set_test_end_hook(self, hook: Callable) -> None:
        """Set a hook to be called when a test ends."""
        self.test_end_hook = hook
    
    def create_execution_groups(self) -> None:
        """
        Create execution groups based on test dependencies.
        
        This uses the dependency manager to generate groups of tests
        that can be executed in parallel.
        """
        # Ensure dependencies are validated
        if not self.dependency_manager.is_validated:
            is_valid, errors = self.dependency_manager.validate_dependencies()
            if not is_valid:
                error_msg = f"Invalid dependencies: {len(errors)} errors"
                for error in errors[:5]:  # Show first 5 errors
                    error_msg += f"\n  - {error}"
                logger.error(error_msg)
                raise ValueError(error_msg)
        
        # Get parallel execution groups from dependency manager
        parallel_groups = self.dependency_manager.get_parallel_execution_groups()
        
        # Create execution groups
        execution_groups = {}
        for level, test_ids in enumerate(parallel_groups):
            group_id = f"group_{level}"
            
            # Create dependencies on previous levels
            dependencies = set()
            if level > 0:
                for prev_level in range(level):
                    dependencies.add(f"group_{prev_level}")
            
            # Create execution group
            execution_groups[group_id] = ExecutionGroup(
                group_id=group_id,
                test_ids=test_ids,
                dependencies=dependencies,
                is_ready=level == 0,  # First level is always ready
                metadata={"level": level}
            )
        
        self.execution_groups = execution_groups
        
        # Create execution contexts for all tests
        for test_id in self.dependency_manager.test_dependencies:
            self.execution_contexts[test_id] = ExecutionContext(test_id=test_id)
        
        # Update metrics
        self.metrics["total_tests"] = len(self.execution_contexts)
        
        # Add first level to execution queue if it's ready
        if "group_0" in self.execution_groups and self.execution_groups["group_0"].is_ready:
            for test_id in self.execution_groups["group_0"].test_ids:
                self.execution_contexts[test_id].status = ExecutionStatus.READY
                self.execution_queue.append(test_id)
        
        logger.info(f"Created {len(self.execution_groups)} execution groups with "
                   f"{len(self.execution_contexts)} total tests")
    
    def adjust_max_workers(self) -> None:
        """
        Adjust max workers based on execution strategy and system resources.
        
        Different strategies may adjust worker count differently based on
        available resources and execution characteristics.
        """
        import psutil
        
        # Get system info
        cpu_count = psutil.cpu_count(logical=False)
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        # Default to CPU count
        new_max_workers = cpu_count
        
        # Adjust based on strategy
        if self.strategy == ExecutionStrategy.SIMPLE:
            # Simple strategy uses fixed worker count
            new_max_workers = cpu_count
        
        elif self.strategy == ExecutionStrategy.MAX_PARALLEL:
            # Max parallel strategy uses as many workers as possible
            # but still considers memory constraints
            memory_per_worker_gb = 1.0  # Estimated memory per worker
            memory_based_workers = int(available_memory_gb / memory_per_worker_gb)
            new_max_workers = min(len(self.execution_contexts), 
                                max(cpu_count * 2, memory_based_workers))
        
        elif self.strategy == ExecutionStrategy.RESOURCE_AWARE:
            # Resource-aware strategy considers system load and available resources
            system_load = psutil.getloadavg()[0] / cpu_count
            
            if system_load < 0.3:
                # Low load, use more workers
                new_max_workers = cpu_count * 2
            elif system_load < 0.7:
                # Medium load, use CPU count
                new_max_workers = cpu_count
            else:
                # High load, use fewer workers
                new_max_workers = max(1, cpu_count // 2)
        
        elif self.strategy == ExecutionStrategy.DEADLINE_BASED:
            # Deadline-based uses more workers to meet deadlines
            new_max_workers = cpu_count * 2
        
        elif self.strategy == ExecutionStrategy.ADAPTIVE:
            # Adaptive strategy adjusts based on observed performance
            system_load = psutil.getloadavg()[0] / cpu_count
            
            # Start with CPU count
            new_max_workers = cpu_count
            
            # Adjust based on system load trend and execution metrics
            # (This is simplified - a real implementation would track metrics over time)
            if system_load > 0.8:
                # High load, reduce workers
                new_max_workers = max(1, int(new_max_workers * 0.8))
            elif system_load < 0.4:
                # Low load, increase workers
                new_max_workers = int(new_max_workers * 1.2)
        
        # Ensure at least one worker
        new_max_workers = max(1, new_max_workers)
        
        # Update max workers if changed
        if new_max_workers != self.max_workers:
            logger.info(f"Adjusted max workers from {self.max_workers} to {new_max_workers} "
                       f"based on {self.strategy.name} strategy")
            self.max_workers = new_max_workers
    
    def update_group_status(self) -> None:
        """
        Update the status of execution groups.
        
        Checks if groups are ready to execute based on dependencies
        and marks completed groups.
        """
        # Check each group
        for group_id, group in self.execution_groups.items():
            # Completed groups stay completed.
            if group.is_completed:
                continue
            
            # Only transition to ready once.
            if not group.is_ready:
                # Check if all dependencies are completed
                all_deps_completed = True
                for dep_group_id in group.dependencies:
                    if dep_group_id not in self.execution_groups:
                        logger.warning(f"Group {group_id} depends on unknown group {dep_group_id}")
                        all_deps_completed = False
                        break

                    if not self.execution_groups[dep_group_id].is_completed:
                        all_deps_completed = False
                        break

                # Mark group as ready if all dependencies are completed
                if all_deps_completed:
                    group.is_ready = True
                    logger.debug(f"Group {group_id} is now ready")

                    # Add tests to execution queue
                    for test_id in group.test_ids:
                        # Check if test is pending
                        if self.execution_contexts[test_id].status == ExecutionStatus.PENDING:
                            # Check if test dependencies are satisfied
                            can_execute, _ = self.dependency_manager.resolve_dependencies(test_id)

                            if can_execute:
                                self.execution_contexts[test_id].status = ExecutionStatus.READY
                                self.execution_queue.append(test_id)
                                logger.debug(f"Test {test_id} is now ready")
                            else:
                                # Test dependencies not satisfied, might be due to failures
                                self.execution_contexts[test_id].status = ExecutionStatus.SKIPPED
                                self.metrics["skipped_tests"] += 1
                                logger.debug(f"Test {test_id} skipped due to dependencies")
            
            # Check if group is completed
            all_tests_complete = True
            for test_id in group.test_ids:
                status = self.execution_contexts[test_id].status
                if status not in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.SKIPPED]:
                    all_tests_complete = False
                    break
            
            # Mark group as completed if all tests are done
            if all_tests_complete:
                group.is_completed = True
                logger.debug(f"Group {group_id} is now completed")
    
    def select_tests_for_execution(self) -> List[str]:
        """
        Select tests for execution based on the current strategy.
        
        Returns:
            List of test IDs selected for execution
        """
        # Calculate how many workers are available
        available_workers = self.max_workers - len(self.active_workers)
        
        if available_workers <= 0 or not self.execution_queue:
            return []
        
        selected_tests = []
        
        if self.strategy == ExecutionStrategy.SIMPLE:
            # Simple strategy just selects tests in order
            selected_tests = self.execution_queue[:available_workers]
        
        elif self.strategy == ExecutionStrategy.MAX_PARALLEL:
            # Max parallel strategy selects as many tests as possible
            selected_tests = self.execution_queue[:available_workers]
        
        elif self.strategy == ExecutionStrategy.RESOURCE_AWARE:
            # Resource-aware strategy considers resource requirements
            # This would typically involve hardware capability matching
            # For now, just select tests in order
            selected_tests = self.execution_queue[:available_workers]
        
        elif self.strategy == ExecutionStrategy.DEADLINE_BASED:
            # Deadline-based strategy prioritizes tests with deadlines
            # Sort by deadline if available in metadata
            prioritized_queue = sorted(
                self.execution_queue,
                key=lambda test_id: self.execution_contexts[test_id].metadata.get('deadline', float('inf'))
            )
            selected_tests = prioritized_queue[:available_workers]
        
        elif self.strategy == ExecutionStrategy.ADAPTIVE:
            # Adaptive strategy adjusts based on observed performance
            # For now, just select tests in order
            selected_tests = self.execution_queue[:available_workers]
        
        # Remove selected tests from queue
        for test_id in selected_tests:
            self.execution_queue.remove(test_id)
        
        return selected_tests
    
    def execute_test(self, test_id: str, worker_id: str) -> None:
        """
        Execute a single test.
        
        Args:
            test_id: ID of the test to execute
            worker_id: ID of the worker executing the test
        """
        # Get execution context
        context = self.execution_contexts[test_id]
        
        # Update context
        context.worker_id = worker_id
        context.start_time = time.time()
        context.status = ExecutionStatus.RUNNING
        
        # Call test start hook if provided
        if self.test_start_hook:
            try:
                self.test_start_hook(test_id, worker_id)
            except Exception as e:
                logger.error(f"Error in test start hook: {str(e)}")
        
        logger.info(f"Started execution of test {test_id} on worker {worker_id}")
        
        try:
            # For demonstration, simulate test execution
            # In a real implementation, this would call the actual test
            import random
            
            # Simulate execution time
            execution_time = random.uniform(0.5, 3.0)
            
            # Simulate progress updates
            steps = 10
            for i in range(1, steps + 1):
                time.sleep(execution_time / steps)
                context.progress = i / steps
            
            # Simulate success/failure (tests expect success when random() > 0.9)
            success = random.random() > 0.9
            
            if success:
                context.status = ExecutionStatus.COMPLETED
                context.result = {"success": True}
                self.metrics["completed_tests"] += 1
                logger.info(f"Test {test_id} completed successfully")
            else:
                context.status = ExecutionStatus.FAILED
                context.error = "Simulated failure"
                self.metrics["failed_tests"] += 1
                logger.warning(f"Test {test_id} failed")
            
            # Update the dependency manager
            self.dependency_manager.update_test_status(
                test_id=test_id,
                status="completed" if success else "failed",
                result={"success": success},
                metadata={"execution_time": execution_time}
            )
        
        except Exception as e:
            # Handle test execution error
            context.status = ExecutionStatus.FAILED
            context.error = str(e)
            self.metrics["failed_tests"] += 1
            
            logger.error(f"Error executing test {test_id}: {str(e)}")
            
            # Update the dependency manager
            self.dependency_manager.update_test_status(
                test_id=test_id,
                status="failed",
                result={"success": False, "error": str(e)}
            )
        
        finally:
            # Update context
            context.end_time = time.time()
            context.progress = 1.0
            
            # Calculate execution time
            if context.start_time and context.end_time:
                execution_time = context.end_time - context.start_time
                context.metadata["execution_time"] = execution_time
            
            # Call test end hook if provided
            if self.test_end_hook:
                try:
                    self.test_end_hook(test_id, context.status, context.result, context.error)
                except Exception as e:
                    logger.error(f"Error in test end hook: {str(e)}")
            
            # Remove worker from active workers
            if worker_id in self.active_workers:
                del self.active_workers[worker_id]
    
    def async_execute_tests(self, test_ids: List[str]) -> List[concurrent.futures.Future]:
        """
        Execute tests asynchronously using a thread pool.
        
        Args:
            test_ids: List of test IDs to execute
            
        Returns:
            List of futures for the test executions
        """
        # Create thread pool executor
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        futures = []
        
        # Execute each test
        for test_id in test_ids:
            # Generate a worker ID
            worker_id = f"worker_{uuid.uuid4().hex[:8]}"
            
            # Add to active workers
            self.active_workers[worker_id] = test_id
            
            # Submit to executor
            future = executor.submit(self.execute_test, test_id, worker_id)
            futures.append(future)
        
        return futures
    
    def execute_all_tests(self) -> Dict[str, Any]:
        """
        Execute all tests in dependency order.
        
        Returns:
            Dictionary with execution results
        """
        # Create execution groups if not already created
        if not self.execution_groups:
            self.create_execution_groups()
        
        # Adjust max workers
        self.adjust_max_workers()
        
        # Call pre-execution hook if provided
        if self.pre_execution_hook:
            try:
                self.pre_execution_hook(self.max_workers, self.strategy, len(self.execution_contexts))
            except Exception as e:
                logger.error(f"Error in pre-execution hook: {str(e)}")
        
        # Start execution
        self.start_time = time.time()
        logger.info(f"Starting execution of {len(self.execution_contexts)} tests "
                   f"with {self.max_workers} workers using {self.strategy.name} strategy")
        
        # Initialize futures list
        all_futures = []
        
        # Main execution loop
        while True:
            # Update group status
            self.update_group_status()
            
            # Select tests for execution
            selected_tests = self.select_tests_for_execution()
            
            if selected_tests:
                # Execute selected tests
                futures = self.async_execute_tests(selected_tests)
                all_futures.extend(futures)
                
                # Track max parallelism
                current_parallelism = len(self.active_workers)
                self.metrics["max_parallelism"] = max(self.metrics["max_parallelism"], current_parallelism)
                
                logger.debug(f"Executing {len(selected_tests)} tests, "
                           f"{len(self.execution_queue)} remaining in queue")
            
            # Check if all tests are completed
            all_completed = all(
                context.status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.SKIPPED]
                for context in self.execution_contexts.values()
            )
            
            if all_completed:
                logger.info("All tests completed")
                break
            
            # Check timeout
            if self.timeout_seconds and self.start_time:
                elapsed_time = time.time() - self.start_time
                if elapsed_time > self.timeout_seconds:
                    logger.warning(f"Execution timeout after {elapsed_time:.2f} seconds")
                    
                    # Mark remaining tests as skipped
                    for test_id in self.execution_queue:
                        self.execution_contexts[test_id].status = ExecutionStatus.SKIPPED
                        self.metrics["skipped_tests"] += 1
                    
                    # Mark active tests as failed
                    for worker_id, test_id in self.active_workers.items():
                        if self.execution_contexts[test_id].status == ExecutionStatus.RUNNING:
                            self.execution_contexts[test_id].status = ExecutionStatus.FAILED
                            self.execution_contexts[test_id].error = "Execution timeout"
                            self.metrics["failed_tests"] += 1
                    
                    # Break out of loop
                    break
            
            # Sleep to avoid busy waiting
            time.sleep(0.1)
        
        # Wait for all futures to complete
        for future in concurrent.futures.as_completed(all_futures):
            try:
                # Get result (which will be None, but will raise any exceptions)
                future.result()
            except Exception as e:
                logger.error(f"Error in async test execution: {str(e)}")
        
        # End execution
        self.end_time = time.time()
        
        # Calculate execution time
        if self.start_time and self.end_time:
            self.metrics["execution_time"] = self.end_time - self.start_time
        
        # Calculate average metrics
        if self.metrics["total_tests"] > 0:
            execution_times = [
                context.metadata.get("execution_time", 0.0)
                for context in self.execution_contexts.values()
                if context.status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED]
                and "execution_time" in context.metadata
            ]
            
            if execution_times:
                self.metrics["avg_execution_time"] = sum(execution_times) / len(execution_times)
        
        # Call post-execution hook if provided
        if self.post_execution_hook:
            try:
                self.post_execution_hook(self.metrics)
            except Exception as e:
                logger.error(f"Error in post-execution hook: {str(e)}")
        
        logger.info(f"Execution completed in {self.metrics['execution_time']:.2f} seconds with "
                  f"{self.metrics['completed_tests']} successful, "
                  f"{self.metrics['failed_tests']} failed, and "
                  f"{self.metrics['skipped_tests']} skipped tests")
        
        return self.get_execution_results()
    
    def get_execution_results(self) -> Dict[str, Any]:
        """
        Get execution results and metrics.
        
        Returns:
            Dictionary with execution results and metrics
        """
        test_results = {}
        for test_id, context in self.execution_contexts.items():
            test_results[test_id] = {
                "status": context.status.value,
                "result": context.result,
                "error": context.error,
                "worker_id": context.worker_id,
                "execution_time": (context.end_time - context.start_time) if context.start_time and context.end_time else None,
                "metadata": context.metadata
            }
        
        return {
            "metrics": self.metrics,
            "test_results": test_results,
            "execution_groups": {
                group_id: {
                    "tests": group.test_ids,
                    "dependencies": list(group.dependencies),
                    "is_completed": group.is_completed,
                    "metadata": group.metadata
                } for group_id, group in self.execution_groups.items()
            },
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_time": self.end_time - self.start_time if self.start_time and self.end_time else None,
            "strategy": self.strategy.value,
            "max_workers": self.max_workers
        }
    
    async def execute_all_tests_async(self) -> Dict[str, Any]:
        """
        Execute all tests asynchronously.
        
        Returns:
            Dictionary with execution results
        """
        # Create execution groups if not already created
        if not self.execution_groups:
            self.create_execution_groups()
        
        # Adjust max workers
        self.adjust_max_workers()
        
        # Call pre-execution hook if provided
        if self.pre_execution_hook:
            try:
                self.pre_execution_hook(self.max_workers, self.strategy, len(self.execution_contexts))
            except Exception as e:
                logger.error(f"Error in pre-execution hook: {str(e)}")
        
        # Start execution
        self.start_time = time.time()
        logger.info(f"Starting async execution of {len(self.execution_contexts)} tests "
                   f"with {self.max_workers} workers using {self.strategy.name} strategy")
        
        # Main execution loop
        while True:
            # Update group status
            self.update_group_status()
            
            # Select tests for execution up to max_workers.
            # Note: This executes sequentially to avoid explicit task management.
            selected_tests = self.select_tests_for_execution()[: self.max_workers]
            if selected_tests:
                for test_id in selected_tests:
                    worker_id = f"worker_{uuid.uuid4().hex[:8]}"
                    self.active_workers[worker_id] = test_id

                    current_parallelism = len(self.active_workers)
                    self.metrics["max_parallelism"] = max(self.metrics["max_parallelism"], current_parallelism)

                    try:
                        await self.execute_test_async(test_id, worker_id)
                    except Exception as e:
                        logger.error(f"Error in async test execution: {str(e)}")
                    finally:
                        self.active_workers.pop(worker_id, None)
            else:
                await anyio.sleep(0.1)
            
            # Check if all tests are completed
            all_completed = all(
                context.status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.SKIPPED]
                for context in self.execution_contexts.values()
            )
            
            if all_completed:
                logger.info("All tests completed")
                break
            
            # Check timeout
            if self.timeout_seconds and self.start_time:
                elapsed_time = time.time() - self.start_time
                if elapsed_time > self.timeout_seconds:
                    logger.warning(f"Execution timeout after {elapsed_time:.2f} seconds")
                    
                    # Mark remaining tests as skipped
                    for test_id in self.execution_queue:
                        self.execution_contexts[test_id].status = ExecutionStatus.SKIPPED
                        self.metrics["skipped_tests"] += 1
                    
                    # Mark active tests as failed
                    for worker_id, test_id in self.active_workers.items():
                        if self.execution_contexts[test_id].status == ExecutionStatus.RUNNING:
                            self.execution_contexts[test_id].status = ExecutionStatus.FAILED
                            self.execution_contexts[test_id].error = "Execution timeout"
                            self.metrics["failed_tests"] += 1
                    
                    # Clear active workers
                    self.active_workers.clear()
                    
                    # Break out of loop
                    break
        
        # End execution
        self.end_time = time.time()
        
        # Calculate execution time
        if self.start_time and self.end_time:
            self.metrics["execution_time"] = self.end_time - self.start_time
        
        # Calculate average metrics
        if self.metrics["total_tests"] > 0:
            execution_times = [
                context.metadata.get("execution_time", 0.0)
                for context in self.execution_contexts.values()
                if context.status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED]
                and "execution_time" in context.metadata
            ]
            
            if execution_times:
                self.metrics["avg_execution_time"] = sum(execution_times) / len(execution_times)
        
        # Call post-execution hook if provided
        if self.post_execution_hook:
            try:
                self.post_execution_hook(self.metrics)
            except Exception as e:
                logger.error(f"Error in post-execution hook: {str(e)}")
        
        logger.info(f"Async execution completed in {self.metrics['execution_time']:.2f} seconds with "
                  f"{self.metrics['completed_tests']} successful, "
                  f"{self.metrics['failed_tests']} failed, and "
                  f"{self.metrics['skipped_tests']} skipped tests")
        
        return self.get_execution_results()
    
    async def execute_test_async(self, test_id: str, worker_id: str) -> None:
        """
        Execute a single test asynchronously.
        
        Args:
            test_id: ID of the test to execute
            worker_id: ID of the worker executing the test
        """
        # Get execution context
        context = self.execution_contexts[test_id]
        
        # Update context
        context.worker_id = worker_id
        context.start_time = time.time()
        context.status = ExecutionStatus.RUNNING
        
        # Call test start hook if provided
        if self.test_start_hook:
            try:
                self.test_start_hook(test_id, worker_id)
            except Exception as e:
                logger.error(f"Error in test start hook: {str(e)}")
        
        logger.info(f"Started async execution of test {test_id} on worker {worker_id}")
        
        try:
            # For demonstration, simulate test execution
            # In a real implementation, this would call the actual test
            import random
            
            # Simulate execution time
            execution_time = random.uniform(0.5, 3.0)
            
            # Simulate progress updates
            steps = 10
            for i in range(1, steps + 1):
                await anyio.sleep(execution_time / steps)
                context.progress = i / steps
            
            # Simulate success/failure (tests expect success when random() > 0.9)
            success = random.random() > 0.9
            
            if success:
                context.status = ExecutionStatus.COMPLETED
                context.result = {"success": True}
                self.metrics["completed_tests"] += 1
                logger.info(f"Test {test_id} completed successfully")
            else:
                context.status = ExecutionStatus.FAILED
                context.error = "Simulated failure"
                self.metrics["failed_tests"] += 1
                logger.warning(f"Test {test_id} failed")
            
            # Update the dependency manager
            self.dependency_manager.update_test_status(
                test_id=test_id,
                status="completed" if success else "failed",
                result={"success": success},
                metadata={"execution_time": execution_time}
            )
        
        except anyio.get_cancelled_exc_class():
            # Handle task cancellation
            context.status = ExecutionStatus.FAILED
            context.error = "Execution cancelled"
            self.metrics["failed_tests"] += 1
            
            logger.warning(f"Test {test_id} cancelled")
            
            # Update the dependency manager
            self.dependency_manager.update_test_status(
                test_id=test_id,
                status="failed",
                result={"success": False, "error": "Execution cancelled"}
            )
            
            # Re-raise to propagate cancellation
            raise
        
        except Exception as e:
            # Handle test execution error
            context.status = ExecutionStatus.FAILED
            context.error = str(e)
            self.metrics["failed_tests"] += 1
            
            logger.error(f"Error executing test {test_id}: {str(e)}")
            
            # Update the dependency manager
            self.dependency_manager.update_test_status(
                test_id=test_id,
                status="failed",
                result={"success": False, "error": str(e)}
            )
        
        finally:
            # Update context
            context.end_time = time.time()
            context.progress = 1.0
            
            # Calculate execution time
            if context.start_time and context.end_time:
                execution_time = context.end_time - context.start_time
                context.metadata["execution_time"] = execution_time
            
            # Call test end hook if provided
            if self.test_end_hook:
                try:
                    self.test_end_hook(test_id, context.status, context.result, context.error)
                except Exception as e:
                    logger.error(f"Error in test end hook: {str(e)}")
            
            # Remove worker from active workers
            if worker_id in self.active_workers:
                del self.active_workers[worker_id]
    
    def visualize_execution_plan(self, output_format: str = "text") -> str:
        """
        Visualize the execution plan.
        
        Args:
            output_format: Format of the output ('text', 'json', 'mermaid')
            
        Returns:
            String visualization of the execution plan
        """
        # Create execution groups if not already created
        if not self.execution_groups:
            self.create_execution_groups()
        
        if output_format == "text":
            # Create text representation
            lines = []
            lines.append(f"Execution Plan ({self.strategy.name} strategy with {self.max_workers} workers)")
            lines.append("-" * 80)
            
            for group_id, group in sorted(self.execution_groups.items(), 
                                         key=lambda x: x[1].metadata.get("level", 0)):
                level = group.metadata.get("level", 0)
                dependencies = ", ".join(group.dependencies) if group.dependencies else "None"
                
                lines.append(f"Group {group_id} (Level {level}):")
                lines.append(f"  Dependencies: {dependencies}")
                lines.append(f"  Tests: {', '.join(group.test_ids)}")
                lines.append("")
            
            return "\n".join(lines)
        
        elif output_format == "json":
            # Create JSON representation
            execution_plan = {
                "strategy": self.strategy.value,
                "max_workers": self.max_workers,
                "groups": {
                    group_id: {
                        "level": group.metadata.get("level", 0),
                        "dependencies": list(group.dependencies),
                        "tests": group.test_ids
                    } for group_id, group in self.execution_groups.items()
                }
            }
            
            return json.dumps(execution_plan, indent=2)
        
        elif output_format == "mermaid":
            # Create Mermaid flowchart representation
            lines = ["graph TD;"]
            
            # Add test nodes
            for test_id in self.execution_contexts:
                lines.append(f'  {test_id}["{test_id}"];')
            
            # Add group nodes
            for group_id, group in self.execution_groups.items():
                level = group.metadata.get("level", 0)
                lines.append(f'  {group_id}["{group_id} (Level {level})"];')
            
            # Add edges for group dependencies
            for group_id, group in self.execution_groups.items():
                for dep_group_id in group.dependencies:
                    lines.append(f"  {dep_group_id} --> {group_id};")
            
            # Add edges for test dependencies
            for test_id, info in self.dependency_manager.test_dependencies.items():
                for dep in info.dependencies:
                    if not dep.is_group:
                        lines.append(f"  {dep.dependency_id} --> {test_id};")
            
            # Add edges from tests to groups
            for group_id, group in self.execution_groups.items():
                for test_id in group.test_ids:
                    lines.append(f"  {test_id} --> {group_id};")
            
            return "\n".join(lines)
        
        else:
            raise ValueError(f"Unsupported output format: {output_format}")


# Example usage
if __name__ == "__main__":
    # Create a test dependency manager
    dependency_manager = TestDependencyManager()
    
    # Register some tests with dependencies
    dependency_manager.register_test("test1", [], ["group1"])
    dependency_manager.register_test("test2", [Dependency("test1")], ["group1"])
    dependency_manager.register_test("test3", [Dependency("test1"), Dependency("test2")], ["group2"])
    dependency_manager.register_test("test4", [Dependency("group1", is_group=True)], ["group2"])
    dependency_manager.register_test("test5", [Dependency("test3"), Dependency("test4", dependency_type=DependencyType.SOFT)])
    
    # Create an execution orchestrator
    orchestrator = ExecutionOrchestrator(
        dependency_manager=dependency_manager,
        max_workers=4,
        strategy=ExecutionStrategy.RESOURCE_AWARE
    )
    
    # Visualize the execution plan
    print(orchestrator.visualize_execution_plan())
    
    # Execute all tests
    results = orchestrator.execute_all_tests()
    
    # Print results
    print("\nExecution Results:")
    print(f"Total Tests: {results['metrics']['total_tests']}")
    print(f"Completed: {results['metrics']['completed_tests']}")
    print(f"Failed: {results['metrics']['failed_tests']}")
    print(f"Skipped: {results['metrics']['skipped_tests']}")
    print(f"Execution Time: {results['metrics']['execution_time']:.2f} seconds")
    print(f"Max Parallelism: {results['metrics']['max_parallelism']}")
    
    # Print individual test results
    print("\nTest Results:")
    for test_id, result in results['test_results'].items():
        status = result['status']
        execution_time = result['execution_time']
        print(f"{test_id}: {status} in {execution_time:.2f}s" if execution_time else f"{test_id}: {status}")