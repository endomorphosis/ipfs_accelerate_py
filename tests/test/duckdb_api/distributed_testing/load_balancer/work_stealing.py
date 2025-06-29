#!/usr/bin/env python3
"""
Distributed Testing Framework - Work Stealing Algorithm

This module implements the work stealing algorithm for the adaptive load balancing
system in the distributed testing framework. It allows idle workers to proactively
steal tasks from overloaded workers, improving overall resource utilization.

Key features:
- Detects and redistributes workload across worker nodes
- Balances between worker specialization and load distribution
- Implements priority-aware stealing policies
- Provides automatic migration of tasks between workers
- Supports transaction-based state management during migrations
- Implements backpressure mechanisms for system stability
"""

import logging
import time
import random
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass

from .models import (
    WorkerCapabilities, 
    WorkerLoad, 
    TestRequirements, 
    WorkerAssignment
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("work_stealing")


@dataclass
class StealingOpportunity:
    """Represents an opportunity to steal work from an overloaded worker."""
    source_worker_id: str
    target_worker_id: str
    task_id: str
    task_requirements: TestRequirements
    priority: int  # Higher value means higher priority for stealing
    estimated_benefit: float  # Estimated improvement in execution time
    migration_cost: float  # Estimated cost of migration


class WorkStealer:
    """Implements a work stealing algorithm for load balancing."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the work stealer.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = {
            # Thresholds for determining worker states
            "high_load_threshold": 0.8,  # Load above this is considered high
            "low_load_threshold": 0.3,  # Load below this is considered low
            "load_imbalance_threshold": 0.4,  # Min difference to consider stealing
            "idle_threshold": 0.2,  # Load below this is considered idle
            
            # Stealing policies
            "enable_priority_aware_stealing": True,  # Consider task priority in stealing decisions
            "enable_specialization_aware_stealing": True,  # Consider worker specialization in stealing decisions
            "enable_cost_benefit_analysis": True,  # Analyze cost vs benefit of stealing
            
            # Migration parameters
            "min_remaining_time": 10.0,  # Minimum remaining execution time (seconds) to consider stealing
            "max_migration_overhead": 0.5,  # Maximum allowed migration overhead ratio
            "max_simultaneous_migrations": 3,  # Maximum number of migrations at once
            "min_transfer_interval": 30.0,  # Minimum time (seconds) between transfers for same worker
            
            # Backpressure parameters
            "backpressure_threshold": 0.9,  # Load above this triggers backpressure
            "backpressure_cooldown": 60.0,  # Cooldown period (seconds) after backpressure
            
            # Transaction management
            "transaction_timeout": 30.0,  # Timeout for migration transactions (seconds)
            "retry_attempts": 2,  # Number of retry attempts for failed migrations
        }
        
        # Override defaults with provided config
        if config:
            self.config.update(config)
        
        # State tracking
        self.ongoing_migrations = {}  # task_id -> (source_worker, target_worker, start_time)
        self.recent_migrations = {}  # worker_id -> List[timestamp]
        self.backpressure_state = {}  # worker_id -> end_timestamp
        
        logger.info("Work stealer initialized")
    
    def identify_stealing_opportunities(
        self,
        worker_capabilities: Dict[str, WorkerCapabilities],
        worker_loads: Dict[str, WorkerLoad],
        assigned_tasks: Dict[str, Tuple[str, TestRequirements, datetime]],
        performance_data: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> List[StealingOpportunity]:
        """Identify opportunities for work stealing across workers.
        
        Args:
            worker_capabilities: Dict of worker_id -> WorkerCapabilities
            worker_loads: Dict of worker_id -> WorkerLoad
            assigned_tasks: Dict of task_id -> (worker_id, task_requirements, start_time)
            performance_data: Optional dict with worker performance data
            
        Returns:
            List of StealingOpportunity objects representing possible migrations
        """
        # Skip if not enough workers
        if len(worker_loads) < 2:
            return []
        
        # Categorize workers by load
        overloaded_workers = []
        underloaded_workers = []
        
        for worker_id, load in worker_loads.items():
            load_score = load.calculate_load_score()
            
            # Skip workers under backpressure
            if worker_id in self.backpressure_state:
                backpressure_end = self.backpressure_state[worker_id]
                if datetime.now() < backpressure_end:
                    logger.debug(f"Worker {worker_id} under backpressure until {backpressure_end}")
                    continue
                else:
                    # Clear expired backpressure
                    del self.backpressure_state[worker_id]
            
            # Categorize by load
            if load_score >= self.config["high_load_threshold"]:
                overloaded_workers.append(worker_id)
            elif load_score <= self.config["low_load_threshold"]:
                underloaded_workers.append(worker_id)
                
        if not overloaded_workers or not underloaded_workers:
            logger.debug("No work stealing opportunities (no imbalance)")
            return []
            
        logger.debug(f"Found {len(overloaded_workers)} overloaded and {len(underloaded_workers)} underloaded workers")
        
        # Identify tasks that could potentially be migrated
        opportunities = []
        
        for task_id, (source_worker_id, task_requirements, start_time) in assigned_tasks.items():
            # Skip if task already being migrated
            if task_id in self.ongoing_migrations:
                continue
                
            # Skip if source worker not overloaded
            if source_worker_id not in overloaded_workers:
                continue
                
            # Skip if task has been running too long (likely to finish soon)
            elapsed_time = (datetime.now() - start_time).total_seconds()
            estimated_total_time = task_requirements.expected_duration
            
            if estimated_total_time > 0:
                estimated_remaining = max(0, estimated_total_time - elapsed_time)
                
                if estimated_remaining < self.config["min_remaining_time"]:
                    logger.debug(f"Task {task_id} on {source_worker_id} likely to finish soon (est. {estimated_remaining:.1f}s remaining)")
                    continue
            
            # Check compatibility with underloaded workers
            for target_worker_id in underloaded_workers:
                # Skip self-stealing
                if target_worker_id == source_worker_id:
                    continue
                    
                target_capabilities = worker_capabilities.get(target_worker_id)
                target_load = worker_loads.get(target_worker_id)
                
                # Skip if missing capabilities or load info
                if not target_capabilities or not target_load:
                    continue
                    
                # Check if target worker can handle this task
                if not self._is_compatible(task_requirements, target_capabilities, target_load):
                    logger.debug(f"Worker {target_worker_id} not compatible with task {task_id}")
                    continue
                
                # Calculate priority and benefit of this stealing opportunity
                priority, estimated_benefit, migration_cost = self._evaluate_stealing_opportunity(
                    task_id=task_id,
                    task_requirements=task_requirements,
                    source_worker_id=source_worker_id,
                    target_worker_id=target_worker_id,
                    worker_loads=worker_loads,
                    performance_data=performance_data,
                    elapsed_time=elapsed_time
                )
                
                # Skip if not beneficial
                if estimated_benefit <= 0 or migration_cost >= estimated_benefit:
                    continue
                
                # Create stealing opportunity
                opportunity = StealingOpportunity(
                    source_worker_id=source_worker_id,
                    target_worker_id=target_worker_id,
                    task_id=task_id,
                    task_requirements=task_requirements,
                    priority=priority,
                    estimated_benefit=estimated_benefit,
                    migration_cost=migration_cost
                )
                
                opportunities.append(opportunity)
        
        # Sort opportunities by priority (descending)
        opportunities.sort(key=lambda x: x.priority, reverse=True)
        
        return opportunities
    
    def select_tasks_to_steal(
        self,
        opportunities: List[StealingOpportunity],
        max_steals: Optional[int] = None
    ) -> List[StealingOpportunity]:
        """Select which tasks to actually steal from the list of opportunities.
        
        Args:
            opportunities: List of stealing opportunities
            max_steals: Maximum number of tasks to steal (None for automatic)
            
        Returns:
            List of stealing opportunities to execute
        """
        if not opportunities:
            return []
            
        # Determine maximum number of steals
        max_steals = max_steals or self.config["max_simultaneous_migrations"]
        
        # Group opportunities by source worker to prevent excessive stealing from one worker
        by_source = {}
        for opportunity in opportunities:
            if opportunity.source_worker_id not in by_source:
                by_source[opportunity.source_worker_id] = []
            by_source[opportunity.source_worker_id].append(opportunity)
        
        # Select best opportunities while respecting limits
        selected = []
        sources_used = set()
        targets_used = set()
        
        for opportunity in opportunities:
            # Stop if we've reached the maximum
            if len(selected) >= max_steals:
                break
                
            # Limit the number of tasks stolen from a single worker
            if opportunity.source_worker_id in sources_used:
                if sources_used.count(opportunity.source_worker_id) >= 2:  # Max 2 steals per source
                    continue
                    
            # Limit the number of tasks assigned to a single worker
            if opportunity.target_worker_id in targets_used:
                if targets_used.count(opportunity.target_worker_id) >= 2:  # Max 2 steals per target
                    continue
            
            # Check for recent migrations for this worker
            recent_source = self.recent_migrations.get(opportunity.source_worker_id, [])
            recent_target = self.recent_migrations.get(opportunity.target_worker_id, [])
            
            now = datetime.now()
            min_interval = self.config["min_transfer_interval"]
            
            # Filter for recent migrations within the minimum interval
            recent_source = [t for t in recent_source if (now - t).total_seconds() < min_interval]
            recent_target = [t for t in recent_target if (now - t).total_seconds() < min_interval]
            
            # Update recent migrations list
            self.recent_migrations[opportunity.source_worker_id] = recent_source
            self.recent_migrations[opportunity.target_worker_id] = recent_target
            
            # Skip if either worker has had too many recent migrations
            if len(recent_source) >= 2 or len(recent_target) >= 2:
                continue
            
            # Select this opportunity
            selected.append(opportunity)
            sources_used.add(opportunity.source_worker_id)
            targets_used.add(opportunity.target_worker_id)
            
            # Track recent migration
            if opportunity.source_worker_id not in self.recent_migrations:
                self.recent_migrations[opportunity.source_worker_id] = []
            if opportunity.target_worker_id not in self.recent_migrations:
                self.recent_migrations[opportunity.target_worker_id] = []
                
            self.recent_migrations[opportunity.source_worker_id].append(now)
            self.recent_migrations[opportunity.target_worker_id].append(now)
        
        return selected
    
    def execute_stealing(
        self,
        opportunity: StealingOpportunity,
        task_executor
    ) -> bool:
        """Execute a work stealing operation.
        
        Args:
            opportunity: Stealing opportunity to execute
            task_executor: Callable to execute task on target worker
            
        Returns:
            True if stealing succeeded, False otherwise
        """
        task_id = opportunity.task_id
        source_id = opportunity.source_worker_id
        target_id = opportunity.target_worker_id
        
        logger.info(f"Stealing task {task_id} from worker {source_id} to {target_id}")
        
        # Record start of migration
        self.ongoing_migrations[task_id] = (source_id, target_id, datetime.now())
        
        try:
            # Execute task transfer using provided executor
            success = task_executor(
                task_id=task_id,
                source_worker_id=source_id,
                target_worker_id=target_id,
                task_requirements=opportunity.task_requirements
            )
            
            if success:
                logger.info(f"Successfully migrated task {task_id} from {source_id} to {target_id}")
            else:
                logger.warning(f"Failed to migrate task {task_id} from {source_id} to {target_id}")
                
            return success
            
        except Exception as e:
            logger.error(f"Error during task migration: {e}")
            return False
        finally:
            # Clean up regardless of outcome
            if task_id in self.ongoing_migrations:
                del self.ongoing_migrations[task_id]
    
    def apply_backpressure(self, worker_id: str, duration_seconds: Optional[float] = None) -> None:
        """Apply backpressure to a worker to prevent stealing for a period.
        
        Args:
            worker_id: Worker ID to apply backpressure to
            duration_seconds: Optional custom duration in seconds
        """
        duration = duration_seconds or self.config["backpressure_cooldown"]
        end_time = datetime.now() + timedelta(seconds=duration)
        
        self.backpressure_state[worker_id] = end_time
        
        logger.info(f"Applied backpressure to worker {worker_id} until {end_time}")
    
    def _is_compatible(
        self,
        requirements: TestRequirements,
        capabilities: WorkerCapabilities,
        load: WorkerLoad
    ) -> bool:
        """Check if worker is compatible with task requirements."""
        # Check capabilities
        if not capabilities.is_compatible_with(requirements):
            return False
            
        # Check if worker has capacity for this task
        if not load.has_capacity_for(requirements, capabilities):
            return False
            
        return True
    
    def _evaluate_stealing_opportunity(
        self,
        task_id: str,
        task_requirements: TestRequirements,
        source_worker_id: str,
        target_worker_id: str,
        worker_loads: Dict[str, WorkerLoad],
        performance_data: Optional[Dict[str, Dict[str, Any]]],
        elapsed_time: float
    ) -> Tuple[int, float, float]:
        """Evaluate the priority and benefit of a stealing opportunity.
        
        Args:
            task_id: ID of the task
            task_requirements: Requirements for the task
            source_worker_id: ID of the source worker
            target_worker_id: ID of the target worker
            worker_loads: Dict of worker_id -> WorkerLoad
            performance_data: Optional dict with worker performance data
            elapsed_time: Time the task has been running (seconds)
            
        Returns:
            Tuple of (priority, estimated_benefit, migration_cost)
        """
        # Base priority score
        priority = 50
        
        # Factor 1: Load imbalance
        source_load = worker_loads[source_worker_id].calculate_load_score()
        target_load = worker_loads[target_worker_id].calculate_load_score()
        
        load_diff = source_load - target_load
        
        # Only valuable if significant load difference
        if load_diff < self.config["load_imbalance_threshold"]:
            return 0, 0.0, 0.0
            
        # Higher load difference = higher priority
        priority += int(load_diff * 50)  # Up to +50 points
        
        # Factor 2: Task priority (if enabled)
        if self.config["enable_priority_aware_stealing"]:
            # Higher priority tasks (lower number) get higher stealing priority
            task_priority = task_requirements.priority
            if task_priority <= 2:  # High priority
                priority += 30
            elif task_priority >= 4:  # Low priority
                priority -= 20
        
        # Factor 3: Worker specialization (if enabled)
        specialization_boost = 0
        if self.config["enable_specialization_aware_stealing"] and performance_data:
            # Check if target worker has better performance for this type of task
            if target_worker_id in performance_data:
                worker_perf = performance_data[target_worker_id]
                
                # Check model-specific performance if available
                if (task_requirements.model_id and 
                    task_requirements.model_id in worker_perf and 
                    task_requirements.test_type in worker_perf[task_requirements.model_id]):
                    
                    target_perf = worker_perf[task_requirements.model_id][task_requirements.test_type]
                    
                    # Higher specialization score if target has good performance on this task
                    if target_perf.success_rate > 0.9 and target_perf.sample_count > 5:
                        specialization_boost = 30
                
                # Check model family performance as fallback
                elif (task_requirements.model_family and 
                    "model_family" in worker_perf and 
                    task_requirements.model_family in worker_perf["model_family"]):
                    
                    family_perf = worker_perf["model_family"][task_requirements.model_family]
                    
                    # Moderate specialization score if target is good for this model family
                    if family_perf.success_rate > 0.9 and family_perf.sample_count > 5:
                        specialization_boost = 20
                
                # Check test type performance as fallback
                elif (task_requirements.test_type and 
                    "test_type" in worker_perf and 
                    task_requirements.test_type in worker_perf["test_type"]):
                    
                    type_perf = worker_perf["test_type"][task_requirements.test_type]
                    
                    # Small specialization score if target is good for this test type
                    if type_perf.success_rate > 0.9 and type_perf.sample_count > 5:
                        specialization_boost = 10
            
            priority += specialization_boost
        
        # Estimate benefit of stealing (execution time improvement in seconds)
        estimated_benefit = self._estimate_execution_benefit(
            task_requirements=task_requirements,
            source_worker_id=source_worker_id,
            target_worker_id=target_worker_id,
            worker_loads=worker_loads,
            performance_data=performance_data,
            elapsed_time=elapsed_time
        )
        
        # Estimate cost of migration (overhead in seconds)
        migration_cost = self._estimate_migration_cost(
            task_requirements=task_requirements,
            elapsed_time=elapsed_time
        )
        
        # Adjust priority based on benefit and cost
        if self.config["enable_cost_benefit_analysis"]:
            if estimated_benefit > 0 and migration_cost > 0:
                # Calculate benefit/cost ratio
                benefit_ratio = estimated_benefit / migration_cost
                
                if benefit_ratio > 5.0:
                    # Excellent benefit/cost ratio
                    priority += 40
                elif benefit_ratio > 3.0:
                    # Good benefit/cost ratio
                    priority += 20
                elif benefit_ratio > 1.5:
                    # Moderate benefit/cost ratio
                    priority += 10
                elif benefit_ratio < 1.0:
                    # Poor benefit/cost ratio
                    priority -= 30
        
        return priority, estimated_benefit, migration_cost
    
    def _estimate_execution_benefit(
        self,
        task_requirements: TestRequirements,
        source_worker_id: str,
        target_worker_id: str,
        worker_loads: Dict[str, WorkerLoad],
        performance_data: Optional[Dict[str, Dict[str, Any]]],
        elapsed_time: float
    ) -> float:
        """Estimate the benefit (time saved) by migrating a task.
        
        Args:
            task_requirements: Requirements for the task
            source_worker_id: ID of the source worker
            target_worker_id: ID of the target worker
            worker_loads: Dict of worker_id -> WorkerLoad
            performance_data: Optional dict with worker performance data
            elapsed_time: Time the task has been running (seconds)
            
        Returns:
            Estimated time saved in seconds (negative if slower)
        """
        # Get expected total execution time
        total_expected_time = task_requirements.expected_duration
        
        # Calculate estimated completion times on source and target
        # Based on load and any available performance data
        source_load = worker_loads[source_worker_id].calculate_load_score()
        target_load = worker_loads[target_worker_id].calculate_load_score()
        
        # For high load, execution becomes slower due to resource contention
        source_slowdown = self._calculate_load_slowdown(source_load)
        target_slowdown = self._calculate_load_slowdown(target_load)
        
        # Apply performance data if available
        perf_factor = 1.0
        if performance_data:
            # Check if we have performance data for both workers
            if (target_worker_id in performance_data and 
                source_worker_id in performance_data):
                
                # Try to find most specific performance data
                target_perf = None
                source_perf = None
                
                # Check model-specific performance
                if (task_requirements.model_id and task_requirements.test_type):
                    target_data = performance_data.get(target_worker_id, {})
                    source_data = performance_data.get(source_worker_id, {})
                    
                    target_model = target_data.get(task_requirements.model_id, {})
                    source_model = source_data.get(task_requirements.model_id, {})
                    
                    target_perf = target_model.get(task_requirements.test_type)
                    source_perf = source_model.get(task_requirements.test_type)
                
                # Use family-level performance if available and no model-specific data
                if (not target_perf or not source_perf) and task_requirements.model_family:
                    target_data = performance_data.get(target_worker_id, {})
                    source_data = performance_data.get(source_worker_id, {})
                    
                    if "model_family" in target_data and "model_family" in source_data:
                        target_family = target_data["model_family"].get(task_requirements.model_family)
                        source_family = source_data["model_family"].get(task_requirements.model_family)
                        
                        if target_family and source_family:
                            target_perf = target_family
                            source_perf = source_family
                
                # If we have both pieces of performance data, calculate relative speed
                if target_perf and source_perf and source_perf.average_execution_time > 0:
                    perf_ratio = source_perf.average_execution_time / target_perf.average_execution_time
                    perf_factor = perf_ratio
        
        # Remaining time estimate for the source worker
        remaining_time_source = max(0, total_expected_time - elapsed_time) * source_slowdown
        
        # Estimated time on target worker (applying performance factor)
        # We need to finish the remaining work on the target, but might be faster/slower
        # based on target worker's performance characteristics
        remaining_time_target = max(0, total_expected_time - elapsed_time) * target_slowdown / perf_factor
        
        # Benefit is the difference in remaining time
        # Positive value means time saved by migrating
        benefit = remaining_time_source - remaining_time_target
        
        return benefit
    
    def _estimate_migration_cost(
        self,
        task_requirements: TestRequirements,
        elapsed_time: float
    ) -> float:
        """Estimate the cost (overhead) of migrating a task.
        
        Args:
            task_requirements: Requirements for the task
            elapsed_time: Time the task has been running (seconds)
            
        Returns:
            Estimated migration cost in seconds
        """
        # Base migration cost (overhead of stopping and restarting)
        base_cost = 5.0  # Seconds
        
        # Additional cost based on task characteristics
        # For simplicity, we assume migration cost scales with task size/complexity
        complexity_factor = 1.0
        
        # If we know memory requirements, use that as a proxy for state size
        if hasattr(task_requirements, 'memory_gb') and task_requirements.memory_gb > 0:
            # More memory = more state to transfer
            complexity_factor = max(1.0, task_requirements.memory_gb / 2.0)  # Scale with memory (GB)
        
        # If checkpoint/resume is supported, cost is lower after initial phase
        # Assume checkpoint/resume support for simplicity
        if elapsed_time > 30.0:  # Task has been running for a while
            # Cost is lower because we can checkpoint progress
            checkpoint_factor = 0.6  # 40% reduction in cost
        else:
            # No reduction for tasks just starting
            checkpoint_factor = 1.0
        
        # Calculate total migration cost
        migration_cost = base_cost * complexity_factor * checkpoint_factor
        
        return migration_cost
    
    def _calculate_load_slowdown(self, load: float) -> float:
        """Calculate slowdown factor based on worker load.
        
        Args:
            load: Worker load (0.0 to 1.0)
            
        Returns:
            Slowdown factor (1.0 = no slowdown, >1.0 = slower)
        """
        # No slowdown for low load
        if load < 0.5:
            return 1.0
            
        # Exponential slowdown as load approaches 1.0
        # At load=0.5: slowdown=1.0
        # At load=0.8: slowdown≈1.5
        # At load=0.9: slowdown≈2.0
        # At load=0.95: slowdown≈3.0
        if load >= 0.95:
            return 3.0
        elif load >= 0.9:
            return 2.0
        elif load >= 0.8:
            return 1.5
        else:
            # Linear interpolation between 0.5 and 0.8
            return 1.0 + (load - 0.5) * (1.5 - 1.0) / (0.8 - 0.5)