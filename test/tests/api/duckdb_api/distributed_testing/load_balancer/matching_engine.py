#!/usr/bin/env python3
"""
Distributed Testing Framework - Load Balancer Matching Engine

This module implements the matching engine for the adaptive load balancing
system. It's responsible for matching tasks to optimal workers based on
requirements, capabilities, and current load.

Key features:
- Multi-factor scoring system for task-worker combinations
- Capability-based matching to ensure task requirements are satisfied
- Performance-aware matching based on historical execution data
- Load-aware distribution to maintain balanced utilization
- Specialized hardware affinity for optimal resource utilization
"""

import os
import json
import logging
import heapq
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass

# Import models
from test.tests.api.duckdb_api.distributed_testing.load_balancer.models import (
    WorkerCapabilities, 
    WorkerPerformance, 
    WorkerLoad, 
    TestRequirements, 
    WorkerAssignment
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("matching_engine")


@dataclass
class WorkerScore:
    """Score for a worker against a specific task."""
    worker_id: str
    task_id: str
    capability_score: float   # How well worker capabilities match task requirements (0-1)
    performance_score: float  # Historical performance score for similar tasks (0-1)
    load_score: float         # Current load score (0-1, higher is better - less loaded)
    overall_score: float      # Combined ranking score (0-1, higher is better)
    worker_capabilities: Optional[WorkerCapabilities] = None
    worker_load: Optional[WorkerLoad] = None


class MatchingEngine:
    """Matches tasks to optimal workers based on various factors."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the matching engine.
        
        Args:
            config: Optional configuration dictionary with custom weights and settings
        """
        self.config = {
            # Scoring weights for different factors (must sum to 1.0)
            "capability_weight": 0.4,   # Weight for capability matching
            "performance_weight": 0.3,  # Weight for historical performance
            "load_weight": 0.3,         # Weight for current load balance
            
            # Thresholds and limits
            "minimum_capability_score": 0.5,   # Minimum score to be considered compatible
            "load_threshold_high": 0.8,        # High load threshold (0-1)
            "load_threshold_low": 0.2,         # Low load threshold (0-1)
            
            # Specialized matching options
            "enable_affinity_bonus": True,     # Enable bonus for worker-task affinity
            "affinity_bonus_multiplier": 0.2,  # Bonus multiplier for affinity (0-1)
            "enable_penalty_for_migrations": True,  # Enable penalty for task migrations
            "migration_penalty_multiplier": 0.1,    # Penalty multiplier for migrations
            
            # Advanced options
            "enable_predictive_scoring": False,  # Use predicted future load for scoring
            "prediction_weight": 0.2,           # Weight for predictive component
            "consistency_bonus_weight": 0.1,    # Weight for consistent performance
        }
        
        # Override defaults with provided config
        if config:
            self.config.update(config)
        
        logger.info("Matching engine initialized")
    
    def find_best_worker(
        self,
        task_requirements: TestRequirements,
        available_workers: Dict[str, WorkerCapabilities],
        worker_loads: Dict[str, WorkerLoad],
        worker_performances: Optional[Dict[str, Dict[str, WorkerPerformance]]] = None,
        previous_assignments: Optional[Dict[str, str]] = None,
        excluded_workers: Optional[Set[str]] = None
    ) -> Optional[str]:
        """Find the best worker for a specific task.
        
        Args:
            task_requirements: Requirements for the task
            available_workers: Dict of worker_id -> WorkerCapabilities
            worker_loads: Dict of worker_id -> WorkerLoad
            worker_performances: Optional dict of worker_id -> dict of test_type -> WorkerPerformance
            previous_assignments: Optional dict of task_id -> worker_id for previous assignments
            excluded_workers: Optional set of worker_ids to exclude from consideration
            
        Returns:
            worker_id of the best worker, or None if no suitable worker is found
        """
        # Filter out excluded workers
        if excluded_workers:
            worker_ids = [w_id for w_id in available_workers.keys() if w_id not in excluded_workers]
        else:
            worker_ids = list(available_workers.keys())
            
        if not worker_ids:
            logger.warning(f"No available workers for task {task_requirements.test_id}")
            return None
        
        # Score each worker
        worker_scores = []
        for worker_id in worker_ids:
            worker_capabilities = available_workers[worker_id]
            worker_load = worker_loads.get(worker_id)
            
            # Skip if worker load info is missing
            if not worker_load:
                logger.warning(f"Missing load information for worker {worker_id}")
                continue
                
            # Check if worker has capacity for this task
            if not worker_load.has_capacity_for(task_requirements, worker_capabilities):
                logger.debug(f"Worker {worker_id} lacks capacity for task {task_requirements.test_id}")
                continue
                
            # Calculate scores for this worker
            capability_score = self._calculate_capability_score(task_requirements, worker_capabilities)
            
            # Skip if capability score is below minimum threshold
            if capability_score < self.config["minimum_capability_score"]:
                logger.debug(f"Worker {worker_id} capability score {capability_score:.2f} below threshold")
                continue
                
            # Calculate remaining scores
            perf_score = self._calculate_performance_score(
                task_requirements, 
                worker_id, 
                worker_performances
            )
            
            load_score = self._calculate_load_score(
                task_requirements, 
                worker_id, 
                worker_load
            )
            
            # Apply bonus for affinity if enabled
            affinity_bonus = 0.0
            if self.config["enable_affinity_bonus"]:
                affinity_bonus = self._calculate_affinity_bonus(
                    task_requirements, 
                    worker_id, 
                    worker_performances
                )
                
            # Apply penalty for migrations if enabled
            migration_penalty = 0.0
            if (self.config["enable_penalty_for_migrations"] and 
                previous_assignments and 
                task_requirements.test_id in previous_assignments and
                previous_assignments[task_requirements.test_id] != worker_id):
                migration_penalty = self.config["migration_penalty_multiplier"]
            
            # Calculate overall score with weights
            overall_score = (
                self.config["capability_weight"] * capability_score +
                self.config["performance_weight"] * perf_score + 
                self.config["load_weight"] * load_score +
                affinity_bonus - 
                migration_penalty
            )
            
            # Normalize to 0-1 range
            overall_score = max(0.0, min(1.0, overall_score))
            
            # Create and store worker score
            score = WorkerScore(
                worker_id=worker_id, 
                task_id=task_requirements.test_id,
                capability_score=capability_score,
                performance_score=perf_score, 
                load_score=load_score,
                overall_score=overall_score,
                worker_capabilities=worker_capabilities,
                worker_load=worker_load
            )
            
            worker_scores.append(score)
            
        if not worker_scores:
            logger.warning(f"No suitable workers found for task {task_requirements.test_id}")
            return None
            
        # Find the worker with the highest overall score
        best_worker = max(worker_scores, key=lambda x: x.overall_score)
        
        logger.info(f"Selected worker {best_worker.worker_id} for task {task_requirements.test_id} "
                   f"with score {best_worker.overall_score:.2f}")
        
        return best_worker.worker_id
    
    def find_best_task_worker_pairs(
        self,
        tasks: Dict[str, TestRequirements],
        workers: Dict[str, WorkerCapabilities],
        worker_loads: Dict[str, WorkerLoad],
        worker_performances: Optional[Dict[str, Dict[str, WorkerPerformance]]] = None,
        max_assignments: Optional[int] = None
    ) -> List[Tuple[str, str]]:
        """Find optimal task-worker pairs for multiple tasks and workers.
        
        Args:
            tasks: Dict of task_id -> TestRequirements
            workers: Dict of worker_id -> WorkerCapabilities
            worker_loads: Dict of worker_id -> WorkerLoad
            worker_performances: Optional dict of worker_id -> dict of test_type -> WorkerPerformance
            max_assignments: Maximum number of assignments to make (None for no limit)
            
        Returns:
            List of (task_id, worker_id) tuples for the optimal assignments
        """
        # Calculate all valid task-worker scores
        all_scores = []
        for task_id, requirements in tasks.items():
            for worker_id, capabilities in workers.items():
                worker_load = worker_loads.get(worker_id)
                
                # Skip if worker load info is missing
                if not worker_load:
                    continue
                    
                # Check if worker has capacity for this task
                if not worker_load.has_capacity_for(requirements, capabilities):
                    continue
                    
                # Calculate scores for this worker-task combination
                capability_score = self._calculate_capability_score(requirements, capabilities)
                
                # Skip if capability score is below minimum threshold
                if capability_score < self.config["minimum_capability_score"]:
                    continue
                    
                # Calculate remaining scores
                perf_score = self._calculate_performance_score(
                    requirements, 
                    worker_id, 
                    worker_performances
                )
                
                load_score = self._calculate_load_score(
                    requirements, 
                    worker_id, 
                    worker_load
                )
                
                # Apply bonus for affinity if enabled
                affinity_bonus = 0.0
                if self.config["enable_affinity_bonus"]:
                    affinity_bonus = self._calculate_affinity_bonus(
                        requirements, 
                        worker_id, 
                        worker_performances
                    )
                
                # Calculate overall score with weights
                overall_score = (
                    self.config["capability_weight"] * capability_score +
                    self.config["performance_weight"] * perf_score + 
                    self.config["load_weight"] * load_score +
                    affinity_bonus
                )
                
                # Normalize to 0-1 range
                overall_score = max(0.0, min(1.0, overall_score))
                
                # Create and store score entry
                all_scores.append((overall_score, task_id, worker_id, worker_load))
        
        # Sort scores in descending order (highest score first)
        all_scores.sort(reverse=True)
        
        # Make assignments with greedy algorithm
        assignments = []
        assigned_tasks = set()
        updated_worker_loads = worker_loads.copy()
        
        for score, task_id, worker_id, worker_load in all_scores:
            # Skip if task is already assigned
            if task_id in assigned_tasks:
                continue
                
            # Skip if we've reached max assignments
            if max_assignments is not None and len(assignments) >= max_assignments:
                break
                
            # Use the updated load for this worker
            updated_load = updated_worker_loads.get(worker_id, worker_load)
            
            # Check if worker still has capacity for this task
            task_requirements = tasks[task_id]
            if not updated_load.has_capacity_for(task_requirements, workers.get(worker_id)):
                continue
                
            # Add assignment
            assignments.append((task_id, worker_id))
            assigned_tasks.add(task_id)
            
            # Update worker load (simulate resource reservation)
            updated_load.reserve_resources(
                task_id, 
                task_requirements, 
                workers.get(worker_id)
            )
            updated_worker_loads[worker_id] = updated_load
        
        logger.info(f"Made {len(assignments)} assignments out of {len(tasks)} tasks")
        return assignments
    
    def _calculate_capability_score(
        self, 
        requirements: TestRequirements, 
        capabilities: WorkerCapabilities
    ) -> float:
        """Calculate capability score based on how well worker capabilities match task requirements.
        
        Args:
            requirements: Task requirements
            capabilities: Worker capabilities
            
        Returns:
            Capability score between 0.0 and 1.0 (higher is better)
        """
        # Base score starts at 1.0
        score = 1.0
        
        # Required backend check
        if requirements.required_backend:
            if requirements.required_backend not in capabilities.supported_backends:
                return 0.0  # Hard requirement not met
            else:
                # Bonus for having the required backend
                score += 0.1
        
        # Preferred backend check
        if requirements.preferred_backend:
            if requirements.preferred_backend in capabilities.supported_backends:
                score += 0.1
            else:
                score -= 0.1
        
        # Memory check
        if capabilities.available_memory < requirements.minimum_memory:
            return 0.0  # Hard requirement not met
        else:
            # Score based on available memory vs required memory
            # Higher score if worker has more memory than required
            memory_ratio = min(3.0, capabilities.available_memory / requirements.minimum_memory)
            memory_score = 0.5 + (memory_ratio - 1.0) * 0.25  # Range: 0.5 to 1.0
            score *= memory_score
        
        # Accelerator check
        for accel_type, count in requirements.required_accelerators.items():
            if accel_type not in capabilities.available_accelerators:
                return 0.0  # Hard requirement not met
            
            if capabilities.available_accelerators[accel_type] < count:
                return 0.0  # Hard requirement not met
            
            # Score based on available vs required accelerators
            accel_ratio = min(3.0, capabilities.available_accelerators[accel_type] / count)
            accel_score = 0.5 + (accel_ratio - 1.0) * 0.25  # Range: 0.5 to 1.0
            score *= accel_score
        
        # Software check
        for sw_name, min_version in requirements.required_software.items():
            if sw_name not in capabilities.software_versions:
                return 0.0  # Hard requirement not met
            
            # TODO: Implement version comparison
            
        # CPU cores check - soft requirement
        if hasattr(requirements, 'minimum_cpu_cores') and requirements.minimum_cpu_cores > 0:
            if capabilities.cpu_cores < requirements.minimum_cpu_cores:
                score *= 0.5  # Penalty for not meeting soft requirement
            else:
                core_ratio = min(2.0, capabilities.cpu_cores / requirements.minimum_cpu_cores)
                core_score = 0.8 + (core_ratio - 1.0) * 0.2  # Range: 0.8 to 1.0
                score *= core_score
        
        # Normalize score to 0-1 range
        return max(0.0, min(1.0, score))
    
    def _calculate_performance_score(
        self, 
        requirements: TestRequirements, 
        worker_id: str,
        worker_performances: Optional[Dict[str, Dict[str, WorkerPerformance]]]
    ) -> float:
        """Calculate performance score based on historical data.
        
        Args:
            requirements: Task requirements
            worker_id: Worker ID
            worker_performances: Optional dict of worker_id -> dict of test_type -> WorkerPerformance
            
        Returns:
            Performance score between 0.0 and 1.0 (higher is better)
        """
        # Default score if no performance data
        if not worker_performances or worker_id not in worker_performances:
            return 0.5
        
        worker_perf = worker_performances[worker_id]
        
        # Check for exact match on model and test type
        if (requirements.model_id and requirements.test_type and 
            requirements.model_id in worker_perf and 
            requirements.test_type in worker_perf[requirements.model_id]):
            perf = worker_perf[requirements.model_id][requirements.test_type]
            
            # High score for successful history
            success_score = perf.success_rate
            
            # Execution time score - normalize across range
            time_score = 1.0  # Default high score
            if perf.average_execution_time > 0:
                expected_time = requirements.expected_duration if requirements.expected_duration > 0 else 60.0
                time_ratio = expected_time / perf.average_execution_time
                if time_ratio >= 1.0:
                    # Faster than expected
                    time_score = min(1.0, 0.8 + (time_ratio - 1.0) * 0.2)
                else:
                    # Slower than expected
                    time_score = max(0.2, 0.8 * time_ratio)
            
            # Sample count score - more samples means more confidence
            sample_weight = min(1.0, perf.sample_count / 10.0)  # Max weight after 10 samples
            
            # Weight the success score higher than the time score
            combined_score = 0.7 * success_score + 0.3 * time_score
            
            # Apply sample weight
            final_score = 0.5 + (combined_score - 0.5) * sample_weight
            
            return final_score
        
        # Check for model family match
        elif requirements.model_family and "model_family" in worker_perf:
            # TODO: Implement model family performance logic
            return 0.5
        
        # Check for test type match
        elif requirements.test_type and requirements.test_type in worker_perf:
            # TODO: Implement test type performance logic
            return 0.5
        
        # No relevant performance data
        return 0.5
    
    def _calculate_load_score(
        self, 
        requirements: TestRequirements, 
        worker_id: str,
        worker_load: WorkerLoad
    ) -> float:
        """Calculate load score based on current worker load.
        
        Args:
            requirements: Task requirements
            worker_id: Worker ID
            worker_load: Current worker load
            
        Returns:
            Load score between 0.0 and 1.0 (higher is better - less loaded)
        """
        # Calculate current load percentage (inverse, so higher is better)
        load_score = 1.0 - worker_load.calculate_load_score()
        
        # Adjust based on active tests count - better to use worker with fewer active tests
        active_tests_factor = max(0.5, 1.0 - (worker_load.active_tests * 0.1))
        
        # Adjust for worker warming/cooling state
        state_factor = 1.0
        if worker_load.warming_state:
            # Warming workers get lower score - prefer fully warmed workers
            state_factor = 0.7
        elif worker_load.cooling_state:
            # Cooling workers get lower score - prefer fully cooled workers
            state_factor = 0.5
        
        # Combine factors
        combined_score = load_score * active_tests_factor * state_factor
        
        # Adjust based on task priority - high priority tasks care less about load
        priority_factor = max(0.5, 1.0 - (requirements.priority - 1) * 0.1)
        adjusted_score = combined_score * priority_factor
        
        return max(0.0, min(1.0, adjusted_score))
    
    def _calculate_affinity_bonus(
        self, 
        requirements: TestRequirements, 
        worker_id: str,
        worker_performances: Optional[Dict[str, Dict[str, WorkerPerformance]]]
    ) -> float:
        """Calculate bonus for worker-task affinity based on historical data.
        
        Args:
            requirements: Task requirements
            worker_id: Worker ID
            worker_performances: Optional dict of worker_id -> dict of test_type -> WorkerPerformance
            
        Returns:
            Affinity bonus between 0.0 and self.config["affinity_bonus_multiplier"]
        """
        # No bonus if no performance data
        if not worker_performances or worker_id not in worker_performances:
            return 0.0
        
        worker_perf = worker_performances[worker_id]
        
        # Check for model family affinity
        affinity_score = 0.0
        if requirements.model_family and "model_family" in worker_perf:
            family_perf = worker_perf["model_family"].get(requirements.model_family)
            if family_perf and family_perf.sample_count > 5:
                # High success rate means good affinity
                affinity_score = family_perf.success_rate * 0.5
        
        # Check for test type affinity
        if requirements.test_type and "test_type" in worker_perf:
            type_perf = worker_perf["test_type"].get(requirements.test_type)
            if type_perf and type_perf.sample_count > 5:
                # High success rate means good affinity
                type_affinity = type_perf.success_rate * 0.5
                affinity_score = max(affinity_score, type_affinity)
        
        # Apply multiplier to get final bonus
        return affinity_score * self.config["affinity_bonus_multiplier"]