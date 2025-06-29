#!/usr/bin/env python3
"""
Distributed Testing Framework - Scheduling Algorithms

This module implements various scheduling algorithms for the adaptive load balancer
in the distributed testing framework.
"""

import logging
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass

from .models import TestRequirements, WorkerCapabilities, WorkerLoad, WorkerPerformance

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("scheduling_algorithms")


class SchedulingAlgorithm(ABC):
    """Base class for scheduling algorithms."""
    
    @abstractmethod
    def select_worker(self, test_requirements: TestRequirements, 
                    available_workers: Dict[str, WorkerCapabilities],
                    worker_loads: Dict[str, WorkerLoad],
                    performance_data: Dict[str, Dict[str, WorkerPerformance]]) -> Optional[str]:
        """Select the best worker for the given test requirements.
        
        Args:
            test_requirements: Requirements for the test to schedule
            available_workers: Dict of worker_id to WorkerCapabilities
            worker_loads: Dict of worker_id to WorkerLoad
            performance_data: Performance history for workers (worker_id -> test_type -> WorkerPerformance)
            
        Returns:
            Selected worker ID, or None if no suitable worker found
        """
        pass


class RoundRobinScheduler(SchedulingAlgorithm):
    """Round-robin scheduling algorithm."""
    
    def __init__(self):
        """Initialize the round-robin scheduler."""
        self.last_worker_index = -1
        self.worker_ids = []
        
    def select_worker(self, test_requirements: TestRequirements, 
                    available_workers: Dict[str, WorkerCapabilities],
                    worker_loads: Dict[str, WorkerLoad],
                    performance_data: Dict[str, Dict[str, WorkerPerformance]]) -> Optional[str]:
        """Select the next worker in round-robin fashion."""
        # Update worker list if it changed
        current_workers = list(available_workers.keys())
        if set(current_workers) != set(self.worker_ids):
            self.worker_ids = current_workers
            self.last_worker_index = -1
            
        if not self.worker_ids:
            return None
            
        # Filter out incompatible workers
        compatible_workers = []
        for worker_id in self.worker_ids:
            if self._is_compatible(test_requirements, available_workers[worker_id], worker_loads.get(worker_id)):
                compatible_workers.append(worker_id)
                
        if not compatible_workers:
            return None
            
        # Select next worker in round-robin fashion
        self.last_worker_index = (self.last_worker_index + 1) % len(compatible_workers)
        return compatible_workers[self.last_worker_index]
        
    def _is_compatible(self, requirements: TestRequirements, 
                     capabilities: WorkerCapabilities,
                     load: Optional[WorkerLoad]) -> bool:
        """Check if worker is compatible with test requirements."""
        # Check capabilities
        if not capabilities.is_compatible_with(requirements):
            return False
            
        # Check load if available
        if load and not load.has_capacity_for(requirements):
            return False
            
        return True


class WeightedRoundRobinScheduler(SchedulingAlgorithm):
    """Weighted round-robin scheduling algorithm based on worker load."""
    
    def __init__(self):
        """Initialize the weighted round-robin scheduler."""
        self.worker_weights = {}
        
    def select_worker(self, test_requirements: TestRequirements, 
                    available_workers: Dict[str, WorkerCapabilities],
                    worker_loads: Dict[str, WorkerLoad],
                    performance_data: Dict[str, Dict[str, WorkerPerformance]]) -> Optional[str]:
        """Select worker using weighted round-robin based on current load."""
        # Filter out incompatible workers
        compatible_workers = {}
        for worker_id, capabilities in available_workers.items():
            load = worker_loads.get(worker_id)
            if self._is_compatible(test_requirements, capabilities, load):
                compatible_workers[worker_id] = capabilities
                
        if not compatible_workers:
            return None
            
        # Calculate weights based on current load
        weights = {}
        for worker_id in compatible_workers:
            load = worker_loads.get(worker_id)
            if load:
                # Invert load score to get weight (less loaded = higher weight)
                load_score = load.get_effective_load_score() if hasattr(load, 'get_effective_load_score') else load.calculate_load_score()
                weights[worker_id] = 1.0 - load_score
            else:
                # No load data, assume neutral weight
                weights[worker_id] = 0.5
                
        # Normalize weights to sum to 1.0
        total_weight = sum(weights.values())
        if total_weight > 0:
            for worker_id in weights:
                weights[worker_id] /= total_weight
                
        # Weighted random selection
        r = random.random()
        cumulative_weight = 0.0
        for worker_id, weight in weights.items():
            cumulative_weight += weight
            if r <= cumulative_weight:
                return worker_id
                
        # Fallback to last worker if something went wrong
        return list(compatible_workers.keys())[-1]
        
    def _is_compatible(self, requirements: TestRequirements, 
                     capabilities: WorkerCapabilities,
                     load: Optional[WorkerLoad]) -> bool:
        """Check if worker is compatible with test requirements."""
        # Check capabilities
        if not capabilities.is_compatible_with(requirements):
            return False
            
        # Check load if available
        if load and not load.has_capacity_for(requirements):
            return False
            
        return True


class PerformanceBasedScheduler(SchedulingAlgorithm):
    """Performance-based scheduling algorithm using historical performance data."""
    
    def select_worker(self, test_requirements: TestRequirements, 
                    available_workers: Dict[str, WorkerCapabilities],
                    worker_loads: Dict[str, WorkerLoad],
                    performance_data: Dict[str, Dict[str, WorkerPerformance]]) -> Optional[str]:
        """Select worker based on historical performance."""
        # Filter out incompatible workers
        compatible_workers = {}
        for worker_id, capabilities in available_workers.items():
            load = worker_loads.get(worker_id)
            if self._is_compatible(test_requirements, capabilities, load):
                compatible_workers[worker_id] = capabilities
                
        if not compatible_workers:
            return None
            
        # Calculate scores based on performance data
        worker_scores = {}
        for worker_id in compatible_workers:
            worker_scores[worker_id] = self._calculate_performance_score(
                worker_id, test_requirements, performance_data, worker_loads.get(worker_id)
            )
            
        # Select worker with highest score
        if worker_scores:
            return max(worker_scores.items(), key=lambda x: x[1])[0]
            
        # Fallback to random selection if no performance data
        return random.choice(list(compatible_workers.keys()))
        
    def _is_compatible(self, requirements: TestRequirements, 
                     capabilities: WorkerCapabilities,
                     load: Optional[WorkerLoad]) -> bool:
        """Check if worker is compatible with test requirements."""
        # Check capabilities
        if not capabilities.is_compatible_with(requirements):
            return False
            
        # Check load if available
        if load and not load.has_capacity_for(requirements):
            return False
            
        return True
        
    def _calculate_performance_score(self, worker_id: str, 
                                   requirements: TestRequirements,
                                   performance_data: Dict[str, Dict[str, WorkerPerformance]],
                                   load: Optional[WorkerLoad]) -> float:
        """Calculate performance score for a worker based on historical data."""
        # Base score
        score = 0.5
        
        # Get performance data for this test type
        test_type = requirements.test_type or "default"
        worker_perf = performance_data.get(worker_id, {})
        perf = worker_perf.get(test_type)
        
        if perf:
            # Higher success rate = higher score (30% weight)
            success_score = perf.success_rate  # 0.0 to 1.0
            
            # Lower execution time = higher score (30% weight)
            # Normalize against expected duration
            time_ratio = perf.average_execution_time / max(1.0, requirements.expected_duration)
            time_score = max(0.0, min(1.0, 1.0 - (time_ratio - 1.0) / 3.0))  # Scale within reasonable bounds
            
            # More experience = higher score (10% weight)
            experience_score = min(1.0, perf.sample_count / 10.0)  # Scale up to 10 samples
            
            # Calculate weighted score from performance data (70% of total)
            perf_score = 0.3 * success_score + 0.3 * time_score + 0.1 * experience_score
            score = perf_score
        
        # Adjust for current load (30% of total)
        if load:
            # Lower load = higher score
            effective_load = load.get_effective_load_score() if hasattr(load, 'get_effective_load_score') else load.calculate_load_score()
            load_score = 1.0 - effective_load
            score = 0.7 * score + 0.3 * load_score
            
        return score


class PriorityBasedScheduler(SchedulingAlgorithm):
    """Priority-based scheduling algorithm that considers test priority."""
    
    def select_worker(self, test_requirements: TestRequirements, 
                    available_workers: Dict[str, WorkerCapabilities],
                    worker_loads: Dict[str, WorkerLoad],
                    performance_data: Dict[str, Dict[str, WorkerPerformance]]) -> Optional[str]:
        """Select worker based on test priority and worker capabilities."""
        # Filter out incompatible workers
        compatible_workers = {}
        for worker_id, capabilities in available_workers.items():
            load = worker_loads.get(worker_id)
            if self._is_compatible(test_requirements, capabilities, load):
                compatible_workers[worker_id] = capabilities
                
        if not compatible_workers:
            return None
            
        # For high-priority tests, select the fastest worker
        if test_requirements.priority <= 2:  # High priority (1-2)
            # Calculate scores emphasizing speed
            worker_scores = {}
            for worker_id in compatible_workers:
                worker_scores[worker_id] = self._calculate_speed_score(
                    worker_id, test_requirements, performance_data, worker_loads.get(worker_id)
                )
                
            # Select worker with highest score
            if worker_scores:
                return max(worker_scores.items(), key=lambda x: x[1])[0]
        
        # For low-priority tests, use weighted round-robin
        elif test_requirements.priority >= 4:  # Low priority (4-5)
            # Calculate weights based on current load
            weights = {}
            for worker_id in compatible_workers:
                load = worker_loads.get(worker_id)
                if load:
                    # Invert load score to get weight (less loaded = higher weight)
                    effective_load = load.get_effective_load_score() if hasattr(load, 'get_effective_load_score') else load.calculate_load_score()
                    weights[worker_id] = 1.0 - effective_load
                else:
                    # No load data, assume neutral weight
                    weights[worker_id] = 0.5
                    
            # Normalize weights to sum to 1.0
            total_weight = sum(weights.values())
            if total_weight > 0:
                for worker_id in weights:
                    weights[worker_id] /= total_weight
                    
            # Weighted random selection
            r = random.random()
            cumulative_weight = 0.0
            for worker_id, weight in weights.items():
                cumulative_weight += weight
                if r <= cumulative_weight:
                    return worker_id
        
        # For medium-priority tests, balanced approach
        else:  # Medium priority (3)
            # Calculate balanced scores
            worker_scores = {}
            for worker_id in compatible_workers:
                worker_scores[worker_id] = self._calculate_balanced_score(
                    worker_id, test_requirements, performance_data, worker_loads.get(worker_id)
                )
                
            # Select worker with highest score
            if worker_scores:
                return max(worker_scores.items(), key=lambda x: x[1])[0]
                
        # Fallback to random selection
        return random.choice(list(compatible_workers.keys()))
        
    def _is_compatible(self, requirements: TestRequirements, 
                     capabilities: WorkerCapabilities,
                     load: Optional[WorkerLoad]) -> bool:
        """Check if worker is compatible with test requirements."""
        # Check capabilities
        if not capabilities.is_compatible_with(requirements):
            return False
            
        # Check load if available
        if load and not load.has_capacity_for(requirements):
            return False
            
        return True
        
    def _calculate_speed_score(self, worker_id: str, 
                             requirements: TestRequirements,
                             performance_data: Dict[str, Dict[str, WorkerPerformance]],
                             load: Optional[WorkerLoad]) -> float:
        """Calculate score emphasizing speed for high-priority tests."""
        # Base score
        score = 0.5
        
        # Get performance data for this test type
        test_type = requirements.test_type or "default"
        worker_perf = performance_data.get(worker_id, {})
        perf = worker_perf.get(test_type)
        
        if perf:
            # Higher success rate = higher score (20% weight)
            success_score = perf.success_rate  # 0.0 to 1.0
            
            # Lower execution time = higher score (60% weight for high-priority)
            # Normalize against expected duration
            time_ratio = perf.average_execution_time / max(1.0, requirements.expected_duration)
            time_score = max(0.0, min(1.0, 1.0 - (time_ratio - 1.0) / 3.0))  # Scale within reasonable bounds
            
            # Calculate weighted score from performance data (80% of total)
            perf_score = 0.2 * success_score + 0.6 * time_score
            score = perf_score
        
        # Adjust for current load (20% of total)
        if load:
            # Lower load = higher score
            effective_load = load.get_effective_load_score() if hasattr(load, 'get_effective_load_score') else load.calculate_load_score()
            load_score = 1.0 - effective_load
            score = 0.8 * score + 0.2 * load_score
            
        return score
        
    def _calculate_balanced_score(self, worker_id: str, 
                                requirements: TestRequirements,
                                performance_data: Dict[str, Dict[str, WorkerPerformance]],
                                load: Optional[WorkerLoad]) -> float:
        """Calculate balanced score for medium-priority tests."""
        # Base score
        score = 0.5
        
        # Get performance data for this test type
        test_type = requirements.test_type or "default"
        worker_perf = performance_data.get(worker_id, {})
        perf = worker_perf.get(test_type)
        
        if perf:
            # Higher success rate = higher score (30% weight)
            success_score = perf.success_rate  # 0.0 to 1.0
            
            # Lower execution time = higher score (30% weight)
            # Normalize against expected duration
            time_ratio = perf.average_execution_time / max(1.0, requirements.expected_duration)
            time_score = max(0.0, min(1.0, 1.0 - (time_ratio - 1.0) / 3.0))  # Scale within reasonable bounds
            
            # Calculate weighted score from performance data (60% of total)
            perf_score = 0.3 * success_score + 0.3 * time_score
            score = perf_score
        
        # Adjust for current load (40% of total)
        if load:
            # Lower load = higher score
            effective_load = load.get_effective_load_score() if hasattr(load, 'get_effective_load_score') else load.calculate_load_score()
            load_score = 1.0 - effective_load
            score = 0.6 * score + 0.4 * load_score
            
        return score


class CompositeScheduler(SchedulingAlgorithm):
    """Composite scheduling algorithm that combines multiple algorithms."""
    
    def __init__(self, algorithms: List[Tuple[SchedulingAlgorithm, float]]):
        """Initialize the composite scheduler.
        
        Args:
            algorithms: List of (algorithm, weight) tuples
        """
        self.algorithms = algorithms
        
    def select_worker(self, test_requirements: TestRequirements, 
                    available_workers: Dict[str, WorkerCapabilities],
                    worker_loads: Dict[str, WorkerLoad],
                    performance_data: Dict[str, Dict[str, WorkerPerformance]]) -> Optional[str]:
        """Select worker by combining results from multiple algorithms."""
        if not available_workers:
            return None
            
        # Get worker scores from each algorithm
        worker_scores = {worker_id: 0.0 for worker_id in available_workers}
        
        for algorithm, weight in self.algorithms:
            selected = algorithm.select_worker(
                test_requirements, available_workers, worker_loads, performance_data
            )
            
            if selected:
                # Give points to selected worker
                worker_scores[selected] += weight
                
        # Select worker with highest score
        if worker_scores:
            return max(worker_scores.items(), key=lambda x: x[1])[0]
            
        return None


class AffinityBasedScheduler(SchedulingAlgorithm):
    """Affinity-based scheduler that tries to assign similar tests to the same worker."""
    
    def __init__(self):
        """Initialize the affinity-based scheduler."""
        self.model_affinities = {}  # model_id -> worker_id
        self.family_affinities = {}  # model_family -> worker_id
        
    def select_worker(self, test_requirements: TestRequirements, 
                    available_workers: Dict[str, WorkerCapabilities],
                    worker_loads: Dict[str, WorkerLoad],
                    performance_data: Dict[str, Dict[str, WorkerPerformance]]) -> Optional[str]:
        """Select worker based on test affinities."""
        # Filter out incompatible workers
        compatible_workers = {}
        for worker_id, capabilities in available_workers.items():
            load = worker_loads.get(worker_id)
            if self._is_compatible(test_requirements, capabilities, load):
                compatible_workers[worker_id] = capabilities
                
        if not compatible_workers:
            return None
            
        # Check if we have affinity for this model
        if test_requirements.model_id and test_requirements.model_id in self.model_affinities:
            affinity_worker = self.model_affinities[test_requirements.model_id]
            if affinity_worker in compatible_workers:
                return affinity_worker
                
        # Check if we have affinity for this model family
        if test_requirements.model_family and test_requirements.model_family in self.family_affinities:
            affinity_worker = self.family_affinities[test_requirements.model_family]
            if affinity_worker in compatible_workers:
                return affinity_worker
                
        # No affinity or affinity worker not available, select based on load
        lowest_load = float('inf')
        selected_worker = None
        
        for worker_id in compatible_workers:
            load = worker_loads.get(worker_id)
            if load:
                effective_load = load.get_effective_load_score() if hasattr(load, 'get_effective_load_score') else load.calculate_load_score()
                if effective_load < lowest_load:
                    lowest_load = effective_load
                    selected_worker = worker_id
                    
        if selected_worker:
            # Update affinities
            if test_requirements.model_id:
                self.model_affinities[test_requirements.model_id] = selected_worker
            if test_requirements.model_family:
                self.family_affinities[test_requirements.model_family] = selected_worker
                
            return selected_worker
                
        # Fallback to random selection
        selected_worker = random.choice(list(compatible_workers.keys()))
        
        # Update affinities
        if test_requirements.model_id:
            self.model_affinities[test_requirements.model_id] = selected_worker
        if test_requirements.model_family:
            self.family_affinities[test_requirements.model_family] = selected_worker
            
        return selected_worker
        
    def _is_compatible(self, requirements: TestRequirements, 
                     capabilities: WorkerCapabilities,
                     load: Optional[WorkerLoad]) -> bool:
        """Check if worker is compatible with test requirements."""
        # Check capabilities
        if not capabilities.is_compatible_with(requirements):
            return False
            
        # Check load if available
        if load and not load.has_capacity_for(requirements):
            return False
            
        return True


class AdaptiveScheduler(SchedulingAlgorithm):
    """Adaptive scheduler that selects algorithm based on workload characteristics."""
    
    def __init__(self):
        """Initialize the adaptive scheduler with sub-algorithms."""
        self.round_robin = RoundRobinScheduler()
        self.weighted_round_robin = WeightedRoundRobinScheduler()
        self.performance_based = PerformanceBasedScheduler()
        self.priority_based = PriorityBasedScheduler()
        self.affinity_based = AffinityBasedScheduler()
        
        # State tracking
        self.high_load_threshold = 0.8
        self.low_load_threshold = 0.3
        self.high_priority_threshold = 2
        
    def select_worker(self, test_requirements: TestRequirements, 
                    available_workers: Dict[str, WorkerCapabilities],
                    worker_loads: Dict[str, WorkerLoad],
                    performance_data: Dict[str, Dict[str, WorkerPerformance]]) -> Optional[str]:
        """Adaptively select scheduling algorithm based on current conditions."""
        if not available_workers:
            return None
            
        # Calculate current system-wide load
        system_load = self._calculate_system_load(worker_loads)
        
        # High load condition
        if system_load > self.high_load_threshold:
            # High priority test under high load
            if test_requirements.priority <= self.high_priority_threshold:
                # Use performance-based scheduling for high priority tests
                return self.performance_based.select_worker(
                    test_requirements, available_workers, worker_loads, performance_data
                )
            else:
                # Use weighted round-robin for normal priority tests
                return self.weighted_round_robin.select_worker(
                    test_requirements, available_workers, worker_loads, performance_data
                )
                
        # Low load condition
        elif system_load < self.low_load_threshold:
            # Under low load, use affinity-based scheduling
            return self.affinity_based.select_worker(
                test_requirements, available_workers, worker_loads, performance_data
            )
            
        # Medium load, use priority-based scheduling
        else:
            return self.priority_based.select_worker(
                test_requirements, available_workers, worker_loads, performance_data
            )
            
    def _calculate_system_load(self, worker_loads: Dict[str, WorkerLoad]) -> float:
        """Calculate system-wide load average."""
        if not worker_loads:
            return 0.0
            
        load_sum = sum(load.get_effective_load_score() if hasattr(load, 'get_effective_load_score') 
                      else load.calculate_load_score() 
                      for load in worker_loads.values())
        return load_sum / len(worker_loads)