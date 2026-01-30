#!/usr/bin/env python3
"""
Advanced Scheduling Strategies for Distributed Testing Framework

This module implements enhanced scheduling strategies for the Distributed Testing Framework, 
building on the existing Hardware-Aware Scheduler to provide more sophisticated scheduling capabilities:

1. Historical Performance Scheduling: Leverages past execution data for optimized scheduling
2. Deadline-Aware Scheduling: Prioritizes tests based on deadlines and estimated execution times
3. Test Type-Specific Scheduling: Optimizes scheduling strategies for different test types
4. Machine Learning-Based Scheduling: Uses simple ML techniques to optimize scheduling decisions

These strategies enhance the framework's ability to efficiently distribute tests across
heterogeneous hardware environments while meeting deadlines and optimizing resource usage.
"""

import logging
import random
import math
import time
import json
import datetime
import uuid
import statistics
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Set, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import heapq
import pickle
from pathlib import Path
import threading
from datetime import datetime, timedelta

# Import hardware-aware scheduler components
from hardware_aware_scheduler import (
    HardwareAwareScheduler,
    hardware_profile_from_dict,
    workload_profile_from_dict
)

# Import hardware workload management components
from hardware_workload_management import (
    HardwareWorkloadManager, WorkloadProfile, WorkloadType, WorkloadProfileMetric,
    create_workload_profile, WorkloadExecutionPlan
)

# Import hardware taxonomy components
from enhanced_hardware_taxonomy import (
    EnhancedHardwareTaxonomy, CapabilityDefinition,
    HardwareHierarchy, HardwareRelationship
)

# Import load balancer components
from data.duckdb.distributed_testing.load_balancer.models import (
    TestRequirements, WorkerCapabilities, WorkerLoad, WorkerPerformance
)
from data.duckdb.distributed_testing.load_balancer.scheduling_algorithms import SchedulingAlgorithm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("advanced_scheduling_strategies")


# ============================================================================================
#  Historical Performance-Based Scheduler
# ============================================================================================

class PerformanceMetric(Enum):
    """Performance metrics tracked for scheduling optimization."""
    EXECUTION_TIME = "execution_time"
    THROUGHPUT = "throughput"
    SUCCESS_RATE = "success_rate"
    RESOURCE_UTILIZATION = "resource_utilization"
    ENERGY_EFFICIENCY = "energy_efficiency"
    THERMAL_EFFICIENCY = "thermal_efficiency"


@dataclass
class HistoricalPerformanceRecord:
    """Record of a test execution for performance tracking."""
    test_id: str
    worker_id: str
    hardware_id: str
    execution_time: float
    memory_usage: float
    success: bool
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class HistoricalPerformanceScheduler(HardwareAwareScheduler):
    """
    Scheduler that uses historical performance data to enhance resource-aware scheduling.
    
    This scheduler extends the hardware-aware scheduler by incorporating historical
    execution data to make more informed scheduling decisions, leading to better
    resource utilization and faster test execution.
    """
    
    def __init__(
            self, 
            hardware_workload_manager: HardwareWorkloadManager, 
            hardware_taxonomy: EnhancedHardwareTaxonomy,
            db_path: Optional[str] = None,
            performance_history_window: int = 20,
            performance_weight: float = 0.7,
            min_history_entries: int = 3
        ):
        """
        Initialize the historical performance scheduler.
        
        Args:
            hardware_workload_manager: Manager for hardware workloads
            hardware_taxonomy: Taxonomy of hardware capabilities
            db_path: Optional path to database for persistence
            performance_history_window: Number of recent executions to consider in history
            performance_weight: Weight given to historical performance in scheduling decisions
            min_history_entries: Minimum number of history entries required for performance-based decisions
        """
        super().__init__(hardware_workload_manager, hardware_taxonomy)
        
        # Performance history database
        self.db_path = db_path
        self.performance_history: Dict[str, List[HistoricalPerformanceRecord]] = {}
        
        # Historical performance configuration
        self.performance_history_window = performance_history_window
        self.performance_weight = performance_weight
        self.min_history_entries = min_history_entries
        
        # Performance prediction models
        self.execution_time_models: Dict[str, Dict[str, Any]] = {}
        
        # Scheduling optimization settings
        self.enable_performance_prediction = True
        self.enable_adaptive_weighting = True
        self.adaptive_weight_learning_rate = 0.1
        
        # Initialize performance history from database if available
        if self.db_path:
            self._load_performance_history()
    
    def select_worker(
            self, 
            test_requirements: TestRequirements, 
            available_workers: Dict[str, WorkerCapabilities],
            worker_loads: Dict[str, WorkerLoad],
            performance_data: Dict[str, Dict[str, WorkerPerformance]]
        ) -> Optional[str]:
        """
        Select the best worker based on hardware capabilities and historical performance.
        
        Args:
            test_requirements: Requirements for the test
            available_workers: Available workers with their capabilities
            worker_loads: Current load on each worker
            performance_data: Historical performance data
            
        Returns:
            ID of selected worker or None if no suitable worker found
        """
        # Use the base scheduler to get compatible hardware with efficiency scores
        workload_profile = self._test_to_workload_profile(test_requirements)
        self._update_worker_hardware_profiles(available_workers)
        
        # Use workload manager to find compatible hardware with efficiency scores
        compatible_hardware = self.workload_manager.get_compatible_hardware(workload_profile)
        
        if not compatible_hardware:
            logger.warning(f"No compatible hardware found for test {test_requirements.test_id}")
            return None
        
        # Filter by available workers and check load
        available_hardware = []
        for hardware_id, hardware_profile, efficiency in compatible_hardware:
            # Extract worker_id from hardware_id (format: worker_id_model_name)
            worker_id = hardware_id.split("_")[0]
            
            if worker_id in available_workers:
                # Check if worker has capacity
                load = worker_loads.get(worker_id)
                if load and load.has_capacity_for(test_requirements):
                    # Adjust efficiency based on current load and thermal state
                    adjusted_efficiency = self._adjust_efficiency_for_load_and_thermal(
                        efficiency, worker_id, load, hardware_profile
                    )
                    
                    # Further adjust based on historical performance if available
                    performance_adjusted_efficiency = self._adjust_for_historical_performance(
                        adjusted_efficiency, 
                        test_requirements.test_id, 
                        worker_id,
                        hardware_id,
                        hardware_profile
                    )
                    
                    available_hardware.append((worker_id, hardware_profile, performance_adjusted_efficiency))
        
        if not available_hardware:
            logger.warning(f"No available workers with capacity for test {test_requirements.test_id}")
            return None
        
        # Sort by adjusted efficiency score (highest first)
        available_hardware.sort(key=lambda x: x[2], reverse=True)
        
        # Select the worker with highest adjusted efficiency
        selected_worker_id = available_hardware[0][0]
        selected_hardware = available_hardware[0][1]
        selected_efficiency = available_hardware[0][2]
        
        logger.info(f"Selected worker {selected_worker_id} for test {test_requirements.test_id} "
                  f"with performance-adjusted efficiency {selected_efficiency:.2f}")
        
        # Update preferences for this workload type
        self._update_workload_preferences(workload_profile.workload_type.value, selected_worker_id, selected_efficiency)
        
        # Cache the workload ID for the test
        self.test_workload_cache[test_requirements.test_id] = workload_profile.workload_id
        
        return selected_worker_id
    
    def _adjust_for_historical_performance(
            self, 
            base_efficiency: float, 
            test_id: str, 
            worker_id: str,
            hardware_id: str,
            hardware_profile: Any
        ) -> float:
        """
        Adjust efficiency score based on historical performance data.
        
        Args:
            base_efficiency: Base efficiency score from hardware matching
            test_id: ID of the test being scheduled
            worker_id: ID of the worker being considered
            hardware_id: ID of the hardware being considered
            hardware_profile: Hardware capability profile
            
        Returns:
            Adjusted efficiency score
        """
        # Start with the base efficiency
        adjusted_efficiency = base_efficiency
        
        # If no historical data available, return base efficiency
        test_key = test_id
        if "_" in test_id:
            # Extract base test type from parameterized test IDs
            test_key = test_id.split("_")[0]
        
        if test_key not in self.performance_history:
            return base_efficiency
        
        # Filter records for this worker
        worker_records = [r for r in self.performance_history[test_key] 
                        if r.worker_id == worker_id]
        
        # If insufficient history, return base efficiency
        if len(worker_records) < self.min_history_entries:
            return base_efficiency
        
        # Calculate performance factor based on execution time
        # Sort by timestamp and take the most recent ones up to history window
        recent_records = sorted(worker_records, key=lambda r: r.timestamp, reverse=True)
        recent_records = recent_records[:self.performance_history_window]
        
        # Extract execution times and check success rate
        execution_times = [r.execution_time for r in recent_records if r.success]
        success_count = sum(1 for r in recent_records if r.success)
        
        if not execution_times:
            # No successful executions, penalize this worker
            return base_efficiency * 0.7
        
        # Calculate success rate
        success_rate = success_count / len(recent_records)
        
        # Calculate average execution time
        avg_execution_time = sum(execution_times) / len(execution_times)
        
        # Get execution times for all workers for this test
        all_worker_records = self.performance_history[test_key]
        all_execution_times = [r.execution_time for r in all_worker_records if r.success]
        
        if not all_execution_times:
            return base_efficiency
        
        # Calculate performance ratio relative to average across all workers
        all_avg_execution_time = sum(all_execution_times) / len(all_execution_times)
        perf_ratio = all_avg_execution_time / avg_execution_time if avg_execution_time > 0 else 1.0
        
        # Limit the performance ratio to a reasonable range (0.5 to 2.0)
        perf_ratio = max(0.5, min(2.0, perf_ratio))
        
        # Calculate combined performance factor
        # - Higher success rate is better
        # - Higher perf_ratio (faster than average) is better
        performance_factor = success_rate * perf_ratio
        
        # Apply performance factor with configured weight
        # Base efficiency gets (1 - performance_weight), performance factor gets performance_weight
        adjusted_efficiency = (1.0 - self.performance_weight) * base_efficiency + self.performance_weight * performance_factor
        
        logger.debug(f"Adjusted efficiency for test {test_id} on worker {worker_id}: "
                    f"base={base_efficiency:.2f}, perf_ratio={perf_ratio:.2f}, "
                    f"success_rate={success_rate:.2f}, adjusted={adjusted_efficiency:.2f}")
        
        return adjusted_efficiency
    
    def record_performance(self, performance_record: HistoricalPerformanceRecord) -> None:
        """
        Record performance data from a test execution.
        
        Args:
            performance_record: Record of test execution performance
        """
        test_id = performance_record.test_id
        
        # Extract base test type from parameterized test IDs
        if "_" in test_id:
            test_key = test_id.split("_")[0]
        else:
            test_key = test_id
        
        # Initialize list if needed
        if test_key not in self.performance_history:
            self.performance_history[test_key] = []
        
        # Add record to history
        self.performance_history[test_key].append(performance_record)
        
        # Limit history size if needed
        if len(self.performance_history[test_key]) > self.performance_history_window * 10:
            # Sort by timestamp and keep the most recent ones
            self.performance_history[test_key] = sorted(
                self.performance_history[test_key],
                key=lambda r: r.timestamp,
                reverse=True
            )[:self.performance_history_window * 10]
        
        # Persist to database if available
        if self.db_path:
            self._save_performance_record(performance_record)
        
        # Update prediction models
        if self.enable_performance_prediction:
            self._update_prediction_model(test_key)
        
        logger.debug(f"Recorded performance for test {test_id}: "
                   f"time={performance_record.execution_time:.2f}s, "
                   f"success={performance_record.success}")
    
    def predict_execution_time(
            self, 
            test_id: str, 
            worker_id: str, 
            hardware_id: str
        ) -> Optional[float]:
        """
        Predict execution time for a test on specific hardware.
        
        Args:
            test_id: ID of the test
            worker_id: ID of the worker
            hardware_id: ID of the hardware
            
        Returns:
            Predicted execution time in seconds, or None if prediction not possible
        """
        # Extract base test type from parameterized test IDs
        if "_" in test_id:
            test_key = test_id.split("_")[0]
        else:
            test_key = test_id
        
        # Check if we have a model for this test
        if test_key not in self.execution_time_models:
            return None
        
        # Check if we have data for this worker
        model = self.execution_time_models[test_key]
        worker_key = f"{worker_id}_{hardware_id}"
        
        if worker_key not in model['worker_averages']:
            # Fall back to average across all workers
            if 'global_average' in model:
                return model['global_average']
            return None
        
        # Return predicted execution time
        return model['worker_averages'][worker_key]
    
    def _update_prediction_model(self, test_key: str) -> None:
        """
        Update prediction model for a test based on historical data.
        
        Args:
            test_key: Base test key to update model for
        """
        if test_key not in self.performance_history:
            return
        
        # Get all successful executions
        successful_records = [r for r in self.performance_history[test_key] if r.success]
        
        if len(successful_records) < self.min_history_entries:
            return
        
        # Calculate global average execution time
        all_execution_times = [r.execution_time for r in successful_records]
        global_average = sum(all_execution_times) / len(all_execution_times)
        
        # Calculate worker-specific averages
        worker_averages = {}
        
        for record in successful_records:
            worker_key = f"{record.worker_id}_{record.hardware_id}"
            
            if worker_key not in worker_averages:
                worker_averages[worker_key] = {
                    'times': [],
                    'average': 0.0
                }
            
            worker_averages[worker_key]['times'].append(record.execution_time)
        
        # Calculate averages for each worker
        for worker_key, data in worker_averages.items():
            if len(data['times']) >= 3:  # Only if we have enough data
                data['average'] = sum(data['times']) / len(data['times'])
            else:
                # If insufficient data, use global average
                data['average'] = global_average
        
        # Create final model
        model = {
            'global_average': global_average,
            'worker_averages': {k: v['average'] for k, v in worker_averages.items()},
            'updated_at': datetime.now()
        }
        
        self.execution_time_models[test_key] = model
    
    def _load_performance_history(self) -> None:
        """Load performance history from database."""
        # This would be implemented to load from the database
        # For the prototype, we'll just initialize an empty history
        self.performance_history = defaultdict(list)
        logger.info("Performance history initialized")
    
    def _save_performance_record(self, record: HistoricalPerformanceRecord) -> None:
        """
        Save a performance record to the database.
        
        Args:
            record: Performance record to save
        """
        # This would be implemented to save to the database
        # For the prototype, we'll just log it
        logger.debug(f"Saving performance record for test {record.test_id}")


# ============================================================================================
#  Deadline-Aware Scheduler
# ============================================================================================

@dataclass
class TestDeadline:
    """Represents a deadline for a test."""
    test_id: str
    deadline: datetime
    priority: int = 3  # 1-5 (1 = highest)
    estimated_duration: Optional[float] = None
    actual_duration: Optional[float] = None
    completed: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DeadlineAwareScheduler(HistoricalPerformanceScheduler):
    """
    Scheduler that incorporates test deadlines and priorities for scheduling decisions.
    
    This scheduler extends the historical performance scheduler by considering test deadlines
    and priorities when making scheduling decisions, ensuring that time-critical tests
    are executed in time to meet their deadlines.
    """
    
    def __init__(
            self, 
            hardware_workload_manager: HardwareWorkloadManager, 
            hardware_taxonomy: EnhancedHardwareTaxonomy,
            db_path: Optional[str] = None,
            performance_history_window: int = 20,
            performance_weight: float = 0.7,
            min_history_entries: int = 3,
            deadline_weight: float = 0.8,
            urgency_threshold: int = 30  # Minutes before deadline for urgent status
        ):
        """
        Initialize the deadline-aware scheduler.
        
        Args:
            hardware_workload_manager: Manager for hardware workloads
            hardware_taxonomy: Taxonomy of hardware capabilities
            db_path: Optional path to database for persistence
            performance_history_window: Number of recent executions to consider in history
            performance_weight: Weight given to historical performance in scheduling decisions
            min_history_entries: Minimum number of history entries required for performance-based decisions
            deadline_weight: Weight given to deadline pressure in scheduling decisions
            urgency_threshold: Minutes before deadline to consider a test urgent
        """
        super().__init__(
            hardware_workload_manager, 
            hardware_taxonomy,
            db_path,
            performance_history_window,
            performance_weight,
            min_history_entries
        )
        
        # Deadline management
        self.test_deadlines: Dict[str, TestDeadline] = {}
        self.deadline_weight = deadline_weight
        self.urgency_threshold = urgency_threshold
        
        # Priority estimation
        self.enable_dynamic_priority = True
        self.priority_boost_factor = 0.5  # How much to boost priority based on urgency
        
        # Safety margin for execution time estimates
        self.execution_time_safety_margin = 1.2  # 20% safety margin
    
    def select_worker(
            self, 
            test_requirements: TestRequirements, 
            available_workers: Dict[str, WorkerCapabilities],
            worker_loads: Dict[str, WorkerLoad],
            performance_data: Dict[str, Dict[str, WorkerPerformance]]
        ) -> Optional[str]:
        """
        Select the best worker considering hardware capabilities, performance, and deadlines.
        
        Args:
            test_requirements: Requirements for the test
            available_workers: Available workers with their capabilities
            worker_loads: Current load on each worker
            performance_data: Historical performance data
            
        Returns:
            ID of selected worker or None if no suitable worker found
        """
        # Update the test's priority based on deadline
        if test_requirements.test_id in self.test_deadlines:
            self._update_test_priority(test_requirements)
        
        # Get initial worker selection from historical performance scheduler
        workload_profile = self._test_to_workload_profile(test_requirements)
        self._update_worker_hardware_profiles(available_workers)
        
        # Use workload manager to find compatible hardware with efficiency scores
        compatible_hardware = self.workload_manager.get_compatible_hardware(workload_profile)
        
        if not compatible_hardware:
            logger.warning(f"No compatible hardware found for test {test_requirements.test_id}")
            return None
        
        # Filter by available workers and check load
        available_hardware = []
        for hardware_id, hardware_profile, efficiency in compatible_hardware:
            # Extract worker_id from hardware_id (format: worker_id_model_name)
            worker_id = hardware_id.split("_")[0]
            
            if worker_id in available_workers:
                # Check if worker has capacity
                load = worker_loads.get(worker_id)
                if load and load.has_capacity_for(test_requirements):
                    # Adjust efficiency based on current load and thermal state
                    adjusted_efficiency = self._adjust_efficiency_for_load_and_thermal(
                        efficiency, worker_id, load, hardware_profile
                    )
                    
                    # Further adjust based on historical performance if available
                    performance_adjusted_efficiency = self._adjust_for_historical_performance(
                        adjusted_efficiency, 
                        test_requirements.test_id, 
                        worker_id,
                        hardware_id,
                        hardware_profile
                    )
                    
                    # Finally, adjust for deadline pressure
                    deadline_adjusted_efficiency = self._adjust_for_deadline_pressure(
                        performance_adjusted_efficiency,
                        test_requirements.test_id,
                        worker_id,
                        hardware_id
                    )
                    
                    available_hardware.append((worker_id, hardware_profile, deadline_adjusted_efficiency))
        
        if not available_hardware:
            logger.warning(f"No available workers with capacity for test {test_requirements.test_id}")
            return None
        
        # Sort by adjusted efficiency score (highest first)
        available_hardware.sort(key=lambda x: x[2], reverse=True)
        
        # Handle urgent tests by selecting fastest available worker
        if self._is_test_urgent(test_requirements.test_id):
            # Select worker with fastest estimated execution time
            fastest_worker = self._find_fastest_worker(
                test_requirements.test_id,
                available_hardware
            )
            
            if fastest_worker:
                selected_worker_id, selected_hardware, selected_efficiency = fastest_worker
                
                logger.info(f"Selected worker {selected_worker_id} for URGENT test {test_requirements.test_id} "
                          f"with deadline-adjusted efficiency {selected_efficiency:.2f}")
                
                # Update preferences for this workload type
                self._update_workload_preferences(
                    workload_profile.workload_type.value, 
                    selected_worker_id, 
                    selected_efficiency
                )
                
                # Cache the workload ID for the test
                self.test_workload_cache[test_requirements.test_id] = workload_profile.workload_id
                
                return selected_worker_id
        
        # For non-urgent tests, select the worker with highest adjusted efficiency
        selected_worker_id = available_hardware[0][0]
        selected_hardware = available_hardware[0][1]
        selected_efficiency = available_hardware[0][2]
        
        logger.info(f"Selected worker {selected_worker_id} for test {test_requirements.test_id} "
                  f"with deadline-adjusted efficiency {selected_efficiency:.2f}")
        
        # Update preferences for this workload type
        self._update_workload_preferences(
            workload_profile.workload_type.value, 
            selected_worker_id, 
            selected_efficiency
        )
        
        # Cache the workload ID for the test
        self.test_workload_cache[test_requirements.test_id] = workload_profile.workload_id
        
        return selected_worker_id
    
    def register_test_deadline(self, deadline: TestDeadline) -> None:
        """
        Register a deadline for a test.
        
        Args:
            deadline: Deadline details for the test
        """
        self.test_deadlines[deadline.test_id] = deadline
        logger.info(f"Registered deadline for test {deadline.test_id}: "
                  f"{deadline.deadline.isoformat()}, priority {deadline.priority}")
    
    def update_test_deadline_status(self, test_id: str, completed: bool, actual_duration: Optional[float] = None) -> None:
        """
        Update the status of a test with a deadline.
        
        Args:
            test_id: ID of the test
            completed: Whether the test completed successfully
            actual_duration: Actual execution duration in seconds
        """
        if test_id not in self.test_deadlines:
            return
        
        deadline = self.test_deadlines[test_id]
        deadline.completed = completed
        
        if actual_duration is not None:
            deadline.actual_duration = actual_duration
        
        logger.debug(f"Updated deadline status for test {test_id}: "
                   f"completed={completed}, duration={actual_duration}")
    
    def _update_test_priority(self, test_requirements: TestRequirements) -> None:
        """
        Update a test's priority based on its deadline.
        
        Args:
            test_requirements: Test requirements to update
        """
        if not self.enable_dynamic_priority:
            return
        
        test_id = test_requirements.test_id
        if test_id not in self.test_deadlines:
            return
        
        deadline = self.test_deadlines[test_id]
        
        # Calculate time remaining until deadline
        now = datetime.now()
        time_remaining = (deadline.deadline - now).total_seconds()
        
        if time_remaining <= 0:
            # Past deadline, set to highest priority
            test_requirements.priority = 1
            return
        
        # Calculate urgency factor (0.0 to 1.0, higher is more urgent)
        urgency_seconds = self.urgency_threshold * 60
        urgency_factor = max(0, min(1.0, 1.0 - (time_remaining / urgency_seconds)))
        
        # Only boost priority if we're within urgency threshold
        if urgency_factor > 0:
            # Calculate priority boost
            priority_boost = math.floor(self.priority_boost_factor * urgency_factor * 4)
            
            # Apply boost to base priority (ensuring we don't go below 1)
            boosted_priority = max(1, deadline.priority - priority_boost)
            
            # Update test requirements priority
            test_requirements.priority = boosted_priority
            
            if boosted_priority < deadline.priority:
                logger.info(f"Boosted priority for test {test_id} from {deadline.priority} to {boosted_priority} "
                          f"due to deadline in {time_remaining/60:.1f} minutes")
    
    def _adjust_for_deadline_pressure(
            self,
            base_efficiency: float,
            test_id: str,
            worker_id: str,
            hardware_id: str
        ) -> float:
        """
        Adjust efficiency based on deadline pressure.
        
        Args:
            base_efficiency: Base efficiency score
            test_id: ID of the test
            worker_id: ID of the worker
            hardware_id: ID of the hardware
            
        Returns:
            Adjusted efficiency score
        """
        # If no deadline, return base efficiency
        if test_id not in self.test_deadlines:
            return base_efficiency
        
        deadline = self.test_deadlines[test_id]
        
        # Calculate time remaining until deadline
        now = datetime.now()
        time_remaining = (deadline.deadline - now).total_seconds()
        
        # For past deadlines, use maximum urgency
        if time_remaining <= 0:
            urgency_factor = 1.0
        else:
            # Calculate urgency factor (0.0 to 1.0, higher is more urgent)
            urgency_seconds = self.urgency_threshold * 60
            urgency_factor = max(0, min(1.0, 1.0 - (time_remaining / urgency_seconds)))
        
        # If not urgent at all, return base efficiency
        if urgency_factor <= 0:
            return base_efficiency
        
        # Predict execution time on this worker
        predicted_time = self.predict_execution_time(test_id, worker_id, hardware_id)
        
        # If we can't predict, use estimated duration from deadline
        if predicted_time is None:
            if deadline.estimated_duration is None:
                # If no estimated duration, assume it's fine
                return base_efficiency
            predicted_time = deadline.estimated_duration
        
        # Apply safety margin
        predicted_time = predicted_time * self.execution_time_safety_margin
        
        # Check if we can complete before deadline
        can_complete_in_time = predicted_time < time_remaining
        
        if urgency_factor > 0.7:  # High urgency
            if can_complete_in_time:
                # High urgency and can complete in time - significant boost
                deadline_factor = 1.0 + urgency_factor
            else:
                # High urgency but can't complete - penalize
                deadline_factor = 0.5
        else:  # Moderate urgency
            if can_complete_in_time:
                # Moderate urgency and can complete - moderate boost
                deadline_factor = 1.0 + (0.5 * urgency_factor)
            else:
                # Moderate urgency but can't complete - slight penalty
                deadline_factor = 0.8
        
        # Apply deadline factor with configured weight
        adjusted_efficiency = (1.0 - self.deadline_weight) * base_efficiency + self.deadline_weight * base_efficiency * deadline_factor
        
        logger.debug(f"Adjusted efficiency for test {test_id} based on deadline: "
                   f"base={base_efficiency:.2f}, urgency={urgency_factor:.2f}, "
                   f"deadline_factor={deadline_factor:.2f}, adjusted={adjusted_efficiency:.2f}")
        
        return adjusted_efficiency
    
    def _is_test_urgent(self, test_id: str) -> bool:
        """
        Check if a test is urgent based on its deadline.
        
        Args:
            test_id: ID of the test
            
        Returns:
            True if the test is urgent, False otherwise
        """
        if test_id not in self.test_deadlines:
            return False
        
        deadline = self.test_deadlines[test_id]
        
        # Calculate time remaining until deadline
        now = datetime.now()
        time_remaining_minutes = (deadline.deadline - now).total_seconds() / 60
        
        # Urgent if time remaining is less than urgency threshold
        return time_remaining_minutes <= self.urgency_threshold
    
    def _find_fastest_worker(
            self,
            test_id: str,
            available_hardware: List[Tuple[str, Any, float]]
        ) -> Optional[Tuple[str, Any, float]]:
        """
        Find the worker that can execute the test fastest.
        
        Args:
            test_id: ID of the test
            available_hardware: List of (worker_id, hardware_profile, efficiency) tuples
            
        Returns:
            Tuple of (worker_id, hardware_profile, efficiency) for fastest worker,
            or None if no predictions available
        """
        fastest_worker = None
        fastest_time = float('inf')
        
        for worker_id, hardware_profile, efficiency in available_hardware:
            # Extract hardware_id from worker's hardware profile
            hardware_id = f"{worker_id}_{hardware_profile.model_name}"
            
            # Predict execution time
            predicted_time = self.predict_execution_time(test_id, worker_id, hardware_id)
            
            # If we can't predict, use efficiency as a proxy
            if predicted_time is None:
                if fastest_worker is None or efficiency > fastest_worker[2]:
                    fastest_worker = (worker_id, hardware_profile, efficiency)
            else:
                # Use actual prediction
                if predicted_time < fastest_time:
                    fastest_time = predicted_time
                    fastest_worker = (worker_id, hardware_profile, efficiency)
        
        return fastest_worker


# ============================================================================================
#  Test Type-Specific Scheduler
# ============================================================================================

class TestTypeSchedulingStrategy(Enum):
    """Scheduling strategies for different test types."""
    COMPUTE_OPTIMIZED = "compute_optimized"
    MEMORY_OPTIMIZED = "memory_optimized"
    IO_OPTIMIZED = "io_optimized"
    NETWORK_OPTIMIZED = "network_optimized"
    LATENCY_OPTIMIZED = "latency_optimized"
    THROUGHPUT_OPTIMIZED = "throughput_optimized"
    DEFAULT = "default"


@dataclass
class TestTypeConfiguration:
    """Configuration for a specific test type."""
    test_type: str
    strategy: TestTypeSchedulingStrategy
    preferred_hardware_types: List[str]
    weight_adjustments: Dict[str, float]
    scheduling_parameters: Dict[str, Any]
    description: Optional[str] = None


class TestTypeSpecificScheduler(DeadlineAwareScheduler):
    """
    Scheduler that applies specialized strategies for different test types.
    
    This scheduler extends the deadline-aware scheduler by applying different
    scheduling strategies depending on the type of test being scheduled, optimizing
    for specific test characteristics.
    """
    
    def __init__(
            self, 
            hardware_workload_manager: HardwareWorkloadManager, 
            hardware_taxonomy: EnhancedHardwareTaxonomy,
            db_path: Optional[str] = None,
            performance_history_window: int = 20,
            performance_weight: float = 0.7,
            min_history_entries: int = 3,
            deadline_weight: float = 0.8,
            urgency_threshold: int = 30
        ):
        """
        Initialize the test type-specific scheduler.
        
        Args:
            hardware_workload_manager: Manager for hardware workloads
            hardware_taxonomy: Taxonomy of hardware capabilities
            db_path: Optional path to database for persistence
            performance_history_window: Number of recent executions to consider in history
            performance_weight: Weight given to historical performance in scheduling decisions
            min_history_entries: Minimum number of history entries required for performance-based decisions
            deadline_weight: Weight given to deadline pressure in scheduling decisions
            urgency_threshold: Minutes before deadline to consider a test urgent
        """
        super().__init__(
            hardware_workload_manager, 
            hardware_taxonomy,
            db_path,
            performance_history_window,
            performance_weight,
            min_history_entries,
            deadline_weight,
            urgency_threshold
        )
        
        # Test type configurations
        self.test_type_configs: Dict[str, TestTypeConfiguration] = {}
        
        # Initialize default configurations
        self._initialize_test_type_configs()
    
    def _initialize_test_type_configs(self) -> None:
        """Initialize default configurations for different test types."""
        # Compute-intensive tests (e.g. model training, complex data processing)
        self.test_type_configs["compute_intensive"] = TestTypeConfiguration(
            test_type="compute_intensive",
            strategy=TestTypeSchedulingStrategy.COMPUTE_OPTIMIZED,
            preferred_hardware_types=["GPU", "TPU", "CPU"],
            weight_adjustments={
                "hardware_type_compatibility": 1.0,
                "compute_capability": 1.2,
                "memory_compatibility": 0.8,
                "historical_performance": 0.9,
                "deadline_pressure": 0.7
            },
            scheduling_parameters={
                "min_compute_units": 4,
                "batch_scheduling": True,
                "dynamic_scaling": True
            },
            description="Tests with high computational requirements"
        )
        
        # Memory-intensive tests (e.g. large model inference, data preprocessing)
        self.test_type_configs["memory_intensive"] = TestTypeConfiguration(
            test_type="memory_intensive",
            strategy=TestTypeSchedulingStrategy.MEMORY_OPTIMIZED,
            preferred_hardware_types=["CPU", "GPU"],
            weight_adjustments={
                "hardware_type_compatibility": 0.9,
                "compute_capability": 0.7,
                "memory_compatibility": 1.2,
                "historical_performance": 0.8,
                "deadline_pressure": 0.7
            },
            scheduling_parameters={
                "min_memory_gb": 8,
                "memory_safety_factor": 1.2,
                "avoid_memory_fragmentation": True
            },
            description="Tests with high memory requirements"
        )
        
        # I/O-intensive tests (e.g. data loading, storage tests)
        self.test_type_configs["io_intensive"] = TestTypeConfiguration(
            test_type="io_intensive",
            strategy=TestTypeSchedulingStrategy.IO_OPTIMIZED,
            preferred_hardware_types=["CPU"],
            weight_adjustments={
                "hardware_type_compatibility": 0.8,
                "compute_capability": 0.6,
                "memory_compatibility": 0.7,
                "io_capability": 1.2,
                "historical_performance": 0.9,
                "deadline_pressure": 0.7
            },
            scheduling_parameters={
                "batch_scheduling": False,
                "exclusive_execution": True,
                "io_priority": "high"
            },
            description="Tests with high I/O requirements"
        )
        
        # Network-intensive tests (e.g. distributed training, API tests)
        self.test_type_configs["network_intensive"] = TestTypeConfiguration(
            test_type="network_intensive",
            strategy=TestTypeSchedulingStrategy.NETWORK_OPTIMIZED,
            preferred_hardware_types=["CPU"],
            weight_adjustments={
                "hardware_type_compatibility": 0.8,
                "compute_capability": 0.6,
                "memory_compatibility": 0.7,
                "network_capability": 1.2,
                "historical_performance": 0.8,
                "deadline_pressure": 0.7
            },
            scheduling_parameters={
                "network_isolation": True,
                "bandwidth_reservation": True,
                "latency_sensitive": True
            },
            description="Tests with high network requirements"
        )
        
        # Latency-sensitive tests (e.g. interactive model serving)
        self.test_type_configs["latency_sensitive"] = TestTypeConfiguration(
            test_type="latency_sensitive",
            strategy=TestTypeSchedulingStrategy.LATENCY_OPTIMIZED,
            preferred_hardware_types=["GPU", "CPU"],
            weight_adjustments={
                "hardware_type_compatibility": 0.9,
                "compute_capability": 0.8,
                "memory_compatibility": 0.7,
                "historical_performance": 1.0,
                "deadline_pressure": 1.0
            },
            scheduling_parameters={
                "exclusive_execution": True,
                "preemptive_scheduling": True,
                "real_time_priority": True
            },
            description="Tests with strict latency requirements"
        )
        
        # Throughput-oriented tests (e.g. batch processing)
        self.test_type_configs["throughput_oriented"] = TestTypeConfiguration(
            test_type="throughput_oriented",
            strategy=TestTypeSchedulingStrategy.THROUGHPUT_OPTIMIZED,
            preferred_hardware_types=["GPU", "TPU", "CPU"],
            weight_adjustments={
                "hardware_type_compatibility": 0.9,
                "compute_capability": 1.0,
                "memory_compatibility": 0.8,
                "historical_performance": 0.9,
                "deadline_pressure": 0.5
            },
            scheduling_parameters={
                "batch_scheduling": True,
                "pipeline_execution": True,
                "max_concurrency": 4
            },
            description="Tests optimized for throughput rather than latency"
        )
        
        # WebGPU/WebNN-specific tests
        self.test_type_configs["webgpu_webnn"] = TestTypeConfiguration(
            test_type="webgpu_webnn",
            strategy=TestTypeSchedulingStrategy.COMPUTE_OPTIMIZED,
            preferred_hardware_types=["GPU", "CPU"],
            weight_adjustments={
                "hardware_type_compatibility": 1.0,
                "compute_capability": 0.9,
                "memory_compatibility": 0.8,
                "browser_compatibility": 1.2,
                "historical_performance": 1.0,
                "deadline_pressure": 0.7
            },
            scheduling_parameters={
                "browser_preference": "chrome",
                "webgpu_enabled": True,
                "shader_compilation_optimization": True
            },
            description="Tests for WebGPU and WebNN capabilities"
        )
        
        # Default configuration
        self.test_type_configs["default"] = TestTypeConfiguration(
            test_type="default",
            strategy=TestTypeSchedulingStrategy.DEFAULT,
            preferred_hardware_types=["CPU", "GPU"],
            weight_adjustments={
                "hardware_type_compatibility": 0.9,
                "compute_capability": 0.8,
                "memory_compatibility": 0.8,
                "historical_performance": 0.8,
                "deadline_pressure": 0.8
            },
            scheduling_parameters={
                "batch_scheduling": False,
                "dynamic_scaling": False
            },
            description="Default configuration for general tests"
        )
    
    def select_worker(
            self, 
            test_requirements: TestRequirements, 
            available_workers: Dict[str, WorkerCapabilities],
            worker_loads: Dict[str, WorkerLoad],
            performance_data: Dict[str, Dict[str, WorkerPerformance]]
        ) -> Optional[str]:
        """
        Select the best worker using test type-specific strategies.
        
        Args:
            test_requirements: Requirements for the test
            available_workers: Available workers with their capabilities
            worker_loads: Current load on each worker
            performance_data: Historical performance data
            
        Returns:
            ID of selected worker or None if no suitable worker found
        """
        # Get test type and apply specific strategy
        test_type = test_requirements.test_type
        config = self._get_test_type_config(test_type)
        
        # Apply test type specific weights
        original_weights = self.match_factor_weights.copy()
        self._apply_test_type_weights(config)
        
        # Apply test type specific hardware preferences
        original_test_type_to_hardware_type = self.test_type_to_hardware_type.copy()
        self._apply_test_type_hardware_preferences(config)
        
        try:
            # Use the base class to select a worker
            selected_worker = super().select_worker(
                test_requirements,
                available_workers,
                worker_loads,
                performance_data
            )
            
            # Implement strategy-specific post-processing
            if config.strategy == TestTypeSchedulingStrategy.COMPUTE_OPTIMIZED:
                # For compute-optimized, we might add additional checks or enhancements here
                pass
            elif config.strategy == TestTypeSchedulingStrategy.MEMORY_OPTIMIZED:
                # For memory-optimized, ensure we have enough memory with safety factor
                pass
            elif config.strategy == TestTypeSchedulingStrategy.LATENCY_OPTIMIZED:
                # For latency-optimized, we might prioritize less loaded workers
                pass
            
            return selected_worker
        finally:
            # Restore original weights and preferences
            self.match_factor_weights = original_weights
            self.test_type_to_hardware_type = original_test_type_to_hardware_type
    
    def register_test_type_config(self, config: TestTypeConfiguration) -> None:
        """
        Register a configuration for a specific test type.
        
        Args:
            config: Test type configuration
        """
        self.test_type_configs[config.test_type] = config
        logger.info(f"Registered configuration for test type {config.test_type} with strategy {config.strategy.value}")
    
    def _get_test_type_config(self, test_type: str) -> TestTypeConfiguration:
        """
        Get configuration for a specific test type.
        
        Args:
            test_type: Test type
            
        Returns:
            Test type configuration
        """
        # Try to match directly
        if test_type in self.test_type_configs:
            return self.test_type_configs[test_type]
        
        # Try to match by substring (e.g. "webgpu_inference" should match "webgpu")
        for config_type, config in self.test_type_configs.items():
            if config_type != "default" and config_type in test_type.lower():
                return config
        
        # Fall back to default
        return self.test_type_configs["default"]
    
    def _apply_test_type_weights(self, config: TestTypeConfiguration) -> None:
        """
        Apply weight adjustments for a specific test type.
        
        Args:
            config: Test type configuration
        """
        for factor, adjustment in config.weight_adjustments.items():
            if factor in self.match_factor_weights:
                self.match_factor_weights[factor] = adjustment
    
    def _apply_test_type_hardware_preferences(self, config: TestTypeConfiguration) -> None:
        """
        Apply hardware preferences for a specific test type.
        
        Args:
            config: Test type configuration
        """
        # Convert string hardware types to HardwareType enum
        hardware_types = []
        for hw_type in config.preferred_hardware_types:
            try:
                hardware_type = HardwareType[hw_type]
                hardware_types.append(hardware_type)
            except (KeyError, ValueError):
                logger.warning(f"Invalid hardware type: {hw_type}")
        
        # Update the test type to hardware type mapping for the test's type
        if hardware_types and config.test_type in self.test_type_to_hardware_type:
            self.test_type_to_hardware_type[TestType[config.test_type.upper()]] = hardware_types


# ============================================================================================
#  Machine Learning Based Scheduler
# ============================================================================================

@dataclass
class SchedulingFeature:
    """Feature used for machine learning-based scheduling."""
    name: str
    value: float
    weight: float = 1.0
    feature_type: str = "continuous"  # continuous, categorical, boolean


@dataclass
class SchedulingDecision:
    """Record of a scheduling decision for training."""
    test_id: str
    worker_id: str
    features: Dict[str, float]
    execution_time: float
    success: bool
    timestamp: datetime = field(default_factory=datetime.now)


class MLBasedScheduler(TestTypeSpecificScheduler):
    """
    Scheduler that uses machine learning techniques to optimize scheduling decisions.
    
    This scheduler extends the test type-specific scheduler by incorporating machine
    learning models to predict execution time and optimize scheduling decisions based
    on past execution data.
    """
    
    def __init__(
            self, 
            hardware_workload_manager: HardwareWorkloadManager, 
            hardware_taxonomy: EnhancedHardwareTaxonomy,
            db_path: Optional[str] = None,
            performance_history_window: int = 20,
            performance_weight: float = 0.7,
            min_history_entries: int = 3,
            deadline_weight: float = 0.8,
            urgency_threshold: int = 30,
            model_path: Optional[str] = None
        ):
        """
        Initialize the machine learning-based scheduler.
        
        Args:
            hardware_workload_manager: Manager for hardware workloads
            hardware_taxonomy: Taxonomy of hardware capabilities
            db_path: Optional path to database for persistence
            performance_history_window: Number of recent executions to consider in history
            performance_weight: Weight given to historical performance in scheduling decisions
            min_history_entries: Minimum number of history entries required for performance-based decisions
            deadline_weight: Weight given to deadline pressure in scheduling decisions
            urgency_threshold: Minutes before deadline to consider a test urgent
            model_path: Path to pre-trained model file
        """
        super().__init__(
            hardware_workload_manager, 
            hardware_taxonomy,
            db_path,
            performance_history_window,
            performance_weight,
            min_history_entries,
            deadline_weight,
            urgency_threshold
        )
        
        # ML model settings
        self.enable_ml_scheduling = True
        self.ml_weight = 0.7  # Weight given to ML predictions
        
        # Training data
        self.scheduling_decisions: List[SchedulingDecision] = []
        self.min_training_samples = 50
        self.training_interval = 100  # Train after this many new samples
        
        # Feature extraction
        self.feature_definitions: Dict[str, Dict[str, Any]] = self._initialize_feature_definitions()
        
        # Model state
        self.model = None
        self.model_version = 0
        self.last_training_time = None
        
        # Load pre-trained model if available
        if model_path and Path(model_path).exists():
            self._load_model(model_path)
    
    def select_worker(
            self, 
            test_requirements: TestRequirements, 
            available_workers: Dict[str, WorkerCapabilities],
            worker_loads: Dict[str, WorkerLoad],
            performance_data: Dict[str, Dict[str, WorkerPerformance]]
        ) -> Optional[str]:
        """
        Select the best worker using machine learning-based predictions.
        
        Args:
            test_requirements: Requirements for the test
            available_workers: Available workers with their capabilities
            worker_loads: Current load on each worker
            performance_data: Historical performance data
            
        Returns:
            ID of selected worker or None if no suitable worker found
        """
        # If ML scheduling is disabled or model not ready, fall back to base implementation
        if not self.enable_ml_scheduling or self.model is None:
            return super().select_worker(
                test_requirements,
                available_workers,
                worker_loads,
                performance_data
            )
        
        # Get initial worker selection from base scheduler
        workload_profile = self._test_to_workload_profile(test_requirements)
        self._update_worker_hardware_profiles(available_workers)
        
        # Use workload manager to find compatible hardware with efficiency scores
        compatible_hardware = self.workload_manager.get_compatible_hardware(workload_profile)
        
        if not compatible_hardware:
            logger.warning(f"No compatible hardware found for test {test_requirements.test_id}")
            return None
        
        # Filter by available workers and check load
        candidate_workers = []
        for hardware_id, hardware_profile, efficiency in compatible_hardware:
            # Extract worker_id from hardware_id (format: worker_id_model_name)
            worker_id = hardware_id.split("_")[0]
            
            if worker_id in available_workers:
                # Check if worker has capacity
                load = worker_loads.get(worker_id)
                if load and load.has_capacity_for(test_requirements):
                    # Calculate base score using standard methods
                    adjusted_efficiency = self._adjust_efficiency_for_load_and_thermal(
                        efficiency, worker_id, load, hardware_profile
                    )
                    
                    performance_adjusted_efficiency = self._adjust_for_historical_performance(
                        adjusted_efficiency, 
                        test_requirements.test_id, 
                        worker_id,
                        hardware_id,
                        hardware_profile
                    )
                    
                    deadline_adjusted_efficiency = self._adjust_for_deadline_pressure(
                        performance_adjusted_efficiency,
                        test_requirements.test_id,
                        worker_id,
                        hardware_id
                    )
                    
                    # Extract features for ML prediction
                    features = self._extract_features(
                        test_requirements,
                        worker_id,
                        hardware_id,
                        hardware_profile,
                        load,
                        deadline_adjusted_efficiency
                    )
                    
                    # Use ML model to predict execution time
                    predicted_time = self._predict_execution_time(features)
                    
                    # Calculate ML-based score (lower execution time is better)
                    if predicted_time and predicted_time > 0:
                        ml_score = 1.0 / predicted_time
                    else:
                        # Use base score if prediction failed
                        ml_score = deadline_adjusted_efficiency
                    
                    # Combine base score and ML score
                    combined_score = (1.0 - self.ml_weight) * deadline_adjusted_efficiency + self.ml_weight * ml_score
                    
                    candidate_workers.append({
                        "worker_id": worker_id,
                        "hardware_profile": hardware_profile,
                        "hardware_id": hardware_id,
                        "base_score": deadline_adjusted_efficiency,
                        "ml_score": ml_score,
                        "combined_score": combined_score,
                        "features": features,
                        "predicted_time": predicted_time
                    })
        
        if not candidate_workers:
            logger.warning(f"No available workers with capacity for test {test_requirements.test_id}")
            return None
        
        # Sort by combined score (highest first)
        candidate_workers.sort(key=lambda x: x["combined_score"], reverse=True)
        
        # Select the worker with highest combined score
        selected = candidate_workers[0]
        selected_worker_id = selected["worker_id"]
        
        logger.info(f"Selected worker {selected_worker_id} for test {test_requirements.test_id} "
                  f"with ML-based score {selected['combined_score']:.2f} "
                  f"(predicted time: {selected['predicted_time']:.2f}s)")
        
        # Update preferences for this workload type
        self._update_workload_preferences(
            workload_profile.workload_type.value, 
            selected_worker_id, 
            selected["combined_score"]
        )
        
        # Cache the workload ID for the test
        self.test_workload_cache[test_requirements.test_id] = workload_profile.workload_id
        
        # Record this decision for training
        self._record_scheduling_decision(
            test_requirements.test_id,
            selected_worker_id,
            selected["features"]
        )
        
        return selected_worker_id
    
    def record_execution_result(
            self, 
            test_id: str, 
            worker_id: str, 
            execution_time: float, 
            success: bool
        ) -> None:
        """
        Record the result of a test execution for model training.
        
        Args:
            test_id: ID of the test
            worker_id: ID of the worker that executed the test
            execution_time: Execution time in seconds
            success: Whether the execution was successful
        """
        # Update base performance tracking
        record = HistoricalPerformanceRecord(
            test_id=test_id,
            worker_id=worker_id,
            hardware_id="unknown",  # Will be inferred if possible
            execution_time=execution_time,
            memory_usage=0.0,  # Unknown at this point
            success=success
        )
        super().record_performance(record)
        
        # Update ML training data
        for decision in self.scheduling_decisions:
            if decision.test_id == test_id and decision.worker_id == worker_id:
                # Found the corresponding decision
                decision.execution_time = execution_time
                decision.success = success
                
                logger.debug(f"Updated scheduling decision for test {test_id} with execution time {execution_time:.2f}s")
                
                # Check if we should train the model
                if len(self.scheduling_decisions) >= self.min_training_samples:
                    if len(self.scheduling_decisions) % self.training_interval == 0:
                        self._train_model()
                
                break
    
    def _initialize_feature_definitions(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize feature definitions for the ML model.
        
        Returns:
            Dictionary of feature definitions
        """
        return {
            "test_type": {
                "type": "categorical",
                "encoding": "one_hot",
                "description": "Type of test being scheduled"
            },
            "hardware_type": {
                "type": "categorical",
                "encoding": "one_hot",
                "description": "Type of hardware (CPU, GPU, etc.)"
            },
            "compute_units": {
                "type": "continuous",
                "normalization": "standard",
                "description": "Number of compute units available"
            },
            "memory_gb": {
                "type": "continuous",
                "normalization": "standard",
                "description": "Amount of memory in GB"
            },
            "worker_load": {
                "type": "continuous",
                "normalization": "minmax",
                "description": "Current load on the worker (0.0-1.0)"
            },
            "base_efficiency": {
                "type": "continuous",
                "normalization": "minmax",
                "description": "Base efficiency score from hardware matching"
            },
            "deadline_pressure": {
                "type": "continuous",
                "normalization": "minmax",
                "description": "Pressure from deadline (0.0-1.0)"
            },
            "historical_success_rate": {
                "type": "continuous",
                "normalization": "minmax",
                "description": "Historical success rate for this test-worker combination"
            },
            "historical_avg_time": {
                "type": "continuous",
                "normalization": "standard",
                "description": "Historical average execution time for this test-worker combination"
            },
            "temperature": {
                "type": "continuous",
                "normalization": "minmax",
                "description": "Current thermal state of the worker"
            }
        }
    
    def _extract_features(
            self, 
            test_requirements: TestRequirements,
            worker_id: str,
            hardware_id: str,
            hardware_profile: Any,
            worker_load: WorkerLoad,
            base_efficiency: float
        ) -> Dict[str, float]:
        """
        Extract features for ML prediction.
        
        Args:
            test_requirements: Test requirements
            worker_id: ID of the worker
            hardware_id: ID of the hardware
            hardware_profile: Hardware capability profile
            worker_load: Current load on the worker
            base_efficiency: Base efficiency score from standard methods
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Test features
        features["test_type"] = self._encode_categorical(test_requirements.test_type or "unknown")
        features["test_priority"] = float(test_requirements.priority) / 5.0  # Normalize to 0.0-1.0
        features["estimated_duration"] = float(test_requirements.expected_duration or 60.0) / 3600.0  # Normalize to hours
        
        # Hardware features
        features["hardware_type"] = self._encode_categorical(hardware_profile.hardware_class.value)
        features["compute_units"] = float(hardware_profile.compute_units) / 100.0  # Normalize to reasonable range
        features["memory_gb"] = float(hardware_profile.memory.total_bytes) / (1024**3 * 32.0)  # Normalize to 32GB
        
        # Worker state
        features["worker_load"] = worker_load.calculate_load_score() if hasattr(worker_load, 'calculate_load_score') else 0.5
        
        # Get thermal state if available
        if worker_id in self.worker_thermal_states:
            features["temperature"] = self.worker_thermal_states[worker_id].get("temperature", 0.0)
        else:
            features["temperature"] = 0.0
        
        # Historical performance
        test_key = test_requirements.test_id
        if "_" in test_key:
            test_key = test_key.split("_")[0]
        
        if test_key in self.performance_history:
            worker_records = [r for r in self.performance_history[test_key] if r.worker_id == worker_id]
            
            if worker_records:
                # Success rate
                features["historical_success_rate"] = sum(1 for r in worker_records if r.success) / len(worker_records)
                
                # Average execution time
                successful_times = [r.execution_time for r in worker_records if r.success]
                if successful_times:
                    features["historical_avg_time"] = sum(successful_times) / len(successful_times) / 3600.0  # Normalize to hours
                else:
                    features["historical_avg_time"] = 0.0
            else:
                features["historical_success_rate"] = 0.5  # Default
                features["historical_avg_time"] = 0.0
        else:
            features["historical_success_rate"] = 0.5  # Default
            features["historical_avg_time"] = 0.0
        
        # Scheduling factors
        features["base_efficiency"] = base_efficiency
        
        # Deadline pressure (if available)
        if test_requirements.test_id in self.test_deadlines:
            deadline = self.test_deadlines[test_requirements.test_id]
            now = datetime.now()
            time_remaining = (deadline.deadline - now).total_seconds()
            
            if time_remaining <= 0:
                features["deadline_pressure"] = 1.0
            else:
                urgency_seconds = self.urgency_threshold * 60
                features["deadline_pressure"] = max(0, min(1.0, 1.0 - (time_remaining / urgency_seconds)))
        else:
            features["deadline_pressure"] = 0.0
        
        return features
    
    def _encode_categorical(self, value: str) -> float:
        """
        Simple encoding for categorical features.
        
        Args:
            value: Categorical value
            
        Returns:
            Encoded value as float
        """
        # This is a simplified encoding - in a real implementation,
        # we would use one-hot encoding or embedding techniques
        
        # Hash the string to get a numeric representation
        hash_val = hash(value) % 10000
        return hash_val / 10000.0  # Normalize to 0.0-1.0
    
    def _record_scheduling_decision(self, test_id: str, worker_id: str, features: Dict[str, float]) -> None:
        """
        Record a scheduling decision for future training.
        
        Args:
            test_id: ID of the test
            worker_id: ID of the selected worker
            features: Features used for the decision
        """
        # Create new decision record (execution_time and success will be updated later)
        decision = SchedulingDecision(
            test_id=test_id,
            worker_id=worker_id,
            features=features,
            execution_time=0.0,
            success=False
        )
        
        self.scheduling_decisions.append(decision)
        
        # Limit history size
        if len(self.scheduling_decisions) > 1000:
            # Keep the most recent ones
            self.scheduling_decisions = sorted(
                self.scheduling_decisions,
                key=lambda d: d.timestamp,
                reverse=True
            )[:1000]
    
    def _predict_execution_time(self, features: Dict[str, float]) -> Optional[float]:
        """
        Use ML model to predict execution time.
        
        Args:
            features: Features for prediction
            
        Returns:
            Predicted execution time in seconds, or None if prediction failed
        """
        if self.model is None:
            return None
        
        try:
            # Simple linear model prediction
            prediction = 0.0
            
            for feature_name, feature_value in features.items():
                if feature_name in self.model["weights"]:
                    prediction += feature_value * self.model["weights"][feature_name]
            
            # Add bias term
            prediction += self.model["bias"]
            
            # Ensure positive prediction
            prediction = max(1.0, prediction * 3600.0)  # Convert back to seconds
            
            return prediction
        except Exception as e:
            logger.warning(f"Error predicting execution time: {str(e)}")
            return None
    
    def _train_model(self) -> None:
        """Train the ML model using recorded scheduling decisions."""
        # Filter to completed decisions (those with execution time > 0)
        training_data = [d for d in self.scheduling_decisions if d.execution_time > 0]
        
        if len(training_data) < self.min_training_samples:
            logger.info(f"Not enough training data: {len(training_data)} samples, need {self.min_training_samples}")
            return
        
        try:
            logger.info(f"Training ML scheduler model with {len(training_data)} samples")
            
            # Simple linear regression model
            # In a real implementation, we would use more sophisticated techniques
            
            # Collect features and target values
            all_features = set()
            for decision in training_data:
                all_features.update(decision.features.keys())
            
            # Initialize weights
            weights = {feature: 0.0 for feature in all_features}
            bias = 0.0
            
            # Simple gradient descent
            learning_rate = 0.01
            epochs = 100
            
            for epoch in range(epochs):
                total_loss = 0.0
                
                for decision in training_data:
                    if not decision.success:
                        continue  # Skip failed executions
                    
                    # Predict execution time (in hours for numerical stability)
                    predicted_time = bias
                    for feature, weight in weights.items():
                        feature_value = decision.features.get(feature, 0.0)
                        predicted_time += feature_value * weight
                    
                    # Actual execution time (in hours)
                    actual_time = decision.execution_time / 3600.0
                    
                    # Error
                    error = predicted_time - actual_time
                    total_loss += error ** 2
                    
                    # Update bias
                    bias -= learning_rate * error
                    
                    # Update weights
                    for feature in weights:
                        feature_value = decision.features.get(feature, 0.0)
                        weights[feature] -= learning_rate * error * feature_value
                
                # Early stopping if loss is small enough
                if total_loss < 0.0001:
                    break
            
            # Create model
            self.model = {
                "weights": weights,
                "bias": bias,
                "version": self.model_version + 1,
                "trained_at": datetime.now(),
                "num_samples": len(training_data)
            }
            
            self.model_version += 1
            self.last_training_time = datetime.now()
            
            logger.info(f"Trained ML scheduler model version {self.model_version} with {len(training_data)} samples")
            
            # Save model if database is available
            if self.db_path:
                self._save_model()
        except Exception as e:
            logger.error(f"Error training ML model: {str(e)}")
    
    def _save_model(self) -> None:
        """Save the trained model to disk or database."""
        try:
            # For simplicity, we'll use pickle to save the model
            model_path = Path(self.db_path).parent / "ml_scheduler_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            logger.info(f"Saved ML scheduler model version {self.model_version} to {model_path}")
        except Exception as e:
            logger.error(f"Error saving ML model: {str(e)}")
    
    def _load_model(self, model_path: str) -> None:
        """
        Load a pre-trained model from disk.
        
        Args:
            model_path: Path to the model file
        """
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            self.model_version = self.model.get("version", 1)
            self.last_training_time = self.model.get("trained_at", datetime.now())
            
            logger.info(f"Loaded ML scheduler model version {self.model_version}")
        except Exception as e:
            logger.error(f"Error loading ML model: {str(e)}")
            self.model = None


# ============================================================================================
#  Advanced Scheduling Strategy Factory
# ============================================================================================

class SchedulingStrategyType(Enum):
    """Types of scheduling strategies."""
    HARDWARE_AWARE = "hardware_aware"
    HISTORICAL_PERFORMANCE = "historical_performance"
    DEADLINE_AWARE = "deadline_aware"
    TEST_TYPE_SPECIFIC = "test_type_specific"
    ML_BASED = "ml_based"


class AdvancedSchedulingStrategyFactory:
    """Factory class for creating different scheduling strategies."""
    
    @staticmethod
    def create_scheduler(
            strategy_type: SchedulingStrategyType,
            hardware_workload_manager: HardwareWorkloadManager,
            hardware_taxonomy: EnhancedHardwareTaxonomy,
            config: Dict[str, Any] = None
        ) -> SchedulingAlgorithm:
        """
        Create a scheduler of the specified type.
        
        Args:
            strategy_type: Type of scheduling strategy to create
            hardware_workload_manager: Manager for hardware workloads
            hardware_taxonomy: Taxonomy of hardware capabilities
            config: Optional configuration for the scheduler
            
        Returns:
            Scheduling algorithm implementation
        """
        if config is None:
            config = {}
        
        if strategy_type == SchedulingStrategyType.HARDWARE_AWARE:
            return HardwareAwareScheduler(hardware_workload_manager, hardware_taxonomy)
        
        elif strategy_type == SchedulingStrategyType.HISTORICAL_PERFORMANCE:
            return HistoricalPerformanceScheduler(
                hardware_workload_manager=hardware_workload_manager,
                hardware_taxonomy=hardware_taxonomy,
                db_path=config.get("db_path"),
                performance_history_window=config.get("performance_history_window", 20),
                performance_weight=config.get("performance_weight", 0.7),
                min_history_entries=config.get("min_history_entries", 3)
            )
        
        elif strategy_type == SchedulingStrategyType.DEADLINE_AWARE:
            return DeadlineAwareScheduler(
                hardware_workload_manager=hardware_workload_manager,
                hardware_taxonomy=hardware_taxonomy,
                db_path=config.get("db_path"),
                performance_history_window=config.get("performance_history_window", 20),
                performance_weight=config.get("performance_weight", 0.7),
                min_history_entries=config.get("min_history_entries", 3),
                deadline_weight=config.get("deadline_weight", 0.8),
                urgency_threshold=config.get("urgency_threshold", 30)
            )
        
        elif strategy_type == SchedulingStrategyType.TEST_TYPE_SPECIFIC:
            return TestTypeSpecificScheduler(
                hardware_workload_manager=hardware_workload_manager,
                hardware_taxonomy=hardware_taxonomy,
                db_path=config.get("db_path"),
                performance_history_window=config.get("performance_history_window", 20),
                performance_weight=config.get("performance_weight", 0.7),
                min_history_entries=config.get("min_history_entries", 3),
                deadline_weight=config.get("deadline_weight", 0.8),
                urgency_threshold=config.get("urgency_threshold", 30)
            )
        
        elif strategy_type == SchedulingStrategyType.ML_BASED:
            return MLBasedScheduler(
                hardware_workload_manager=hardware_workload_manager,
                hardware_taxonomy=hardware_taxonomy,
                db_path=config.get("db_path"),
                performance_history_window=config.get("performance_history_window", 20),
                performance_weight=config.get("performance_weight", 0.7),
                min_history_entries=config.get("min_history_entries", 3),
                deadline_weight=config.get("deadline_weight", 0.8),
                urgency_threshold=config.get("urgency_threshold", 30),
                model_path=config.get("model_path")
            )
        
        else:
            raise ValueError(f"Unknown scheduling strategy type: {strategy_type}")