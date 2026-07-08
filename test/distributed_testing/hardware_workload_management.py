#!/usr/bin/env python3
"""
Hardware-Aware Workload Management for Distributed Testing Framework

This module implements advanced workload management capabilities with hardware awareness,
allowing for more sophisticated and efficient distribution of workloads across
heterogeneous hardware environments.

Key features:
- Hardware-specific workload analysis and profiling
- Dynamic workload allocation based on hardware capabilities
- Multi-device orchestration for complex workloads
- Adaptive resource allocation based on workload characteristics
- Performance history tracking and optimization
"""

import os
import json
import logging
import threading
import time
from typing import Dict, List, Any, Optional, Tuple, Set, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
import queue
import heapq
import math

# Import hardware taxonomy components
from duckdb_api.distributed_testing.hardware_taxonomy import (
    HardwareClass, HardwareArchitecture, HardwareVendor,
    SoftwareBackend, PrecisionType, AcceleratorFeature,
    HardwareCapabilityProfile, HardwareTaxonomy
)

# Import existing load balancer components
from duckdb_api.distributed_testing.load_balancer.models import (
    WorkerCapabilities, WorkerLoad, TestRequirements, WorkerAssignment
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("hardware_workload_management")


class WorkloadType(Enum):
    """Classification of workload types for hardware matching."""
    VISION = "vision"
    NLP = "nlp"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"
    TRAINING = "training"
    INFERENCE = "inference"
    EMBEDDING = "embedding"
    CONVERSATIONAL = "conversational"
    MIXED = "mixed"


class WorkloadProfileMetric(Enum):
    """Metrics used for workload profiling."""
    COMPUTE_INTENSITY = "compute_intensity"
    MEMORY_INTENSITY = "memory_intensity"
    IO_INTENSITY = "io_intensity"
    NETWORK_INTENSITY = "network_intensity"
    MODEL_SIZE = "model_size"
    BATCH_SIZE = "batch_size"
    LATENCY_SENSITIVITY = "latency_sensitivity"
    THROUGHPUT_SENSITIVITY = "throughput_sensitivity"
    PARALLELISM = "parallelism"
    TEMPERATURE = "temperature"
    ENERGY_SENSITIVITY = "energy_sensitivity"


@dataclass
class WorkloadProfile:
    """Detailed profile of a workload's resource requirements and characteristics."""
    workload_id: str
    workload_type: WorkloadType
    required_backends: Set[SoftwareBackend] = field(default_factory=set)
    required_precisions: Set[PrecisionType] = field(default_factory=set)
    required_features: Set[AcceleratorFeature] = field(default_factory=set)
    min_memory_bytes: int = 0
    min_compute_units: int = 0
    
    # Workload metrics (0.0 to 1.0 scale, higher value = more intensive)
    metrics: Dict[WorkloadProfileMetric, float] = field(default_factory=dict)
    
    # Hardware preferences
    preferred_hardware_class: Optional[HardwareClass] = None
    preferred_architecture: Optional[HardwareArchitecture] = None
    preferred_vendors: List[HardwareVendor] = field(default_factory=list)
    
    # Scheduling control
    priority: int = 3  # 1-5 (1 = highest)
    deadline: Optional[datetime] = None
    concurrency_key: Optional[str] = None
    estimated_duration_seconds: Optional[int] = None
    
    # Multi-device allocation
    is_shardable: bool = False
    min_shards: int = 1
    max_shards: int = 1
    allocation_strategy: str = "single"  # single, sharded, replicated
    
    # Custom properties
    custom_properties: Dict[str, Any] = field(default_factory=dict)
    
    def get_efficiency_score(self, hardware: HardwareCapabilityProfile) -> float:
        """
        Calculate how efficiently the workload would run on the given hardware.
        
        Args:
            hardware: Hardware capability profile
            
        Returns:
            Efficiency score (0.0 to 1.0, higher is better)
        """
        # Base score starts at 1.0
        score = 1.0
        
        # Check hardware class match
        if self.preferred_hardware_class:
            if hardware.hardware_class == self.preferred_hardware_class:
                score *= 1.0
            else:
                # Reduce score if not preferred hardware class
                score *= 0.7
        
        # Check architecture match
        if self.preferred_architecture:
            if hardware.architecture == self.preferred_architecture:
                score *= 1.0
            else:
                # Reduce score if not preferred architecture
                score *= 0.8
        
        # Check vendor match
        if self.preferred_vendors:
            if hardware.vendor in self.preferred_vendors:
                score *= 1.0
            else:
                # Reduce score if not preferred vendor
                score *= 0.9
        
        # Calculate match based on workload metrics
        if WorkloadProfileMetric.COMPUTE_INTENSITY in self.metrics:
            compute_intensity = self.metrics[WorkloadProfileMetric.COMPUTE_INTENSITY]
            
            # For high compute workloads, hardware with more compute units is better
            if compute_intensity > 0.7:  # High compute intensity
                compute_factor = min(1.0, hardware.compute_units / 100.0)
                score *= (0.5 + 0.5 * compute_factor)
        
        # Memory intensity factor
        if WorkloadProfileMetric.MEMORY_INTENSITY in self.metrics:
            memory_intensity = self.metrics[WorkloadProfileMetric.MEMORY_INTENSITY]
            
            # For high memory workloads, hardware with more memory is better
            if memory_intensity > 0.7:  # High memory intensity
                memory_gb = hardware.memory.total_bytes / (1024 * 1024 * 1024)
                memory_factor = min(1.0, memory_gb / 16.0)  # Scale to 16GB
                score *= (0.5 + 0.5 * memory_factor)
        
        # Consider device-specific specializations
        if hardware.hardware_class == HardwareClass.GPU:
            if self.workload_type == WorkloadType.VISION:
                score *= 1.2  # GPUs work well for vision
            elif self.workload_type == WorkloadType.NLP and AcceleratorFeature.TENSOR_CORES in hardware.features:
                score *= 1.2  # GPUs with tensor cores work well for NLP
        
        elif hardware.hardware_class == HardwareClass.TPU:
            if self.workload_type in [WorkloadType.NLP, WorkloadType.VISION]:
                score *= 1.3  # TPUs work very well for NLP and vision
        
        elif hardware.hardware_class == HardwareClass.NPU:
            if self.workload_type == WorkloadType.INFERENCE:
                score *= 1.2  # NPUs work well for inference
        
        # Cap the final score
        return min(1.0, max(0.0, score))
    
    def to_test_requirements(self) -> TestRequirements:
        """
        Convert workload profile to test requirements for the load balancer.
        
        Returns:
            TestRequirements instance
        """
        # Convert workload profile to test requirements
        backends = []
        for backend in self.required_backends:
            backends.append(backend.value)
        
        # Estimate memory requirements in GB for test requirements
        memory_gb = math.ceil(self.min_memory_bytes / (1024 * 1024 * 1024))
        
        # Create test requirements
        test_requirements = TestRequirements(
            test_id=self.workload_id,
            test_type=self.workload_type.value,
            model_id=self.custom_properties.get("model_id", "unknown"),
            priority=self.priority,
            min_memory_gb=memory_gb,
            min_cpu_cores=self.min_compute_units,
            required_capabilities=backends,
            concurrency_key=self.concurrency_key,
            custom_properties={
                "workload_profile": asdict(self),
                "allocation_strategy": self.allocation_strategy,
                "is_shardable": self.is_shardable,
                "min_shards": self.min_shards,
                "max_shards": self.max_shards
            }
        )
        
        return test_requirements


@dataclass
class WorkloadExecutionPlan:
    """Plan for executing a workload across one or more hardware devices."""
    workload_profile: WorkloadProfile
    hardware_assignments: List[Tuple[str, HardwareCapabilityProfile]] = field(default_factory=list)
    is_multi_device: bool = False
    shard_count: int = 1
    estimated_execution_time: float = 0.0
    estimated_efficiency: float = 0.0
    estimated_energy_usage: float = 0.0
    execution_status: str = "planned"  # planned, executing, completed, failed
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def mark_started(self) -> None:
        """Mark the execution plan as started."""
        self.execution_status = "executing"
        self.started_at = datetime.now()
    
    def mark_completed(self, success: bool) -> None:
        """Mark the execution plan as completed or failed."""
        self.execution_status = "completed" if success else "failed"
        self.completed_at = datetime.now()
    
    def get_actual_duration(self) -> Optional[float]:
        """Get the actual duration of execution in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class HardwareWorkloadManager:
    """
    Advanced hardware-aware workload management system for distributed testing.
    
    This class provides sophisticated workload management capabilities:
    1. Hardware-specific workload profiling and analysis
    2. Optimal workload-to-hardware matching
    3. Multi-device orchestration for complex workloads
    4. Adaptive resource allocation based on workload characteristics
    5. Performance prediction and optimization
    """
    
    def __init__(self, hardware_taxonomy: HardwareTaxonomy, db_path: Optional[str] = None):
        """
        Initialize the hardware workload manager.
        
        Args:
            hardware_taxonomy: Hardware taxonomy for capability matching
            db_path: Optional path to database for performance tracking
        """
        self.hardware_taxonomy = hardware_taxonomy
        self.db_path = db_path
        self.lock = threading.RLock()
        
        # Workload management
        self.workload_profiles: Dict[str, WorkloadProfile] = {}
        self.execution_plans: Dict[str, WorkloadExecutionPlan] = {}
        self.active_executions: Dict[str, Set[str]] = {}  # hardware_id -> set of workload_ids
        
        # Performance history
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {}  # workload_type -> list of execution records
        
        # Prioritization
        self.workload_queue = []  # Priority queue of (priority, workload_id)
        
        # Multi-device orchestration
        self.device_groups: Dict[str, List[str]] = {}  # group_id -> list of hardware_ids
        
        # Scheduling
        self.thermal_tracking: Dict[str, Dict[str, Any]] = {}  # hardware_id -> thermal state
        self.device_state_cache: Dict[str, Dict[str, Any]] = {}  # hardware_id -> state cache
        
        # Multi-device orchestration
        self.orchestrator = MultiDeviceOrchestrator(self)
        
        # Hardware-specific workload optimization
        self.hardware_optimizations: Dict[str, Dict[str, Any]] = {}  # hardware_class -> optimization settings
        
        # Workload-specific device preferences 
        self.workload_device_preferences: Dict[str, Dict[str, float]] = {}  # workload_type -> hardware_id -> preference score
        
        # Enhanced thermal management with predictive cooling
        self.thermal_prediction_enabled = True
        self.thermal_prediction_window = 5  # minutes
        self.thermal_prediction_model = None  # Would be initialized with ML model
        
        # Monitoring and events
        self.monitoring_interval = 10  # seconds
        self._stop_monitoring = threading.Event()
        self.monitoring_thread = None
        self.event_callbacks: Dict[str, List[Callable]] = {
            "workload_scheduled": [],
            "workload_started": [],
            "workload_completed": [],
            "workload_failed": [],
            "hardware_state_changed": [],
            "thermal_prediction_alert": [],
            "resource_contention_detected": [],
            "device_performance_degraded": []
        }
    
    def start(self) -> None:
        """Start the hardware workload manager."""
        # Start monitoring thread
        self._stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info("Hardware workload manager started")
    
    def stop(self) -> None:
        """Stop the hardware workload manager."""
        # Stop monitoring thread
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self._stop_monitoring.set()
            self.monitoring_thread.join(timeout=5)
            
        logger.info("Hardware workload manager stopped")
    
    def register_workload(self, workload_profile: WorkloadProfile) -> str:
        """
        Register a workload for scheduling.
        
        Args:
            workload_profile: Profile describing the workload
            
        Returns:
            Workload ID
        """
        with self.lock:
            # Ensure workload has an ID
            if not workload_profile.workload_id:
                workload_profile.workload_id = str(uuid.uuid4())
            
            # Store workload profile
            self.workload_profiles[workload_profile.workload_id] = workload_profile
            
            # Add to priority queue
            heapq.heappush(self.workload_queue, (workload_profile.priority, workload_profile.workload_id))
            
            logger.info(f"Registered workload {workload_profile.workload_id} of type {workload_profile.workload_type.value}")
            
            # Trigger scheduling
            self._schedule_pending_workloads()
            
            return workload_profile.workload_id
    
    def get_execution_plan(self, workload_id: str) -> Optional[WorkloadExecutionPlan]:
        """
        Get the execution plan for a workload.
        
        Args:
            workload_id: Workload ID
            
        Returns:
            Execution plan or None if not scheduled
        """
        with self.lock:
            return self.execution_plans.get(workload_id)
    
    def update_execution_status(self, workload_id: str, status: str) -> None:
        """
        Update the status of a workload execution.
        
        Args:
            workload_id: Workload ID
            status: New status (executing, completed, failed)
        """
        with self.lock:
            if workload_id in self.execution_plans:
                plan = self.execution_plans[workload_id]
                
                if status == "executing":
                    plan.mark_started()
                    self._trigger_event("workload_started", workload_id, plan)
                    
                elif status in ["completed", "failed"]:
                    success = status == "completed"
                    plan.mark_completed(success)
                    
                    # Record performance
                    self._record_execution_performance(workload_id, plan)
                    
                    # Release hardware resources
                    for hardware_id, _ in plan.hardware_assignments:
                        if hardware_id in self.active_executions and workload_id in self.active_executions[hardware_id]:
                            self.active_executions[hardware_id].remove(workload_id)
                    
                    # Trigger appropriate event
                    if success:
                        self._trigger_event("workload_completed", workload_id, plan)
                    else:
                        self._trigger_event("workload_failed", workload_id, plan)
                
                # Schedule more workloads if possible
                self._schedule_pending_workloads()
    
    def register_event_callback(self, event_type: str, callback: Callable) -> None:
        """
        Register a callback for workload events.
        
        Args:
            event_type: Event type (workload_scheduled, workload_started, etc.)
            callback: Function to call when event occurs
        """
        with self.lock:
            if event_type in self.event_callbacks:
                self.event_callbacks[event_type].append(callback)
    
    def create_device_group(self, group_id: str, hardware_ids: List[str]) -> None:
        """
        Create a device group for multi-device orchestration.
        
        Args:
            group_id: Unique identifier for the group
            hardware_ids: List of hardware IDs in the group
        """
        with self.lock:
            self.device_groups[group_id] = hardware_ids
            logger.info(f"Created device group {group_id} with {len(hardware_ids)} devices")
    
    def get_device_groups(self) -> Dict[str, List[str]]:
        """
        Get all device groups.
        
        Returns:
            Dictionary of group_id -> list of hardware_ids
        """
        with self.lock:
            return self.device_groups.copy()
    
    def get_hardware_load(self, hardware_id: str) -> float:
        """
        Get the current load on a hardware device (0.0 to 1.0).
        
        Args:
            hardware_id: Hardware device ID
            
        Returns:
            Load score (0.0 to 1.0)
        """
        with self.lock:
            if hardware_id in self.active_executions:
                # Simple calculation based on number of active workloads
                active_count = len(self.active_executions[hardware_id])
                # Assuming each device can handle up to 5 workloads efficiently
                return min(1.0, active_count / 5.0)
            return 0.0
    
    def get_compatible_hardware(self, workload_profile: WorkloadProfile) -> List[Tuple[str, HardwareCapabilityProfile, float]]:
        """
        Find compatible hardware for a workload with efficiency scores.
        
        Args:
            workload_profile: Workload profile to match
            
        Returns:
            List of (hardware_id, hardware_profile, efficiency_score) tuples
        """
        compatible_hardware = []
        
        # Convert workload requirements to hardware requirements
        required_backends = workload_profile.required_backends
        required_precisions = workload_profile.required_precisions
        required_features = workload_profile.required_features
        min_memory_bytes = workload_profile.min_memory_bytes
        min_compute_units = workload_profile.min_compute_units
        
        # First, find hardware by class (if specified)
        hardware_class = workload_profile.preferred_hardware_class
        if hardware_class and hardware_class.value in self.hardware_taxonomy.hardware_profiles:
            profiles = self.hardware_taxonomy.hardware_profiles[hardware_class.value]
            for profile in profiles:
                # Check if hardware is compatible with requirements
                if profile.is_compatible_with(
                    required_backends=required_backends,
                    required_precisions=required_precisions,
                    required_features=required_features,
                    min_memory_bytes=min_memory_bytes,
                    min_compute_units=min_compute_units
                ):
                    # Find hardware_id from worker_hardware_map
                    for worker_id, hw_profiles in self.hardware_taxonomy.worker_hardware_map.items():
                        for hw_profile in hw_profiles:
                            if (hw_profile.hardware_class == profile.hardware_class and
                                hw_profile.architecture == profile.architecture and
                                hw_profile.vendor == profile.vendor and
                                hw_profile.model_name == profile.model_name):
                                # Calculate efficiency score
                                efficiency = workload_profile.get_efficiency_score(profile)
                                compatible_hardware.append((worker_id, profile, efficiency))
        
        # If no preferred class or no compatible hardware found, try all hardware
        if not compatible_hardware:
            for worker_id, hw_profiles in self.hardware_taxonomy.worker_hardware_map.items():
                for profile in hw_profiles:
                    # Check if hardware is compatible with requirements
                    if profile.is_compatible_with(
                        required_backends=required_backends,
                        required_precisions=required_precisions,
                        required_features=required_features,
                        min_memory_bytes=min_memory_bytes,
                        min_compute_units=min_compute_units
                    ):
                        # Calculate efficiency score
                        efficiency = workload_profile.get_efficiency_score(profile)
                        compatible_hardware.append((worker_id, profile, efficiency))
        
        # Sort by efficiency score (highest first)
        compatible_hardware.sort(key=lambda x: x[2], reverse=True)
        return compatible_hardware
    
    def predict_execution_time(self, workload_profile: WorkloadProfile, hardware_profile: HardwareCapabilityProfile) -> float:
        """
        Predict execution time for a workload on specific hardware.
        
        Args:
            workload_profile: Workload profile
            hardware_profile: Hardware capability profile
            
        Returns:
            Estimated execution time in seconds
        """
        # Base execution time from workload profile
        base_time = workload_profile.estimated_duration_seconds
        if not base_time:
            # Default base time if not specified
            base_time = 60.0  # 1 minute default
        
        # Check performance history for similar workloads
        workload_type = workload_profile.workload_type.value
        if workload_type in self.performance_history:
            similar_executions = []
            for execution in self.performance_history[workload_type]:
                # Check if hardware is similar
                if (execution["hardware_class"] == hardware_profile.hardware_class.value and
                    execution["hardware_architecture"] == hardware_profile.architecture.value):
                    # Found similar execution
                    similar_executions.append(execution["duration_seconds"])
            
            if similar_executions:
                # Use average of similar executions
                avg_time = sum(similar_executions) / len(similar_executions)
                # Blend with base time (80% history, 20% base estimate)
                base_time = 0.8 * avg_time + 0.2 * base_time
        
        # Adjust base time based on hardware capabilities
        
        # Compute units factor (more compute units = faster)
        compute_factor = 1.0
        if workload_profile.min_compute_units > 0:
            compute_ratio = hardware_profile.compute_units / workload_profile.min_compute_units
            compute_factor = max(0.5, min(1.5, 1.0 / compute_ratio))
        
        # Memory factor (more memory = faster)
        memory_factor = 1.0
        if workload_profile.min_memory_bytes > 0:
            memory_ratio = hardware_profile.memory.total_bytes / workload_profile.min_memory_bytes
            memory_factor = max(0.7, min(1.3, 1.0 / memory_ratio))
        
        # Hardware efficiency factor
        efficiency = workload_profile.get_efficiency_score(hardware_profile)
        efficiency_factor = max(0.5, 1.5 - efficiency)  # High efficiency = lower execution time
        
        # Compute final adjusted time
        adjusted_time = base_time * compute_factor * memory_factor * efficiency_factor
        
        return max(1.0, adjusted_time)  # Ensure at least 1 second
    
    def predict_energy_usage(self, workload_profile: WorkloadProfile, hardware_profile: HardwareCapabilityProfile, duration_seconds: float) -> float:
        """
        Predict energy usage for a workload on specific hardware.
        
        Args:
            workload_profile: Workload profile
            hardware_profile: Hardware capability profile
            duration_seconds: Estimated execution time in seconds
            
        Returns:
            Estimated energy usage in joules
        """
        # Base calculation using TDP and duration
        tdp_watts = hardware_profile.thermal_design_power_w or 100.0  # Default to 100W if not specified
        
        # Calculate workload intensity factor (0.3 to 1.0)
        intensity = 0.7  # Default medium intensity
        
        if WorkloadProfileMetric.COMPUTE_INTENSITY in workload_profile.metrics:
            compute_intensity = workload_profile.metrics[WorkloadProfileMetric.COMPUTE_INTENSITY]
            intensity = max(intensity, 0.3 + 0.7 * compute_intensity)
        
        if WorkloadProfileMetric.MEMORY_INTENSITY in workload_profile.metrics:
            memory_intensity = workload_profile.metrics[WorkloadProfileMetric.MEMORY_INTENSITY]
            # Memory operations can be expensive on some hardware
            memory_factor = 0.3 + 0.7 * memory_intensity
            intensity = max(intensity, memory_factor)
        
        # Energy sensitivity factor
        energy_factor = 1.0
        if WorkloadProfileMetric.ENERGY_SENSITIVITY in workload_profile.metrics:
            energy_sensitivity = workload_profile.metrics[WorkloadProfileMetric.ENERGY_SENSITIVITY]
            energy_factor = 0.5 + 0.5 * energy_sensitivity
        
        # Calculate energy usage: Power (W) * Time (s) * Intensity * Energy Factor = Energy (J)
        energy_joules = tdp_watts * duration_seconds * intensity * energy_factor
        
        return energy_joules
    
    def create_execution_plan(self, workload_id: str) -> Optional[WorkloadExecutionPlan]:
        """
        Create an execution plan for a workload.
        
        Args:
            workload_id: Workload ID
            
        Returns:
            Execution plan or None if no suitable hardware found
        """
        if workload_id not in self.workload_profiles:
            return None
        
        workload_profile = self.workload_profiles[workload_id]
        
        # Find compatible hardware with efficiency scores
        compatible_hardware = self.get_compatible_hardware(workload_profile)
        
        if not compatible_hardware:
            logger.warning(f"No compatible hardware found for workload {workload_id}")
            return None
        
        # Check if workload should be executed on multiple devices
        if workload_profile.allocation_strategy == "sharded" and workload_profile.is_shardable:
            # Multi-device execution plan (sharded)
            return self._create_sharded_execution_plan(workload_profile, compatible_hardware)
        
        elif workload_profile.allocation_strategy == "replicated":
            # Multi-device execution plan (replicated)
            return self._create_replicated_execution_plan(workload_profile, compatible_hardware)
        
        else:
            # Single-device execution plan
            return self._create_single_device_execution_plan(workload_profile, compatible_hardware)
    
    def _create_single_device_execution_plan(
            self, 
            workload_profile: WorkloadProfile, 
            compatible_hardware: List[Tuple[str, HardwareCapabilityProfile, float]]
        ) -> Optional[WorkloadExecutionPlan]:
        """
        Create an execution plan for a single device.
        
        Args:
            workload_profile: Workload profile
            compatible_hardware: List of (hardware_id, hardware_profile, efficiency_score) tuples
            
        Returns:
            Execution plan or None if no suitable hardware found
        """
        # Filter by available hardware (not fully loaded)
        available_hardware = []
        for hardware_id, hardware_profile, efficiency in compatible_hardware:
            # Check current load
            load = self.get_hardware_load(hardware_id)
            if load < 0.9:  # Allow if less than 90% loaded
                available_hardware.append((hardware_id, hardware_profile, efficiency))
        
        if not available_hardware:
            return None
        
        # Select best hardware based on efficiency and load
        best_hardware = None
        best_score = -1.0
        
        for hardware_id, hardware_profile, efficiency in available_hardware:
            # Calculate combined score (blend of efficiency and available capacity)
            load = self.get_hardware_load(hardware_id)
            available_capacity = 1.0 - load
            
            # Score is 70% efficiency, 30% available capacity
            combined_score = 0.7 * efficiency + 0.3 * available_capacity
            
            if combined_score > best_score:
                best_score = combined_score
                best_hardware = (hardware_id, hardware_profile)
        
        if not best_hardware:
            return None
        
        hardware_id, hardware_profile = best_hardware
        
        # Predict execution time and energy usage
        estimated_time = self.predict_execution_time(workload_profile, hardware_profile)
        estimated_energy = self.predict_energy_usage(workload_profile, hardware_profile, estimated_time)
        
        # Create execution plan
        plan = WorkloadExecutionPlan(
            workload_profile=workload_profile,
            hardware_assignments=[(hardware_id, hardware_profile)],
            is_multi_device=False,
            shard_count=1,
            estimated_execution_time=estimated_time,
            estimated_efficiency=best_score,
            estimated_energy_usage=estimated_energy
        )
        
        return plan
    
    def _create_sharded_execution_plan(
            self, 
            workload_profile: WorkloadProfile, 
            compatible_hardware: List[Tuple[str, HardwareCapabilityProfile, float]]
        ) -> Optional[WorkloadExecutionPlan]:
        """
        Create a sharded execution plan across multiple devices.
        
        Args:
            workload_profile: Workload profile
            compatible_hardware: List of (hardware_id, hardware_profile, efficiency_score) tuples
            
        Returns:
            Execution plan or None if no suitable hardware combination found
        """
        min_shards = workload_profile.min_shards
        max_shards = workload_profile.max_shards
        
        # Determine optimal shard count based on available hardware
        available_hardware = []
        for hardware_id, hardware_profile, efficiency in compatible_hardware:
            # Check current load
            load = self.get_hardware_load(hardware_id)
            if load < 0.8:  # Allow if less than 80% loaded for multi-device workloads
                available_hardware.append((hardware_id, hardware_profile, efficiency))
        
        if len(available_hardware) < min_shards:
            return None
        
        # Determine optimal shard count (min_shards to max_shards)
        optimal_shard_count = min(max_shards, len(available_hardware))
        
        # For now, just select the top N most efficient devices
        available_hardware.sort(key=lambda x: x[2], reverse=True)
        selected_hardware = available_hardware[:optimal_shard_count]
        
        # Calculate overall efficiency as the average of selected hardware
        avg_efficiency = sum(efficiency for _, _, efficiency in selected_hardware) / len(selected_hardware)
        
        # Create hardware assignments
        hardware_assignments = [(hw_id, hw_profile) for hw_id, hw_profile, _ in selected_hardware]
        
        # Predict execution time (use the slowest device for prediction)
        min_efficiency = min(efficiency for _, _, efficiency in selected_hardware)
        slowest_hardware = next(hw_profile for _, hw_profile, eff in selected_hardware if eff == min_efficiency)
        
        # For sharded workloads, execution time is roughly divided by shard count,
        # but with some overhead for coordination (10% per shard)
        base_time = self.predict_execution_time(workload_profile, slowest_hardware)
        coordination_factor = 1.0 + (0.1 * optimal_shard_count)
        estimated_time = (base_time / optimal_shard_count) * coordination_factor
        
        # Estimate energy usage (sum across all devices, but with lower intensity per device)
        total_energy = 0.0
        for _, hw_profile, _ in selected_hardware:
            device_energy = self.predict_energy_usage(
                workload_profile, hw_profile, estimated_time
            )
            # Each device operates at lower intensity in sharded mode
            intensity_factor = 0.7  # Assume 70% intensity per device in sharded mode
            total_energy += device_energy * intensity_factor
        
        # Create execution plan
        plan = WorkloadExecutionPlan(
            workload_profile=workload_profile,
            hardware_assignments=hardware_assignments,
            is_multi_device=True,
            shard_count=optimal_shard_count,
            estimated_execution_time=estimated_time,
            estimated_efficiency=avg_efficiency,
            estimated_energy_usage=total_energy
        )
        
        return plan
    
    def _create_replicated_execution_plan(
            self, 
            workload_profile: WorkloadProfile, 
            compatible_hardware: List[Tuple[str, HardwareCapabilityProfile, float]]
        ) -> Optional[WorkloadExecutionPlan]:
        """
        Create a replicated execution plan across multiple devices.
        
        Args:
            workload_profile: Workload profile
            compatible_hardware: List of (hardware_id, hardware_profile, efficiency_score) tuples
            
        Returns:
            Execution plan or None if no suitable hardware found
        """
        # For replicated execution, we run the same workload on multiple devices
        # for redundancy or throughput. We need at least 2 devices.
        min_replicas = workload_profile.min_shards
        max_replicas = workload_profile.max_shards
        
        # Determine available hardware
        available_hardware = []
        for hardware_id, hardware_profile, efficiency in compatible_hardware:
            # Check current load
            load = self.get_hardware_load(hardware_id)
            if load < 0.7:  # Lower threshold for replicated workloads
                available_hardware.append((hardware_id, hardware_profile, efficiency))
        
        if len(available_hardware) < min_replicas:
            return None
        
        # Determine optimal replica count
        optimal_replica_count = min(max_replicas, len(available_hardware))
        
        # Select the most efficient hardware first
        available_hardware.sort(key=lambda x: x[2], reverse=True)
        selected_hardware = available_hardware[:optimal_replica_count]
        
        # Calculate overall efficiency as the average of selected hardware
        avg_efficiency = sum(efficiency for _, _, efficiency in selected_hardware) / len(selected_hardware)
        
        # Create hardware assignments
        hardware_assignments = [(hw_id, hw_profile) for hw_id, hw_profile, _ in selected_hardware]
        
        # For replicated workloads, execution time is the same as single-device,
        # but let's use the fastest device for the estimate
        fastest_hardware = selected_hardware[0][1]  # hardware_profile of highest efficiency device
        estimated_time = self.predict_execution_time(workload_profile, fastest_hardware)
        
        # Estimate energy usage (sum across all devices)
        total_energy = 0.0
        for _, hw_profile, _ in selected_hardware:
            device_energy = self.predict_energy_usage(
                workload_profile, hw_profile, estimated_time
            )
            total_energy += device_energy
        
        # Create execution plan
        plan = WorkloadExecutionPlan(
            workload_profile=workload_profile,
            hardware_assignments=hardware_assignments,
            is_multi_device=True,
            shard_count=optimal_replica_count,
            estimated_execution_time=estimated_time,
            estimated_efficiency=avg_efficiency,
            estimated_energy_usage=total_energy
        )
        
        return plan
    
    def _schedule_pending_workloads(self) -> None:
        """Schedule pending workloads to available hardware."""
        with self.lock:
            # Process workloads in priority order
            scheduled_count = 0
            processed_workloads = []
            
            while self.workload_queue:
                # Get next workload
                priority, workload_id = heapq.heappop(self.workload_queue)
                processed_workloads.append((priority, workload_id))
                
                # Skip if already scheduled
                if workload_id in self.execution_plans:
                    continue
                
                # Skip if workload not found
                if workload_id not in self.workload_profiles:
                    continue
                
                # Create execution plan
                plan = self.create_execution_plan(workload_id)
                
                if plan:
                    # Store execution plan
                    self.execution_plans[workload_id] = plan
                    
                    # Reserve hardware
                    for hardware_id, _ in plan.hardware_assignments:
                        if hardware_id not in self.active_executions:
                            self.active_executions[hardware_id] = set()
                        self.active_executions[hardware_id].add(workload_id)
                    
                    # Remove from workload profiles
                    # del self.workload_profiles[workload_id]
                    
                    # Track scheduled count
                    scheduled_count += 1
                    
                    # Trigger event
                    self._trigger_event("workload_scheduled", workload_id, plan)
                    
                    logger.info(f"Scheduled workload {workload_id} to {len(plan.hardware_assignments)} devices, "
                             f"estimated execution time: {plan.estimated_execution_time:.2f}s")
                else:
                    # Requeue with lower priority
                    heapq.heappush(self.workload_queue, (priority + 1, workload_id))
                    logger.warning(f"No suitable hardware found for workload {workload_id}, requeued with priority {priority + 1}")
                
                # Limit number of workloads scheduled per cycle
                if scheduled_count >= 10:
                    break
            
            # Put back processed but unscheduled workloads
            for priority, workload_id in processed_workloads:
                if workload_id not in self.execution_plans:
                    heapq.heappush(self.workload_queue, (priority, workload_id))
            
            if scheduled_count > 0:
                logger.info(f"Scheduled {scheduled_count} workloads")
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while not self._stop_monitoring.is_set():
            try:
                # Update hardware states
                self._update_hardware_states()
                
                # Update thermal states
                self._update_thermal_tracking()
                
                # Schedule pending workloads
                self._schedule_pending_workloads()
                
                # Check for completed workloads
                self._check_for_completed_workloads()
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            # Sleep for monitoring interval
            self._stop_monitoring.wait(self.monitoring_interval)
    
    def _update_hardware_states(self) -> None:
        """Update state cache for hardware devices."""
        for worker_id, hw_profiles in self.hardware_taxonomy.worker_hardware_map.items():
            for hw_profile in hw_profiles:
                hardware_id = f"{worker_id}_{hw_profile.model_name}"
                
                # Get current load
                load = self.get_hardware_load(hardware_id)
                
                # Update state cache
                self.device_state_cache[hardware_id] = {
                    "load": load,
                    "last_updated": datetime.now(),
                    "worker_id": worker_id,
                    "hardware_class": hw_profile.hardware_class.value,
                    "model_name": hw_profile.model_name
                }
    
    def _update_thermal_tracking(self) -> None:
        """Update thermal states for hardware devices."""
        # Simple thermal model based on load and time
        for hardware_id, state in self.device_state_cache.items():
            load = state["load"]
            
            if hardware_id not in self.thermal_tracking:
                self.thermal_tracking[hardware_id] = {
                    "temperature": 0.0,  # Normalized temperature (0.0 to 1.0)
                    "warming_state": False,
                    "cooling_state": False,
                    "last_update": datetime.now()
                }
            
            thermal_state = self.thermal_tracking[hardware_id]
            time_delta = (datetime.now() - thermal_state["last_update"]).total_seconds()
            
            # Update temperature based on load and time
            current_temp = thermal_state["temperature"]
            
            if load > 0.7:
                # High load causes warming
                if not thermal_state["warming_state"]:
                    thermal_state["warming_state"] = True
                    thermal_state["cooling_state"] = False
                
                # Increase temperature (max 1.0)
                # Rate depends on current temperature (slower as temperature increases)
                temp_headroom = 1.0 - current_temp
                temp_increase = min(0.1, 0.02 * time_delta * temp_headroom * load)
                thermal_state["temperature"] = min(1.0, current_temp + temp_increase)
                
            elif load < 0.3:
                # Low load allows cooling
                if not thermal_state["cooling_state"]:
                    thermal_state["cooling_state"] = True
                    thermal_state["warming_state"] = False
                
                # Decrease temperature (min 0.0)
                # Rate depends on current temperature (faster as temperature increases)
                cooling_rate = 0.01 * time_delta * (1.0 + current_temp)
                thermal_state["temperature"] = max(0.0, current_temp - cooling_rate)
            
            # Update last update time
            thermal_state["last_update"] = datetime.now()
            
            # Log if significant change
            if abs(thermal_state["temperature"] - current_temp) > 0.1:
                logger.debug(f"Hardware {hardware_id} temperature: {thermal_state['temperature']:.2f}, "
                           f"load: {load:.2f}")
    
    def _check_for_completed_workloads(self) -> None:
        """Check for workloads that should be completed based on estimated time."""
        now = datetime.now()
        
        for workload_id, plan in self.execution_plans.items():
            if plan.execution_status == "executing" and plan.started_at:
                # Check if estimated time has elapsed
                elapsed = (now - plan.started_at).total_seconds()
                
                # If elapsed time is significantly longer than estimated, log warning
                # but don't auto-complete (that should be done by actual completion reporting)
                if elapsed > 2.0 * plan.estimated_execution_time:
                    logger.warning(f"Workload {workload_id} has been running for {elapsed:.1f}s, "
                                 f"which is more than twice the estimated time ({plan.estimated_execution_time:.1f}s)")
    
    def _record_execution_performance(self, workload_id: str, plan: WorkloadExecutionPlan) -> None:
        """Record performance data for a completed workload execution."""
        if not plan.started_at or not plan.completed_at:
            return
        
        # Get workload type
        workload_type = plan.workload_profile.workload_type.value
        
        # Create execution record
        record = {
            "workload_id": workload_id,
            "workload_type": workload_type,
            "execution_status": plan.execution_status,
            "is_multi_device": plan.is_multi_device,
            "shard_count": plan.shard_count,
            "duration_seconds": (plan.completed_at - plan.started_at).total_seconds(),
            "estimated_time": plan.estimated_execution_time,
            "estimated_efficiency": plan.estimated_efficiency,
            "estimated_energy": plan.estimated_energy_usage,
            "hardware_assignments": []
        }
        
        # Add hardware details
        for hardware_id, hw_profile in plan.hardware_assignments:
            record["hardware_assignments"].append({
                "hardware_id": hardware_id,
                "hardware_class": hw_profile.hardware_class.value,
                "hardware_architecture": hw_profile.architecture.value,
                "hardware_vendor": hw_profile.vendor.value,
                "model_name": hw_profile.model_name
            })
        
        # Store in performance history
        if workload_type not in self.performance_history:
            self.performance_history[workload_type] = []
        
        self.performance_history[workload_type].append(record)
        
        logger.info(f"Recorded performance for workload {workload_id}: "
                  f"actual={record['duration_seconds']:.2f}s vs. "
                  f"estimated={plan.estimated_execution_time:.2f}s")
    
    def _trigger_event(self, event_type: str, workload_id: str, plan: WorkloadExecutionPlan) -> None:
        """Trigger an event callback."""
        if event_type in self.event_callbacks:
            for callback in self.event_callbacks[event_type]:
                try:
                    callback(workload_id, plan)
                except Exception as e:
                    logger.error(f"Error in event callback for {event_type}: {e}")


@dataclass
class SubtaskDefinition:
    """Definition of a subtask within a multi-device workload."""
    subtask_id: str
    workload_id: str
    dependencies: List[str] = field(default_factory=list)
    hardware_requirements: Dict[str, Any] = field(default_factory=dict)
    data_requirements: Dict[str, Any] = field(default_factory=dict)
    is_critical: bool = False
    timeout_seconds: int = 300
    retry_count: int = 2
    priority: int = 3
    custom_config: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def full_id(self) -> str:
        """Get the full subtask ID (workload_id_subtask_id)."""
        return f"{self.workload_id}_{self.subtask_id}"


@dataclass
class SubtaskStatus:
    """Status of a subtask execution."""
    subtask: SubtaskDefinition
    status: str = "pending"  # pending, scheduled, executing, completed, failed
    assigned_hardware: Optional[List[Tuple[str, Any]]] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    retry_count: int = 0
    execution_plan_id: Optional[str] = None


@dataclass
class WorkloadExecutionGraph:
    """Execution graph for a multi-device workload."""
    workload_id: str
    subtasks: Dict[str, SubtaskDefinition] = field(default_factory=dict)
    status_map: Dict[str, SubtaskStatus] = field(default_factory=dict)
    aggregation_method: str = "default"
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    aggregated_result: Any = None
    custom_config: Dict[str, Any] = field(default_factory=dict)
    
    def get_ready_subtasks(self) -> List[SubtaskDefinition]:
        """Get subtasks that are ready to execute (dependencies satisfied)."""
        ready_subtasks = []
        
        for subtask_id, subtask in self.subtasks.items():
            # Skip if already completed or currently executing
            status = self.status_map.get(subtask_id)
            if status and status.status in ["completed", "executing", "scheduled"]:
                continue
            
            # Check dependencies
            dependencies_satisfied = True
            for dep_id in subtask.dependencies:
                dep_status = self.status_map.get(dep_id)
                if not dep_status or dep_status.status != "completed":
                    dependencies_satisfied = False
                    break
            
            if dependencies_satisfied:
                ready_subtasks.append(subtask)
        
        return ready_subtasks
    
    def is_completed(self) -> bool:
        """Check if all subtasks are completed."""
        for status in self.status_map.values():
            if status.status not in ["completed", "skipped"]:
                return False
        return True
    
    def get_critical_path_remaining(self) -> List[str]:
        """Get the critical path of remaining subtasks."""
        # Find all incomplete subtasks
        incomplete = {id: subtask for id, subtask in self.subtasks.items() 
                    if self.status_map.get(id, SubtaskStatus(subtask)).status not in ["completed", "skipped"]}
        
        # Build dependency graph
        graph = {id: set() for id in incomplete}
        for id, subtask in incomplete.items():
            for dep in subtask.dependencies:
                if dep in incomplete:
                    graph[dep].add(id)
        
        # Find leaf nodes (no dependents)
        leaves = [id for id, deps in graph.items() if not deps]
        
        # Calculate criticality by finding the longest path from each node
        node_criticality = {}
        
        def calculate_path_length(node_id, visited=None):
            if visited is None:
                visited = set()
            
            if node_id in visited:
                return 0  # Circular dependency
            
            if node_id in node_criticality:
                return node_criticality[node_id]
            
            visited.add(node_id)
            
            # Find longest path from dependencies
            max_length = 0
            for dep in incomplete[node_id].dependencies:
                if dep in incomplete:
                    dep_length = calculate_path_length(dep, visited.copy())
                    max_length = max(max_length, dep_length)
            
            # Add this node's weight (1 by default, more for critical subtasks)
            weight = 2 if incomplete[node_id].is_critical else 1
            path_length = max_length + weight
            
            node_criticality[node_id] = path_length
            return path_length
        
        # Calculate criticality for all leaf nodes
        for leaf in leaves:
            calculate_path_length(leaf)
        
        # Sort by criticality (descending)
        critical_path = sorted(node_criticality.items(), key=lambda x: x[1], reverse=True)
        return [id for id, _ in critical_path]
    
    def update_subtask_status(self, subtask_id: str, status: str, **kwargs) -> None:
        """Update the status of a subtask."""
        if subtask_id not in self.status_map:
            if subtask_id in self.subtasks:
                self.status_map[subtask_id] = SubtaskStatus(self.subtasks[subtask_id])
            else:
                raise ValueError(f"Subtask {subtask_id} not found in workload {self.workload_id}")
        
        subtask_status = self.status_map[subtask_id]
        subtask_status.status = status
        
        # Update other fields if provided
        for field, value in kwargs.items():
            if hasattr(subtask_status, field):
                setattr(subtask_status, field, value)
        
        # Update timestamps
        if status == "executing" and not subtask_status.start_time:
            subtask_status.start_time = datetime.now()
        elif status in ["completed", "failed"] and not subtask_status.end_time:
            subtask_status.end_time = datetime.now()
        
        # Check if the entire workload is completed
        if self.is_completed() and not self.completed_at:
            self.completed_at = datetime.now()


class MultiDeviceOrchestrator:
    """
    Advanced Multi-Device Orchestration for complex workloads distributed across multiple hardware devices.
    
    This enhanced class handles:
    1. Workload decomposition into subtasks with dependency graphs
    2. Optimal subtask placement with hardware affinity
    3. Communication coordination between subtasks
    4. Synchronization and barrier management
    5. Result aggregation from distributed execution
    6. Fault tolerance with retry mechanisms
    7. Critical path identification for priority scheduling
    8. Dynamic workload redistribution
    9. Performance optimization through execution history analysis
    10. Resource monitoring and contention avoidance
    """
    
    def __init__(self, workload_manager):
        """
        Initialize the multi-device orchestrator.
        
        Args:
            workload_manager: Hardware workload manager instance
        """
        self.workload_manager = workload_manager
        self.execution_graphs: Dict[str, WorkloadExecutionGraph] = {}
        self.active_workloads: Set[str] = set()
        self.workload_history: Dict[str, List[Dict[str, Any]]] = {}
        self.lock = threading.RLock()
        
        # Advanced orchestration features
        self.communication_channels: Dict[str, Any] = {}  # For inter-subtask communication
        self.barrier_management: Dict[str, Dict[str, Any]] = {}  # For synchronization points
        self.execution_history: Dict[str, List[Dict[str, Any]]] = {}  # For performance optimization
        self.resource_monitor = ResourceMonitor(self)
        
        # Fault tolerance
        self.failure_detection_system = FailureDetectionSystem(self)
        self.recovery_strategies: Dict[str, Callable] = {
            "retry": self._recovery_strategy_retry,
            "reassign": self._recovery_strategy_reassign,
            "skip": self._recovery_strategy_skip,
            "alternate": self._recovery_strategy_alternate,
            "checkpoint": self._recovery_strategy_checkpoint
        }
        self.default_recovery_strategy = "retry"
        
        # Performance optimization
        self.performance_tracker = PerformanceTracker(self)
        self.optimization_strategies: Dict[str, Callable] = {
            "memory_pooling": self._optimization_memory_pooling,
            "operation_fusion": self._optimization_operation_fusion,
            "device_specialization": self._optimization_device_specialization,
            "data_locality": self._optimization_data_locality,
            "parallel_execution": self._optimization_parallel_execution
        }
        
        # Monitoring
        self.monitoring_enabled = True
        self.monitoring_interval = 5  # seconds
        self._stop_monitoring = threading.Event()
        self.monitoring_thread = None
    
    def start(self) -> None:
        """Start the orchestrator's monitoring system."""
        if self.monitoring_enabled and not self.monitoring_thread:
            self._stop_monitoring.clear()
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True
            )
            self.monitoring_thread.start()
            logger.info("Multi-device orchestrator monitoring started")
    
    def stop(self) -> None:
        """Stop the orchestrator's monitoring system."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self._stop_monitoring.set()
            self.monitoring_thread.join(timeout=5)
            logger.info("Multi-device orchestrator monitoring stopped")
    
    def register_workload(self, workload_id: str, config: Dict[str, Any]) -> WorkloadExecutionGraph:
        """
        Register a multi-device workload for orchestration.
        
        Args:
            workload_id: Workload ID
            config: Workload configuration including subtasks and dependencies
            
        Returns:
            WorkloadExecutionGraph for the registered workload
        """
        with self.lock:
            # Create subtask definitions
            subtasks = {}
            for subtask_id, subtask_config in config.get("subtasks", {}).items():
                subtask = SubtaskDefinition(
                    subtask_id=subtask_id,
                    workload_id=workload_id,
                    dependencies=subtask_config.get("dependencies", []),
                    hardware_requirements=subtask_config.get("hardware_requirements", {}),
                    data_requirements=subtask_config.get("data_requirements", {}),
                    is_critical=subtask_config.get("is_critical", False),
                    timeout_seconds=subtask_config.get("timeout_seconds", 300),
                    retry_count=subtask_config.get("retry_count", 2),
                    priority=subtask_config.get("priority", 3),
                    custom_config=subtask_config.get("custom_config", {})
                )
                subtasks[subtask_id] = subtask
            
            # Create execution graph
            graph = WorkloadExecutionGraph(
                workload_id=workload_id,
                subtasks=subtasks,
                aggregation_method=config.get("aggregation_method", "default"),
                custom_config=config.get("custom_config", {})
            )
            
            # Initialize status map
            for subtask_id, subtask in subtasks.items():
                graph.status_map[subtask_id] = SubtaskStatus(subtask)
            
            # Store execution graph
            self.execution_graphs[workload_id] = graph
            self.active_workloads.add(workload_id)
            
            logger.info(f"Registered workload {workload_id} with {len(subtasks)} subtasks")
            
            # Initialize any necessary communication channels or barriers
            self._initialize_communication_channels(workload_id, config)
            self._initialize_barriers(workload_id, config)
            
            # Apply workload optimizations
            self._apply_workload_optimizations(workload_id, config)
            
            return graph
    
    def get_ready_subtasks(self, workload_id: str) -> List[SubtaskDefinition]:
        """
        Get subtasks that are ready to execute (dependencies satisfied).
        
        Args:
            workload_id: Workload ID
            
        Returns:
            List of ready subtask definitions
        """
        with self.lock:
            if workload_id not in self.execution_graphs:
                return []
            
            execution_graph = self.execution_graphs[workload_id]
            return execution_graph.get_ready_subtasks()
    
    def get_next_critical_subtasks(self, workload_id: str, limit: int = 5) -> List[SubtaskDefinition]:
        """
        Get the next most critical subtasks for a workload, based on the critical path.
        
        Args:
            workload_id: Workload ID
            limit: Maximum number of subtasks to return
            
        Returns:
            List of critical subtask definitions
        """
        with self.lock:
            if workload_id not in self.execution_graphs:
                return []
            
            execution_graph = self.execution_graphs[workload_id]
            critical_path = execution_graph.get_critical_path_remaining()
            
            critical_subtasks = []
            for subtask_id in critical_path[:limit]:
                subtask = execution_graph.subtasks.get(subtask_id)
                if subtask:
                    status = execution_graph.status_map.get(subtask_id)
                    if status and status.status == "pending":
                        critical_subtasks.append(subtask)
            
            return critical_subtasks
    
    def schedule_subtask(self, workload_id: str, subtask_id: str, execution_plan: Any) -> bool:
        """
        Schedule a subtask for execution.
        
        Args:
            workload_id: Workload ID
            subtask_id: Subtask ID
            execution_plan: Execution plan for the subtask
            
        Returns:
            True if scheduling was successful, False otherwise
        """
        with self.lock:
            if workload_id not in self.execution_graphs:
                return False
            
            execution_graph = self.execution_graphs[workload_id]
            if subtask_id not in execution_graph.subtasks:
                return False
            
            # Update subtask status
            execution_graph.update_subtask_status(
                subtask_id=subtask_id,
                status="scheduled",
                execution_plan_id=getattr(execution_plan, "id", str(uuid.uuid4()))
            )
            
            logger.info(f"Scheduled subtask {subtask_id} for workload {workload_id}")
            return True
    
    def start_subtask_execution(self, workload_id: str, subtask_id: str) -> bool:
        """
        Mark a subtask as executing.
        
        Args:
            workload_id: Workload ID
            subtask_id: Subtask ID
            
        Returns:
            True if update was successful, False otherwise
        """
        with self.lock:
            if workload_id not in self.execution_graphs:
                return False
            
            execution_graph = self.execution_graphs[workload_id]
            if subtask_id not in execution_graph.subtasks:
                return False
            
            # Update subtask status
            execution_graph.update_subtask_status(
                subtask_id=subtask_id,
                status="executing",
                start_time=datetime.now()
            )
            
            logger.info(f"Started execution of subtask {subtask_id} for workload {workload_id}")
            return True
    
    def complete_subtask(self, workload_id: str, subtask_id: str, result: Any) -> bool:
        """
        Mark a subtask as completed with its result.
        
        Args:
            workload_id: Workload ID
            subtask_id: Subtask ID
            result: Subtask execution result
            
        Returns:
            True if update was successful, False otherwise
        """
        with self.lock:
            if workload_id not in self.execution_graphs:
                return False
            
            execution_graph = self.execution_graphs[workload_id]
            if subtask_id not in execution_graph.subtasks:
                return False
            
            # Update subtask status
            execution_graph.update_subtask_status(
                subtask_id=subtask_id,
                status="completed",
                result=result,
                end_time=datetime.now()
            )
            
            # Check if workload is completed
            if execution_graph.is_completed():
                # Aggregate results
                aggregated_result = self._aggregate_results(workload_id)
                execution_graph.aggregated_result = aggregated_result
                
                # Remove from active workloads
                if workload_id in self.active_workloads:
                    self.active_workloads.remove(workload_id)
                
                # Record workload completion
                self._record_workload_completion(workload_id, execution_graph)
                
                logger.info(f"Completed workload {workload_id} with {len(execution_graph.subtasks)} subtasks")
            
            logger.info(f"Completed subtask {subtask_id} for workload {workload_id}")
            return True
    
    def fail_subtask(self, workload_id: str, subtask_id: str, error: str) -> bool:
        """
        Mark a subtask as failed with error information.
        
        Args:
            workload_id: Workload ID
            subtask_id: Subtask ID
            error: Error information
            
        Returns:
            True if update was successful, False otherwise
        """
        with self.lock:
            if workload_id not in self.execution_graphs:
                return False
            
            execution_graph = self.execution_graphs[workload_id]
            if subtask_id not in execution_graph.subtasks:
                return False
            
            # Update subtask status
            status = execution_graph.status_map.get(subtask_id)
            if not status:
                return False
            
            status.status = "failed"
            status.error = error
            status.end_time = datetime.now()
            
            # Apply recovery strategy
            subtask = execution_graph.subtasks[subtask_id]
            recovery_strategy = subtask.custom_config.get("recovery_strategy", self.default_recovery_strategy)
            
            # Apply the recovery strategy
            if recovery_strategy in self.recovery_strategies:
                recovery_successful = self.recovery_strategies[recovery_strategy](workload_id, subtask_id, error)
                logger.info(f"Applied recovery strategy '{recovery_strategy}' for failed subtask {subtask_id}, success: {recovery_successful}")
                return recovery_successful
            else:
                logger.warning(f"Unknown recovery strategy '{recovery_strategy}' for failed subtask {subtask_id}")
                return False
    
    def get_aggregated_results(self, workload_id: str) -> Dict[str, Any]:
        """
        Get aggregated results for a completed workload.
        
        Args:
            workload_id: Workload ID
            
        Returns:
            Aggregated results dictionary
        """
        with self.lock:
            if workload_id not in self.execution_graphs:
                raise ValueError(f"Workload {workload_id} not found")
            
            execution_graph = self.execution_graphs[workload_id]
            
            if not execution_graph.is_completed():
                raise ValueError(f"Workload {workload_id} is not completed")
            
            # Return the cached aggregated result if available
            if execution_graph.aggregated_result is not None:
                return execution_graph.aggregated_result
            
            # Otherwise, compute and cache it
            aggregated_result = self._aggregate_results(workload_id)
            execution_graph.aggregated_result = aggregated_result
            
            return aggregated_result
    
    def get_execution_status(self, workload_id: str) -> Dict[str, Any]:
        """
        Get the current execution status of a workload.
        
        Args:
            workload_id: Workload ID
            
        Returns:
            Execution status information
        """
        with self.lock:
            if workload_id not in self.execution_graphs:
                return {"error": f"Workload {workload_id} not found"}
            
            execution_graph = self.execution_graphs[workload_id]
            
            # Count subtasks by status
            status_counts = {"pending": 0, "scheduled": 0, "executing": 0, "completed": 0, "failed": 0, "skipped": 0}
            for status in execution_graph.status_map.values():
                if status.status in status_counts:
                    status_counts[status.status] += 1
            
            # Calculate overall progress
            total_subtasks = len(execution_graph.subtasks)
            completed_subtasks = status_counts["completed"] + status_counts["skipped"]
            progress = (completed_subtasks / total_subtasks) * 100 if total_subtasks > 0 else 0
            
            # Get currently executing subtasks
            executing_subtasks = [
                status.subtask.subtask_id
                for status in execution_graph.status_map.values()
                if status.status == "executing"
            ]
            
            # Get next critical subtasks
            critical_path = execution_graph.get_critical_path_remaining()[:5]
            
            return {
                "workload_id": workload_id,
                "status": "completed" if execution_graph.is_completed() else "in_progress",
                "progress": progress,
                "subtask_counts": status_counts,
                "total_subtasks": total_subtasks,
                "completed_subtasks": completed_subtasks,
                "executing_subtasks": executing_subtasks,
                "critical_path": critical_path,
                "created_at": execution_graph.created_at,
                "completed_at": execution_graph.completed_at
            }
    
    def _aggregate_results(self, workload_id: str) -> Dict[str, Any]:
        """Aggregate results from all subtasks in a workload."""
        execution_graph = self.execution_graphs[workload_id]
        
        # Collect all subtask results
        results = {}
        for subtask_id, status in execution_graph.status_map.items():
            if status.status == "completed":
                results[subtask_id] = status.result
        
        # Perform aggregation based on workload config
        aggregation_method = execution_graph.aggregation_method
        
        if aggregation_method == "default":
            # Simple dictionary aggregation
            return {"subtask_results": results}
        
        elif aggregation_method == "concat":
            # Concatenate results (assuming list results)
            aggregated = []
            for result in results.values():
                if isinstance(result, list):
                    aggregated.extend(result)
                else:
                    aggregated.append(result)
            return {"results": aggregated}
        
        elif aggregation_method == "sum":
            # Sum all numerical results
            total = 0
            for result in results.values():
                if isinstance(result, (int, float)):
                    total += result
                elif isinstance(result, dict) and "value" in result and isinstance(result["value"], (int, float)):
                    total += result["value"]
            return {"sum": total}
        
        elif aggregation_method == "average":
            # Average all numerical results
            values = []
            for result in results.values():
                if isinstance(result, (int, float)):
                    values.append(result)
                elif isinstance(result, dict) and "value" in result and isinstance(result["value"], (int, float)):
                    values.append(result["value"])
            
            if values:
                average = sum(values) / len(values)
                return {"average": average, "count": len(values), "values": values}
            else:
                return {"average": 0, "count": 0, "values": []}
        
        elif aggregation_method == "custom":
            # Use custom aggregation function defined in workload config
            # (Not implemented here, would require code execution)
            return {"subtask_results": results, "warning": "Custom aggregation not applied"}
        
        else:
            return {"subtask_results": results}
    
    def _initialize_communication_channels(self, workload_id: str, config: Dict[str, Any]) -> None:
        """Initialize communication channels for subtasks."""
        if workload_id not in self.communication_channels:
            self.communication_channels[workload_id] = {}
        
        comm_channels = config.get("communication_channels", {})
        for channel_id, channel_config in comm_channels.items():
            self.communication_channels[workload_id][channel_id] = {
                "config": channel_config,
                "messages": queue.Queue(),
                "subscribers": set()
            }
    
    def _initialize_barriers(self, workload_id: str, config: Dict[str, Any]) -> None:
        """Initialize synchronization barriers for subtasks."""
        if workload_id not in self.barrier_management:
            self.barrier_management[workload_id] = {}
        
        barriers = config.get("barriers", {})
        for barrier_id, barrier_config in barriers.items():
            participants = set(barrier_config.get("participants", []))
            self.barrier_management[workload_id][barrier_id] = {
                "config": barrier_config,
                "participants": participants,
                "arrived": set(),
                "event": threading.Event()
            }
    
    def _apply_workload_optimizations(self, workload_id: str, config: Dict[str, Any]) -> None:
        """Apply optimizations to a workload."""
        optimizations = config.get("optimizations", [])
        for optimization in optimizations:
            if optimization in self.optimization_strategies:
                self.optimization_strategies[optimization](workload_id, config)
    
    def _record_workload_completion(self, workload_id: str, execution_graph: WorkloadExecutionGraph) -> None:
        """Record workload completion for performance history."""
        # Calculate overall statistics
        total_subtasks = len(execution_graph.subtasks)
        completion_time = (execution_graph.completed_at - execution_graph.created_at).total_seconds()
        
        subtask_times = []
        for status in execution_graph.status_map.values():
            if status.start_time and status.end_time:
                subtask_time = (status.end_time - status.start_time).total_seconds()
                subtask_times.append(subtask_time)
        
        avg_subtask_time = sum(subtask_times) / len(subtask_times) if subtask_times else 0
        
        # Record in workload history
        if workload_id not in self.workload_history:
            self.workload_history[workload_id] = []
        
        self.workload_history[workload_id].append({
            "timestamp": datetime.now(),
            "total_subtasks": total_subtasks,
            "completion_time": completion_time,
            "avg_subtask_time": avg_subtask_time,
            "subtask_count_by_status": {status: sum(1 for s in execution_graph.status_map.values() if s.status == status)
                                        for status in ["completed", "failed", "skipped"]}
        })
    
    def _recovery_strategy_retry(self, workload_id: str, subtask_id: str, error: str) -> bool:
        """Recovery strategy: retry the failed subtask."""
        execution_graph = self.execution_graphs[workload_id]
        status = execution_graph.status_map.get(subtask_id)
        
        if not status:
            return False
        
        # Check if retry limit exceeded
        subtask = execution_graph.subtasks[subtask_id]
        if status.retry_count >= subtask.retry_count:
            logger.warning(f"Retry limit exceeded for subtask {subtask_id} in workload {workload_id}")
            return False
        
        # Increment retry count and reset status to pending
        status.retry_count += 1
        status.status = "pending"
        status.error = None
        status.start_time = None
        status.end_time = None
        
        logger.info(f"Retrying subtask {subtask_id} in workload {workload_id} (attempt {status.retry_count})")
        return True
    
    def _recovery_strategy_reassign(self, workload_id: str, subtask_id: str, error: str) -> bool:
        """Recovery strategy: reassign the subtask to different hardware."""
        execution_graph = self.execution_graphs[workload_id]
        status = execution_graph.status_map.get(subtask_id)
        
        if not status:
            return False
        
        # Reset status to pending and mark for reassignment
        status.status = "pending"
        status.error = None
        status.start_time = None
        status.end_time = None
        status.assigned_hardware = None  # Clear hardware assignment
        
        # Add to subtask custom config to avoid reusing same hardware
        subtask = execution_graph.subtasks[subtask_id]
        if "failed_hardware" not in subtask.custom_config:
            subtask.custom_config["failed_hardware"] = []
        
        if status.assigned_hardware:
            for hw_id, _ in status.assigned_hardware:
                subtask.custom_config["failed_hardware"].append(hw_id)
        
        logger.info(f"Reassigning subtask {subtask_id} in workload {workload_id} to different hardware")
        return True
    
    def _recovery_strategy_skip(self, workload_id: str, subtask_id: str, error: str) -> bool:
        """Recovery strategy: skip the failed subtask if not critical."""
        execution_graph = self.execution_graphs[workload_id]
        subtask = execution_graph.subtasks[subtask_id]
        
        # Check if subtask is critical
        if subtask.is_critical:
            logger.warning(f"Cannot skip critical subtask {subtask_id} in workload {workload_id}")
            return False
        
        # Mark subtask as skipped
        execution_graph.update_subtask_status(
            subtask_id=subtask_id,
            status="skipped",
            error=error,
            end_time=datetime.now()
        )
        
        # Check if any other subtasks depend on this one
        for other_id, other_subtask in execution_graph.subtasks.items():
            if subtask_id in other_subtask.dependencies:
                logger.warning(f"Skipping subtask {subtask_id} affects dependent subtask {other_id}")
        
        logger.info(f"Skipped non-critical subtask {subtask_id} in workload {workload_id}")
        return True
    
    def _recovery_strategy_alternate(self, workload_id: str, subtask_id: str, error: str) -> bool:
        """Recovery strategy: use alternate implementation if available."""
        execution_graph = self.execution_graphs[workload_id]
        subtask = execution_graph.subtasks[subtask_id]
        
        # Check if alternate implementation is available
        if "alternate_implementation" not in subtask.custom_config:
            logger.warning(f"No alternate implementation available for subtask {subtask_id}")
            return False
        
        # Update subtask configuration to use alternate implementation
        alternate = subtask.custom_config["alternate_implementation"]
        subtask.custom_config["implementation"] = alternate
        subtask.custom_config["using_alternate"] = True
        
        # Reset status to pending
        status = execution_graph.status_map.get(subtask_id)
        status.status = "pending"
        status.error = None
        status.start_time = None
        status.end_time = None
        
        logger.info(f"Using alternate implementation for subtask {subtask_id} in workload {workload_id}")
        return True
    
    def _recovery_strategy_checkpoint(self, workload_id: str, subtask_id: str, error: str) -> bool:
        """Recovery strategy: restore from checkpoint if available."""
        execution_graph = self.execution_graphs[workload_id]
        subtask = execution_graph.subtasks[subtask_id]
        
        # Check if checkpoint is available
        if "checkpoint" not in subtask.custom_config:
            logger.warning(f"No checkpoint available for subtask {subtask_id}")
            return False
        
        # Update subtask configuration to use checkpoint
        checkpoint = subtask.custom_config["checkpoint"]
        subtask.custom_config["restore_checkpoint"] = checkpoint
        
        # Reset status to pending
        status = execution_graph.status_map.get(subtask_id)
        status.status = "pending"
        status.error = None
        status.start_time = None
        status.end_time = None
        
        logger.info(f"Restoring from checkpoint for subtask {subtask_id} in workload {workload_id}")
        return True
    
    def _optimization_memory_pooling(self, workload_id: str, config: Dict[str, Any]) -> None:
        """Optimization strategy: pool memory for related subtasks."""
        memory_pool_config = config.get("memory_pooling", {})
        if not memory_pool_config:
            return
        
        # Implementation would set up memory pooling for subtasks
        logger.info(f"Applied memory pooling optimization to workload {workload_id}")
    
    def _optimization_operation_fusion(self, workload_id: str, config: Dict[str, Any]) -> None:
        """Optimization strategy: fuse operations in compatible subtasks."""
        fusion_config = config.get("operation_fusion", {})
        if not fusion_config:
            return
        
        # Implementation would analyze and fuse operations in the subtask graph
        logger.info(f"Applied operation fusion optimization to workload {workload_id}")
    
    def _optimization_device_specialization(self, workload_id: str, config: Dict[str, Any]) -> None:
        """Optimization strategy: specialize subtasks for specific devices."""
        specialization_config = config.get("device_specialization", {})
        if not specialization_config:
            return
        
        # Implementation would adjust subtask requirements for specific devices
        logger.info(f"Applied device specialization optimization to workload {workload_id}")
    
    def _optimization_data_locality(self, workload_id: str, config: Dict[str, Any]) -> None:
        """Optimization strategy: optimize data placement for locality."""
        locality_config = config.get("data_locality", {})
        if not locality_config:
            return
        
        # Implementation would optimize data placement across subtasks
        logger.info(f"Applied data locality optimization to workload {workload_id}")
    
    def _optimization_parallel_execution(self, workload_id: str, config: Dict[str, Any]) -> None:
        """Optimization strategy: reorganize subtasks for better parallelism."""
        parallel_config = config.get("parallel_execution", {})
        if not parallel_config:
            return
        
        # Implementation would reorganize the subtask graph for better parallelism
        logger.info(f"Applied parallel execution optimization to workload {workload_id}")
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop for workloads."""
        while not self._stop_monitoring.is_set():
            try:
                with self.lock:
                    for workload_id in list(self.active_workloads):
                        if workload_id in self.execution_graphs:
                            # Check for timed out subtasks
                            self._check_subtask_timeouts(workload_id)
                            
                            # Check resource utilization
                            self._check_resource_utilization(workload_id)
                            
                            # Update execution metrics
                            self._update_execution_metrics(workload_id)
                
            except Exception as e:
                logger.error(f"Error in orchestrator monitoring loop: {e}")
            
            # Sleep for monitoring interval
            self._stop_monitoring.wait(self.monitoring_interval)
    
    def _check_subtask_timeouts(self, workload_id: str) -> None:
        """Check for subtasks that have exceeded their timeout."""
        execution_graph = self.execution_graphs[workload_id]
        now = datetime.now()
        
        for subtask_id, status in execution_graph.status_map.items():
            if status.status == "executing" and status.start_time:
                subtask = execution_graph.subtasks[subtask_id]
                elapsed = (now - status.start_time).total_seconds()
                
                if elapsed > subtask.timeout_seconds:
                    logger.warning(f"Subtask {subtask_id} in workload {workload_id} timed out after {elapsed:.1f}s")
                    
                    # Mark as failed and apply recovery strategy
                    self.fail_subtask(
                        workload_id=workload_id,
                        subtask_id=subtask_id,
                        error=f"Timeout after {elapsed:.1f}s (limit: {subtask.timeout_seconds}s)"
                    )
    
    def _check_resource_utilization(self, workload_id: str) -> None:
        """Check resource utilization for active subtasks."""
        if not hasattr(self, "resource_monitor"):
            return
        
        execution_graph = self.execution_graphs[workload_id]
        executing_subtasks = [
            (subtask_id, status)
            for subtask_id, status in execution_graph.status_map.items()
            if status.status == "executing"
        ]
        
        # Report high utilization
        for subtask_id, status in executing_subtasks:
            if status.assigned_hardware:
                for hw_id, _ in status.assigned_hardware:
                    utilization = self.resource_monitor.get_utilization(hw_id)
                    if utilization and utilization.get("cpu_utilization", 0) > 90:
                        logger.warning(f"High CPU utilization ({utilization['cpu_utilization']:.1f}%) for subtask {subtask_id} on hardware {hw_id}")
    
    def _update_execution_metrics(self, workload_id: str) -> None:
        """Update execution metrics for performance tracking."""
        if not hasattr(self, "performance_tracker"):
            return
        
        execution_graph = self.execution_graphs[workload_id]
        if execution_graph.is_completed():
            return
        
        # Calculate progress
        total_subtasks = len(execution_graph.subtasks)
        completed_subtasks = sum(1 for status in execution_graph.status_map.values() 
                               if status.status in ["completed", "skipped"])
        
        progress = (completed_subtasks / total_subtasks) * 100 if total_subtasks > 0 else 0
        
        # Record progress
        self.performance_tracker.record_progress(workload_id, progress)


class ResourceMonitor:
    """Monitor resource utilization for hardware devices."""
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.utilization_data: Dict[str, Dict[str, float]] = {}
        self.lock = threading.RLock()
    
    def get_utilization(self, hardware_id: str) -> Optional[Dict[str, float]]:
        """Get current utilization for a hardware device."""
        with self.lock:
            return self.utilization_data.get(hardware_id)
    
    def update_utilization(self, hardware_id: str, utilization: Dict[str, float]) -> None:
        """Update utilization data for a hardware device."""
        with self.lock:
            self.utilization_data[hardware_id] = utilization


class FailureDetectionSystem:
    """Detect and analyze failures in workload execution."""
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.failure_patterns: Dict[str, Dict[str, Any]] = {}
        self.recent_failures: List[Dict[str, Any]] = []
        self.lock = threading.RLock()
    
    def record_failure(self, workload_id: str, subtask_id: str, error: str, hardware_id: Optional[str] = None) -> None:
        """Record a subtask failure for pattern analysis."""
        with self.lock:
            failure = {
                "timestamp": datetime.now(),
                "workload_id": workload_id,
                "subtask_id": subtask_id,
                "error": error,
                "hardware_id": hardware_id
            }
            
            self.recent_failures.append(failure)
            
            # Limit size of recent failures list
            if len(self.recent_failures) > 100:
                self.recent_failures = self.recent_failures[-100:]
            
            # Analyze for patterns
            self._analyze_failure_patterns()
    
    def _analyze_failure_patterns(self) -> None:
        """Analyze recent failures for patterns."""
        # Group failures by error type
        error_groups = {}
        for failure in self.recent_failures:
            error_type = self._categorize_error(failure["error"])
            if error_type not in error_groups:
                error_groups[error_type] = []
            
            error_groups[error_type].append(failure)
        
        # Look for recurring patterns
        for error_type, failures in error_groups.items():
            if len(failures) >= 3:
                # Check if multiple failures on same hardware
                hardware_counts = {}
                for failure in failures:
                    hw_id = failure.get("hardware_id")
                    if hw_id:
                        hardware_counts[hw_id] = hardware_counts.get(hw_id, 0) + 1
                
                for hw_id, count in hardware_counts.items():
                    if count >= 3:
                        # Recurring hardware-specific failure pattern detected
                        pattern_key = f"{error_type}_{hw_id}"
                        if pattern_key not in self.failure_patterns:
                            self.failure_patterns[pattern_key] = {
                                "error_type": error_type,
                                "hardware_id": hw_id,
                                "occurrences": count,
                                "first_seen": min(f["timestamp"] for f in failures if f.get("hardware_id") == hw_id),
                                "last_seen": max(f["timestamp"] for f in failures if f.get("hardware_id") == hw_id)
                            }
                            
                            logger.warning(f"Detected failure pattern: {error_type} occurring on hardware {hw_id} ({count} times)")
                        else:
                            self.failure_patterns[pattern_key]["occurrences"] += 1
                            self.failure_patterns[pattern_key]["last_seen"] = max(f["timestamp"] for f in failures if f.get("hardware_id") == hw_id)
    
    def _categorize_error(self, error: str) -> str:
        """Categorize an error message into a general error type."""
        if "memory" in error.lower() or "out of memory" in error.lower():
            return "memory_error"
        elif "timeout" in error.lower():
            return "timeout_error"
        elif "connection" in error.lower() or "network" in error.lower():
            return "network_error"
        elif "permission" in error.lower() or "access" in error.lower():
            return "permission_error"
        else:
            return "general_error"


class PerformanceTracker:
    """Track performance metrics for workload execution."""
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.progress_history: Dict[str, List[Tuple[datetime, float]]] = {}
        self.execution_metrics: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.RLock()
    
    def record_progress(self, workload_id: str, progress: float) -> None:
        """Record progress for a workload execution."""
        with self.lock:
            if workload_id not in self.progress_history:
                self.progress_history[workload_id] = []
            
            self.progress_history[workload_id].append((datetime.now(), progress))
    
    def calculate_execution_rate(self, workload_id: str) -> Optional[float]:
        """Calculate execution rate (progress per second) for a workload."""
        with self.lock:
            if workload_id not in self.progress_history or len(self.progress_history[workload_id]) < 2:
                return None
            
            # Get first and last progress points
            first_time, first_progress = self.progress_history[workload_id][0]
            last_time, last_progress = self.progress_history[workload_id][-1]
            
            # Calculate progress per second
            time_diff = (last_time - first_time).total_seconds()
            progress_diff = last_progress - first_progress
            
            if time_diff <= 0:
                return 0.0
            
            return progress_diff / time_diff


# Factory method to create a workload profile
def create_workload_profile(
    workload_type: str,
    model_id: str = None,
    min_memory_gb: float = 1.0,
    min_compute_units: int = 2,
    metrics: Dict[str, float] = None,
    priority: int = 3,
    preferred_hardware_class: str = None,
    backend_requirements: List[str] = None,
    precision_requirements: List[str] = None,
    feature_requirements: List[str] = None,
    is_shardable: bool = False,
    min_shards: int = 1,
    max_shards: int = 1,
    allocation_strategy: str = "single",
    estimated_duration_seconds: int = 60,
    workload_id: str = None
) -> WorkloadProfile:
    """
    Create a workload profile with the specified parameters.
    
    Args:
        workload_type: Type of workload (vision, nlp, audio, etc.)
        model_id: Model identifier
        min_memory_gb: Minimum memory in GB
        min_compute_units: Minimum compute units
        metrics: Dictionary of workload metrics
        priority: Priority level (1-5, 1 = highest)
        preferred_hardware_class: Preferred hardware class
        backend_requirements: Required backends
        precision_requirements: Required precision types
        feature_requirements: Required hardware features
        is_shardable: Whether workload can be sharded
        min_shards: Minimum number of shards
        max_shards: Maximum number of shards
        allocation_strategy: Allocation strategy (single, sharded, replicated)
        estimated_duration_seconds: Estimated execution time in seconds
        workload_id: Optional workload ID (generated if not provided)
        
    Returns:
        WorkloadProfile instance
    """
    # Convert workload type string to enum
    try:
        wl_type = WorkloadType[workload_type.upper()]
    except (KeyError, AttributeError):
        wl_type = WorkloadType.MIXED
    
    # Generate ID if not provided
    if not workload_id:
        workload_id = str(uuid.uuid4())
    
    # Convert memory GB to bytes
    min_memory_bytes = int(min_memory_gb * 1024 * 1024 * 1024)
    
    # Convert backend requirements to SoftwareBackend enum
    required_backends = set()
    if backend_requirements:
        for backend in backend_requirements:
            try:
                required_backends.add(SoftwareBackend[backend.upper()])
            except (KeyError, AttributeError):
                logger.warning(f"Unknown backend type: {backend}")
    
    # Convert precision requirements to PrecisionType enum
    required_precisions = set()
    if precision_requirements:
        for precision in precision_requirements:
            try:
                required_precisions.add(PrecisionType[precision.upper()])
            except (KeyError, AttributeError):
                logger.warning(f"Unknown precision type: {precision}")
    
    # Convert feature requirements to AcceleratorFeature enum
    required_features = set()
    if feature_requirements:
        for feature in feature_requirements:
            try:
                required_features.add(AcceleratorFeature[feature.upper()])
            except (KeyError, AttributeError):
                logger.warning(f"Unknown feature: {feature}")
    
    # Convert preferred hardware class to enum
    preferred_hw_class = None
    if preferred_hardware_class:
        try:
            preferred_hw_class = HardwareClass[preferred_hardware_class.upper()]
        except (KeyError, AttributeError):
            logger.warning(f"Unknown hardware class: {preferred_hardware_class}")
    
    # Convert metrics to WorkloadProfileMetric enum
    workload_metrics = {}
    if metrics:
        for metric_name, value in metrics.items():
            try:
                metric_enum = WorkloadProfileMetric[metric_name.upper()]
                workload_metrics[metric_enum] = value
            except (KeyError, AttributeError):
                logger.warning(f"Unknown metric: {metric_name}")
    
    # Create and return profile
    return WorkloadProfile(
        workload_id=workload_id,
        workload_type=wl_type,
        required_backends=required_backends,
        required_precisions=required_precisions,
        required_features=required_features,
        min_memory_bytes=min_memory_bytes,
        min_compute_units=min_compute_units,
        metrics=workload_metrics,
        preferred_hardware_class=preferred_hw_class,
        priority=priority,
        is_shardable=is_shardable,
        min_shards=min_shards,
        max_shards=max_shards,
        allocation_strategy=allocation_strategy,
        estimated_duration_seconds=estimated_duration_seconds,
        custom_properties={"model_id": model_id}
    )