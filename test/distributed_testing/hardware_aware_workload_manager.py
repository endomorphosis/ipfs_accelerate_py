#!/usr/bin/env python3
"""
Hardware-Aware Workload Manager for Distributed Testing Framework

This module implements a sophisticated workload management system that understands
hardware characteristics and optimizes workload execution across heterogeneous
hardware environments.

Key features:
- Workload decomposition into hardware-specific components
- Adaptive batch sizing based on hardware capabilities
- Dynamic precision selection for optimal performance vs. accuracy trade-offs
- Memory footprint optimization across devices
- Operation fusion for specific hardware capabilities
"""

import os
import logging
import threading
import time
import uuid
import math
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json

from data.duckdb.distributed_testing.hardware_taxonomy import (
    HardwareCapabilityProfile, HardwareClass, SoftwareBackend, PrecisionType
)

from .enhanced_hardware_taxonomy import (
    EnhancedHardwareTaxonomy
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("hardware_aware_workload_manager")


@dataclass
class WorkloadRequirement:
    """Specific requirement for a workload."""
    requirement_id: str
    requirement_type: str  # "resource", "capability", "performance", "precision"
    value: Any
    importance: float = 1.0  # 0.0 to 1.0, higher means more important
    is_hard_requirement: bool = False  # If True, must be satisfied


@dataclass
class OperationProfile:
    """Profile of a specific operation within a workload."""
    operation_id: str
    operation_type: str  # e.g., "matmul", "conv2d", "attention"
    input_shapes: List[List[int]]
    output_shapes: List[List[int]]
    required_precision: PrecisionType = PrecisionType.FP32
    estimated_flops: Optional[int] = None
    estimated_memory_bytes: Optional[int] = None
    parallelism_degree: float = 1.0  # 0.0 to 1.0, higher means more parallelizable
    dependencies: List[str] = field(default_factory=list)  # IDs of operations this depends on


@dataclass
class Workload:
    """
    A workload specification with detailed requirements and characteristics.
    This is more specific than the WorkloadProfile in the base system.
    """
    workload_id: str
    workload_name: str
    workload_type: str  # e.g., "vision", "nlp", "audio", "training", "inference"
    requirements: List[WorkloadRequirement] = field(default_factory=list)
    operations: List[OperationProfile] = field(default_factory=list)
    required_backends: Set[SoftwareBackend] = field(default_factory=set)
    min_memory_bytes: int = 0
    priority: int = 3  # 1 (highest) to 5 (lowest)
    deadline: Optional[datetime] = None
    target_batch_size: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    custom_properties: Dict[str, Any] = field(default_factory=dict)
    
    def can_decompose(self) -> bool:
        """Check if this workload can be decomposed into smaller parts."""
        # Workload can be decomposed if operations have dependencies defined
        # and not all operations depend on all others
        if not self.operations or len(self.operations) <= 1:
            return False
        
        # Create a dependency graph
        all_dependencies = set()
        for operation in self.operations:
            all_dependencies.update(operation.dependencies)
            
        # If some operations have no dependencies, or not all operations are dependencies,
        # then workload can be decomposed
        operation_ids = {op.operation_id for op in self.operations}
        return len(all_dependencies) < len(operation_ids)
    
    def get_capability_requirements(self) -> Dict[str, Any]:
        """Get capability requirements for this workload."""
        capability_requirements = {}
        
        # Extract capability requirements from workload requirements
        for req in self.requirements:
            if req.requirement_type == "capability":
                capability_requirements[req.requirement_id] = req.value
        
        # Add default capability requirements based on workload type
        if self.workload_type == "vision":
            if "specialized.vision" not in capability_requirements:
                capability_requirements["specialized.vision"] = {"effectiveness": 0.7}
            if "compute.convolution" not in capability_requirements:
                capability_requirements["compute.convolution"] = {"performance": 50.0}
                
        elif self.workload_type == "nlp":
            if "specialized.nlp" not in capability_requirements:
                capability_requirements["specialized.nlp"] = {"effectiveness": 0.7}
            if "compute.matrix_multiplication" not in capability_requirements:
                capability_requirements["compute.matrix_multiplication"] = {"performance": 100.0}
                
        elif self.workload_type == "audio":
            if "specialized.audio" not in capability_requirements:
                capability_requirements["specialized.audio"] = {"effectiveness": 0.7}
        
        # Add requirements based on operations
        total_flops = 0
        has_matmul = False
        has_conv = False
        
        for op in self.operations:
            if op.estimated_flops is not None:
                total_flops += op.estimated_flops
            
            if op.operation_type == "matmul" or op.operation_type == "gemm":
                has_matmul = True
            elif op.operation_type.startswith("conv"):
                has_conv = True
        
        # Add compute requirements based on operations
        if has_matmul and "compute.matrix_multiplication" not in capability_requirements:
            capability_requirements["compute.matrix_multiplication"] = {
                "performance": total_flops / 1e9 if total_flops > 0 else 50.0
            }
            
        if has_conv and "compute.convolution" not in capability_requirements:
            capability_requirements["compute.convolution"] = {
                "performance": total_flops / 1e9 if total_flops > 0 else 30.0
            }
        
        # Add precision requirements
        required_precisions = set()
        for op in self.operations:
            required_precisions.add(op.required_precision)
        
        if len(required_precisions) > 1:
            if "precision.mixed" not in capability_requirements:
                capability_requirements["precision.mixed"] = {
                    "supported_precisions": [p.value for p in required_precisions]
                }
        
        return capability_requirements


@dataclass
class WorkloadDecomposition:
    """Decomposition of a workload into smaller parts."""
    original_workload_id: str
    subworkloads: List[Workload] = field(default_factory=list)
    dependencies: Dict[str, List[str]] = field(default_factory=dict)  # subworkload_id -> list of dependencies
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AllocationSlot:
    """Allocation slot for a workload on a hardware device."""
    hardware_id: str
    hardware_profile: HardwareCapabilityProfile
    start_time: datetime
    end_time: datetime
    workload_id: str
    allocation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: str = "scheduled"  # scheduled, running, completed, failed
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionPlan:
    """Plan for executing a workload."""
    workload: Workload
    allocations: List[AllocationSlot] = field(default_factory=list)
    is_decomposed: bool = False
    decomposition: Optional[WorkloadDecomposition] = None
    estimated_completion_time: datetime = field(default_factory=lambda: datetime.now() + timedelta(minutes=10))
    estimated_energy_joules: float = 0.0
    efficiency_score: float = 0.0
    plan_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: str = "created"  # created, validated, executing, completed, failed
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    custom_properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionStats:
    """Statistics from workload execution."""
    workload_id: str
    execution_time_seconds: float
    energy_joules: float
    hardware_ids: List[str]
    start_time: datetime
    end_time: datetime
    achieved_throughput: float = 0.0
    achieved_latency: float = 0.0
    peak_memory_bytes: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HardwareState:
    """Current state of a hardware device."""
    hardware_id: str
    hardware_profile: HardwareCapabilityProfile
    current_workloads: List[str] = field(default_factory=list)
    utilization_percentage: float = 0.0
    memory_usage_bytes: int = 0
    temperature: float = 0.0  # 0.0 to 1.0 scale
    power_usage_watts: float = 0.0
    current_allocation_slots: List[AllocationSlot] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)
    status: str = "available"  # available, busy, overloaded, error, offline
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkloadOptimizationResult:
    """Result of workload optimization."""
    workload_id: str
    optimized_workload: Workload
    optimization_type: str  # "batch_size", "precision", "memory", "operation_fusion"
    expected_improvement: float  # Ratio of improvement (1.1 = 10% better)
    trade_offs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class HardwareAwareWorkloadManager:
    """
    Hardware-aware workload manager for distributed testing.
    
    This class handles the advanced management of workloads in heterogeneous
    hardware environments, including workload decomposition, hardware matching,
    and optimization of execution parameters.
    """
    
    def __init__(self, hardware_taxonomy: EnhancedHardwareTaxonomy, db_path: Optional[str] = None):
        """
        Initialize the hardware-aware workload manager.
        
        Args:
            hardware_taxonomy: Enhanced hardware taxonomy for capability matching
            db_path: Optional path to database for performance tracking
        """
        self.taxonomy = hardware_taxonomy
        self.db_path = db_path
        self.lock = threading.RLock()
        
        # Workload management
        self.workloads: Dict[str, Workload] = {}
        self.execution_plans: Dict[str, ExecutionPlan] = {}
        self.current_hardware_states: Dict[str, HardwareState] = {}
        
        # Performance history
        self.execution_history: List[ExecutionStats] = []
        
        # Workload optimizations
        self.optimization_history: Dict[str, List[WorkloadOptimizationResult]] = {}
        
        # Scheduler state
        self.scheduled_workloads: List[Tuple[int, str]] = []  # List of (priority, workload_id) tuples
        self.workload_waiting_map: Dict[str, Optional[datetime]] = {}  # Workload ID -> deadline
        
        # Event handling
        self.event_callbacks: Dict[str, List[Callable]] = {
            "workload_registered": [],
            "plan_created": [],
            "execution_started": [],
            "execution_completed": [],
            "execution_failed": [],
            "hardware_state_changed": [],
            "optimization_applied": []
        }
        
        # Monitoring
        self.monitoring_interval = 10  # seconds
        self._stop_monitoring = threading.Event()
        self.monitoring_thread = None
        
        # Initialize hardware states based on taxonomy
        self._initialize_hardware_states()
    
    def _initialize_hardware_states(self):
        """Initialize hardware states based on taxonomy."""
        with self.lock:
            # Clear current states
            self.current_hardware_states = {}
            
            # Iterate through hardware profiles in taxonomy
            for hardware_class, profiles in self.taxonomy.hardware_profiles.items():
                for profile in profiles:
                    # Create hardware ID
                    hardware_id = f"{profile.hardware_class.value}_{profile.model_name}"
                    
                    # Create hardware state
                    state = HardwareState(
                        hardware_id=hardware_id,
                        hardware_profile=profile,
                        last_updated=datetime.now()
                    )
                    
                    # Add to state map
                    self.current_hardware_states[hardware_id] = state
            
            logger.info(f"Initialized {len(self.current_hardware_states)} hardware states")
    
    def start(self):
        """Start the hardware-aware workload manager."""
        # Start monitoring thread
        self._stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info("Hardware-aware workload manager started")
    
    def stop(self):
        """Stop the hardware-aware workload manager."""
        # Stop monitoring thread
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self._stop_monitoring.set()
            self.monitoring_thread.join(timeout=5)
        
        logger.info("Hardware-aware workload manager stopped")
    
    def _monitoring_loop(self):
        """Monitoring loop for workload manager."""
        while not self._stop_monitoring.is_set():
            try:
                # Update hardware states
                self._update_hardware_states()
                
                # Schedule pending workloads
                self._schedule_pending_workloads()
                
                # Check for completed or timed out allocations
                self._check_allocation_status()
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            # Sleep until next interval
            self._stop_monitoring.wait(self.monitoring_interval)
    
    def _update_hardware_states(self):
        """Update hardware states with latest information."""
        with self.lock:
            # For each hardware state
            for hardware_id, state in self.current_hardware_states.items():
                # Update last_updated
                state.last_updated = datetime.now()
                
                # Update current workloads
                state.current_workloads = [
                    slot.workload_id for slot in state.current_allocation_slots
                    if slot.status == "running"
                ]
                
                # Update utilization based on allocation slots
                if state.current_allocation_slots:
                    # Simple estimation: 0.2 to 0.9 based on number of slots
                    state.utilization_percentage = min(0.9, 0.2 + (len(state.current_allocation_slots) * 0.1))
                else:
                    state.utilization_percentage = 0.0
                
                # Update memory usage (simplified simulation)
                state.memory_usage_bytes = sum(
                    self.workloads.get(slot.workload_id, Workload("", "", "")).min_memory_bytes
                    for slot in state.current_allocation_slots
                    if slot.status in ["running", "scheduled"]
                )
                
                # Simulate temperature increase with utilization
                # Temperature decay: previous * 0.9 + new * 0.1
                target_temp = state.utilization_percentage
                state.temperature = state.temperature * 0.9 + target_temp * 0.1
                
                # Simulate power usage based on utilization and hardware type
                base_power = 5.0  # Base power in watts
                if state.hardware_profile.hardware_class == HardwareClass.CPU:
                    max_power = 65.0
                elif state.hardware_profile.hardware_class == HardwareClass.GPU:
                    max_power = 300.0
                elif state.hardware_profile.hardware_class == HardwareClass.TPU:
                    max_power = 200.0
                elif state.hardware_profile.hardware_class == HardwareClass.NPU:
                    max_power = 10.0
                else:
                    max_power = 50.0
                
                # Power is base_power + utilization * (max_power - base_power)
                state.power_usage_watts = base_power + (state.utilization_percentage * (max_power - base_power))
                
                # Update status based on utilization
                if state.utilization_percentage > 0.9:
                    state.status = "overloaded"
                elif state.utilization_percentage > 0.0:
                    state.status = "busy"
                else:
                    state.status = "available"
    
    def _schedule_pending_workloads(self):
        """Schedule pending workloads to available hardware."""
        with self.lock:
            if not self.scheduled_workloads:
                return
            
            # Sort by priority (lower value = higher priority)
            self.scheduled_workloads.sort()
            
            # Process up to 5 workloads per cycle
            processed_count = 0
            processed_workloads = []
            
            for priority, workload_id in self.scheduled_workloads:
                processed_workloads.append((priority, workload_id))
                
                # Skip if workload doesn't exist
                if workload_id not in self.workloads:
                    continue
                
                # Get workload
                workload = self.workloads[workload_id]
                
                # Check if deadline passed
                deadline = self.workload_waiting_map.get(workload_id)
                if deadline and deadline < datetime.now():
                    logger.warning(f"Workload {workload_id} missed deadline, cancelling")
                    # Remove from scheduling
                    if workload_id in self.workload_waiting_map:
                        del self.workload_waiting_map[workload_id]
                    continue
                
                # Create execution plan
                plan = self.create_execution_plan(workload_id)
                
                if plan:
                    # Store plan
                    self.execution_plans[plan.plan_id] = plan
                    
                    # Remove from waiting map
                    if workload_id in self.workload_waiting_map:
                        del self.workload_waiting_map[workload_id]
                    
                    # Update hardware states with allocation slots
                    for slot in plan.allocations:
                        hardware_id = slot.hardware_id
                        if hardware_id in self.current_hardware_states:
                            self.current_hardware_states[hardware_id].current_allocation_slots.append(slot)
                    
                    # Trigger event
                    self._trigger_event("plan_created", plan)
                    
                    logger.info(f"Created execution plan for workload {workload_id} with {len(plan.allocations)} allocations")
                    
                    processed_count += 1
                    if processed_count >= 5:
                        break
                else:
                    # Couldn't create plan, adjust priority and re-queue
                    new_priority = priority + 1
                    processed_workloads.append((new_priority, workload_id))
                    logger.debug(f"Couldn't create plan for workload {workload_id}, re-queued with priority {new_priority}")
            
            # Remove processed workloads
            self.scheduled_workloads = [item for item in self.scheduled_workloads if item not in processed_workloads]
            
            # Add back workloads that couldn't be scheduled
            for priority, workload_id in processed_workloads:
                if workload_id in self.workload_waiting_map:
                    self.scheduled_workloads.append((priority, workload_id))
    
    def _check_allocation_status(self):
        """Check status of allocation slots and update accordingly."""
        with self.lock:
            now = datetime.now()
            
            # Keep track of plans that need status updates
            plans_to_update = set()
            
            # Check all hardware states
            for hardware_id, state in self.current_hardware_states.items():
                # Find slots that have passed their end time
                completed_slots = []
                
                for slot in state.current_allocation_slots:
                    # Check if end time has passed
                    if slot.end_time <= now and slot.status in ["scheduled", "running"]:
                        # Mark as completed
                        slot.status = "completed"
                        completed_slots.append(slot)
                        
                        # Find the plan this slot belongs to
                        for plan_id, plan in self.execution_plans.items():
                            if any(a.allocation_id == slot.allocation_id for a in plan.allocations):
                                plans_to_update.add(plan_id)
                                break
                
                # Remove completed slots from current allocations
                state.current_allocation_slots = [
                    slot for slot in state.current_allocation_slots 
                    if slot not in completed_slots
                ]
            
            # Update plan status
            for plan_id in plans_to_update:
                plan = self.execution_plans[plan_id]
                
                # Check if all allocations are completed
                all_completed = all(
                    slot.status == "completed" 
                    for slot in plan.allocations
                )
                
                any_failed = any(
                    slot.status == "failed" 
                    for slot in plan.allocations
                )
                
                if all_completed:
                    # Mark plan as completed
                    plan.status = "completed"
                    
                    # Create execution stats
                    stats = self._create_execution_stats(plan)
                    
                    # Add to history
                    self.execution_history.append(stats)
                    
                    # Trigger event
                    self._trigger_event("execution_completed", plan, stats)
                    
                    logger.info(f"Execution plan {plan_id} completed successfully")
                elif any_failed:
                    # Mark plan as failed
                    plan.status = "failed"
                    
                    # Create execution stats
                    stats = self._create_execution_stats(plan)
                    
                    # Add to history
                    self.execution_history.append(stats)
                    
                    # Trigger event
                    self._trigger_event("execution_failed", plan, stats)
                    
                    logger.warning(f"Execution plan {plan_id} failed")
    
    def _create_execution_stats(self, plan: ExecutionPlan) -> ExecutionStats:
        """Create execution statistics from a completed plan."""
        # Find earliest start and latest end
        start_times = [slot.start_time for slot in plan.allocations]
        end_times = [slot.end_time for slot in plan.allocations]
        
        start_time = min(start_times) if start_times else plan.created_at
        end_time = max(end_times) if end_times else datetime.now()
        
        # Calculate duration
        duration_seconds = (end_time - start_time).total_seconds()
        
        # Collect hardware IDs
        hardware_ids = [slot.hardware_id for slot in plan.allocations]
        
        # Calculate energy usage (simplified)
        energy_joules = 0.0
        for slot in plan.allocations:
            # Get hardware state
            if slot.hardware_id in self.current_hardware_states:
                state = self.current_hardware_states[slot.hardware_id]
                # Energy = power * time
                slot_duration = (slot.end_time - slot.start_time).total_seconds()
                energy_joules += state.power_usage_watts * slot_duration
        
        # Create stats
        stats = ExecutionStats(
            workload_id=plan.workload.workload_id,
            execution_time_seconds=duration_seconds,
            energy_joules=energy_joules,
            hardware_ids=hardware_ids,
            start_time=start_time,
            end_time=end_time
        )
        
        # Add throughput/latency if available
        if "batch_size" in plan.metadata and duration_seconds > 0:
            batch_size = plan.metadata["batch_size"]
            stats.achieved_throughput = batch_size / duration_seconds
            stats.achieved_latency = duration_seconds / batch_size
        
        return stats
    
    def _trigger_event(self, event_type: str, *args, **kwargs):
        """Trigger event callbacks."""
        if event_type in self.event_callbacks:
            for callback in self.event_callbacks[event_type]:
                try:
                    callback(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error in event callback for {event_type}: {e}")
    
    def register_workload(self, workload: Workload) -> str:
        """
        Register a workload for scheduling.
        
        Args:
            workload: Workload to register
            
        Returns:
            Workload ID
        """
        with self.lock:
            # Ensure workload has an ID
            if not workload.workload_id:
                workload.workload_id = str(uuid.uuid4())
            
            # Store workload
            self.workloads[workload.workload_id] = workload
            
            # Add to scheduling queue
            self.scheduled_workloads.append((workload.priority, workload.workload_id))
            
            # Add to waiting map with deadline
            self.workload_waiting_map[workload.workload_id] = workload.deadline
            
            # Trigger event
            self._trigger_event("workload_registered", workload)
            
            logger.info(f"Registered workload {workload.workload_id} of type {workload.workload_type}")
            
            return workload.workload_id
    
    def create_execution_plan(self, workload_id: str) -> Optional[ExecutionPlan]:
        """
        Create an execution plan for a workload.
        
        Args:
            workload_id: ID of the workload to plan
            
        Returns:
            Execution plan or None if no suitable hardware found
        """
        with self.lock:
            # Check if workload exists
            if workload_id not in self.workloads:
                logger.warning(f"Workload {workload_id} not found")
                return None
            
            workload = self.workloads[workload_id]
            
            # Check if workload can be decomposed
            if workload.can_decompose():
                # Create decomposed plan
                return self._create_decomposed_execution_plan(workload)
            else:
                # Create single-device plan
                return self._create_single_device_execution_plan(workload)
    
    def _create_single_device_execution_plan(self, workload: Workload) -> Optional[ExecutionPlan]:
        """Create execution plan for a single-device workload."""
        # Get capability requirements
        capability_requirements = workload.get_capability_requirements()
        
        # Find optimal hardware
        hardware_matches = self.taxonomy.find_optimal_hardware_for_workload(
            capability_requirements,
            available_workers=None  # Consider all hardware
        )
        
        if not hardware_matches:
            logger.warning(f"No suitable hardware found for workload {workload.workload_id}")
            return None
        
        # Filter based on hardware availability
        available_matches = []
        for worker_id, profile, match_score in hardware_matches:
            # Get hardware state
            hardware_id = f"{profile.hardware_class.value}_{profile.model_name}"
            if hardware_id in self.current_hardware_states:
                state = self.current_hardware_states[hardware_id]
                
                # Check if hardware is available or busy (not overloaded or error)
                if state.status in ["available", "busy"]:
                    # Check if hardware has enough memory
                    if workload.min_memory_bytes <= (profile.memory.total_bytes - state.memory_usage_bytes):
                        # Adjust match score based on current load
                        adjusted_score = match_score * (1.0 - (state.utilization_percentage * 0.5))
                        available_matches.append((hardware_id, profile, adjusted_score))
        
        if not available_matches:
            logger.warning(f"No available hardware found for workload {workload.workload_id}")
            return None
        
        # Sort by adjusted score
        available_matches.sort(key=lambda x: x[2], reverse=True)
        
        # Select best match
        selected_hardware_id, selected_profile, selected_score = available_matches[0]
        
        # Optimize workload for selected hardware
        optimized_workload = self._optimize_workload_for_hardware(workload, selected_profile)
        
        # Estimate execution time
        execution_time_seconds = self._estimate_execution_time(optimized_workload, selected_profile)
        
        # Create allocation slot
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=execution_time_seconds)
        
        allocation = AllocationSlot(
            hardware_id=selected_hardware_id,
            hardware_profile=selected_profile,
            start_time=start_time,
            end_time=end_time,
            workload_id=workload.workload_id,
            status="scheduled"
        )
        
        # Create execution plan
        plan = ExecutionPlan(
            workload=optimized_workload,
            allocations=[allocation],
            is_decomposed=False,
            estimated_completion_time=end_time,
            efficiency_score=selected_score,
            status="validated"
        )
        
        # Estimate energy usage
        plan.estimated_energy_joules = self._estimate_energy_usage(optimized_workload, selected_profile, execution_time_seconds)
        
        # Add metadata
        plan.metadata["batch_size"] = optimized_workload.target_batch_size
        
        return plan
    
    def _create_decomposed_execution_plan(self, workload: Workload) -> Optional[ExecutionPlan]:
        """Create execution plan for a decomposable workload."""
        # Decompose workload
        decomposition = self._decompose_workload(workload)
        
        if not decomposition or not decomposition.subworkloads:
            logger.warning(f"Failed to decompose workload {workload.workload_id}")
            return None
        
        # Create execution plans for subworkloads
        sub_plans = []
        success = True
        
        for subworkload in decomposition.subworkloads:
            # Create single-device plan for subworkload
            sub_plan = self._create_single_device_execution_plan(subworkload)
            
            if sub_plan:
                sub_plans.append(sub_plan)
            else:
                success = False
                break
        
        if not success:
            logger.warning(f"Failed to create plans for all subworkloads of {workload.workload_id}")
            return None
        
        # Combine sub-plans into a single plan
        all_allocations = []
        for plan in sub_plans:
            all_allocations.extend(plan.allocations)
        
        # Find latest completion time
        completion_times = [plan.estimated_completion_time for plan in sub_plans]
        latest_completion = max(completion_times) if completion_times else (datetime.now() + timedelta(minutes=10))
        
        # Sum up efficiency scores and energy usage
        total_efficiency = sum(plan.efficiency_score for plan in sub_plans) / len(sub_plans)
        total_energy = sum(plan.estimated_energy_joules for plan in sub_plans)
        
        # Create combined plan
        combined_plan = ExecutionPlan(
            workload=workload,
            allocations=all_allocations,
            is_decomposed=True,
            decomposition=decomposition,
            estimated_completion_time=latest_completion,
            efficiency_score=total_efficiency,
            estimated_energy_joules=total_energy,
            status="validated"
        )
        
        return combined_plan
    
    def _decompose_workload(self, workload: Workload) -> Optional[WorkloadDecomposition]:
        """
        Decompose a workload into smaller parts based on operation dependencies.
        
        Args:
            workload: Workload to decompose
            
        Returns:
            Workload decomposition or None if decomposition not possible
        """
        if not workload.can_decompose():
            return None
        
        # Create dependency graph
        dependency_graph = {}
        operation_map = {}
        
        for op in workload.operations:
            operation_map[op.operation_id] = op
            dependency_graph[op.operation_id] = set(op.dependencies)
        
        # Identify groups of operations that form connected subgraphs
        # This is a simplified approach - more sophisticated algorithms could be used
        
        # Start with leaf operations (no other operations depend on them)
        all_ops = set(operation_map.keys())
        all_dependencies = set()
        for deps in dependency_graph.values():
            all_dependencies.update(deps)
        
        leaf_ops = all_ops - all_dependencies
        
        # Group operations based on dependencies
        groups = []
        remaining_ops = set(all_ops)
        
        # First group: leaf operations
        if leaf_ops:
            groups.append(leaf_ops)
            remaining_ops -= leaf_ops
        
        # Continue grouping operations by dependency level
        while remaining_ops:
            next_group = set()
            for op_id in remaining_ops:
                # Check if all dependencies are in previous groups
                deps = dependency_graph.get(op_id, set())
                if all(dep not in remaining_ops for dep in deps):
                    next_group.add(op_id)
            
            if not next_group:
                # If we can't find a next group, put all remaining ops in one group
                groups.append(remaining_ops)
                break
            
            groups.append(next_group)
            remaining_ops -= next_group
        
        # Create subworkloads from operation groups
        subworkloads = []
        dependencies = {}
        
        for i, group in enumerate(groups):
            # Create subworkload for this group
            subworkload_id = f"{workload.workload_id}_sub{i+1}"
            
            # Get operations for this group
            ops = [operation_map[op_id] for op_id in group]
            
            # Calculate minimum memory for this subworkload
            sub_memory = sum(op.estimated_memory_bytes or 0 for op in ops)
            
            # Create subworkload with same requirements but specific operations
            subworkload = Workload(
                workload_id=subworkload_id,
                workload_name=f"{workload.workload_name} (Part {i+1})",
                workload_type=workload.workload_type,
                requirements=workload.requirements,
                operations=ops,
                required_backends=workload.required_backends,
                min_memory_bytes=max(sub_memory, workload.min_memory_bytes // len(groups)),
                priority=workload.priority,
                deadline=workload.deadline,
                target_batch_size=workload.target_batch_size,
                metadata=dict(workload.metadata),
                custom_properties=dict(workload.custom_properties)
            )
            
            # Add parent workload reference
            subworkload.metadata["parent_workload_id"] = workload.workload_id
            subworkload.metadata["group_index"] = i
            
            # Add to subworkloads list
            subworkloads.append(subworkload)
            
            # Track dependencies between subworkloads
            sub_deps = set()
            for op_id in group:
                for dep_id in dependency_graph.get(op_id, set()):
                    # Find which subworkload contains this dependency
                    for j, other_group in enumerate(groups):
                        if dep_id in other_group:
                            sub_deps.add(f"{workload.workload_id}_sub{j+1}")
                            break
            
            dependencies[subworkload_id] = list(sub_deps)
        
        # Create decomposition
        decomposition = WorkloadDecomposition(
            original_workload_id=workload.workload_id,
            subworkloads=subworkloads,
            dependencies=dependencies
        )
        
        return decomposition
    
    def _optimize_workload_for_hardware(self, workload: Workload, hardware_profile: HardwareCapabilityProfile) -> Workload:
        """
        Optimize a workload for specific hardware.
        
        Args:
            workload: Workload to optimize
            hardware_profile: Hardware profile to optimize for
            
        Returns:
            Optimized workload
        """
        # Create copy of workload
        optimized = Workload(
            workload_id=workload.workload_id,
            workload_name=workload.workload_name,
            workload_type=workload.workload_type,
            requirements=list(workload.requirements),
            operations=list(workload.operations),
            required_backends=set(workload.required_backends),
            min_memory_bytes=workload.min_memory_bytes,
            priority=workload.priority,
            deadline=workload.deadline,
            target_batch_size=workload.target_batch_size,
            metadata=dict(workload.metadata),
            custom_properties=dict(workload.custom_properties)
        )
        
        # Apply optimizations
        optimized = self._optimize_batch_size(optimized, hardware_profile)
        optimized = self._optimize_precision(optimized, hardware_profile)
        optimized = self._optimize_memory_usage(optimized, hardware_profile)
        optimized = self._optimize_operation_fusion(optimized, hardware_profile)
        
        return optimized
    
    def _optimize_batch_size(self, workload: Workload, hardware_profile: HardwareCapabilityProfile) -> Workload:
        """Optimize batch size for hardware."""
        # Create working copy
        optimized = Workload(
            workload_id=workload.workload_id,
            workload_name=workload.workload_name,
            workload_type=workload.workload_type,
            requirements=list(workload.requirements),
            operations=list(workload.operations),
            required_backends=set(workload.required_backends),
            min_memory_bytes=workload.min_memory_bytes,
            priority=workload.priority,
            deadline=workload.deadline,
            target_batch_size=workload.target_batch_size,
            metadata=dict(workload.metadata),
            custom_properties=dict(workload.custom_properties)
        )
        
        # Start with current batch size
        current_batch_size = workload.target_batch_size
        
        # Calculate suggested batch size based on hardware
        suggested_batch_size = current_batch_size
        
        # Adjust based on hardware class
        if hardware_profile.hardware_class == HardwareClass.GPU:
            # GPUs benefit from larger batch sizes
            suggested_batch_size = max(16, current_batch_size)
            # Scale with compute units
            compute_factor = max(1, hardware_profile.compute_units // 16)
            suggested_batch_size = min(128, suggested_batch_size * compute_factor)
            
        elif hardware_profile.hardware_class == HardwareClass.TPU:
            # TPUs often work well with power-of-2 batch sizes
            suggested_batch_size = max(8, current_batch_size)
            suggested_batch_size = 2 ** int(math.log2(suggested_batch_size))
            
        elif hardware_profile.hardware_class == HardwareClass.NPU:
            # NPUs may benefit from smaller batch sizes
            suggested_batch_size = min(8, max(1, current_batch_size))
            
        elif hardware_profile.hardware_class == HardwareClass.HYBRID:
            # Browser-based hardware may have memory limitations
            suggested_batch_size = min(4, max(1, current_batch_size))
        
        # Check workload type specifics
        if workload.workload_type == "inference":
            # Inference can often use larger batch sizes
            suggested_batch_size = max(suggested_batch_size, 4)
        elif workload.workload_type == "training":
            # Training already has optimal batch sizes specified usually
            pass
        
        # Adjust operations for new batch size
        batch_size_ratio = suggested_batch_size / max(1, current_batch_size)
        
        for op in optimized.operations:
            # Adjust input/output shapes for batch dimension
            for shapes in [op.input_shapes, op.output_shapes]:
                for i, shape in enumerate(shapes):
                    if shape:  # Non-empty shape
                        shapes[i] = [int(suggested_batch_size)] + shape[1:]
            
            # Adjust memory requirements proportionally
            if op.estimated_memory_bytes is not None:
                op.estimated_memory_bytes = int(op.estimated_memory_bytes * batch_size_ratio)
        
        # Update workload batch size
        optimized.target_batch_size = suggested_batch_size
        
        # Adjust memory requirements proportionally
        optimized.min_memory_bytes = int(workload.min_memory_bytes * batch_size_ratio)
        
        # Record optimization
        optimization_result = WorkloadOptimizationResult(
            workload_id=workload.workload_id,
            optimized_workload=optimized,
            optimization_type="batch_size",
            expected_improvement=1.0 + (0.1 * (suggested_batch_size / max(1, current_batch_size) - 1)),
            trade_offs={
                "memory_increase": batch_size_ratio,
                "original_batch_size": current_batch_size,
                "new_batch_size": suggested_batch_size
            }
        )
        
        # Store optimization result
        if workload.workload_id not in self.optimization_history:
            self.optimization_history[workload.workload_id] = []
        self.optimization_history[workload.workload_id].append(optimization_result)
        
        # Trigger event
        self._trigger_event("optimization_applied", optimization_result)
        
        return optimized
    
    def _optimize_precision(self, workload: Workload, hardware_profile: HardwareCapabilityProfile) -> Workload:
        """Optimize precision for hardware."""
        # Create working copy
        optimized = Workload(
            workload_id=workload.workload_id,
            workload_name=workload.workload_name,
            workload_type=workload.workload_type,
            requirements=list(workload.requirements),
            operations=list(workload.operations),
            required_backends=set(workload.required_backends),
            min_memory_bytes=workload.min_memory_bytes,
            priority=workload.priority,
            deadline=workload.deadline,
            target_batch_size=workload.target_batch_size,
            metadata=dict(workload.metadata),
            custom_properties=dict(workload.custom_properties)
        )
        
        # Check if hardware supports mixed precision
        supports_fp16 = PrecisionType.FP16 in hardware_profile.supported_precisions
        supports_int8 = PrecisionType.INT8 in hardware_profile.supported_precisions
        
        # Get capabilities
        hardware_capabilities = self.taxonomy.get_inherited_capabilities(hardware_profile)
        
        supports_mixed_precision = "precision.mixed" in hardware_capabilities
        supports_quantization = "precision.quantization" in hardware_capabilities
        
        has_tensor_cores = "compute.tensor_core_acceleration" in hardware_capabilities
        
        # If neither mixed precision nor quantization are supported, return original
        if not (supports_mixed_precision or supports_quantization):
            return optimized
        
        # Count operations by type
        op_types = {}
        total_ops = len(optimized.operations)
        
        for op in optimized.operations:
            op_type = op.operation_type
            op_types[op_type] = op_types.get(op_type, 0) + 1
        
        # Determine which operations benefit from which precision
        fp16_candidates = ["matmul", "gemm", "conv1d", "conv2d", "conv3d"]
        int8_candidates = ["relu", "sigmoid", "tanh", "pool"]
        
        # Determine if mixed precision would be beneficial
        fp16_benefit = sum(op_types.get(op, 0) for op in fp16_candidates) / max(1, total_ops)
        int8_benefit = sum(op_types.get(op, 0) for op in int8_candidates) / max(1, total_ops)
        
        # Apply mixed precision where beneficial
        precision_changed = False
        memory_ratio = 1.0
        
        for op in optimized.operations:
            original_precision = op.required_precision
            
            # Apply FP16 for matrix multiplication and convolution
            if (op.operation_type in fp16_candidates and 
                supports_fp16 and 
                supports_mixed_precision and
                has_tensor_cores):
                op.required_precision = PrecisionType.FP16
                if original_precision != op.required_precision:
                    precision_changed = True
                    memory_ratio *= 0.5  # FP16 is half the size of FP32
            
            # Apply INT8 for activation functions and pooling
            elif (op.operation_type in int8_candidates and 
                 supports_int8 and
                 supports_quantization):
                op.required_precision = PrecisionType.INT8
                if original_precision != op.required_precision:
                    precision_changed = True
                    memory_ratio *= 0.25  # INT8 is quarter the size of FP32
        
        # If precision was changed, adjust memory requirements
        if precision_changed:
            # Apply memory reduction factor, but be conservative (don't reduce too much)
            memory_ratio = max(0.6, memory_ratio)
            optimized.min_memory_bytes = int(workload.min_memory_bytes * memory_ratio)
            
            # Record optimization
            optimization_result = WorkloadOptimizationResult(
                workload_id=workload.workload_id,
                optimized_workload=optimized,
                optimization_type="precision",
                expected_improvement=1.0 + (0.2 * fp16_benefit + 0.4 * int8_benefit),
                trade_offs={
                    "memory_ratio": memory_ratio,
                    "fp16_benefit": fp16_benefit,
                    "int8_benefit": int8_benefit
                }
            )
            
            # Store optimization result
            if workload.workload_id not in self.optimization_history:
                self.optimization_history[workload.workload_id] = []
            self.optimization_history[workload.workload_id].append(optimization_result)
            
            # Trigger event
            self._trigger_event("optimization_applied", optimization_result)
        
        return optimized
    
    def _optimize_memory_usage(self, workload: Workload, hardware_profile: HardwareCapabilityProfile) -> Workload:
        """Optimize memory usage for hardware."""
        # Create working copy
        optimized = Workload(
            workload_id=workload.workload_id,
            workload_name=workload.workload_name,
            workload_type=workload.workload_type,
            requirements=list(workload.requirements),
            operations=list(workload.operations),
            required_backends=set(workload.required_backends),
            min_memory_bytes=workload.min_memory_bytes,
            priority=workload.priority,
            deadline=workload.deadline,
            target_batch_size=workload.target_batch_size,
            metadata=dict(workload.metadata),
            custom_properties=dict(workload.custom_properties)
        )
        
        # Check if memory optimization is needed
        if workload.min_memory_bytes < hardware_profile.memory.total_bytes * 0.5:
            # Memory usage is reasonable, no optimization needed
            return optimized
        
        # Check if we have operation memory estimates
        has_memory_estimates = all(
            op.estimated_memory_bytes is not None and op.estimated_memory_bytes > 0
            for op in optimized.operations
        )
        
        if not has_memory_estimates:
            # Without memory estimates, we can't optimize effectively
            return optimized
        
        # Sort operations by memory usage
        sorted_ops = sorted(
            optimized.operations,
            key=lambda op: op.estimated_memory_bytes or 0,
            reverse=True
        )
        
        # Identify high-memory operations
        high_memory_ops = [
            op for op in sorted_ops
            if (op.estimated_memory_bytes or 0) > 100 * 1024 * 1024  # >100 MB
        ]
        
        if not high_memory_ops:
            # No high-memory operations to optimize
            return optimized
        
        # Apply memory optimizations
        for op in high_memory_ops:
            # Apply operation-specific optimizations
            if op.operation_type == "matmul" or op.operation_type == "gemm":
                # For matrix multiplication, consider block-wise computation
                if op.estimated_memory_bytes is not None:
                    op.estimated_memory_bytes = int(op.estimated_memory_bytes * 0.7)
                    op.metadata = op.metadata or {}
                    op.metadata["block_size"] = 1024  # Block size for block-wise matmul
            
            elif op.operation_type.startswith("conv"):
                # For convolution, consider depthwise separable convolution if appropriate
                if op.estimated_memory_bytes is not None:
                    op.estimated_memory_bytes = int(op.estimated_memory_bytes * 0.8)
                    op.metadata = op.metadata or {}
                    op.metadata["use_depthwise"] = True
            
            elif op.operation_type == "attention":
                # For attention, consider optimizations like FlashAttention
                if op.estimated_memory_bytes is not None:
                    op.estimated_memory_bytes = int(op.estimated_memory_bytes * 0.6)
                    op.metadata = op.metadata or {}
                    op.metadata["use_flash_attention"] = True
        
        # Recalculate total memory
        total_memory = sum(op.estimated_memory_bytes or 0 for op in optimized.operations)
        
        # Add overhead
        optimized.min_memory_bytes = int(total_memory * 1.1)  # 10% overhead
        
        # Record memory optimization
        memory_ratio = optimized.min_memory_bytes / workload.min_memory_bytes
        
        optimization_result = WorkloadOptimizationResult(
            workload_id=workload.workload_id,
            optimized_workload=optimized,
            optimization_type="memory",
            expected_improvement=1.0 / memory_ratio,
            trade_offs={
                "memory_ratio": memory_ratio,
                "original_memory": workload.min_memory_bytes,
                "new_memory": optimized.min_memory_bytes
            }
        )
        
        # Store optimization result
        if workload.workload_id not in self.optimization_history:
            self.optimization_history[workload.workload_id] = []
        self.optimization_history[workload.workload_id].append(optimization_result)
        
        # Trigger event
        self._trigger_event("optimization_applied", optimization_result)
        
        return optimized
    
    def _optimize_operation_fusion(self, workload: Workload, hardware_profile: HardwareCapabilityProfile) -> Workload:
        """Optimize operation fusion for hardware."""
        # Create working copy
        optimized = Workload(
            workload_id=workload.workload_id,
            workload_name=workload.workload_name,
            workload_type=workload.workload_type,
            requirements=list(workload.requirements),
            operations=list(workload.operations),
            required_backends=set(workload.required_backends),
            min_memory_bytes=workload.min_memory_bytes,
            priority=workload.priority,
            deadline=workload.deadline,
            target_batch_size=workload.target_batch_size,
            metadata=dict(workload.metadata),
            custom_properties=dict(workload.custom_properties)
        )
        
        # Check if operation fusion is applicable
        if len(optimized.operations) <= 1:
            # No operations to fuse
            return optimized
        
        # Create operation dependency graph
        op_graph = {}
        op_map = {}
        
        for op in optimized.operations:
            op_graph[op.operation_id] = set(op.dependencies)
            op_map[op.operation_id] = op
        
        # Identify fusion patterns
        # Pattern 1: LinearActivation (matmul/conv + activation)
        fusions = []
        
        for op_id, op in op_map.items():
            if op.operation_type in ["relu", "sigmoid", "tanh"]:
                # This is an activation function, check dependencies
                if len(op_graph.get(op_id, set())) == 1:
                    # Single dependency, check if it's a linear operation
                    dep_id = list(op_graph[op_id])[0]
                    dep_op = op_map.get(dep_id)
                    
                    if dep_op and dep_op.operation_type in ["matmul", "gemm", "conv1d", "conv2d", "conv3d"]:
                        # Found a fusion pattern: linear + activation
                        fusions.append(("LinearActivation", dep_id, op_id))
        
        # Pattern 2: ElementWiseChain (multiple element-wise ops in a row)
        element_wise_ops = ["add", "mul", "sub", "div", "pow", "relu", "sigmoid", "tanh"]
        
        for op_id, op in op_map.items():
            if op.operation_type in element_wise_ops:
                # Check for chains of element-wise operations
                chain = [op_id]
                current = op_id
                
                while True:
                    # Get dependencies of current operation
                    deps = op_graph.get(current, set())
                    
                    if len(deps) != 1:
                        # Not a single dependency, chain ends
                        break
                    
                    dep_id = list(deps)[0]
                    dep_op = op_map.get(dep_id)
                    
                    if not dep_op or dep_op.operation_type not in element_wise_ops:
                        # Not an element-wise op, chain ends
                        break
                    
                    # Add to chain and continue
                    chain.append(dep_id)
                    current = dep_id
                
                if len(chain) >= 3:
                    # Found a chain of at least 3 element-wise operations
                    fusions.append(("ElementWiseChain", chain))
        
        # Apply fusions to create new operations
        if fusions:
            # Create new operations list
            new_operations = []
            removed_ops = set()
            
            for fusion_type, *args in fusions:
                if fusion_type == "LinearActivation":
                    linear_id, activation_id = args
                    
                    # Skip if already removed
                    if linear_id in removed_ops or activation_id in removed_ops:
                        continue
                    
                    # Get original operations
                    linear_op = op_map[linear_id]
                    activation_op = op_map[activation_id]
                    
                    # Create fused operation
                    fused_id = f"{linear_id}_{activation_id}_fused"
                    fused_op = OperationProfile(
                        operation_id=fused_id,
                        operation_type=f"{linear_op.operation_type}_{activation_op.operation_type}",
                        input_shapes=linear_op.input_shapes,
                        output_shapes=activation_op.output_shapes,
                        required_precision=linear_op.required_precision,
                        estimated_flops=(linear_op.estimated_flops or 0) + (activation_op.estimated_flops or 0),
                        estimated_memory_bytes=max(linear_op.estimated_memory_bytes or 0, 
                                                activation_op.estimated_memory_bytes or 0),
                        parallelism_degree=min(linear_op.parallelism_degree, activation_op.parallelism_degree),
                        dependencies=linear_op.dependencies
                    )
                    
                    # Add metadata
                    fused_op.metadata = {
                        "fusion_type": "LinearActivation",
                        "original_ops": [linear_id, activation_id]
                    }
                    
                    # Add to new operations
                    new_operations.append(fused_op)
                    
                    # Mark original operations as removed
                    removed_ops.add(linear_id)
                    removed_ops.add(activation_id)
                    
                elif fusion_type == "ElementWiseChain":
                    chain = args[0]
                    
                    # Skip if any operation already removed
                    if any(op_id in removed_ops for op_id in chain):
                        continue
                    
                    # Get original operations
                    chain_ops = [op_map[op_id] for op_id in chain]
                    
                    # Create fused operation
                    fused_id = f"{'_'.join(chain)}_fused"
                    fused_op = OperationProfile(
                        operation_id=fused_id,
                        operation_type="element_wise_chain",
                        input_shapes=chain_ops[-1].input_shapes,
                        output_shapes=chain_ops[0].output_shapes,
                        required_precision=chain_ops[0].required_precision,
                        estimated_flops=sum(op.estimated_flops or 0 for op in chain_ops),
                        estimated_memory_bytes=max(op.estimated_memory_bytes or 0 for op in chain_ops),
                        parallelism_degree=min(op.parallelism_degree for op in chain_ops),
                        dependencies=chain_ops[-1].dependencies
                    )
                    
                    # Add metadata
                    fused_op.metadata = {
                        "fusion_type": "ElementWiseChain",
                        "original_ops": chain,
                        "operation_types": [op_map[op_id].operation_type for op_id in chain]
                    }
                    
                    # Add to new operations
                    new_operations.append(fused_op)
                    
                    # Mark original operations as removed
                    removed_ops.update(chain)
            
            # Add operations that were not fused
            for op in optimized.operations:
                if op.operation_id not in removed_ops:
                    new_operations.append(op)
            
            # Update dependencies for new operations
            for op in new_operations:
                # Update dependencies to point to fused operations
                new_deps = []
                for dep in op.dependencies:
                    if dep in removed_ops:
                        # Find fused operation that contains this dependency
                        for fused_op in new_operations:
                            if hasattr(fused_op, "metadata") and fused_op.metadata and "original_ops" in fused_op.metadata:
                                if dep in fused_op.metadata["original_ops"]:
                                    new_deps.append(fused_op.operation_id)
                                    break
                    else:
                        new_deps.append(dep)
                
                op.dependencies = new_deps
            
            # Update operations list
            optimized.operations = new_operations
            
            # Record optimization
            optimization_result = WorkloadOptimizationResult(
                workload_id=workload.workload_id,
                optimized_workload=optimized,
                optimization_type="operation_fusion",
                expected_improvement=1.0 + (0.05 * len(fusions)),
                trade_offs={
                    "num_fusions": len(fusions),
                    "original_op_count": len(workload.operations),
                    "new_op_count": len(new_operations),
                    "fusion_types": {fusion[0]: 0 for fusion in fusions}
                }
            )
            
            # Count fusion types
            for fusion_type, *_ in fusions:
                optimization_result.trade_offs["fusion_types"][fusion_type] += 1
            
            # Store optimization result
            if workload.workload_id not in self.optimization_history:
                self.optimization_history[workload.workload_id] = []
            self.optimization_history[workload.workload_id].append(optimization_result)
            
            # Trigger event
            self._trigger_event("optimization_applied", optimization_result)
        
        return optimized
    
    def _estimate_execution_time(self, workload: Workload, hardware_profile: HardwareCapabilityProfile) -> float:
        """
        Estimate execution time for a workload on specific hardware.
        
        Args:
            workload: Workload to estimate
            hardware_profile: Hardware profile to use
            
        Returns:
            Estimated execution time in seconds
        """
        # Start with base time estimate
        base_time = 1.0  # 1 second minimum
        
        # Calculate based on operations
        total_flops = 0
        
        for op in workload.operations:
            if op.estimated_flops is not None:
                total_flops += op.estimated_flops
        
        # If we have operation FLOPS, estimate based on hardware capabilities
        if total_flops > 0:
            # Get hardware compute capability based on operation types
            compute_capability = 1e9  # Default: 1 GFLOPS
            
            has_matmul = any(op.operation_type in ["matmul", "gemm"] for op in workload.operations)
            has_conv = any(op.operation_type.startswith("conv") for op in workload.operations)
            
            if has_matmul and has_conv:
                # Mixed workload, check for tensor cores
                if hardware_profile.hardware_class == HardwareClass.GPU:
                    if "compute.tensor_core_acceleration" in self.taxonomy.get_inherited_capabilities(hardware_profile):
                        compute_capability = 100e9  # 100 TFLOPS with tensor cores
                    else:
                        compute_capability = 20e9  # 20 TFLOPS for modern GPU
                elif hardware_profile.hardware_class == HardwareClass.TPU:
                    compute_capability = 45e9  # 45 TFLOPS for TPU
                elif hardware_profile.hardware_class == HardwareClass.NPU:
                    compute_capability = 8e9  # 8 TFLOPS for NPU
                elif hardware_profile.hardware_class == HardwareClass.CPU:
                    compute_capability = 2e9  # 2 TFLOPS for modern CPU
                elif hardware_profile.hardware_class == HardwareClass.HYBRID:
                    compute_capability = 5e9  # 5 TFLOPS for browser-based
            
            # Estimate time based on FLOPS and compute capability
            compute_time = total_flops / compute_capability
            
            # Adjust for batch size
            batch_size = workload.target_batch_size
            # Higher batch sizes are more efficient (sub-linear scaling)
            batch_factor = math.sqrt(batch_size) / batch_size if batch_size > 0 else 1.0
            
            # Calculate estimated time
            base_time = max(1.0, compute_time / batch_factor)
        else:
            # Fallback: estimate based on workload type and hardware class
            if workload.workload_type == "vision":
                if hardware_profile.hardware_class == HardwareClass.GPU:
                    base_time = 0.5
                elif hardware_profile.hardware_class == HardwareClass.TPU:
                    base_time = 0.3
                elif hardware_profile.hardware_class == HardwareClass.NPU:
                    base_time = 0.8
                elif hardware_profile.hardware_class == HardwareClass.CPU:
                    base_time = 2.0
                else:
                    base_time = 1.5
            
            elif workload.workload_type == "nlp":
                if hardware_profile.hardware_class == HardwareClass.GPU:
                    base_time = 0.7
                elif hardware_profile.hardware_class == HardwareClass.TPU:
                    base_time = 0.5
                elif hardware_profile.hardware_class == HardwareClass.NPU:
                    base_time = 1.0
                elif hardware_profile.hardware_class == HardwareClass.CPU:
                    base_time = 3.0
                else:
                    base_time = 2.0
            
            else:
                # Default for other workload types
                base_time = 1.0
            
            # Adjust for batch size
            batch_size = workload.target_batch_size
            base_time *= max(1, batch_size // 2)
        
        return base_time
    
    def _estimate_energy_usage(self, workload: Workload, hardware_profile: HardwareCapabilityProfile, duration_seconds: float) -> float:
        """
        Estimate energy usage for a workload on specific hardware.
        
        Args:
            workload: Workload to estimate
            hardware_profile: Hardware profile to use
            duration_seconds: Estimated execution time in seconds
            
        Returns:
            Estimated energy usage in joules
        """
        # Estimate power usage based on hardware class
        if hardware_profile.hardware_class == HardwareClass.CPU:
            power_watts = 65.0
        elif hardware_profile.hardware_class == HardwareClass.GPU:
            power_watts = 250.0
        elif hardware_profile.hardware_class == HardwareClass.TPU:
            power_watts = 200.0
        elif hardware_profile.hardware_class == HardwareClass.NPU:
            power_watts = 10.0
        elif hardware_profile.hardware_class == HardwareClass.HYBRID:
            power_watts = 30.0
        else:
            power_watts = 50.0
        
        # Adjust based on hardware TDP if available
        if hardware_profile.thermal_design_power_w is not None:
            power_watts = hardware_profile.thermal_design_power_w
        
        # Adjust based on workload utilization
        # Assume workload uses 70-90% of maximum power during execution
        utilization_factor = 0.7 + (0.2 * random.random())
        
        # Calculate energy: power * time
        energy_joules = power_watts * utilization_factor * duration_seconds
        
        return energy_joules
    
    def get_workload_optimization_history(self, workload_id: str) -> List[WorkloadOptimizationResult]:
        """
        Get optimization history for a workload.
        
        Args:
            workload_id: Workload ID
            
        Returns:
            List of optimization results
        """
        with self.lock:
            return self.optimization_history.get(workload_id, [])
    
    def get_hardware_state(self, hardware_id: str) -> Optional[HardwareState]:
        """
        Get current state of a hardware device.
        
        Args:
            hardware_id: Hardware ID
            
        Returns:
            Hardware state or None if not found
        """
        with self.lock:
            return self.current_hardware_states.get(hardware_id)
    
    def get_all_hardware_states(self) -> Dict[str, HardwareState]:
        """
        Get current states of all hardware devices.
        
        Returns:
            Dictionary of hardware ID -> hardware state
        """
        with self.lock:
            return dict(self.current_hardware_states)
    
    def register_event_callback(self, event_type: str, callback: Callable):
        """
        Register a callback for workload events.
        
        Args:
            event_type: Event type to register for
            callback: Callback function
        """
        with self.lock:
            if event_type in self.event_callbacks:
                self.event_callbacks[event_type].append(callback)
    
    def save_state(self, file_path: str) -> bool:
        """
        Save manager state to a JSON file.
        
        Args:
            file_path: Path to save state to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create state dictionary
            state = {
                "timestamp": datetime.now().isoformat(),
                "hardware_states": {},
                "execution_history": []
            }
            
            # Add hardware states
            for hardware_id, hardware_state in self.current_hardware_states.items():
                state["hardware_states"][hardware_id] = {
                    "hardware_id": hardware_state.hardware_id,
                    "utilization_percentage": hardware_state.utilization_percentage,
                    "memory_usage_bytes": hardware_state.memory_usage_bytes,
                    "temperature": hardware_state.temperature,
                    "power_usage_watts": hardware_state.power_usage_watts,
                    "status": hardware_state.status,
                    "current_workload_count": len(hardware_state.current_workloads)
                }
            
            # Add execution history
            for stats in self.execution_history:
                state["execution_history"].append({
                    "workload_id": stats.workload_id,
                    "execution_time_seconds": stats.execution_time_seconds,
                    "energy_joules": stats.energy_joules,
                    "start_time": stats.start_time.isoformat(),
                    "end_time": stats.end_time.isoformat(),
                    "achieved_throughput": stats.achieved_throughput,
                    "achieved_latency": stats.achieved_latency,
                    "peak_memory_bytes": stats.peak_memory_bytes
                })
            
            # Save to file
            with open(file_path, 'w') as f:
                json.dump(state, f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"Error saving state to {file_path}: {e}")
            return False
    
    def load_state(self, file_path: str) -> bool:
        """
        Load manager state from a JSON file.
        
        Args:
            file_path: Path to load state from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load from file
            with open(file_path, 'r') as f:
                state = json.load(f)
            
            # Process execution history
            execution_history = []
            
            for stats_dict in state.get("execution_history", []):
                stats = ExecutionStats(
                    workload_id=stats_dict["workload_id"],
                    execution_time_seconds=stats_dict["execution_time_seconds"],
                    energy_joules=stats_dict["energy_joules"],
                    hardware_ids=stats_dict.get("hardware_ids", []),
                    start_time=datetime.fromisoformat(stats_dict["start_time"]),
                    end_time=datetime.fromisoformat(stats_dict["end_time"]),
                    achieved_throughput=stats_dict.get("achieved_throughput", 0.0),
                    achieved_latency=stats_dict.get("achieved_latency", 0.0),
                    peak_memory_bytes=stats_dict.get("peak_memory_bytes", 0)
                )
                
                execution_history.append(stats)
            
            # Update execution history
            with self.lock:
                self.execution_history = execution_history
            
            return True
        except Exception as e:
            logger.error(f"Error loading state from {file_path}: {e}")
            return False
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        Get summary of optimizations applied.
        
        Returns:
            Dictionary with optimization summary
        """
        with self.lock:
            # Count optimizations by type
            optimization_counts = {}
            
            for workload_id, optimizations in self.optimization_history.items():
                for opt in optimizations:
                    opt_type = opt.optimization_type
                    optimization_counts[opt_type] = optimization_counts.get(opt_type, 0) + 1
            
            # Calculate average improvement by type
            improvement_by_type = {}
            
            for workload_id, optimizations in self.optimization_history.items():
                for opt in optimizations:
                    opt_type = opt.optimization_type
                    
                    if opt_type not in improvement_by_type:
                        improvement_by_type[opt_type] = []
                    
                    improvement_by_type[opt_type].append(opt.expected_improvement)
            
            # Calculate averages
            avg_improvement = {}
            
            for opt_type, improvements in improvement_by_type.items():
                avg_improvement[opt_type] = sum(improvements) / len(improvements)
            
            # Create summary
            summary = {
                "optimization_counts": optimization_counts,
                "average_improvement": avg_improvement,
                "total_optimizations": sum(optimization_counts.values()),
                "unique_workloads_optimized": len(self.optimization_history)
            }
            
            return summary