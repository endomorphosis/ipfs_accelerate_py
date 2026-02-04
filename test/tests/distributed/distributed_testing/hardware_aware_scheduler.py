#!/usr/bin/env python3
"""
Hardware-Aware Scheduler for Distributed Testing Framework

This module implements a hardware-aware scheduling algorithm that integrates the 
Hardware-Aware Workload Management system with the Load Balancer component, enabling
more efficient and intelligent distribution of tests based on hardware capabilities
and workload characteristics.
"""

import logging
import random
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
import math

# Import load balancer components
from data.duckdb.distributed_testing.load_balancer.models import (
    TestRequirements, WorkerCapabilities, WorkerLoad, WorkerPerformance
)
from data.duckdb.distributed_testing.load_balancer.scheduling_algorithms import SchedulingAlgorithm

# Import hardware workload management components
from test.tests.distributed.distributed_testing.hardware_workload_management import (
    HardwareWorkloadManager, WorkloadProfile, WorkloadType, WorkloadProfileMetric,
    create_workload_profile
)

# Import hardware taxonomy
from data.duckdb.distributed_testing.hardware_taxonomy import (
    HardwareTaxonomy, HardwareCapabilityProfile, HardwareClass,
    SoftwareBackend, PrecisionType, AcceleratorFeature
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("hardware_aware_scheduler")


class HardwareAwareScheduler(SchedulingAlgorithm):
    """
    Hardware-aware scheduling algorithm that uses the Hardware-Aware Workload Management
    system to make more intelligent scheduling decisions based on hardware capabilities
    and workload characteristics.
    """
    
    def __init__(self, hardware_workload_manager: HardwareWorkloadManager, hardware_taxonomy: HardwareTaxonomy):
        """
        Initialize the hardware-aware scheduler.
        
        Args:
            hardware_workload_manager: Hardware workload manager instance
            hardware_taxonomy: Hardware taxonomy instance
        """
        self.workload_manager = hardware_workload_manager
        self.hardware_taxonomy = hardware_taxonomy
        
        # Cache of worker hardware profiles
        self.worker_hardware_cache: Dict[str, List[HardwareCapabilityProfile]] = {}
        
        # Cache of test-to-workload mappings
        self.test_workload_cache: Dict[str, str] = {}
        
        # Preferred worker for workload types
        self.workload_worker_preferences: Dict[str, Dict[str, float]] = {}
        
        # Thermal state tracking
        self.worker_thermal_states: Dict[str, Dict[str, Any]] = {}
    
    def select_worker(self, test_requirements: TestRequirements, 
                    available_workers: Dict[str, WorkerCapabilities],
                    worker_loads: Dict[str, WorkerLoad],
                    performance_data: Dict[str, Dict[str, WorkerPerformance]]) -> Optional[str]:
        """
        Select the best worker for the given test requirements using hardware-aware scheduling.
        
        Args:
            test_requirements: Requirements for the test to schedule
            available_workers: Dict of worker_id to WorkerCapabilities
            worker_loads: Dict of worker_id to WorkerLoad
            performance_data: Performance history for workers (worker_id -> test_type -> WorkerPerformance)
            
        Returns:
            Selected worker ID, or None if no suitable worker found
        """
        # Convert test requirements to workload profile
        workload_profile = self._test_to_workload_profile(test_requirements)
        
        # Ensure worker hardware profiles are in the taxonomy
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
                    available_hardware.append((worker_id, hardware_profile, adjusted_efficiency))
        
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
                  f"with efficiency {selected_efficiency:.2f}")
        
        # Update preferences for this workload type
        self._update_workload_preferences(workload_profile.workload_type.value, selected_worker_id, selected_efficiency)
        
        # Cache the workload ID for the test
        self.test_workload_cache[test_requirements.test_id] = workload_profile.workload_id
        
        return selected_worker_id
    
    def _test_to_workload_profile(self, test_requirements: TestRequirements) -> WorkloadProfile:
        """
        Convert test requirements to a workload profile.
        
        Args:
            test_requirements: Test requirements to convert
            
        Returns:
            Workload profile
        """
        # Check if test already has a workload profile in custom properties
        if hasattr(test_requirements, 'custom_properties') and test_requirements.custom_properties:
            if 'workload_profile' in test_requirements.custom_properties:
                # Workload profile already exists, deserialize and return
                wp_dict = test_requirements.custom_properties['workload_profile']
                
                # Convert dictionary back to workload profile
                # This is a simplified version, real implementation would need proper deserialization
                workload_type = wp_dict.get('workload_type', 'MIXED')
                if isinstance(workload_type, dict) and 'value' in workload_type:
                    workload_type = workload_type['value']
                
                return workload_profile_from_dict(wp_dict)
        
        # Determine workload type from test type
        workload_type = self._determine_workload_type(test_requirements.test_type)
        
        # Determine compute intensity (0.0-1.0)
        compute_intensity = 0.5  # Default medium intensity
        if test_requirements.expected_duration > 120:
            compute_intensity = 0.8  # Long tests are typically compute intensive
        elif test_requirements.expected_duration < 10:
            compute_intensity = 0.3  # Short tests are typically less compute intensive
        
        # Determine memory intensity (0.0-1.0)
        memory_intensity = min(1.0, test_requirements.minimum_memory / 16.0)  # Scale based on memory requirements
        
        # Determine backend requirements
        backend_requirements = []
        if test_requirements.required_backend:
            backend_requirements.append(test_requirements.required_backend)
        elif test_requirements.preferred_backend:
            backend_requirements.append(test_requirements.preferred_backend)
        
        # Create metrics dictionary
        metrics = {
            "COMPUTE_INTENSITY": compute_intensity,
            "MEMORY_INTENSITY": memory_intensity,
            "LATENCY_SENSITIVITY": 0.5,  # Default medium sensitivity
            "THROUGHPUT_SENSITIVITY": 0.5  # Default medium sensitivity
        }
        
        # Set shardable flag based on test requirements
        is_shardable = False
        min_shards = 1
        max_shards = 1
        allocation_strategy = "single"
        
        # Check custom properties for multi-device configuration
        if hasattr(test_requirements, 'custom_properties') and test_requirements.custom_properties:
            is_shardable = test_requirements.custom_properties.get('is_shardable', False)
            min_shards = test_requirements.custom_properties.get('min_shards', 1)
            max_shards = test_requirements.custom_properties.get('max_shards', 1)
            allocation_strategy = test_requirements.custom_properties.get('allocation_strategy', 'single')
        
        # Create workload profile
        return create_workload_profile(
            workload_type=workload_type,
            model_id=test_requirements.model_id,
            min_memory_gb=test_requirements.minimum_memory,
            min_compute_units=2,  # Default, could be better estimated
            metrics=metrics,
            priority=test_requirements.priority,
            backend_requirements=backend_requirements,
            is_shardable=is_shardable,
            min_shards=min_shards,
            max_shards=max_shards,
            allocation_strategy=allocation_strategy,
            estimated_duration_seconds=int(test_requirements.expected_duration),
            workload_id=test_requirements.test_id
        )
    
    def _determine_workload_type(self, test_type: Optional[str]) -> str:
        """
        Determine workload type from test type.
        
        Args:
            test_type: Test type string
            
        Returns:
            Workload type string
        """
        if not test_type:
            return "MIXED"
        
        test_type_lower = test_type.lower()
        
        if "vision" in test_type_lower or "image" in test_type_lower:
            return "VISION"
        elif "nlp" in test_type_lower or "text" in test_type_lower or "language" in test_type_lower:
            return "NLP"
        elif "audio" in test_type_lower or "speech" in test_type_lower:
            return "AUDIO"
        elif "embedding" in test_type_lower:
            return "EMBEDDING"
        elif "training" in test_type_lower:
            return "TRAINING"
        elif "inference" in test_type_lower:
            return "INFERENCE"
        elif "conversational" in test_type_lower:
            return "CONVERSATIONAL"
        elif "multi" in test_type_lower:
            return "MULTIMODAL"
        else:
            return "MIXED"
    
    def _update_worker_hardware_profiles(self, available_workers: Dict[str, WorkerCapabilities]) -> None:
        """
        Update worker hardware profiles in the hardware taxonomy.
        
        Args:
            available_workers: Dict of worker_id to WorkerCapabilities
        """
        for worker_id, capabilities in available_workers.items():
            # Skip if already processed
            if worker_id in self.worker_hardware_cache:
                continue
            
            # Convert worker capabilities to hardware profiles
            hardware_profiles = self._worker_capabilities_to_hardware_profiles(worker_id, capabilities)
            
            # Store in cache
            self.worker_hardware_cache[worker_id] = hardware_profiles
            
            # Register with hardware taxonomy
            self.hardware_taxonomy.register_worker_hardware(worker_id, hardware_profiles)
    
    def _worker_capabilities_to_hardware_profiles(
            self, worker_id: str, capabilities: WorkerCapabilities
        ) -> List[HardwareCapabilityProfile]:
        """
        Convert worker capabilities to hardware capability profiles.
        
        Args:
            worker_id: Worker ID
            capabilities: Worker capabilities
            
        Returns:
            List of hardware capability profiles
        """
        profiles = []
        
        # Common memory profile for all hardware
        memory_profile = {
            "total_bytes": int(capabilities.available_memory * 1024 * 1024 * 1024),
            "available_bytes": int(capabilities.available_memory * 0.9 * 1024 * 1024 * 1024),
            "is_shared": False,
            "hierarchy_levels": 2,
            "has_unified_memory": False,
            "memory_type": "DDR4"
        }
        
        # Check for CPU
        if capabilities.cpu_cores > 0:
            # Create CPU profile
            cpu_profile = {
                "hardware_class": HardwareClass.CPU.value,
                "architecture": "X86_64",  # Assume x86_64 for simplicity
                "vendor": "INTEL",  # Assume Intel for simplicity
                "model_name": f"CPU-{capabilities.cpu_cores}cores",
                "supported_backends": ["PYTORCH", "TENSORFLOW", "ONNX"],
                "supported_precisions": ["FP32", "INT8"],
                "features": ["SIMD", "AVX"],
                "memory": memory_profile,
                "compute_units": capabilities.cpu_cores,
                "clock_speed_mhz": 2500  # Default assumption
            }
            profiles.append(hardware_profile_from_dict(cpu_profile))
        
        # Check for GPU
        for backend in capabilities.supported_backends:
            if "cuda" in backend.lower() or "gpu" in backend.lower():
                # Create GPU profile
                gpu_profile = {
                    "hardware_class": HardwareClass.GPU.value,
                    "architecture": "GPU_CUDA",  # Assume CUDA for simplicity
                    "vendor": "NVIDIA",  # Assume NVIDIA for simplicity
                    "model_name": f"GPU-{backend}",
                    "supported_backends": ["PYTORCH", "TENSORFLOW", "ONNX", "CUDA"],
                    "supported_precisions": ["FP32", "FP16", "INT8"],
                    "features": ["COMPUTE_SHADERS", "TENSOR_CORES"],
                    "memory": memory_profile,
                    "compute_units": 80,  # Default assumption
                    "clock_speed_mhz": 1500  # Default assumption
                }
                profiles.append(hardware_profile_from_dict(gpu_profile))
        
        # Check for TPU
        for backend in capabilities.supported_backends:
            if "tpu" in backend.lower():
                # Create TPU profile
                tpu_profile = {
                    "hardware_class": HardwareClass.TPU.value,
                    "architecture": "TPU",
                    "vendor": "GOOGLE",
                    "model_name": f"TPU-{backend}",
                    "supported_backends": ["TENSORFLOW", "ONNX"],
                    "supported_precisions": ["FP32", "BF16", "INT8"],
                    "features": ["TENSOR_CORES"],
                    "memory": memory_profile,
                    "compute_units": 8,  # Default assumption
                    "clock_speed_mhz": 1000  # Default assumption
                }
                profiles.append(hardware_profile_from_dict(tpu_profile))
        
        # Check for NPU
        for backend in capabilities.supported_backends:
            if "npu" in backend.lower() or "qnn" in backend.lower():
                # Create NPU profile
                npu_profile = {
                    "hardware_class": HardwareClass.NPU.value,
                    "architecture": "NPU_QUALCOMM",  # Assume Qualcomm for simplicity
                    "vendor": "QUALCOMM",
                    "model_name": f"NPU-{backend}",
                    "supported_backends": ["ONNX", "QNN"],
                    "supported_precisions": ["FP32", "FP16", "INT8", "INT4"],
                    "features": ["QUANTIZATION"],
                    "memory": memory_profile,
                    "compute_units": 4,  # Default assumption
                    "clock_speed_mhz": 800  # Default assumption
                }
                profiles.append(hardware_profile_from_dict(npu_profile))
        
        # Check for WebNN/WebGPU
        for backend in capabilities.supported_backends:
            if "webnn" in backend.lower() or "webgpu" in backend.lower():
                # Create browser profile
                browser_profile = {
                    "hardware_class": HardwareClass.HYBRID.value,
                    "architecture": "GPU_WEBGPU" if "webgpu" in backend.lower() else "OTHER",
                    "vendor": "OTHER",
                    "model_name": f"Browser-{backend}",
                    "supported_backends": ["WEBNN" if "webnn" in backend.lower() else "WEBGPU"],
                    "supported_precisions": ["FP32", "FP16"],
                    "features": ["COMPUTE_SHADERS"],
                    "memory": {
                        "total_bytes": int(2 * 1024 * 1024 * 1024),  # Assume 2GB
                        "available_bytes": int(1 * 1024 * 1024 * 1024),  # Assume 1GB
                        "is_shared": True,
                        "hierarchy_levels": 1,
                        "has_unified_memory": True,
                        "memory_type": "Shared"
                    },
                    "compute_units": 2,  # Default assumption
                    "clock_speed_mhz": 1000  # Default assumption
                }
                profiles.append(hardware_profile_from_dict(browser_profile))
        
        # If no profiles created, add a generic one
        if not profiles:
            generic_profile = {
                "hardware_class": HardwareClass.CPU.value,
                "architecture": "X86_64",
                "vendor": "OTHER",
                "model_name": "Generic-CPU",
                "supported_backends": capabilities.supported_backends,
                "supported_precisions": ["FP32"],
                "features": [],
                "memory": memory_profile,
                "compute_units": max(1, capabilities.cpu_cores),
                "clock_speed_mhz": 2000  # Default assumption
            }
            profiles.append(hardware_profile_from_dict(generic_profile))
        
        # Create composite hardware ID for each profile
        for profile in profiles:
            profile_hw_id = f"{worker_id}_{profile.model_name}"
            # We could store this mapping if needed
        
        return profiles
    
    def _adjust_efficiency_for_load_and_thermal(
            self, efficiency: float, worker_id: str, load: WorkerLoad, 
            hardware_profile: HardwareCapabilityProfile
        ) -> float:
        """
        Adjust efficiency score based on current load and thermal state.
        
        Args:
            efficiency: Base efficiency score
            worker_id: Worker ID
            load: Current worker load
            hardware_profile: Hardware capability profile
            
        Returns:
            Adjusted efficiency score
        """
        # Start with base efficiency
        adjusted_efficiency = efficiency
        
        # Get current load score
        load_score = load.get_effective_load_score() if hasattr(load, 'get_effective_load_score') else load.calculate_load_score()
        
        # Reduce efficiency based on load (more loaded = less efficient)
        # Load factor: 1.0 when load=0, 0.7 when load=1.0
        load_factor = 1.0 - (0.3 * load_score)
        adjusted_efficiency *= load_factor
        
        # Check for warming/cooling state
        if load.warming_state:
            # Reduce efficiency for warming workers
            warming_factor = load.performance_level if hasattr(load, 'performance_level') else 0.8
            adjusted_efficiency *= warming_factor
        
        if load.cooling_state:
            # Reduce efficiency for cooling workers
            cooling_factor = load.performance_level if hasattr(load, 'performance_level') else 0.9
            adjusted_efficiency *= cooling_factor
        
        # Apply thermal state adjustment
        thermal_state = self.worker_thermal_states.get(worker_id, {"temperature": 0.0})
        temperature = thermal_state.get("temperature", 0.0)
        
        # Reduce efficiency based on temperature (higher temperature = less efficient)
        # Temp factor: 1.0 when temp=0, 0.7 when temp=1.0
        temp_factor = 1.0 - (0.3 * temperature)
        adjusted_efficiency *= temp_factor
        
        return adjusted_efficiency
    
    def _update_workload_preferences(self, workload_type: str, worker_id: str, efficiency: float) -> None:
        """
        Update preferences for workload types based on selected workers.
        
        Args:
            workload_type: Workload type
            worker_id: Selected worker ID
            efficiency: Efficiency score
        """
        if workload_type not in self.workload_worker_preferences:
            self.workload_worker_preferences[workload_type] = {}
        
        # Update preference with exponential moving average
        alpha = 0.3  # Learning rate
        current = self.workload_worker_preferences[workload_type].get(worker_id, 0.5)
        updated = (1 - alpha) * current + alpha * efficiency
        
        self.workload_worker_preferences[workload_type][worker_id] = updated


# Helper function to convert dictionary to HardwareCapabilityProfile
def hardware_profile_from_dict(profile_dict: Dict[str, Any]) -> HardwareCapabilityProfile:
    """
    Convert a dictionary to a HardwareCapabilityProfile instance.
    
    Args:
        profile_dict: Dictionary with hardware profile attributes
        
    Returns:
        HardwareCapabilityProfile instance
    """
    # Convert string enums to actual enum values
    if "hardware_class" in profile_dict:
        try:
            profile_dict["hardware_class"] = HardwareClass(profile_dict["hardware_class"])
        except ValueError:
            profile_dict["hardware_class"] = HardwareClass.UNKNOWN
    
    # Convert supported backends
    if "supported_backends" in profile_dict:
        backends = set()
        for backend in profile_dict["supported_backends"]:
            try:
                backends.add(SoftwareBackend[backend])
            except (KeyError, TypeError):
                # Skip invalid backends
                pass
        profile_dict["supported_backends"] = backends
    
    # Convert supported precisions
    if "supported_precisions" in profile_dict:
        precisions = set()
        for precision in profile_dict["supported_precisions"]:
            try:
                precisions.add(PrecisionType[precision])
            except (KeyError, TypeError):
                # Skip invalid precisions
                pass
        profile_dict["supported_precisions"] = precisions
    
    # Convert features
    if "features" in profile_dict:
        features = set()
        for feature in profile_dict["features"]:
            try:
                features.add(AcceleratorFeature[feature])
            except (KeyError, TypeError):
                # Skip invalid features
                pass
        profile_dict["features"] = features
    
    # Create HardwareCapabilityProfile using the dictionary
    # This is a simplified approach - in practice, you would need to 
    # handle more fields and ensure all required fields are present
    profile = HardwareCapabilityProfile(
        hardware_class=profile_dict.get("hardware_class", HardwareClass.UNKNOWN),
        architecture=profile_dict.get("architecture", "OTHER"),
        vendor=profile_dict.get("vendor", "OTHER"),
        model_name=profile_dict.get("model_name", "unknown"),
        supported_backends=profile_dict.get("supported_backends", set()),
        supported_precisions=profile_dict.get("supported_precisions", set()),
        features=profile_dict.get("features", set()),
        compute_units=profile_dict.get("compute_units", 0),
        clock_speed_mhz=profile_dict.get("clock_speed_mhz")
    )
    
    # Set memory attributes if provided
    if "memory" in profile_dict:
        memory_dict = profile_dict["memory"]
        if hasattr(profile, "memory"):
            profile.memory.total_bytes = memory_dict.get("total_bytes", 0)
            profile.memory.available_bytes = memory_dict.get("available_bytes", 0)
            profile.memory.is_shared = memory_dict.get("is_shared", False)
            profile.memory.hierarchy_levels = memory_dict.get("hierarchy_levels", 1)
            profile.memory.has_unified_memory = memory_dict.get("has_unified_memory", False)
            profile.memory.memory_type = memory_dict.get("memory_type", "unknown")
    
    return profile


# Helper function to convert dictionary to WorkloadProfile
def workload_profile_from_dict(profile_dict: Dict[str, Any]) -> WorkloadProfile:
    """
    Convert a dictionary to a WorkloadProfile instance.
    
    Args:
        profile_dict: Dictionary with workload profile attributes
        
    Returns:
        WorkloadProfile instance
    """
    # Convert workload type
    workload_type = profile_dict.get("workload_type", "MIXED")
    if isinstance(workload_type, dict) and "value" in workload_type:
        workload_type = workload_type["value"]
    
    try:
        workload_type_enum = WorkloadType[workload_type]
    except (KeyError, TypeError):
        workload_type_enum = WorkloadType.MIXED
    
    # Convert metrics
    metrics = {}
    raw_metrics = profile_dict.get("metrics", {})
    for metric_name, value in raw_metrics.items():
        if isinstance(metric_name, dict) and "value" in metric_name:
            metric_name = metric_name["value"]
        try:
            metric_enum = WorkloadProfileMetric[metric_name]
            metrics[metric_enum] = value
        except (KeyError, TypeError):
            # Skip invalid metrics
            pass
    
    # Create WorkloadProfile
    profile = WorkloadProfile(
        workload_id=profile_dict.get("workload_id", ""),
        workload_type=workload_type_enum,
        min_memory_bytes=profile_dict.get("min_memory_bytes", 0),
        min_compute_units=profile_dict.get("min_compute_units", 0),
        metrics=metrics,
        priority=profile_dict.get("priority", 3),
        is_shardable=profile_dict.get("is_shardable", False),
        min_shards=profile_dict.get("min_shards", 1),
        max_shards=profile_dict.get("max_shards", 1),
        allocation_strategy=profile_dict.get("allocation_strategy", "single"),
        estimated_duration_seconds=profile_dict.get("estimated_duration_seconds"),
        custom_properties=profile_dict.get("custom_properties", {})
    )
    
    return profile