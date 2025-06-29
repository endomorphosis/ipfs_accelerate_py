#!/usr/bin/env python3
"""
Advanced Scheduling Strategies Test Script

This script demonstrates and tests the advanced scheduling strategies for the Distributed Testing Framework,
including historical performance-based scheduling, deadline-aware scheduling, test type-specific scheduling,
and machine learning-based scheduling.
"""

import os
import sys
import time
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
import argparse
import json
from dataclasses import dataclass, field
from enum import Enum

# Mock load balancer models for testing
@dataclass
class TestRequirements:
    """Requirements for a test."""
    test_id: str
    test_type: Optional[str] = None
    model_id: Optional[str] = None
    priority: int = 3
    min_memory_gb: float = 4.0
    min_cpu_cores: int = 2
    required_capabilities: List[str] = field(default_factory=list)
    preferred_backend: Optional[str] = None
    concurrency_key: Optional[str] = None
    expected_duration: Optional[float] = None
    custom_properties: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorkerCapabilities:
    """Capabilities of a worker."""
    worker_id: str
    worker_type: str
    cpu_cores: int
    available_memory: float
    supported_backends: List[str]
    custom_capabilities: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorkerLoad:
    """Current load on a worker."""
    worker_id: str
    current_tests: int
    max_tests: int
    current_cpu_usage: float
    current_memory_usage: float
    warming_state: bool = False
    cooling_state: bool = False
    
    def calculate_load_score(self) -> float:
        """Calculate load score (0.0-1.0)."""
        return min(1.0, self.current_tests / max(1, self.max_tests))
    
    def has_capacity_for(self, test: TestRequirements) -> bool:
        """Check if worker has capacity for a test."""
        return self.current_tests < self.max_tests

@dataclass
class WorkerPerformance:
    """Performance metrics for a worker."""
    worker_id: str
    test_type: str
    avg_execution_time: float
    success_rate: float
    last_updated: datetime = field(default_factory=datetime.now)

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
    """Workload profile."""
    workload_id: str
    workload_type: WorkloadType
    
    def get_efficiency_score(self, hardware) -> float:
        """Mock efficiency score calculation."""
        return random.random()

@dataclass
class HardwareCapabilityProfile:
    """Mock hardware capability profile."""
    hardware_class: Any
    architecture: str
    vendor: str
    model_name: str
    compute_units: int = 0
    
    @property
    def memory(self):
        """Mock memory property."""
        return MemoryProfile()

@dataclass
class MemoryProfile:
    """Mock memory profile."""
    total_bytes: int = 8 * 1024 * 1024 * 1024  # 8 GB
    available_bytes: int = 6 * 1024 * 1024 * 1024  # 6 GB
    is_shared: bool = False
    hierarchy_levels: int = 2
    has_unified_memory: bool = False
    memory_type: str = "DDR4"

class HardwareClass(Enum):
    """Mock hardware class."""
    CPU = "CPU"
    GPU = "GPU"
    TPU = "TPU"
    NPU = "NPU"
    HYBRID = "HYBRID"
    UNKNOWN = "UNKNOWN"

class HardwareType(Enum):
    """Mock hardware type."""
    CPU = "CPU"
    GPU = "GPU"
    TPU = "TPU"
    NPU = "NPU"
    HYBRID = "HYBRID"
    UNKNOWN = "UNKNOWN"

class HardwareTaxonomy:
    """Mock hardware taxonomy."""
    def __init__(self):
        self.hardware_profiles = {}
        self.worker_hardware_map = {}
    
    def register_worker_hardware(self, worker_id, hardware_profiles):
        """Register worker hardware."""
        self.worker_hardware_map[worker_id] = hardware_profiles

class HardwareWorkloadManager:
    """Mock hardware workload manager."""
    def __init__(self, hardware_taxonomy, db_path=None):
        self.hardware_taxonomy = hardware_taxonomy
        self.active_executions = {}
        self.thermal_tracking = {}
    
    def get_compatible_hardware(self, workload_profile):
        """Get compatible hardware for a workload."""
        return [(f"worker_{i}_GPU", HardwareCapabilityProfile(
            hardware_class=HardwareClass.GPU,
            architecture="GPU_CUDA",
            vendor="NVIDIA",
            model_name=f"GPU-{i}",
            compute_units=80
        ), random.random()) for i in range(3)]

def create_workload_profile(**kwargs):
    """Create a workload profile."""
    return WorkloadProfile(
        workload_id=kwargs.get("workload_id", "test"),
        workload_type=kwargs.get("workload_type", WorkloadType.MIXED)
    )

# Mock enhanced hardware taxonomy
class EnhancedHardwareTaxonomy(HardwareTaxonomy):
    """Enhanced hardware taxonomy."""
    pass

@dataclass
class CapabilityDefinition:
    """Mock capability definition."""
    capability_id: str
    capability_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    related_capabilities: List[str] = field(default_factory=list)
    hardware_requirements: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None

@dataclass
class HardwareHierarchy:
    """Mock hardware hierarchy."""
    parent_class: HardwareClass
    child_class: HardwareClass
    inheritance_factor: float = 1.0
    capability_filters: Set[str] = field(default_factory=set)

@dataclass
class HardwareRelationship:
    """Mock hardware relationship."""
    source_id: str
    target_id: str
    relationship_type: str
    strength: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)

# Mock advanced scheduling components for testing
class SchedulingAlgorithm:
    """Base scheduling algorithm."""
    def select_worker(self, test_requirements, available_workers, worker_loads, performance_data):
        """Select a worker for a test."""
        return list(available_workers.keys())[0] if available_workers else None

class HardwareAwareScheduler(SchedulingAlgorithm):
    """Mock hardware-aware scheduler."""
    def __init__(self, hardware_workload_manager, hardware_taxonomy):
        self.workload_manager = hardware_workload_manager
        self.hardware_taxonomy = hardware_taxonomy
        self.test_workload_cache = {}
        self.worker_hardware_cache = {}
        self.workload_worker_preferences = {}
        self.worker_thermal_states = {}
        self.match_factor_weights = {}
        self.test_type_to_hardware_type = {}
    
    def select_worker(self, test_requirements, available_workers, worker_loads, performance_data):
        """Select a worker for a test."""
        return list(available_workers.keys())[0] if available_workers else None
    
    def _test_to_workload_profile(self, test_requirements):
        """Convert test requirements to workload profile."""
        return create_workload_profile(
            workload_id=test_requirements.test_id,
            workload_type=WorkloadType.MIXED
        )
    
    def _update_worker_hardware_profiles(self, available_workers):
        """Update worker hardware profiles."""
        for worker_id, capabilities in available_workers.items():
            self.worker_hardware_cache[worker_id] = [HardwareCapabilityProfile(
                hardware_class=HardwareClass.GPU,
                architecture="GPU_CUDA",
                vendor="NVIDIA",
                model_name=f"GPU-{worker_id}",
                compute_units=80
            )]
    
    def _adjust_efficiency_for_load_and_thermal(self, efficiency, worker_id, load, hardware_profile):
        """Adjust efficiency score for load and thermal state."""
        return efficiency
    
    def _update_workload_preferences(self, workload_type, worker_id, efficiency):
        """Update workload preferences."""
        pass

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

@dataclass
class SchedulingDecision:
    """Record of a scheduling decision for training."""
    test_id: str
    worker_id: str
    features: Dict[str, float]
    execution_time: float
    success: bool
    timestamp: datetime = field(default_factory=datetime.now)

class SchedulingStrategyType(Enum):
    """Types of scheduling strategies."""
    HARDWARE_AWARE = "hardware_aware"
    HISTORICAL_PERFORMANCE = "historical_performance"
    DEADLINE_AWARE = "deadline_aware"
    TEST_TYPE_SPECIFIC = "test_type_specific"
    ML_BASED = "ml_based"

class HistoricalPerformanceScheduler(HardwareAwareScheduler):
    """Historical performance-based scheduler."""
    def __init__(self, hardware_workload_manager, hardware_taxonomy, db_path=None, 
                 performance_history_window=20, performance_weight=0.7, min_history_entries=3):
        super().__init__(hardware_workload_manager, hardware_taxonomy)
        self.db_path = db_path
        self.performance_history = {}
        self.performance_history_window = performance_history_window
        self.performance_weight = performance_weight
        self.min_history_entries = min_history_entries
        self.execution_time_models = {}
    
    def record_performance(self, record):
        """Record performance data."""
        test_id = record.test_id
        if "_" in test_id:
            test_key = test_id.split("_")[0]
        else:
            test_key = test_id
        
        if test_key not in self.performance_history:
            self.performance_history[test_key] = []
        
        self.performance_history[test_key].append(record)
    
    def predict_execution_time(self, test_id, worker_id, hardware_id):
        """Predict execution time."""
        return random.uniform(10, 100)

class DeadlineAwareScheduler(HistoricalPerformanceScheduler):
    """Deadline-aware scheduler."""
    def __init__(self, hardware_workload_manager, hardware_taxonomy, db_path=None, 
                 performance_history_window=20, performance_weight=0.7, min_history_entries=3,
                 deadline_weight=0.8, urgency_threshold=30):
        super().__init__(hardware_workload_manager, hardware_taxonomy, db_path, 
                        performance_history_window, performance_weight, min_history_entries)
        self.test_deadlines = {}
        self.deadline_weight = deadline_weight
        self.urgency_threshold = urgency_threshold
    
    def register_test_deadline(self, deadline):
        """Register a deadline for a test."""
        self.test_deadlines[deadline.test_id] = deadline
    
    def update_test_deadline_status(self, test_id, completed, actual_duration=None):
        """Update the status of a test deadline."""
        if test_id in self.test_deadlines:
            self.test_deadlines[test_id].completed = completed
            if actual_duration is not None:
                self.test_deadlines[test_id].actual_duration = actual_duration

class TestTypeSpecificScheduler(DeadlineAwareScheduler):
    """Test type-specific scheduler."""
    def __init__(self, hardware_workload_manager, hardware_taxonomy, db_path=None, 
                 performance_history_window=20, performance_weight=0.7, min_history_entries=3,
                 deadline_weight=0.8, urgency_threshold=30):
        super().__init__(hardware_workload_manager, hardware_taxonomy, db_path, 
                        performance_history_window, performance_weight, min_history_entries,
                        deadline_weight, urgency_threshold)
        self.test_type_configs = {}

class MLBasedScheduler(TestTypeSpecificScheduler):
    """Machine learning-based scheduler."""
    def __init__(self, hardware_workload_manager, hardware_taxonomy, db_path=None, 
                 performance_history_window=20, performance_weight=0.7, min_history_entries=3,
                 deadline_weight=0.8, urgency_threshold=30, model_path=None):
        super().__init__(hardware_workload_manager, hardware_taxonomy, db_path, 
                        performance_history_window, performance_weight, min_history_entries,
                        deadline_weight, urgency_threshold)
        self.model = {"weights": {}, "bias": 0.0}
        self.scheduling_decisions = []
    
    def record_execution_result(self, test_id, worker_id, execution_time, success):
        """Record execution result."""
        pass

class AdvancedSchedulingStrategyFactory:
    """Factory for creating scheduling strategies."""
    @staticmethod
    def create_scheduler(strategy_type, hardware_workload_manager, hardware_taxonomy, config=None):
        """Create a scheduler of the specified type."""
        if config is None:
            config = {}
        
        if strategy_type == SchedulingStrategyType.HARDWARE_AWARE:
            return HardwareAwareScheduler(hardware_workload_manager, hardware_taxonomy)
        elif strategy_type == SchedulingStrategyType.HISTORICAL_PERFORMANCE:
            return HistoricalPerformanceScheduler(hardware_workload_manager, hardware_taxonomy, config.get("db_path"))
        elif strategy_type == SchedulingStrategyType.DEADLINE_AWARE:
            return DeadlineAwareScheduler(hardware_workload_manager, hardware_taxonomy, config.get("db_path"))
        elif strategy_type == SchedulingStrategyType.TEST_TYPE_SPECIFIC:
            return TestTypeSpecificScheduler(hardware_workload_manager, hardware_taxonomy, config.get("db_path"))
        elif strategy_type == SchedulingStrategyType.ML_BASED:
            return MLBasedScheduler(hardware_workload_manager, hardware_taxonomy, config.get("db_path"))
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("run_test_advanced_scheduling")


class TestEnvironment:
    """
    Test environment for the advanced scheduling strategies.
    
    This class sets up a test environment with simulated workers, hardware capabilities,
    and test requirements to demonstrate and test the advanced scheduling strategies.
    """
    
    def __init__(self, num_workers: int = 5, db_path: Optional[str] = None):
        """
        Initialize the test environment.
        
        Args:
            num_workers: Number of workers to simulate
            db_path: Optional path to database for persistence
        """
        self.num_workers = num_workers
        self.db_path = db_path
        
        # Initialize hardware taxonomy
        self.hardware_taxonomy = EnhancedHardwareTaxonomy()
        
        # Initialize hardware workload manager
        self.workload_manager = HardwareWorkloadManager(self.hardware_taxonomy, db_path)
        
        # Initialize scheduler factory
        self.scheduler_factory = AdvancedSchedulingStrategyFactory()
        
        # Initialize simulated workers
        self.workers = self._create_simulated_workers(num_workers)
        
        # Initialize simulated tests
        self.test_requirements = self._create_simulated_tests()
        
        # Performance tracking
        self.worker_performance = {}  # worker_id -> test_type -> WorkerPerformance
        
        # Execution tracking
        self.execution_history = []  # list of execution records
        
        logger.info(f"Test environment initialized with {num_workers} workers")
    
    def _create_simulated_workers(self, num_workers: int) -> Dict[str, WorkerCapabilities]:
        """
        Create simulated workers with different capabilities.
        
        Args:
            num_workers: Number of workers to create
            
        Returns:
            Dictionary of worker_id -> WorkerCapabilities
        """
        workers = {}
        
        # Create workers with different capabilities
        for i in range(num_workers):
            worker_id = f"worker_{i}"
            
            # Randomize capabilities to create diversity
            cpu_cores = random.choice([4, 8, 16, 32])
            available_memory = random.choice([8, 16, 32, 64])
            
            # Determine hardware type
            if i % 5 == 0:
                # CPU-only worker
                supported_backends = ["pytorch", "tensorflow", "onnx"]
                worker_type = "cpu"
            elif i % 5 == 1:
                # GPU worker
                supported_backends = ["pytorch", "tensorflow", "onnx", "cuda"]
                worker_type = "gpu"
            elif i % 5 == 2:
                # TPU worker
                supported_backends = ["tensorflow", "tpu"]
                worker_type = "tpu"
            elif i % 5 == 3:
                # Web worker (WebGPU)
                supported_backends = ["webgpu"]
                worker_type = "webgpu"
            else:
                # Hybrid worker
                supported_backends = ["pytorch", "tensorflow", "onnx", "cuda", "webgpu"]
                worker_type = "hybrid"
            
            # Create worker capabilities
            capabilities = WorkerCapabilities(
                worker_id=worker_id,
                worker_type=worker_type,
                cpu_cores=cpu_cores,
                available_memory=available_memory,
                supported_backends=supported_backends,
                custom_capabilities={
                    "hardware_class": worker_type.upper(),
                    "supports_sharding": worker_type in ["gpu", "tpu", "hybrid"],
                    "supports_webgpu": "webgpu" in supported_backends,
                    "max_concurrent_tests": random.randint(2, 6)
                }
            )
            
            workers[worker_id] = capabilities
            
            # Create worker load
            load = WorkerLoad(
                worker_id=worker_id,
                current_tests=0,
                max_tests=random.randint(4, 10),
                current_cpu_usage=random.random() * 0.3,  # Initial low CPU usage
                current_memory_usage=random.random() * 0.2 * available_memory,  # Initial low memory usage
                warming_state=False,
                cooling_state=False
            )
            
            # Add to workload manager
            self.workload_manager.active_executions[worker_id] = set()
            
            # Add thermal state
            self.workload_manager.thermal_tracking[worker_id] = {
                "temperature": random.random() * 0.3,  # Initial low temperature
                "last_updated": datetime.now()
            }
        
        return workers
    
    def _create_simulated_tests(self) -> List[TestRequirements]:
        """
        Create simulated test requirements with different characteristics.
        
        Returns:
            List of TestRequirements
        """
        test_requirements = []
        
        # Test types
        test_types = [
            "compute_intensive", "memory_intensive", "io_intensive",
            "network_intensive", "latency_sensitive", "throughput_oriented",
            "webgpu_inference", "webgpu_training"
        ]
        
        # Create tests with different characteristics
        for i in range(50):
            test_id = f"test_{i}"
            test_type = random.choice(test_types)
            
            # Randomize requirements based on test type
            if "compute_intensive" in test_type:
                min_memory = random.choice([2, 4, 8])
                min_cpu_cores = random.randint(4, 16)
                expected_duration = random.randint(60, 300)
                preferred_backend = random.choice(["cuda", "pytorch", "tensorflow"])
            elif "memory_intensive" in test_type:
                min_memory = random.choice([16, 32, 64])
                min_cpu_cores = random.randint(2, 8)
                expected_duration = random.randint(30, 180)
                preferred_backend = random.choice(["pytorch", "tensorflow", None])
            elif "webgpu" in test_type:
                min_memory = random.choice([2, 4, 8])
                min_cpu_cores = random.randint(2, 4)
                expected_duration = random.randint(20, 120)
                preferred_backend = "webgpu"
            else:
                min_memory = random.choice([2, 4, 8, 16])
                min_cpu_cores = random.randint(2, 8)
                expected_duration = random.randint(10, 120)
                preferred_backend = random.choice(["pytorch", "tensorflow", "onnx", None])
            
            # Create test requirements
            requirements = TestRequirements(
                test_id=test_id,
                test_type=test_type,
                model_id=f"model_{i % 10}",
                priority=random.randint(1, 5),  # 1 = highest, 5 = lowest
                min_memory_gb=min_memory,
                min_cpu_cores=min_cpu_cores,
                required_capabilities=[],
                preferred_backend=preferred_backend,
                expected_duration=expected_duration,
                custom_properties={
                    "is_shardable": random.random() > 0.7,
                    "batch_size": random.choice([1, 2, 4, 8, 16]),
                    "workload_profile": {
                        "workload_type": test_type.upper(),
                        "metrics": {
                            "COMPUTE_INTENSITY": random.random(),
                            "MEMORY_INTENSITY": random.random(),
                            "LATENCY_SENSITIVITY": random.random(),
                            "THROUGHPUT_SENSITIVITY": random.random()
                        }
                    }
                }
            )
            
            test_requirements.append(requirements)
        
        return test_requirements
    
    def create_worker_loads(self) -> Dict[str, WorkerLoad]:
        """
        Create worker loads based on current state.
        
        Returns:
            Dictionary of worker_id -> WorkerLoad
        """
        worker_loads = {}
        
        for worker_id, capabilities in self.workers.items():
            # Get current executions
            current_tests = len(self.workload_manager.active_executions.get(worker_id, set()))
            
            # Get max tests from capabilities
            max_tests = capabilities.custom_capabilities.get("max_concurrent_tests", 5)
            
            # Calculate current CPU and memory usage
            cpu_usage = min(0.2 + (current_tests / max_tests) * 0.7 + random.random() * 0.1, 1.0)
            memory_usage = min(0.1 + (current_tests / max_tests) * 0.8 + random.random() * 0.1, 1.0) * capabilities.available_memory
            
            # Create worker load
            load = WorkerLoad(
                worker_id=worker_id,
                current_tests=current_tests,
                max_tests=max_tests,
                current_cpu_usage=cpu_usage,
                current_memory_usage=memory_usage,
                warming_state=False,
                cooling_state=random.random() > 0.95  # Occasionally in cooling state
            )
            
            # Add custom method for load score calculation
            load.calculate_load_score = lambda: min(current_tests / max_tests, 0.95)
            
            # Add custom method for capacity check
            def has_capacity_for(test_req):
                return current_tests < max_tests and memory_usage + test_req.min_memory_gb < capabilities.available_memory * 0.9
            
            load.has_capacity_for = has_capacity_for
            
            worker_loads[worker_id] = load
            
            # Update thermal state
            if worker_id in self.workload_manager.thermal_tracking:
                # Increase temperature based on load
                current_temp = self.workload_manager.thermal_tracking[worker_id]["temperature"]
                new_temp = min(current_temp + (current_tests / max_tests) * 0.1 + random.random() * 0.05, 1.0)
                
                # Cool down slightly if in cooling state
                if load.cooling_state:
                    new_temp = max(0.1, new_temp - 0.2)
                
                self.workload_manager.thermal_tracking[worker_id]["temperature"] = new_temp
                self.workload_manager.thermal_tracking[worker_id]["last_updated"] = datetime.now()
        
        return worker_loads
    
    def test_hardware_aware_scheduler(self) -> None:
        """Test the base hardware-aware scheduler."""
        logger.info("Testing Hardware-Aware Scheduler")
        
        # Create scheduler
        scheduler = self.scheduler_factory.create_scheduler(
            SchedulingStrategyType.HARDWARE_AWARE,
            self.workload_manager,
            self.hardware_taxonomy
        )
        
        # Run scheduling test
        self._run_scheduling_test(scheduler, "hardware_aware", num_tests=10)
    
    def test_historical_performance_scheduler(self) -> None:
        """Test the historical performance-based scheduler."""
        logger.info("Testing Historical Performance Scheduler")
        
        # Create scheduler
        scheduler = self.scheduler_factory.create_scheduler(
            SchedulingStrategyType.HISTORICAL_PERFORMANCE,
            self.workload_manager,
            self.hardware_taxonomy,
            {"db_path": self.db_path}
        )
        
        # Add some historical performance data
        self._add_simulated_performance_history(scheduler)
        
        # Run scheduling test
        self._run_scheduling_test(scheduler, "historical_performance", num_tests=10)
    
    def test_deadline_aware_scheduler(self) -> None:
        """Test the deadline-aware scheduler."""
        logger.info("Testing Deadline-Aware Scheduler")
        
        # Create scheduler
        scheduler = self.scheduler_factory.create_scheduler(
            SchedulingStrategyType.DEADLINE_AWARE,
            self.workload_manager,
            self.hardware_taxonomy,
            {"db_path": self.db_path}
        )
        
        # Add some historical performance data
        self._add_simulated_performance_history(scheduler)
        
        # Add deadlines to tests
        self._add_simulated_deadlines(scheduler)
        
        # Run scheduling test
        self._run_scheduling_test(scheduler, "deadline_aware", num_tests=10)
    
    def test_test_type_specific_scheduler(self) -> None:
        """Test the test type-specific scheduler."""
        logger.info("Testing Test Type-Specific Scheduler")
        
        # Create scheduler
        scheduler = self.scheduler_factory.create_scheduler(
            SchedulingStrategyType.TEST_TYPE_SPECIFIC,
            self.workload_manager,
            self.hardware_taxonomy,
            {"db_path": self.db_path}
        )
        
        # Add some historical performance data
        self._add_simulated_performance_history(scheduler)
        
        # Add deadlines to tests
        self._add_simulated_deadlines(scheduler)
        
        # Run scheduling test
        self._run_scheduling_test(scheduler, "test_type_specific", num_tests=10)
    
    def test_ml_based_scheduler(self) -> None:
        """Test the machine learning-based scheduler."""
        logger.info("Testing Machine Learning-Based Scheduler")
        
        # Create scheduler
        scheduler = self.scheduler_factory.create_scheduler(
            SchedulingStrategyType.ML_BASED,
            self.workload_manager,
            self.hardware_taxonomy,
            {"db_path": self.db_path}
        )
        
        # Add some historical performance data
        self._add_simulated_performance_history(scheduler)
        
        # Add deadlines to tests
        self._add_simulated_deadlines(scheduler)
        
        # Add training data
        self._add_simulated_training_data(scheduler)
        
        # Run scheduling test
        self._run_scheduling_test(scheduler, "ml_based", num_tests=10)
    
    def _run_scheduling_test(self, scheduler, scheduler_type: str, num_tests: int = 10) -> List[Dict[str, Any]]:
        """
        Run a scheduling test with the given scheduler.
        
        Args:
            scheduler: Scheduler to test
            scheduler_type: Type of scheduler for logging
            num_tests: Number of tests to schedule
            
        Returns:
            List of scheduling results
        """
        results = []
        
        # Use a subset of tests
        tests = random.sample(self.test_requirements, min(num_tests, len(self.test_requirements)))
        
        # Update worker loads
        worker_loads = self.create_worker_loads()
        
        for test in tests:
            # Select worker
            start_time = time.time()
            selected_worker = scheduler.select_worker(
                test,
                self.workers,
                worker_loads,
                self.worker_performance
            )
            end_time = time.time()
            
            # Log result
            if selected_worker:
                result = {
                    "test_id": test.test_id,
                    "test_type": test.test_type,
                    "worker_id": selected_worker,
                    "worker_type": self.workers[selected_worker].worker_type,
                    "decision_time_ms": (end_time - start_time) * 1000,
                    "scheduler_type": scheduler_type
                }
                
                logger.info(f"[{scheduler_type}] Selected worker {selected_worker} ({self.workers[selected_worker].worker_type}) "
                          f"for test {test.test_id} ({test.test_type})")
                
                # Update worker load
                worker_loads[selected_worker].current_tests += 1
                
                # Add to active executions
                if selected_worker not in self.workload_manager.active_executions:
                    self.workload_manager.active_executions[selected_worker] = set()
                self.workload_manager.active_executions[selected_worker].add(test.test_id)
                
                # Simulate execution and record performance
                execution_time, success = self._simulate_test_execution(test, selected_worker)
                
                result["execution_time"] = execution_time
                result["success"] = success
                
                # Record performance for the scheduler
                if hasattr(scheduler, 'record_performance'):
                    record = HistoricalPerformanceRecord(
                        test_id=test.test_id,
                        worker_id=selected_worker,
                        hardware_id=f"{selected_worker}_{self.workers[selected_worker].worker_type}",
                        execution_time=execution_time,
                        memory_usage=test.min_memory_gb,
                        success=success
                    )
                    scheduler.record_performance(record)
                
                # Record execution result for ML scheduler
                if hasattr(scheduler, 'record_execution_result'):
                    scheduler.record_execution_result(
                        test.test_id,
                        selected_worker,
                        execution_time,
                        success
                    )
                
                # Update deadline status if applicable
                if hasattr(scheduler, 'update_test_deadline_status'):
                    scheduler.update_test_deadline_status(
                        test.test_id,
                        success,
                        execution_time
                    )
                
                results.append(result)
            else:
                logger.warning(f"[{scheduler_type}] No suitable worker found for test {test.test_id} ({test.test_type})")
        
        return results
    
    def _add_simulated_performance_history(self, scheduler) -> None:
        """
        Add simulated performance history to a scheduler.
        
        Args:
            scheduler: Scheduler to add history to
        """
        if not hasattr(scheduler, 'record_performance'):
            return
        
        logger.info("Adding simulated performance history")
        
        # Generate some history for each test type
        for test_type in set(req.test_type for req in self.test_requirements):
            # Find tests of this type
            type_tests = [req for req in self.test_requirements if req.test_type == test_type]
            
            if not type_tests:
                continue
            
            # Generate history for each worker
            for worker_id, capabilities in self.workers.items():
                # Determine base execution time for this worker-test combination
                if capabilities.worker_type == "gpu" and "compute_intensive" in test_type:
                    # GPU good for compute-intensive
                    base_time = random.uniform(30, 60)
                    success_prob = 0.95
                elif capabilities.worker_type == "tpu" and "compute_intensive" in test_type:
                    # TPU great for compute-intensive
                    base_time = random.uniform(20, 40)
                    success_prob = 0.98
                elif capabilities.worker_type == "webgpu" and "webgpu" in test_type:
                    # WebGPU good for WebGPU tests
                    base_time = random.uniform(15, 30)
                    success_prob = 0.9
                elif capabilities.worker_type == "cpu" and "memory_intensive" in test_type:
                    # CPU decent for memory-intensive
                    base_time = random.uniform(40, 80)
                    success_prob = 0.85
                else:
                    # Default case
                    base_time = random.uniform(50, 100)
                    success_prob = 0.8
                
                # Generate 5-10 historical records
                for _ in range(random.randint(5, 10)):
                    test = random.choice(type_tests)
                    
                    # Apply random variation
                    execution_time = base_time * (0.8 + random.random() * 0.4)
                    success = random.random() < success_prob
                    
                    # Create record
                    record = HistoricalPerformanceRecord(
                        test_id=test.test_id,
                        worker_id=worker_id,
                        hardware_id=f"{worker_id}_{capabilities.worker_type}",
                        execution_time=execution_time,
                        memory_usage=test.min_memory_gb,
                        success=success,
                        timestamp=datetime.now() - timedelta(hours=random.randint(1, 24))
                    )
                    
                    # Add to scheduler
                    scheduler.record_performance(record)
    
    def _add_simulated_deadlines(self, scheduler) -> None:
        """
        Add simulated deadlines to a scheduler.
        
        Args:
            scheduler: Scheduler to add deadlines to
        """
        if not hasattr(scheduler, 'register_test_deadline'):
            return
        
        logger.info("Adding simulated deadlines")
        
        # Add deadlines to some tests
        for test in self.test_requirements[:20]:  # Add to first 20 tests
            # Randomly assign deadline
            minutes_to_deadline = random.randint(5, 120)
            deadline_time = datetime.now() + timedelta(minutes=minutes_to_deadline)
            
            # Randomize priority (1-5, 1 = highest)
            priority = random.randint(1, 5)
            
            # Create deadline
            deadline = TestDeadline(
                test_id=test.test_id,
                deadline=deadline_time,
                priority=priority,
                estimated_duration=test.expected_duration
            )
            
            # Register with scheduler
            scheduler.register_test_deadline(deadline)
    
    def _add_simulated_training_data(self, scheduler) -> None:
        """
        Add simulated training data to an ML-based scheduler.
        
        Args:
            scheduler: Scheduler to add training data to
        """
        if not hasattr(scheduler, 'record_execution_result'):
            return
        
        logger.info("Adding simulated training data")
        
        # Add some simulated decisions and results
        for i in range(100):
            # Random test and worker
            test = random.choice(self.test_requirements)
            worker_id = random.choice(list(self.workers.keys()))
            capabilities = self.workers[worker_id]
            
            # Create features
            features = {
                "test_type": float(hash(test.test_type) % 10000) / 10000.0,
                "test_priority": float(test.priority) / 5.0,
                "hardware_type": float(hash(capabilities.worker_type) % 10000) / 10000.0,
                "compute_units": float(capabilities.cpu_cores) / 32.0,
                "memory_gb": float(capabilities.available_memory) / 64.0,
                "worker_load": random.random(),
                "temperature": random.random(),
                "base_efficiency": random.random(),
                "deadline_pressure": random.random() if i % 4 == 0 else 0.0,
                "historical_success_rate": 0.5 + random.random() * 0.5,
                "historical_avg_time": random.random()
            }
            
            # Record decision
            self._record_scheduling_decision(scheduler, test.test_id, worker_id, features)
            
            # Simulate execution
            execution_time, success = self._simulate_test_execution(test, worker_id)
            
            # Record result
            scheduler.record_execution_result(
                test.test_id,
                worker_id,
                execution_time,
                success
            )
    
    def _record_scheduling_decision(self, scheduler, test_id: str, worker_id: str, features: Dict[str, float]) -> None:
        """
        Record a scheduling decision for an ML-based scheduler.
        
        Args:
            scheduler: ML-based scheduler
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
        
        # Add to scheduler's decisions
        if hasattr(scheduler, 'scheduling_decisions'):
            scheduler.scheduling_decisions.append(decision)
    
    def _simulate_test_execution(self, test: TestRequirements, worker_id: str) -> Tuple[float, bool]:
        """
        Simulate test execution and return results.
        
        Args:
            test: Test requirements
            worker_id: ID of the worker executing the test
            
        Returns:
            Tuple of (execution_time, success)
        """
        capabilities = self.workers[worker_id]
        
        # Determine base execution time
        if capabilities.worker_type == "gpu" and "compute_intensive" in test.test_type:
            # GPU good for compute-intensive
            base_time = test.expected_duration * 0.6
            success_prob = 0.95
        elif capabilities.worker_type == "tpu" and "compute_intensive" in test.test_type:
            # TPU great for compute-intensive
            base_time = test.expected_duration * 0.4
            success_prob = 0.98
        elif capabilities.worker_type == "webgpu" and "webgpu" in test.test_type:
            # WebGPU good for WebGPU tests
            base_time = test.expected_duration * 0.5
            success_prob = 0.9
        elif capabilities.worker_type == "cpu" and "memory_intensive" in test.test_type:
            # CPU decent for memory-intensive
            base_time = test.expected_duration * 0.8
            success_prob = 0.85
        else:
            # Default case
            base_time = test.expected_duration
            success_prob = 0.8
        
        # Apply random variation
        execution_time = base_time * (0.8 + random.random() * 0.4)
        success = random.random() < success_prob
        
        # Record history
        execution_record = {
            "test_id": test.test_id,
            "test_type": test.test_type,
            "worker_id": worker_id,
            "worker_type": capabilities.worker_type,
            "execution_time": execution_time,
            "success": success,
            "timestamp": datetime.now()
        }
        
        self.execution_history.append(execution_record)
        
        return execution_time, success
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """
        Run comprehensive tests for all schedulers.
        
        Returns:
            Test results for comparison
        """
        results = {}
        
        # Test each scheduler
        results["hardware_aware"] = self.test_hardware_aware_scheduler()
        results["historical_performance"] = self.test_historical_performance_scheduler()
        results["deadline_aware"] = self.test_deadline_aware_scheduler()
        results["test_type_specific"] = self.test_test_type_specific_scheduler()
        results["ml_based"] = self.test_ml_based_scheduler()
        
        # Analyze and compare results
        comparison = self._compare_scheduler_results(results)
        
        return {
            "results": results,
            "comparison": comparison,
            "execution_history": self.execution_history
        }
    
    def _compare_scheduler_results(self, results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Compare results from different schedulers.
        
        Args:
            results: Results from each scheduler
            
        Returns:
            Comparison metrics
        """
        comparison = {}
        
        # Calculate average execution time for each scheduler
        for scheduler_type, scheduler_results in results.items():
            if not scheduler_results:
                # Initialize with empty results if none
                scheduler_results = []
            
            # Calculate average execution time
            total_time = sum(result.get("execution_time", 0) for result in scheduler_results)
            avg_time = total_time / len(scheduler_results) if scheduler_results else 0
            
            # Calculate success rate
            success_count = sum(1 for result in scheduler_results if result.get("success", False))
            success_rate = success_count / len(scheduler_results) if scheduler_results else 0
            
            # Calculate average decision time
            total_decision_time = sum(result.get("decision_time_ms", 0) for result in scheduler_results)
            avg_decision_time = total_decision_time / len(scheduler_results) if scheduler_results else 0
            
            # Store metrics
            comparison[scheduler_type] = {
                "avg_execution_time": avg_time,
                "success_rate": success_rate,
                "avg_decision_time_ms": avg_decision_time,
                "test_count": len(scheduler_results)
            }
        
        return comparison


def run_tests():
    """Run the advanced scheduling strategy tests."""
    parser = argparse.ArgumentParser(description='Test Advanced Scheduling Strategies')
    parser.add_argument('--workers', type=int, default=5, help='Number of workers to simulate')
    parser.add_argument('--db-path', type=str, default=None, help='Path to database for persistence')
    parser.add_argument('--output', type=str, default='scheduling_results.json', help='Output file for results')
    args = parser.parse_args()
    
    logger.info(f"Running advanced scheduling tests with {args.workers} workers")
    
    # Create test environment
    env = TestEnvironment(num_workers=args.workers, db_path=args.db_path)
    
    # Run comprehensive test
    results = env.run_comprehensive_test()
    
    # Save results
    with open(args.output, 'w') as f:
        # Convert datetime objects to strings
        for scheduler_type, scheduler_results in results["results"].items():
            if scheduler_results:  # Check if not None
                for result in scheduler_results:
                    # Convert any datetime objects to strings
                    for key, value in list(result.items()):
                        if isinstance(value, datetime):
                            result[key] = value.isoformat()
        
        for record in results["execution_history"]:
            for key, value in list(record.items()):
                if isinstance(value, datetime):
                    record[key] = value.isoformat()
        
        # Convert to JSON-serializable format
        json_results = {
            "results": {k: (v if v is not None else []) for k, v in results["results"].items()},
            "comparison": results["comparison"],
            "execution_history": results["execution_history"]
        }
        
        json.dump(json_results, f, indent=2)
    
    logger.info(f"Test results saved to {args.output}")
    
    # Print comparison
    print("\nScheduler Comparison:")
    print("-" * 80)
    print(f"{'Scheduler':<20} {'Avg Time (s)':<15} {'Success Rate':<15} {'Decision Time (ms)':<20}")
    print("-" * 80)
    
    for scheduler_type, metrics in results["comparison"].items():
        try:
            print(f"{scheduler_type:<20} {metrics.get('avg_execution_time', 0):<15.2f} {metrics.get('success_rate', 0):<15.2f} {metrics.get('avg_decision_time_ms', 0):<20.2f}")
        except Exception as e:
            print(f"{scheduler_type:<20} {'N/A':<15} {'N/A':<15} {'N/A':<20}")
    
    print("-" * 80)


if __name__ == "__main__":
    run_tests()