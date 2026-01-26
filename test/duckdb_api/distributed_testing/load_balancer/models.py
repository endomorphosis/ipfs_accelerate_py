#!/usr/bin/env python3
"""
Distributed Testing Framework - Load Balancer Data Models

This module contains the data models for the adaptive load balancing system
in the Distributed Testing Framework.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
import json


@dataclass
class WorkerCapabilities:
    """Worker hardware and software capabilities."""
    worker_id: str
    hostname: str
    hardware_specs: Dict[str, Any] = field(default_factory=dict)  # CPU, GPU, memory, etc.
    software_versions: Dict[str, str] = field(default_factory=dict)  # Python, libraries, etc.
    supported_backends: List[str] = field(default_factory=list)  # CUDA, CPU, etc.
    network_bandwidth: float = 0.0  # Mbps
    storage_capacity: float = 0.0  # GB
    available_accelerators: Dict[str, int] = field(default_factory=dict)  # Type: count
    available_memory: float = 0.0  # GB
    available_disk: float = 0.0  # GB
    cpu_cores: int = 0
    cpu_threads: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "worker_id": self.worker_id,
            "hostname": self.hostname,
            "hardware_specs": self.hardware_specs,
            "software_versions": self.software_versions,
            "supported_backends": self.supported_backends,
            "network_bandwidth": self.network_bandwidth,
            "storage_capacity": self.storage_capacity,
            "available_accelerators": self.available_accelerators,
            "available_memory": self.available_memory,
            "available_disk": self.available_disk,
            "cpu_cores": self.cpu_cores,
            "cpu_threads": self.cpu_threads,
            "last_updated": self.last_updated.isoformat()
        }
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkerCapabilities':
        """Create instance from dictionary."""
        # Convert string timestamp back to datetime
        if isinstance(data.get("last_updated"), str):
            data["last_updated"] = datetime.fromisoformat(data["last_updated"])
        return cls(**data)
    
    def is_compatible_with(self, requirements: 'TestRequirements') -> bool:
        """Check if this worker is compatible with the given test requirements."""
        # Check minimum memory
        if self.available_memory < requirements.minimum_memory:
            return False
            
        # Check backend support
        if (requirements.required_backend and 
            requirements.required_backend not in self.supported_backends):
            return False
            
        # Check for required accelerators
        for accel_type, count in requirements.required_accelerators.items():
            if accel_type not in self.available_accelerators:
                return False
            if self.available_accelerators.get(accel_type, 0) < count:
                return False
                
        # Check for required software
        for sw_name, min_version in requirements.required_software.items():
            if sw_name not in self.software_versions:
                return False
            # TODO: Implement version comparison logic
                
        return True


@dataclass
class WorkerPerformance:
    """Worker performance history for a specific test type."""
    worker_id: str
    test_type: str
    model_id: Optional[str] = None
    model_family: Optional[str] = None
    hardware_type: Optional[str] = None
    average_execution_time: float = 0.0  # seconds
    success_rate: float = 1.0  # 0.0 to 1.0
    last_execution_time: datetime = field(default_factory=datetime.now)
    sample_count: int = 0
    min_execution_time: float = 0.0  # seconds
    max_execution_time: float = 0.0  # seconds
    std_execution_time: float = 0.0  # seconds
    total_failures: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "worker_id": self.worker_id,
            "test_type": self.test_type,
            "model_id": self.model_id,
            "model_family": self.model_family,
            "hardware_type": self.hardware_type,
            "average_execution_time": self.average_execution_time,
            "success_rate": self.success_rate,
            "last_execution_time": self.last_execution_time.isoformat(),
            "sample_count": self.sample_count,
            "min_execution_time": self.min_execution_time,
            "max_execution_time": self.max_execution_time,
            "std_execution_time": self.std_execution_time,
            "total_failures": self.total_failures
        }
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkerPerformance':
        """Create instance from dictionary."""
        # Convert string timestamp back to datetime
        if isinstance(data.get("last_execution_time"), str):
            data["last_execution_time"] = datetime.fromisoformat(data["last_execution_time"])
        return cls(**data)
    
    def update_with_result(self, execution_time: float, success: bool) -> None:
        """Update performance metrics with a new test result."""
        # Update sample count
        self.sample_count += 1
        
        if success:
            # Update execution time statistics
            if self.sample_count == 1:
                # First sample
                self.average_execution_time = execution_time
                self.min_execution_time = execution_time
                self.max_execution_time = execution_time
                self.std_execution_time = 0.0
            else:
                # Update min/max
                self.min_execution_time = min(self.min_execution_time, execution_time)
                self.max_execution_time = max(self.max_execution_time, execution_time)
                
                # Update average (rolling)
                old_avg = self.average_execution_time
                self.average_execution_time = old_avg + (execution_time - old_avg) / self.sample_count
                
                # Update standard deviation (approximation for rolling calculation)
                # This is a simplification and not statistically perfect
                if self.sample_count > 1:
                    self.std_execution_time = ((self.std_execution_time ** 2) * (self.sample_count - 2) + 
                                            (execution_time - old_avg) * (execution_time - self.average_execution_time)) / (self.sample_count - 1)
                    self.std_execution_time = max(0.0, self.std_execution_time) ** 0.5
        else:
            # Update failure count
            self.total_failures += 1
            
        # Update success rate
        self.success_rate = (self.sample_count - self.total_failures) / self.sample_count
        
        # Update timestamp
        self.last_execution_time = datetime.now()


@dataclass
class WorkerLoad:
    """Current worker load status."""
    worker_id: str
    active_tests: int = 0
    queued_tests: int = 0
    cpu_utilization: float = 0.0  # percentage
    memory_utilization: float = 0.0  # percentage
    gpu_utilization: float = 0.0  # percentage
    io_utilization: float = 0.0  # percentage
    network_utilization: float = 0.0  # percentage
    queue_depth: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    active_test_ids: Set[str] = field(default_factory=set)
    reserved_memory: float = 0.0  # GB
    reserved_accelerators: Dict[str, int] = field(default_factory=dict)  # Type: count
    warming_state: bool = False  # Worker is warming up
    cooling_state: bool = False  # Worker is cooling down
    warming_until: Optional[datetime] = None  # Time until warmed
    cooling_until: Optional[datetime] = None  # Time until cooled
    performance_level: float = 1.0  # Performance level (0.0-1.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "worker_id": self.worker_id,
            "active_tests": self.active_tests,
            "queued_tests": self.queued_tests,
            "cpu_utilization": self.cpu_utilization,
            "memory_utilization": self.memory_utilization,
            "gpu_utilization": self.gpu_utilization,
            "io_utilization": self.io_utilization,
            "network_utilization": self.network_utilization,
            "queue_depth": self.queue_depth,
            "last_updated": self.last_updated.isoformat(),
            "active_test_ids": list(self.active_test_ids),
            "reserved_memory": self.reserved_memory,
            "reserved_accelerators": self.reserved_accelerators,
            "warming_state": self.warming_state,
            "cooling_state": self.cooling_state,
            "warming_until": self.warming_until.isoformat() if self.warming_until else None,
            "cooling_until": self.cooling_until.isoformat() if self.cooling_until else None,
            "performance_level": self.performance_level
        }
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkerLoad':
        """Create instance from dictionary."""
        # Convert string timestamp back to datetime
        if isinstance(data.get("last_updated"), str):
            data["last_updated"] = datetime.fromisoformat(data["last_updated"])
            
        # Convert warming_until timestamp
        if isinstance(data.get("warming_until"), str) and data["warming_until"]:
            data["warming_until"] = datetime.fromisoformat(data["warming_until"])
            
        # Convert cooling_until timestamp
        if isinstance(data.get("cooling_until"), str) and data["cooling_until"]:
            data["cooling_until"] = datetime.fromisoformat(data["cooling_until"])
            
        # Convert active_test_ids from list to set
        if "active_test_ids" in data and isinstance(data["active_test_ids"], list):
            data["active_test_ids"] = set(data["active_test_ids"])
            
        return cls(**data)
    
    def calculate_load_score(self) -> float:
        """Calculate a composite load score (0.0 to 1.0).
        
        Higher score means higher load (less available capacity).
        """
        # Simple weighted average of different utilization metrics
        weights = {
            "cpu": 0.3,
            "memory": 0.3,
            "gpu": 0.2,
            "io": 0.1,
            "network": 0.1
        }
        
        score = (
            weights["cpu"] * self.cpu_utilization / 100.0 +
            weights["memory"] * self.memory_utilization / 100.0 +
            weights["gpu"] * self.gpu_utilization / 100.0 +
            weights["io"] * self.io_utilization / 100.0 +
            weights["network"] * self.network_utilization / 100.0
        )
        
        return max(0.0, min(1.0, score))
    
    def has_capacity_for(self, requirements: 'TestRequirements', worker_capabilities: Optional['WorkerCapabilities'] = None) -> bool:
        """Check if this worker has enough free capacity for the test."""
        # Check if we have room for another test
        if self.calculate_load_score() > 0.9:  # Threshold can be configurable
            return False
            
        # Check memory - compare with actual available memory if worker_capabilities provided
        if worker_capabilities:
            # Check against actual worker available memory
            if self.reserved_memory + requirements.minimum_memory > worker_capabilities.available_memory:
                return False
        else:
            # No worker_capabilities provided, use the requirement's limit as a fallback
            # This is less accurate but prevents errors when capabilities are not available
            if self.reserved_memory + requirements.minimum_memory > requirements.required_memory_limit:
                return False
            
        # Check accelerators
        for accel_type, count in requirements.required_accelerators.items():
            reserved = self.reserved_accelerators.get(accel_type, 0)
            
            if worker_capabilities and accel_type in worker_capabilities.available_accelerators:
                # Check against actual worker available accelerators
                if reserved + count > worker_capabilities.available_accelerators.get(accel_type, 0):
                    return False
            else:
                # Use the requirement's limit as a fallback
                if reserved + count > requirements.required_accelerator_limit.get(accel_type, 0):
                    return False
                
        return True
    
    def reserve_resources(self, test_id: str, requirements: 'TestRequirements', 
                           worker_capabilities: Optional['WorkerCapabilities'] = None) -> bool:
        """Reserve resources for a test execution. Returns success/failure."""
        if not self.has_capacity_for(requirements, worker_capabilities):
            return False
            
        # Reserve memory
        self.reserved_memory += requirements.minimum_memory
        
        # Reserve accelerators
        for accel_type, count in requirements.required_accelerators.items():
            self.reserved_accelerators[accel_type] = self.reserved_accelerators.get(accel_type, 0) + count
            
        # Update active tests
        self.active_tests += 1
        self.active_test_ids.add(test_id)
        
        # Update timestamp
        self.last_updated = datetime.now()
        
        return True
    
    def release_resources(self, test_id: str, requirements: 'TestRequirements') -> None:
        """Release resources after test completion."""
        if test_id not in self.active_test_ids:
            return
            
        # Release memory
        self.reserved_memory = max(0.0, self.reserved_memory - requirements.minimum_memory)
        
        # Release accelerators
        for accel_type, count in requirements.required_accelerators.items():
            current = self.reserved_accelerators.get(accel_type, 0)
            self.reserved_accelerators[accel_type] = max(0, current - count)
            
        # Update active tests
        self.active_tests = max(0, self.active_tests - 1)
        self.active_test_ids.remove(test_id)
        
        # Update timestamp
        self.last_updated = datetime.now()
        
    def start_warming(self, duration_seconds: float = 30.0) -> None:
        """Start warming up the worker.
        
        Args:
            duration_seconds: Duration of warm-up period in seconds
        """
        self.warming_state = True
        self.cooling_state = False
        self.warming_until = datetime.now() + timedelta(seconds=duration_seconds)
        self.cooling_until = None
        
        # Start with reduced performance that will gradually increase
        self.performance_level = 0.6
        
    def start_cooling(self, duration_seconds: float = 60.0) -> None:
        """Start cooling down the worker.
        
        Args:
            duration_seconds: Duration of cool-down period in seconds
        """
        self.cooling_state = True
        self.warming_state = False
        self.cooling_until = datetime.now() + timedelta(seconds=duration_seconds)
        self.warming_until = None
        
    def update_thermal_state(self) -> None:
        """Update thermal state (warming/cooling) based on time elapsed.
        
        This should be called periodically to adjust performance level.
        """
        now = datetime.now()
        
        # Check if warming period is over
        if self.warming_state and self.warming_until and now >= self.warming_until:
            self.warming_state = False
            self.warming_until = None
            self.performance_level = 1.0
            return
            
        # Check if cooling period is over
        if self.cooling_state and self.cooling_until and now >= self.cooling_until:
            self.cooling_state = False
            self.cooling_until = None
            self.performance_level = 1.0
            return
            
        # Update performance level based on current state
        if self.warming_state and self.warming_until:
            # Gradually increase performance during warm-up
            total_warming_seconds = (self.warming_until - (now - timedelta(seconds=30))).total_seconds()
            elapsed_seconds = (now - (now - timedelta(seconds=30))).total_seconds()
            
            if total_warming_seconds > 0:
                progress = min(1.0, elapsed_seconds / total_warming_seconds)
                self.performance_level = 0.6 + (0.4 * progress)
                
        elif self.cooling_state and self.cooling_until:
            # Gradually decrease performance during cool-down
            total_cooling_seconds = (self.cooling_until - (now - timedelta(seconds=60))).total_seconds()
            elapsed_seconds = (now - (now - timedelta(seconds=60))).total_seconds()
            
            if total_cooling_seconds > 0:
                progress = min(1.0, elapsed_seconds / total_cooling_seconds)
                self.performance_level = 1.0 - (0.3 * progress)
                
    def get_effective_load_score(self) -> float:
        """Calculate effective load score considering thermal state.
        
        Returns:
            Adjusted load score (0.0 to 1.0)
        """
        base_score = self.calculate_load_score()
        
        # Adjust score based on warming/cooling state
        if self.warming_state:
            # Higher effective load during warm-up
            return min(1.0, base_score + (1.0 - self.performance_level))
            
        elif self.cooling_state:
            # Higher effective load during cool-down
            return min(1.0, base_score + (1.0 - self.performance_level))
            
        return base_score


@dataclass
class TestRequirements:
    """Test execution requirements."""
    __test__ = False
    test_id: str
    model_id: Optional[str] = None
    model_family: Optional[str] = None
    test_type: Optional[str] = None
    minimum_memory: float = 0.5  # GB
    required_memory_limit: float = 1000.0  # GB (upper limit for checking capacity)
    preferred_backend: Optional[str] = None
    required_backend: Optional[str] = None
    expected_duration: float = 60.0  # seconds
    priority: int = 3  # 1 (highest) to 5 (lowest)
    required_accelerators: Dict[str, int] = field(default_factory=dict)  # Type: count
    required_accelerator_limit: Dict[str, int] = field(default_factory=dict)  # Type: upper limit
    required_software: Dict[str, str] = field(default_factory=dict)  # Name: min version
    timeout: float = 3600.0  # seconds
    retries: int = 3
    concurrency_key: Optional[str] = None  # Tests with same key cannot run concurrently
    
    def __post_init__(self):
        """Post-initialization processing to set defaults."""
        # Ensure required_accelerator_limit has entries for all required_accelerators
        # with sensible defaults if not specified
        for accel_type, count in self.required_accelerators.items():
            if accel_type not in self.required_accelerator_limit:
                # Default limit is same as requirement (no sharing of accelerators)
                self.required_accelerator_limit[accel_type] = count
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "test_id": self.test_id,
            "model_id": self.model_id,
            "model_family": self.model_family,
            "test_type": self.test_type,
            "minimum_memory": self.minimum_memory,
            "required_memory_limit": self.required_memory_limit,
            "preferred_backend": self.preferred_backend,
            "required_backend": self.required_backend,
            "expected_duration": self.expected_duration,
            "priority": self.priority,
            "required_accelerators": self.required_accelerators,
            "required_accelerator_limit": self.required_accelerator_limit,
            "required_software": self.required_software,
            "timeout": self.timeout,
            "retries": self.retries,
            "concurrency_key": self.concurrency_key
        }
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestRequirements':
        """Create instance from dictionary."""
        return cls(**data)


@dataclass
class WorkerAssignment:
    """Assignment of a test to a worker."""
    worker_id: str
    test_id: str
    test_requirements: TestRequirements
    assigned_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "assigned"  # assigned, running, completed, failed
    result: Optional[Dict[str, Any]] = None
    execution_time: float = 0.0  # seconds
    success: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "worker_id": self.worker_id,
            "test_id": self.test_id,
            "test_requirements": self.test_requirements.to_dict(),
            "assigned_at": self.assigned_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status,
            "result": self.result,
            "execution_time": self.execution_time,
            "success": self.success
        }
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkerAssignment':
        """Create instance from dictionary."""
        # Convert string timestamps back to datetime
        if isinstance(data.get("assigned_at"), str):
            data["assigned_at"] = datetime.fromisoformat(data["assigned_at"])
        if isinstance(data.get("started_at"), str) and data["started_at"]:
            data["started_at"] = datetime.fromisoformat(data["started_at"])
        if isinstance(data.get("completed_at"), str) and data["completed_at"]:
            data["completed_at"] = datetime.fromisoformat(data["completed_at"])
            
        # Convert test_requirements from dict to TestRequirements
        if "test_requirements" in data and isinstance(data["test_requirements"], dict):
            data["test_requirements"] = TestRequirements.from_dict(data["test_requirements"])
            
        return cls(**data)
    
    def mark_started(self) -> None:
        """Mark this assignment as started."""
        self.started_at = datetime.now()
        self.status = "running"
    
    def mark_completed(self, success: bool, result: Optional[Dict[str, Any]] = None) -> None:
        """Mark this assignment as completed."""
        self.completed_at = datetime.now()
        self.status = "completed" if success else "failed"
        self.success = success
        self.result = result
        
        if self.started_at:
            self.execution_time = (self.completed_at - self.started_at).total_seconds()