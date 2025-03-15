#!/usr/bin/env python3
"""
Hardware-Test Matching Algorithms for Distributed Testing Framework

This module provides intelligent algorithms for matching tests to appropriate hardware resources
in a distributed testing environment. It leverages the enhanced hardware capability system to 
match test requirements with hardware capabilities for optimal resource utilization.

Key features:
- Multi-factor matching algorithm considering hardware capabilities, test requirements, and historical performance
- Historical performance-based scoring for test-hardware combinations
- Adaptive weight adjustment based on execution results
- Specialized matchers for different test types (compute-intensive, memory-intensive, etc.)
- Integration with enhanced hardware capability and distributed error handler
"""

import logging
import json
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Set, Optional, Tuple, Any, Union, DefaultDict, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque

# Import related modules
from enhanced_hardware_capability import (
    HardwareCapability, WorkerHardwareCapabilities, 
    HardwareCapabilityDetector, HardwareCapabilityComparator,
    HardwareType, PrecisionType, CapabilityScore
)

from distributed_error_handler import (
    DistributedErrorHandler, ErrorType, ErrorSeverity,
    ErrorContext, ErrorReport, safe_execute
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("hardware_test_matcher")


class TestRequirementType(Enum):
    """Types of test requirements."""
    COMPUTE = "compute"         # Computational requirements
    MEMORY = "memory"           # Memory requirements
    STORAGE = "storage"         # Storage requirements
    PRECISION = "precision"     # Numerical precision requirements
    HARDWARE_TYPE = "hardware_type"  # Specific hardware type requirements
    NETWORK = "network"         # Network requirements
    FEATURE = "feature"         # Feature-specific requirements


class TestType(Enum):
    """Types of tests with different resource profiles."""
    COMPUTE_INTENSIVE = "compute_intensive"    # Tests with high computational requirements
    MEMORY_INTENSIVE = "memory_intensive"      # Tests with high memory requirements
    IO_INTENSIVE = "io_intensive"              # Tests with high I/O requirements
    NETWORK_INTENSIVE = "network_intensive"    # Tests with high network requirements
    GPU_ACCELERATED = "gpu_accelerated"        # Tests that benefit from GPU acceleration
    PRECISION_SENSITIVE = "precision_sensitive"  # Tests sensitive to numerical precision
    GENERAL = "general"                        # General-purpose tests


@dataclass
class TestRequirement:
    """Represents a requirement for a test."""
    requirement_type: TestRequirementType
    value: Any
    importance: float = 1.0  # Importance weight (0.0-1.0)
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestProfile:
    """Profile of a test with its requirements."""
    test_id: str
    test_type: TestType = TestType.GENERAL
    requirements: List[TestRequirement] = field(default_factory=list)
    estimated_duration_seconds: Optional[float] = None
    estimated_memory_mb: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


@dataclass
class HardwareTestMatch:
    """Represents a match between a test and hardware."""
    test_id: str
    worker_id: str
    hardware_id: str
    hardware_type: HardwareType
    match_score: float
    match_factors: Dict[str, float]
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestPerformanceRecord:
    """Record of test performance on a specific hardware configuration."""
    test_id: str
    worker_id: str
    hardware_id: str
    execution_time_seconds: float
    memory_usage_mb: Optional[float] = None
    success: bool = True
    error_type: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TestHardwareMatcher:
    """
    Intelligent test-to-hardware matching system.
    
    This class is responsible for matching tests to the most appropriate hardware
    resources based on test requirements, hardware capabilities, and historical
    performance data.
    """
    
    def __init__(
            self, 
            hardware_capability_detector: Optional[HardwareCapabilityDetector] = None,
            error_handler: Optional[DistributedErrorHandler] = None,
            db_connection: Any = None
        ):
        """
        Initialize the test-hardware matcher.
        
        Args:
            hardware_capability_detector: Detector for hardware capabilities
            error_handler: Error handler for handling matching failures
            db_connection: Optional database connection for persistence
        """
        # Initialize hardware components
        self.hardware_detector = hardware_capability_detector or HardwareCapabilityDetector()
        self.hardware_comparator = HardwareCapabilityComparator()
        self.error_handler = error_handler
        self.db_connection = db_connection
        
        # Test profiles and requirements
        self.test_profiles: Dict[str, TestProfile] = {}
        
        # Performance history
        self.performance_history: Dict[str, List[TestPerformanceRecord]] = defaultdict(list)
        
        # Hardware capabilities cache
        self.worker_capabilities: Dict[str, WorkerHardwareCapabilities] = {}
        
        # Matching configuration
        self.match_factor_weights = {
            "hardware_type_compatibility": 0.9,
            "compute_capability": 0.8,
            "memory_compatibility": 0.8,
            "precision_compatibility": 0.7,
            "historical_performance": 0.7,
            "hardware_preference": 0.6,
            "error_history": 0.6,
            "load_balancing": 0.5,
            "feature_support": 0.7
        }
        
        # Default test type to hardware type mappings
        self.test_type_to_hardware_type = {
            TestType.COMPUTE_INTENSIVE: [HardwareType.GPU, HardwareType.TPU, HardwareType.CPU],
            TestType.MEMORY_INTENSIVE: [HardwareType.CPU, HardwareType.GPU],
            TestType.IO_INTENSIVE: [HardwareType.CPU],
            TestType.NETWORK_INTENSIVE: [HardwareType.CPU],
            TestType.GPU_ACCELERATED: [HardwareType.GPU, HardwareType.TPU],
            TestType.PRECISION_SENSITIVE: [HardwareType.GPU, HardwareType.TPU, HardwareType.CPU],
            TestType.GENERAL: [HardwareType.CPU, HardwareType.GPU]
        }
        
        # Adaptive weight adjustment settings
        self.enable_adaptive_weights = True
        self.adaptation_rate = 0.05
        self.weight_min = 0.1
        self.weight_max = 1.0
        
        # Performance tracking
        self.performance_weight_window = 10  # Number of recent executions for performance weighting
        
        # Create schema if needed
        if self.db_connection:
            self._create_schema()
        
        logger.info("Test Hardware Matcher initialized")
    
    def _create_schema(self) -> None:
        """Create database schema for test matching."""
        if not self.db_connection:
            return
        
        try:
            # Create test profiles table
            self.db_connection.execute("""
            CREATE TABLE IF NOT EXISTS test_profiles (
                test_id VARCHAR PRIMARY KEY,
                test_type VARCHAR,
                estimated_duration_seconds FLOAT,
                estimated_memory_mb FLOAT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                metadata JSON
            )
            """)
            
            # Create test requirements table
            self.db_connection.execute("""
            CREATE TABLE IF NOT EXISTS test_requirements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id VARCHAR,
                requirement_type VARCHAR,
                value TEXT,
                importance FLOAT,
                description TEXT,
                metadata JSON,
                FOREIGN KEY (test_id) REFERENCES test_profiles(test_id)
            )
            """)
            
            # Create test performance history table
            self.db_connection.execute("""
            CREATE TABLE IF NOT EXISTS test_performance_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id VARCHAR,
                worker_id VARCHAR,
                hardware_id VARCHAR,
                execution_time_seconds FLOAT,
                memory_usage_mb FLOAT,
                success BOOLEAN,
                error_type VARCHAR,
                timestamp TIMESTAMP,
                metadata JSON,
                FOREIGN KEY (test_id) REFERENCES test_profiles(test_id)
            )
            """)
            
            # Create hardware-test matches table
            self.db_connection.execute("""
            CREATE TABLE IF NOT EXISTS hardware_test_matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id VARCHAR,
                worker_id VARCHAR,
                hardware_id VARCHAR,
                hardware_type VARCHAR,
                match_score FLOAT,
                match_factors JSON,
                created_at TIMESTAMP,
                metadata JSON,
                FOREIGN KEY (test_id) REFERENCES test_profiles(test_id)
            )
            """)
            
            logger.debug("Test hardware matching database schema created")
        except Exception as e:
            if self.error_handler:
                context = {
                    "component": "test_hardware_matcher",
                    "operation": "create_schema"
                }
                self.error_handler.handle_error(e, context)
            else:
                logger.error(f"Failed to create test matching schema: {str(e)}")
    
    def register_test_profile(self, test_profile: TestProfile) -> bool:
        """
        Register a test profile with requirements.
        
        Args:
            test_profile: Profile of the test to register
            
        Returns:
            True if registration was successful, False otherwise
        """
        try:
            # Store in memory
            self.test_profiles[test_profile.test_id] = test_profile
            
            # Persist to database if available
            if self.db_connection:
                self._persist_test_profile(test_profile)
            
            logger.debug(f"Registered test profile for {test_profile.test_id} of type {test_profile.test_type.value}")
            return True
        except Exception as e:
            if self.error_handler:
                context = {
                    "component": "test_hardware_matcher",
                    "operation": "register_test_profile",
                    "test_id": test_profile.test_id
                }
                self.error_handler.handle_error(e, context)
            else:
                logger.error(f"Failed to register test profile: {str(e)}")
            return False
    
    def _persist_test_profile(self, test_profile: TestProfile) -> None:
        """Persist test profile to database."""
        if not self.db_connection:
            return
        
        try:
            # Insert profile
            self.db_connection.execute("""
            INSERT INTO test_profiles (
                test_id, test_type, estimated_duration_seconds, estimated_memory_mb,
                created_at, updated_at, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(test_id) DO UPDATE SET
                test_type = excluded.test_type,
                estimated_duration_seconds = excluded.estimated_duration_seconds,
                estimated_memory_mb = excluded.estimated_memory_mb,
                updated_at = excluded.updated_at,
                metadata = excluded.metadata
            """, (
                test_profile.test_id,
                test_profile.test_type.value,
                test_profile.estimated_duration_seconds,
                test_profile.estimated_memory_mb,
                datetime.now(),
                datetime.now(),
                json.dumps({"metadata": test_profile.metadata, "tags": test_profile.tags})
            ))
            
            # Delete existing requirements to avoid duplicates
            self.db_connection.execute("""
            DELETE FROM test_requirements WHERE test_id = ?
            """, (test_profile.test_id,))
            
            # Insert requirements
            for req in test_profile.requirements:
                # Convert requirement value to string for storage
                value_str = json.dumps(req.value) if isinstance(req.value, (dict, list)) else str(req.value)
                
                self.db_connection.execute("""
                INSERT INTO test_requirements (
                    test_id, requirement_type, value, importance, description, metadata
                ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    test_profile.test_id,
                    req.requirement_type.value,
                    value_str,
                    req.importance,
                    req.description,
                    json.dumps(req.metadata)
                ))
                
            logger.debug(f"Persisted test profile for {test_profile.test_id}")
        except Exception as e:
            logger.error(f"Failed to persist test profile: {str(e)}")
    
    def register_worker_capabilities(self, worker_capabilities: WorkerHardwareCapabilities) -> bool:
        """
        Register hardware capabilities for a worker.
        
        Args:
            worker_capabilities: Hardware capabilities of the worker
            
        Returns:
            True if registration was successful, False otherwise
        """
        try:
            # Store in memory
            self.worker_capabilities[worker_capabilities.worker_id] = worker_capabilities
            
            logger.debug(f"Registered capabilities for worker {worker_capabilities.worker_id} "
                        f"with {len(worker_capabilities.hardware_capabilities)} hardware capabilities")
            return True
        except Exception as e:
            if self.error_handler:
                context = {
                    "component": "test_hardware_matcher",
                    "operation": "register_worker_capabilities",
                    "worker_id": worker_capabilities.worker_id
                }
                self.error_handler.handle_error(e, context)
            else:
                logger.error(f"Failed to register worker capabilities: {str(e)}")
            return False
    
    def register_test_performance(self, performance_record: TestPerformanceRecord) -> bool:
        """
        Register test performance data for a specific hardware configuration.
        
        Args:
            performance_record: Performance record to register
            
        Returns:
            True if registration was successful, False otherwise
        """
        try:
            # Store in memory
            key = f"{performance_record.test_id}:{performance_record.hardware_id}"
            self.performance_history[key].append(performance_record)
            
            # Keep only recent history for memory efficiency
            max_history = 100
            if len(self.performance_history[key]) > max_history:
                self.performance_history[key] = self.performance_history[key][-max_history:]
            
            # Persist to database if available
            if self.db_connection:
                self._persist_performance_record(performance_record)
            
            # Adaptive weight adjustment if enabled
            if self.enable_adaptive_weights:
                self._adjust_weights_based_on_performance(performance_record)
            
            logger.debug(f"Registered performance for test {performance_record.test_id} "
                        f"on worker {performance_record.worker_id}: "
                        f"{performance_record.execution_time_seconds:.2f}s, success={performance_record.success}")
            return True
        except Exception as e:
            if self.error_handler:
                context = {
                    "component": "test_hardware_matcher",
                    "operation": "register_test_performance",
                    "test_id": performance_record.test_id,
                    "worker_id": performance_record.worker_id
                }
                self.error_handler.handle_error(e, context)
            else:
                logger.error(f"Failed to register test performance: {str(e)}")
            return False
    
    def _persist_performance_record(self, performance_record: TestPerformanceRecord) -> None:
        """Persist performance record to database."""
        if not self.db_connection:
            return
        
        try:
            self.db_connection.execute("""
            INSERT INTO test_performance_history (
                test_id, worker_id, hardware_id, execution_time_seconds,
                memory_usage_mb, success, error_type, timestamp, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                performance_record.test_id,
                performance_record.worker_id,
                performance_record.hardware_id,
                performance_record.execution_time_seconds,
                performance_record.memory_usage_mb,
                performance_record.success,
                performance_record.error_type,
                performance_record.timestamp,
                json.dumps(performance_record.metadata)
            ))
            
            logger.debug(f"Persisted performance record for test {performance_record.test_id}")
        except Exception as e:
            logger.error(f"Failed to persist performance record: {str(e)}")
    
    def _adjust_weights_based_on_performance(self, performance_record: TestPerformanceRecord) -> None:
        """
        Adjust matching factor weights based on test performance.
        
        Args:
            performance_record: Recent performance record to learn from
        """
        # Skip adjustment if test was successful
        if performance_record.success:
            return
        
        # Get test profile
        test_profile = self.test_profiles.get(performance_record.test_id)
        if not test_profile:
            return
        
        # Adjust weights based on error type and test type
        error_type = performance_record.error_type or "unknown"
        
        # Determine which factors to adjust
        factors_to_adjust = []
        
        if error_type in ["resource", "memory"]:
            # Memory-related error
            factors_to_adjust.append("memory_compatibility")
        
        elif error_type in ["timeout", "performance"]:
            # Performance-related error
            factors_to_adjust.append("compute_capability")
            factors_to_adjust.append("historical_performance")
        
        elif error_type in ["precision", "assertion"]:
            # Precision-related error
            factors_to_adjust.append("precision_compatibility")
        
        elif error_type in ["network"]:
            # Network-related error
            factors_to_adjust.append("feature_support")
        
        elif error_type in ["hardware", "system"]:
            # Hardware-related error
            factors_to_adjust.append("hardware_type_compatibility")
            factors_to_adjust.append("error_history")
        
        # Adjust weights
        for factor in factors_to_adjust:
            if factor in self.match_factor_weights:
                # Increase weight (give it more importance)
                new_weight = min(
                    self.weight_max,
                    self.match_factor_weights[factor] + self.adaptation_rate
                )
                
                # Log adjustment
                logger.debug(f"Adjusting weight for {factor} from {self.match_factor_weights[factor]:.2f} to {new_weight:.2f}")
                
                # Update weight
                self.match_factor_weights[factor] = new_weight
    
    def _get_test_performance_metrics(self, test_id: str, hardware_id: str) -> Dict[str, Any]:
        """
        Get performance metrics for a test on specific hardware.
        
        Args:
            test_id: ID of the test
            hardware_id: ID of the hardware
            
        Returns:
            Dictionary with performance metrics
        """
        key = f"{test_id}:{hardware_id}"
        records = self.performance_history.get(key, [])
        
        if not records:
            return {
                "success_rate": None,
                "avg_execution_time": None,
                "avg_memory_usage": None,
                "execution_count": 0,
                "last_execution": None,
                "performance_score": None
            }
        
        # Calculate metrics
        success_count = sum(1 for r in records if r.success)
        total_count = len(records)
        success_rate = success_count / total_count if total_count > 0 else 0.0
        
        # Get execution times for successful runs
        execution_times = [r.execution_time_seconds for r in records if r.success]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else None
        
        # Get memory usage for successful runs
        memory_usages = [r.memory_usage_mb for r in records if r.success and r.memory_usage_mb is not None]
        avg_memory_usage = sum(memory_usages) / len(memory_usages) if memory_usages else None
        
        # Get last execution time
        last_execution = max((r.timestamp for r in records), default=None)
        
        # Calculate performance score (1.0 is best)
        performance_score = None
        if avg_execution_time and success_rate > 0:
            # Penalize by failures
            performance_score = success_rate / (avg_execution_time ** 0.5)
            
            # Scale to reasonable range
            performance_score = min(1.0, performance_score)
        
        return {
            "success_rate": success_rate,
            "avg_execution_time": avg_execution_time,
            "avg_memory_usage": avg_memory_usage,
            "execution_count": total_count,
            "last_execution": last_execution,
            "performance_score": performance_score
        }
    
    def match_test_to_hardware(self, test_id: str, 
                            available_workers: List[str] = None) -> Optional[HardwareTestMatch]:
        """
        Match a test to the most appropriate hardware.
        
        Args:
            test_id: ID of the test to match
            available_workers: List of available worker IDs (None means all registered workers)
            
        Returns:
            Best hardware test match if found, None otherwise
        """
        # Get test profile
        test_profile = self.test_profiles.get(test_id)
        if not test_profile:
            logger.warning(f"Cannot match test {test_id}: no test profile available")
            return None
        
        # Determine available workers
        if available_workers is None:
            available_workers = list(self.worker_capabilities.keys())
        
        if not available_workers:
            logger.warning(f"Cannot match test {test_id}: no workers available")
            return None
        
        # Evaluate matches for each worker
        matches = []
        for worker_id in available_workers:
            worker_capability = self.worker_capabilities.get(worker_id)
            if not worker_capability:
                continue
            
            # Evaluate match for each hardware capability
            for hardware in worker_capability.hardware_capabilities:
                match_score, match_factors = self._evaluate_match_score(
                    test_profile, worker_capability, hardware
                )
                
                # Only consider viable matches
                if match_score > 0:
                    hardware_id = f"{worker_id}:{hardware.hardware_type.value}"
                    if hasattr(hardware, "model") and hardware.model:
                        hardware_id += f":{hardware.model}"
                    
                    match = HardwareTestMatch(
                        test_id=test_id,
                        worker_id=worker_id,
                        hardware_id=hardware_id,
                        hardware_type=hardware.hardware_type,
                        match_score=match_score,
                        match_factors=match_factors
                    )
                    matches.append(match)
        
        # Find best match
        if matches:
            best_match = max(matches, key=lambda m: m.match_score)
            
            # Persist match to database if available
            if self.db_connection:
                self._persist_match(best_match)
            
            logger.info(f"Matched test {test_id} to worker {best_match.worker_id} "
                       f"({best_match.hardware_type.value}) with score {best_match.match_score:.4f}")
            return best_match
        else:
            logger.warning(f"No suitable hardware match found for test {test_id}")
            return None
    
    def _persist_match(self, match: HardwareTestMatch) -> None:
        """Persist hardware test match to database."""
        if not self.db_connection:
            return
        
        try:
            self.db_connection.execute("""
            INSERT INTO hardware_test_matches (
                test_id, worker_id, hardware_id, hardware_type,
                match_score, match_factors, created_at, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                match.test_id,
                match.worker_id,
                match.hardware_id,
                match.hardware_type.value,
                match.match_score,
                json.dumps(match.match_factors),
                match.created_at,
                json.dumps(match.metadata)
            ))
            
            logger.debug(f"Persisted hardware test match for test {match.test_id}")
        except Exception as e:
            logger.error(f"Failed to persist hardware test match: {str(e)}")
    
    def _evaluate_match_score(self, test_profile: TestProfile,
                           worker_capability: WorkerHardwareCapabilities,
                           hardware: HardwareCapability) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate the match score between a test and a hardware capability.
        
        Args:
            test_profile: Test profile
            worker_capability: Worker hardware capabilities
            hardware: Specific hardware capability to evaluate
            
        Returns:
            Tuple of (match_score, match_factors)
        """
        # Initialize match factors
        match_factors = {
            "hardware_type_compatibility": 0.0,
            "compute_capability": 0.0,
            "memory_compatibility": 0.0,
            "precision_compatibility": 0.0,
            "historical_performance": 0.0,
            "hardware_preference": 0.0,
            "error_history": 0.0,
            "load_balancing": 0.0,
            "feature_support": 0.0
        }
        
        # Check hardware type compatibility
        hardware_type_compat = self._evaluate_hardware_type_compatibility(
            test_profile, hardware.hardware_type
        )
        match_factors["hardware_type_compatibility"] = hardware_type_compat
        
        # If hardware type is incompatible, return zero score
        if hardware_type_compat == 0:
            return 0.0, match_factors
        
        # Evaluate compute capability
        match_factors["compute_capability"] = self._evaluate_compute_capability(
            test_profile, hardware
        )
        
        # Evaluate memory compatibility
        match_factors["memory_compatibility"] = self._evaluate_memory_compatibility(
            test_profile, hardware
        )
        
        # Evaluate precision compatibility
        match_factors["precision_compatibility"] = self._evaluate_precision_compatibility(
            test_profile, hardware
        )
        
        # Evaluate historical performance
        hardware_id = f"{worker_capability.worker_id}:{hardware.hardware_type.value}"
        if hasattr(hardware, "model") and hardware.model:
            hardware_id += f":{hardware.model}"
        
        match_factors["historical_performance"] = self._evaluate_historical_performance(
            test_profile.test_id, hardware_id
        )
        
        # Evaluate hardware preference
        match_factors["hardware_preference"] = self._evaluate_hardware_preference(
            test_profile, hardware.hardware_type
        )
        
        # Evaluate error history
        match_factors["error_history"] = self._evaluate_error_history(
            test_profile.test_id, worker_capability.worker_id, hardware_id
        )
        
        # Evaluate load balancing
        match_factors["load_balancing"] = self._evaluate_load_balancing(
            worker_capability.worker_id
        )
        
        # Evaluate feature support
        match_factors["feature_support"] = self._evaluate_feature_support(
            test_profile, hardware
        )
        
        # Calculate weighted score
        weighted_score = 0.0
        total_weight = 0.0
        
        for factor, score in match_factors.items():
            weight = self.match_factor_weights.get(factor, 0.0)
            weighted_score += score * weight
            total_weight += weight
        
        # Normalize score
        final_score = weighted_score / total_weight if total_weight > 0 else 0.0
        
        return final_score, match_factors
    
    def _evaluate_hardware_type_compatibility(self, test_profile: TestProfile, 
                                           hardware_type: HardwareType) -> float:
        """
        Evaluate hardware type compatibility for a test.
        
        Args:
            test_profile: Test profile
            hardware_type: Hardware type to evaluate
            
        Returns:
            Compatibility score (0.0-1.0)
        """
        # Get preferred hardware types for this test type
        preferred_types = self.test_type_to_hardware_type.get(test_profile.test_type, [])
        
        # Check if hardware type is in preferred list
        if not preferred_types:
            return 0.5  # No preference, moderate compatibility
        
        # Check if exact match for first preference
        if preferred_types and hardware_type == preferred_types[0]:
            return 1.0  # Perfect match
        
        # Check if in preferred list
        if hardware_type in preferred_types:
            # Score based on position in preference list
            position = preferred_types.index(hardware_type)
            return 1.0 - (position * 0.2)
        
        # Check specific requirements
        for req in test_profile.requirements:
            if req.requirement_type == TestRequirementType.HARDWARE_TYPE:
                if isinstance(req.value, HardwareType) and req.value == hardware_type:
                    return 1.0
                if isinstance(req.value, str) and req.value == hardware_type.value:
                    return 1.0
        
        # Default low compatibility
        return 0.1
    
    def _evaluate_compute_capability(self, test_profile: TestProfile,
                                 hardware: HardwareCapability) -> float:
        """
        Evaluate compute capability compatibility for a test.
        
        Args:
            test_profile: Test profile
            hardware: Hardware capability to evaluate
            
        Returns:
            Compatibility score (0.0-1.0)
        """
        # For compute-intensive tests, check compute score
        if test_profile.test_type == TestType.COMPUTE_INTENSIVE:
            # Get compute score if available
            compute_score = hardware.scores.get("compute")
            if compute_score:
                # Map CapabilityScore to float
                score_mapping = {
                    CapabilityScore.EXCELLENT: 1.0,
                    CapabilityScore.GOOD: 0.8,
                    CapabilityScore.AVERAGE: 0.6,
                    CapabilityScore.BASIC: 0.4,
                    CapabilityScore.MINIMAL: 0.2,
                    CapabilityScore.UNKNOWN: 0.5
                }
                return score_mapping.get(compute_score, 0.5)
        
        # For GPU-accelerated tests, prioritize GPUs with more compute units
        elif test_profile.test_type == TestType.GPU_ACCELERATED:
            if hardware.hardware_type == HardwareType.GPU:
                # Check compute units
                if hardware.compute_units:
                    # Normalize to a reasonable range (assuming up to 128 compute units)
                    return min(1.0, hardware.compute_units / 128)
                
                # Check for tensor cores
                tensor_cores = hardware.capabilities.get("tensor_cores", False)
                if tensor_cores:
                    return 0.9
                
                # Default good score for GPUs
                return 0.7
        
        # For general tests, return moderate score
        return 0.5
    
    def _evaluate_memory_compatibility(self, test_profile: TestProfile,
                                   hardware: HardwareCapability) -> float:
        """
        Evaluate memory compatibility for a test.
        
        Args:
            test_profile: Test profile
            hardware: Hardware capability to evaluate
            
        Returns:
            Compatibility score (0.0-1.0)
        """
        # Check if test has memory requirements
        memory_requirement = None
        memory_priority = 0.5
        
        for req in test_profile.requirements:
            if req.requirement_type == TestRequirementType.MEMORY:
                memory_requirement = req.value
                memory_priority = req.importance
                break
        
        # Use estimated memory if no explicit requirement
        if memory_requirement is None and test_profile.estimated_memory_mb:
            memory_requirement = test_profile.estimated_memory_mb
        
        # If no memory requirement, assign moderate score
        if memory_requirement is None:
            return 0.5
        
        # Check if hardware has memory information
        if hardware.memory_gb is None:
            return 0.3  # Unknown memory size is not ideal
        
        # Convert to same unit (MB)
        hardware_memory_mb = hardware.memory_gb * 1024
        
        # Calculate ratio of available to required
        memory_ratio = hardware_memory_mb / memory_requirement
        
        # Score based on ratio
        if memory_ratio < 1.0:
            # Not enough memory
            return 0.0
        elif memory_ratio < 1.2:
            # Just enough memory
            return 0.3
        elif memory_ratio < 2.0:
            # Sufficient memory
            return 0.7
        else:
            # Abundant memory
            return 1.0
    
    def _evaluate_precision_compatibility(self, test_profile: TestProfile,
                                      hardware: HardwareCapability) -> float:
        """
        Evaluate numerical precision compatibility for a test.
        
        Args:
            test_profile: Test profile
            hardware: Hardware capability to evaluate
            
        Returns:
            Compatibility score (0.0-1.0)
        """
        # Check if test has precision requirements
        required_precision = None
        precision_priority = 0.5
        
        for req in test_profile.requirements:
            if req.requirement_type == TestRequirementType.PRECISION:
                required_precision = req.value
                precision_priority = req.importance
                break
        
        # If no precision requirement, assign moderate score
        if required_precision is None:
            return 0.5
        
        # Convert to PrecisionType if string
        if isinstance(required_precision, str):
            try:
                required_precision = PrecisionType(required_precision)
            except ValueError:
                logger.warning(f"Invalid precision value: {required_precision}")
                return 0.3
        
        # Check if hardware supports the required precision
        if not hardware.supported_precisions:
            return 0.3  # Unknown precision support is not ideal
        
        if required_precision in hardware.supported_precisions:
            return 1.0  # Perfect match
        
        # Check if hardware supports higher precision
        precision_order = [
            PrecisionType.INT2,
            PrecisionType.INT4,
            PrecisionType.INT8,
            PrecisionType.INT16,
            PrecisionType.INT32,
            PrecisionType.INT64,
            PrecisionType.FP16,
            PrecisionType.BF16,
            PrecisionType.FP32,
            PrecisionType.FP64,
            PrecisionType.MIXED
        ]
        
        if required_precision in precision_order and any(precision in hardware.supported_precisions for precision in precision_order[precision_order.index(required_precision):]):
            return 0.7  # Has higher precision
        
        # No suitable precision
        return 0.0
    
    def _evaluate_historical_performance(self, test_id: str, hardware_id: str) -> float:
        """
        Evaluate historical performance for a test on specific hardware.
        
        Args:
            test_id: ID of the test
            hardware_id: ID of the hardware
            
        Returns:
            Performance score (0.0-1.0)
        """
        # Get performance metrics
        metrics = self._get_test_performance_metrics(test_id, hardware_id)
        
        # If no history, return moderate score
        if metrics["execution_count"] == 0:
            return 0.5
        
        # Factor in success rate heavily
        if metrics["success_rate"] is not None:
            if metrics["success_rate"] < 0.5:
                return 0.1  # Poor success rate
        
        # Use performance score if available
        if metrics["performance_score"] is not None:
            return metrics["performance_score"]
        
        # Default moderate score
        return 0.5
    
    def _evaluate_hardware_preference(self, test_profile: TestProfile,
                                   hardware_type: HardwareType) -> float:
        """
        Evaluate hardware preference for a test.
        
        Args:
            test_profile: Test profile
            hardware_type: Hardware type to evaluate
            
        Returns:
            Preference score (0.0-1.0)
        """
        # Check for explicit hardware preference in test metadata
        hardware_preferences = test_profile.metadata.get("hardware_preferences", [])
        if hardware_preferences:
            # Convert string values to HardwareType if needed
            normalized_preferences = []
            for pref in hardware_preferences:
                if isinstance(pref, str):
                    try:
                        normalized_preferences.append(HardwareType(pref))
                    except ValueError:
                        continue
                else:
                    normalized_preferences.append(pref)
            
            # Check if hardware type is in preferences
            if hardware_type in normalized_preferences:
                position = normalized_preferences.index(hardware_type)
                return 1.0 - (position * 0.2)
        
        # Check for explicit tags that might indicate preference
        for tag in test_profile.tags:
            if tag.lower() == hardware_type.value.lower():
                return 0.9
            if tag.lower() == f"prefer_{hardware_type.value.lower()}":
                return 1.0
            if tag.lower() == f"avoid_{hardware_type.value.lower()}":
                return 0.1
        
        # Default to moderate preference
        return 0.5
    
    def _evaluate_error_history(self, test_id: str, worker_id: str, hardware_id: str) -> float:
        """
        Evaluate error history for a test on specific hardware.
        
        Args:
            test_id: ID of the test
            worker_id: ID of the worker
            hardware_id: ID of the hardware
            
        Returns:
            Error history score (0.0-1.0, higher means fewer errors)
        """
        # Get performance metrics for this test-hardware combination
        key = f"{test_id}:{hardware_id}"
        records = self.performance_history.get(key, [])
        
        if not records:
            return 0.5  # No history, moderate score
        
        # Calculate error rate
        error_count = sum(1 for r in records if not r.success)
        total_count = len(records)
        error_rate = error_count / total_count if total_count > 0 else 0.0
        
        # Calculate recent error rate (last 5 runs)
        recent_records = records[-5:] if len(records) >= 5 else records
        recent_error_count = sum(1 for r in recent_records if not r.success)
        recent_error_rate = recent_error_count / len(recent_records) if recent_records else 0.0
        
        # Weight recent errors more heavily
        weighted_error_rate = (error_rate * 0.3) + (recent_error_rate * 0.7)
        
        # Score is inverse of error rate (0 errors = 1.0 score)
        return 1.0 - weighted_error_rate
    
    def _evaluate_load_balancing(self, worker_id: str) -> float:
        """
        Evaluate load balancing factor for a worker.
        
        Args:
            worker_id: ID of the worker
            
        Returns:
            Load balancing score (0.0-1.0, higher means less loaded)
        """
        # In a real implementation, this would check current worker load
        # For now, return a fixed value
        return 0.7
    
    def _evaluate_feature_support(self, test_profile: TestProfile,
                               hardware: HardwareCapability) -> float:
        """
        Evaluate feature support for a test.
        
        Args:
            test_profile: Test profile
            hardware: Hardware capability to evaluate
            
        Returns:
            Feature support score (0.0-1.0)
        """
        # Check if test has feature requirements
        feature_requirements = []
        
        for req in test_profile.requirements:
            if req.requirement_type == TestRequirementType.FEATURE:
                feature_requirements.append((req.value, req.importance))
        
        # If no feature requirements, assign high score
        if not feature_requirements:
            return 0.8
        
        # Check each required feature
        supported_count = 0
        total_importance = 0.0
        supported_importance = 0.0
        
        for feature, importance in feature_requirements:
            total_importance += importance
            
            # Check if hardware supports the feature
            feature_supported = False
            
            if isinstance(feature, str):
                feature_supported = hardware.capabilities.get(feature, False)
            elif isinstance(feature, dict):
                # Check multiple feature attributes
                all_attributes_match = True
                for attr, value in feature.items():
                    if hardware.capabilities.get(attr) != value:
                        all_attributes_match = False
                        break
                feature_supported = all_attributes_match
            
            if feature_supported:
                supported_count += 1
                supported_importance += importance
        
        # Calculate score based on importance-weighted support
        if total_importance > 0:
            score = supported_importance / total_importance
        else:
            score = supported_count / len(feature_requirements) if feature_requirements else 0.8
        
        return score
    
    def match_tests_to_workers(self, test_ids: List[str], 
                            available_workers: List[str] = None) -> Dict[str, HardwareTestMatch]:
        """
        Match multiple tests to appropriate workers.
        
        Args:
            test_ids: List of test IDs to match
            available_workers: List of available worker IDs (None means all registered workers)
            
        Returns:
            Dictionary mapping test IDs to hardware test matches
        """
        matches = {}
        
        # Match each test individually
        for test_id in test_ids:
            match = self.match_test_to_hardware(test_id, available_workers)
            if match:
                matches[test_id] = match
        
        return matches
    
    def create_test_profile_from_dict(self, profile_data: Dict[str, Any]) -> TestProfile:
        """
        Create a test profile from a dictionary.
        
        Args:
            profile_data: Dictionary with test profile data
            
        Returns:
            Test profile object
        """
        # Extract test profile data
        test_id = profile_data.get("test_id")
        if not test_id:
            raise ValueError("Test ID is required")
        
        # Get test type
        test_type_str = profile_data.get("test_type", "general")
        try:
            test_type = TestType(test_type_str)
        except ValueError:
            logger.warning(f"Invalid test type: {test_type_str}, using GENERAL")
            test_type = TestType.GENERAL
        
        # Create test profile
        profile = TestProfile(
            test_id=test_id,
            test_type=test_type,
            estimated_duration_seconds=profile_data.get("estimated_duration_seconds"),
            estimated_memory_mb=profile_data.get("estimated_memory_mb"),
            metadata=profile_data.get("metadata", {}),
            tags=profile_data.get("tags", [])
        )
        
        # Add requirements
        requirements_data = profile_data.get("requirements", [])
        for req_data in requirements_data:
            requirement_type_str = req_data.get("requirement_type")
            if not requirement_type_str:
                continue
            
            try:
                requirement_type = TestRequirementType(requirement_type_str)
            except ValueError:
                logger.warning(f"Invalid requirement type: {requirement_type_str}")
                continue
            
            requirement = TestRequirement(
                requirement_type=requirement_type,
                value=req_data.get("value"),
                importance=req_data.get("importance", 1.0),
                description=req_data.get("description"),
                metadata=req_data.get("metadata", {})
            )
            
            profile.requirements.append(requirement)
        
        return profile
    
    def get_specialized_matcher(self, test_type: TestType) -> 'TestHardwareMatcher':
        """
        Get a specialized matcher for a specific test type.
        
        Args:
            test_type: Test type to create specialized matcher for
            
        Returns:
            Specialized test hardware matcher
        """
        # Create new matcher with same base configuration
        specialized_matcher = TestHardwareMatcher(
            hardware_capability_detector=self.hardware_detector,
            error_handler=self.error_handler,
            db_connection=self.db_connection
        )
        
        # Copy base configuration
        specialized_matcher.worker_capabilities = self.worker_capabilities
        specialized_matcher.performance_history = self.performance_history
        
        # Adjust weights based on test type
        if test_type == TestType.COMPUTE_INTENSIVE:
            # Prioritize compute capability
            specialized_matcher.match_factor_weights["compute_capability"] = 1.0
            specialized_matcher.match_factor_weights["historical_performance"] = 0.9
            specialized_matcher.match_factor_weights["memory_compatibility"] = 0.7
        
        elif test_type == TestType.MEMORY_INTENSIVE:
            # Prioritize memory compatibility
            specialized_matcher.match_factor_weights["memory_compatibility"] = 1.0
            specialized_matcher.match_factor_weights["historical_performance"] = 0.9
            specialized_matcher.match_factor_weights["compute_capability"] = 0.6
        
        elif test_type == TestType.GPU_ACCELERATED:
            # Prioritize GPU hardware and features
            specialized_matcher.match_factor_weights["hardware_type_compatibility"] = 1.0
            specialized_matcher.match_factor_weights["compute_capability"] = 0.9
            specialized_matcher.match_factor_weights["feature_support"] = 0.8
        
        elif test_type == TestType.PRECISION_SENSITIVE:
            # Prioritize precision support
            specialized_matcher.match_factor_weights["precision_compatibility"] = 1.0
            specialized_matcher.match_factor_weights["hardware_type_compatibility"] = 0.9
            specialized_matcher.match_factor_weights["historical_performance"] = 0.8
        
        return specialized_matcher
    
    def get_test_profile(self, test_id: str) -> Optional[TestProfile]:
        """
        Get a test profile by ID.
        
        Args:
            test_id: ID of the test
            
        Returns:
            Test profile if found, None otherwise
        """
        return self.test_profiles.get(test_id)
    
    def get_worker_capability(self, worker_id: str) -> Optional[WorkerHardwareCapabilities]:
        """
        Get worker capabilities by ID.
        
        Args:
            worker_id: ID of the worker
            
        Returns:
            Worker capabilities if found, None otherwise
        """
        return self.worker_capabilities.get(worker_id)
    
    def get_test_performance_history(self, test_id: str,
                                   hardware_id: Optional[str] = None) -> List[TestPerformanceRecord]:
        """
        Get performance history for a test.
        
        Args:
            test_id: ID of the test
            hardware_id: Optional hardware ID to filter by
            
        Returns:
            List of performance records
        """
        if hardware_id:
            key = f"{test_id}:{hardware_id}"
            return self.performance_history.get(key, [])
        else:
            # Collect all records for this test across hardware
            records = []
            for key, history in self.performance_history.items():
                if key.startswith(f"{test_id}:"):
                    records.extend(history)
            return records


# Example usage
if __name__ == "__main__":
    # Create hardware detector and matcher
    hardware_detector = HardwareCapabilityDetector()
    matcher = TestHardwareMatcher(hardware_detector=hardware_detector)
    
    # Detect worker capabilities
    worker_capabilities = hardware_detector.detect_all_capabilities()
    matcher.register_worker_capabilities(worker_capabilities)
    
    # Create some test profiles
    compute_test = TestProfile(
        test_id="compute_test",
        test_type=TestType.COMPUTE_INTENSIVE,
        estimated_duration_seconds=60,
        estimated_memory_mb=500,
        requirements=[
            TestRequirement(
                requirement_type=TestRequirementType.COMPUTE,
                value="high",
                importance=0.9
            ),
            TestRequirement(
                requirement_type=TestRequirementType.HARDWARE_TYPE,
                value=HardwareType.GPU,
                importance=0.8
            )
        ],
        tags=["gpu", "compute"]
    )
    
    memory_test = TestProfile(
        test_id="memory_test",
        test_type=TestType.MEMORY_INTENSIVE,
        estimated_duration_seconds=120,
        estimated_memory_mb=4000,
        requirements=[
            TestRequirement(
                requirement_type=TestRequirementType.MEMORY,
                value=3000,
                importance=0.9
            )
        ],
        tags=["memory"]
    )
    
    precision_test = TestProfile(
        test_id="precision_test",
        test_type=TestType.PRECISION_SENSITIVE,
        estimated_duration_seconds=30,
        estimated_memory_mb=1000,
        requirements=[
            TestRequirement(
                requirement_type=TestRequirementType.PRECISION,
                value=PrecisionType.FP32,
                importance=1.0
            )
        ],
        tags=["precision"]
    )
    
    # Register test profiles
    matcher.register_test_profile(compute_test)
    matcher.register_test_profile(memory_test)
    matcher.register_test_profile(precision_test)
    
    # Match tests to hardware
    print("Matching tests to hardware:")
    for test_id in ["compute_test", "memory_test", "precision_test"]:
        match = matcher.match_test_to_hardware(test_id)
        if match:
            print(f"{test_id}: Matched to {match.worker_id} ({match.hardware_type.value}) with score {match.match_score:.4f}")
            print(f"  Match factors: {match.match_factors}")
        else:
            print(f"{test_id}: No suitable match found")
    
    # Register some test performance data
    matcher.register_test_performance(TestPerformanceRecord(
        test_id="compute_test",
        worker_id=worker_capabilities.worker_id,
        hardware_id=f"{worker_capabilities.worker_id}:gpu",
        execution_time_seconds=45.2,
        memory_usage_mb=450,
        success=True
    ))
    
    # Match again (should use performance history)
    print("\nMatching after performance history:")
    match = matcher.match_test_to_hardware("compute_test")
    if match:
        print(f"compute_test: Matched to {match.worker_id} ({match.hardware_type.value}) with score {match.match_score:.4f}")
        print(f"  Match factors: {match.match_factors}")