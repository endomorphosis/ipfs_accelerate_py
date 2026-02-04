#!/usr/bin/env python3
"""
Integration Demo for Hardware-Test Matching Algorithms

This script demonstrates the functionality of the Hardware-Test Matching Algorithms
in the Distributed Testing Framework, showing how it integrates with the Enhanced 
Hardware Capability system, Error Handler, Test Dependency Manager, and Execution 
Orchestrator to enable intelligent test distribution based on hardware capabilities.
"""

import os
import sys
import time
import anyio
import logging
import uuid
import json
from datetime import datetime, timedelta
from typing import Dict, List, Set, Any, Optional

# Import required modules
from enhanced_hardware_capability import (
    HardwareCapability, WorkerHardwareCapabilities, 
    HardwareCapabilityDetector, HardwareCapabilityComparator,
    HardwareType, PrecisionType, CapabilityScore, HardwareVendor
)

from distributed_error_handler import (
    DistributedErrorHandler, ErrorType, ErrorSeverity,
    ErrorContext, ErrorReport, RetryPolicy, safe_execute
)

from test_dependency_manager import (
    TestDependencyManager, DependencyType, Dependency
)

from execution_orchestrator import (
    ExecutionOrchestrator, ExecutionStrategy
)

from hardware_test_matcher import (
    TestRequirementType, TestType, TestRequirement, TestProfile,
    HardwareTestMatch, TestPerformanceRecord, TestHardwareMatcher
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("hardware_test_matcher_demo")


class HardwareTestMatcherDemo:
    """Demo class for Hardware-Test Matching Algorithms."""
    
    def __init__(self):
        """Initialize the demo."""
        # Create core components
        self.error_handler = DistributedErrorHandler()
        self.dependency_manager = TestDependencyManager()
        
        # Create hardware capability detector and test matcher
        self.hardware_detector = HardwareCapabilityDetector()
        self.test_matcher = TestHardwareMatcher(
            hardware_capability_detector=self.hardware_detector,
            error_handler=self.error_handler
        )
        
        # Create execution orchestrator
        self.execution_orchestrator = ExecutionOrchestrator(
            dependency_manager=self.dependency_manager,
            max_workers=4,
            strategy=ExecutionStrategy.RESOURCE_AWARE
        )
        
        # Integrate error handler with execution orchestrator
        self.error_handler.integrate_with_execution_orchestrator(self.execution_orchestrator)
        self.error_handler.integrate_with_dependency_manager(self.dependency_manager)
        
        # Worker ID for this demo
        self.worker_id = f"worker_{uuid.uuid4().hex[:8]}"
        
        logger.info("Hardware Test Matcher Demo initialized")
    
    def setup_test_profiles(self):
        """Set up test profiles for the demo."""
        logger.info("Setting up test profiles...")
        
        # Create compute-intensive test
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
            tags=["gpu", "compute"],
            metadata={"priority": "high"}
        )
        
        # Create memory-intensive test
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
            tags=["memory"],
            metadata={"priority": "medium"}
        )
        
        # Create precision-sensitive test
        precision_test = TestProfile(
            test_id="precision_test",
            test_type=TestType.PRECISION_SENSITIVE,
            estimated_duration_seconds=30,
            estimated_memory_mb=1000,
            requirements=[
                TestRequirement(
                    requirement_type=TestRequirementType.PRECISION,
                    value=PrecisionType.FP16,
                    importance=1.0
                )
            ],
            tags=["precision"],
            metadata={"priority": "medium"}
        )
        
        # Create general-purpose test
        general_test = TestProfile(
            test_id="general_test",
            test_type=TestType.GENERAL,
            estimated_duration_seconds=15,
            estimated_memory_mb=200,
            tags=["general"],
            metadata={"priority": "low"}
        )
        
        # Create GPU-accelerated test
        gpu_test = TestProfile(
            test_id="gpu_test",
            test_type=TestType.GPU_ACCELERATED,
            estimated_duration_seconds=90,
            estimated_memory_mb=2000,
            requirements=[
                TestRequirement(
                    requirement_type=TestRequirementType.FEATURE,
                    value="tensor_cores",
                    importance=0.7
                )
            ],
            tags=["gpu", "ml"],
            metadata={"priority": "high"}
        )
        
        # Register test profiles
        self.test_matcher.register_test_profile(compute_test)
        self.test_matcher.register_test_profile(memory_test)
        self.test_matcher.register_test_profile(precision_test)
        self.test_matcher.register_test_profile(general_test)
        self.test_matcher.register_test_profile(gpu_test)
        
        # Add test dependencies
        self.dependency_manager.register_test("compute_test")
        self.dependency_manager.register_test("memory_test")
        self.dependency_manager.register_test("precision_test")
        self.dependency_manager.register_test("general_test")
        self.dependency_manager.register_test("gpu_test", [Dependency("compute_test")], ["gpu"])
        
        logger.info("Registered 5 test profiles with dependencies")
    
    def setup_simulated_hardware(self):
        """Set up simulated hardware capabilities."""
        logger.info("Setting up simulated hardware capabilities...")
        
        # Create CPU capability
        cpu_capability = HardwareCapability(
            hardware_type=HardwareType.CPU,
            vendor=HardwareVendor.INTEL,
            model="Intel Core i7-11700K",
            cores=8,
            memory_gb=32.0,
            supported_precisions=[
                PrecisionType.FP32, 
                PrecisionType.FP64,
                PrecisionType.INT32
            ],
            scores={
                "compute": CapabilityScore.GOOD,
                "memory": CapabilityScore.GOOD,
                "vector": CapabilityScore.GOOD,
                "overall": CapabilityScore.GOOD
            },
            capabilities={
                "avx2": True,
                "avx512": False,
                "hyperthreading": True,
                "frequency_mhz": 3600
            }
        )
        
        # Create high-end GPU capability
        gpu_high_capability = HardwareCapability(
            hardware_type=HardwareType.GPU,
            vendor=HardwareVendor.NVIDIA,
            model="NVIDIA RTX 3080",
            compute_units=68,
            memory_gb=10.0,
            supported_precisions=[
                PrecisionType.FP32, 
                PrecisionType.FP16,
                PrecisionType.INT32,
                PrecisionType.INT8
            ],
            scores={
                "compute": CapabilityScore.EXCELLENT,
                "memory": CapabilityScore.GOOD,
                "precision": CapabilityScore.EXCELLENT,
                "overall": CapabilityScore.EXCELLENT
            },
            capabilities={
                "tensor_cores": True,
                "cuda_cores": 8704,
                "memory_bandwidth_gbps": 760,
                "compute_capability": "8.6"
            }
        )
        
        # Create mid-range GPU capability
        gpu_mid_capability = HardwareCapability(
            hardware_type=HardwareType.GPU,
            vendor=HardwareVendor.NVIDIA,
            model="NVIDIA GTX 1660",
            compute_units=22,
            memory_gb=6.0,
            supported_precisions=[
                PrecisionType.FP32, 
                PrecisionType.FP16,
                PrecisionType.INT32
            ],
            scores={
                "compute": CapabilityScore.AVERAGE,
                "memory": CapabilityScore.AVERAGE,
                "precision": CapabilityScore.AVERAGE,
                "overall": CapabilityScore.AVERAGE
            },
            capabilities={
                "tensor_cores": False,
                "cuda_cores": 1408,
                "memory_bandwidth_gbps": 192,
                "compute_capability": "7.5"
            }
        )
        
        # Create simulated worker capabilities
        worker_capabilities = WorkerHardwareCapabilities(
            worker_id=self.worker_id,
            os_type="Linux",
            os_version="Ubuntu 22.04",
            hostname="demo-worker",
            cpu_count=8,
            total_memory_gb=32.0,
            hardware_capabilities=[cpu_capability, gpu_high_capability, gpu_mid_capability]
        )
        
        # Register worker capabilities
        self.test_matcher.register_worker_capabilities(worker_capabilities)
        
        logger.info(f"Registered worker {self.worker_id} with CPU and GPU capabilities")
    
    def match_and_execute_tests(self):
        """Match tests to hardware and execute them."""
        logger.info("Matching tests to hardware...")
        
        # Match tests to hardware
        test_ids = ["compute_test", "memory_test", "precision_test", "general_test", "gpu_test"]
        matches = self.test_matcher.match_tests_to_workers(test_ids)
        
        # Print matches
        print("\n" + "="*80)
        print("Hardware-Test Matches:")
        print("="*80)
        for test_id, match in matches.items():
            print(f"Test: {test_id}")
            print(f"  Matched to: {match.hardware_type.value} ({match.hardware_id})")
            print(f"  Match score: {match.match_score:.4f}")
            print(f"  Match factors:")
            for factor, score in match.match_factors.items():
                print(f"    - {factor}: {score:.4f}")
            print()
        
        # Execute tests based on matches
        print("\n" + "="*80)
        print("Test Execution Results:")
        print("="*80)
        
        # Process each test
        for test_id in test_ids:
            # Check if test has a match
            if test_id not in matches:
                print(f"Test {test_id}: NO SUITABLE HARDWARE FOUND")
                continue
            
            match = matches[test_id]
            
            # Simulate test execution
            success = True
            execution_time = 0
            memory_usage = 0
            error_type = None
            
            # Get test profile
            test_profile = self.test_matcher.get_test_profile(test_id)
            
            # Simulate different execution results based on test type
            if test_id == "compute_test":
                if match.hardware_type == HardwareType.GPU:
                    execution_time = 45.2
                    memory_usage = 450
                else:
                    execution_time = 120.5
                    memory_usage = 400
            
            elif test_id == "memory_test":
                execution_time = 110.8
                memory_usage = 3800
                
                # Simulate an error if matched to GPU with low memory
                if match.hardware_type == HardwareType.GPU and "memory_gb" in match.metadata:
                    gpu_memory = match.metadata["memory_gb"]
                    if gpu_memory and gpu_memory < 8:
                        success = False
                        error_type = "resource"
            
            elif test_id == "precision_test":
                if match.hardware_type == HardwareType.GPU:
                    execution_time = 28.3
                    memory_usage = 950
                else:
                    execution_time = 40.1
                    memory_usage = 920
            
            elif test_id == "general_test":
                execution_time = 14.2
                memory_usage = 180
            
            elif test_id == "gpu_test":
                if match.hardware_type == HardwareType.GPU:
                    # Check if tensor cores are available
                    has_tensor_cores = False
                    for hw in self.test_matcher.get_worker_capability(match.worker_id).hardware_capabilities:
                        if hw.hardware_type == HardwareType.GPU and hw.capabilities.get("tensor_cores", False):
                            has_tensor_cores = True
                            break
                    
                    if has_tensor_cores:
                        execution_time = 65.7
                        memory_usage = 1850
                    else:
                        execution_time = 85.3
                        memory_usage = 1900
                else:
                    execution_time = 180.5
                    memory_usage = 1800
                    # Simulate an error for non-GPU hardware
                    success = False
                    error_type = "hardware"
            
            # Print execution results
            print(f"Test {test_id}:")
            print(f"  Hardware: {match.hardware_type.value} ({match.hardware_id})")
            print(f"  Success: {success}")
            print(f"  Execution time: {execution_time:.2f}s")
            print(f"  Memory usage: {memory_usage} MB")
            if not success:
                print(f"  Error type: {error_type}")
            print()
            
            # Register performance record
            performance_record = TestPerformanceRecord(
                test_id=test_id,
                worker_id=match.worker_id,
                hardware_id=match.hardware_id,
                execution_time_seconds=execution_time,
                memory_usage_mb=memory_usage,
                success=success,
                error_type=error_type
            )
            
            self.test_matcher.register_test_performance(performance_record)
        
        # Update dependency manager with test statuses
        for test_id in test_ids:
            if test_id in matches:
                match = matches[test_id]
                # Get performance metrics
                metrics = self.test_matcher._get_test_performance_metrics(test_id, match.hardware_id)
                
                # Update dependency manager
                if metrics["success_rate"] == 1.0:
                    self.dependency_manager.update_test_status(
                        test_id=test_id,
                        status="completed",
                        result={"success": True}
                    )
                else:
                    self.dependency_manager.update_test_status(
                        test_id=test_id,
                        status="failed",
                        result={"success": False}
                    )
    
    def analyze_performance_impact(self):
        """Analyze the impact of performance history on matching."""
        logger.info("Analyzing impact of performance history on matching...")
        
        print("\n" + "="*80)
        print("Impact of Performance History on Matching:")
        print("="*80)
        
        # Match again with performance history
        test_ids = ["compute_test", "memory_test", "precision_test", "general_test", "gpu_test"]
        updated_matches = self.test_matcher.match_tests_to_workers(test_ids)
        
        # Print updated matches
        for test_id, match in updated_matches.items():
            # Get original match
            original_match = None
            for orig_test_id, orig_match in self.original_matches.items():
                if orig_test_id == test_id:
                    original_match = orig_match
                    break
            
            # Skip if no original match
            if not original_match:
                continue
            
            # Compare match scores
            score_diff = match.match_score - original_match.match_score
            hardware_changed = match.hardware_id != original_match.hardware_id
            
            print(f"Test: {test_id}")
            print(f"  Original match: {original_match.hardware_type.value} ({original_match.hardware_id})")
            print(f"  Original score: {original_match.match_score:.4f}")
            print(f"  Updated match: {match.hardware_type.value} ({match.hardware_id})")
            print(f"  Updated score: {match.match_score:.4f}")
            print(f"  Score change: {score_diff:+.4f}")
            
            if hardware_changed:
                print(f"  Hardware changed: YES - Performance history caused hardware reassignment")
            else:
                print(f"  Hardware changed: NO - Same hardware selected but with adjusted score")
            
            # Print updated match factors to show impact of performance history
            print(f"  Updated match factors:")
            for factor, score in match.match_factors.items():
                # Get original factor score
                original_score = original_match.match_factors.get(factor, 0.0)
                factor_diff = score - original_score
                
                if abs(factor_diff) > 0.01:
                    print(f"    - {factor}: {score:.4f} ({factor_diff:+.4f})")
                else:
                    print(f"    - {factor}: {score:.4f}")
            
            print()
    
    def demonstrate_specialized_matchers(self):
        """Demonstrate specialized matchers for different test types."""
        logger.info("Demonstrating specialized matchers...")
        
        print("\n" + "="*80)
        print("Specialized Matchers for Different Test Types:")
        print("="*80)
        
        # Create specialized matchers
        compute_matcher = self.test_matcher.get_specialized_matcher(TestType.COMPUTE_INTENSIVE)
        memory_matcher = self.test_matcher.get_specialized_matcher(TestType.MEMORY_INTENSIVE)
        gpu_matcher = self.test_matcher.get_specialized_matcher(TestType.GPU_ACCELERATED)
        precision_matcher = self.test_matcher.get_specialized_matcher(TestType.PRECISION_SENSITIVE)
        
        # Print weight configurations
        print("Weight configurations for specialized matchers:\n")
        
        # Get factor names
        factor_names = list(self.test_matcher.match_factor_weights.keys())
        
        # Print header
        print(f"{'Factor':<25} {'Default':<10} {'Compute':<10} {'Memory':<10} {'GPU':<10} {'Precision':<10}")
        print("-" * 75)
        
        # Print weights for each factor
        for factor in factor_names:
            default_weight = self.test_matcher.match_factor_weights.get(factor, 0.0)
            compute_weight = compute_matcher.match_factor_weights.get(factor, 0.0)
            memory_weight = memory_matcher.match_factor_weights.get(factor, 0.0)
            gpu_weight = gpu_matcher.match_factor_weights.get(factor, 0.0)
            precision_weight = precision_matcher.match_factor_weights.get(factor, 0.0)
            
            print(f"{factor:<25} {default_weight:<10.2f} {compute_weight:<10.2f} "
                  f"{memory_weight:<10.2f} {gpu_weight:<10.2f} {precision_weight:<10.2f}")
        
        print("\nMatching same test with different specialized matchers:\n")
        
        # Use the same test with different matchers
        test_id = "general_test"
        
        # Match with each matcher
        default_match = self.test_matcher.match_test_to_hardware(test_id)
        compute_match = compute_matcher.match_test_to_hardware(test_id)
        memory_match = memory_matcher.match_test_to_hardware(test_id)
        gpu_match = gpu_matcher.match_test_to_hardware(test_id)
        precision_match = precision_matcher.match_test_to_hardware(test_id)
        
        # Print results
        print(f"Test: {test_id}")
        print(f"  Default matcher: {default_match.hardware_type.value} (Score: {default_match.match_score:.4f})")
        print(f"  Compute matcher: {compute_match.hardware_type.value} (Score: {compute_match.match_score:.4f})")
        print(f"  Memory matcher: {memory_match.hardware_type.value} (Score: {memory_match.match_score:.4f})")
        print(f"  GPU matcher: {gpu_match.hardware_type.value} (Score: {gpu_match.match_score:.4f})")
        print(f"  Precision matcher: {precision_match.hardware_type.value} (Score: {precision_match.match_score:.4f})")
    
    def demonstrate_adaptive_weights(self):
        """Demonstrate adaptive weight adjustment based on errors."""
        logger.info("Demonstrating adaptive weight adjustment...")
        
        print("\n" + "="*80)
        print("Adaptive Weight Adjustment Based on Errors:")
        print("="*80)
        
        # Print initial weights
        print("Initial weights:")
        for factor, weight in self.test_matcher.match_factor_weights.items():
            print(f"  {factor}: {weight:.4f}")
        
        # Simulate errors for memory-intensive test
        print("\nSimulating memory errors for memory_test...")
        for i in range(3):
            performance_record = TestPerformanceRecord(
                test_id="memory_test",
                worker_id=self.worker_id,
                hardware_id=f"{self.worker_id}:{HardwareType.CPU.value}",
                execution_time_seconds=150,
                memory_usage_mb=None,
                success=False,
                error_type="resource"
            )
            self.test_matcher.register_test_performance(performance_record)
            print(f"  Registered error {i+1}: resource error for memory_test")
        
        # Print adjusted weights
        print("\nAdjusted weights after memory errors:")
        for factor, weight in self.test_matcher.match_factor_weights.items():
            print(f"  {factor}: {weight:.4f}")
        
        # Simulate errors for precision-sensitive test
        print("\nSimulating precision errors for precision_test...")
        for i in range(3):
            performance_record = TestPerformanceRecord(
                test_id="precision_test",
                worker_id=self.worker_id,
                hardware_id=f"{self.worker_id}:{HardwareType.CPU.value}",
                execution_time_seconds=50,
                memory_usage_mb=1000,
                success=False,
                error_type="precision"
            )
            self.test_matcher.register_test_performance(performance_record)
            print(f"  Registered error {i+1}: precision error for precision_test")
        
        # Print adjusted weights
        print("\nAdjusted weights after precision errors:")
        for factor, weight in self.test_matcher.match_factor_weights.items():
            print(f"  {factor}: {weight:.4f}")
        
        # Match tests again with adjusted weights
        print("\nMatching tests again with adjusted weights...")
        test_ids = ["compute_test", "memory_test", "precision_test", "general_test", "gpu_test"]
        updated_matches = self.test_matcher.match_tests_to_workers(test_ids)
        
        # Print updated matches for affected tests
        for test_id in ["memory_test", "precision_test"]:
            if test_id in updated_matches:
                match = updated_matches[test_id]
                print(f"\nTest: {test_id}")
                print(f"  Matched to: {match.hardware_type.value} ({match.hardware_id})")
                print(f"  Match score: {match.match_score:.4f}")
                print(f"  Match factors:")
                for factor, score in match.match_factors.items():
                    print(f"    - {factor}: {score:.4f}")
    
    def run_demo(self):
        """Run the complete demo."""
        logger.info("Starting Hardware Test Matcher Demo")
        
        # Set up test profiles and simulated hardware
        self.setup_test_profiles()
        self.setup_simulated_hardware()
        
        # Match and execute tests
        self.original_matches = self.test_matcher.match_tests_to_workers(
            ["compute_test", "memory_test", "precision_test", "general_test", "gpu_test"]
        )
        self.match_and_execute_tests()
        
        # Analyze performance impact
        self.analyze_performance_impact()
        
        # Demonstrate specialized matchers
        self.demonstrate_specialized_matchers()
        
        # Demonstrate adaptive weights
        self.demonstrate_adaptive_weights()
        
        logger.info("Hardware Test Matcher Demo completed")


# Run the demo
if __name__ == "__main__":
    demo = HardwareTestMatcherDemo()
    demo.run_demo()