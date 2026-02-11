#!/usr/bin/env python3
"""
Unit tests for Enhanced Hardware Taxonomy

This module provides comprehensive tests for the Enhanced Hardware Taxonomy
capabilities, including capability registry, hardware hierarchies, capability
inheritance, and workload matching.
"""

import os
import sys
import unittest
from typing import Dict, List, Any, Set, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import components
from .enhanced_hardware_taxonomy import (
    EnhancedHardwareTaxonomy,
    CapabilityDefinition,
    HardwareHierarchy,
    HardwareRelationship
)

from data.duckdb.distributed_testing.hardware_taxonomy import (
    HardwareClass,
    HardwareArchitecture,
    HardwareVendor,
    SoftwareBackend,
    PrecisionType,
    AcceleratorFeature,
    HardwareCapabilityProfile,
    MemoryProfile,
    create_cpu_profile,
    create_gpu_profile,
    create_npu_profile,
    create_browser_profile
)


class TestEnhancedHardwareTaxonomy(unittest.TestCase):
    """Test cases for the EnhancedHardwareTaxonomy class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create enhanced taxonomy
        self.taxonomy = EnhancedHardwareTaxonomy()
        
        # Create and register sample hardware profiles
        self.cpu_profile = create_cpu_profile(
            model_name="Intel i9-12900K",
            vendor=HardwareVendor.INTEL,
            cores=16,
            memory_gb=64.0,
            clock_speed_mhz=3200,
            has_avx=True,
            has_avx2=True,
            has_avx512=True
        )
        
        self.gpu_profile = create_gpu_profile(
            model_name="NVIDIA RTX 4090",
            vendor=HardwareVendor.NVIDIA,
            compute_units=128,
            memory_gb=24.0,
            clock_speed_mhz=2235,
            has_tensor_cores=True,
            has_ray_tracing=True,
            memory_bandwidth_gbps=1008.0,
            tdp_w=450.0
        )
        
        self.npu_profile = create_npu_profile(
            model_name="Qualcomm Hexagon 780",
            vendor=HardwareVendor.QUALCOMM,
            compute_units=8,
            memory_gb=4.0,
            clock_speed_mhz=1000,
            has_quantization=True,
            tdp_w=5.0
        )
        
        self.browser_profile = create_browser_profile(
            browser_name="Chrome",
            supports_webgpu=True,
            supports_webnn=True,
            gpu_profile=self.gpu_profile
        )
        
        # Register with taxonomy
        self.taxonomy.register_hardware_profile_with_capabilities(self.cpu_profile)
        self.taxonomy.register_hardware_profile_with_capabilities(self.gpu_profile)
        self.taxonomy.register_hardware_profile_with_capabilities(self.npu_profile)
        self.taxonomy.register_hardware_profile_with_capabilities(self.browser_profile)
        
        # Register worker hardware
        self.taxonomy.register_worker_hardware("worker1", [self.cpu_profile])
        self.taxonomy.register_worker_hardware("worker2", [self.gpu_profile])
        self.taxonomy.register_worker_hardware("worker3", [self.npu_profile])
        self.taxonomy.register_worker_hardware("worker4", [self.browser_profile])
        self.taxonomy.register_worker_hardware("worker5", [self.cpu_profile, self.gpu_profile])
    
    def test_capability_registry(self):
        """Test capability registry functionality."""
        # Check if common capabilities were initialized
        self.assertIn("compute.matrix_multiplication", self.taxonomy.capabilities_registry)
        self.assertIn("compute.convolution", self.taxonomy.capabilities_registry)
        self.assertIn("compute.tensor_core_acceleration", self.taxonomy.capabilities_registry)
        self.assertIn("memory.high_bandwidth", self.taxonomy.capabilities_registry)
        self.assertIn("precision.mixed", self.taxonomy.capabilities_registry)
        self.assertIn("specialized.vision", self.taxonomy.capabilities_registry)
        self.assertIn("specialized.nlp", self.taxonomy.capabilities_registry)
        
        # Get capability definition
        tensor_core_def = self.taxonomy.get_capability_definition("compute.tensor_core_acceleration")
        self.assertIsNotNone(tensor_core_def)
        self.assertEqual(tensor_core_def.capability_type, "compute")
        self.assertIn("compute.matrix_multiplication", tensor_core_def.related_capabilities)
        
        # Register a new capability
        new_capability = CapabilityDefinition(
            capability_id="test.custom_capability",
            capability_type="test",
            properties={"test_property": "test_value"},
            description="Test capability"
        )
        self.taxonomy.register_capability(new_capability)
        
        # Check if registered
        self.assertIn("test.custom_capability", self.taxonomy.capabilities_registry)
        retrieved_capability = self.taxonomy.get_capability_definition("test.custom_capability")
        self.assertEqual(retrieved_capability.description, "Test capability")
    
    def test_hardware_capability_registration(self):
        """Test hardware capability registration."""
        # Register a capability for a hardware profile
        hardware_id = f"{self.gpu_profile.hardware_class.value}_{self.gpu_profile.model_name}"
        self.taxonomy.register_hardware_capability(
            hardware_id,
            "test.custom_capability",
            {"performance": 100.0}
        )
        
        # Check if registered
        capabilities = self.taxonomy.get_hardware_capabilities(hardware_id)
        self.assertIn("test.custom_capability", capabilities)
        self.assertEqual(capabilities["test.custom_capability"]["performance"], 100.0)
    
    def test_inferred_capabilities(self):
        """Test capability inference from hardware properties."""
        # Get inferred capabilities for GPU profile
        gpu_id = f"{self.gpu_profile.hardware_class.value}_{self.gpu_profile.model_name}"
        
        # Get all capabilities (including inferred)
        capabilities = self.taxonomy.get_inherited_capabilities(self.gpu_profile)
        
        # Check inferred compute capabilities
        self.assertIn("compute.matrix_multiplication", capabilities)
        self.assertIn("compute.tensor_core_acceleration", capabilities)
        
        # Check inferred memory capabilities
        self.assertIn("memory.high_bandwidth", capabilities)
        
        # Check inferred specialized capabilities
        self.assertIn("specialized.vision", capabilities)
        self.assertIn("specialized.nlp", capabilities)
        
        # Check inferred precision capabilities
        self.assertIn("precision.mixed", capabilities)
        self.assertIn("precision.quantization", capabilities)
    
    def test_capability_compatibility(self):
        """Test capability compatibility calculation."""
        # Same capability should have perfect compatibility
        self.assertEqual(
            self.taxonomy.get_capability_compatibility(
                "compute.matrix_multiplication", 
                "compute.matrix_multiplication"
            ),
            1.0
        )
        
        # Related capabilities should have high compatibility
        self.assertAlmostEqual(
            self.taxonomy.get_capability_compatibility(
                "compute.matrix_multiplication", 
                "compute.tensor_core_acceleration"
            ),
            0.8
        )
        
        # Same type capabilities should have medium compatibility
        self.assertAlmostEqual(
            self.taxonomy.get_capability_compatibility(
                "compute.matrix_multiplication", 
                "compute.convolution"
            ),
            0.5
        )
        
        # Different type capabilities should have low compatibility
        self.assertAlmostEqual(
            self.taxonomy.get_capability_compatibility(
                "compute.matrix_multiplication", 
                "memory.high_bandwidth"
            ),
            0.1
        )
    
    def test_find_hardware_with_capability(self):
        """Test finding hardware with specific capabilities."""
        # Find hardware with tensor core acceleration
        tensor_core_hardware = self.taxonomy.find_hardware_with_capability("compute.tensor_core_acceleration")
        self.assertTrue(len(tensor_core_hardware) > 0)
        
        # All hardware should have the profile and tensor core feature
        for profile, props in tensor_core_hardware:
            self.assertIn(AcceleratorFeature.TENSOR_CORES, profile.features)
            self.assertIn("performance_multiplier", props)
        
        # Find hardware with high bandwidth memory
        high_bandwidth_hardware = self.taxonomy.find_hardware_with_capability("memory.high_bandwidth")
        self.assertTrue(len(high_bandwidth_hardware) > 0)
        
        # All hardware should have high bandwidth memory
        for profile, props in high_bandwidth_hardware:
            self.assertTrue(profile.memory.bandwidth_gbps > 500)
    
    def test_workload_hardware_match(self):
        """Test workload-to-hardware matching calculation."""
        # Define workload requirements for vision
        vision_requirements = {
            "compute.matrix_multiplication": {"performance": 100.0},
            "compute.convolution": {"performance": 50.0},
            "specialized.vision": {"effectiveness": 0.7}
        }
        
        # Calculate match scores
        cpu_match = self.taxonomy.calculate_workload_hardware_match(vision_requirements, self.cpu_profile)
        gpu_match = self.taxonomy.calculate_workload_hardware_match(vision_requirements, self.gpu_profile)
        npu_match = self.taxonomy.calculate_workload_hardware_match(vision_requirements, self.npu_profile)
        browser_match = self.taxonomy.calculate_workload_hardware_match(vision_requirements, self.browser_profile)
        
        # GPU should be best for vision
        self.assertGreater(gpu_match, cpu_match)
        self.assertGreater(gpu_match, npu_match)
        self.assertGreater(gpu_match, browser_match)
        
        # Define workload requirements for NLP
        nlp_requirements = {
            "compute.matrix_multiplication": {"performance": 200.0},
            "specialized.nlp": {"effectiveness": 0.8}
        }
        
        # Calculate match scores for NLP
        cpu_match_nlp = self.taxonomy.calculate_workload_hardware_match(nlp_requirements, self.cpu_profile)
        gpu_match_nlp = self.taxonomy.calculate_workload_hardware_match(nlp_requirements, self.gpu_profile)
        
        # GPU should be best for NLP too due to tensor cores
        self.assertGreater(gpu_match_nlp, cpu_match_nlp)
    
    def test_find_optimal_hardware(self):
        """Test finding optimal hardware for workloads."""
        # Define workload requirements for vision
        vision_requirements = {
            "compute.matrix_multiplication": {"performance": 100.0},
            "compute.convolution": {"performance": 50.0},
            "specialized.vision": {"effectiveness": 0.7}
        }
        
        # Find optimal hardware
        optimal_hardware = self.taxonomy.find_optimal_hardware_for_workload(vision_requirements)
        
        # Should return at least one result
        self.assertTrue(len(optimal_hardware) > 0)
        
        # First result should be the best match
        best_worker_id, best_profile, best_score = optimal_hardware[0]
        
        # Should be worker2 (GPU) or worker5 (CPU+GPU)
        self.assertIn(best_worker_id, ["worker2", "worker5"])
        
        # And the hardware class should be GPU
        self.assertEqual(best_profile.hardware_class, HardwareClass.GPU)
    
    def test_capabilities_map_for_worker(self):
        """Test getting capability map for a worker."""
        # Get capability map for worker5 (has CPU and GPU)
        capability_map = self.taxonomy.get_capability_map_for_worker("worker5")
        
        # Should have entries for both CPU and GPU
        cpu_id = f"{self.cpu_profile.hardware_class.value}_{self.cpu_profile.model_name}"
        gpu_id = f"{self.gpu_profile.hardware_class.value}_{self.gpu_profile.model_name}"
        
        self.assertIn(cpu_id, capability_map)
        self.assertIn(gpu_id, capability_map)
        
        # CPU should have audio capability
        self.assertIn("specialized.audio", capability_map[cpu_id])
        
        # GPU should have tensor core acceleration
        self.assertIn("compute.tensor_core_acceleration", capability_map[gpu_id])
    
    def test_hardware_relationships(self):
        """Test hardware relationship registration and querying."""
        # Register a relationship
        relationship = HardwareRelationship(
            source_id=f"{self.cpu_profile.hardware_class.value}_{self.cpu_profile.model_name}",
            target_id=f"{self.gpu_profile.hardware_class.value}_{self.gpu_profile.model_name}",
            relationship_type="accelerates",
            strength=0.9,
            properties={"data_transfer_speed_gbps": 16.0}
        )
        self.taxonomy.register_hardware_relationship(relationship)
        
        # Get relationships for CPU
        cpu_id = f"{self.cpu_profile.hardware_class.value}_{self.cpu_profile.model_name}"
        relationships = self.taxonomy.get_relationships_for_hardware(cpu_id)
        
        # Should have one relationship
        self.assertEqual(len(relationships), 1)
        
        # Find related hardware
        related_hardware = self.taxonomy.find_related_hardware(cpu_id)
        
        # Should have one result
        self.assertEqual(len(related_hardware), 1)
        
        # Should be to GPU
        target_id, rel_type, strength = related_hardware[0]
        self.assertEqual(target_id, f"{self.gpu_profile.hardware_class.value}_{self.gpu_profile.model_name}")
        self.assertEqual(rel_type, "accelerates")
        self.assertAlmostEqual(strength, 0.9)


if __name__ == "__main__":
    unittest.main()