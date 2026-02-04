"""
Test script for the enhanced hardware taxonomy.

This script tests the capabilities of the enhanced hardware taxonomy,
including the capability registry, hardware relationship modeling,
and capability inheritance support.
"""

import unittest
from typing import Dict, Set, Any

from test.tests.api.duckdb_api.distributed_testing.hardware_taxonomy import (
    HardwareClass,
    HardwareArchitecture,
    HardwareVendor,
    SoftwareBackend,
    PrecisionType,
    AcceleratorFeature,
    MemoryProfile,
    HardwareCapabilityProfile,
    create_cpu_profile,
    create_gpu_profile,
    create_npu_profile
)
from test.tests.api.duckdb_api.distributed_testing.enhanced_hardware_taxonomy import (
    EnhancedHardwareTaxonomy,
    CapabilityScope,
    CapabilityDefinition,
    HardwareRelationship
)


class TestEnhancedHardwareTaxonomy(unittest.TestCase):
    """Test cases for the EnhancedHardwareTaxonomy class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.taxonomy = EnhancedHardwareTaxonomy()
        
        # Create some test hardware profiles
        self.cpu_profile = create_cpu_profile(
            model_name="Intel Core i9-12900K",
            vendor=HardwareVendor.INTEL,
            cores=16,
            memory_gb=64.0,
            clock_speed_mhz=5200,
            has_avx=True,
            has_avx2=True,
            has_avx512=True
        )
        
        self.gpu_profile = create_gpu_profile(
            model_name="NVIDIA RTX 4090",
            vendor=HardwareVendor.NVIDIA,
            compute_units=128,
            memory_gb=24.0,
            clock_speed_mhz=2520,
            has_tensor_cores=True,
            has_ray_tracing=True,
            compute_capability="8.9",
            memory_bandwidth_gbps=1008.0,
            tdp_w=450.0
        )
        
        self.npu_profile = create_npu_profile(
            model_name="Qualcomm Hexagon NPU",
            vendor=HardwareVendor.QUALCOMM,
            compute_units=8,
            memory_gb=8.0,
            clock_speed_mhz=1000,
            has_quantization=True,
            tdp_w=5.0
        )
        
        # Register the profiles
        self.taxonomy.register_hardware_profile(self.cpu_profile, auto_discover=False)
        self.taxonomy.register_hardware_profile(self.gpu_profile, auto_discover=False)
        self.taxonomy.register_hardware_profile(self.npu_profile, auto_discover=False)
    
    def test_capability_registry(self):
        """Test registering and retrieving capabilities from the registry."""
        # Register a new test capability
        test_cap = self.taxonomy.register_capability(
            capability_id="test_capability",
            name="Test Capability",
            description="A test capability for unit testing",
            scope=CapabilityScope.GLOBAL,
            properties={"test_property": "test_value"},
            supported_hardware_classes={HardwareClass.CPU, HardwareClass.GPU}
        )
        
        # Verify the capability was registered
        self.assertIn("test_capability", self.taxonomy.capabilities_registry)
        
        # Retrieve the capability
        retrieved_cap = self.taxonomy.get_capability("test_capability")
        self.assertEqual(retrieved_cap.name, "Test Capability")
        self.assertEqual(retrieved_cap.properties["test_property"], "test_value")
        self.assertEqual(retrieved_cap.supported_hardware_classes, {HardwareClass.CPU, HardwareClass.GPU})
    
    def test_hardware_hierarchy(self):
        """Test defining and retrieving hardware hierarchies."""
        # Define a new hardware hierarchy
        self.taxonomy.define_hardware_hierarchy(
            parent_hardware=HardwareClass.GPU,
            child_hardware=HardwareClass.TPU,
            inheritance_factor=0.8
        )
        
        # Verify the hierarchy was defined
        self.assertIn(HardwareClass.GPU, self.taxonomy.hardware_hierarchies)
        self.assertIn((HardwareClass.TPU, 0.8), self.taxonomy.hardware_hierarchies[HardwareClass.GPU])
        
        # Check the relationship was created
        relationships = self.taxonomy.get_hardware_relationships(
            hardware=HardwareClass.GPU,
            relationship_type="parent_of"
        )
        self.assertTrue(any(r.target_hardware == HardwareClass.TPU for r in relationships))
    
    def test_hardware_relationship(self):
        """Test registering and retrieving hardware relationships."""
        # Register a new relationship
        relationship = self.taxonomy.register_hardware_relationship(
            source_hardware=HardwareClass.GPU,
            source_type="class",
            target_hardware=HardwareClass.CPU,
            target_type="class",
            relationship_type="accelerates",
            compatibility_score=0.9,
            data_transfer_efficiency=0.8,
            shared_memory=False,
            properties={"acceleration_factor": 10.0}
        )
        
        # Verify the relationship was registered
        self.assertIn("class:gpu_accelerates_class:cpu", self.taxonomy.hardware_relationships)
        
        # Retrieve relationships
        relationships = self.taxonomy.get_hardware_relationships(
            hardware=HardwareClass.GPU,
            relationship_type="accelerates"
        )
        self.assertEqual(len(relationships), 1)
        self.assertEqual(relationships[0].target_hardware, HardwareClass.CPU)
        self.assertEqual(relationships[0].compatibility_score, 0.9)
        self.assertEqual(relationships[0].properties["acceleration_factor"], 10.0)
    
    def test_capability_assignment(self):
        """Test assigning capabilities to hardware profiles."""
        # Assign a capability to a hardware profile
        self.taxonomy.assign_capability_to_hardware(
            hardware_profile=self.gpu_profile,
            capability_id="matrix_multiplication"
        )
        
        # Verify the capability was assigned
        self.assertTrue(self.taxonomy.has_capability(self.gpu_profile, "matrix_multiplication"))
        
        # Verify the capability shows up in the hardware capabilities
        capabilities = self.taxonomy.get_hardware_capabilities(self.gpu_profile, include_inherited=False)
        self.assertIn("matrix_multiplication", capabilities)
    
    def test_capability_inheritance(self):
        """Test capability inheritance through hardware hierarchies."""
        # Define a hierarchy with the CPU as a parent of GPU
        self.taxonomy.define_hardware_hierarchy(
            parent_hardware=HardwareClass.CPU,
            child_hardware=HardwareClass.GPU,
            inheritance_factor=0.7
        )
        
        # Assign a capability to the CPU
        self.taxonomy.assign_capability_to_hardware(
            hardware_profile=self.cpu_profile,
            capability_id="matrix_multiplication"
        )
        
        # Get inherited capabilities for the GPU
        inherited_capabilities = self.taxonomy.get_inherited_capabilities(self.gpu_profile)
        self.assertIn("matrix_multiplication", inherited_capabilities)
        
        # Verify inheritance through get_hardware_capabilities with include_inherited=True
        all_capabilities = self.taxonomy.get_hardware_capabilities(self.gpu_profile, include_inherited=True)
        self.assertIn("matrix_multiplication", all_capabilities)
    
    def test_auto_discover_capabilities(self):
        """Test automatic discovery of capabilities based on hardware characteristics."""
        # Run auto-discovery on the GPU profile
        discovered = self.taxonomy.discover_capabilities(self.gpu_profile)
        
        # Since the GPU has tensor cores, it should discover tensor_core_acceleration
        self.assertIn("tensor_core_acceleration", discovered)
        
        # First assign matrix_multiplication since it's a prerequisite for tensor_core_acceleration
        self.taxonomy.assign_capability_to_hardware(
            hardware_profile=self.gpu_profile,
            capability_id="matrix_multiplication"
        )
        
        # Then auto-assign the capabilities
        assigned = self.taxonomy.auto_assign_capabilities(self.gpu_profile)
        self.assertIn("tensor_core_acceleration", assigned)
        
        # Verify both capabilities were assigned
        self.assertTrue(self.taxonomy.has_capability(self.gpu_profile, "tensor_core_acceleration"))
        self.assertTrue(self.taxonomy.has_capability(self.gpu_profile, "matrix_multiplication"))
    
    def test_workload_capability_match(self):
        """Test matching workload capability requirements to hardware profiles."""
        # First assign matrix_multiplication since it's a prerequisite
        self.taxonomy.assign_capability_to_hardware(
            hardware_profile=self.gpu_profile,
            capability_id="matrix_multiplication"
        )
        
        # Then assign tensor_core_acceleration
        self.taxonomy.assign_capability_to_hardware(
            hardware_profile=self.gpu_profile,
            capability_id="tensor_core_acceleration"
        )
        
        # Assign to NPU
        self.taxonomy.assign_capability_to_hardware(
            hardware_profile=self.npu_profile,
            capability_id="low_precision_computation"
        )
        
        # Calculate match scores for a workload requiring tensor operations
        gpu_score = self.taxonomy.calculate_workload_capability_match(
            workload_type="nlp",
            required_capabilities={"tensor_core_acceleration", "matrix_multiplication"},
            hardware_profile=self.gpu_profile
        )
        cpu_score = self.taxonomy.calculate_workload_capability_match(
            workload_type="nlp",
            required_capabilities={"tensor_core_acceleration", "matrix_multiplication"},
            hardware_profile=self.cpu_profile
        )
        npu_score = self.taxonomy.calculate_workload_capability_match(
            workload_type="nlp",
            required_capabilities={"tensor_core_acceleration", "matrix_multiplication"},
            hardware_profile=self.npu_profile
        )
        
        # GPU should have a perfect match
        self.assertEqual(gpu_score, 1.0)
        # CPU should have a low or zero match
        self.assertLess(cpu_score, 0.5)
        # NPU should have a partial match if it has matrix_multiplication
        self.assertLessEqual(npu_score, 0.5)
    
    def test_register_profile_with_auto_discover(self):
        """Test registering a hardware profile with auto-discovery of capabilities."""
        # Create a new test capability that doesn't have prerequisites
        self.taxonomy.register_capability(
            capability_id="mixed_precision",
            name="Mixed Precision",
            description="Support for mixed precision operations",
            scope=CapabilityScope.GLOBAL,
            supported_hardware_classes={HardwareClass.GPU}
        )
        
        # Add discovery rule for our new capability
        def original_discover_capabilities(self, hardware_profile):
            discovered = self.__original_discover_capabilities(hardware_profile)
            
            # Add our new capability to be discovered for GPU profiles
            if hardware_profile.hardware_class == HardwareClass.GPU:
                discovered.add("mixed_precision")
                
            return discovered
        
        # Save original method and monkey patch with our version
        self.taxonomy.__original_discover_capabilities = self.taxonomy.discover_capabilities
        self.taxonomy.discover_capabilities = lambda hp: original_discover_capabilities(self.taxonomy, hp)
        
        # Create a new profile with features that should trigger auto-discovery
        new_gpu_profile = create_gpu_profile(
            model_name="NVIDIA RTX 3090",
            vendor=HardwareVendor.NVIDIA,
            compute_units=82,
            memory_gb=24.0,
            clock_speed_mhz=1695,
            has_tensor_cores=True,
            has_ray_tracing=True,
            compute_capability="8.6",
            memory_bandwidth_gbps=936.0,
            tdp_w=350.0
        )
        
        # Register with auto-discovery
        self.taxonomy.register_hardware_profile(new_gpu_profile, auto_discover=True)
        
        # Verify our new capability was auto-discovered and assigned
        self.assertTrue(self.taxonomy.has_capability(new_gpu_profile, "mixed_precision"))


if __name__ == "__main__":
    unittest.main()