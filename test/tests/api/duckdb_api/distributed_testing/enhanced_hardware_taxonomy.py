"""
Enhanced Hardware Taxonomy for Distributed Testing Framework

This module extends the base hardware taxonomy with a capability registry,
hardware relationship modeling, and capability inheritance support, enabling
more sophisticated hardware detection, matching, and optimization strategies.

The enhanced taxonomy allows the system to model hierarchical relationships
between different hardware types, provides a centralized registry of hardware
capabilities, and supports runtime discovery of hardware capabilities.
"""

import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union, Any

from .hardware_taxonomy import (
    HardwareClass,
    HardwareArchitecture,
    HardwareVendor,
    SoftwareBackend,
    PrecisionType,
    AcceleratorFeature,
    MemoryProfile,
    HardwareCapabilityProfile,
    HardwareSpecialization,
    HardwareTaxonomy
)


class CapabilityScope(enum.Enum):
    """Defines the scope of a capability in the registry."""
    GLOBAL = "global"  # Capability applies globally
    CLASS = "class"    # Capability applies to a hardware class
    VENDOR = "vendor"  # Capability applies to a specific vendor
    MODEL = "model"    # Capability applies to a specific model
    DEVICE = "device"  # Capability applies to a specific device instance


@dataclass
class CapabilityDefinition:
    """Definition of a capability in the registry."""
    capability_id: str
    name: str
    description: str
    scope: CapabilityScope
    properties: Dict[str, Any] = field(default_factory=dict)
    supported_hardware_classes: Set[HardwareClass] = field(default_factory=set)
    supported_architectures: Set[HardwareArchitecture] = field(default_factory=set)
    supported_vendors: Set[HardwareVendor] = field(default_factory=set)
    supported_models: Set[str] = field(default_factory=set)
    requires_capabilities: Set[str] = field(default_factory=set)
    incompatible_capabilities: Set[str] = field(default_factory=set)
    performance_impact: Dict[str, float] = field(default_factory=dict)
    power_impact: Optional[float] = None
    thermal_impact: Optional[float] = None
    memory_impact: Optional[float] = None


@dataclass
class HardwareRelationship:
    """Defines a relationship between hardware types."""
    source_hardware: Union[HardwareClass, HardwareArchitecture, str]
    target_hardware: Union[HardwareClass, HardwareArchitecture, str]
    relationship_type: str  # e.g., "parent_of", "compatible_with", "accelerates"
    compatibility_score: float = 0.0  # 0.0 to 1.0
    data_transfer_efficiency: Optional[float] = None  # 0.0 to 1.0
    shared_memory: bool = False
    properties: Dict[str, Any] = field(default_factory=dict)


class EnhancedHardwareTaxonomy(HardwareTaxonomy):
    """
    Enhanced hardware taxonomy with capability registry and relationship modeling.
    
    Extends the base HardwareTaxonomy with:
    - Centralized registry of hardware capabilities
    - Hardware relationship modeling
    - Capability inheritance for hardware hierarchies
    - Dynamic capability discovery
    """
    
    def __init__(self):
        super().__init__()
        # Central registry of capabilities
        self.capabilities_registry: Dict[str, CapabilityDefinition] = {}
        
        # Hardware hierarchy relationships (parent-child)
        self.hardware_hierarchies: Dict[Union[HardwareClass, str], List[Tuple[Union[HardwareClass, str], float]]] = {}
        
        # General hardware relationships (compatibility, acceleration, etc.)
        self.hardware_relationships: Dict[str, HardwareRelationship] = {}
        
        # Hardware capability instances (which hardware has which capabilities)
        self.hardware_capabilities: Dict[str, Set[str]] = {}
        
        # Cached inherited capabilities for performance
        self._inherited_capabilities_cache: Dict[str, Dict[str, Any]] = {}
        
        # Initialize default hardware hierarchies
        self._initialize_default_hierarchies()
        
        # Initialize default capabilities
        self._initialize_default_capabilities()
    
    def _initialize_default_hierarchies(self):
        """Initialize default hardware hierarchies."""
        # CPU is a general processor class
        self.define_hardware_hierarchy(HardwareClass.CPU, HardwareClass.GPU, 0.7)
        self.define_hardware_hierarchy(HardwareClass.CPU, HardwareClass.TPU, 0.4)
        self.define_hardware_hierarchy(HardwareClass.CPU, HardwareClass.NPU, 0.5)
        
        # Architecture hierarchies
        self.define_hardware_hierarchy(HardwareArchitecture.X86_64, HardwareArchitecture.GPU_CUDA, 0.8)
        self.define_hardware_hierarchy(HardwareArchitecture.X86_64, HardwareArchitecture.GPU_ROCM, 0.8)
        self.define_hardware_hierarchy(HardwareArchitecture.ARM64, HardwareArchitecture.GPU_METAL, 0.9)
        self.define_hardware_hierarchy(HardwareArchitecture.ARM64, HardwareArchitecture.NPU_QUALCOMM, 0.9)
    
    def _initialize_default_capabilities(self):
        """Initialize default capabilities in the registry."""
        # Matrix multiplication capability
        self.register_capability(
            capability_id="matrix_multiplication",
            name="Matrix Multiplication",
            description="Ability to perform efficient matrix multiplication operations",
            scope=CapabilityScope.GLOBAL,
            properties={
                "variants": ["gemm", "batched_gemm", "strided_gemm"],
                "datatypes": ["float32", "float16", "int8"],
            },
            supported_hardware_classes={
                HardwareClass.CPU, HardwareClass.GPU, HardwareClass.TPU, 
                HardwareClass.NPU, HardwareClass.FPGA
            }
        )
        
        # Tensor core acceleration
        self.register_capability(
            capability_id="tensor_core_acceleration",
            name="Tensor Core Acceleration",
            description="Hardware-accelerated tensor operations using specialized cores",
            scope=CapabilityScope.CLASS,
            properties={
                "acceleration_factor": 4.0,
                "supported_operations": ["matmul", "conv"]
            },
            supported_hardware_classes={HardwareClass.GPU, HardwareClass.TPU},
            supported_vendors={HardwareVendor.NVIDIA, HardwareVendor.GOOGLE},
            requires_capabilities={"matrix_multiplication"}
        )
        
        # Low-precision computation
        self.register_capability(
            capability_id="low_precision_computation",
            name="Low-Precision Computation",
            description="Support for efficient low-precision (INT8/INT4) computation",
            scope=CapabilityScope.VENDOR,
            properties={
                "min_precision": "int4",
                "optimal_precision": "int8"
            },
            supported_hardware_classes={
                HardwareClass.GPU, HardwareClass.TPU, HardwareClass.NPU
            },
            performance_impact={"throughput": 2.5, "latency": 0.8},
            power_impact=0.6  # Reduces power consumption to 60% of FP32
        )
        
        # WebGPU compute shaders
        self.register_capability(
            capability_id="webgpu_compute_shaders",
            name="WebGPU Compute Shaders",
            description="Support for WebGPU compute shader operations",
            scope=CapabilityScope.MODEL,
            properties={
                "workgroup_size": 256,
                "max_compute_invocations": 16384
            },
            supported_hardware_classes={HardwareClass.GPU, HardwareClass.HYBRID},
            supported_architectures={HardwareArchitecture.GPU_WEBGPU}
        )
    
    def register_capability(self, capability_id: str, name: str, description: str,
                           scope: CapabilityScope, properties: Dict[str, Any] = None,
                           supported_hardware_classes: Set[HardwareClass] = None,
                           supported_architectures: Set[HardwareArchitecture] = None,
                           supported_vendors: Set[HardwareVendor] = None,
                           supported_models: Set[str] = None,
                           requires_capabilities: Set[str] = None,
                           incompatible_capabilities: Set[str] = None,
                           performance_impact: Dict[str, float] = None,
                           power_impact: Optional[float] = None,
                           thermal_impact: Optional[float] = None,
                           memory_impact: Optional[float] = None) -> CapabilityDefinition:
        """
        Register a hardware capability in the central registry.
        
        Args:
            capability_id: Unique identifier for the capability
            name: Human-readable name for the capability
            description: Detailed description of the capability
            scope: Scope at which the capability applies
            properties: Dictionary of capability-specific properties
            supported_hardware_classes: Set of hardware classes supporting this capability
            supported_architectures: Set of architectures supporting this capability
            supported_vendors: Set of vendors supporting this capability
            supported_models: Set of model names supporting this capability
            requires_capabilities: Set of capability IDs that must be present
            incompatible_capabilities: Set of capability IDs that cannot be present
            performance_impact: Dictionary of performance impacts (e.g., throughput, latency)
            power_impact: Impact on power consumption (multiplier, < 1 means reduction)
            thermal_impact: Impact on thermal output (multiplier, < 1 means reduction)
            memory_impact: Impact on memory consumption (multiplier, < 1 means reduction)
            
        Returns:
            CapabilityDefinition: The registered capability definition
        """
        capability = CapabilityDefinition(
            capability_id=capability_id,
            name=name,
            description=description,
            scope=scope,
            properties=properties or {},
            supported_hardware_classes=supported_hardware_classes or set(),
            supported_architectures=supported_architectures or set(),
            supported_vendors=supported_vendors or set(),
            supported_models=supported_models or set(),
            requires_capabilities=requires_capabilities or set(),
            incompatible_capabilities=incompatible_capabilities or set(),
            performance_impact=performance_impact or {},
            power_impact=power_impact,
            thermal_impact=thermal_impact,
            memory_impact=memory_impact
        )
        
        self.capabilities_registry[capability_id] = capability
        return capability
    
    def get_capability(self, capability_id: str) -> Optional[CapabilityDefinition]:
        """
        Get a capability definition from the registry.
        
        Args:
            capability_id: ID of the capability to retrieve
            
        Returns:
            Optional[CapabilityDefinition]: The capability definition, or None if not found
        """
        return self.capabilities_registry.get(capability_id)
    
    def define_hardware_hierarchy(self, parent_hardware: Union[HardwareClass, HardwareArchitecture, str],
                                child_hardware: Union[HardwareClass, HardwareArchitecture, str] = None,
                                inheritance_factor: float = 1.0):
        """
        Define a hierarchical relationship between hardware types.
        
        Args:
            parent_hardware: Parent hardware class, architecture, or model
            child_hardware: Child hardware class, architecture, or model
            inheritance_factor: Factor for capability inheritance (0.0 to 1.0)
        """
        if parent_hardware not in self.hardware_hierarchies:
            self.hardware_hierarchies[parent_hardware] = []
        
        # Add child to parent's hierarchy with inheritance factor
        if child_hardware is not None:
            self.hardware_hierarchies[parent_hardware].append((child_hardware, inheritance_factor))
            
            # Create a relationship record for more detailed information
            parent_type = "class" if isinstance(parent_hardware, HardwareClass) else "architecture"
            child_type = "class" if isinstance(child_hardware, HardwareClass) else "architecture"
            
            relationship_id = f"{parent_type}:{parent_hardware.value if hasattr(parent_hardware, 'value') else parent_hardware}_" \
                             f"{child_type}:{child_hardware.value if hasattr(child_hardware, 'value') else child_hardware}"
            
            self.hardware_relationships[relationship_id] = HardwareRelationship(
                source_hardware=parent_hardware,
                target_hardware=child_hardware,
                relationship_type="parent_of",
                compatibility_score=inheritance_factor,
                properties={
                    "inheritance_factor": inheritance_factor,
                    "parent_type": parent_type,
                    "child_type": child_type
                }
            )
        
        # Clear the cache when relationships change
        self._inherited_capabilities_cache.clear()
    
    def register_hardware_relationship(self, source_hardware: Union[HardwareClass, HardwareArchitecture, str],
                                     source_type: str,
                                     target_hardware: Union[HardwareClass, HardwareArchitecture, str],
                                     target_type: str,
                                     relationship_type: str,
                                     compatibility_score: float = 0.0,
                                     data_transfer_efficiency: Optional[float] = None,
                                     shared_memory: bool = False,
                                     properties: Dict[str, Any] = None) -> HardwareRelationship:
        """
        Register a general relationship between hardware types.
        
        Args:
            source_hardware: Source hardware class, architecture, or model
            source_type: Type of the source identifier ("class", "architecture", "model")
            target_hardware: Target hardware class, architecture, or model
            target_type: Type of the target identifier ("class", "architecture", "model")
            relationship_type: Type of relationship (e.g., "compatible_with", "accelerates")
            compatibility_score: Compatibility score between hardware (0.0 to 1.0)
            data_transfer_efficiency: Efficiency of data transfer between hardware (0.0 to 1.0)
            shared_memory: Whether the hardware shares memory
            properties: Additional properties of the relationship
            
        Returns:
            HardwareRelationship: The registered relationship
        """
        relationship_id = f"{source_type}:{source_hardware.value if hasattr(source_hardware, 'value') else source_hardware}_" \
                         f"{relationship_type}_" \
                         f"{target_type}:{target_hardware.value if hasattr(target_hardware, 'value') else target_hardware}"
        
        relationship = HardwareRelationship(
            source_hardware=source_hardware,
            target_hardware=target_hardware,
            relationship_type=relationship_type,
            compatibility_score=compatibility_score,
            data_transfer_efficiency=data_transfer_efficiency,
            shared_memory=shared_memory,
            properties=properties or {}
        )
        
        self.hardware_relationships[relationship_id] = relationship
        return relationship
    
    def get_hardware_relationships(self, hardware: Union[HardwareClass, HardwareArchitecture, str],
                                 hardware_type: str = "class",
                                 relationship_type: Optional[str] = None) -> List[HardwareRelationship]:
        """
        Get all relationships for a specific hardware.
        
        Args:
            hardware: Hardware class, architecture, or model
            hardware_type: Type of the hardware identifier ("class", "architecture", "model")
            relationship_type: Optional filter for relationship type
            
        Returns:
            List[HardwareRelationship]: Matching relationships
        """
        hw_value = hardware.value if hasattr(hardware, 'value') else hardware
        prefix = f"{hardware_type}:{hw_value}"
        
        relationships = []
        for rel_id, relationship in self.hardware_relationships.items():
            if rel_id.startswith(prefix) and (relationship_type is None or relationship.relationship_type == relationship_type):
                relationships.append(relationship)
        
        return relationships
    
    def assign_capability_to_hardware(self, hardware_profile: HardwareCapabilityProfile,
                                    capability_id: str, 
                                    property_overrides: Dict[str, Any] = None):
        """
        Assign a capability to a specific hardware profile.
        
        Args:
            hardware_profile: Hardware profile to assign the capability to
            capability_id: ID of the capability from the registry
            property_overrides: Optional overrides for capability properties
        """
        if capability_id not in self.capabilities_registry:
            raise ValueError(f"Capability '{capability_id}' not found in registry")
        
        # Get capability definition
        capability = self.capabilities_registry[capability_id]
        
        # Check if hardware is compatible with capability
        if capability.supported_hardware_classes and hardware_profile.hardware_class not in capability.supported_hardware_classes:
            raise ValueError(f"Hardware class {hardware_profile.hardware_class} is not compatible with capability '{capability_id}'")
        
        if capability.supported_architectures and hardware_profile.architecture not in capability.supported_architectures:
            raise ValueError(f"Architecture {hardware_profile.architecture} is not compatible with capability '{capability_id}'")
        
        if capability.supported_vendors and hardware_profile.vendor not in capability.supported_vendors:
            raise ValueError(f"Vendor {hardware_profile.vendor} is not compatible with capability '{capability_id}'")
        
        if capability.supported_models and hardware_profile.model_name not in capability.supported_models:
            raise ValueError(f"Model {hardware_profile.model_name} is not compatible with capability '{capability_id}'")
        
        # Check for required capabilities
        hardware_id = self._get_hardware_id(hardware_profile)
        current_capabilities = self.hardware_capabilities.get(hardware_id, set())
        
        for required_cap in capability.requires_capabilities:
            if required_cap not in current_capabilities:
                raise ValueError(f"Capability '{capability_id}' requires capability '{required_cap}' which is not present")
        
        # Check for incompatible capabilities
        for incompatible_cap in capability.incompatible_capabilities:
            if incompatible_cap in current_capabilities:
                raise ValueError(f"Capability '{capability_id}' is incompatible with capability '{incompatible_cap}' which is present")
        
        # Add capability to hardware
        if hardware_id not in self.hardware_capabilities:
            self.hardware_capabilities[hardware_id] = set()
        
        self.hardware_capabilities[hardware_id].add(capability_id)
        
        # Store property overrides if provided
        if property_overrides:
            # This would require additional storage for hardware-specific capability properties
            # Simplified implementation for now
            pass
        
        # Clear cache for this hardware
        if hardware_id in self._inherited_capabilities_cache:
            del self._inherited_capabilities_cache[hardware_id]
    
    def has_capability(self, hardware_profile: HardwareCapabilityProfile, capability_id: str) -> bool:
        """
        Check if a hardware profile has a specific capability.
        
        Args:
            hardware_profile: Hardware profile to check
            capability_id: ID of the capability
            
        Returns:
            bool: True if the hardware has the capability, False otherwise
        """
        hardware_id = self._get_hardware_id(hardware_profile)
        capabilities = self.hardware_capabilities.get(hardware_id, set())
        return capability_id in capabilities
    
    def get_hardware_capabilities(self, hardware_profile: HardwareCapabilityProfile, 
                                include_inherited: bool = True) -> Dict[str, CapabilityDefinition]:
        """
        Get all capabilities for a hardware profile.
        
        Args:
            hardware_profile: Hardware profile to get capabilities for
            include_inherited: Whether to include inherited capabilities
            
        Returns:
            Dict[str, CapabilityDefinition]: Dictionary of capability IDs to capability definitions
        """
        hardware_id = self._get_hardware_id(hardware_profile)
        direct_capabilities = self.hardware_capabilities.get(hardware_id, set())
        
        result = {}
        for cap_id in direct_capabilities:
            if cap_id in self.capabilities_registry:
                result[cap_id] = self.capabilities_registry[cap_id]
        
        if include_inherited:
            # Get inherited capabilities
            inherited_caps = self.get_inherited_capabilities(hardware_profile)
            for cap_id, cap_def in inherited_caps.items():
                if cap_id not in result:  # Direct capabilities take precedence
                    result[cap_id] = cap_def
        
        return result
    
    def get_inherited_capabilities(self, hardware_profile: HardwareCapabilityProfile) -> Dict[str, CapabilityDefinition]:
        """
        Get all inherited capabilities for a hardware profile.
        
        This method traverses the hardware hierarchy to find capabilities that might
        be inherited from parent hardware classes/architectures.
        
        Args:
            hardware_profile: Hardware profile to get inherited capabilities for
            
        Returns:
            Dict[str, CapabilityDefinition]: Dictionary of inherited capability IDs to capability definitions
        """
        hardware_id = self._get_hardware_id(hardware_profile)
        
        # Check cache first
        if hardware_id in self._inherited_capabilities_cache:
            return self._inherited_capabilities_cache[hardware_id]
        
        result = {}
        
        # Check class hierarchy
        self._add_inherited_capabilities_from_class(hardware_profile.hardware_class, result)
        
        # Check architecture hierarchy
        self._add_inherited_capabilities_from_architecture(hardware_profile.architecture, result)
        
        # Store in cache for future use
        self._inherited_capabilities_cache[hardware_id] = result
        return result
    
    def _add_inherited_capabilities_from_class(self, hardware_class: HardwareClass, 
                                             result: Dict[str, CapabilityDefinition],
                                             visited: Set[HardwareClass] = None):
        """
        Add inherited capabilities from a hardware class hierarchy.
        
        Args:
            hardware_class: Hardware class to get capabilities from
            result: Dictionary to add capabilities to
            visited: Set of already visited classes to prevent cycles
        """
        if visited is None:
            visited = set()
        
        if hardware_class in visited:
            return
        
        visited.add(hardware_class)
        
        # Check for parent classes in the hierarchy
        for parent, children in self.hardware_hierarchies.items():
            if not isinstance(parent, HardwareClass):
                continue
                
            for child, inheritance_factor in children:
                if child == hardware_class and inheritance_factor > 0:
                    # Found a parent, check for capabilities
                    for cap_id, cap_def in self.capabilities_registry.items():
                        if parent in cap_def.supported_hardware_classes:
                            # Inherit the capability
                            result[cap_id] = cap_def
                    
                    # Recursively check parent's hierarchy
                    self._add_inherited_capabilities_from_class(parent, result, visited)
    
    def _add_inherited_capabilities_from_architecture(self, architecture: HardwareArchitecture,
                                                   result: Dict[str, CapabilityDefinition],
                                                   visited: Set[HardwareArchitecture] = None):
        """
        Add inherited capabilities from an architecture hierarchy.
        
        Args:
            architecture: Hardware architecture to get capabilities from
            result: Dictionary to add capabilities to
            visited: Set of already visited architectures to prevent cycles
        """
        if visited is None:
            visited = set()
        
        if architecture in visited:
            return
        
        visited.add(architecture)
        
        # Check for parent architectures in the hierarchy
        for parent, children in self.hardware_hierarchies.items():
            if not isinstance(parent, HardwareArchitecture):
                continue
                
            for child, inheritance_factor in children:
                if child == architecture and inheritance_factor > 0:
                    # Found a parent, check for capabilities
                    for cap_id, cap_def in self.capabilities_registry.items():
                        if parent in cap_def.supported_architectures:
                            # Inherit the capability
                            result[cap_id] = cap_def
                    
                    # Recursively check parent's hierarchy
                    self._add_inherited_capabilities_from_architecture(parent, result, visited)
    
    def discover_capabilities(self, hardware_profile: HardwareCapabilityProfile) -> Set[str]:
        """
        Dynamically discover capabilities for a hardware profile.
        
        This method analyzes the hardware profile to identify capabilities
        that it might have based on its characteristics, even if not explicitly
        assigned.
        
        Args:
            hardware_profile: Hardware profile to discover capabilities for
            
        Returns:
            Set[str]: Set of discovered capability IDs
        """
        discovered = set()
        
        # Use hardware class, architecture, vendor, features, and other attributes
        # to infer capabilities
        
        # Example: GPUs with compute_units >= 30 might have tensor operations capability
        if hardware_profile.hardware_class == HardwareClass.GPU and hardware_profile.compute_units >= 30:
            discovered.add("tensor_operations")
        
        # Example: Hardware with AVX2 feature might have SIMD capability
        if AcceleratorFeature.AVX2 in hardware_profile.features:
            discovered.add("simd_256bit")
        
        # Example: GPUs with CUDA architecture might have unified memory capability
        if hardware_profile.architecture == HardwareArchitecture.GPU_CUDA and \
           hardware_profile.memory.has_unified_memory:
            discovered.add("unified_memory")
        
        # Example: Tensor cores imply tensor core acceleration
        if AcceleratorFeature.TENSOR_CORES in hardware_profile.features:
            discovered.add("tensor_core_acceleration")
        
        # Example: FP16 precision support implies mixed precision capability
        if PrecisionType.FP16 in hardware_profile.supported_precisions:
            discovered.add("mixed_precision")
        
        # Example: INT8 precision support implies quantization capability
        if PrecisionType.INT8 in hardware_profile.supported_precisions:
            discovered.add("quantization")
        
        # Return only capabilities that exist in the registry
        return {cap_id for cap_id in discovered if cap_id in self.capabilities_registry}
    
    def auto_assign_capabilities(self, hardware_profile: HardwareCapabilityProfile) -> Set[str]:
        """
        Automatically assign discovered capabilities to a hardware profile.
        
        Args:
            hardware_profile: Hardware profile to assign capabilities to
            
        Returns:
            Set[str]: Set of assigned capability IDs
        """
        discovered = self.discover_capabilities(hardware_profile)
        
        for cap_id in discovered:
            try:
                self.assign_capability_to_hardware(hardware_profile, cap_id)
            except ValueError:
                # Skip capabilities that can't be assigned
                pass
        
        hardware_id = self._get_hardware_id(hardware_profile)
        return self.hardware_capabilities.get(hardware_id, set())
    
    def register_hardware_profile(self, profile: HardwareCapabilityProfile, auto_discover: bool = True):
        """
        Override to add auto-discovery of capabilities.
        
        Args:
            profile: The hardware capability profile to register
            auto_discover: Whether to automatically discover capabilities
        """
        # Call parent method to register the profile
        super().register_hardware_profile(profile)
        
        # Automatically discover and assign capabilities if requested
        if auto_discover:
            self.auto_assign_capabilities(profile)
    
    def calculate_workload_capability_match(self, workload_type: str, 
                                         required_capabilities: Set[str],
                                         hardware_profile: HardwareCapabilityProfile) -> float:
        """
        Calculate how well a hardware profile matches required capabilities for a workload.
        
        Args:
            workload_type: Type of workload
            required_capabilities: Set of capability IDs required by the workload
            hardware_profile: Hardware profile to evaluate
            
        Returns:
            float: Match score (0.0 to 1.0)
        """
        if not required_capabilities:
            return 1.0
        
        # Get all capabilities for the hardware
        hardware_capabilities = self.get_hardware_capabilities(hardware_profile)
        
        # Count matching capabilities
        matching = 0
        for req_cap in required_capabilities:
            if req_cap in hardware_capabilities:
                matching += 1
        
        return matching / len(required_capabilities)
    
    def _get_hardware_id(self, hardware_profile: HardwareCapabilityProfile) -> str:
        """
        Generate a unique ID for a hardware profile.
        
        Args:
            hardware_profile: Hardware profile to generate ID for
            
        Returns:
            str: Unique hardware ID
        """
        return f"{hardware_profile.hardware_class.value}:{hardware_profile.architecture.value}:{hardware_profile.vendor.value}:{hardware_profile.model_name}"