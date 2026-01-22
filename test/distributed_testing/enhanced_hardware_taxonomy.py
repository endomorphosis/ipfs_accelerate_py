#!/usr/bin/env python3
"""
Enhanced Hardware Taxonomy for Distributed Testing Framework

This module extends the base Hardware Taxonomy to provide a more sophisticated
classification system for heterogeneous hardware environments, enabling advanced
hardware-aware workload management.

Key enhancements:
- Capabilities Registry for centralized capability tracking
- Hardware Relationship Modeling for understanding hardware hierarchies
- Capability Inheritance for propagating capabilities through hierarchies
- Dynamic Capability Discovery for runtime capability detection
"""

from typing import Dict, List, Any, Optional, Set, Tuple, Union
import logging
from dataclasses import dataclass, field

from duckdb_api.distributed_testing.hardware_taxonomy import (
    HardwareTaxonomy, HardwareCapabilityProfile, HardwareClass,
    HardwareArchitecture, HardwareVendor, SoftwareBackend,
    PrecisionType, AcceleratorFeature
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("enhanced_hardware_taxonomy")


@dataclass
class CapabilityDefinition:
    """Definition of a hardware capability with properties and relationships."""
    capability_id: str
    capability_type: str  # e.g., "compute", "memory", "io", "network"
    properties: Dict[str, Any] = field(default_factory=dict)
    related_capabilities: List[str] = field(default_factory=list)
    hardware_requirements: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None


@dataclass
class HardwareHierarchy:
    """Represents a hierarchical relationship between hardware classes."""
    parent_class: HardwareClass
    child_class: HardwareClass
    inheritance_factor: float = 1.0  # How strongly capabilities are inherited (0.0-1.0)
    capability_filters: Set[str] = field(default_factory=set)  # Capabilities that are not inherited


@dataclass
class HardwareRelationship:
    """Represents a relationship between hardware profiles."""
    source_id: str
    target_id: str
    relationship_type: str  # e.g., "depends_on", "accelerates", "cooperates_with"
    strength: float = 1.0  # Strength of the relationship (0.0-1.0)
    properties: Dict[str, Any] = field(default_factory=dict)


class EnhancedHardwareTaxonomy(HardwareTaxonomy):
    """
    Enhanced hardware taxonomy with capabilities registry and relationship modeling.
    
    This class extends the base HardwareTaxonomy with more sophisticated features
    for hardware classification, capability tracking, and relationship modeling.
    """
    
    def __init__(self):
        """Initialize the enhanced hardware taxonomy."""
        super().__init__()
        
        # Capabilities registry
        self.capabilities_registry: Dict[str, CapabilityDefinition] = {}
        
        # Hardware hierarchies (parent -> children)
        self.hardware_hierarchies: Dict[HardwareClass, List[HardwareHierarchy]] = {}
        
        # Hardware relationships (profile_id -> relationships)
        self.hardware_relationships: Dict[str, List[HardwareRelationship]] = {}
        
        # Hardware capabilities (profile_id -> capabilities)
        self.hardware_capabilities: Dict[str, Dict[str, Any]] = {}
        
        # Capability inheritance cache
        self.capability_inheritance_cache: Dict[str, Dict[str, Any]] = {}
        
        # Initialize with common capability definitions
        self._initialize_common_capabilities()
        
        # Initialize hardware hierarchies
        self._initialize_hardware_hierarchies()
    
    def _initialize_common_capabilities(self):
        """Initialize common capability definitions in the registry."""
        # Compute capabilities
        self.register_capability(
            CapabilityDefinition(
                capability_id="compute.matrix_multiplication",
                capability_type="compute",
                properties={
                    "operations": ["matmul", "gemm"],
                    "dimensionality": ["2D"]
                },
                description="Ability to perform matrix multiplication operations"
            )
        )
        
        self.register_capability(
            CapabilityDefinition(
                capability_id="compute.convolution",
                capability_type="compute",
                properties={
                    "operations": ["conv1d", "conv2d", "conv3d"],
                    "dimensionality": ["1D", "2D", "3D"]
                },
                description="Ability to perform convolution operations"
            )
        )
        
        self.register_capability(
            CapabilityDefinition(
                capability_id="compute.tensor_core_acceleration",
                capability_type="compute",
                properties={
                    "operations": ["matmul", "conv2d"],
                    "precisions": ["fp16", "int8"]
                },
                related_capabilities=["compute.matrix_multiplication", "compute.convolution"],
                hardware_requirements={
                    "features": ["tensor_cores"]
                },
                description="Hardware acceleration for matrix and tensor operations using tensor cores"
            )
        )
        
        # Memory capabilities
        self.register_capability(
            CapabilityDefinition(
                capability_id="memory.high_bandwidth",
                capability_type="memory",
                properties={
                    "min_bandwidth_gbps": 500.0
                },
                description="High bandwidth memory access"
            )
        )
        
        self.register_capability(
            CapabilityDefinition(
                capability_id="memory.unified",
                capability_type="memory",
                properties={
                    "shared_address_space": True
                },
                description="Unified memory architecture with shared address space"
            )
        )
        
        # Precision capabilities
        self.register_capability(
            CapabilityDefinition(
                capability_id="precision.mixed",
                capability_type="precision",
                properties={
                    "supported_precisions": ["fp32", "fp16", "int8"],
                    "automatic_casting": True
                },
                description="Support for mixed precision operations with automatic casting"
            )
        )
        
        self.register_capability(
            CapabilityDefinition(
                capability_id="precision.quantization",
                capability_type="precision",
                properties={
                    "supported_precisions": ["int8", "int4", "int2"],
                    "quantization_schemes": ["symmetric", "asymmetric"]
                },
                description="Support for quantized operations at various bit widths"
            )
        )
        
        # Specialized capabilities
        self.register_capability(
            CapabilityDefinition(
                capability_id="specialized.vision",
                capability_type="specialized",
                properties={
                    "operations": ["conv2d", "pooling", "normalization"],
                    "optimized_for": ["image_classification", "object_detection"]
                },
                related_capabilities=["compute.convolution"],
                description="Specialized support for vision workloads"
            )
        )
        
        self.register_capability(
            CapabilityDefinition(
                capability_id="specialized.nlp",
                capability_type="specialized",
                properties={
                    "operations": ["matmul", "attention"],
                    "optimized_for": ["transformer", "sequence_modeling"]
                },
                related_capabilities=["compute.matrix_multiplication"],
                description="Specialized support for NLP workloads"
            )
        )
        
        self.register_capability(
            CapabilityDefinition(
                capability_id="specialized.audio",
                capability_type="specialized",
                properties={
                    "operations": ["fft", "conv1d"],
                    "optimized_for": ["audio_processing", "speech_recognition"]
                },
                description="Specialized support for audio workloads"
            )
        )
    
    def _initialize_hardware_hierarchies(self):
        """Initialize hardware class hierarchies."""
        # CPU-based hierarchies
        self.define_hardware_hierarchy(
            HardwareClass.CPU, HardwareClass.CPU, 1.0
        )
        
        # GPU-based hierarchies
        self.define_hardware_hierarchy(
            HardwareClass.GPU, HardwareClass.GPU, 0.8
        )
        
        # CPU to other accelerators
        self.define_hardware_hierarchy(
            HardwareClass.CPU, HardwareClass.GPU, 0.5,
            capability_filters={"memory.unified"}
        )
        self.define_hardware_hierarchy(
            HardwareClass.CPU, HardwareClass.TPU, 0.3
        )
        self.define_hardware_hierarchy(
            HardwareClass.CPU, HardwareClass.NPU, 0.4
        )
        
        # GPU to other accelerators
        self.define_hardware_hierarchy(
            HardwareClass.GPU, HardwareClass.TPU, 0.4
        )
        self.define_hardware_hierarchy(
            HardwareClass.GPU, HardwareClass.NPU, 0.4
        )
        
        # Hybrid devices (e.g., browser-based acceleration)
        self.define_hardware_hierarchy(
            HardwareClass.HYBRID, HardwareClass.GPU, 0.7,
            capability_filters={"memory.high_bandwidth"}
        )
        self.define_hardware_hierarchy(
            HardwareClass.HYBRID, HardwareClass.CPU, 0.9
        )
    
    def register_capability(self, capability: CapabilityDefinition):
        """
        Register a hardware capability in the central registry.
        
        Args:
            capability: Capability definition to register
        """
        self.capabilities_registry[capability.capability_id] = capability
        logger.debug(f"Registered capability: {capability.capability_id}")
    
    def register_hardware_capability(self, hardware_id: str, capability_id: str, properties: Dict[str, Any] = None):
        """
        Register a capability for a specific hardware profile.
        
        Args:
            hardware_id: ID of the hardware profile
            capability_id: ID of the capability to register
            properties: Optional specific properties for this hardware's implementation
                       of the capability
        """
        if hardware_id not in self.hardware_capabilities:
            self.hardware_capabilities[hardware_id] = {}
        
        if capability_id not in self.capabilities_registry:
            logger.warning(f"Unknown capability ID: {capability_id}, not registered")
            return
        
        # Register capability with hardware-specific properties
        self.hardware_capabilities[hardware_id][capability_id] = properties or {}
        
        # Invalidate inheritance cache for this hardware
        if hardware_id in self.capability_inheritance_cache:
            del self.capability_inheritance_cache[hardware_id]
    
    def define_hardware_hierarchy(self, parent_class: HardwareClass, 
                                 child_class: HardwareClass, 
                                 inheritance_factor: float = 1.0,
                                 capability_filters: Set[str] = None):
        """
        Define hierarchical relationship between hardware classes.
        
        Args:
            parent_class: Parent hardware class
            child_class: Child hardware class
            inheritance_factor: Factor for capability inheritance strength (0.0-1.0)
            capability_filters: Set of capability IDs that should not be inherited
        """
        if parent_class not in self.hardware_hierarchies:
            self.hardware_hierarchies[parent_class] = []
        
        hierarchy = HardwareHierarchy(
            parent_class=parent_class,
            child_class=child_class,
            inheritance_factor=inheritance_factor,
            capability_filters=capability_filters or set()
        )
        
        self.hardware_hierarchies[parent_class].append(hierarchy)
        logger.debug(f"Defined hardware hierarchy: {parent_class.value} -> {child_class.value} "
                   f"(factor: {inheritance_factor})")
        
        # Clear inheritance cache as hierarchies have changed
        self.capability_inheritance_cache.clear()
    
    def register_hardware_relationship(self, relationship: HardwareRelationship):
        """
        Register a relationship between hardware profiles.
        
        Args:
            relationship: Relationship to register
        """
        if relationship.source_id not in self.hardware_relationships:
            self.hardware_relationships[relationship.source_id] = []
        
        self.hardware_relationships[relationship.source_id].append(relationship)
        logger.debug(f"Registered hardware relationship: {relationship.source_id} -> {relationship.target_id} "
                   f"({relationship.relationship_type})")
    
    def get_capability_definition(self, capability_id: str) -> Optional[CapabilityDefinition]:
        """
        Get the definition of a capability.
        
        Args:
            capability_id: ID of the capability
            
        Returns:
            Capability definition or None if not found
        """
        return self.capabilities_registry.get(capability_id)
    
    def get_hardware_capabilities(self, hardware_id: str) -> Dict[str, Any]:
        """
        Get the capabilities of a hardware profile.
        
        Args:
            hardware_id: ID of the hardware profile
            
        Returns:
            Dictionary of capability_id -> properties
        """
        return self.hardware_capabilities.get(hardware_id, {})
    
    def get_inherited_capabilities(self, hardware_profile: HardwareCapabilityProfile) -> Dict[str, Any]:
        """
        Get all capabilities including inherited ones for a hardware profile.
        
        Args:
            hardware_profile: Hardware capability profile
            
        Returns:
            Dictionary of capability_id -> properties with inherited capabilities
        """
        # Create a unique ID for this profile
        profile_id = f"{hardware_profile.hardware_class.value}_{hardware_profile.model_name}"
        
        # Check cache first
        if profile_id in self.capability_inheritance_cache:
            return self.capability_inheritance_cache[profile_id]
        
        # Start with directly registered capabilities
        capabilities = self.get_hardware_capabilities(profile_id).copy()
        
        # Add inferred capabilities based on hardware properties
        inferred_capabilities = self._infer_capabilities(hardware_profile)
        for cap_id, props in inferred_capabilities.items():
            if cap_id not in capabilities:
                capabilities[cap_id] = props
        
        # Add inherited capabilities from hardware hierarchies
        inherited_capabilities = self._get_inherited_capabilities_from_hierarchies(hardware_profile)
        for cap_id, props in inherited_capabilities.items():
            if cap_id not in capabilities:
                capabilities[cap_id] = props
        
        # Cache the result
        self.capability_inheritance_cache[profile_id] = capabilities
        
        return capabilities
    
    def _infer_capabilities(self, hardware_profile: HardwareCapabilityProfile) -> Dict[str, Any]:
        """
        Infer capabilities based on hardware properties.
        
        Args:
            hardware_profile: Hardware capability profile
            
        Returns:
            Dictionary of inferred capability_id -> properties
        """
        inferred_capabilities = {}
        
        # Infer compute capabilities
        if hardware_profile.compute_units > 0:
            inferred_capabilities["compute.matrix_multiplication"] = {
                "performance": hardware_profile.compute_units * (hardware_profile.clock_speed_mhz or 1000) * 0.001
            }
        
        # Infer tensor core acceleration
        if AcceleratorFeature.TENSOR_CORES in hardware_profile.features:
            inferred_capabilities["compute.tensor_core_acceleration"] = {
                "performance_multiplier": 4.0
            }
        
        # Infer memory capabilities
        if hardware_profile.memory.bandwidth_gbps and hardware_profile.memory.bandwidth_gbps > 500:
            inferred_capabilities["memory.high_bandwidth"] = {
                "bandwidth_gbps": hardware_profile.memory.bandwidth_gbps
            }
        
        if hardware_profile.memory.has_unified_memory:
            inferred_capabilities["memory.unified"] = {
                "shared_address_space": True
            }
        
        # Infer precision capabilities
        if len(hardware_profile.supported_precisions) >= 3:
            inferred_capabilities["precision.mixed"] = {
                "supported_precisions": [p.value for p in hardware_profile.supported_precisions]
            }
        
        if (PrecisionType.INT8 in hardware_profile.supported_precisions or
            PrecisionType.INT4 in hardware_profile.supported_precisions):
            inferred_capabilities["precision.quantization"] = {
                "supported_precisions": [p.value for p in hardware_profile.supported_precisions 
                                      if p in [PrecisionType.INT8, PrecisionType.INT4, PrecisionType.INT2, PrecisionType.INT1]]
            }
        
        # Infer specialized capabilities
        if hardware_profile.hardware_class == HardwareClass.GPU and hardware_profile.compute_units > 20:
            inferred_capabilities["specialized.vision"] = {
                "effectiveness": min(1.0, hardware_profile.compute_units / 100.0)
            }
            
            if AcceleratorFeature.TENSOR_CORES in hardware_profile.features:
                inferred_capabilities["specialized.nlp"] = {
                    "effectiveness": min(1.0, hardware_profile.compute_units / 80.0)
                }
        
        if hardware_profile.hardware_class == HardwareClass.CPU and AcceleratorFeature.AVX2 in hardware_profile.features:
            inferred_capabilities["specialized.audio"] = {
                "effectiveness": 0.8
            }
        
        return inferred_capabilities
    
    def _get_inherited_capabilities_from_hierarchies(self, hardware_profile: HardwareCapabilityProfile) -> Dict[str, Any]:
        """
        Get capabilities inherited from hardware hierarchies.
        
        Args:
            hardware_profile: Hardware capability profile
            
        Returns:
            Dictionary of inherited capability_id -> properties
        """
        inherited_capabilities = {}
        profile_class = hardware_profile.hardware_class
        
        # Look for hierarchies where this profile is a child
        for parent_class, hierarchies in self.hardware_hierarchies.items():
            for hierarchy in hierarchies:
                if hierarchy.child_class == profile_class:
                    # Get parent profiles
                    parent_profiles = []
                    for hw_class, profiles in self.hardware_profiles.items():
                        if HardwareClass(hw_class) == parent_class:
                            parent_profiles.extend(profiles)
                    
                    # Inherit capabilities from each parent
                    for parent_profile in parent_profiles:
                        parent_id = f"{parent_profile.hardware_class.value}_{parent_profile.model_name}"
                        parent_capabilities = self.get_hardware_capabilities(parent_id)
                        
                        # Apply inheritance with factor and filters
                        for cap_id, props in parent_capabilities.items():
                            if cap_id not in hierarchy.capability_filters:
                                # Inherit with factor
                                if cap_id not in inherited_capabilities:
                                    # Make a copy of the properties
                                    inherited_props = props.copy() if isinstance(props, dict) else props
                                    
                                    # Apply inheritance factor to numeric properties
                                    if isinstance(inherited_props, dict):
                                        for key, value in inherited_props.items():
                                            if isinstance(value, (int, float)):
                                                inherited_props[key] = value * hierarchy.inheritance_factor
                                    
                                    inherited_capabilities[cap_id] = inherited_props
        
        return inherited_capabilities
    
    def get_capability_compatibility(self, capability_id1: str, capability_id2: str) -> float:
        """
        Calculate compatibility between two capabilities.
        
        Args:
            capability_id1: First capability ID
            capability_id2: Second capability ID
            
        Returns:
            Compatibility score (0.0-1.0, higher is more compatible)
        """
        # Get definitions
        def1 = self.get_capability_definition(capability_id1)
        def2 = self.get_capability_definition(capability_id2)
        
        if not def1 or not def2:
            return 0.0
        
        # Same capability is perfectly compatible
        if capability_id1 == capability_id2:
            return 1.0
        
        # Check if they are related
        if capability_id2 in def1.related_capabilities:
            return 0.8
        elif capability_id1 in def2.related_capabilities:
            return 0.8
        
        # Check if they are of the same type
        if def1.capability_type == def2.capability_type:
            return 0.5
        
        # Default low compatibility
        return 0.1
    
    def find_hardware_with_capability(self, capability_id: str, min_threshold: float = 0.0) -> List[Tuple[HardwareCapabilityProfile, Dict[str, Any]]]:
        """
        Find hardware profiles with a specific capability.
        
        Args:
            capability_id: Capability ID to search for
            min_threshold: Minimum threshold for capability properties (if numeric)
            
        Returns:
            List of (hardware_profile, capability_properties) tuples
        """
        results = []
        
        # Check all profiles
        for hw_class, profiles in self.hardware_profiles.items():
            for profile in profiles:
                profile_id = f"{profile.hardware_class.value}_{profile.model_name}"
                all_capabilities = self.get_inherited_capabilities(profile)
                
                if capability_id in all_capabilities:
                    # Check threshold if applicable
                    props = all_capabilities[capability_id]
                    if isinstance(props, dict):
                        # For dictionary properties, check numeric values against threshold
                        meets_threshold = True
                        for key, value in props.items():
                            if isinstance(value, (int, float)) and value < min_threshold:
                                meets_threshold = False
                                break
                        
                        if meets_threshold:
                            results.append((profile, props))
                    else:
                        # For non-dictionary properties, add directly
                        results.append((profile, props))
        
        return results
    
    def calculate_workload_hardware_match(self, 
                                        required_capabilities: Dict[str, Any],
                                        hardware_profile: HardwareCapabilityProfile) -> float:
        """
        Calculate how well a hardware profile matches required capabilities.
        
        Args:
            required_capabilities: Dictionary of capability_id -> required properties
            hardware_profile: Hardware capability profile
            
        Returns:
            Match score (0.0-1.0, higher is better)
        """
        # Get all capabilities for this hardware
        hardware_capabilities = self.get_inherited_capabilities(hardware_profile)
        
        # If no requirements, return medium score
        if not required_capabilities:
            return 0.5
        
        # Calculate match score
        total_score = 0.0
        total_weight = 0.0
        
        for cap_id, required_props in required_capabilities.items():
            # Assign weight based on capability type
            cap_def = self.get_capability_definition(cap_id)
            weight = 1.0
            if cap_def:
                if cap_def.capability_type == "compute":
                    weight = 1.2
                elif cap_def.capability_type == "memory":
                    weight = 1.0
                elif cap_def.capability_type == "specialized":
                    weight = 1.5
            
            # Check if capability exists in hardware
            if cap_id in hardware_capabilities:
                hardware_props = hardware_capabilities[cap_id]
                
                # Calculate property match
                prop_match = 1.0
                if isinstance(required_props, dict) and isinstance(hardware_props, dict):
                    # For dictionaries, check property by property
                    for key, required_value in required_props.items():
                        if key in hardware_props:
                            hardware_value = hardware_props[key]
                            
                            # Different matching based on value type
                            if isinstance(required_value, (int, float)) and isinstance(hardware_value, (int, float)):
                                # For numeric values, check if hardware meets or exceeds requirement
                                if hardware_value >= required_value:
                                    # Scale based on how much it exceeds (up to 2x improvement)
                                    ratio = min(2.0, hardware_value / max(1e-6, required_value))
                                    prop_match *= (0.75 + 0.25 * ratio)
                                else:
                                    # Reduce score based on how much it falls short
                                    ratio = hardware_value / max(1e-6, required_value)
                                    prop_match *= max(0.0, ratio)
                            elif isinstance(required_value, list) and isinstance(hardware_value, list):
                                # For lists, check overlap
                                required_set = set(required_value)
                                hardware_set = set(hardware_value)
                                if required_set.issubset(hardware_set):
                                    prop_match *= 1.0
                                else:
                                    overlap = len(required_set.intersection(hardware_set))
                                    total = len(required_set)
                                    prop_match *= max(0.0, overlap / max(1, total))
                            elif required_value != hardware_value:
                                # For other types, check equality
                                prop_match *= 0.5
                        else:
                            # Missing property
                            prop_match *= 0.3
                
                # Add to total score
                total_score += weight * prop_match
                total_weight += weight
            else:
                # Missing capability
                # If it's a critical capability, severely reduce score
                cap_def = self.get_capability_definition(cap_id)
                if cap_def and cap_def.capability_type in ["compute", "specialized"]:
                    # Still add weight, but with zero score for this capability
                    total_weight += weight
                    # No addition to total_score
                else:
                    # For non-critical capabilities, add partial weight
                    total_weight += weight * 0.5
        
        # Calculate final score
        if total_weight > 0:
            return total_score / total_weight
        else:
            return 0.0
    
    def find_optimal_hardware_for_workload(self, 
                                         required_capabilities: Dict[str, Any],
                                         available_workers: Optional[List[str]] = None) -> List[Tuple[str, HardwareCapabilityProfile, float]]:
        """
        Find optimal hardware for a workload based on required capabilities.
        
        Args:
            required_capabilities: Dictionary of capability_id -> required properties
            available_workers: Optional list of available worker IDs, or None for all
            
        Returns:
            List of (worker_id, hardware_profile, match_score) tuples, sorted by match score
        """
        results = []
        valid_worker_ids = available_workers if available_workers is not None else list(self.worker_hardware_map.keys())
        
        for worker_id in valid_worker_ids:
            if worker_id in self.worker_hardware_map:
                worker_profiles = self.worker_hardware_map[worker_id]
                for profile in worker_profiles:
                    match_score = self.calculate_workload_hardware_match(required_capabilities, profile)
                    results.append((worker_id, profile, match_score))
        
        # Sort by match score (descending)
        results.sort(key=lambda item: item[2], reverse=True)
        return results
    
    # Add a method to register a profile with automatic capability inference
    def register_hardware_profile_with_capabilities(self, profile: HardwareCapabilityProfile):
        """
        Register a hardware profile and automatically infer and register its capabilities.
        
        Args:
            profile: Hardware capability profile to register
        """
        # First register the profile with the base taxonomy
        super().register_hardware_profile(profile)
        
        # Then infer capabilities
        profile_id = f"{profile.hardware_class.value}_{profile.model_name}"
        inferred_capabilities = self._infer_capabilities(profile)
        
        # Register inferred capabilities
        for cap_id, props in inferred_capabilities.items():
            self.register_hardware_capability(profile_id, cap_id, props)
        
        logger.info(f"Registered hardware profile with {len(inferred_capabilities)} inferred capabilities: {profile_id}")
    
    # Override the existing method to update specialization map
    def register_worker_hardware(self, worker_id: str, profiles: List[HardwareCapabilityProfile]):
        """
        Register hardware profiles for a worker and update specialization map.
        
        Args:
            worker_id: ID of the worker
            profiles: List of hardware profiles for the worker
        """
        # Register with base taxonomy
        super().register_worker_hardware(worker_id, profiles)
        
        # Register capabilities for each profile
        for profile in profiles:
            profile_id = f"{profile.hardware_class.value}_{profile.model_name}"
            inferred_capabilities = self._infer_capabilities(profile)
            
            # Register inferred capabilities
            for cap_id, props in inferred_capabilities.items():
                self.register_hardware_capability(profile_id, cap_id, props)
        
        # Update specialization map with new profiles
        self.update_specialization_map()
        
        logger.info(f"Registered worker {worker_id} with {len(profiles)} hardware profiles")
    
    def get_capability_map_for_worker(self, worker_id: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Get a detailed capability map for a worker.
        
        Args:
            worker_id: ID of the worker
            
        Returns:
            Nested dictionary: hardware_id -> capability_id -> properties
        """
        if worker_id not in self.worker_hardware_map:
            return {}
        
        result = {}
        worker_profiles = self.worker_hardware_map[worker_id]
        
        for profile in worker_profiles:
            profile_id = f"{profile.hardware_class.value}_{profile.model_name}"
            capabilities = self.get_inherited_capabilities(profile)
            
            # Add to result
            result[profile_id] = capabilities
        
        return result
    
    def get_relationships_for_hardware(self, hardware_id: str) -> List[HardwareRelationship]:
        """
        Get all relationships for a hardware profile.
        
        Args:
            hardware_id: ID of the hardware profile
            
        Returns:
            List of hardware relationships
        """
        return self.hardware_relationships.get(hardware_id, [])
    
    def find_related_hardware(self, 
                           hardware_id: str, 
                           relationship_type: Optional[str] = None,
                           min_strength: float = 0.0) -> List[Tuple[str, str, float]]:
        """
        Find hardware related to a given hardware profile.
        
        Args:
            hardware_id: ID of the hardware profile
            relationship_type: Optional type of relationship to filter by
            min_strength: Minimum relationship strength
            
        Returns:
            List of (target_id, relationship_type, strength) tuples
        """
        results = []
        
        for rel in self.get_relationships_for_hardware(hardware_id):
            if (relationship_type is None or rel.relationship_type == relationship_type) and rel.strength >= min_strength:
                results.append((rel.target_id, rel.relationship_type, rel.strength))
        
        # Sort by strength (descending)
        results.sort(key=lambda item: item[2], reverse=True)
        return results