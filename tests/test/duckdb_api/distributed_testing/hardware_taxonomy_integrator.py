"""
Hardware Taxonomy Integrator for connecting the Enhanced Hardware Taxonomy with the Heterogeneous Scheduler.

This module provides the integration layer between the Enhanced Hardware Taxonomy and
the existing Heterogeneous Scheduler, enabling more sophisticated hardware selection
based on detailed capability matching.
"""

import logging
from typing import Dict, Set, Any, List, Optional, Tuple
from dataclasses import dataclass, field

from duckdb_api.distributed_testing.enhanced_hardware_taxonomy import (
    EnhancedHardwareTaxonomy,
    HardwareCapabilityProfile,
    CapabilityDefinition
)
from duckdb_api.distributed_testing.heterogeneous_scheduler import (
    WorkerState,
    TestTask,
    WorkloadProfile
)
from duckdb_api.distributed_testing.hardware_taxonomy import (
    HardwareClass,
    HardwareVendor,
    HardwareArchitecture
)

logger = logging.getLogger(__name__)

class HardwareTaxonomyIntegrator:
    """
    Integrates the Enhanced Hardware Taxonomy with the Heterogeneous Scheduler
    to provide more sophisticated hardware selection based on capabilities.
    """
    
    def __init__(self, taxonomy: Optional[EnhancedHardwareTaxonomy] = None):
        """
        Initialize with an optional taxonomy instance.
        
        Args:
            taxonomy: EnhancedHardwareTaxonomy instance to use, or None to create a new one
        """
        self.taxonomy = taxonomy or EnhancedHardwareTaxonomy()
        self.workload_capability_requirements = {}
        self._setup_common_workload_requirements()
        
    def _setup_common_workload_requirements(self):
        """Set up common workload requirements for standard workload types."""
        # NLP workloads typically need matrix multiplication and prefer tensor cores
        self.register_workload_requirements(
            "nlp", 
            required_capabilities={"matrix_multiplication"}, 
            preferred_capabilities={"tensor_core_acceleration", "int8_acceleration"}
        )
        
        # Vision workloads typically need convolutions and matrix multiplication
        self.register_workload_requirements(
            "vision", 
            required_capabilities={"matrix_multiplication"}, 
            preferred_capabilities={"conv_acceleration", "int8_vision_optimization"}
        )
        
        # Audio workloads benefit from FFT acceleration
        self.register_workload_requirements(
            "audio", 
            required_capabilities={"matrix_multiplication"}, 
            preferred_capabilities={"fft_acceleration", "audio_dsp_support"}
        )
        
        # Browser-based workloads 
        self.register_workload_requirements(
            "browser", 
            required_capabilities={"browser_compatibility"}, 
            preferred_capabilities={"webgpu_support", "webnn_support"}
        )
        
        # Multi-modal workloads need diverse computation support
        self.register_workload_requirements(
            "multimodal", 
            required_capabilities={"matrix_multiplication"}, 
            preferred_capabilities={
                "conv_acceleration", 
                "tensor_core_acceleration", 
                "parallel_execution"
            }
        )
        
    def register_workload_requirements(
        self, 
        workload_type: str, 
        required_capabilities: Set[str],
        preferred_capabilities: Optional[Set[str]] = None
    ):
        """
        Register capability requirements for a workload type.
        
        Args:
            workload_type: Type of workload (e.g., "nlp", "vision")
            required_capabilities: Set of capability IDs required for this workload
            preferred_capabilities: Set of capability IDs preferred but not required
        """
        self.workload_capability_requirements[workload_type] = {
            "required": required_capabilities,
            "preferred": preferred_capabilities or set()
        }
        logger.debug(
            f"Registered requirements for workload type '{workload_type}': "
            f"required={required_capabilities}, preferred={preferred_capabilities or set()}"
        )
    
    def enhance_worker_state(self, worker_state: WorkerState) -> WorkerState:
        """
        Enhance a worker state with taxonomy-based capability information.
        
        Args:
            worker_state: Worker state to enhance
            
        Returns:
            Enhanced worker state with capability profiles and updated specializations
        """
        # Convert worker hardware profiles to HardwareCapabilityProfile objects
        capability_profiles = []
        
        for profile_dict in worker_state.hardware_profiles:
            # Extract core hardware information
            hw_class = HardwareClass(profile_dict.get("hardware_class", "cpu"))
            architecture = profile_dict.get("architecture", "unknown")
            vendor = profile_dict.get("vendor", "unknown")
            model_name = profile_dict.get("model_name", "unknown")
            
            # Create a capability profile
            profile = HardwareCapabilityProfile(
                hardware_class=hw_class,
                architecture=architecture,
                vendor=vendor,
                model_name=model_name,
                memory_gb=profile_dict.get("memory_gb", 0.0),
                compute_units=profile_dict.get("compute_units", 0),
                features=profile_dict.get("features", []),
                supported_backends=profile_dict.get("supported_backends", [])
            )
            
            # Auto-discover capabilities based on hardware profile information
            self.taxonomy.auto_assign_capabilities(profile)
            capability_profiles.append(profile)
            
            logger.debug(
                f"Created capability profile for {hw_class}/{vendor}/{model_name} "
                f"with {len(profile.capabilities)} capabilities"
            )
        
        # Store reference to capability profiles for later use
        worker_state.capability_profiles = capability_profiles
        
        # Enhance workload specializations based on capability match
        for workload_type, requirements in self.workload_capability_requirements.items():
            # Find best matching profile for this workload
            best_match_score = 0.0
            best_profile = None
            
            for profile in capability_profiles:
                match_score = self.taxonomy.calculate_capability_match_score(
                    profile.capabilities,
                    requirements["required"],
                    requirements["preferred"]
                )
                
                if match_score > best_match_score:
                    best_match_score = match_score
                    best_profile = profile
            
            # Update worker's workload specialization score if we found a match
            if best_profile and best_match_score > 0:
                # Scale by existing specialization if present
                existing_score = worker_state.workload_specializations.get(workload_type, 0.5)
                # Blend the scores giving more weight to capability match
                new_score = 0.4 * existing_score + 0.6 * best_match_score
                worker_state.workload_specializations[workload_type] = new_score
                
                logger.debug(
                    f"Updated specialization for workload '{workload_type}': "
                    f"{existing_score:.2f} -> {new_score:.2f} (match score: {best_match_score:.2f})"
                )
        
        return worker_state
    
    def calculate_enhanced_affinity(self, worker_state: WorkerState, task: TestTask) -> float:
        """
        Calculate an enhanced affinity score using capability matching.
        
        Args:
            worker_state: Worker state with capability profiles
            task: Task to calculate affinity for
            
        Returns:
            Enhanced affinity score (0.0 to 1.0)
        """
        # Get workload type from task profile
        workload_type = task.workload_profile.workload_type
        
        # Get base score from existing affinity calculation
        base_score = worker_state.calculate_affinity_score(task)
        
        # Check if we have capability requirements for this workload
        if (workload_type not in self.workload_capability_requirements or 
            not hasattr(worker_state, 'capability_profiles')):
            # If no capability information, return the base score
            return base_score
        
        # Get required and preferred capabilities for this workload
        requirements = self.workload_capability_requirements[workload_type]
        required_caps = requirements["required"]
        preferred_caps = requirements["preferred"]
        
        # Also include task-specific capability requirements if available
        task_required_caps = getattr(task.workload_profile, "required_capabilities", set())
        task_preferred_caps = getattr(task.workload_profile, "preferred_capabilities", set())
        
        # Combine workload type requirements with task-specific requirements
        all_required = required_caps.union(task_required_caps)
        all_preferred = preferred_caps.union(task_preferred_caps)
        
        # Find best matching profile across all worker hardware
        best_match_score = 0.0
        for profile in worker_state.capability_profiles:
            match_score = self.taxonomy.calculate_capability_match_score(
                profile.capabilities, all_required, all_preferred
            )
            best_match_score = max(best_match_score, match_score)
        
        # Blend the scores, giving more weight to capability match for precision
        # but still considering the base score which accounts for current load and thermal state
        enhanced_score = 0.4 * base_score + 0.6 * best_match_score
        
        logger.debug(
            f"Affinity calculation for task on worker {worker_state.worker_id}: "
            f"base={base_score:.2f}, capability={best_match_score:.2f}, "
            f"enhanced={enhanced_score:.2f}"
        )
        
        return enhanced_score
    
    def get_capability_breakdown(self, worker_state: WorkerState) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get a breakdown of capabilities by workload type for a worker.
        
        Args:
            worker_state: Worker state with capability profiles
            
        Returns:
            Dictionary mapping workload types to lists of (capability, impact) tuples
        """
        if not hasattr(worker_state, 'capability_profiles'):
            return {}
        
        result = {}
        for workload_type, requirements in self.workload_capability_requirements.items():
            capability_impacts = []
            
            # For each profile, analyze which capabilities have the most impact
            for profile in worker_state.capability_profiles:
                # Get all capabilities for this profile
                all_caps = profile.capabilities
                
                # For each capability, calculate its impact on the match score
                for cap_id in all_caps:
                    # Check if this capability is required or preferred
                    if cap_id in requirements["required"]:
                        impact = 1.0  # Required capabilities have maximum impact
                    elif cap_id in requirements["preferred"]:
                        impact = 0.5  # Preferred capabilities have medium impact
                    else:
                        impact = 0.0  # Capabilities not in requirements have no impact
                    
                    if impact > 0:
                        # Get the capability definition for additional info
                        cap_def = self.taxonomy.get_capability_definition(cap_id)
                        capability_impacts.append((cap_id, impact))
            
            if capability_impacts:
                # Sort by impact (highest first)
                capability_impacts.sort(key=lambda x: x[1], reverse=True)
                result[workload_type] = capability_impacts
        
        return result
    
    def enhance_workload_profile(self, profile: WorkloadProfile) -> WorkloadProfile:
        """
        Enhance a workload profile with capability requirements based on its type.
        
        Args:
            profile: WorkloadProfile to enhance
            
        Returns:
            Enhanced WorkloadProfile with capability requirements
        """
        workload_type = profile.workload_type
        
        # If this workload type has registered capability requirements, add them
        if workload_type in self.workload_capability_requirements:
            requirements = self.workload_capability_requirements[workload_type]
            
            # Add required capabilities
            for cap_id in requirements["required"]:
                # Add to profile's required capabilities
                if hasattr(profile, "required_capabilities"):
                    profile.required_capabilities.add(cap_id)
                else:
                    # If the profile doesn't have the attribute, add it via __dict__
                    if "required_capabilities" not in profile.__dict__:
                        profile.__dict__["required_capabilities"] = set()
                    profile.__dict__["required_capabilities"].add(cap_id)
            
            # Add preferred capabilities
            for cap_id in requirements["preferred"]:
                # Add to profile's preferred capabilities
                if hasattr(profile, "preferred_capabilities"):
                    profile.preferred_capabilities.add(cap_id)
                else:
                    # If the profile doesn't have the attribute, add it via __dict__
                    if "preferred_capabilities" not in profile.__dict__:
                        profile.__dict__["preferred_capabilities"] = set()
                    profile.__dict__["preferred_capabilities"].add(cap_id)
        
        return profile