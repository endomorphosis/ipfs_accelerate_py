"""
Hardware Taxonomy for Distributed Testing Framework

This module provides a comprehensive classification system for heterogeneous
hardware environments, enabling more sophisticated hardware detection,
matching, and optimization strategies.

The taxonomy defines hardware types, capabilities, and relationships, allowing
the system to reason about hardware compatibility, specialization, and
performance characteristics.
"""

import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union


class HardwareClass(enum.Enum):
    """High-level classification of hardware devices."""
    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"
    NPU = "npu"
    DSP = "dsp"
    FPGA = "fpga"
    ASIC = "asic"
    HYBRID = "hybrid"
    UNKNOWN = "unknown"


class HardwareArchitecture(enum.Enum):
    """Processor architecture classification."""
    X86 = "x86"
    X86_64 = "x86_64"
    ARM = "arm"
    ARM64 = "arm64"
    PPC = "ppc"
    PPC64 = "ppc64"
    MIPS = "mips"
    RISCV = "riscv"
    GPU_CUDA = "gpu_cuda"
    GPU_ROCM = "gpu_rocm"
    GPU_METAL = "gpu_metal"
    GPU_WEBGPU = "gpu_webgpu"
    TPU = "tpu"
    NPU_QUALCOMM = "npu_qualcomm"
    NPU_MEDIATEK = "npu_mediatek"
    NPU_SAMSUNG = "npu_samsung"
    DSP_HEXAGON = "dsp_hexagon"
    FPGA_INTEL = "fpga_intel"
    FPGA_XILINX = "fpga_xilinx"
    OTHER = "other"


class HardwareVendor(enum.Enum):
    """Hardware vendor classification."""
    INTEL = "intel"
    AMD = "amd"
    NVIDIA = "nvidia"
    ARM = "arm"
    APPLE = "apple"
    QUALCOMM = "qualcomm"
    MEDIATEK = "mediatek"
    SAMSUNG = "samsung"
    GOOGLE = "google"
    IBM = "ibm"
    MICROSOFT = "microsoft"
    AMAZON = "amazon"
    XILINX = "xilinx"
    ALTERA = "altera"
    HUAWEI = "huawei"
    OTHER = "other"


class SoftwareBackend(enum.Enum):
    """Software frameworks and backends."""
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    ONNX = "onnx"
    WEBNN = "webnn"
    WEBGPU = "webgpu"
    DIRECTML = "directml"
    OPENVINO = "openvino"
    QNN = "qnn"
    CUDA = "cuda"
    ROCM = "rocm"
    MPS = "mps"
    VULKAN = "vulkan"
    METAL = "metal"
    OPENCL = "opencl"
    TVM = "tvm"
    NNAPI = "nnapi"
    COREML = "coreml"
    TFLITE = "tflite"
    TENSORRT = "tensorrt"
    OTHER = "other"


class PrecisionType(enum.Enum):
    """Numeric precision formats."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"
    INT2 = "int2"
    INT1 = "int1"
    MIXED = "mixed"
    OTHER = "other"


class AcceleratorFeature(enum.Enum):
    """Special hardware acceleration features."""
    TENSOR_CORES = "tensor_cores"
    RAY_TRACING = "ray_tracing"
    NEURAL_ENGINE = "neural_engine"
    QUANTIZATION = "quantization"
    SPARSITY = "sparsity"
    DYNAMIC_PRECISION = "dynamic_precision"
    MULTI_INSTANCE = "multi_instance"
    SHARED_MEMORY = "shared_memory"
    UNIFIED_MEMORY = "unified_memory"
    DEDICATED_MEMORY = "dedicated_memory"
    COMPUTE_SHADERS = "compute_shaders"
    SIMD = "simd"
    SIMT = "simt"
    AVX = "avx"
    AVX2 = "avx2"
    AVX512 = "avx512"
    NEON = "neon"
    SVE = "sve"
    SMT = "smt"
    TSX = "tsx"
    OTHER = "other"


@dataclass
class MemoryProfile:
    """Memory characteristics and capabilities."""
    total_bytes: int = 0
    available_bytes: int = 0
    bandwidth_gbps: Optional[float] = None  # Memory bandwidth in GB/s
    is_shared: bool = False  # Whether memory is shared with other hardware
    hierarchy_levels: int = 1  # Number of cache/memory hierarchy levels
    has_unified_memory: bool = False  # Whether the hardware has unified memory architecture
    memory_type: str = "unknown"  # Type of memory (DDR4, HBM, GDDR6, etc.)


@dataclass
class HardwareCapabilityProfile:
    """Detailed hardware capability information."""
    hardware_class: HardwareClass = HardwareClass.UNKNOWN
    architecture: HardwareArchitecture = HardwareArchitecture.OTHER
    vendor: HardwareVendor = HardwareVendor.OTHER
    model_name: str = "unknown"
    supported_backends: Set[SoftwareBackend] = field(default_factory=set)
    supported_precisions: Set[PrecisionType] = field(default_factory=set)
    features: Set[AcceleratorFeature] = field(default_factory=set)
    memory: MemoryProfile = field(default_factory=MemoryProfile)
    compute_units: int = 0  # Cores, SMs, etc.
    clock_speed_mhz: Optional[int] = None
    thermal_design_power_w: Optional[float] = None  # TDP in watts
    compute_capability: Optional[str] = None  # For NVIDIA GPUs
    firmware_version: Optional[str] = None
    driver_version: Optional[str] = None
    is_virtualized: bool = False
    is_simulated: bool = False  # Whether this is a simulated device

    # Performance characteristics for different operations
    performance_profile: Dict[str, float] = field(default_factory=dict)
    # Format: {operation_type: operations_per_second}
    # Example: {"fp32_matmul": 1200, "int8_inference": 4500}
    
    # Efficiency metrics
    power_efficiency: Optional[float] = None  # Operations per watt
    memory_efficiency: Optional[float] = None  # Operations per memory bandwidth
    thermal_efficiency: Optional[float] = None  # Operations per degree Celsius increase

    def is_compatible_with(self, 
                          required_backends: Set[SoftwareBackend] = None,
                          required_precisions: Set[PrecisionType] = None,
                          required_features: Set[AcceleratorFeature] = None,
                          min_memory_bytes: int = 0,
                          min_compute_units: int = 0) -> bool:
        """
        Check if this hardware is compatible with the specified requirements.
        
        Args:
            required_backends: Set of required software backends
            required_precisions: Set of required precision types
            required_features: Set of required hardware features
            min_memory_bytes: Minimum required memory in bytes
            min_compute_units: Minimum required compute units
            
        Returns:
            bool: True if compatible, False otherwise
        """
        if required_backends and not required_backends.issubset(self.supported_backends):
            return False
            
        if required_precisions and not required_precisions.issubset(self.supported_precisions):
            return False
            
        if required_features and not required_features.issubset(self.features):
            return False
            
        if min_memory_bytes > self.memory.total_bytes:
            return False
            
        if min_compute_units > self.compute_units:
            return False
            
        return True
    
    def estimate_performance(self, operation_type: str, 
                            size: int, 
                            precision: PrecisionType) -> Optional[float]:
        """
        Estimate performance for a specific operation.
        
        Args:
            operation_type: Type of operation (matmul, conv, etc.)
            size: Size of the operation (elements, parameters, etc.)
            precision: Precision type for the operation
            
        Returns:
            Optional[float]: Estimated operations per second, or None if unknown
        """
        key = f"{precision.value}_{operation_type}"
        if key in self.performance_profile:
            base_performance = self.performance_profile[key]
            # Apply scaling factor based on size (simplified)
            return base_performance * (1.0 / (1.0 + size/1000000.0)) 
        return None


@dataclass
class HardwareSpecialization:
    """
    Hardware specialization for specific workloads.
    """
    workload_type: str  # e.g., "vision", "nlp", "audio", "reinforcement_learning"
    effectiveness_score: float  # 0.0 to 1.0, higher is better
    hardware_profile: HardwareCapabilityProfile
    optimal_batch_size: Optional[int] = None
    optimal_precision: Optional[PrecisionType] = None
    notes: Optional[str] = None
    
    @classmethod
    def create_specialization_map(cls, 
                                 hardware_profiles: List[HardwareCapabilityProfile]) -> Dict[str, List["HardwareSpecialization"]]:
        """
        Create a map of workload types to specialized hardware.
        
        Args:
            hardware_profiles: List of hardware capability profiles
            
        Returns:
            Dict mapping workload types to list of HardwareSpecialization objects
        """
        specialization_map = {}
        
        # These rules could be learned from performance data
        # Here we use a rule-based approach for demonstration
        for profile in hardware_profiles:
            specializations = []
            
            # NLP specialization rules
            if HardwareClass.GPU == profile.hardware_class and profile.compute_units >= 30:
                effectiveness = min(1.0, profile.compute_units / 100.0)
                specializations.append(
                    cls("nlp", effectiveness, profile, 
                        optimal_batch_size=16, 
                        optimal_precision=PrecisionType.FP16 
                            if PrecisionType.FP16 in profile.supported_precisions 
                            else PrecisionType.FP32)
                )
            
            # Vision specialization rules
            if profile.hardware_class in [HardwareClass.GPU, HardwareClass.TPU, HardwareClass.NPU]:
                if AcceleratorFeature.TENSOR_CORES in profile.features:
                    effectiveness = 0.9
                else:
                    effectiveness = 0.7
                specializations.append(
                    cls("vision", effectiveness, profile, 
                        optimal_batch_size=32, 
                        optimal_precision=PrecisionType.FP16 
                            if PrecisionType.FP16 in profile.supported_precisions 
                            else PrecisionType.FP32)
                )
            
            # Audio specialization rules
            if profile.hardware_class == HardwareClass.CPU and AcceleratorFeature.AVX2 in profile.features:
                effectiveness = 0.85
                specializations.append(
                    cls("audio", effectiveness, profile, 
                        optimal_batch_size=8, 
                        optimal_precision=PrecisionType.FP32)
                )
            
            # Add specializations to map
            for spec in specializations:
                if spec.workload_type not in specialization_map:
                    specialization_map[spec.workload_type] = []
                specialization_map[spec.workload_type].append(spec)
        
        # Sort each list by effectiveness_score (descending)
        for workload_type in specialization_map:
            specialization_map[workload_type].sort(
                key=lambda spec: spec.effectiveness_score, reverse=True)
                
        return specialization_map


class HardwareTaxonomy:
    """
    Main class for hardware taxonomy and classification.
    """
    def __init__(self):
        # Maps hardware class to capability profiles
        self.hardware_profiles: Dict[str, List[HardwareCapabilityProfile]] = {}
        
        # Maps workload types to specialized hardware
        self.specialization_map: Dict[str, List[HardwareSpecialization]] = {}
        
        # Maps worker IDs to hardware profiles
        self.worker_hardware_map: Dict[str, List[HardwareCapabilityProfile]] = {}
        
        # Performance index for quick lookup
        self.performance_index: Dict[Tuple[str, PrecisionType], List[Tuple[float, HardwareCapabilityProfile]]] = {}
        
        # Hardware compatibility matrix
        self.compatibility_matrix: Dict[Tuple[HardwareClass, HardwareClass], float] = {}
        
        self._initialize_compatibility_matrix()
    
    def _initialize_compatibility_matrix(self):
        """Initialize the hardware compatibility matrix with default values."""
        classes = list(HardwareClass)
        for c1 in classes:
            for c2 in classes:
                # Default: perfect compatibility with self, otherwise 0.0
                self.compatibility_matrix[(c1, c2)] = 1.0 if c1 == c2 else 0.0
        
        # Define compatibility values for common combinations
        # Higher values indicate better compatibility (0.0 to 1.0)
        self.compatibility_matrix[(HardwareClass.CPU, HardwareClass.CPU)] = 1.0
        self.compatibility_matrix[(HardwareClass.GPU, HardwareClass.GPU)] = 0.8  # Different GPU architectures
        self.compatibility_matrix[(HardwareClass.CPU, HardwareClass.GPU)] = 0.6
        self.compatibility_matrix[(HardwareClass.GPU, HardwareClass.CPU)] = 0.6
        self.compatibility_matrix[(HardwareClass.TPU, HardwareClass.CPU)] = 0.3
        self.compatibility_matrix[(HardwareClass.CPU, HardwareClass.TPU)] = 0.3
        self.compatibility_matrix[(HardwareClass.NPU, HardwareClass.CPU)] = 0.4
        self.compatibility_matrix[(HardwareClass.CPU, HardwareClass.NPU)] = 0.4
        
    def register_hardware_profile(self, profile: HardwareCapabilityProfile):
        """
        Register a hardware capability profile in the taxonomy.
        
        Args:
            profile: The hardware capability profile to register
        """
        hardware_class = profile.hardware_class.value
        if hardware_class not in self.hardware_profiles:
            self.hardware_profiles[hardware_class] = []
        self.hardware_profiles[hardware_class].append(profile)
        
        # Update performance index
        for operation_type, performance in profile.performance_profile.items():
            # Extract precision from operation type (e.g., "fp32_matmul" -> PrecisionType.FP32)
            precision_str, _ = operation_type.split("_", 1)
            precision = next((p for p in PrecisionType if p.value == precision_str), PrecisionType.OTHER)
            
            key = (operation_type, precision)
            if key not in self.performance_index:
                self.performance_index[key] = []
            self.performance_index[key].append((performance, profile))
            
            # Sort by performance (descending)
            self.performance_index[key].sort(key=lambda item: item[0], reverse=True)
    
    def register_worker_hardware(self, worker_id: str, profiles: List[HardwareCapabilityProfile]):
        """
        Register hardware profiles for a worker.
        
        Args:
            worker_id: ID of the worker
            profiles: List of hardware profiles for the worker
        """
        self.worker_hardware_map[worker_id] = profiles
        
        # Register all profiles in the taxonomy
        for profile in profiles:
            self.register_hardware_profile(profile)
    
    def update_specialization_map(self):
        """Update the specialization map based on registered hardware profiles."""
        # Flatten all profiles into a single list
        all_profiles = []
        for profiles in self.hardware_profiles.values():
            all_profiles.extend(profiles)
            
        self.specialization_map = HardwareSpecialization.create_specialization_map(all_profiles)
    
    def find_best_hardware_for_workload(self, 
                                       workload_type: str, 
                                       worker_ids: List[str] = None,
                                       min_effectiveness: float = 0.0) -> List[Tuple[str, HardwareCapabilityProfile, float]]:
        """
        Find the best hardware for a specific workload type.
        
        Args:
            workload_type: Type of workload (e.g., "nlp", "vision")
            worker_ids: Optional list of worker IDs to consider, or None for all workers
            min_effectiveness: Minimum effectiveness score (0.0 to 1.0)
            
        Returns:
            List of (worker_id, hardware_profile, effectiveness_score) tuples,
            sorted by effectiveness score (descending)
        """
        if workload_type not in self.specialization_map:
            return []
            
        result = []
        valid_worker_ids = worker_ids if worker_ids is not None else list(self.worker_hardware_map.keys())
        
        for spec in self.specialization_map[workload_type]:
            if spec.effectiveness_score < min_effectiveness:
                continue
                
            profile = spec.hardware_profile
            
            # Find workers that have this profile
            for worker_id in valid_worker_ids:
                if worker_id in self.worker_hardware_map:
                    worker_profiles = self.worker_hardware_map[worker_id]
                    for worker_profile in worker_profiles:
                        if (worker_profile.hardware_class == profile.hardware_class and 
                            worker_profile.architecture == profile.architecture and
                            worker_profile.vendor == profile.vendor and
                            worker_profile.model_name == profile.model_name):
                            result.append((worker_id, worker_profile, spec.effectiveness_score))
        
        # Sort by effectiveness score (descending)
        result.sort(key=lambda item: item[2], reverse=True)
        return result
    
    def calculate_compatibility_score(self, profile1: HardwareCapabilityProfile, 
                                     profile2: HardwareCapabilityProfile) -> float:
        """
        Calculate compatibility score between two hardware profiles.
        
        Args:
            profile1: First hardware profile
            profile2: Second hardware profile
            
        Returns:
            float: Compatibility score (0.0 to 1.0, higher is more compatible)
        """
        # Base compatibility from the matrix
        base_score = self.compatibility_matrix.get(
            (profile1.hardware_class, profile2.hardware_class), 0.0)
        
        # Adjust for shared backends
        backend_overlap = len(profile1.supported_backends.intersection(profile2.supported_backends))
        backend_score = backend_overlap / max(1, len(profile1.supported_backends.union(profile2.supported_backends)))
        
        # Adjust for shared precision types
        precision_overlap = len(profile1.supported_precisions.intersection(profile2.supported_precisions))
        precision_score = precision_overlap / max(1, len(profile1.supported_precisions.union(profile2.supported_precisions)))
        
        # Calculate final score
        final_score = 0.5 * base_score + 0.3 * backend_score + 0.2 * precision_score
        return min(1.0, max(0.0, final_score))
    
    def find_most_compatible_workers(self, source_worker_id: str, 
                                    candidate_worker_ids: List[str]) -> List[Tuple[str, float]]:
        """
        Find workers most compatible with a source worker.
        
        Useful for finding fallback workers or workers for distributed execution.
        
        Args:
            source_worker_id: ID of the source worker
            candidate_worker_ids: List of candidate worker IDs
            
        Returns:
            List of (worker_id, compatibility_score) tuples, sorted by score (descending)
        """
        if source_worker_id not in self.worker_hardware_map:
            return []
            
        source_profiles = self.worker_hardware_map[source_worker_id]
        result = []
        
        for candidate_id in candidate_worker_ids:
            if candidate_id == source_worker_id or candidate_id not in self.worker_hardware_map:
                continue
                
            candidate_profiles = self.worker_hardware_map[candidate_id]
            
            # Calculate average compatibility score between all profile pairs
            total_score = 0.0
            count = 0
            
            for source_profile in source_profiles:
                for candidate_profile in candidate_profiles:
                    total_score += self.calculate_compatibility_score(source_profile, candidate_profile)
                    count += 1
            
            avg_score = total_score / max(1, count)
            result.append((candidate_id, avg_score))
        
        # Sort by compatibility score (descending)
        result.sort(key=lambda item: item[1], reverse=True)
        return result
    
    def get_performance_ranking(self, operation_type: str, 
                               precision: PrecisionType) -> List[Tuple[HardwareCapabilityProfile, float]]:
        """
        Get hardware profiles ranked by performance for a specific operation.
        
        Args:
            operation_type: Type of operation (e.g., "matmul", "conv")
            precision: Precision type
            
        Returns:
            List of (profile, performance) tuples, sorted by performance (descending)
        """
        key = (f"{precision.value}_{operation_type}", precision)
        if key not in self.performance_index:
            return []
            
        return [(profile, perf) for perf, profile in self.performance_index[key]]


# Factory methods for easy creation of common hardware profiles
def create_cpu_profile(
    model_name: str,
    vendor: HardwareVendor,
    cores: int,
    memory_gb: float,
    clock_speed_mhz: int,
    has_avx: bool = False,
    has_avx2: bool = False,
    has_avx512: bool = False
) -> HardwareCapabilityProfile:
    """Create a CPU hardware profile."""
    memory = MemoryProfile(
        total_bytes=int(memory_gb * 1024 * 1024 * 1024),
        available_bytes=int(memory_gb * 0.8 * 1024 * 1024 * 1024),  # Assume 80% available
        memory_type="DDR4",
        is_shared=False,
        hierarchy_levels=3,
        has_unified_memory=True
    )
    
    features = set()
    if has_avx:
        features.add(AcceleratorFeature.AVX)
    if has_avx2:
        features.add(AcceleratorFeature.AVX2)
    if has_avx512:
        features.add(AcceleratorFeature.AVX512)
    features.add(AcceleratorFeature.SIMD)
    
    supported_backends = {
        SoftwareBackend.PYTORCH, 
        SoftwareBackend.TENSORFLOW,
        SoftwareBackend.ONNX,
        SoftwareBackend.OPENVINO
    }
    
    supported_precisions = {
        PrecisionType.FP32,
        PrecisionType.INT8
    }
    
    performance_profile = {
        "fp32_matmul": cores * clock_speed_mhz * 0.1,
        "fp32_conv": cores * clock_speed_mhz * 0.05,
        "int8_matmul": cores * clock_speed_mhz * 0.3,
        "int8_conv": cores * clock_speed_mhz * 0.15
    }
    
    # Determine architecture based on vendor
    if vendor == HardwareVendor.INTEL or vendor == HardwareVendor.AMD:
        architecture = HardwareArchitecture.X86_64
    elif vendor == HardwareVendor.ARM or vendor == HardwareVendor.APPLE:
        architecture = HardwareArchitecture.ARM64
    else:
        architecture = HardwareArchitecture.OTHER
    
    return HardwareCapabilityProfile(
        hardware_class=HardwareClass.CPU,
        architecture=architecture,
        vendor=vendor,
        model_name=model_name,
        supported_backends=supported_backends,
        supported_precisions=supported_precisions,
        features=features,
        memory=memory,
        compute_units=cores,
        clock_speed_mhz=clock_speed_mhz,
        thermal_design_power_w=65.0,  # Typical CPU TDP
        performance_profile=performance_profile,
        power_efficiency=cores * clock_speed_mhz / 65.0,
        memory_efficiency=0.7
    )


def create_gpu_profile(
    model_name: str,
    vendor: HardwareVendor,
    compute_units: int,
    memory_gb: float,
    clock_speed_mhz: int,
    has_tensor_cores: bool = False,
    has_ray_tracing: bool = False,
    compute_capability: Optional[str] = None,
    memory_bandwidth_gbps: Optional[float] = None,
    tdp_w: float = 200.0
) -> HardwareCapabilityProfile:
    """Create a GPU hardware profile."""
    memory = MemoryProfile(
        total_bytes=int(memory_gb * 1024 * 1024 * 1024),
        available_bytes=int(memory_gb * 0.9 * 1024 * 1024 * 1024),  # Assume 90% available
        bandwidth_gbps=memory_bandwidth_gbps,
        memory_type="GDDR6" if vendor == HardwareVendor.NVIDIA else "HBM2",
        is_shared=False,
        hierarchy_levels=2,
        has_unified_memory=False
    )
    
    features = {AcceleratorFeature.SIMT, AcceleratorFeature.COMPUTE_SHADERS}
    if has_tensor_cores:
        features.add(AcceleratorFeature.TENSOR_CORES)
    if has_ray_tracing:
        features.add(AcceleratorFeature.RAY_TRACING)
    
    supported_backends = {SoftwareBackend.PYTORCH, SoftwareBackend.TENSORFLOW, SoftwareBackend.ONNX}
    if vendor == HardwareVendor.NVIDIA:
        supported_backends.add(SoftwareBackend.CUDA)
        supported_backends.add(SoftwareBackend.TENSORRT)
    elif vendor == HardwareVendor.AMD:
        supported_backends.add(SoftwareBackend.ROCM)
    elif vendor == HardwareVendor.APPLE:
        supported_backends.add(SoftwareBackend.METAL)
        supported_backends.add(SoftwareBackend.MPS)
    
    supported_precisions = {PrecisionType.FP32, PrecisionType.FP16}
    if has_tensor_cores:
        supported_precisions.add(PrecisionType.INT8)
        supported_precisions.add(PrecisionType.INT4)
    
    # Tensor core performance factor
    tc_factor = 4.0 if has_tensor_cores else 1.0
    
    performance_profile = {
        "fp32_matmul": compute_units * clock_speed_mhz * 0.5,
        "fp32_conv": compute_units * clock_speed_mhz * 0.3,
        "fp16_matmul": compute_units * clock_speed_mhz * 1.0,
        "fp16_conv": compute_units * clock_speed_mhz * 0.6,
        "int8_matmul": compute_units * clock_speed_mhz * 1.8 * tc_factor,
        "int8_conv": compute_units * clock_speed_mhz * 1.2 * tc_factor
    }
    
    # Determine architecture based on vendor
    if vendor == HardwareVendor.NVIDIA:
        architecture = HardwareArchitecture.GPU_CUDA
    elif vendor == HardwareVendor.AMD:
        architecture = HardwareArchitecture.GPU_ROCM
    elif vendor == HardwareVendor.APPLE:
        architecture = HardwareArchitecture.GPU_METAL
    else:
        architecture = HardwareArchitecture.OTHER
    
    return HardwareCapabilityProfile(
        hardware_class=HardwareClass.GPU,
        architecture=architecture,
        vendor=vendor,
        model_name=model_name,
        supported_backends=supported_backends,
        supported_precisions=supported_precisions,
        features=features,
        memory=memory,
        compute_units=compute_units,
        clock_speed_mhz=clock_speed_mhz,
        thermal_design_power_w=tdp_w,
        compute_capability=compute_capability,
        performance_profile=performance_profile,
        power_efficiency=compute_units * clock_speed_mhz / tdp_w,
        memory_efficiency=0.8 if memory_bandwidth_gbps else 0.6
    )


def create_npu_profile(
    model_name: str,
    vendor: HardwareVendor,
    compute_units: int,
    memory_gb: float,
    clock_speed_mhz: int,
    has_quantization: bool = True,
    tdp_w: float = 10.0
) -> HardwareCapabilityProfile:
    """Create an NPU hardware profile."""
    memory = MemoryProfile(
        total_bytes=int(memory_gb * 1024 * 1024 * 1024),
        available_bytes=int(memory_gb * 0.9 * 1024 * 1024 * 1024),
        memory_type="LPDDR5",
        is_shared=True,
        hierarchy_levels=2,
        has_unified_memory=True
    )
    
    features = {AcceleratorFeature.NEURAL_ENGINE}
    if has_quantization:
        features.add(AcceleratorFeature.QUANTIZATION)
    
    # Different vendors support different backends
    supported_backends = {SoftwareBackend.ONNX}
    if vendor == HardwareVendor.QUALCOMM:
        supported_backends.add(SoftwareBackend.QNN)
        architecture = HardwareArchitecture.NPU_QUALCOMM
    elif vendor == HardwareVendor.MEDIATEK:
        supported_backends.add(SoftwareBackend.NNAPI)
        architecture = HardwareArchitecture.NPU_MEDIATEK
    elif vendor == HardwareVendor.SAMSUNG:
        supported_backends.add(SoftwareBackend.NNAPI)
        architecture = HardwareArchitecture.NPU_SAMSUNG
    else:
        architecture = HardwareArchitecture.OTHER
    
    supported_precisions = {PrecisionType.FP32, PrecisionType.FP16, PrecisionType.INT8}
    if has_quantization:
        supported_precisions.add(PrecisionType.INT4)
    
    # NPUs are efficient at int8 but may have lower clock speeds
    performance_profile = {
        "fp32_matmul": compute_units * clock_speed_mhz * 0.3,
        "fp32_conv": compute_units * clock_speed_mhz * 0.2,
        "fp16_matmul": compute_units * clock_speed_mhz * 0.6,
        "fp16_conv": compute_units * clock_speed_mhz * 0.4,
        "int8_matmul": compute_units * clock_speed_mhz * 1.2,
        "int8_conv": compute_units * clock_speed_mhz * 0.8,
        "int4_matmul": compute_units * clock_speed_mhz * 2.0 if has_quantization else 0.0,
        "int4_conv": compute_units * clock_speed_mhz * 1.5 if has_quantization else 0.0
    }
    
    return HardwareCapabilityProfile(
        hardware_class=HardwareClass.NPU,
        architecture=architecture,
        vendor=vendor,
        model_name=model_name,
        supported_backends=supported_backends,
        supported_precisions=supported_precisions,
        features=features,
        memory=memory,
        compute_units=compute_units,
        clock_speed_mhz=clock_speed_mhz,
        thermal_design_power_w=tdp_w,
        performance_profile=performance_profile,
        power_efficiency=compute_units * clock_speed_mhz / tdp_w * 2.0,  # NPUs are power efficient
        memory_efficiency=0.9  # NPUs often have good memory efficiency
    )


def create_browser_profile(
    browser_name: str,
    supports_webgpu: bool = False,
    supports_webnn: bool = False,
    gpu_profile: Optional[HardwareCapabilityProfile] = None
) -> HardwareCapabilityProfile:
    """Create a browser hardware profile."""
    # Map browser name to vendor
    vendor_map = {
        "chrome": HardwareVendor.GOOGLE,
        "edge": HardwareVendor.MICROSOFT,
        "firefox": HardwareVendor.OTHER,
        "safari": HardwareVendor.APPLE
    }
    vendor = vendor_map.get(browser_name.lower(), HardwareVendor.OTHER)
    
    # Start with base features and supported backends
    features = {AcceleratorFeature.COMPUTE_SHADERS}
    supported_backends = set()
    supported_precisions = {PrecisionType.FP32}
    
    if supports_webgpu:
        supported_backends.add(SoftwareBackend.WEBGPU)
        features.add(AcceleratorFeature.COMPUTE_SHADERS)
        supported_precisions.add(PrecisionType.FP16)
        
    if supports_webnn:
        supported_backends.add(SoftwareBackend.WEBNN)
        
    # If we have a GPU profile, use its memory and compute characteristics
    memory = MemoryProfile(
        total_bytes=1024 * 1024 * 1024,  # Default 1GB
        available_bytes=512 * 1024 * 1024,  # Default 512MB
        memory_type="Shared",
        is_shared=True,
        hierarchy_levels=1,
        has_unified_memory=True
    )
    
    compute_units = 2  # Default low value
    clock_speed_mhz = 1000  # Default value
    tdp_w = 5.0  # Default low value
    
    if gpu_profile:
        memory = MemoryProfile(
            total_bytes=gpu_profile.memory.total_bytes // 4,  # Conservative estimate
            available_bytes=gpu_profile.memory.available_bytes // 4,
            memory_type=gpu_profile.memory.memory_type,
            is_shared=True,
            hierarchy_levels=gpu_profile.memory.hierarchy_levels,
            has_unified_memory=gpu_profile.memory.has_unified_memory
        )
        compute_units = gpu_profile.compute_units // 2
        clock_speed_mhz = gpu_profile.clock_speed_mhz
        tdp_w = gpu_profile.thermal_design_power_w
    
    # Performance profiles - based on browser benchmarks
    # Chrome is strong in WebGPU, Edge in WebNN, Firefox in audio compute shaders
    performance_factor = {
        "chrome": {"webgpu": 1.0, "webnn": 0.7, "audio": 0.7},
        "edge": {"webgpu": 0.8, "webnn": 1.0, "audio": 0.6},
        "firefox": {"webgpu": 0.8, "webnn": 0.6, "audio": 1.0},
        "safari": {"webgpu": 0.9, "webnn": 0.8, "audio": 0.7}
    }.get(browser_name.lower(), {"webgpu": 0.6, "webnn": 0.6, "audio": 0.6})
    
    base_performance = compute_units * clock_speed_mhz * 0.01
    
    performance_profile = {
        "fp32_matmul": base_performance * performance_factor["webgpu"] if supports_webgpu else 0.0,
        "fp32_conv": base_performance * 0.8 * performance_factor["webgpu"] if supports_webgpu else 0.0,
        "fp16_matmul": base_performance * 1.5 * performance_factor["webgpu"] if supports_webgpu else 0.0,
        "fp16_conv": base_performance * 1.2 * performance_factor["webgpu"] if supports_webgpu else 0.0,
        "int8_matmul": base_performance * 2.0 * performance_factor["webnn"] if supports_webnn else 0.0,
        "int8_conv": base_performance * 1.5 * performance_factor["webnn"] if supports_webnn else 0.0,
        "fp32_audio": base_performance * 1.2 * performance_factor["audio"] if supports_webgpu else 0.0
    }
    
    return HardwareCapabilityProfile(
        hardware_class=HardwareClass.HYBRID,
        architecture=HardwareArchitecture.GPU_WEBGPU if supports_webgpu else HardwareArchitecture.OTHER,
        vendor=vendor,
        model_name=f"{browser_name.capitalize()} Browser",
        supported_backends=supported_backends,
        supported_precisions=supported_precisions,
        features=features,
        memory=memory,
        compute_units=compute_units,
        clock_speed_mhz=clock_speed_mhz,
        thermal_design_power_w=tdp_w,
        performance_profile=performance_profile,
        power_efficiency=base_performance / tdp_w,
        memory_efficiency=0.5  # Browsers generally have lower memory efficiency
    )