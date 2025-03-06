#!/usr/bin/env python3
"""
Auto-detection utility for hardware and optimal precision configuration.
Detects available hardware, determines optimal precision for each platform,
and suggests configuration for optimal performance.
"""

import os
import sys
import json
import logging
import argparse
import torch
import platform
import subprocess
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("auto_detect")

# Global constants
PRECISION_TYPES = ["fp32", "fp16", "bf16", "int8", "int4", "uint4", "fp8", "fp4"]
HARDWARE_TYPES = ["cpu", "cuda", "rocm", "mps", "openvino", "qnn", "webnn", "webgpu"]

@dataclass
class HardwareInfo:
    """Stores detailed information about detected hardware"""
    type: str
    detected: bool = False
    count: int = 0
    names: List[str] = field(default_factory=list)
    memory_gb: List[float] = field(default_factory=list)
    compute_capability: Optional[Tuple[int, int]] = None
    driver_version: Optional[str] = None
    api_version: Optional[str] = None
    architecture: Optional[str] = None
    extra_info: Dict[str, Any] = field(default_factory=dict)
    
    def as_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "type": self.type,
            "detected": self.detected,
            "count": self.count,
            "names": self.names,
            "memory_gb": self.memory_gb,
            "compute_capability": self.compute_capability,
            "driver_version": self.driver_version,
            "api_version": self.api_version,
            "architecture": self.architecture,
            "extra_info": self.extra_info
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'HardwareInfo':
        """Create instance from dictionary"""
        return cls(**data)


@dataclass
class PrecisionInfo:
    """Stores information about precision support for hardware"""
    hardware_type: str
    supported: Dict[str, bool] = field(default_factory=dict)
    optimal: Optional[str] = None
    performance_ranking: List[str] = field(default_factory=list)
    memory_ranking: List[str] = field(default_factory=list)
    extra_info: Dict[str, Any] = field(default_factory=dict)
    
    def as_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "hardware_type": self.hardware_type,
            "supported": self.supported,
            "optimal": self.optimal,
            "performance_ranking": self.performance_ranking,
            "memory_ranking": self.memory_ranking,
            "extra_info": self.extra_info
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PrecisionInfo':
        """Create instance from dictionary"""
        return cls(**data)


@dataclass
class DetectionResult:
    """Stores complete detection results"""
    hardware: Dict[str, HardwareInfo] = field(default_factory=dict)
    precision: Dict[str, PrecisionInfo] = field(default_factory=dict)
    recommended_config: Dict[str, Any] = field(default_factory=dict)
    
    def as_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "hardware": {k: v.as_dict() for k, v in self.hardware.items()},
            "precision": {k: v.as_dict() for k, v in self.precision.items()},
            "recommended_config": self.recommended_config
        }
    
    def save(self, filepath: str) -> None:
        """Save detection results to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.as_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'DetectionResult':
        """Load detection results from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
            result = cls()
            result.hardware = {k: HardwareInfo.from_dict(v) for k, v in data["hardware"].items()}
            result.precision = {k: PrecisionInfo.from_dict(v) for k, v in data["precision"].items()}
            result.recommended_config = data["recommended_config"]
            return result


def detect_cpu_capabilities() -> HardwareInfo:
    """Detect CPU capabilities and features"""
    info = HardwareInfo(type="cpu", detected=True, count=1)
    
    # Basic CPU information
    info.names = [platform.processor()]
    
    # Try to get more detailed CPU info
    try:
        import cpuinfo
        cpu_data = cpuinfo.get_cpu_info()
        
        # Update with detailed information if available
        if cpu_data:
            info.architecture = cpu_data.get('arch', None)
            info.extra_info["brand_raw"] = cpu_data.get('brand_raw', None)
            info.extra_info["vendor_id_raw"] = cpu_data.get('vendor_id_raw', None)
            
            # Check for specific CPU features
            flags = cpu_data.get('flags', [])
            
            # Features relevant to machine learning
            feature_checks = {
                "avx": "avx" in flags,
                "avx2": "avx2" in flags,
                "avx512f": "avx512f" in flags,
                "fma3": "fma" in flags or "fma3" in flags,
                "sse4_1": "sse4_1" in flags,
                "sse4_2": "sse4_2" in flags
            }
            
            info.extra_info["features"] = feature_checks
            info.extra_info["hz_actual"] = cpu_data.get('hz_actual', None)
            
            # Count physical cores if available
            if 'count' in cpu_data:
                info.count = cpu_data.get('count')
    
    except ImportError:
        # Fallback if py-cpuinfo is not available
        import multiprocessing
        info.count = multiprocessing.cpu_count()
    
    # Try to estimate memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        info.memory_gb = [memory.total / (1024**3)]  # Convert bytes to GB
    except ImportError:
        pass
    
    return info


def detect_cuda_hardware() -> HardwareInfo:
    """Detect CUDA/NVIDIA GPU hardware"""
    info = HardwareInfo(type="cuda", detected=torch.cuda.is_available())
    
    if not info.detected:
        return info
    
    # Basic CUDA information
    info.count = torch.cuda.device_count()
    info.names = [torch.cuda.get_device_name(i) for i in range(info.count)]
    
    # Get compute capabilities for first device
    if info.count > 0:
        info.compute_capability = torch.cuda.get_device_capability(0)
    
    # Get driver version if available
    if hasattr(torch.version, 'cuda'):
        info.api_version = torch.version.cuda
    
    if hasattr(torch.cuda, 'driver_version'):
        try:
            info.driver_version = str(torch.cuda.driver_version())
        except:
            # Some PyTorch versions have driver_version as a property
            info.driver_version = str(torch.cuda.driver_version)
    
    # Get memory information
    info.memory_gb = []
    for i in range(info.count):
        try:
            mem_info = torch.cuda.get_device_properties(i).total_memory
            info.memory_gb.append(mem_info / (1024**3))  # Convert bytes to GB
        except:
            info.memory_gb.append(0.0)
    
    # Try to get more detailed info using subprocess
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=gpu_name,memory.total,driver_version,compute_cap', '--format=csv'],
                               capture_output=True, text=True)
        if result.returncode == 0:
            info.extra_info["nvidia_smi"] = result.stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    
    return info


def detect_amd_hardware() -> HardwareInfo:
    """Detect AMD GPU hardware"""
    # Check if PyTorch was built with ROCm support
    has_hip = False
    try:
        import torch.utils.hip
        has_hip = torch.utils.hip.is_available()
    except ImportError:
        pass
    
    info = HardwareInfo(type="amd", detected=has_hip)
    
    if not info.detected:
        return info
    
    # AMD GPU detection - similar to CUDA but using different APIs
    # PyTorch with ROCm uses the CUDA API names
    if has_hip:
        info.count = torch.cuda.device_count()
        info.names = [torch.cuda.get_device_name(i) for i in range(info.count)]
        
        # Get memory info
        info.memory_gb = []
        for i in range(info.count):
            try:
                mem_info = torch.cuda.get_device_properties(i).total_memory
                info.memory_gb.append(mem_info / (1024**3))  # Convert bytes to GB
            except:
                info.memory_gb.append(0.0)
    
    # Try to get more detailed info using rocm-smi
    try:
        result = subprocess.run(['rocm-smi', '--showproductname', '--showmeminfo', 'vram'],
                               capture_output=True, text=True)
        if result.returncode == 0:
            info.extra_info["rocm_smi"] = result.stdout
            
        # Try to get ROCm version
        version_result = subprocess.run(['rocm-smi', '--showdriverversion'],
                                     capture_output=True, text=True)
        if version_result.returncode == 0:
            info.driver_version = version_result.stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    
    return info


def detect_apple_mps_hardware() -> HardwareInfo:
    """Detect Apple Silicon MPS hardware"""
    has_mps = False
    try:
        from torch.backends import mps
        has_mps = mps.is_available()
    except ImportError:
        pass
    
    info = HardwareInfo(type="mps", detected=has_mps)
    
    if not info.detected:
        return info
    
    # Limited information available through PyTorch
    info.count = 1
    info.names = ["Apple Silicon"]
    
    # Try to get more specific information about the chip
    try:
        result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                               capture_output=True, text=True)
        if result.returncode == 0:
            info.names = [result.stdout.strip()]
            
        # Try to determine if it's M1, M2, etc.
        if "Apple M1" in info.names[0]:
            info.architecture = "M1"
        elif "Apple M2" in info.names[0]:
            info.architecture = "M2"
        elif "Apple M3" in info.names[0]:
            info.architecture = "M3"
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    
    # Try to get memory info
    try:
        import psutil
        vm = psutil.virtual_memory()
        info.memory_gb = [vm.total / (1024**3)]  # System memory, as shared with GPU
    except ImportError:
        pass
    
    return info


def detect_openvino_hardware() -> HardwareInfo:
    """Detect OpenVINO hardware acceleration capabilities"""
    has_openvino = False
    try:
        import openvino as ov
        has_openvino = True
    except ImportError:
        pass
    
    info = HardwareInfo(type="openvino", detected=has_openvino)
    
    if not info.detected:
        return info
    
    # Get OpenVINO version
    try:
        import openvino as ov
        info.api_version = ov.__version__
        
        # Try to get available devices
        core = ov.Core()
        available_devices = core.available_devices
        
        info.count = len(available_devices)
        info.names = available_devices
        
        # Check for specific hardware
        has_cpu = "CPU" in available_devices
        has_gpu = "GPU" in available_devices
        has_vpu = any(device.startswith("VPU") for device in available_devices)
        has_gna = any(device.startswith("GNA") for device in available_devices)
        
        info.extra_info = {
            "has_cpu": has_cpu,
            "has_gpu": has_gpu,
            "has_vpu": has_vpu,
            "has_gna": has_gna,
            "available_devices": available_devices
        }
        
        # For each device, try to get properties
        device_properties = {}
        for device in available_devices:
            try:
                props = core.get_property(device, "PROPERTIES")
                device_properties[device] = props
            except:
                pass
        
        if device_properties:
            info.extra_info["device_properties"] = device_properties
    
    except Exception as e:
        info.extra_info["error"] = str(e)
    
    return info


def detect_qualcomm_hardware() -> HardwareInfo:
    """Detect Qualcomm AI hardware"""
    # Check for Qualcomm AI Engine Direct SDK
    has_qualcomm = False
    try:
        import qti.aisw
        has_qualcomm = True
    except ImportError:
        pass
    
    # Alternative check for SNPE
    if not has_qualcomm:
        try:
            import snpe
            has_qualcomm = True
        except ImportError:
            pass
    
    info = HardwareInfo(type="qualcomm", detected=has_qualcomm)
    
    if not info.detected:
        # On Android, we might be able to detect Qualcomm hardware even without the SDK
        if platform.system() == "Linux":
            try:
                with open("/proc/cpuinfo", "r") as f:
                    cpu_info = f.read()
                    if "Qualcomm" in cpu_info:
                        info.detected = True
                        info.count = 1
                        info.names = ["Qualcomm SoC"]
            except:
                pass
    
    return info


def detect_all_hardware() -> Dict[str, HardwareInfo]:
    """Detect all available hardware platforms"""
    hardware = {}
    
    logger.info("Detecting CPU capabilities...")
    hardware["cpu"] = detect_cpu_capabilities()
    
    logger.info("Checking for NVIDIA CUDA GPUs...")
    hardware["cuda"] = detect_cuda_hardware()
    
    logger.info("Checking for AMD ROCm GPUs...")
    hardware["amd"] = detect_amd_hardware()
    
    logger.info("Checking for Apple Silicon MPS...")
    hardware["mps"] = detect_apple_mps_hardware()
    
    logger.info("Checking for OpenVINO support...")
    hardware["openvino"] = detect_openvino_hardware()
    
    logger.info("Checking for Qualcomm AI acceleration...")
    hardware["qualcomm"] = detect_qualcomm_hardware()
    
    # Log summary of detected hardware
    detected = [hw_type for hw_type, info in hardware.items() if info.detected]
    logger.info(f"Detected hardware: {', '.join(detected)}")
    
    return hardware


def get_optimal_precision(hardware_type: str, hardware_info: HardwareInfo) -> PrecisionInfo:
    """Determine optimal precision types for the given hardware"""
    # Initialize with defaults - all precisions unsupported
    precision_info = PrecisionInfo(hardware_type=hardware_type)
    precision_info.supported = {p: False for p in PRECISION_TYPES}
    
    # CPU precision support
    if hardware_type == "cpu":
        # Base support
        precision_info.supported.update({
            "fp32": True,
            "int8": True
        })
        
        # Check for AVX2 support for bfloat16
        if hardware_info.extra_info.get("features", {}).get("avx2", False):
            precision_info.supported["bf16"] = True
        
        # Check for int4/uint4 support
        try:
            import transformers
            from packaging import version
            if version.parse(transformers.__version__) >= version.parse("4.20.0"):
                precision_info.supported["int4"] = True
                precision_info.supported["uint4"] = True
        except ImportError:
            pass
        
        # Set optimal precision based on features
        if hardware_info.extra_info.get("features", {}).get("avx512f", False):
            precision_info.optimal = "bf16"
        else:
            precision_info.optimal = "fp32"
        
        # Performance ranking for CPU
        precision_info.performance_ranking = ["fp32", "bf16", "int8", "int4", "uint4"]
        precision_info.memory_ranking = ["int4", "uint4", "int8", "bf16", "fp32"]
    
    # CUDA/NVIDIA GPU precision support
    elif hardware_type == "cuda" and hardware_info.detected:
        # Default support
        precision_info.supported.update({
            "fp32": True,
            "fp16": True,
            "int8": True
        })
        
        # Check for tensor cores (compute capability 7.0+)
        if hardware_info.compute_capability and hardware_info.compute_capability >= (7, 0):
            precision_info.supported["int4"] = True
            precision_info.supported["uint4"] = True
        
        # BF16 support (Ampere and later, compute capability 8.0+)
        if hardware_info.compute_capability and hardware_info.compute_capability >= (8, 0):
            precision_info.supported["bf16"] = True
        
        # FP8 support (Hopper architecture, compute capability 9.0+)
        if hardware_info.compute_capability and hardware_info.compute_capability >= (9, 0):
            precision_info.supported["fp8"] = True
        
        # Set optimal precision based on compute capability
        if hardware_info.compute_capability and hardware_info.compute_capability >= (9, 0):
            precision_info.optimal = "fp8"  # Latest Hopper GPUs
        elif hardware_info.compute_capability and hardware_info.compute_capability >= (8, 0):
            precision_info.optimal = "bf16"  # Ampere GPUs
        elif hardware_info.compute_capability and hardware_info.compute_capability >= (7, 0):
            precision_info.optimal = "fp16"  # Volta/Turing GPUs
        else:
            precision_info.optimal = "fp32"  # Older GPUs
        
        # Performance ranking for NVIDIA GPUs
        precision_info.performance_ranking = ["fp8", "bf16", "fp16", "fp32", "int8", "int4", "uint4"]
        precision_info.memory_ranking = ["int4", "uint4", "int8", "fp8", "fp16", "bf16", "fp32"]
    
    # AMD GPU precision support
    elif hardware_type == "amd" and hardware_info.detected:
        # Default support
        precision_info.supported.update({
            "fp32": True,
            "fp16": True,
            "int8": True
        })
        
        # BF16 support depends on ROCm version and GPU architecture
        # Limited int4/uint4 support in ROCm
        
        # Set optimal precision (generally fp16 for AMD GPUs)
        precision_info.optimal = "fp16"
        
        # Performance ranking for AMD GPUs
        precision_info.performance_ranking = ["fp16", "fp32", "int8"]
        precision_info.memory_ranking = ["int8", "fp16", "fp32"]
    
    # Apple Silicon MPS
    elif hardware_type == "mps" and hardware_info.detected:
        # Default support
        precision_info.supported.update({
            "fp32": True,
            "fp16": True
        })
        
        # M1/M2/M3 chips also support int8
        if hardware_info.architecture in ["M1", "M2", "M3"]:
            precision_info.supported["int8"] = True
        
        # Set optimal precision (fp16 is generally best for Apple Silicon)
        precision_info.optimal = "fp16"
        
        # Performance ranking for Apple Silicon
        precision_info.performance_ranking = ["fp16", "fp32", "int8"]
        precision_info.memory_ranking = ["int8", "fp16", "fp32"]
    
    # OpenVINO
    elif hardware_type == "openvino" and hardware_info.detected:
        # Default support
        precision_info.supported.update({
            "fp32": True,
            "fp16": True,
            "int8": True,
            "int4": True,
            "uint4": True
        })
        
        # Set optimal precision (depends on the specific hardware)
        if "GPU" in hardware_info.names:
            precision_info.optimal = "fp16"  # Intel GPUs
        else:
            precision_info.optimal = "int8"  # Intel CPUs with OpenVINO
        
        # Performance ranking for OpenVINO
        precision_info.performance_ranking = ["int8", "fp16", "fp32", "int4", "uint4"]
        precision_info.memory_ranking = ["int4", "uint4", "int8", "fp16", "fp32"]
    
    # Qualcomm
    elif hardware_type == "qualcomm" and hardware_info.detected:
        # Default support
        precision_info.supported.update({
            "fp32": True,
            "fp16": True,
            "int8": True
        })
        
        # Set optimal precision
        precision_info.optimal = "fp16"  # Generally best for mobile hardware
        
        # Performance ranking for Qualcomm
        precision_info.performance_ranking = ["fp16", "int8", "fp32"]
        precision_info.memory_ranking = ["int8", "fp16", "fp32"]
    
    # Filter performance and memory rankings to only include supported precisions
    precision_info.performance_ranking = [p for p in precision_info.performance_ranking 
                                         if precision_info.supported.get(p, False)]
    precision_info.memory_ranking = [p for p in precision_info.memory_ranking 
                                    if precision_info.supported.get(p, False)]
    
    return precision_info


def determine_precision_for_all_hardware(hardware: Dict[str, HardwareInfo]) -> Dict[str, PrecisionInfo]:
    """Determine optimal precision for all detected hardware"""
    precision_info = {}
    
    for hw_type, hw_info in hardware.items():
        if hw_info.detected:
            logger.info(f"Determining optimal precision for {hw_type}...")
            precision_info[hw_type] = get_optimal_precision(hw_type, hw_info)
    
    return precision_info


def generate_recommended_config(hardware: Dict[str, HardwareInfo], 
                              precision: Dict[str, PrecisionInfo]) -> Dict[str, Any]:
    """Generate recommended configuration based on detected hardware and precision"""
    config = {
        "primary_hardware": None,
        "fallback_hardware": [],
        "optimal_precision": {},
        "model_recommendations": {},
        "package_requirements": []
    }
    
    # Determine primary hardware in priority order
    hw_priority = ["cuda", "amd", "mps", "openvino", "qualcomm", "cpu"]
    for hw in hw_priority:
        if hw in hardware and hardware[hw].detected:
            config["primary_hardware"] = hw
            break
    
    # Determine fallback hardware
    detected_hw = [hw for hw, info in hardware.items() if info.detected]
    config["fallback_hardware"] = [hw for hw in detected_hw if hw != config["primary_hardware"]]
    
    # Optimal precision for each hardware
    for hw, precision_info in precision.items():
        if precision_info.optimal:
            config["optimal_precision"][hw] = precision_info.optimal
    
    # Model-specific recommendations
    # For BERT models
    config["model_recommendations"]["bert"] = {
        "hardware": config["primary_hardware"],
        "precision": precision.get(config["primary_hardware"], PrecisionInfo("")).optimal,
        "batch_size": 8 if config["primary_hardware"] in ["cuda", "amd"] else 4
    }
    
    # For T5 models
    config["model_recommendations"]["t5"] = {
        "hardware": config["primary_hardware"],
        "precision": precision.get(config["primary_hardware"], PrecisionInfo("")).optimal,
        "batch_size": 4 if config["primary_hardware"] in ["cuda", "amd"] else 1
    }
    
    # For Vision Transformer models
    config["model_recommendations"]["vit"] = {
        "hardware": config["primary_hardware"],
        "precision": precision.get(config["primary_hardware"], PrecisionInfo("")).optimal,
        "batch_size": 16 if config["primary_hardware"] in ["cuda", "amd"] else 8
    }
    
    # Package requirements based on detected hardware
    if hardware.get("cuda", HardwareInfo("cuda")).detected:
        config["package_requirements"].append("torch>=2.0.0")
        config["package_requirements"].append("nvidia-ml-py>=11.495.46")
    
    if hardware.get("amd", HardwareInfo("amd")).detected:
        config["package_requirements"].append("torch>=2.0.0+rocm")
    
    if hardware.get("mps", HardwareInfo("mps")).detected:
        config["package_requirements"].append("torch>=2.0.0")
    
    if hardware.get("openvino", HardwareInfo("openvino")).detected:
        config["package_requirements"].append("openvino>=2023.0.0")
        config["package_requirements"].append("openvino-tensorflow>=2023.0.0")
    
    if hardware.get("qualcomm", HardwareInfo("qualcomm")).detected:
        config["package_requirements"].append("qti-aisw>=1.0.0")
    
    # Add common requirements
    config["package_requirements"].extend([
        "transformers>=4.30.0",
        "accelerate>=0.20.0",
        "optimum>=1.8.0",
        "bitsandbytes>=0.39.0",
        "py-cpuinfo>=9.0.0"
    ])
    
    return config


def print_detection_report(result: DetectionResult) -> None:
    """Print human-readable report of detection results"""
    print("\n" + "="*80)
    print(" HARDWARE AND PRECISION AUTO-DETECTION REPORT ")
    print("="*80)
    
    # Print detected hardware
    print("\nDETECTED HARDWARE:")
    print("-----------------")
    
    for hw_type, hw_info in result.hardware.items():
        if hw_info.detected:
            print(f"✓ {hw_type.upper()}:")
            if hw_info.count > 0:
                for i, name in enumerate(hw_info.names):
                    memory_str = f" ({hw_info.memory_gb[i]:.1f} GB)" if i < len(hw_info.memory_gb) else ""
                    print(f"  - {name}{memory_str}")
            if hw_info.api_version:
                print(f"  - API Version: {hw_info.api_version}")
            if hw_info.driver_version:
                print(f"  - Driver: {hw_info.driver_version}")
            if hw_info.compute_capability:
                print(f"  - Compute Capability: {hw_info.compute_capability[0]}.{hw_info.compute_capability[1]}")
            print()
    
    # Print supported precision
    print("\nSUPPORTED PRECISION:")
    print("-------------------")
    
    for hw_type, precision_info in result.precision.items():
        print(f"{hw_type.upper()}:")
        
        # Print supported precision types
        supported = [p for p, is_supported in precision_info.supported.items() if is_supported]
        print(f"  - Supported: {', '.join(supported)}")
        
        # Print optimal precision
        if precision_info.optimal:
            print(f"  - Optimal: {precision_info.optimal} ⭐")
        
        # Print performance ranking
        if precision_info.performance_ranking:
            print(f"  - Performance (best to worst): {' > '.join(precision_info.performance_ranking)}")
        print()
    
    # Print recommendations
    print("\nRECOMMENDED CONFIGURATION:")
    print("-------------------------")
    
    primary_hw = result.recommended_config.get("primary_hardware")
    fallback_hw = result.recommended_config.get("fallback_hardware", [])
    
    if primary_hw:
        print(f"Primary Hardware: {primary_hw.upper()}")
        optimal_precision = result.recommended_config.get("optimal_precision", {}).get(primary_hw)
        if optimal_precision:
            print(f"Optimal Precision: {optimal_precision}")
    
    if fallback_hw:
        print(f"Fallback Hardware: {', '.join(fallback_hw).upper()}")
    
    # Print model-specific recommendations
    print("\nMODEL-SPECIFIC RECOMMENDATIONS:")
    print("------------------------------")
    
    for model, config in result.recommended_config.get("model_recommendations", {}).items():
        print(f"{model.upper()}:")
        print(f"  - Hardware: {config.get('hardware', 'cpu').upper()}")
        print(f"  - Precision: {config.get('precision', 'fp32')}")
        print(f"  - Batch Size: {config.get('batch_size', 1)}")
        print()
    
    # Print package requirements
    print("\nRECOMMENDED PACKAGE REQUIREMENTS:")
    print("-------------------------------")
    
    for req in result.recommended_config.get("package_requirements", []):
        print(f"- {req}")
    
    print("\n" + "="*80)


def generate_requirements_file(result: DetectionResult, output_path: str) -> None:
    """Generate requirements.txt based on detection results"""
    with open(output_path, 'w') as f:
        for req in result.recommended_config.get("package_requirements", []):
            f.write(f"{req}\n")
    
    logger.info(f"Requirements file generated at {output_path}")


def generate_config_file(result: DetectionResult, output_path: str) -> None:
    """Generate configuration file based on detection results"""
    config = {
        "hardware": {
            "primary": result.recommended_config.get("primary_hardware"),
            "fallback": result.recommended_config.get("fallback_hardware", [])
        },
        "precision": result.recommended_config.get("optimal_precision", {}),
        "models": result.recommended_config.get("model_recommendations", {})
    }
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Configuration file generated at {output_path}")


def main():
    """Main function for hardware and precision auto-detection"""
    parser = argparse.ArgumentParser(description="Hardware and precision auto-detection tool")
    parser.add_argument("--output", default="auto_detection_results.json",
                      help="Output JSON file for detection results")
    parser.add_argument("--generate-requirements", action="store_true",
                      help="Generate requirements.txt based on detected hardware")
    parser.add_argument("--generate-config", action="store_true",
                      help="Generate hardware configuration file")
    parser.add_argument("--requirements-file", default="requirements.txt",
                      help="Path for generated requirements.txt")
    parser.add_argument("--config-file", default="hardware_config.json",
                      help="Path for generated configuration file")
    
    args = parser.parse_args()
    
    logger.info("Starting hardware and precision auto-detection")
    
    # Detect hardware
    hardware = detect_all_hardware()
    
    # Determine precision for detected hardware
    precision = determine_precision_for_all_hardware(hardware)
    
    # Generate recommended configuration
    config = generate_recommended_config(hardware, precision)
    
    # Create result
    result = DetectionResult(
        hardware=hardware,
        precision=precision,
        recommended_config=config
    )
    
    # Save detection results
    result.save(args.output)
    logger.info(f"Detection results saved to {args.output}")
    
    # Print report
    print_detection_report(result)
    
    # Generate requirements file if requested
    if args.generate_requirements:
        generate_requirements_file(result, args.requirements_file)
    
    # Generate configuration file if requested
    if args.generate_config:
        generate_config_file(result, args.config_file)


if __name__ == "__main__":
    main()