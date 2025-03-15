#!/usr/bin/env python3
"""
Enhanced Hardware Capability Reporting System

This module provides a comprehensive system for detecting, reporting, and
comparing hardware capabilities in the Distributed Testing Framework. It
integrates with the Hardware Taxonomy to provide detailed classification
and comparison of hardware capabilities.

Key features:
- Automatic hardware capability detection
- Structured capability representation
- Capability comparison and compatibility checking
- Integration with Hardware Taxonomy
- Support for specialized hardware types (GPU, TPU, NPU, WebGPU, etc.)
"""

import logging
import json
import os
import platform
import re
import subprocess
from enum import Enum
from typing import Dict, List, Set, Optional, Tuple, Any, Union, DefaultDict
from dataclasses import dataclass, field

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("enhanced_hardware_capability")


class HardwareType(Enum):
    """Types of hardware capabilities."""
    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"
    NPU = "npu"
    DSP = "dsp"
    FPGA = "fpga"
    ASIC = "asic"
    WEBGPU = "webgpu"
    WEBNN = "webnn"
    OTHER = "other"


class PrecisionType(Enum):
    """Precision types supported by hardware."""
    FP64 = "fp64"
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT64 = "int64"
    INT32 = "int32"
    INT16 = "int16"
    INT8 = "int8"
    INT4 = "int4"
    INT2 = "int2"
    MIXED = "mixed"


class HardwareVendor(Enum):
    """Common hardware vendors."""
    INTEL = "intel"
    AMD = "amd"
    NVIDIA = "nvidia"
    APPLE = "apple"
    QUALCOMM = "qualcomm"
    MEDIATEK = "mediatek"
    SAMSUNG = "samsung"
    GOOGLE = "google"
    MICROSOFT = "microsoft"
    AMAZON = "amazon"
    HUAWEI = "huawei"
    ARM = "arm"
    XILINX = "xilinx"
    UNKNOWN = "unknown"


class CapabilityScore(Enum):
    """Capability score levels."""
    EXCELLENT = 5
    GOOD = 4
    AVERAGE = 3
    BASIC = 2
    MINIMAL = 1
    UNKNOWN = 0


@dataclass
class HardwareCapability:
    """Represents a hardware capability with detailed information."""
    hardware_type: HardwareType
    vendor: HardwareVendor = HardwareVendor.UNKNOWN
    model: str = "Unknown"
    version: Optional[str] = None
    driver_version: Optional[str] = None
    compute_units: Optional[int] = None
    cores: Optional[int] = None
    memory_gb: Optional[float] = None
    memory_bandwidth_gbps: Optional[float] = None
    supported_precisions: List[PrecisionType] = field(default_factory=list)
    theoretical_flops: Optional[float] = None
    capabilities: Dict[str, Union[bool, str, int, float]] = field(default_factory=dict)
    scores: Dict[str, CapabilityScore] = field(default_factory=dict)
    taxonomy_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkerHardwareCapabilities:
    """Hardware capabilities for a worker node."""
    worker_id: str
    os_type: str
    os_version: str
    hostname: str
    cpu_count: int
    total_memory_gb: float
    hardware_capabilities: List[HardwareCapability] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    last_updated: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class HardwareCapabilityDetector:
    """
    Detects hardware capabilities on the current system.
    
    This class provides methods to automatically detect CPU, GPU, and other
    hardware capabilities, creating structured capability information.
    """
    
    def __init__(self, worker_id: Optional[str] = None):
        """
        Initialize the hardware capability detector.
        
        Args:
            worker_id: Optional worker ID (will be auto-generated if not provided)
        """
        self.worker_id = worker_id or self._generate_worker_id()
        self.os_info = self._get_os_info()
        self.hostname = self._get_hostname()
        
        logger.info(f"Hardware Capability Detector initialized for worker {self.worker_id}")
    
    def _generate_worker_id(self) -> str:
        """Generate a unique worker ID based on hostname and timestamp."""
        import socket
        import time
        import hashlib
        
        hostname = socket.gethostname()
        timestamp = int(time.time())
        worker_hash = hashlib.md5(f"{hostname}_{timestamp}".encode()).hexdigest()[:12]
        
        return f"worker_{worker_hash}"
    
    def _get_os_info(self) -> Tuple[str, str]:
        """Get the OS type and version."""
        os_type = platform.system()
        os_version = platform.version()
        
        return os_type, os_version
    
    def _get_hostname(self) -> str:
        """Get the hostname."""
        import socket
        return socket.gethostname()
    
    def detect_cpu_capabilities(self) -> HardwareCapability:
        """
        Detect CPU capabilities on the current system.
        
        Returns:
            HardwareCapability for the CPU
        """
        import psutil
        import cpuinfo
        
        try:
            # Get CPU info
            cpu_info = cpuinfo.get_cpu_info()
            
            # Extract vendor
            vendor_str = cpu_info.get('vendor_id', '').lower()
            vendor = HardwareVendor.UNKNOWN
            if 'intel' in vendor_str:
                vendor = HardwareVendor.INTEL
            elif 'amd' in vendor_str:
                vendor = HardwareVendor.AMD
            elif 'apple' in vendor_str:
                vendor = HardwareVendor.APPLE
            elif 'arm' in vendor_str:
                vendor = HardwareVendor.ARM
            
            # Create capability
            capability = HardwareCapability(
                hardware_type=HardwareType.CPU,
                vendor=vendor,
                model=cpu_info.get('brand_raw', 'Unknown CPU'),
                version=cpu_info.get('model', None),
                compute_units=None,  # Not directly available
                cores=psutil.cpu_count(logical=False),
                memory_gb=psutil.virtual_memory().total / (1024**3),
                supported_precisions=[
                    PrecisionType.FP64,
                    PrecisionType.FP32,
                    PrecisionType.INT64,
                    PrecisionType.INT32,
                ],
                capabilities={
                    'threads': psutil.cpu_count(logical=True),
                    'architecture': cpu_info.get('arch', 'unknown'),
                    'frequency_mhz': cpu_info.get('hz_advertised_raw', [0])[0] / 1000000,
                    'l1_cache_kb': cpu_info.get('l1_data_cache_size', 0) / 1024 if 'l1_data_cache_size' in cpu_info else None,
                    'l2_cache_kb': cpu_info.get('l2_cache_size', 0) / 1024 if 'l2_cache_size' in cpu_info else None,
                    'l3_cache_kb': cpu_info.get('l3_cache_size', 0) / 1024 if 'l3_cache_size' in cpu_info else None,
                    'avx': 'avx' in cpu_info.get('flags', []),
                    'avx2': 'avx2' in cpu_info.get('flags', []),
                    'avx512': any('avx512' in flag for flag in cpu_info.get('flags', [])),
                    'sse': 'sse' in cpu_info.get('flags', []),
                    'sse2': 'sse2' in cpu_info.get('flags', []),
                    'sse3': 'sse3' in cpu_info.get('flags', []),
                    'sse4': 'sse4_1' in cpu_info.get('flags', []) or 'sse4_2' in cpu_info.get('flags', []),
                    'vnni': 'avx512_vnni' in cpu_info.get('flags', []),
                    'fma': 'fma' in cpu_info.get('flags', []),
                }
            )
            
            # Add ARM-specific capabilities if applicable
            if vendor == HardwareVendor.ARM:
                capability.capabilities.update({
                    'neon': cpu_info.get('flags', '') and 'neon' in cpu_info['flags'],
                    'asimd': cpu_info.get('flags', '') and 'asimd' in cpu_info['flags'],
                    'sve': cpu_info.get('flags', '') and 'sve' in cpu_info['flags'],
                })
                
                # Add Apple-specific capabilities
                if platform.system() == 'Darwin' and vendor == HardwareVendor.APPLE:
                    capability.capabilities.update({
                        'apple_silicon': 'Apple' in cpu_info.get('brand_raw', ''),
                        'neural_engine': 'Apple' in cpu_info.get('brand_raw', ''),  # Assumption for Apple Silicon
                    })
                    
                    # Update supported precisions for Apple Silicon
                    if 'Apple' in cpu_info.get('brand_raw', ''):
                        if PrecisionType.BF16 not in capability.supported_precisions:
                            capability.supported_precisions.append(PrecisionType.BF16)
                        if PrecisionType.FP16 not in capability.supported_precisions:
                            capability.supported_precisions.append(PrecisionType.FP16)
            
            # Calculate scores
            scores = {
                'compute': self._calculate_cpu_compute_score(capability),
                'memory': self._calculate_cpu_memory_score(capability),
                'vector': self._calculate_cpu_vector_score(capability),
                'precision': self._calculate_cpu_precision_score(capability),
                'overall': CapabilityScore.UNKNOWN  # Will be calculated
            }
            
            # Calculate overall score (average of other scores)
            score_values = [score.value for score in scores.values() if score != CapabilityScore.UNKNOWN]
            if score_values:
                avg_score = sum(score_values) / len(score_values)
                overall_score = CapabilityScore.UNKNOWN
                for score in CapabilityScore:
                    if score != CapabilityScore.UNKNOWN and abs(score.value - avg_score) <= 0.5:
                        overall_score = score
                        break
                scores['overall'] = overall_score
            
            capability.scores = scores
            
            logger.info(f"Detected CPU: {capability.model} with {capability.cores} cores")
            return capability
        
        except Exception as e:
            logger.error(f"Error detecting CPU capabilities: {str(e)}")
            # Return a minimal CPU capability
            return HardwareCapability(
                hardware_type=HardwareType.CPU,
                model="Unknown CPU",
                cores=psutil.cpu_count(logical=False),
                memory_gb=psutil.virtual_memory().total / (1024**3)
            )
    
    def _calculate_cpu_compute_score(self, capability: HardwareCapability) -> CapabilityScore:
        """Calculate CPU compute capability score."""
        cores = capability.cores or 0
        threads = capability.capabilities.get('threads', 0)
        freq = capability.capabilities.get('frequency_mhz', 0)
        
        if cores >= 32 and freq >= 3000:
            return CapabilityScore.EXCELLENT
        elif cores >= 16 and freq >= 2500:
            return CapabilityScore.GOOD
        elif cores >= 8 and freq >= 2000:
            return CapabilityScore.AVERAGE
        elif cores >= 4 and freq >= 1500:
            return CapabilityScore.BASIC
        elif cores > 0:
            return CapabilityScore.MINIMAL
        else:
            return CapabilityScore.UNKNOWN
    
    def _calculate_cpu_memory_score(self, capability: HardwareCapability) -> CapabilityScore:
        """Calculate CPU memory capability score."""
        memory_gb = capability.memory_gb or 0
        l3_cache = capability.capabilities.get('l3_cache_kb', 0)
        
        if memory_gb >= 128 and l3_cache >= 32768:
            return CapabilityScore.EXCELLENT
        elif memory_gb >= 64 and l3_cache >= 16384:
            return CapabilityScore.GOOD
        elif memory_gb >= 16 and l3_cache >= 8192:
            return CapabilityScore.AVERAGE
        elif memory_gb >= 4:
            return CapabilityScore.BASIC
        elif memory_gb > 0:
            return CapabilityScore.MINIMAL
        else:
            return CapabilityScore.UNKNOWN
    
    def _calculate_cpu_vector_score(self, capability: HardwareCapability) -> CapabilityScore:
        """Calculate CPU vector capability score."""
        caps = capability.capabilities
        
        if caps.get('avx512', False):
            return CapabilityScore.EXCELLENT
        elif caps.get('avx2', False) and caps.get('fma', False):
            return CapabilityScore.GOOD
        elif caps.get('avx', False):
            return CapabilityScore.AVERAGE
        elif caps.get('sse4', False):
            return CapabilityScore.BASIC
        elif caps.get('sse', False):
            return CapabilityScore.MINIMAL
        else:
            return CapabilityScore.UNKNOWN
    
    def _calculate_cpu_precision_score(self, capability: HardwareCapability) -> CapabilityScore:
        """Calculate CPU precision capability score."""
        precisions = capability.supported_precisions
        
        if (PrecisionType.FP64 in precisions and 
            PrecisionType.BF16 in precisions):
            return CapabilityScore.EXCELLENT
        elif (PrecisionType.FP64 in precisions and 
              PrecisionType.FP32 in precisions and 
              any(p in precisions for p in [PrecisionType.FP16, PrecisionType.INT8])):
            return CapabilityScore.GOOD
        elif PrecisionType.FP32 in precisions and PrecisionType.INT32 in precisions:
            return CapabilityScore.AVERAGE
        elif PrecisionType.FP32 in precisions:
            return CapabilityScore.BASIC
        elif precisions:
            return CapabilityScore.MINIMAL
        else:
            return CapabilityScore.UNKNOWN
    
    def detect_gpu_capabilities(self) -> List[HardwareCapability]:
        """
        Detect GPU capabilities on the current system.
        
        Returns:
            List of HardwareCapability objects for GPUs
        """
        gpu_capabilities = []
        
        # Check for NVIDIA GPUs using pynvml
        nvidia_gpus = self._detect_nvidia_gpus()
        if nvidia_gpus:
            gpu_capabilities.extend(nvidia_gpus)
        
        # Check for AMD GPUs using rocm-smi
        amd_gpus = self._detect_amd_gpus()
        if amd_gpus:
            gpu_capabilities.extend(amd_gpus)
        
        # Check for Apple GPUs (Metal)
        apple_gpus = self._detect_apple_gpus()
        if apple_gpus:
            gpu_capabilities.extend(apple_gpus)
        
        logger.info(f"Detected {len(gpu_capabilities)} GPUs")
        return gpu_capabilities
    
    def _detect_nvidia_gpus(self) -> List[HardwareCapability]:
        """Detect NVIDIA GPUs using pynvml."""
        try:
            import pynvml
            
            pynvml.nvmlInit()
            nvidia_gpus = []
            
            try:
                device_count = pynvml.nvmlDeviceGetCount()
                
                for i in range(device_count):
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        device_info = pynvml.nvmlDeviceGetName(handle)
                        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        compute_capability = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                        
                        # Get driver version
                        driver_version = pynvml.nvmlSystemGetDriverVersion()
                        
                        # Determine supported precisions based on compute capability
                        supported_precisions = [PrecisionType.FP32, PrecisionType.INT32]
                        
                        if compute_capability[0] >= 6:
                            supported_precisions.append(PrecisionType.FP16)
                        
                        if compute_capability[0] >= 7:
                            supported_precisions.append(PrecisionType.INT8)
                            
                        if compute_capability[0] >= 8:
                            supported_precisions.append(PrecisionType.BF16)
                            supported_precisions.append(PrecisionType.INT4)
                            
                        if compute_capability[0] >= 9:
                            supported_precisions.append(PrecisionType.INT2)
                            
                        # Create capability object
                        gpu_capability = HardwareCapability(
                            hardware_type=HardwareType.GPU,
                            vendor=HardwareVendor.NVIDIA,
                            model=device_info.decode('utf-8') if isinstance(device_info, bytes) else device_info,
                            version=f"{compute_capability[0]}.{compute_capability[1]}",
                            driver_version=driver_version,
                            compute_units=None,  # Not directly available
                            memory_gb=memory_info.total / (1024**3),
                            supported_precisions=supported_precisions,
                            capabilities={
                                'compute_capability': f"{compute_capability[0]}.{compute_capability[1]}",
                                'cuda_cores': None,  # Not directly available
                                'tensor_cores': compute_capability[0] >= 7,
                                'ecc_enabled': pynvml.nvmlDeviceGetTotalEccErrors(handle, 0) != -1,
                                'tcc_driver': False,  # Default value
                                'memory_used_gb': memory_info.used / (1024**3),
                            }
                        )
                        
                        # Calculate scores
                        scores = {
                            'compute': self._calculate_nvidia_gpu_compute_score(gpu_capability),
                            'memory': self._calculate_gpu_memory_score(gpu_capability),
                            'precision': self._calculate_gpu_precision_score(gpu_capability),
                            'overall': CapabilityScore.UNKNOWN
                        }
                        
                        # Calculate overall score
                        score_values = [score.value for score in scores.values() if score != CapabilityScore.UNKNOWN]
                        if score_values:
                            avg_score = sum(score_values) / len(score_values)
                            overall_score = CapabilityScore.UNKNOWN
                            for score in CapabilityScore:
                                if score != CapabilityScore.UNKNOWN and abs(score.value - avg_score) <= 0.5:
                                    overall_score = score
                                    break
                            scores['overall'] = overall_score
                        
                        gpu_capability.scores = scores
                        
                        # Add to list
                        nvidia_gpus.append(gpu_capability)
                    
                    except Exception as e:
                        logger.error(f"Error getting info for NVIDIA GPU {i}: {str(e)}")
                
                return nvidia_gpus
            
            finally:
                pynvml.nvmlShutdown()
        
        except ImportError:
            logger.warning("pynvml not installed, cannot detect NVIDIA GPUs using NVML")
            return []
        
        except Exception as e:
            logger.error(f"Error initializing NVML: {str(e)}")
            return []
    
    def _calculate_nvidia_gpu_compute_score(self, capability: HardwareCapability) -> CapabilityScore:
        """Calculate NVIDIA GPU compute capability score."""
        cc_str = capability.capabilities.get('compute_capability', '0.0')
        try:
            cc_major = int(cc_str.split('.')[0])
            
            if cc_major >= 9:
                return CapabilityScore.EXCELLENT
            elif cc_major >= 8:
                return CapabilityScore.EXCELLENT
            elif cc_major >= 7:
                return CapabilityScore.GOOD
            elif cc_major >= 6:
                return CapabilityScore.AVERAGE
            elif cc_major >= 5:
                return CapabilityScore.BASIC
            else:
                return CapabilityScore.MINIMAL
        
        except (ValueError, IndexError):
            return CapabilityScore.UNKNOWN
    
    def _detect_amd_gpus(self) -> List[HardwareCapability]:
        """Detect AMD GPUs using rocm-smi if available."""
        if platform.system() != 'Linux':
            return []
        
        try:
            # Check if rocm-smi is available
            result = subprocess.run(['which', 'rocm-smi'], 
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE)
            
            if result.returncode != 0:
                logger.warning("rocm-smi not found, cannot detect AMD GPUs")
                return []
            
            # Run rocm-smi to get GPU information
            result = subprocess.run(['rocm-smi', '--showallinfo', '--json'], 
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE)
            
            if result.returncode != 0:
                logger.error(f"Error running rocm-smi: {result.stderr.decode('utf-8').strip()}")
                return []
            
            # Parse JSON output
            try:
                gpu_data = json.loads(result.stdout.decode('utf-8'))
                amd_gpus = []
                
                for gpu_id, info in gpu_data.items():
                    if not isinstance(info, dict):
                        continue
                    
                    gpu_info = info.get('Card series', 'Unknown AMD GPU')
                    memory_info = info.get('Memory Total', '0 GB')
                    memory_gb = 0
                    
                    # Extract memory size
                    memory_match = re.search(r'(\d+(?:\.\d+)?)\s*GB', memory_info)
                    if memory_match:
                        memory_gb = float(memory_match.group(1))
                    
                    # Create capability object
                    gpu_capability = HardwareCapability(
                        hardware_type=HardwareType.GPU,
                        vendor=HardwareVendor.AMD,
                        model=gpu_info,
                        version=info.get('Card model', None),
                        driver_version=info.get('Driver version', None),
                        compute_units=info.get('Compute Units', None),
                        memory_gb=memory_gb,
                        supported_precisions=[
                            PrecisionType.FP32,
                            PrecisionType.FP16,
                            PrecisionType.INT32,
                            PrecisionType.INT16
                        ],
                        capabilities={
                            'memory_clock': info.get('Memory Clock', None),
                            'gpu_clock': info.get('GPU Clock', None),
                            'memory_used': info.get('Memory Used', None),
                            'temperature': info.get('Temperature', None),
                            'power': info.get('Power', None),
                        }
                    )
                    
                    # Calculate scores - more basic for AMD due to less info
                    scores = {
                        'compute': self._calculate_amd_gpu_compute_score(gpu_capability),
                        'memory': self._calculate_gpu_memory_score(gpu_capability),
                        'precision': self._calculate_gpu_precision_score(gpu_capability),
                        'overall': CapabilityScore.UNKNOWN
                    }
                    
                    # Calculate overall score
                    score_values = [score.value for score in scores.values() if score != CapabilityScore.UNKNOWN]
                    if score_values:
                        avg_score = sum(score_values) / len(score_values)
                        overall_score = CapabilityScore.UNKNOWN
                        for score in CapabilityScore:
                            if score != CapabilityScore.UNKNOWN and abs(score.value - avg_score) <= 0.5:
                                overall_score = score
                                break
                        scores['overall'] = overall_score
                    
                    gpu_capability.scores = scores
                    
                    # Add to list
                    amd_gpus.append(gpu_capability)
                
                return amd_gpus
            
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing rocm-smi output: {str(e)}")
                return []
        
        except Exception as e:
            logger.error(f"Error detecting AMD GPUs: {str(e)}")
            return []
    
    def _calculate_amd_gpu_compute_score(self, capability: HardwareCapability) -> CapabilityScore:
        """Calculate AMD GPU compute capability score based on model and compute units."""
        model = capability.model.lower() if capability.model else ""
        compute_units = capability.compute_units or 0
        
        # MI series
        if "mi300" in model or "mi250" in model or "mi210" in model:
            return CapabilityScore.EXCELLENT
        elif "mi100" in model or "mi50" in model:
            return CapabilityScore.GOOD
        # RDNA series
        elif "rdna 3" in model or "navi 3" in model:
            return CapabilityScore.EXCELLENT
        elif "rdna 2" in model or "navi 2" in model:
            return CapabilityScore.GOOD
        elif "rdna" in model or "navi" in model:
            return CapabilityScore.AVERAGE
        # Check compute units if model-based detection fails
        elif compute_units >= 100:
            return CapabilityScore.EXCELLENT
        elif compute_units >= 60:
            return CapabilityScore.GOOD
        elif compute_units >= 40:
            return CapabilityScore.AVERAGE
        elif compute_units >= 20:
            return CapabilityScore.BASIC
        elif compute_units > 0:
            return CapabilityScore.MINIMAL
        else:
            return CapabilityScore.UNKNOWN
    
    def _detect_apple_gpus(self) -> List[HardwareCapability]:
        """Detect Apple GPUs (Metal) if on macOS."""
        if platform.system() != 'Darwin':
            return []
        
        try:
            # There's no simple CLI tool for Metal info, but we can use system_profiler
            result = subprocess.run(['system_profiler', 'SPDisplaysDataType', '-json'], 
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE)
            
            if result.returncode != 0:
                logger.error(f"Error running system_profiler: {result.stderr.decode('utf-8').strip()}")
                return []
            
            # Parse JSON output
            try:
                gpu_data = json.loads(result.stdout.decode('utf-8'))
                displays_info = gpu_data.get('SPDisplaysDataType', [])
                apple_gpus = []
                
                for display_info in displays_info:
                    if 'spdisplays_metal' not in display_info and 'spdisplays_mtlgpufamilysupport' not in display_info:
                        continue
                    
                    gpu_model = display_info.get('spdisplays_device_name', 'Unknown Apple GPU')
                    metal_support = display_info.get('spdisplays_metal', False)
                    metal_family = display_info.get('spdisplays_mtlgpufamilysupport', 'Apple')
                    
                    # Determine if this is Apple Silicon
                    is_apple_silicon = 'Apple' in gpu_model and not ('Intel' in gpu_model or 'AMD' in gpu_model)
                    
                    # Determine vendor
                    vendor = HardwareVendor.APPLE
                    if 'Intel' in gpu_model:
                        vendor = HardwareVendor.INTEL
                    elif 'AMD' in gpu_model:
                        vendor = HardwareVendor.AMD
                    
                    # Create capability object
                    gpu_capability = HardwareCapability(
                        hardware_type=HardwareType.GPU,
                        vendor=vendor,
                        model=gpu_model,
                        version=metal_family,
                        supported_precisions=[
                            PrecisionType.FP32,
                            PrecisionType.FP16,
                            PrecisionType.INT32,
                            PrecisionType.INT16
                        ],
                        capabilities={
                            'metal_support': metal_support,
                            'metal_family': metal_family,
                            'apple_silicon': is_apple_silicon
                        }
                    )
                    
                    # Add specific precisions for Apple Silicon
                    if is_apple_silicon:
                        gpu_capability.supported_precisions.append(PrecisionType.INT8)
                        if "M1" in gpu_model or "M2" in gpu_model or "M3" in gpu_model:
                            gpu_capability.supported_precisions.append(PrecisionType.BF16)
                    
                    # Calculate scores
                    scores = {
                        'compute': self._calculate_apple_gpu_compute_score(gpu_capability),
                        'precision': self._calculate_gpu_precision_score(gpu_capability),
                        'overall': CapabilityScore.UNKNOWN
                    }
                    
                    # Calculate overall score
                    score_values = [score.value for score in scores.values() if score != CapabilityScore.UNKNOWN]
                    if score_values:
                        avg_score = sum(score_values) / len(score_values)
                        overall_score = CapabilityScore.UNKNOWN
                        for score in CapabilityScore:
                            if score != CapabilityScore.UNKNOWN and abs(score.value - avg_score) <= 0.5:
                                overall_score = score
                                break
                        scores['overall'] = overall_score
                    
                    gpu_capability.scores = scores
                    
                    # Add to list
                    apple_gpus.append(gpu_capability)
                
                return apple_gpus
            
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing system_profiler output: {str(e)}")
                return []
        
        except Exception as e:
            logger.error(f"Error detecting Apple GPUs: {str(e)}")
            return []
    
    def _calculate_apple_gpu_compute_score(self, capability: HardwareCapability) -> CapabilityScore:
        """Calculate Apple GPU compute capability score."""
        model = capability.model.lower() if capability.model else ""
        is_apple_silicon = capability.capabilities.get('apple_silicon', False)
        
        if is_apple_silicon:
            if "m3 max" in model or "m3 ultra" in model or "m3 pro" in model:
                return CapabilityScore.EXCELLENT
            elif "m3" in model or "m2 max" in model or "m2 ultra" in model or "m2 pro" in model:
                return CapabilityScore.EXCELLENT
            elif "m2" in model or "m1 max" in model or "m1 ultra" in model or "m1 pro" in model:
                return CapabilityScore.GOOD
            elif "m1" in model:
                return CapabilityScore.AVERAGE
            else:
                return CapabilityScore.BASIC
        else:
            # Integrated or external GPU
            if "radeon" in model and ("pro" in model or "vega" in model):
                return CapabilityScore.AVERAGE
            elif "radeon" in model:
                return CapabilityScore.BASIC
            elif "intel" in model and "iris" in model:
                return CapabilityScore.MINIMAL
            else:
                return CapabilityScore.MINIMAL
    
    def _calculate_gpu_memory_score(self, capability: HardwareCapability) -> CapabilityScore:
        """Calculate GPU memory capability score."""
        memory_gb = capability.memory_gb or 0
        
        if memory_gb >= 64:
            return CapabilityScore.EXCELLENT
        elif memory_gb >= 24:
            return CapabilityScore.GOOD
        elif memory_gb >= 8:
            return CapabilityScore.AVERAGE
        elif memory_gb >= 4:
            return CapabilityScore.BASIC
        elif memory_gb > 0:
            return CapabilityScore.MINIMAL
        else:
            return CapabilityScore.UNKNOWN
    
    def _calculate_gpu_precision_score(self, capability: HardwareCapability) -> CapabilityScore:
        """Calculate GPU precision capability score."""
        precisions = capability.supported_precisions
        
        if (PrecisionType.FP32 in precisions and
            PrecisionType.FP16 in precisions and
            PrecisionType.INT8 in precisions and
            any(p in precisions for p in [PrecisionType.INT4, PrecisionType.INT2])):
            return CapabilityScore.EXCELLENT
        elif (PrecisionType.FP32 in precisions and
              PrecisionType.FP16 in precisions and
              PrecisionType.INT8 in precisions):
            return CapabilityScore.GOOD
        elif (PrecisionType.FP32 in precisions and
              any(p in precisions for p in [PrecisionType.FP16, PrecisionType.INT8])):
            return CapabilityScore.AVERAGE
        elif PrecisionType.FP32 in precisions:
            return CapabilityScore.BASIC
        elif precisions:
            return CapabilityScore.MINIMAL
        else:
            return CapabilityScore.UNKNOWN
    
    def detect_webgpu_capabilities(self) -> Optional[HardwareCapability]:
        """
        Detect WebGPU capabilities if available.
        
        Returns:
            HardwareCapability for WebGPU or None if not available
        """
        # WebGPU detection typically requires browser automation
        # This is just a placeholder that would be implemented with actual browser testing
        logger.info("WebGPU detection requires browser automation")
        return None
    
    def detect_webnn_capabilities(self) -> Optional[HardwareCapability]:
        """
        Detect WebNN capabilities if available.
        
        Returns:
            HardwareCapability for WebNN or None if not available
        """
        # WebNN detection typically requires browser automation
        # This is just a placeholder that would be implemented with actual browser testing
        logger.info("WebNN detection requires browser automation")
        return None
    
    def detect_tpu_capabilities(self) -> List[HardwareCapability]:
        """
        Detect TPU capabilities if available.
        
        Returns:
            List of HardwareCapability objects for TPUs
        """
        # TPU detection is typically only available on Google Cloud or specialized hardware
        try:
            # Check if TPU library is available
            import tensorflow as tf
            
            # Check if TPUs are available
            tpus = []
            try:
                tpu_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
                tf.config.experimental_connect_to_cluster(tpu_resolver)
                tf.tpu.experimental.initialize_tpu_system(tpu_resolver)
                
                # Get TPU information
                tpu_devices = tf.config.list_logical_devices('TPU')
                
                if tpu_devices:
                    # Create capability object
                    tpu_capability = HardwareCapability(
                        hardware_type=HardwareType.TPU,
                        vendor=HardwareVendor.GOOGLE,
                        model="Google TPU",
                        version=None,  # Not directly available
                        compute_units=len(tpu_devices),
                        supported_precisions=[
                            PrecisionType.FP32,
                            PrecisionType.FP16,
                            PrecisionType.BF16,
                            PrecisionType.INT8
                        ],
                        capabilities={
                            'device_count': len(tpu_devices),
                            'tensorflow_version': tf.__version__
                        }
                    )
                    
                    # Add to list
                    tpus.append(tpu_capability)
                    
                    logger.info(f"Detected TPU with {len(tpu_devices)} devices")
                
                return tpus
            
            except (ValueError, tf.errors.NotFoundError):
                logger.info("No TPUs detected")
                return []
        
        except ImportError:
            logger.info("TensorFlow not installed, cannot detect TPUs")
            return []
        
        except Exception as e:
            logger.error(f"Error detecting TPUs: {str(e)}")
            return []
    
    def detect_npu_capabilities(self) -> List[HardwareCapability]:
        """
        Detect NPU capabilities if available (e.g., for Qualcomm, Samsung, Mediatek, etc.).
        
        Returns:
            List of HardwareCapability objects for NPUs
        """
        # NPU detection depends on the specific hardware and drivers
        npus = []
        
        # Check for Qualcomm NPUs
        qualcomm_npus = self._detect_qualcomm_npus()
        if qualcomm_npus:
            npus.extend(qualcomm_npus)
        
        # Check for Samsung NPUs
        samsung_npus = self._detect_samsung_npus()
        if samsung_npus:
            npus.extend(samsung_npus)
        
        # Check for Mediatek NPUs
        mediatek_npus = self._detect_mediatek_npus()
        if mediatek_npus:
            npus.extend(mediatek_npus)
        
        # Check for Apple Neural Engine
        apple_npus = self._detect_apple_neural_engine()
        if apple_npus:
            npus.extend(apple_npus)
        
        logger.info(f"Detected {len(npus)} NPUs")
        return npus
    
    def _detect_qualcomm_npus(self) -> List[HardwareCapability]:
        """Detect Qualcomm NPUs (AI Engine, HTA, etc.)."""
        try:
            # First check if this is a Qualcomm platform
            if platform.system() == 'Linux':
                try:
                    with open('/proc/cpuinfo', 'r') as f:
                        cpuinfo = f.read().lower()
                        is_qualcomm = 'qualcomm' in cpuinfo or 'qcom' in cpuinfo
                    
                    if not is_qualcomm:
                        return []
                except Exception:
                    return []
            elif platform.system() != 'Android':
                return []
            
            # Check if QNN library is available
            try:
                import qti.aisw.dlc_utils as qnn_utils
                
                # Get QNN version
                qnn_version = getattr(qnn_utils, 'VERSION', 'Unknown')
                
                # Create capability object
                qnn_capability = HardwareCapability(
                    hardware_type=HardwareType.NPU,
                    vendor=HardwareVendor.QUALCOMM,
                    model="Qualcomm AI Engine",
                    version=qnn_version,
                    supported_precisions=[
                        PrecisionType.FP32,
                        PrecisionType.FP16,
                        PrecisionType.INT8
                    ],
                    capabilities={
                        'sdk_version': qnn_version,
                        'backend': 'QNN'
                    }
                )
                
                return [qnn_capability]
            except ImportError:
                # Try SNPE SDK
                try:
                    import snpe
                    
                    # Get SNPE version
                    snpe_version = getattr(snpe, 'version', 'Unknown')
                    
                    # Create capability object
                    snpe_capability = HardwareCapability(
                        hardware_type=HardwareType.NPU,
                        vendor=HardwareVendor.QUALCOMM,
                        model="Qualcomm SNPE",
                        version=snpe_version,
                        supported_precisions=[
                            PrecisionType.FP32,
                            PrecisionType.FP16,
                            PrecisionType.INT8
                        ],
                        capabilities={
                            'sdk_version': snpe_version,
                            'backend': 'SNPE'
                        }
                    )
                    
                    return [snpe_capability]
                except ImportError:
                    # No Qualcomm AI libraries found
                    return []
        
        except Exception as e:
            logger.error(f"Error detecting Qualcomm NPUs: {str(e)}")
            return []
    
    def _detect_samsung_npus(self) -> List[HardwareCapability]:
        """Detect Samsung NPUs (Neural Processing Unit)."""
        try:
            # First check if this is a Samsung platform
            if platform.system() == 'Linux':
                try:
                    with open('/proc/cpuinfo', 'r') as f:
                        cpuinfo = f.read().lower()
                        is_samsung = 'samsung' in cpuinfo or 'exynos' in cpuinfo
                    
                    if not is_samsung:
                        return []
                except Exception:
                    return []
            elif platform.system() != 'Android':
                return []
            
            # Check if ONE library is available (Samsung NPU)
            try:
                import one
                
                # Get ONE version
                one_version = getattr(one, 'version', 'Unknown')
                
                # Create capability object
                one_capability = HardwareCapability(
                    hardware_type=HardwareType.NPU,
                    vendor=HardwareVendor.SAMSUNG,
                    model="Samsung Neural Processing Unit",
                    version=one_version,
                    supported_precisions=[
                        PrecisionType.FP32,
                        PrecisionType.FP16,
                        PrecisionType.INT8
                    ],
                    capabilities={
                        'sdk_version': one_version,
                        'backend': 'ONE'
                    }
                )
                
                return [one_capability]
            except ImportError:
                # No Samsung NPU libraries found
                return []
        
        except Exception as e:
            logger.error(f"Error detecting Samsung NPUs: {str(e)}")
            return []
    
    def _detect_mediatek_npus(self) -> List[HardwareCapability]:
        """Detect Mediatek NPUs (APU)."""
        try:
            # First check if this is a Mediatek platform
            if platform.system() == 'Linux':
                try:
                    with open('/proc/cpuinfo', 'r') as f:
                        cpuinfo = f.read().lower()
                        is_mediatek = 'mediatek' in cpuinfo or 'mt' in cpuinfo
                    
                    if not is_mediatek:
                        return []
                except Exception:
                    return []
            elif platform.system() != 'Android':
                return []
            
            # Check if NNAPIE library is available (Mediatek NPU)
            try:
                import nnapie
                
                # Get NNAPIE version
                nnapie_version = getattr(nnapie, 'version', 'Unknown')
                
                # Create capability object
                nnapie_capability = HardwareCapability(
                    hardware_type=HardwareType.NPU,
                    vendor=HardwareVendor.MEDIATEK,
                    model="Mediatek APU",
                    version=nnapie_version,
                    supported_precisions=[
                        PrecisionType.FP32,
                        PrecisionType.FP16,
                        PrecisionType.INT8
                    ],
                    capabilities={
                        'sdk_version': nnapie_version,
                        'backend': 'NNAPIE'
                    }
                )
                
                return [nnapie_capability]
            except ImportError:
                # No Mediatek NPU libraries found
                return []
        
        except Exception as e:
            logger.error(f"Error detecting Mediatek NPUs: {str(e)}")
            return []
    
    def _detect_apple_neural_engine(self) -> List[HardwareCapability]:
        """Detect Apple Neural Engine (ANE)."""
        try:
            # Only available on macOS with Apple Silicon
            if platform.system() != 'Darwin':
                return []
            
            # Check if this is Apple Silicon
            result = subprocess.run(['sysctl', 'hw.optional.arm64'], 
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE)
            
            if result.returncode != 0 or b'1' not in result.stdout:
                return []
            
            # Check if Core ML is available
            try:
                import coremltools as ct
                
                # Create capability object
                ane_capability = HardwareCapability(
                    hardware_type=HardwareType.NPU,
                    vendor=HardwareVendor.APPLE,
                    model="Apple Neural Engine",
                    version=ct.__version__,
                    supported_precisions=[
                        PrecisionType.FP32,
                        PrecisionType.FP16,
                        PrecisionType.INT8
                    ],
                    capabilities={
                        'sdk_version': ct.__version__,
                        'backend': 'CoreML'
                    }
                )
                
                return [ane_capability]
            except ImportError:
                # Apple Neural Engine detected but no Core ML library
                # Still report it as available
                ane_capability = HardwareCapability(
                    hardware_type=HardwareType.NPU,
                    vendor=HardwareVendor.APPLE,
                    model="Apple Neural Engine",
                    supported_precisions=[
                        PrecisionType.FP32,
                        PrecisionType.FP16,
                        PrecisionType.INT8
                    ],
                    capabilities={
                        'backend': 'Unknown'
                    }
                )
                
                return [ane_capability]
        
        except Exception as e:
            logger.error(f"Error detecting Apple Neural Engine: {str(e)}")
            return []
    
    def detect_all_capabilities(self) -> WorkerHardwareCapabilities:
        """
        Detect all hardware capabilities on the current system.
        
        Returns:
            WorkerHardwareCapabilities with all detected hardware
        """
        import psutil
        import time
        
        # Get CPU capabilities
        cpu_capability = self.detect_cpu_capabilities()
        
        # Get GPU capabilities
        gpu_capabilities = self.detect_gpu_capabilities()
        
        # Get TPU capabilities
        tpu_capabilities = self.detect_tpu_capabilities()
        
        # Get NPU capabilities
        npu_capabilities = self.detect_npu_capabilities()
        
        # Get WebGPU capabilities (if applicable)
        webgpu_capability = self.detect_webgpu_capabilities()
        
        # Get WebNN capabilities (if applicable)
        webnn_capability = self.detect_webnn_capabilities()
        
        # Combine all capabilities
        all_capabilities = [cpu_capability] + gpu_capabilities + tpu_capabilities + npu_capabilities
        
        if webgpu_capability:
            all_capabilities.append(webgpu_capability)
        
        if webnn_capability:
            all_capabilities.append(webnn_capability)
        
        # Create worker hardware capabilities
        worker_capabilities = WorkerHardwareCapabilities(
            worker_id=self.worker_id,
            os_type=self.os_info[0],
            os_version=self.os_info[1],
            hostname=self.hostname,
            cpu_count=psutil.cpu_count(logical=False),
            total_memory_gb=psutil.virtual_memory().total / (1024**3),
            hardware_capabilities=all_capabilities,
            last_updated=time.time()
        )
        
        logger.info(f"Detected {len(all_capabilities)} hardware capabilities for worker {self.worker_id}")
        
        return worker_capabilities


class HardwareCapabilityComparator:
    """
    Compare hardware capabilities for compatibility and performance estimation.
    
    This class provides methods to compare hardware capabilities, determine
    compatibility, and estimate performance for different workloads.
    """
    
    def __init__(self):
        """Initialize the hardware capability comparator."""
        logger.info("Hardware Capability Comparator initialized")
    
    def compare_capabilities(self, capability1: HardwareCapability, 
                            capability2: HardwareCapability) -> Dict[str, Any]:
        """
        Compare two hardware capabilities.
        
        Args:
            capability1: First hardware capability
            capability2: Second hardware capability
            
        Returns:
            Dictionary with comparison results
        """
        # Compare basic properties
        same_type = capability1.hardware_type == capability2.hardware_type
        same_vendor = capability1.vendor == capability2.vendor
        
        # Compare scores
        score_comparisons = {}
        for score_type in set(capability1.scores.keys()) | set(capability2.scores.keys()):
            score1 = capability1.scores.get(score_type, CapabilityScore.UNKNOWN)
            score2 = capability2.scores.get(score_type, CapabilityScore.UNKNOWN)
            
            if score1 != CapabilityScore.UNKNOWN and score2 != CapabilityScore.UNKNOWN:
                score_comparisons[score_type] = {
                    'first': score1.name,
                    'second': score2.name,
                    'difference': score1.value - score2.value
                }
        
        # Compare memory
        memory_ratio = None
        if capability1.memory_gb and capability2.memory_gb and capability2.memory_gb > 0:
            memory_ratio = capability1.memory_gb / capability2.memory_gb
        
        # Compare precision support
        precision_comparison = {
            'first_only': [p.name for p in capability1.supported_precisions 
                         if p not in capability2.supported_precisions],
            'second_only': [p.name for p in capability2.supported_precisions 
                          if p not in capability1.supported_precisions],
            'common': [p.name for p in capability1.supported_precisions 
                      if p in capability2.supported_precisions]
        }
        
        # Overall comparison result
        return {
            'same_type': same_type,
            'same_vendor': same_vendor,
            'score_comparisons': score_comparisons,
            'memory_ratio': memory_ratio,
            'precision_comparison': precision_comparison
        }
    
    def is_compatible(self, capability: HardwareCapability, 
                     requirements: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if a hardware capability is compatible with requirements.
        
        Args:
            capability: Hardware capability to check
            requirements: Dictionary with hardware requirements
            
        Returns:
            Tuple of (is_compatible, compatibility_details)
        """
        compatibility_details = {}
        
        # Check hardware type
        if 'hardware_type' in requirements:
            req_type = requirements['hardware_type']
            if isinstance(req_type, str):
                req_type = HardwareType(req_type)
            
            type_compatible = capability.hardware_type == req_type
            compatibility_details['hardware_type'] = {
                'required': req_type.name,
                'actual': capability.hardware_type.name,
                'compatible': type_compatible
            }
            
            if not type_compatible:
                return False, compatibility_details
        
        # Check memory requirements
        if 'min_memory_gb' in requirements:
            min_memory = requirements['min_memory_gb']
            actual_memory = capability.memory_gb or 0
            
            memory_compatible = actual_memory >= min_memory
            compatibility_details['memory'] = {
                'required': min_memory,
                'actual': actual_memory,
                'compatible': memory_compatible
            }
            
            if not memory_compatible:
                return False, compatibility_details
        
        # Check precision requirements
        if 'required_precisions' in requirements:
            req_precisions = requirements['required_precisions']
            if isinstance(req_precisions[0], str):
                req_precisions = [PrecisionType(p) for p in req_precisions]
            
            actual_precisions = capability.supported_precisions
            
            missing_precisions = [p for p in req_precisions if p not in actual_precisions]
            precision_compatible = len(missing_precisions) == 0
            
            compatibility_details['precision'] = {
                'required': [p.name for p in req_precisions],
                'actual': [p.name for p in actual_precisions],
                'missing': [p.name for p in missing_precisions],
                'compatible': precision_compatible
            }
            
            if not precision_compatible:
                return False, compatibility_details
        
        # Check vendor requirements (if specified)
        if 'vendor' in requirements:
            req_vendor = requirements['vendor']
            if isinstance(req_vendor, str):
                req_vendor = HardwareVendor(req_vendor)
            
            vendor_compatible = capability.vendor == req_vendor
            compatibility_details['vendor'] = {
                'required': req_vendor.name,
                'actual': capability.vendor.name,
                'compatible': vendor_compatible
            }
            
            if not vendor_compatible:
                return False, compatibility_details
        
        # Check minimum score requirements
        if 'min_scores' in requirements:
            min_scores = requirements['min_scores']
            
            for score_type, min_score in min_scores.items():
                if isinstance(min_score, str):
                    min_score = CapabilityScore[min_score]
                
                actual_score = capability.scores.get(score_type, CapabilityScore.UNKNOWN)
                if actual_score == CapabilityScore.UNKNOWN:
                    score_compatible = False
                else:
                    score_compatible = actual_score.value >= min_score.value
                
                compatibility_details[f'score_{score_type}'] = {
                    'required': min_score.name,
                    'actual': actual_score.name,
                    'compatible': score_compatible
                }
                
                if not score_compatible:
                    return False, compatibility_details
        
        # Check custom capability requirements
        if 'required_capabilities' in requirements:
            for cap_name, cap_value in requirements['required_capabilities'].items():
                actual_value = capability.capabilities.get(cap_name)
                
                if isinstance(cap_value, bool):
                    cap_compatible = actual_value == cap_value
                elif isinstance(cap_value, (int, float)):
                    if actual_value is None:
                        cap_compatible = False
                    else:
                        cap_compatible = actual_value >= cap_value
                else:
                    cap_compatible = actual_value == cap_value
                
                compatibility_details[f'capability_{cap_name}'] = {
                    'required': cap_value,
                    'actual': actual_value,
                    'compatible': cap_compatible
                }
                
                if not cap_compatible:
                    return False, compatibility_details
        
        # All requirements are compatible
        return True, compatibility_details
    
    def estimate_performance(self, capability: HardwareCapability, 
                            workload_type: str,
                            workload_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Estimate performance for a hardware capability with a specific workload.
        
        Args:
            capability: Hardware capability to estimate performance for
            workload_type: Type of workload (e.g., "inference", "training")
            workload_params: Dictionary with workload parameters
            
        Returns:
            Dictionary with performance estimates
        """
        workload_params = workload_params or {}
        
        # Base estimates
        estimates = {
            'relative_score': 1.0,
            'confidence': 0.5,
            'recommendation_level': 'unknown'
        }
        
        # Adjust based on hardware type and workload type
        if workload_type == 'inference':
            if capability.hardware_type == HardwareType.GPU:
                estimates['relative_score'] = 4.0
            elif capability.hardware_type == HardwareType.TPU:
                estimates['relative_score'] = 5.0
            elif capability.hardware_type == HardwareType.NPU:
                estimates['relative_score'] = 4.5
            else:
                estimates['relative_score'] = 2.0
        elif workload_type == 'training':
            if capability.hardware_type == HardwareType.GPU:
                estimates['relative_score'] = 4.5
            elif capability.hardware_type == HardwareType.TPU:
                estimates['relative_score'] = 4.0
            else:
                estimates['relative_score'] = 1.5
        
        # Adjust based on overall score
        overall_score = capability.scores.get('overall', CapabilityScore.UNKNOWN)
        if overall_score != CapabilityScore.UNKNOWN:
            score_adjustment = (overall_score.value - 3) * 0.5
            estimates['relative_score'] += score_adjustment
            estimates['confidence'] += 0.1
        
        # Adjust based on precision requirements
        if 'precision' in workload_params:
            required_precision = workload_params['precision']
            if isinstance(required_precision, str):
                required_precision = PrecisionType(required_precision)
            
            precision_compatible = required_precision in capability.supported_precisions
            
            if precision_compatible:
                estimates['confidence'] += 0.2
            else:
                estimates['relative_score'] *= 0.5
                estimates['confidence'] -= 0.2
        
        # Adjust based on memory requirements
        if 'memory_gb' in workload_params and capability.memory_gb:
            required_memory = workload_params['memory_gb']
            
            if capability.memory_gb >= required_memory * 2:
                estimates['relative_score'] += 1.0
                estimates['confidence'] += 0.1
            elif capability.memory_gb >= required_memory:
                estimates['confidence'] += 0.1
            elif capability.memory_gb < required_memory:
                memory_ratio = capability.memory_gb / required_memory
                estimates['relative_score'] *= memory_ratio
                estimates['confidence'] -= 0.2
        
        # Adjust based on batch size
        if 'batch_size' in workload_params:
            batch_size = workload_params['batch_size']
            
            if capability.hardware_type == HardwareType.GPU and batch_size > 1:
                # GPUs generally benefit from larger batch sizes
                batch_factor = min(batch_size / 4, 2.0)
                estimates['relative_score'] *= 1.0 + batch_factor * 0.2
            elif capability.hardware_type == HardwareType.TPU and batch_size > 1:
                # TPUs generally benefit significantly from larger batch sizes
                batch_factor = min(batch_size / 8, 3.0)
                estimates['relative_score'] *= 1.0 + batch_factor * 0.3
        
        # Clamp and normalize values
        estimates['relative_score'] = max(0.1, min(10.0, estimates['relative_score']))
        estimates['confidence'] = max(0.1, min(1.0, estimates['confidence']))
        
        # Set recommendation level based on relative score
        if estimates['relative_score'] >= 7.0:
            estimates['recommendation_level'] = 'excellent'
        elif estimates['relative_score'] >= 5.0:
            estimates['recommendation_level'] = 'good'
        elif estimates['relative_score'] >= 3.0:
            estimates['recommendation_level'] = 'average'
        elif estimates['relative_score'] >= 1.0:
            estimates['recommendation_level'] = 'minimal'
        else:
            estimates['recommendation_level'] = 'not_recommended'
        
        return estimates
    
    def find_best_hardware(self, capabilities: List[HardwareCapability], 
                          workload_type: str,
                          workload_params: Dict[str, Any] = None) -> Tuple[HardwareCapability, Dict[str, Any]]:
        """
        Find the best hardware for a specific workload from a list of capabilities.
        
        Args:
            capabilities: List of hardware capabilities to choose from
            workload_type: Type of workload (e.g., "inference", "training")
            workload_params: Dictionary with workload parameters
            
        Returns:
            Tuple of (best_capability, performance_estimate)
        """
        if not capabilities:
            raise ValueError("No hardware capabilities provided")
        
        best_capability = None
        best_estimate = None
        
        for capability in capabilities:
            estimate = self.estimate_performance(capability, workload_type, workload_params)
            
            if not best_estimate or estimate['relative_score'] > best_estimate['relative_score']:
                best_capability = capability
                best_estimate = estimate
        
        return best_capability, best_estimate


# Example usage
if __name__ == "__main__":
    # Create a hardware capability detector
    detector = HardwareCapabilityDetector()
    
    # Detect all capabilities
    worker_capabilities = detector.detect_all_capabilities()
    
    # Print information about each capability
    print(f"\nWorker ID: {worker_capabilities.worker_id}")
    print(f"OS: {worker_capabilities.os_type} {worker_capabilities.os_version}")
    print(f"Hostname: {worker_capabilities.hostname}")
    print(f"CPU Count: {worker_capabilities.cpu_count}")
    print(f"Total Memory: {worker_capabilities.total_memory_gb:.2f} GB")
    print(f"Detected {len(worker_capabilities.hardware_capabilities)} hardware capabilities")
    
    for idx, capability in enumerate(worker_capabilities.hardware_capabilities):
        print(f"\nCapability {idx+1}: {capability.hardware_type.name} - {capability.model}")
        print(f"  Vendor: {capability.vendor.name}")
        print(f"  Memory: {capability.memory_gb:.2f} GB" if capability.memory_gb else "  Memory: Unknown")
        print(f"  Supported Precisions: {', '.join(p.name for p in capability.supported_precisions)}")
        
        if capability.scores:
            print("  Scores:")
            for score_type, score in capability.scores.items():
                print(f"    {score_type}: {score.name}")
    
    # Create a hardware capability comparator
    comparator = HardwareCapabilityComparator()
    
    # Check if we have at least one capability
    if worker_capabilities.hardware_capabilities:
        # Get the first capability
        capability = worker_capabilities.hardware_capabilities[0]
        
        # Check if it's compatible with some requirements
        requirements = {
            'hardware_type': capability.hardware_type,
            'min_memory_gb': 1.0,
            'required_precisions': [PrecisionType.FP32]
        }
        
        is_compatible, details = comparator.is_compatible(capability, requirements)
        print(f"\nCompatible with requirements: {is_compatible}")
        
        # Estimate performance
        estimate = comparator.estimate_performance(capability, "inference", {
            'precision': PrecisionType.FP32,
            'batch_size': 4
        })
        
        print(f"\nPerformance Estimate:")
        print(f"  Relative Score: {estimate['relative_score']:.2f}")
        print(f"  Confidence: {estimate['confidence']:.2f}")
        print(f"  Recommendation: {estimate['recommendation_level']}")