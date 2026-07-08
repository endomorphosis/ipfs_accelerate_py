"""
Enhanced Hardware Detector for Distributed Testing Framework

This module extends the basic hardware detection capabilities to provide
more comprehensive hardware profiling, classification, and specialization
for heterogeneous computing environments.

It integrates with the hardware taxonomy system to create detailed hardware
profiles that can be used by the load balancer for more intelligent
workload distribution.
"""

import os
import platform
import sys
import json
import logging
import subprocess
import socket
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from dataclasses import dataclass, field
import re
import threading

try:
    import psutil
except ImportError:
    psutil = None

try:
    import torch
except ImportError:
    torch = None

try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    import GPUtil
except ImportError:
    GPUtil = None

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.edge.options import Options as EdgeOptions
    from selenium.webdriver.firefox.options import Options as FirefoxOptions
    from selenium.common.exceptions import WebDriverException
except ImportError:
    webdriver = None

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
    HardwareTaxonomy,
    create_cpu_profile,
    create_gpu_profile,
    create_npu_profile,
    create_browser_profile
)

# Configure logging
logger = logging.getLogger(__name__)


class EnhancedHardwareDetector:
    """
    Enhanced hardware detector that provides detailed hardware profiles
    for heterogeneous computing environments.
    """
    
    def __init__(self):
        """Initialize the enhanced hardware detector."""
        self.taxonomy = HardwareTaxonomy()
        self.worker_id = socket.gethostname()
        self._detection_lock = threading.Lock()
        self._detected = False
        self._hardware_profiles = []
        
        # CPU detection results
        self._cpu_info = {}
        
        # GPU detection results
        self._gpu_info = []
        
        # Memory detection results
        self._memory_info = {}
        
        # Platform detection results
        self._platform_info = {}
        
        # Browser detection results
        self._browser_info = {}
        
        # Specialized hardware detection results
        self._specialized_hardware = {}
        
        # Detection flags to avoid redundant detection
        self._cpu_detected = False
        self._gpu_detected = False
        self._memory_detected = False
        self._platform_detected = False
        self._browser_detected = False
        self._specialized_hardware_detected = False
    
    def detect_hardware(self, force_detect: bool = False) -> List[HardwareCapabilityProfile]:
        """
        Detect all hardware capabilities and create hardware profiles.
        
        Args:
            force_detect: Force re-detection even if already detected
            
        Returns:
            List of hardware capability profiles
        """
        with self._detection_lock:
            if self._detected and not force_detect:
                return self._hardware_profiles
            
            # Reset detection flags if forcing detection
            if force_detect:
                self._cpu_detected = False
                self._gpu_detected = False
                self._memory_detected = False
                self._platform_detected = False
                self._browser_detected = False
                self._specialized_hardware_detected = False
            
            # Perform detection
            self._detect_cpu()
            self._detect_memory()
            self._detect_gpu()
            self._detect_platform()
            self._detect_browsers()
            self._detect_specialized_hardware()
            
            # Create hardware profiles
            self._hardware_profiles = []
            
            # Add CPU profile
            if self._cpu_info:
                try:
                    cpu_profile = self._create_cpu_profile()
                    self._hardware_profiles.append(cpu_profile)
                except Exception as e:
                    logger.error(f"Error creating CPU profile: {e}")
            
            # Add GPU profiles
            for gpu_info in self._gpu_info:
                try:
                    gpu_profile = self._create_gpu_profile(gpu_info)
                    self._hardware_profiles.append(gpu_profile)
                except Exception as e:
                    logger.error(f"Error creating GPU profile: {e}")
            
            # Add specialized hardware profiles
            for hw_type, hw_info in self._specialized_hardware.items():
                try:
                    if hw_type == "npu":
                        for npu_info in hw_info:
                            npu_profile = self._create_npu_profile(npu_info)
                            self._hardware_profiles.append(npu_profile)
                    # Add more specialized hardware types as needed
                except Exception as e:
                    logger.error(f"Error creating {hw_type} profile: {e}")
            
            # Add browser profiles
            for browser_name, browser_info in self._browser_info.items():
                try:
                    if browser_info.get("available", False):
                        # Find matching GPU profile if available
                        gpu_profile = None
                        if self._gpu_info:
                            gpu_profile = self._hardware_profiles[1] if len(self._hardware_profiles) > 1 else None
                        
                        browser_profile = create_browser_profile(
                            browser_name=browser_name,
                            supports_webgpu=browser_info.get("webgpu", False),
                            supports_webnn=browser_info.get("webnn", False),
                            gpu_profile=gpu_profile
                        )
                        self._hardware_profiles.append(browser_profile)
                except Exception as e:
                    logger.error(f"Error creating browser profile for {browser_name}: {e}")
            
            # Register profiles with taxonomy
            self.taxonomy.register_worker_hardware(self.worker_id, self._hardware_profiles)
            self.taxonomy.update_specialization_map()
            
            self._detected = True
            return self._hardware_profiles
    
    def _detect_cpu(self) -> Dict[str, Any]:
        """
        Detect CPU information including cores, features, and architecture.
        """
        if self._cpu_detected:
            return self._cpu_info
        
        cpu_info = {
            "cores_physical": 1,
            "cores_logical": 1,
            "architecture": platform.machine(),
            "brand": "Unknown",
            "features": [],
            "has_avx": False,
            "has_avx2": False,
            "has_avx512": False,
            "frequency_mhz": 0,
            "vendor": "unknown"
        }
        
        try:
            # Use psutil if available
            if psutil:
                cpu_info["cores_physical"] = psutil.cpu_count(logical=False) or 1
                cpu_info["cores_logical"] = psutil.cpu_count(logical=True) or 1
                
                # Get CPU frequency
                freq_info = psutil.cpu_freq()
                if freq_info:
                    cpu_info["frequency_mhz"] = int(freq_info.current)
            else:
                # Fallback to os.cpu_count
                cpu_info["cores_logical"] = os.cpu_count() or 1
                cpu_info["cores_physical"] = cpu_info["cores_logical"]
            
            # Try to get CPU brand string
            if platform.system() == "Linux":
                try:
                    with open("/proc/cpuinfo", "r") as f:
                        for line in f:
                            if "model name" in line:
                                cpu_info["brand"] = line.split(":", 1)[1].strip()
                                break
                except Exception:
                    pass
            elif platform.system() == "Darwin":  # macOS
                try:
                    brand = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
                    cpu_info["brand"] = brand
                except Exception:
                    pass
            elif platform.system() == "Windows":
                try:
                    brand = subprocess.check_output(["wmic", "cpu", "get", "name"]).decode().strip()
                    if "Name" in brand:
                        cpu_info["brand"] = brand.split("\n")[1].strip()
                except Exception:
                    pass
            
            # Determine vendor
            brand_lower = cpu_info["brand"].lower()
            if "intel" in brand_lower:
                cpu_info["vendor"] = "intel"
            elif "amd" in brand_lower:
                cpu_info["vendor"] = "amd"
            elif "apple" in brand_lower or "m1" in brand_lower or "m2" in brand_lower:
                cpu_info["vendor"] = "apple"
            elif "arm" in brand_lower or "snapdragon" in brand_lower:
                cpu_info["vendor"] = "arm"
            elif "ibm" in brand_lower or "power" in brand_lower:
                cpu_info["vendor"] = "ibm"
            
            # Detect CPU features
            features = []
            
            # Check for AVX support
            if platform.system() == "Linux":
                try:
                    with open("/proc/cpuinfo", "r") as f:
                        for line in f:
                            if "flags" in line:
                                features = line.split(":", 1)[1].strip().split()
                                break
                except Exception:
                    pass
            elif platform.system() == "Darwin":  # macOS
                try:
                    feature_output = subprocess.check_output(["sysctl", "-n", "machdep.cpu.features"]).decode().strip()
                    features = feature_output.split()
                except Exception:
                    pass
            
            # Check for specific AVX features
            cpu_info["has_avx"] = "avx" in [f.lower() for f in features]
            cpu_info["has_avx2"] = "avx2" in [f.lower() for f in features]
            cpu_info["has_avx512f"] = any(f.lower().startswith("avx512") for f in features)
            cpu_info["features"] = features
            
        except Exception as e:
            logger.error(f"Error detecting CPU: {e}")
        
        self._cpu_info = cpu_info
        self._cpu_detected = True
        return cpu_info
    
    def _detect_memory(self) -> Dict[str, Any]:
        """
        Detect memory information including total and available memory.
        """
        if self._memory_detected:
            return self._memory_info
        
        memory_info = {
            "total_bytes": 0,
            "available_bytes": 0,
            "memory_type": "unknown",
            "is_shared": False,
            "hierarchy_levels": 3,
            "has_unified_memory": False
        }
        
        try:
            # Use psutil if available
            if psutil:
                mem = psutil.virtual_memory()
                memory_info["total_bytes"] = mem.total
                memory_info["available_bytes"] = mem.available
            else:
                # Fallback to a reasonable default
                memory_info["total_bytes"] = 8 * 1024 * 1024 * 1024  # 8 GB
                memory_info["available_bytes"] = 4 * 1024 * 1024 * 1024  # 4 GB
            
            # Try to detect memory type (this is platform-specific and may not always work)
            if platform.system() == "Linux":
                try:
                    with open("/proc/meminfo", "r") as f:
                        for line in f:
                            if "MemTotal" in line:
                                memory_info["total_bytes"] = int(line.split()[1]) * 1024
                            elif "MemAvailable" in line:
                                memory_info["available_bytes"] = int(line.split()[1]) * 1024
                except Exception:
                    pass
                
                # Try to detect memory type using dmidecode (requires root)
                try:
                    dmi_output = subprocess.check_output(["sudo", "dmidecode", "-t", "memory"]).decode()
                    if "DDR4" in dmi_output:
                        memory_info["memory_type"] = "DDR4"
                    elif "DDR3" in dmi_output:
                        memory_info["memory_type"] = "DDR3"
                    elif "DDR5" in dmi_output:
                        memory_info["memory_type"] = "DDR5"
                    elif "LPDDR4" in dmi_output:
                        memory_info["memory_type"] = "LPDDR4"
                    elif "LPDDR5" in dmi_output:
                        memory_info["memory_type"] = "LPDDR5"
                except Exception:
                    # Default to a reasonable guess based on CPU architecture and year
                    memory_info["memory_type"] = "DDR4"
                    
            elif platform.system() == "Darwin":  # macOS
                # Apple Silicon has unified memory
                if "Apple" in platform.processor():
                    memory_info["has_unified_memory"] = True
                    memory_info["memory_type"] = "LPDDR4"  # or LPDDR5 for newer models
            
        except Exception as e:
            logger.error(f"Error detecting memory: {e}")
        
        self._memory_info = memory_info
        self._memory_detected = True
        return memory_info
    
    def _detect_gpu(self) -> List[Dict[str, Any]]:
        """
        Detect GPU information including CUDA, ROCm, and MPS capabilities.
        """
        if self._gpu_detected:
            return self._gpu_info
        
        gpu_info = []
        
        try:
            # Check for CUDA GPUs using PyTorch
            if torch and hasattr(torch, "cuda") and torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    cuda_gpu = {
                        "type": "cuda",
                        "name": props.name,
                        "compute_capability": f"{props.major}.{props.minor}",
                        "compute_units": props.multi_processor_count,
                        "memory_total": props.total_memory,
                        "memory_available": props.total_memory,  # Approximation
                        "clock_rate_mhz": props.clock_rate / 1000,
                        "vendor": "nvidia",
                        "has_tensor_cores": props.major >= 7,  # Volta+ has tensor cores
                        "has_ray_tracing": False,  # Only in specific RTX GPUs
                        "memory_bandwidth_gbps": None,  # Not directly available
                        "tdp_w": None  # Not directly available
                    }
                    gpu_info.append(cuda_gpu)
            
            # If no CUDA GPUs found, try using GPUtil
            if not gpu_info and GPUtil:
                try:
                    for gpu in GPUtil.getGPUs():
                        gpu_info.append({
                            "type": "cuda",
                            "name": gpu.name,
                            "compute_capability": None,  # Not available from GPUtil
                            "compute_units": None,  # Not available from GPUtil
                            "memory_total": gpu.memoryTotal * 1024 * 1024,  # Convert from MB to bytes
                            "memory_available": gpu.memoryFree * 1024 * 1024,  # Convert from MB to bytes
                            "clock_rate_mhz": None,  # Not available from GPUtil
                            "vendor": "nvidia",
                            "has_tensor_cores": "RTX" in gpu.name or "A100" in gpu.name or "H100" in gpu.name,
                            "has_ray_tracing": "RTX" in gpu.name,
                            "memory_bandwidth_gbps": None,
                            "tdp_w": None
                        })
                except Exception as e:
                    logger.warning(f"Error using GPUtil: {e}")
            
            # Check for ROCm GPUs using command-line tools
            if platform.system() == "Linux":
                try:
                    rocm_path = "/opt/rocm/bin/rocm-smi"
                    if os.path.exists(rocm_path):
                        rocm_output = subprocess.check_output([rocm_path, "--showproductname", "--showmeminfo"]).decode()
                        for line in rocm_output.split("\n"):
                            if "GPU" in line and ":" in line:
                                # Extract GPU name
                                gpu_name = line.split(":", 1)[1].strip()
                                
                                # AMD GPUs typically have compute units
                                compute_units = 64  # Default estimate
                                
                                # Create AMD GPU entry
                                rocm_gpu = {
                                    "type": "rocm",
                                    "name": gpu_name,
                                    "compute_capability": None,
                                    "compute_units": compute_units,
                                    "memory_total": 8 * 1024 * 1024 * 1024,  # Default 8GB
                                    "memory_available": 8 * 1024 * 1024 * 1024,  # Default 8GB
                                    "clock_rate_mhz": 1500,  # Default estimate
                                    "vendor": "amd",
                                    "has_tensor_cores": False,
                                    "has_ray_tracing": "RX 6000" in gpu_name or "RX 7000" in gpu_name,
                                    "memory_bandwidth_gbps": None,
                                    "tdp_w": None
                                }
                                gpu_info.append(rocm_gpu)
                except Exception as e:
                    logger.warning(f"Error detecting ROCm GPUs: {e}")
            
            # Check for Apple MPS (Metal Performance Shaders)
            if platform.system() == "Darwin" and hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                # Get processor info
                try:
                    processor_info = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
                    is_apple_silicon = "Apple" in processor_info
                    
                    # For Apple Silicon, extract the model (M1, M2, etc.)
                    model_match = re.search(r'(M\d+)', processor_info)
                    model = model_match.group(1) if model_match else "M1"
                    
                    # Estimate compute units based on the model
                    compute_units = {
                        "M1": 8,
                        "M2": 10,
                        "M1 Pro": 16,
                        "M1 Max": 32,
                        "M1 Ultra": 64,
                        "M2 Pro": 19,
                        "M2 Max": 38,
                        "M2 Ultra": 76
                    }.get(model, 8)
                    
                    # Get total memory
                    memory_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip())
                    
                    mps_gpu = {
                        "type": "mps",
                        "name": f"Apple {model} GPU",
                        "compute_capability": None,
                        "compute_units": compute_units,
                        "memory_total": memory_bytes,  # Unified memory
                        "memory_available": memory_bytes // 2,  # Rough estimate
                        "clock_rate_mhz": 1278,  # Default for M1
                        "vendor": "apple",
                        "has_tensor_cores": True,  # Apple Neural Engine
                        "has_ray_tracing": False,
                        "memory_bandwidth_gbps": 200.0 if "M1" in model else 300.0,  # Estimates
                        "tdp_w": 15.0  # Estimate
                    }
                    gpu_info.append(mps_gpu)
                except Exception as e:
                    logger.warning(f"Error detecting Apple MPS: {e}")
            
            # Try detecting NVIDIA GPUs using nvidia-smi if other methods failed
            if not gpu_info and platform.system() in ["Linux", "Windows"]:
                try:
                    nvidia_smi_output = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,clocks.sm", "--format=csv,noheader"]).decode()
                    for line in nvidia_smi_output.split("\n"):
                        if line.strip():
                            parts = [part.strip() for part in line.split(",")]
                            if len(parts) >= 3:
                                name = parts[0]
                                memory_total = int(parts[1].split()[0]) * 1024 * 1024  # Convert from MiB to bytes
                                memory_free = int(parts[2].split()[0]) * 1024 * 1024  # Convert from MiB to bytes
                                clock_rate = int(parts[3].split()[0]) if len(parts) > 3 else 1000
                                
                                # Check for tensor cores based on architecture
                                has_tensor_cores = (
                                    "RTX" in name or 
                                    "A100" in name or 
                                    "H100" in name or 
                                    "Titan V" in name or
                                    "V100" in name or
                                    any(arch in name for arch in ["Volta", "Turing", "Ampere", "Ada", "Hopper"])
                                )
                                
                                has_ray_tracing = "RTX" in name or "Ada" in name
                                
                                gpu_info.append({
                                    "type": "cuda",
                                    "name": name,
                                    "compute_capability": None,
                                    "compute_units": None,
                                    "memory_total": memory_total,
                                    "memory_available": memory_free,
                                    "clock_rate_mhz": clock_rate,
                                    "vendor": "nvidia",
                                    "has_tensor_cores": has_tensor_cores,
                                    "has_ray_tracing": has_ray_tracing,
                                    "memory_bandwidth_gbps": None,
                                    "tdp_w": None
                                })
                except Exception as e:
                    logger.warning(f"Error using nvidia-smi: {e}")
            
        except Exception as e:
            logger.error(f"Error detecting GPUs: {e}")
        
        self._gpu_info = gpu_info
        self._gpu_detected = True
        return gpu_info
    
    def _detect_platform(self) -> Dict[str, Any]:
        """
        Detect platform information including OS, Python version, and architecture.
        """
        if self._platform_detected:
            return self._platform_info
        
        platform_info = {
            "os": platform.system(),
            "os_version": platform.release(),
            "os_name": platform.platform(),
            "python_version": platform.python_version(),
            "architecture": platform.machine(),
            "hostname": platform.node(),
            "cpu_architecture": platform.processor() or platform.machine(),
            "distribution": None
        }
        
        # Try to get Linux distribution information
        if platform.system() == "Linux":
            try:
                # Try using lsb_release
                distro = subprocess.check_output(["lsb_release", "-a"]).decode()
                for line in distro.split("\n"):
                    if "Description:" in line:
                        platform_info["distribution"] = line.split(":", 1)[1].strip()
                        break
            except Exception:
                # Fallback to reading os-release
                try:
                    with open("/etc/os-release") as f:
                        for line in f:
                            if line.startswith("PRETTY_NAME="):
                                platform_info["distribution"] = line.split("=", 1)[1].strip().strip('"')
                                break
                except Exception:
                    pass
        
        self._platform_info = platform_info
        self._platform_detected = True
        return platform_info
    
    def _detect_browsers(self) -> Dict[str, Dict[str, Any]]:
        """
        Detect available browsers and their WebGPU/WebNN support.
        """
        if self._browser_detected:
            return self._browser_info
        
        browser_info = {
            "chrome": {"available": False, "webgpu": False, "webnn": False, "version": None},
            "edge": {"available": False, "webgpu": False, "webnn": False, "version": None},
            "firefox": {"available": False, "webgpu": False, "webnn": False, "version": None},
            "safari": {"available": False, "webgpu": False, "webnn": False, "version": None}
        }
        
        if not webdriver:
            logger.warning("Selenium webdriver not available for browser detection")
            self._browser_info = browser_info
            self._browser_detected = True
            return browser_info
        
        # Check for Chrome
        try:
            options = ChromeOptions()
            options.add_argument("--headless")
            options.add_argument("--disable-gpu")
            driver = webdriver.Chrome(options=options)
            browser_info["chrome"]["available"] = True
            
            # Get Chrome version
            version = driver.capabilities.get("browserVersion") or driver.capabilities.get("version")
            browser_info["chrome"]["version"] = version
            
            # Check for WebGPU (available in Chrome 113+)
            if version and int(version.split(".")[0]) >= 113:
                browser_info["chrome"]["webgpu"] = True
            
            # Check for WebNN (available in Chrome 113+ with flags)
            if version and int(version.split(".")[0]) >= 113:
                browser_info["chrome"]["webnn"] = True
            
            driver.quit()
        except Exception as e:
            logger.warning(f"Error detecting Chrome: {e}")
        
        # Check for Edge
        try:
            options = EdgeOptions()
            options.add_argument("--headless")
            options.add_argument("--disable-gpu")
            driver = webdriver.Edge(options=options)
            browser_info["edge"]["available"] = True
            
            # Get Edge version
            version = driver.capabilities.get("browserVersion") or driver.capabilities.get("version")
            browser_info["edge"]["version"] = version
            
            # Check for WebGPU (available in Edge 113+)
            if version and int(version.split(".")[0]) >= 113:
                browser_info["edge"]["webgpu"] = True
            
            # Check for WebNN (available in Edge 113+ with better support than Chrome)
            if version and int(version.split(".")[0]) >= 113:
                browser_info["edge"]["webnn"] = True
            
            driver.quit()
        except Exception as e:
            logger.warning(f"Error detecting Edge: {e}")
        
        # Check for Firefox
        try:
            options = FirefoxOptions()
            options.add_argument("--headless")
            driver = webdriver.Firefox(options=options)
            browser_info["firefox"]["available"] = True
            
            # Get Firefox version
            version = driver.capabilities.get("browserVersion") or driver.capabilities.get("version")
            browser_info["firefox"]["version"] = version
            
            # Check for WebGPU (available in Firefox 113+ with flags)
            if version and int(version.split(".")[0]) >= 113:
                browser_info["firefox"]["webgpu"] = True
            
            # WebNN is still experimental in Firefox
            browser_info["firefox"]["webnn"] = False
            
            driver.quit()
        except Exception as e:
            logger.warning(f"Error detecting Firefox: {e}")
        
        # Check for Safari (macOS only)
        if platform.system() == "Darwin":
            try:
                # Safari WebDriver is only available on macOS
                driver = webdriver.Safari()
                browser_info["safari"]["available"] = True
                
                # Get Safari version (format is different)
                version = driver.capabilities.get("browserVersion") or driver.capabilities.get("version")
                browser_info["safari"]["version"] = version
                
                # Check for WebGPU (available in Safari 16.4+)
                if version:
                    major_version = int(version.split(".")[0])
                    if major_version >= 17:
                        browser_info["safari"]["webgpu"] = True
                    elif major_version == 16:
                        minor_version = int(version.split(".")[1]) if len(version.split(".")) > 1 else 0
                        if minor_version >= 4:
                            browser_info["safari"]["webgpu"] = True
                
                # WebNN is not yet available in Safari
                browser_info["safari"]["webnn"] = False
                
                driver.quit()
            except Exception as e:
                logger.warning(f"Error detecting Safari: {e}")
        
        self._browser_info = browser_info
        self._browser_detected = True
        return browser_info
    
    def _detect_specialized_hardware(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Detect specialized hardware like TPUs, NPUs, FPGAs, etc.
        """
        if self._specialized_hardware_detected:
            return self._specialized_hardware
        
        specialized_hardware = {
            "tpu": [],
            "npu": [],
            "fpga": [],
            "dsp": []
        }
        
        # Check for Qualcomm NPUs
        if platform.system() == "Linux" and os.path.exists("/usr/lib/libQNNHtp.so"):
            try:
                # Try to get Qualcomm NPU information
                npu_info = {
                    "type": "npu",
                    "vendor": "qualcomm",
                    "name": "Qualcomm NPU",
                    "compute_units": 8,  # Default estimate
                    "memory_total": 512 * 1024 * 1024,  # Default 512MB estimate
                    "memory_available": 384 * 1024 * 1024,  # Default 384MB estimate
                    "clock_rate_mhz": 800,  # Default estimate
                    "has_quantization": True,
                    "tdp_w": 5.0  # Default estimate
                }
                specialized_hardware["npu"].append(npu_info)
            except Exception as e:
                logger.warning(f"Error detecting Qualcomm NPU: {e}")
        
        # Check for Google TPUs (Cloud TPUs)
        if tf and hasattr(tf, "config") and hasattr(tf.config, "list_physical_devices"):
            try:
                tpus = tf.config.list_physical_devices("TPU")
                if tpus:
                    for i, tpu in enumerate(tpus):
                        tpu_info = {
                            "type": "tpu",
                            "vendor": "google",
                            "name": f"Google TPU v{3 if 'v3' in str(tpu) else 4 if 'v4' in str(tpu) else '3'}",
                            "compute_units": 8,  # TPU v3 has 8 cores
                            "memory_total": 16 * 1024 * 1024 * 1024,  # TPU v3 has 16GB per chip
                            "memory_available": 16 * 1024 * 1024 * 1024,  # Estimate
                            "clock_rate_mhz": 1000,  # Estimate
                            "has_quantization": True,
                            "tdp_w": 200.0  # Estimate
                        }
                        specialized_hardware["tpu"].append(tpu_info)
            except Exception as e:
                logger.warning(f"Error detecting TPUs: {e}")
        
        # Check for Intel FPGAs
        if platform.system() == "Linux" and os.path.exists("/opt/intel/fpga"):
            try:
                fpga_info = {
                    "type": "fpga",
                    "vendor": "intel",
                    "name": "Intel FPGA",
                    "compute_units": 1,  # Not applicable for FPGAs in the same way
                    "memory_total": 8 * 1024 * 1024 * 1024,  # Estimate
                    "memory_available": 8 * 1024 * 1024 * 1024,  # Estimate
                    "clock_rate_mhz": 400,  # Estimate
                    "has_quantization": True,
                    "tdp_w": 75.0  # Estimate
                }
                specialized_hardware["fpga"].append(fpga_info)
            except Exception as e:
                logger.warning(f"Error detecting Intel FPGA: {e}")
        
        # Check for Qualcomm Hexagon DSP
        if platform.system() == "Linux" and (
            os.path.exists("/usr/lib/libhexagon.so") or
            os.path.exists("/vendor/lib/libhexagon_nn_skel.so") or
            os.path.exists("/system/lib/libhexagon_nn_skel.so")
        ):
            try:
                dsp_info = {
                    "type": "dsp",
                    "vendor": "qualcomm",
                    "name": "Qualcomm Hexagon DSP",
                    "compute_units": 4,  # Estimate
                    "memory_total": 256 * 1024 * 1024,  # Estimate
                    "memory_available": 256 * 1024 * 1024,  # Estimate
                    "clock_rate_mhz": 1000,  # Estimate
                    "has_quantization": True,
                    "tdp_w": 2.0  # Estimate
                }
                specialized_hardware["dsp"].append(dsp_info)
            except Exception as e:
                logger.warning(f"Error detecting Qualcomm Hexagon DSP: {e}")
        
        self._specialized_hardware = specialized_hardware
        self._specialized_hardware_detected = True
        return specialized_hardware
    
    def _create_cpu_profile(self) -> HardwareCapabilityProfile:
        """
        Create a CPU hardware capability profile.
        """
        cpu_info = self._cpu_info
        memory_info = self._memory_info
        
        vendor_map = {
            "intel": HardwareVendor.INTEL,
            "amd": HardwareVendor.AMD,
            "apple": HardwareVendor.APPLE,
            "arm": HardwareVendor.ARM,
            "ibm": HardwareVendor.IBM
        }
        
        vendor = vendor_map.get(cpu_info.get("vendor", "unknown").lower(), HardwareVendor.OTHER)
        
        return create_cpu_profile(
            model_name=cpu_info.get("brand", "Unknown CPU"),
            vendor=vendor,
            cores=cpu_info.get("cores_logical", 1),
            memory_gb=memory_info.get("total_bytes", 0) / (1024 * 1024 * 1024),
            clock_speed_mhz=cpu_info.get("frequency_mhz", 1000),
            has_avx=cpu_info.get("has_avx", False),
            has_avx2=cpu_info.get("has_avx2", False),
            has_avx512=cpu_info.get("has_avx512f", False)
        )
    
    def _create_gpu_profile(self, gpu_info: Dict[str, Any]) -> HardwareCapabilityProfile:
        """
        Create a GPU hardware capability profile.
        """
        vendor_map = {
            "nvidia": HardwareVendor.NVIDIA,
            "amd": HardwareVendor.AMD,
            "apple": HardwareVendor.APPLE
        }
        
        vendor = vendor_map.get(gpu_info.get("vendor", "unknown").lower(), HardwareVendor.OTHER)
        
        # Assume some reasonable values for missing information
        compute_units = gpu_info.get("compute_units") or 30  # Default estimate
        memory_gb = gpu_info.get("memory_total", 8 * 1024 * 1024 * 1024) / (1024 * 1024 * 1024)
        clock_speed_mhz = gpu_info.get("clock_rate_mhz") or 1500  # Default estimate
        
        return create_gpu_profile(
            model_name=gpu_info.get("name", "Unknown GPU"),
            vendor=vendor,
            compute_units=compute_units,
            memory_gb=memory_gb,
            clock_speed_mhz=clock_speed_mhz,
            has_tensor_cores=gpu_info.get("has_tensor_cores", False),
            has_ray_tracing=gpu_info.get("has_ray_tracing", False),
            compute_capability=gpu_info.get("compute_capability"),
            memory_bandwidth_gbps=gpu_info.get("memory_bandwidth_gbps"),
            tdp_w=gpu_info.get("tdp_w", 200.0)  # Default estimate
        )
    
    def _create_npu_profile(self, npu_info: Dict[str, Any]) -> HardwareCapabilityProfile:
        """
        Create an NPU hardware capability profile.
        """
        vendor_map = {
            "qualcomm": HardwareVendor.QUALCOMM,
            "mediatek": HardwareVendor.MEDIATEK,
            "samsung": HardwareVendor.SAMSUNG,
            "apple": HardwareVendor.APPLE
        }
        
        vendor = vendor_map.get(npu_info.get("vendor", "unknown").lower(), HardwareVendor.OTHER)
        
        # Assume some reasonable values for missing information
        compute_units = npu_info.get("compute_units") or 8  # Default estimate
        memory_gb = npu_info.get("memory_total", 512 * 1024 * 1024) / (1024 * 1024 * 1024)
        clock_speed_mhz = npu_info.get("clock_rate_mhz") or 800  # Default estimate
        
        return create_npu_profile(
            model_name=npu_info.get("name", "Unknown NPU"),
            vendor=vendor,
            compute_units=compute_units,
            memory_gb=memory_gb,
            clock_speed_mhz=clock_speed_mhz,
            has_quantization=npu_info.get("has_quantization", True),
            tdp_w=npu_info.get("tdp_w", 5.0)  # Default estimate
        )
    
    def get_hardware_profiles(self) -> List[HardwareCapabilityProfile]:
        """Get hardware capability profiles (detecting if needed)."""
        if not self._detected:
            self.detect_hardware()
        return self._hardware_profiles
    
    def get_taxonomy(self) -> HardwareTaxonomy:
        """Get the hardware taxonomy (detecting if needed)."""
        if not self._detected:
            self.detect_hardware()
        return self.taxonomy
    
    def find_optimal_hardware_for_workload(self, workload_type: str, min_effectiveness: float = 0.5) -> Dict:
        """
        Find the optimal hardware for a specific workload type.
        
        Args:
            workload_type: Type of workload (e.g., "nlp", "vision", "audio")
            min_effectiveness: Minimum effectiveness score (0.0 to 1.0)
            
        Returns:
            Dict with hardware information, or None if no suitable hardware found
        """
        if not self._detected:
            self.detect_hardware()
        
        best_hardware = self.taxonomy.find_best_hardware_for_workload(
            workload_type=workload_type,
            worker_ids=[self.worker_id],
            min_effectiveness=min_effectiveness
        )
        
        if not best_hardware:
            return None
        
        # Get the best match (first item)
        worker_id, profile, score = best_hardware[0]
        
        return {
            "hardware_class": profile.hardware_class.value,
            "architecture": profile.architecture.value,
            "vendor": profile.vendor.value,
            "model_name": profile.model_name,
            "effectiveness_score": score,
            "supported_backends": [backend.value for backend in profile.supported_backends],
            "supported_precisions": [precision.value for precision in profile.supported_precisions],
            "features": [feature.value for feature in profile.features],
            "memory_total_gb": profile.memory.total_bytes / (1024 * 1024 * 1024),
            "compute_units": profile.compute_units,
            "performance_profile": profile.performance_profile
        }
    
    def get_performance_ranking(self, operation_type: str, precision: str) -> List[Dict]:
        """
        Get hardware ranked by performance for a specific operation.
        
        Args:
            operation_type: Type of operation (e.g., "matmul", "conv")
            precision: Precision type ("fp32", "fp16", "int8", etc.)
            
        Returns:
            List of dicts with hardware information and performance scores
        """
        if not self._detected:
            self.detect_hardware()
        
        # Convert string to PrecisionType enum
        precision_type = next((p for p in PrecisionType if p.value == precision), PrecisionType.OTHER)
        
        # Get full operation type if only the base operation was provided
        if "_" not in operation_type:
            operation_type = f"{precision}_{operation_type}"
        
        # Get performance ranking
        ranking = self.taxonomy.get_performance_ranking(operation_type, precision_type)
        
        # Convert to simple dicts
        results = []
        for profile, performance in ranking:
            results.append({
                "hardware_class": profile.hardware_class.value,
                "architecture": profile.architecture.value,
                "vendor": profile.vendor.value,
                "model_name": profile.model_name,
                "performance": performance,
                "supported_backends": [backend.value for backend in profile.supported_backends]
            })
        
        return results


def get_enhanced_hardware_info() -> Dict[str, Any]:
    """
    Get comprehensive hardware information using the enhanced detector.
    
    Returns:
        Dict with detailed hardware information
    """
    detector = EnhancedHardwareDetector()
    profiles = detector.detect_hardware()
    
    # Convert hardware profiles to a more serializable format
    serialized_profiles = []
    for profile in profiles:
        serialized_profiles.append({
            "hardware_class": profile.hardware_class.value,
            "architecture": profile.architecture.value,
            "vendor": profile.vendor.value,
            "model_name": profile.model_name,
            "supported_backends": [backend.value for backend in profile.supported_backends],
            "supported_precisions": [precision.value for precision in profile.supported_precisions],
            "features": [feature.value for feature in profile.features],
            "memory_total_gb": profile.memory.total_bytes / (1024 * 1024 * 1024),
            "memory_available_gb": profile.memory.available_bytes / (1024 * 1024 * 1024),
            "compute_units": profile.compute_units,
            "clock_speed_mhz": profile.clock_speed_mhz,
            "performance_profile": profile.performance_profile
        })
    
    # Get optimal hardware for common workloads
    optimal_hardware = {
        "nlp": detector.find_optimal_hardware_for_workload("nlp"),
        "vision": detector.find_optimal_hardware_for_workload("vision"),
        "audio": detector.find_optimal_hardware_for_workload("audio")
    }
    
    return {
        "worker_id": detector.worker_id,
        "hardware_profiles": serialized_profiles,
        "optimal_hardware": optimal_hardware,
        "platform_info": detector._platform_info,
        "browser_info": detector._browser_info,
        "cpu_info": detector._cpu_info,
        "memory_info": detector._memory_info,
        "gpu_info": detector._gpu_info,
        "specialized_hardware": detector._specialized_hardware
    }


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    # Get and print hardware information
    hardware_info = get_enhanced_hardware_info()
    print(json.dumps(hardware_info, indent=2))