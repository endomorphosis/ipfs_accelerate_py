"""
Comprehensive hardware detection module for the IPFS Accelerate framework.
This module provides robust hardware detection with detailed capabilities reporting.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import subprocess
import re
import platform

logger = logging.getLogger(__name__)

# Hardware type constants
CPU = "cpu"
CUDA = "cuda"
ROCM = "rocm"
MPS = "mps"
OPENVINO = "openvino"
WEBNN = "webnn"
WEBGPU = "webgpu"
QUALCOMM = "qualcomm"

# Device map for OpenVINO
OPENVINO_DEVICE_MAP = {
    "CPU": "cpu",
    "GPU": "gpu",
    "MYRIAD": "vpu",
    "HDDL": "vpu",
    "GNA": "gna",
    "HETERO": "hetero",
    "MULTI": "multi",
    "AUTO": "auto"
}

class HardwareDetector:
    """
    Comprehensive hardware detection with detailed capabilities reporting.
    Supports CPU, CUDA, ROCm, MPS (Apple Silicon), OpenVINO, WebNN, WebGPU, and Qualcomm.
    """
    
    def __init__(self, cache_file: Optional[str] = None, force_refresh: bool = False):
        """
        Initialize the hardware detector.
        
        Args:
            cache_file: Optional path to cache detection results
            force_refresh: Force refreshing the cache regardless of its existence
        """
        self.cache_file = cache_file
        self._hardware_info = {}
        self._details = {}
        self._errors = {}
        
        # Load from cache if available and not forcing refresh
        if cache_file and os.path.exists(cache_file) and not force_refresh:
            self._load_from_cache()
        else:
            self._detect_hardware()
            if cache_file:
                self._save_to_cache()
    
    def _load_from_cache(self):
        """Load hardware detection results from cache file"""
        try:
            with open(self.cache_file, 'r') as f:
                cached_data = json.load(f)
                self._hardware_info = cached_data.get('hardware', {})
                self._details = cached_data.get('details', {})
                self._errors = cached_data.get('errors', {})
                logger.info(f"Loaded hardware detection from cache: {self.cache_file}")
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load from cache, performing fresh detection: {str(e)}")
            self._detect_hardware()
            self._save_to_cache()
    
    def _save_to_cache(self):
        """Save hardware detection results to cache file"""
        try:
            cache_data = {
                'hardware': self._hardware_info,
                'details': self._details,
                'errors': self._errors
            }
            
            # Create directory if it doesn't exist
            cache_dir = os.path.dirname(self.cache_file)
            if cache_dir and not os.path.exists(cache_dir):
                os.makedirs(cache_dir, exist_ok=True)
                
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
                logger.info(f"Saved hardware detection to cache: {self.cache_file}")
        except IOError as e:
            logger.warning(f"Failed to save to cache: {str(e)}")
    
    def _detect_hardware(self):
        """Detect available hardware capabilities"""
        logger.info("Detecting hardware capabilities...")
        
        # Always detect CPU
        self._hardware_info[CPU] = True
        self._details[CPU] = self._detect_cpu_capabilities()
        
        # CUDA detection
        self._detect_cuda()
        
        # ROCm (AMD) detection
        self._detect_rocm()
        
        # MPS (Apple Silicon) detection
        self._detect_mps()
        
        # OpenVINO detection
        self._detect_openvino()
        
        # WebNN detection
        self._detect_webnn()
        
        # WebGPU detection
        self._detect_webgpu()
        
        # Qualcomm AI detection
        self._detect_qualcomm()
        
        logger.info(f"Hardware detection complete. Available: {', '.join(hw for hw, available in self._hardware_info.items() if available)}")
    
    def _detect_cpu_capabilities(self) -> Dict[str, Any]:
        """Detect detailed CPU capabilities"""
        cpu_info = {
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "cores": self._get_cpu_cores(),
            "system": platform.system(),
            "sse_support": self._check_cpu_feature("sse"),
            "avx_support": self._check_cpu_feature("avx"),
            "avx2_support": self._check_cpu_feature("avx2"),
            "memory": self._get_system_memory()
        }
        
        # Add basic Python and capabilities info
        cpu_info["python_version"] = platform.python_version()
        cpu_info["python_implementation"] = platform.python_implementation()
        cpu_info["python_compiler"] = platform.python_compiler()
        
        return cpu_info
    
    def _get_cpu_cores(self) -> Dict[str, int]:
        """Get CPU core count information"""
        import multiprocessing
        
        cores = {
            "logical": multiprocessing.cpu_count(),
            "physical": None
        }
        
        # Try to get physical core count (platform-specific)
        try:
            if platform.system() == "Linux":
                # Try using lscpu if available
                try:
                    output = subprocess.check_output("lscpu", shell=True).decode()
                    for line in output.splitlines():
                        if "Core(s) per socket" in line:
                            sockets_line = next((l for l in output.splitlines() if "Socket(s)" in l), None)
                            sockets = 1
                            if sockets_line:
                                sockets = int(sockets_line.split(':')[1].strip())
                            cores_per_socket = int(line.split(':')[1].strip())
                            cores["physical"] = cores_per_socket * sockets
                            break
                except (subprocess.SubprocessError, ValueError):
                    pass
                
                # Fallback to /proc/cpuinfo
                if cores["physical"] is None:
                    try:
                        with open('/proc/cpuinfo', 'r') as f:
                            cpuinfo = f.read()
                        
                        # Count unique combinations of physical id and core id
                        physical_ids = set()
                        for line in cpuinfo.splitlines():
                            if "physical id" in line and "core id" in line:
                                parts = line.split()
                                physical_id = next((p.split(':')[1].strip() for p in parts if "physical id" in p), None)
                                core_id = next((p.split(':')[1].strip() for p in parts if "core id" in p), None)
                                if physical_id and core_id:
                                    physical_ids.add((physical_id, core_id))
                        
                        if physical_ids:
                            cores["physical"] = len(physical_ids)
                    except (IOError, ValueError):
                        pass
            
            elif platform.system() == "Darwin":
                # macOS - use sysctl
                try:
                    output = subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"]).decode().strip()
                    cores["physical"] = int(output)
                except (subprocess.SubprocessError, ValueError):
                    pass
            
            elif platform.system() == "Windows":
                # Windows - use wmic
                try:
                    output = subprocess.check_output("wmic cpu get NumberOfCores", shell=True).decode()
                    lines = output.splitlines()
                    if len(lines) >= 2:
                        cores["physical"] = int(lines[1].strip())
                except (subprocess.SubprocessError, ValueError):
                    pass
        except Exception as e:
            logger.warning(f"Error getting physical core count: {str(e)}")
        
        # If we couldn't determine physical cores, use logical cores
        if cores["physical"] is None:
            cores["physical"] = cores["logical"]
        
        return cores
    
    def _check_cpu_feature(self, feature: str) -> bool:
        """Check if CPU supports a specific feature like SSE, AVX, etc."""
        if platform.system() == "Linux":
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo = f.read()
                return feature.lower() in cpuinfo.lower()
            except IOError:
                pass
        
        # For other platforms or if above method fails
        try:
            import torch
            if feature.lower() == "avx":
                return torch.backends.cpu.supports_avx()
            elif feature.lower() == "avx2":
                return torch.backends.cpu.supports_avx2()
        except (ImportError, AttributeError):
            pass
            
        # Default fallback based on architecture
        arch = platform.machine().lower()
        if feature.lower() in ["sse", "sse2"]:
            # Most x86_64 CPUs support at least SSE2
            return "x86_64" in arch
        elif feature.lower() in ["avx", "avx2"]:
            # Conservatively assume no AVX support by default
            return False
            
        return False
    
    def _get_system_memory(self) -> Dict[str, Any]:
        """Get system memory information"""
        memory_info = {"total": None, "available": None, "unit": "MB"}
        
        try:
            import psutil
            vm = psutil.virtual_memory()
            memory_info["total"] = vm.total / (1024 * 1024)  # Convert to MB
            memory_info["available"] = vm.available / (1024 * 1024)  # Convert to MB
        except ImportError:
            # Fallback methods if psutil is not available
            if platform.system() == "Linux":
                try:
                    with open('/proc/meminfo', 'r') as f:
                        meminfo = f.read()
                    
                    # Extract total memory
                    match = re.search(r'MemTotal:\s+(\d+)', meminfo)
                    if match:
                        memory_info["total"] = int(match.group(1)) / 1024  # Convert from KB to MB
                    
                    # Extract available memory
                    match = re.search(r'MemAvailable:\s+(\d+)', meminfo)
                    if match:
                        memory_info["available"] = int(match.group(1)) / 1024  # Convert from KB to MB
                except IOError:
                    pass
            
            elif platform.system() == "Darwin":
                try:
                    # Get total RAM
                    output = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip()
                    memory_info["total"] = int(output) / (1024 * 1024)  # Convert from bytes to MB
                    
                    # Available memory is harder to get without psutil on macOS
                except (subprocess.SubprocessError, ValueError):
                    pass
                    
            elif platform.system() == "Windows":
                try:
                    output = subprocess.check_output("wmic ComputerSystem get TotalPhysicalMemory", shell=True).decode()
                    lines = output.splitlines()
                    if len(lines) >= 2:
                        memory_info["total"] = int(lines[1].strip()) / (1024 * 1024)  # Convert from bytes to MB
                    
                    # Available memory is harder to get without psutil on Windows
                except (subprocess.SubprocessError, ValueError):
                    pass
        
        return memory_info
    
    def _detect_cuda(self):
        """Detect CUDA availability and capabilities"""
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            
            if cuda_available:
                # Test actual CUDA functionality by creating a small tensor
                try:
                    test_tensor = torch.zeros(1).cuda()
                    del test_tensor
                    torch.cuda.empty_cache()
                    
                    # Get detailed CUDA information
                    self._hardware_info[CUDA] = True
                    self._details[CUDA] = {
                        "device_count": torch.cuda.device_count(),
                        "current_device": torch.cuda.current_device(),
                        "devices": []
                    }
                    
                    # Get info for each device
                    for i in range(torch.cuda.device_count()):
                        device_info = {
                            "name": torch.cuda.get_device_name(i),
                            "capability": torch.cuda.get_device_capability(i),
                            "total_memory": torch.cuda.get_device_properties(i).total_memory / (1024**3),  # Convert to GB
                            "compute_capability_major": torch.cuda.get_device_properties(i).major,
                            "compute_capability_minor": torch.cuda.get_device_properties(i).minor,
                        }
                        
                        # Add information about tensor cores if available
                        if hasattr(torch.cuda, 'get_device_properties'):
                            props = torch.cuda.get_device_properties(i)
                            
                            # Determine if the GPU likely has tensor cores (architecture >= Volta)
                            has_tensor_cores = props.major >= 7
                            device_info["has_tensor_cores"] = has_tensor_cores
                            
                            # Add more device properties if available
                            if hasattr(props, 'multi_processor_count'):
                                device_info["multi_processor_count"] = props.multi_processor_count
                                
                            if hasattr(props, 'max_shared_memory_per_block'):
                                device_info["max_shared_memory_per_block"] = props.max_shared_memory_per_block
                        
                        self._details[CUDA]["devices"].append(device_info)
                    
                    # Add CUDA version information
                    if hasattr(torch.version, 'cuda'):
                        self._details[CUDA]["cuda_version"] = torch.version.cuda
                    
                    # Check for CUDNN
                    if hasattr(torch.backends, 'cudnn'):
                        self._details[CUDA]["cudnn_available"] = torch.backends.cudnn.is_available()
                        if torch.backends.cudnn.is_available() and hasattr(torch.backends.cudnn, 'version'):
                            self._details[CUDA]["cudnn_version"] = torch.backends.cudnn.version()
                except RuntimeError as e:
                    self._hardware_info[CUDA] = False
                    self._errors[CUDA] = f"CUDA initialization failed: {str(e)}"
                    logger.warning(f"CUDA available but initialization failed: {str(e)}")
            else:
                self._hardware_info[CUDA] = False
                self._details[CUDA] = {"reason": "CUDA not available in PyTorch"}
        except ImportError:
            self._hardware_info[CUDA] = False
            self._details[CUDA] = {"reason": "PyTorch not installed"}
        except Exception as e:
            self._hardware_info[CUDA] = False
            self._errors[CUDA] = f"Unexpected error detecting CUDA: {str(e)}"
            logger.error(f"Error detecting CUDA: {str(e)}", exc_info=True)
    
    def _detect_rocm(self):
        """Detect AMD ROCm availability and capabilities"""
        try:
            import torch
            
            # Check if PyTorch was built with ROCm
            is_rocm = False
            if hasattr(torch.version, 'hip') and torch.version.hip is not None:
                is_rocm = True
            
            # For PyTorch with ROCm, CUDA APIs are used, so check CUDA availability
            if is_rocm and torch.cuda.is_available():
                # Try to create a tensor on the GPU to confirm it works
                try:
                    test_tensor = torch.zeros(1).cuda()
                    del test_tensor
                    torch.cuda.empty_cache()
                    
                    self._hardware_info[ROCM] = True
                    self._details[ROCM] = {
                        "device_count": torch.cuda.device_count(),
                        "current_device": torch.cuda.current_device(),
                        "hip_version": torch.version.hip,
                        "devices": []
                    }
                    
                    # Get info for each device
                    for i in range(torch.cuda.device_count()):
                        device_info = {
                            "name": torch.cuda.get_device_name(i),
                            "total_memory": torch.cuda.get_device_properties(i).total_memory / (1024**3)  # Convert to GB
                        }
                        self._details[ROCM]["devices"].append(device_info)
                except RuntimeError as e:
                    self._hardware_info[ROCM] = False
                    self._errors[ROCM] = f"ROCm initialization failed: {str(e)}"
                    logger.warning(f"ROCm available but initialization failed: {str(e)}")
            else:
                self._hardware_info[ROCM] = False
                if not is_rocm:
                    self._details[ROCM] = {"reason": "PyTorch not built with ROCm"}
                else:
                    self._details[ROCM] = {"reason": "No ROCm-compatible GPU available"}
        except ImportError:
            self._hardware_info[ROCM] = False
            self._details[ROCM] = {"reason": "PyTorch not installed"}
        except Exception as e:
            self._hardware_info[ROCM] = False
            self._errors[ROCM] = f"Unexpected error detecting ROCm: {str(e)}"
            logger.error(f"Error detecting ROCm: {str(e)}", exc_info=True)
    
    def _detect_mps(self):
        """Detect Apple Silicon MPS availability and capabilities"""
        # Check if running on macOS first
        if platform.system() != "Darwin":
            self._hardware_info[MPS] = False
            self._details[MPS] = {"reason": "Not running on macOS"}
            return
            
        try:
            import torch
            
            # Check if PyTorch has MPS support
            has_mps_support = hasattr(torch.backends, "mps")
            
            if has_mps_support:
                # Check if MPS is available
                mps_available = torch.backends.mps.is_available()
                
                if mps_available:
                    # Verify MPS works by creating a small tensor
                    try:
                        test_tensor = torch.zeros(1).to('mps')
                        del test_tensor
                        
                        self._hardware_info[MPS] = True
                        self._details[MPS] = {
                            "built_with_mps": torch.backends.mps.is_built(),
                            "macos_version": platform.mac_ver()[0],
                            "device": "mps"
                        }
                        
                        # Try to get model info
                        try:
                            model_name = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
                            self._details[MPS]["model"] = model_name
                        except:
                            pass
                        
                        # Try to get memory info
                        try:
                            ram_info = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip()
                            self._details[MPS]["total_memory_bytes"] = int(ram_info)
                            self._details[MPS]["total_memory_gb"] = int(ram_info) / (1024**3)
                        except:
                            pass
                    except RuntimeError as e:
                        self._hardware_info[MPS] = False
                        self._errors[MPS] = f"MPS initialization failed: {str(e)}"
                        logger.warning(f"MPS available but initialization failed: {str(e)}")
                else:
                    self._hardware_info[MPS] = False
                    self._details[MPS] = {
                        "reason": "MPS not available",
                        "built_with_mps": torch.backends.mps.is_built() if hasattr(torch.backends.mps, "is_built") else False
                    }
            else:
                self._hardware_info[MPS] = False
                self._details[MPS] = {"reason": "PyTorch built without MPS support"}
        except ImportError:
            self._hardware_info[MPS] = False
            self._details[MPS] = {"reason": "PyTorch not installed"}
        except Exception as e:
            self._hardware_info[MPS] = False
            self._errors[MPS] = f"Unexpected error detecting MPS: {str(e)}"
            logger.error(f"Error detecting MPS: {str(e)}", exc_info=True)
    
    def _detect_openvino(self):
        """Detect OpenVINO availability and capabilities"""
        try:
            import openvino
            
            self._hardware_info[OPENVINO] = True
            self._details[OPENVINO] = {
                "version": openvino.__version__,
            }
            
            # Try to get available devices
            try:
                from openvino.runtime import Core
                core = Core()
                available_devices = core.available_devices
                
                self._details[OPENVINO]["available_devices"] = available_devices
                self._details[OPENVINO]["device_info"] = {}
                
                # Get detailed device info for each available device
                for device in available_devices:
                    try:
                        device_name = device
                        device_type = OPENVINO_DEVICE_MAP.get(device.split('.')[0], "unknown")
                        
                        # Get full device info
                        full_device_info = core.get_property(device, "FULL_DEVICE_NAME")
                        
                        self._details[OPENVINO]["device_info"][device] = {
                            "name": device_name,
                            "type": device_type,
                            "full_name": full_device_info
                        }
                        
                        # Get additional properties for specific device types
                        if device_type == "cpu":
                            try:
                                cpu_threads = core.get_property(device, "CPU_THREADS_NUM")
                                self._details[OPENVINO]["device_info"][device]["cpu_threads"] = cpu_threads
                            except:
                                pass
                        elif device_type == "gpu":
                            try:
                                gpu_device_name = core.get_property(device, "DEVICE_ARCHITECTURE")
                                self._details[OPENVINO]["device_info"][device]["architecture"] = gpu_device_name
                            except:
                                pass
                    except Exception as e:
                        logger.warning(f"Could not get detailed info for OpenVINO device {device}: {str(e)}")
            except ImportError:
                self._details[OPENVINO]["available_devices"] = ["CPU"]  # Default assumption
                logger.warning("OpenVINO Core module not available, cannot detect devices")
            except Exception as e:
                self._details[OPENVINO]["available_devices_error"] = str(e)
                logger.warning(f"Error detecting OpenVINO devices: {str(e)}")
        except ImportError:
            self._hardware_info[OPENVINO] = False
            self._details[OPENVINO] = {"reason": "OpenVINO not installed"}
        except Exception as e:
            self._hardware_info[OPENVINO] = False
            self._errors[OPENVINO] = f"Unexpected error detecting OpenVINO: {str(e)}"
            logger.error(f"Error detecting OpenVINO: {str(e)}", exc_info=True)
    
    def _detect_webnn(self):
        """Detect WebNN (Web Neural Network) availability and capabilities"""
        # WebNN is primarily for web browsers, but check for Node.js-based implementations
        self._hardware_info[WEBNN] = False
        
        # Check for Node.js
        try:
            # First check if node is available using 'which'
            node_available = False
            try:
                with open(os.devnull, 'w') as devnull:
                    subprocess.check_call(["which", "node"], stdout=devnull, stderr=devnull)
                node_available = True
            except (subprocess.SubprocessError, FileNotFoundError):
                self._details[WEBNN] = {"reason": "Node.js not available"}
                return
            
            if node_available:
                node_version = subprocess.check_output(["node", "--version"], universal_newlines=True).strip()
                
                # Check for ONNX export capabilities without requiring npm list
                try:
                    import torch
                    import onnx
                    self._hardware_info[WEBNN] = True
                    self._details[WEBNN] = {
                        "node_version": node_version,
                        "dependencies": {},
                        "python_export_capability": {
                            "torch": torch.__version__,
                            "onnx": onnx.__version__
                        }
                    }
                    logger.info("WebNN support detected via Python ONNX export capabilities")
                    return
                except ImportError:
                    pass
                
                # Try to check for NPM packages safely
                try:
                    # Check if npm is available
                    with open(os.devnull, 'w') as devnull:
                        subprocess.check_call(["which", "npm"], stdout=devnull, stderr=devnull)
                    
                    npm_list = subprocess.check_output(["npm", "list", "--json"], universal_newlines=True)
                    npm_packages = json.loads(npm_list)
                    
                    dependencies = npm_packages.get("dependencies", {})
                    has_onnxruntime_web = "onnxruntime-web" in dependencies
                    has_webnn_api = "webnn-api" in dependencies or "webnn-polyfill" in dependencies
                    
                    if has_onnxruntime_web or has_webnn_api:
                        self._hardware_info[WEBNN] = True
                        self._details[WEBNN] = {
                            "node_version": node_version,
                            "dependencies": {
                                "onnxruntime_web": has_onnxruntime_web,
                                "webnn_api": has_webnn_api
                            }
                        }
                    else:
                        self._details[WEBNN] = {
                            "reason": "Required NPM packages not installed",
                            "node_version": node_version
                        }
                except (subprocess.SubprocessError, json.JSONDecodeError, FileNotFoundError) as e:
                    self._details[WEBNN] = {
                        "reason": f"NPM not available or error: {str(e)}",
                        "node_version": node_version
                    }
        except Exception as e:
            self._errors[WEBNN] = f"Unexpected error detecting WebNN: {str(e)}"
            logger.warning(f"Error detecting WebNN: {str(e)}")
    
    def _detect_webgpu(self):
        """Detect WebGPU availability and capabilities"""
        # WebGPU is primarily for web browsers, but check for Node.js-based implementations
        self._hardware_info[WEBGPU] = False
        
        # Check for Node.js
        try:
            # First check if node is available using 'which'
            node_available = False
            try:
                with open(os.devnull, 'w') as devnull:
                    subprocess.check_call(["which", "node"], stdout=devnull, stderr=devnull)
                node_available = True
            except (subprocess.SubprocessError, FileNotFoundError):
                self._details[WEBGPU] = {"reason": "Node.js not available"}
                return
            
            if node_available:
                node_version = subprocess.check_output(["node", "--version"], universal_newlines=True).strip()
                
                # Check for ONNX export capabilities without requiring npm list
                try:
                    import torch
                    
                    # Check if ONNX is available for WebGPU export
                    try:
                        import onnx
                        self._hardware_info[WEBGPU] = True
                        self._details[WEBGPU] = {
                            "node_version": node_version,
                            "dependencies": {},
                            "python_export_capability": {
                                "torch": torch.__version__,
                                "onnx": onnx.__version__
                            }
                        }
                        logger.info("WebGPU support detected via Python ONNX export capabilities")
                        return
                    except ImportError:
                        pass
                except ImportError:
                    pass
                
                # Try to check for NPM packages safely
                try:
                    # Check if npm is available
                    with open(os.devnull, 'w') as devnull:
                        subprocess.check_call(["which", "npm"], stdout=devnull, stderr=devnull)
                    
                    npm_list = subprocess.check_output(["npm", "list", "--json"], universal_newlines=True)
                    npm_packages = json.loads(npm_list)
                    
                    dependencies = npm_packages.get("dependencies", {})
                    has_transformers_js = "@xenova/transformers" in dependencies
                    has_webgpu = "@webgpu/types" in dependencies
                    
                    if has_transformers_js or has_webgpu:
                        self._hardware_info[WEBGPU] = True
                        self._details[WEBGPU] = {
                            "node_version": node_version,
                            "dependencies": {
                                "transformers_js": has_transformers_js,
                                "webgpu_types": has_webgpu
                            }
                        }
                        
                        # Check for additional capabilities
                        if has_transformers_js:
                            try:
                                # Try to get transformers.js version
                                pkg_info = subprocess.check_output(["npm", "view", "@xenova/transformers", "version"], universal_newlines=True).strip()
                                self._details[WEBGPU]["transformers_js_version"] = pkg_info
                            except:
                                pass
                    else:
                        self._details[WEBGPU] = {
                            "reason": "Required NPM packages not installed",
                            "node_version": node_version
                        }
                except (subprocess.SubprocessError, json.JSONDecodeError, FileNotFoundError) as e:
                    self._details[WEBGPU] = {
                        "reason": f"NPM not available or error: {str(e)}",
                        "node_version": node_version
                    }
        except Exception as e:
            self._errors[WEBGPU] = f"Unexpected error detecting WebGPU: {str(e)}"
            logger.warning(f"Error detecting WebGPU: {str(e)}")
    
    def _detect_qualcomm(self):
        """Detect Qualcomm AI capabilities"""
        self._hardware_info[QUALCOMM] = False
        
        # Check for Qualcomm AI Engine Direct
        try:
            # Try to import qnn modules if available
            try:
                import importlib.util
                # Check if the module exists before trying to import it
                if importlib.util.find_spec("qti.aisw.dlc_utils") is not None:
                    import qti.aisw.dlc_utils as dlc_utils
                    import qti.aisw.converters as converters
                    
                    self._hardware_info[QUALCOMM] = True
                    self._details[QUALCOMM] = {
                        "qnn_available": True,
                        "backend": "QNN"
                    }
                else:
                    raise ImportError("qti.aisw.dlc_utils module not found")
            except ImportError:
                # Try checking for SNPE
                try:
                    if importlib.util.find_spec("snpe") is not None:
                        import snpe
                        self._hardware_info[QUALCOMM] = True
                        self._details[QUALCOMM] = {
                            "snpe_available": True,
                            "backend": "SNPE"
                        }
                    else:
                        raise ImportError("snpe module not found")
                except ImportError:
                    # Final check for command-line tools - use which instead of direct execution
                    try:
                        # First check if the command exists using 'which'
                        with open(os.devnull, 'w') as devnull:
                            subprocess.check_call(["which", "snpe-net-run"], stdout=devnull, stderr=devnull)
                        
                        # If we get here, the command exists, so we can try to get its version
                        snpe_version = subprocess.check_output(["snpe-net-run", "--version"], universal_newlines=True).strip()
                        self._hardware_info[QUALCOMM] = True
                        self._details[QUALCOMM] = {
                            "snpe_cli_available": True,
                            "version": snpe_version,
                            "backend": "SNPE CLI"
                        }
                    except (subprocess.SubprocessError, FileNotFoundError):
                        self._details[QUALCOMM] = {"reason": "Qualcomm AI SDK not found"}
        except Exception as e:
            self._errors[QUALCOMM] = f"Unexpected error detecting Qualcomm AI: {str(e)}"
            logger.warning(f"Error detecting Qualcomm AI: {str(e)}")
    
    def get_available_hardware(self) -> Dict[str, bool]:
        """Get dictionary of available hardware platforms"""
        return self._hardware_info
    
    def get_hardware_details(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed information about hardware capabilities"""
        return self._details
    
    def get_errors(self) -> Dict[str, str]:
        """Get errors that occurred during hardware detection"""
        return self._errors
    
    def is_available(self, hardware_type: str) -> bool:
        """Check if specific hardware type is available"""
        return self._hardware_info.get(hardware_type, False)
    
    def get_best_available_hardware(self) -> str:
        """Get the best available hardware platform for inference"""
        # Priority order: CUDA > ROCm > MPS > OpenVINO > CPU
        if self.is_available(CUDA):
            return CUDA
        elif self.is_available(ROCM):
            return ROCM
        elif self.is_available(MPS):
            return MPS
        elif self.is_available(OPENVINO):
            return OPENVINO
        else:
            return CPU
    
    def get_torch_device(self) -> str:
        """Get the appropriate torch device string for the best available hardware"""
        if self.is_available(CUDA):
            return "cuda"
        elif self.is_available(ROCM):
            return "cuda"  # ROCm uses CUDA API
        elif self.is_available(MPS):
            return "mps"
        else:
            return "cpu"
            
    def get_device_with_index(self, preferred_index: int = 0) -> str:
        """
        Get device string with specific index if available
        
        Args:
            preferred_index: The preferred GPU index to use (e.g., cuda:0, cuda:1)
            
        Returns:
            Device string with index if available, otherwise best available device
        """
        device = self.get_torch_device()
        
        # Only add index for CUDA or ROCm devices
        if device == "cuda":
            try:
                import torch
                device_count = torch.cuda.device_count()
                if device_count > 0:
                    # Ensure index is valid
                    if preferred_index >= 0 and preferred_index < device_count:
                        return f"{device}:{preferred_index}"
                    else:
                        return f"{device}:0"  # Default to first device
                    
                    # Additional logging for debugging
                    logger.debug(f"Selected CUDA device index {preferred_index} from {device_count} available devices")
            except (ImportError, AttributeError) as e:
                logger.debug(f"Error selecting CUDA device: {str(e)}")
                pass
        
        # For ROCm devices we also add indices (ROCm uses CUDA API)
        elif self.is_available(ROCM) and device == "cuda":
            try:
                import torch
                device_count = torch.cuda.device_count()
                if device_count > 0:
                    # Ensure index is valid
                    if preferred_index >= 0 and preferred_index < device_count:
                        return f"{device}:{preferred_index}"
                    else:
                        return f"{device}:0"  # Default to first device
                        
                    logger.debug(f"Selected ROCm device index {preferred_index} from {device_count} available devices")
            except (ImportError, AttributeError) as e:
                logger.debug(f"Error selecting ROCm device: {str(e)}")
                pass
                
        return device
    
    def get_compatible_hardware_types(self, model_requirements: Dict[str, Any]) -> List[str]:
        """
        Get list of hardware types compatible with model requirements
        
        Args:
            model_requirements: Dictionary with model hardware requirements
            
        Returns:
            List of compatible hardware types
        """
        compatible = []
        
        # CPU is always compatible
        compatible.append(CPU)
        
        # Check each hardware type
        for hw_type in [CUDA, ROCM, MPS, OPENVINO, WEBNN, WEBGPU, QUALCOMM]:
            if not self.is_available(hw_type):
                continue
                
            # Check if model has explicit compatibility information
            hw_compatibility = model_requirements.get(hw_type, {}).get("compatible", True)
            if hw_compatibility:
                compatible.append(hw_type)
                
            # Check memory requirements
            if hw_type in [CUDA, ROCM, MPS]:
                required_memory = model_requirements.get("memory_requirements", {}).get(hw_type, 0)
                available_memory = 0
                
                if hw_type == CUDA and self._details.get(CUDA, {}).get("devices"):
                    available_memory = self._details[CUDA]["devices"][0].get("total_memory", 0)
                elif hw_type == ROCM and self._details.get(ROCM, {}).get("devices"):
                    available_memory = self._details[ROCM]["devices"][0].get("total_memory", 0)
                
                if required_memory > 0 and available_memory > 0 and required_memory > available_memory:
                    # Remove from compatible list if already added
                    if hw_type in compatible:
                        compatible.remove(hw_type)
        
        return compatible
        
    def get_hardware_by_priority(self, priority_list: Optional[List[str]] = None) -> str:
        """
        Get the best available hardware based on a priority list
        
        Args:
            priority_list: Optional priority list of hardware types
            
        Returns:
            The best available hardware type from the priority list
        """
        # Default priority: CUDA > ROCm > MPS > OpenVINO > CPU
        if priority_list is None:
            priority_list = [CUDA, ROCM, MPS, OPENVINO, CPU]
        
        logger.debug(f"Selecting hardware using priority list: {priority_list}")
        
        # Return the first available hardware type from the priority list
        for hw_type in priority_list:
            if self.is_available(hw_type):
                logger.info(f"Selected hardware {hw_type} based on priority list: {priority_list}")
                return hw_type
        
        # Fallback to CPU if nothing from the priority list is available
        logger.warning(f"No hardware from priority list {priority_list} is available, falling back to CPU")
        return CPU
        
    def get_torch_device_with_priority(self, priority_list: Optional[List[str]] = None, 
                                       preferred_index: int = 0) -> str:
        """
        Get torch device string using priority list and preferred device index
        
        Args:
            priority_list: Optional priority list of hardware types
            preferred_index: Preferred GPU index for CUDA/ROCm devices
            
        Returns:
            PyTorch device string with appropriate format
        """
        # Get best hardware based on priority list
        best_hardware = self.get_hardware_by_priority(priority_list)
        
        # Convert to torch device string
        if best_hardware == CUDA or best_hardware == ROCM:
            # For GPU hardware, add device index
            device_base = "cuda"  # Both CUDA and ROCm use "cuda" in PyTorch
            try:
                import torch
                device_count = torch.cuda.device_count()
                if device_count > 0:
                    # Ensure index is valid
                    if preferred_index >= 0 and preferred_index < device_count:
                        return f"{device_base}:{preferred_index}"
                    else:
                        return f"{device_base}:0"  # Default to first device
            except (ImportError, AttributeError) as e:
                logger.debug(f"Error selecting GPU device index: {str(e)}")
                return device_base
        elif best_hardware == MPS:
            return "mps"
        else:
            return "cpu"
    
    def print_summary(self, detailed: bool = False):
        """Print a summary of detected hardware capabilities"""
        from pprint import pprint
        
        print("\n=== Hardware Detection Summary ===")
        print(f"Available hardware: {', '.join(hw for hw, available in self._hardware_info.items() if available)}")
        print(f"Best available hardware: {self.get_best_available_hardware()}")
        
        if detailed:
            print("\n=== Detailed Hardware Information ===")
            pprint(self._details)
            
            if self._errors:
                print("\n=== Detection Errors ===")
                pprint(self._errors)


def detect_available_hardware(cache_file: Optional[str] = None, 
                             priority_list: Optional[List[str]] = None,
                             preferred_device_index: int = 0) -> Dict[str, Any]:
    """
    Detect available hardware with comprehensive error handling
    
    Args:
        cache_file: Optional path to cache detection results
        priority_list: Optional list of hardware types in priority order
        preferred_device_index: Optional preferred device index for multi-GPU systems
        
    Returns:
        Dictionary with hardware availability and details
    """
    detector = HardwareDetector(cache_file=cache_file)
    
    # Get best hardware based on priority list if provided
    if priority_list:
        best_hardware = detector.get_hardware_by_priority(priority_list)
        # Get torch device with priority and index
        torch_device = detector.get_torch_device_with_priority(priority_list, preferred_device_index)
    else:
        best_hardware = detector.get_best_available_hardware()
        # Get torch device with index
        torch_device = detector.get_device_with_index(preferred_device_index)
    
    # Include the custom priority list in the result if provided
    result = {
        "hardware": detector.get_available_hardware(),
        "details": detector.get_hardware_details(),
        "errors": detector.get_errors(),
        "best_available": best_hardware,
        "torch_device": torch_device
    }
    
    # Include custom priority settings if provided
    if priority_list:
        result["priority_list"] = priority_list
        result["preferred_device_index"] = preferred_device_index
    
    return result

def detect_hardware_with_comprehensive_checks() -> Dict[str, Any]:
    """
    Enhanced hardware detection with robust error handling and comprehensive checks.
    Provides detailed capabilities for all hardware types with extensive fallbacks.
    
    Returns:
        Dictionary with detailed hardware information and capabilities
    """
    hardware = {"cpu": True}
    
    # Test CUDA availability with comprehensive error handling
    try:
        import torch
        if torch.cuda.is_available():
            try:
                # Test actual CUDA functionality, not just library presence
                test_tensor = torch.zeros(1).cuda()
                # Run a simple operation to verify CUDA is actually working
                _ = test_tensor + 1
                del test_tensor
                torch.cuda.empty_cache()
                
                hardware["cuda"] = True
                # Get detailed GPU information
                hardware["cuda_device_count"] = torch.cuda.device_count()
                hardware["cuda_devices"] = []
                
                for device_idx in range(torch.cuda.device_count()):
                    device_props = torch.cuda.get_device_properties(device_idx)
                    hardware["cuda_devices"].append({
                        "name": torch.cuda.get_device_name(device_idx),
                        "total_memory": device_props.total_memory / (1024**3),  # GB
                        "compute_capability": f"{device_props.major}.{device_props.minor}",
                        "multi_processor_count": device_props.multi_processor_count
                    })
                    
                hardware["cuda_current_device"] = torch.cuda.current_device()
                hardware["cuda_arch_list"] = torch.cuda.get_arch_list() if hasattr(torch.cuda, 'get_arch_list') else None
                hardware["cuda_memory_allocated"] = torch.cuda.memory_allocated() / (1024**3)  # GB
                hardware["cuda_memory_reserved"] = torch.cuda.memory_reserved() / (1024**3)  # GB
            except RuntimeError as rt_err:
                hardware["cuda"] = False
                hardware["cuda_error"] = f"CUDA runtime error: {str(rt_err)}"
                hardware["cuda_available_but_not_working"] = True
        else:
            hardware["cuda"] = False
            hardware["cuda_error"] = "torch.cuda.is_available() returned False"
    except Exception as e:
        hardware["cuda"] = False
        hardware["cuda_error"] = f"Exception detecting CUDA: {str(e)}"
    
    # Test MPS (Apple Silicon) availability with detailed checks
    try:
        import torch
        if hasattr(torch, 'mps') and torch.backends.mps.is_available():
            try:
                # Verify MPS works by creating a small tensor and performing an operation
                test_tensor = torch.zeros(1).to('mps')
                _ = test_tensor + 1
                del test_tensor
                
                hardware["mps"] = True
                # Get MPS device information if available
                if hasattr(torch.mps, 'current_allocated_memory'):
                    hardware["mps_allocated_memory"] = torch.mps.current_allocated_memory() / (1024**3)  # GB
                
                # Check if Metal Performance Shaders are actually supported by the current device
                hardware["mps_is_built"] = torch.backends.mps.is_built()
                hardware["mps_is_available"] = torch.backends.mps.is_available()
                
                # Get macOS version if possible
                import platform
                if platform.system() == 'Darwin':
                    hardware["macos_version"] = platform.mac_ver()[0]
            except RuntimeError as rt_err:
                hardware["mps"] = False
                hardware["mps_error"] = f"MPS runtime error: {str(rt_err)}"
                hardware["mps_available_but_not_working"] = True
        else:
            hardware["mps"] = False
            hardware["mps_is_built"] = torch.backends.mps.is_built() if hasattr(torch.backends.mps, 'is_built') else False
            hardware["mps_error"] = "MPS not available in current PyTorch installation"
    except Exception as e:
        hardware["mps"] = False
        hardware["mps_error"] = f"Exception detecting MPS: {str(e)}"
    
    # Test ROCm (AMD) availability with multiple detection methods
    try:
        import torch
        # Check if CUDA is available (ROCm appears as CUDA in PyTorch)
        if torch.cuda.is_available():
            # Check if this is actually ROCm and not NVIDIA CUDA
            is_rocm = False
            
            # Method 1: Check torch.version.hip attribute
            if hasattr(torch.version, 'hip') and torch.version.hip is not None:
                is_rocm = True
            
            # Method 2: Look for AMD in the device name
            if not is_rocm:
                try:
                    device_name = torch.cuda.get_device_name(0).lower()
                    if 'amd' in device_name or 'radeon' in device_name:
                        is_rocm = True
                except:
                    pass
            
            # Method 3: Try to import torch_xla module which might indicate ROCm support
            if not is_rocm:
                try:
                    import torch_xla
                    import torch_xla.core.xla_model as xm
                    # If we got here, there might be ROCm support via XLA
                    hardware["xla_available"] = True
                except ImportError:
                    hardware["xla_available"] = False
            
            if is_rocm:
                try:
                    # Test that ROCm actually works
                    test_tensor = torch.zeros(1).cuda()
                    _ = test_tensor + 1
                    del test_tensor
                    torch.cuda.empty_cache()
                    
                    hardware["rocm"] = True
                    # Get detailed GPU information for ROCm
                    hardware["rocm_device_count"] = torch.cuda.device_count()
                    hardware["rocm_devices"] = []
                    
                    for device_idx in range(torch.cuda.device_count()):
                        device_props = torch.cuda.get_device_properties(device_idx)
                        hardware["rocm_devices"].append({
                            "name": torch.cuda.get_device_name(device_idx),
                            "total_memory": device_props.total_memory / (1024**3),  # GB
                            "multi_processor_count": device_props.multi_processor_count
                        })
                    
                    hardware["rocm_current_device"] = torch.cuda.current_device()
                    hardware["rocm_memory_allocated"] = torch.cuda.memory_allocated() / (1024**3)  # GB
                    hardware["rocm_memory_reserved"] = torch.cuda.memory_reserved() / (1024**3)  # GB
                except RuntimeError as rt_err:
                    hardware["rocm"] = False
                    hardware["rocm_error"] = f"ROCm runtime error: {str(rt_err)}"
                    hardware["rocm_available_but_not_working"] = True
            else:
                hardware["rocm"] = False
                hardware["rocm_error"] = "CUDA available, but no ROCm/AMD GPU detected"
        else:
            hardware["rocm"] = False
            hardware["rocm_error"] = "torch.cuda.is_available() returned False"
    except Exception as e:
        hardware["rocm"] = False
        hardware["rocm_error"] = f"Exception detecting ROCm: {str(e)}"
    
    # Test OpenVINO availability with comprehensive checks
    try:
        import openvino
        hardware["openvino"] = True
        hardware["openvino_version"] = openvino.__version__
        
        # Try to get detailed OpenVINO information
        try:
            from openvino.runtime import Core
            core = Core()
            hardware["openvino_devices"] = core.available_devices
            
            # Get device information for each available device
            hardware["openvino_device_info"] = {}
            for device in core.available_devices:
                try:
                    full_device_name = device
                    if device != "CPU" and not device.startswith("GPU"):
                        full_device_name = f"GPU.{device}"
                    hardware["openvino_device_info"][device] = core.get_property(full_device_name, "FULL_DEVICE_NAME")
                except:
                    hardware["openvino_device_info"][device] = "Property information not available"
            
            # Check for GPU plugin specifically
            hardware["openvino_gpu_plugin"] = "GPU" in core.available_devices
            
            # Test a simple network to verify OpenVINO works
            try:
                import numpy as np
                from openvino.runtime import Type, Layout, Shape, Model, Output, opset
                
                # Create a simple model with a single operation
                input_shape = [1, 3, 224, 224]  # NCHW format
                input_type = Type.f32
                param_shape = [1]
                
                # Create model inputs/parameters
                input_node = opset.parameter(Shape(input_shape), Type.f32, name="data")
                param_node = opset.parameter(Shape(param_shape), Type.f32, name="bias")
                
                # Simple operation: add a scalar bias to all elements
                result_node = opset.add(input_node, param_node, name="output")
                
                # Create model
                model = Model([result_node], [input_node, param_node], "test_model")
                
                # Try to compile on available devices
                hardware["openvino_compiled"] = {}
                for device in core.available_devices:
                    try:
                        compiled_model = core.compile_model(model, device)
                        hardware["openvino_compiled"][device] = True
                    except Exception as e:
                        hardware["openvino_compiled"][device] = False
                        hardware["openvino_compiled_error_" + device] = str(e)
            except Exception as compile_err:
                hardware["openvino_compilation_test_error"] = str(compile_err)
        except Exception as core_err:
            hardware["openvino_core_error"] = str(core_err)
    except ImportError:
        hardware["openvino"] = False
        hardware["openvino_error"] = "OpenVINO not installed"
    except Exception as e:
        hardware["openvino"] = False
        hardware["openvino_error"] = f"Exception detecting OpenVINO: {str(e)}"
    
    # Test WebNN availability with detailed checks
    try:
        # WebNN requires TensorFlow.js or ONNX Runtime Web
        hardware["webnn"] = False
        
        # Try TensorFlow.js approach first
        try:
            import tensorflowjs as tfjs
            # If import succeeded, TensorFlow.js is available which might support WebNN
            hardware["tfjs_available"] = True
            
            # Try to check WebNN support
            try:
                # This is a mock check since true WebNN testing requires a browser environment
                # In a real browser environment, you would use navigator.ml.isWebNNSupported()
                hardware["webnn_potentially_supported"] = True
            except:
                pass
        except ImportError:
            hardware["tfjs_available"] = False
        
        # Try ONNX Runtime Web approach
        try:
            import onnxruntime as ort
            hardware["onnxruntime_available"] = True
            hardware["onnxruntime_version"] = ort.__version__
            
            # Check available providers
            providers = ort.get_available_providers()
            hardware["onnxruntime_providers"] = providers
            
            # WebNN would be accessible in browser contexts with onnxruntime-web
            if "WebNNExecutionProvider" in providers:
                hardware["webnn"] = True
        except ImportError:
            hardware["onnxruntime_available"] = False
    except Exception as e:
        hardware["webnn_error"] = f"Exception detecting WebNN: {str(e)}"
    
    # Check for additional accelerators like TPUs
    try:
        # Try to detect TPU
        try:
            import torch_xla
            import torch_xla.core.xla_model as xm
            hardware["tpu_available"] = True
            try:
                # Get device information
                device = xm.xla_device()
                hardware["tpu_device"] = str(device)
                hardware["tpu_device_type"] = device.type
            except:
                hardware["tpu_device_error"] = "Could not get TPU device information"
        except ImportError:
            hardware["tpu_available"] = False
        
        # Try to detect Qualcomm AI Engine Direct
        try:
            import qti.aisw.dlc_utils
            hardware["qualcomm_ai"] = True
            hardware["qualcomm_ai_version"] = qti.aisw.dlc_utils.__version__
        except ImportError:
            try:
                # Alternative: Check for snpe-dlc-utils
                import snpe.dlc
                hardware["qualcomm_ai"] = True
                hardware["qualcomm_ai_version"] = "SNPE available (version unknown)"
            except ImportError:
                hardware["qualcomm_ai"] = False
    except Exception as e:
        hardware["accelerator_detection_error"] = f"Exception detecting additional accelerators: {str(e)}"
    
    # Get general system info
    try:
        import platform
        import os
        import multiprocessing
        
        hardware["system"] = {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": multiprocessing.cpu_count(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation()
        }
        
        # Get memory information
        try:
            import psutil
            vm = psutil.virtual_memory()
            hardware["system"]["total_memory"] = vm.total / (1024**3)  # GB
            hardware["system"]["available_memory"] = vm.available / (1024**3)  # GB
            hardware["system"]["memory_percent_used"] = vm.percent
        except ImportError:
            pass
    except Exception as e:
        hardware["system_info_error"] = f"Exception getting system info: {str(e)}"
    
    return hardware


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, 
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create detector and output results
    detector = HardwareDetector(cache_file="hardware_detection_cache.json")
    detector.print_summary(detailed=True)
    
    # Export results to JSON
    with open("hardware_detection_results.json", "w") as f:
        json.dump({
            "hardware": detector.get_available_hardware(),
            "details": detector.get_hardware_details(),
            "errors": detector.get_errors(),
            "best_available": detector.get_best_available_hardware(),
            "torch_device": detector.get_torch_device()
        }, f, indent=2)
    
    print(f"\nResults exported to hardware_detection_results.json")