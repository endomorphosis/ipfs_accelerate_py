#!/usr/bin/env python3
"""
IPFS Accelerate Python Framework - Hardware Detection Module

This module provides comprehensive hardware detection for accelerated machine learning,
supporting multiple hardware platforms and providing detailed capabilities reporting.

Supported hardware platforms:
- CPU (always available)
- CUDA (NVIDIA GPUs)
- ROCm (AMD GPUs)
- MPS (Apple Silicon)
- OpenVINO (Intel)
- WebNN (Browser neural network API)
- WebGPU (Browser GPU API)
- Qualcomm (Mobile/Edge AI)

Key features:
- Reliable detection with fallbacks
- Detailed capability reporting
- Performance optimization suggestions
- Web platform optimizations
"""

import os
import sys
import json
import platform
import logging
import subprocess
import importlib.util
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import re
import time
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("hardware_detection")

# Hardware type constants
CPU = "cpu"
CUDA = "cuda"
ROCM = "rocm"
MPS = "mps"
OPENVINO = "openvino"
WEBNN = "webnn"
WEBGPU = "webgpu"
QUALCOMM = "qualcomm"

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
                'errors': self._errors,
                'timestamp': datetime.now().isoformat()
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
            # Try to import psutil for memory info
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
                        
                        # Get full device info
                        try:
                            full_device_info = core.get_property(device, "FULL_DEVICE_NAME")
                            self._details[OPENVINO]["device_info"][device] = {
                                "name": device_name,
                                "full_name": full_device_info
                            }
                        except:
                            self._details[OPENVINO]["device_info"][device] = {
                                "name": device_name
                            }
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
        # Start by assuming WebNN is not available
        self._hardware_info[WEBNN] = False
        
        # First check for browser environment
        try:
            # This would only work if executed in a browser environment like Pyodide
            # or some browser Python runtime
            js_window = eval("window")
            js_navigator = eval("navigator")
            
            # Check for WebNN API in the browser
            if "ml" in js_navigator and "NeuralNetwork" in js_navigator.ml:
                self._hardware_info[WEBNN] = True
                self._details[WEBNN] = {
                    "environment": "browser",
                    "navigator_ml": True,
                    "mode": "direct_browser_detection"
                }
                logger.info("WebNN API detected in browser environment")
                return
            else:
                self._details[WEBNN] = {
                    "environment": "browser",
                    "navigator_ml": False,
                    "reason": "WebNN API not available in browser"
                }
        except:
            # Not in a browser environment
            pass
            
        # Check for simulation mode
        if os.environ.get("WEBNN_SIMULATION") == "1":
            self._hardware_info[WEBNN] = True
            self._details[WEBNN] = {
                "environment": "simulation",
                "mode": "simulated_environment",
                "simulation_enabled": True
            }
            logger.info("WebNN simulation mode enabled")
            return
            
        # Check if explicitly set WEBNN_AVAILABLE=1
        if os.environ.get("WEBNN_AVAILABLE") == "1":
            self._hardware_info[WEBNN] = True
            self._details[WEBNN] = {
                "environment": "override",
                "mode": "environment_variable_override",
                "simulation_enabled": True
            }
            logger.info("WebNN availability forced by environment variable")
            return
        
        # Check for ONNX export capabilities
        try:
            import torch
            import onnx
            self._hardware_info[WEBNN] = True
            self._details[WEBNN] = {
                "environment": "node",
                "mode": "onnx_export",
                "python_export_capability": {
                    "torch": torch.__version__,
                    "onnx": onnx.__version__
                }
            }
            logger.info("WebNN support detected via Python ONNX export capabilities")
            return
        except ImportError:
            pass
        
        # Final details
        self._details[WEBNN] = {
            "reason": "WebNN not available",
            "environment": "node"
        }
    
    def _detect_webgpu(self):
        """Detect WebGPU availability and capabilities"""
        # WebGPU is primarily for web browsers, but check for Node.js-based implementations
        # Start by assuming WebGPU is not available
        self._hardware_info[WEBGPU] = False
        
        # First check for browser environment
        try:
            # This would only work if executed in a browser environment like Pyodide
            js_window = eval("window")
            js_navigator = eval("navigator")
            
            # Check for WebGPU API in browser
            if "gpu" in js_navigator:
                # Try to verify adapter availability
                try:
                    has_adapter = eval("navigator.gpu.requestAdapter() !== null")
                except:
                    has_adapter = False  # Can't verify adapter availability
                
                self._hardware_info[WEBGPU] = True
                self._details[WEBGPU] = {
                    "environment": "browser",
                    "navigator_gpu": True,
                    "adapter_available": has_adapter,
                    "mode": "direct_browser_detection"
                }
                logger.info("WebGPU API detected in browser environment")
                return
            else:
                self._details[WEBGPU] = {
                    "environment": "browser",
                    "navigator_gpu": False,
                    "reason": "WebGPU API not available in browser"
                }
        except:
            # Not in a browser environment
            pass
        
        # Check for WebGPU simulation mode
        if os.environ.get("WEBGPU_SIMULATION") == "1":
            self._hardware_info[WEBGPU] = True
            self._details[WEBGPU] = {
                "environment": "simulation",
                "mode": "simulated_environment",
                "simulation_enabled": True
            }
            logger.info("WebGPU simulation mode enabled")
            return
            
        # Check if explicitly set WEBGPU_AVAILABLE=1
        if os.environ.get("WEBGPU_AVAILABLE") == "1":
            self._hardware_info[WEBGPU] = True
            self._details[WEBGPU] = {
                "environment": "override",
                "mode": "environment_variable_override",
                "simulation_enabled": True
            }
            logger.info("WebGPU availability forced by environment variable")
            return
        
        # Check for Node.js with WebGPU support
        try:
            # First check if node is available
            with open(os.devnull, 'w') as devnull:
                subprocess.check_call(["which", "node"], stdout=devnull, stderr=devnull)
            
            node_version = subprocess.check_output(["node", "--version"], universal_newlines=True).strip()
            
            # Check for Transformers.js or other WebGPU packages
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
                        "environment": "node",
                        "mode": "node_packages",
                        "node_version": node_version,
                        "dependencies": {
                            "transformers_js": has_transformers_js,
                            "webgpu_types": has_webgpu
                        }
                    }
                    
                    # Check for Transformers.js version if available
                    if has_transformers_js:
                        try:
                            # Try to get transformers.js version
                            pkg_info = subprocess.check_output(["npm", "view", "@xenova/transformers", "version"], universal_newlines=True).strip()
                            self._details[WEBGPU]["transformers_js_version"] = pkg_info
                        except:
                            pass
                    
                    logger.info("WebGPU support detected via Node.js packages")
                    return
            except (subprocess.SubprocessError, json.JSONDecodeError, FileNotFoundError):
                pass
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        
        # Check for ONNX export capabilities
        try:
            import torch
            import onnx
            self._hardware_info[WEBGPU] = True
            self._details[WEBGPU] = {
                "environment": "node",
                "mode": "onnx_export",
                "python_export_capability": {
                    "torch": torch.__version__,
                    "onnx": onnx.__version__
                }
            }
            logger.info("WebGPU support detected via Python ONNX export capabilities")
            return
        except ImportError:
            pass
            
        # Final details
        self._details[WEBGPU] = {
            "reason": "WebGPU not available",
            "environment": "node"
        }
    
    def _detect_qualcomm(self):
        """Detect Qualcomm AI capabilities"""
        self._hardware_info[QUALCOMM] = False
        
        # Check for Qualcomm AI Engine Direct
        try:
            # Check for QNN_SDK environment variable
            if "QNN_SDK" in os.environ:
                self._hardware_info[QUALCOMM] = True
                self._details[QUALCOMM] = {
                    "qnn_sdk": os.environ["QNN_SDK"],
                    "backend": "QNN"
                }
                return
                
            # Check for QUALCOMM_SDK environment variable
            if "QUALCOMM_SDK" in os.environ:
                self._hardware_info[QUALCOMM] = True
                self._details[QUALCOMM] = {
                    "qualcomm_sdk": os.environ["QUALCOMM_SDK"],
                    "backend": os.environ.get("QUALCOMM_SDK_TYPE", "Unknown")
                }
                return
                
            # Try to import qnn modules if available
            try:
                # Check if the module exists before trying to import it
                if importlib.util.find_spec("qti.aisw.dlc_utils") is not None:
                    import qti.aisw.dlc_utils as dlc_utils
                    
                    self._hardware_info[QUALCOMM] = True
                    self._details[QUALCOMM] = {
                        "qnn_available": True,
                        "backend": "QNN"
                    }
                    return
            except ImportError:
                pass
                
            # Try checking for SNPE
            try:
                if importlib.util.find_spec("snpe") is not None:
                    import snpe
                    
                    self._hardware_info[QUALCOMM] = True
                    self._details[QUALCOMM] = {
                        "snpe_available": True,
                        "backend": "SNPE"
                    }
                    return
            except ImportError:
                pass
                
            # Check if qnn_wrapper is available
            try:
                if importlib.util.find_spec("qnn_wrapper") is not None:
                    self._hardware_info[QUALCOMM] = True
                    self._details[QUALCOMM] = {
                        "qnn_wrapper_available": True,
                        "backend": "QNN Wrapper"
                    }
                    return
            except ImportError:
                pass
                
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
                return
            except (subprocess.SubprocessError, FileNotFoundError):
                pass
                
            # If we reach here, no Qualcomm AI SDK found
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
        elif self.is_available(QUALCOMM):
            return QUALCOMM
        elif self.is_available(WEBGPU):
            return WEBGPU
        elif self.is_available(WEBNN):
            return WEBNN
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
        # Default priority: CUDA > ROCm > MPS > OpenVINO > QUALCOMM > WEBGPU > WEBNN > CPU
        if priority_list is None:
            priority_list = [CUDA, ROCM, MPS, OPENVINO, QUALCOMM, WEBGPU, WEBNN, CPU]
        
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
        print(f"PyTorch device: {self.get_torch_device()}")
        
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

def get_model_hardware_compatibility(model_name: str) -> Dict[str, bool]:
    """
    Determine hardware compatibility for a specific model.
    
    Args:
        model_name: Name/type of the model to check
        
    Returns:
        Dict indicating which hardware platforms are compatible
    """
    # Detect available hardware
    detector = HardwareDetector()
    
    # Base compatibility that's always available
    compatibility = {
        "cpu": True,
        "cuda": detector.is_available(CUDA),
        "rocm": detector.is_available(ROCM),
        "mps": detector.is_available(MPS),
        "openvino": detector.is_available(OPENVINO),
        "qualcomm": detector.is_available(QUALCOMM),
        "webnn": detector.is_available(WEBNN),
        "webgpu": detector.is_available(WEBGPU)
    }
    
    # Special cases for specific model families
    model_name = model_name.lower()
    
    # Multimodal models like LLaVA may have limited support
    if "llava" in model_name:
        compatibility["mps"] = False  # Limited MPS support for LLaVA
        compatibility["webnn"] = False  # Limited WebNN support for LLaVA
        compatibility["webgpu"] = False  # Limited WebGPU support for LLaVA
    
    # Audio models have limited web support
    if any(audio_model in model_name for audio_model in ["whisper", "wav2vec2", "clap", "hubert"]):
        # Audio models have limited but improving web support
        compatibility["webnn"] = compatibility["webnn"] and "WEBNN_AUDIO_SUPPORT" in os.environ
        compatibility["webgpu"] = compatibility["webgpu"] and "WEBGPU_AUDIO_SUPPORT" in os.environ
    
    # LLMs may have limited web support due to size
    if any(llm in model_name for llm in ["llama", "gpt", "falcon", "mixtral", "qwen"]):
        compatibility["webnn"] = compatibility["webnn"] and "WEBNN_LLM_SUPPORT" in os.environ
        compatibility["webgpu"] = compatibility["webgpu"] and "WEBGPU_LLM_SUPPORT" in os.environ
        
    return compatibility

def get_hardware_detection_code() -> str:
    """
    Generate hardware detection code that can be inserted into templates.
    Returns Python code as a string.
    """
    code = """
# Hardware Detection
import os
import importlib.util

# Try to import torch first (needed for CUDA/ROCm/MPS)
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Initialize hardware capability flags
HAS_CUDA = False
HAS_ROCM = False
HAS_MPS = False
HAS_OPENVINO = False
HAS_WEBNN = False
HAS_WEBGPU = False
HAS_QUALCOMM = False

# CUDA detection
if HAS_TORCH:
    HAS_CUDA = torch.cuda.is_available()
    
    # ROCm detection
    if HAS_CUDA and hasattr(torch, '_C') and hasattr(torch._C, '_rocm_version'):
        HAS_ROCM = True
    elif 'ROCM_HOME' in os.environ:
        HAS_ROCM = True
    
    # Apple MPS detection
    if hasattr(torch, "mps") and hasattr(torch.mps, "is_available"):
        HAS_MPS = torch.mps.is_available()

# OpenVINO detection
HAS_OPENVINO = importlib.util.find_spec("openvino") is not None

# WebNN detection (browser API)
HAS_WEBNN = (
    importlib.util.find_spec("webnn") is not None or 
    importlib.util.find_spec("webnn_js") is not None or
    "WEBNN_AVAILABLE" in os.environ or
    "WEBNN_SIMULATION" in os.environ
)

# WebGPU detection (browser API)
HAS_WEBGPU = (
    importlib.util.find_spec("webgpu") is not None or
    importlib.util.find_spec("wgpu") is not None or
    "WEBGPU_AVAILABLE" in os.environ or
    "WEBGPU_SIMULATION" in os.environ
)

# Qualcomm detection
HAS_QUALCOMM = (
    importlib.util.find_spec("qnn_wrapper") is not None or
    importlib.util.find_spec("qti") is not None or
    "QUALCOMM_SDK" in os.environ
)

# Hardware detection function for comprehensive hardware info
def check_hardware():
    '''Check available hardware and return capabilities.'''
    capabilities = {
        "cpu": True,
        "cuda": HAS_CUDA,
        "rocm": HAS_ROCM,
        "mps": HAS_MPS,
        "openvino": HAS_OPENVINO,
        "qualcomm": HAS_QUALCOMM,
        "webnn": HAS_WEBNN,
        "webgpu": HAS_WEBGPU
    }
    
    # CUDA details if available
    if HAS_CUDA and HAS_TORCH:
        capabilities["cuda_devices"] = torch.cuda.device_count()
        capabilities["cuda_version"] = torch.version.cuda if hasattr(torch.version, 'cuda') else None
    
    return capabilities

# Get hardware capabilities
HW_CAPABILITIES = check_hardware()

# Get appropriate PyTorch device
if HAS_CUDA:
    DEVICE = "cuda"
elif HAS_ROCM:
    DEVICE = "cuda"  # ROCm uses CUDA API in PyTorch
elif HAS_MPS:
    DEVICE = "mps"
else:
    DEVICE = "cpu"
"""
    return code

def detect_browser_features() -> Dict[str, Any]:
    """
    Detect browser-specific features and capabilities for web platform testing.
    This function specifically checks for WebNN and WebGPU support in browser environments.
    
    Returns:
        Dictionary with detected browser features
    """
    features = {
        "running_in_browser": False,
        "webnn_available": False,
        "webgpu_available": False,
        "environment": "node"
    }
    
    # Check if we're running in a browser environment
    try:
        # This would only work if executed in a browser environment like Pyodide
        js_window = eval("window")
        features["running_in_browser"] = True
        features["environment"] = "browser"
        
        # Detect browser information if possible
        try:
            features["browser"] = {
                "user_agent": eval("navigator.userAgent"),
                "platform": eval("navigator.platform"),
                "language": eval("navigator.language")
            }
            
            # Detect browser type
            user_agent = eval("navigator.userAgent.toLowerCase()")
            
            if "firefox" in user_agent:
                features["browser"]["type"] = "firefox"
                features["browser"]["is_firefox"] = True
            elif "chrome" in user_agent:
                features["browser"]["type"] = "chrome"
                features["browser"]["is_chrome"] = True
            elif "safari" in user_agent:
                features["browser"]["type"] = "safari"
                features["browser"]["is_safari"] = True
            elif "edg" in user_agent:
                features["browser"]["type"] = "edge"
                features["browser"]["is_edge"] = True
        except:
            features["browser"] = {"detection_error": "Could not access navigator properties"}
            
        # Check for WebNN API
        try:
            has_webnn = eval("'ml' in navigator && 'NeuralNetwork' in navigator.ml")
            features["webnn_available"] = has_webnn
            
            if has_webnn:
                # Try to get WebNN capabilities if possible
                try:
                    features["webnn_capabilities"] = {
                        "has_gpu": eval("navigator.ml.getNeuralNetworkContext().hasGPU()"),
                        "preferred_backend": eval("navigator.ml.getNeuralNetworkContext().getPreferredBackend()")
                    }
                except:
                    features["webnn_capabilities"] = {"detection_error": "Could not access WebNN capabilities"}
        except:
            features["webnn_available"] = False
            
        # Check for WebGPU API
        try:
            has_webgpu = eval("'gpu' in navigator")
            features["webgpu_available"] = has_webgpu
            
            if has_webgpu:
                # Try to get WebGPU adapter information if possible
                try:
                    # Note: This is asynchronous in reality, but we can't easily handle that here
                    features["webgpu_capabilities"] = {
                        "adapter_available": eval("navigator.gpu.requestAdapter() !== null")
                    }
                except:
                    features["webgpu_capabilities"] = {"detection_error": "Could not access WebGPU capabilities"}
        except:
            features["webgpu_available"] = False
            
        # Get available memory if possible
        try:
            if eval("'performance' in window && 'memory' in performance"):
                features["available_memory_mb"] = eval("performance.memory.totalJSHeapSize / (1024*1024)")
                features["memory_limit_mb"] = eval("performance.memory.jsHeapSizeLimit / (1024*1024)")
        except:
            # Not all browsers expose this API
            pass
            
        # Get GPU information if possible
        try:
            canvas = eval("document.createElement('canvas')")
            gl = eval("canvas.getContext('webgl') || canvas.getContext('experimental-webgl')")
            
            if gl:
                debugInfo = eval("gl.getExtension('WEBGL_debug_renderer_info')")
                if debugInfo:
                    features["gpu_info"] = {
                        "vendor": eval("gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL)"),
                        "renderer": eval("gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL)")
                    }
        except:
            features["gpu_info"] = {"detection_error": "Could not access WebGL GPU information"}
            
    except:
        # Not running in a browser
        features["running_in_browser"] = False
        
        # Check for simulation mode
        if os.environ.get("WEBNN_SIMULATION") == "1":
            features["webnn_available"] = True
            features["environment"] = "simulation"
        
        if os.environ.get("WEBGPU_SIMULATION") == "1":
            features["webgpu_available"] = True
            features["environment"] = "simulation"
            
    return features


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, 
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create detector and output results
    detector = HardwareDetector(cache_file="hardware_detection_cache.json")
    detector.print_summary(detailed=True)
    
    # Create results object
    results = {
        "hardware": detector.get_available_hardware(),
        "details": detector.get_hardware_details(),
        "errors": detector.get_errors(),
        "best_available": detector.get_best_available_hardware(),
        "torch_device": detector.get_torch_device()
    }
    
    # Export results to JSON
    with open("hardware_detection_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults exported to hardware_detection_results.json")