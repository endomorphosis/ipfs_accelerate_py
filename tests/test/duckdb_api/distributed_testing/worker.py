#!/usr/bin/env python3
"""
Distributed Testing Framework - Worker Node

This module implements the worker node for the distributed testing framework,
responsible for executing tasks assigned by the coordinator and reporting results.

Core responsibilities:
- Hardware capability detection
- Registration with coordinator
- Task execution
- Result reporting
- Heartbeat and health monitoring

Usage:
    python worker.py --coordinator http://localhost:8080 --api-key YOUR_API_KEY
"""

import os
import sys
import json
import time
import uuid
import socket
import platform
import asyncio
import logging
import argparse
import threading
import traceback
import importlib
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("worker")

# Try to import optional dependencies
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    logger.warning("psutil not available. Limited hardware detection.")
    PSUTIL_AVAILABLE = False

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    logger.error("websockets not available. Worker cannot function.")
    WEBSOCKETS_AVAILABLE = False

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    logger.warning("GPUtil not available. Limited GPU detection.")
    GPUTIL_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not available. Limited ML capabilities.")
    TORCH_AVAILABLE = False

try:
    import selenium
    from selenium import webdriver
    SELENIUM_AVAILABLE = True
except ImportError:
    logger.warning("Selenium not available. Browser tests unavailable.")
    SELENIUM_AVAILABLE = False

# Add parent directory to path to import modules from parent
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Worker states
WORKER_STATE_INITIALIZING = "initializing"
WORKER_STATE_CONNECTING = "connecting"
WORKER_STATE_REGISTERING = "registering"
WORKER_STATE_ACTIVE = "active"
WORKER_STATE_BUSY = "busy"
WORKER_STATE_DISCONNECTED = "disconnected"
WORKER_STATE_ERROR = "error"

# Task states
TASK_STATE_RECEIVED = "received"
TASK_STATE_RUNNING = "running"
TASK_STATE_COMPLETED = "completed"
TASK_STATE_FAILED = "failed"


class HardwareDetector:
    """Detects hardware capabilities of the worker node."""
    
    def __init__(self):
        """Initialize hardware detector."""
        self.capabilities = {}
        self.detect_hardware()
    
    def detect_hardware(self):
        """Detect hardware capabilities."""
        self.capabilities = {
            "cpu": self._detect_cpu(),
            "memory": self._detect_memory(),
            "gpu": self._detect_gpu(),
            "platform": self._detect_platform(),
            "browsers": self._detect_browsers(),
            "network": self._detect_network(),
            "hardware_types": []
        }
        
        # Determine hardware types
        hardware_types = []
        
        if self.capabilities["cpu"]["count"] > 0:
            hardware_types.append("cpu")
            
        if self.capabilities["gpu"]["count"] > 0:
            for gpu in self.capabilities["gpu"]["devices"]:
                if "nvidia" in gpu["brand"].lower():
                    hardware_types.append("cuda")
                elif "amd" in gpu["brand"].lower():
                    hardware_types.append("rocm")
                elif "intel" in gpu["brand"].lower():
                    hardware_types.append("oneapi")
                    
            if any("nvidia" in gpu["brand"].lower() for gpu in self.capabilities["gpu"]["devices"]):
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    hardware_types.append("torch_cuda")
                    
            if any("amd" in gpu["brand"].lower() for gpu in self.capabilities["gpu"]["devices"]):
                if TORCH_AVAILABLE and hasattr(torch, 'xpu') and torch.xpu.is_available():
                    hardware_types.append("torch_rocm")
        
        # Check for Apple Silicon
        if platform.system() == "Darwin" and platform.processor() == "arm":
            hardware_types.append("mps")
            if TORCH_AVAILABLE and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                hardware_types.append("torch_mps")
                
        # Check for browser hardware acceleration
        if "chrome" in self.capabilities["browsers"]:
            hardware_types.append("webgpu")
            hardware_types.append("webnn")
            
        if "edge" in self.capabilities["browsers"]:
            hardware_types.append("webnn")
            hardware_types.append("webgpu")
            
        if "firefox" in self.capabilities["browsers"]:
            hardware_types.append("webgpu")
        
        # Remove duplicates and store
        self.capabilities["hardware_types"] = list(set(hardware_types))
        
        # Add memory in GB for easy filtering
        self.capabilities["memory_gb"] = self.capabilities["memory"]["total_gb"]
        
        # Add CUDA compute capability if available
        if "cuda" in hardware_types and TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                # Get compute capability of first CUDA device
                major, minor = torch.cuda.get_device_capability(0)
                self.capabilities["cuda_compute"] = float(f"{major}.{minor}")
            except Exception as e:
                logger.warning(f"Could not detect CUDA compute capability: {e}")
        
        logger.info(f"Detected hardware types: {self.capabilities['hardware_types']}")
        
        return self.capabilities
    
    def _detect_cpu(self) -> Dict[str, Any]:
        """Detect CPU capabilities."""
        cpu_info = {
            "count": os.cpu_count() or 0,
            "brand": "Unknown",
            "architecture": platform.machine(),
            "features": []
        }
        
        if PSUTIL_AVAILABLE:
            try:
                cpu_freq = psutil.cpu_freq()
                if cpu_freq:
                    cpu_info["frequency_mhz"] = cpu_freq.current
                    
                cpu_info["physical_cores"] = psutil.cpu_count(logical=False) or 0
                cpu_info["logical_cores"] = psutil.cpu_count(logical=True) or 0
            except Exception as e:
                logger.warning(f"Error detecting CPU details with psutil: {e}")
        
        # Try to get CPU brand from platform info
        if platform.system() == "Linux":
            try:
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if line.startswith("model name"):
                            cpu_info["brand"] = line.split(":", 1)[1].strip()
                            break
            except Exception as e:
                logger.warning(f"Error reading /proc/cpuinfo: {e}")
        elif platform.system() == "Darwin":  # macOS
            try:
                result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], 
                                      capture_output=True, text=True, check=True)
                cpu_info["brand"] = result.stdout.strip()
            except Exception as e:
                logger.warning(f"Error getting CPU brand on macOS: {e}")
        elif platform.system() == "Windows":
            try:
                result = subprocess.run(["wmic", "cpu", "get", "name"], 
                                      capture_output=True, text=True, check=True)
                lines = result.stdout.strip().split("\n")
                if len(lines) > 1:
                    cpu_info["brand"] = lines[1].strip()
            except Exception as e:
                logger.warning(f"Error getting CPU brand on Windows: {e}")
                
        # Detect features
        if platform.system() == "Linux":
            try:
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if line.startswith("flags"):
                            features = line.split(":", 1)[1].strip().split()
                            # Look for specific features
                            if "avx" in features:
                                cpu_info["features"].append("avx")
                            if "avx2" in features:
                                cpu_info["features"].append("avx2")
                            if "sse4_1" in features:
                                cpu_info["features"].append("sse4.1")
                            if "sse4_2" in features:
                                cpu_info["features"].append("sse4.2")
                            break
            except Exception as e:
                logger.warning(f"Error reading CPU features: {e}")
        
        return cpu_info
    
    def _detect_memory(self) -> Dict[str, Any]:
        """Detect memory capabilities."""
        memory_info = {
            "total_gb": 0,
            "available_gb": 0
        }
        
        if PSUTIL_AVAILABLE:
            try:
                memory = psutil.virtual_memory()
                memory_info["total_gb"] = round(memory.total / (1024 ** 3), 2)
                memory_info["available_gb"] = round(memory.available / (1024 ** 3), 2)
            except Exception as e:
                logger.warning(f"Error detecting memory details: {e}")
        
        return memory_info
    
    def _detect_gpu(self) -> Dict[str, Any]:
        """Detect GPU capabilities."""
        gpu_info = {
            "count": 0,
            "devices": []
        }
        
        # Try PyTorch first for CUDA devices
        if TORCH_AVAILABLE:
            try:
                if torch.cuda.is_available():
                    gpu_info["count"] = torch.cuda.device_count()
                    
                    for i in range(gpu_info["count"]):
                        device_info = {
                            "id": i,
                            "brand": torch.cuda.get_device_name(i),
                            "compute_capability": None,
                            "memory_gb": None,
                            "type": "cuda"
                        }
                        
                        try:
                            major, minor = torch.cuda.get_device_capability(i)
                            device_info["compute_capability"] = f"{major}.{minor}"
                        except Exception:
                            pass
                            
                        try:
                            # Get available memory for cards that report it
                            mem_info = torch.cuda.get_device_properties(i).total_memory
                            device_info["memory_gb"] = round(mem_info / (1024 ** 3), 2)
                        except Exception:
                            pass
                            
                        gpu_info["devices"].append(device_info)
                        
                # Check for MPS (Apple Silicon)
                if platform.system() == "Darwin" and platform.processor() == "arm":
                    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        # This is Apple Silicon with MPS
                        device_info = {
                            "id": gpu_info["count"],
                            "brand": "Apple Silicon",
                            "type": "mps"
                        }
                        gpu_info["devices"].append(device_info)
                        gpu_info["count"] += 1
                        
                # Check for ROCm (AMD)
                if hasattr(torch, 'xpu') and torch.xpu.is_available():
                    rocm_count = torch.xpu.device_count()
                    for i in range(rocm_count):
                        device_info = {
                            "id": gpu_info["count"] + i,
                            "brand": f"AMD GPU {i}",
                            "type": "rocm"
                        }
                        gpu_info["devices"].append(device_info)
                    gpu_info["count"] += rocm_count
            
            except Exception as e:
                logger.warning(f"Error detecting GPUs with PyTorch: {e}")
        
        # Try GPUtil for NVIDIA GPUs
        if GPUTIL_AVAILABLE and gpu_info["count"] == 0:
            try:
                gpus = GPUtil.getGPUs()
                gpu_info["count"] = len(gpus)
                
                for i, gpu in enumerate(gpus):
                    device_info = {
                        "id": i,
                        "brand": gpu.name,
                        "memory_gb": round(gpu.memoryTotal / 1024, 2),
                        "type": "cuda"
                    }
                    gpu_info["devices"].append(device_info)
            except Exception as e:
                logger.warning(f"Error detecting GPUs with GPUtil: {e}")
        
        # Check for GPUs using basic system commands if none found so far
        if gpu_info["count"] == 0:
            if platform.system() == "Linux":
                try:
                    # Check for NVIDIA GPUs with nvidia-smi
                    result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        lines = result.stdout.strip().split("\n")
                        for i, line in enumerate(lines):
                            if not line.strip():
                                continue
                            parts = line.split(",")
                            if len(parts) >= 2:
                                name = parts[0].strip()
                                mem_str = parts[1].strip()
                                memory_gb = None
                                if "MiB" in mem_str:
                                    mem_val = float(mem_str.replace("MiB", "").strip())
                                    memory_gb = round(mem_val / 1024, 2)
                                
                                device_info = {
                                    "id": i,
                                    "brand": name,
                                    "memory_gb": memory_gb,
                                    "type": "cuda"
                                }
                                gpu_info["devices"].append(device_info)
                                
                        gpu_info["count"] = len(gpu_info["devices"])
                except Exception:
                    pass
                    
                if gpu_info["count"] == 0:
                    try:
                        # Check for AMD GPUs with rocm-smi
                        result = subprocess.run(["rocm-smi", "--showproductname"], 
                                              capture_output=True, text=True)
                        if result.returncode == 0:
                            lines = result.stdout.strip().split("\n")
                            gpu_names = []
                            for line in lines:
                                if "GPU[" in line and ":" in line:
                                    name = line.split(":", 1)[1].strip()
                                    gpu_names.append(name)
                                    
                            for i, name in enumerate(gpu_names):
                                device_info = {
                                    "id": i,
                                    "brand": name,
                                    "type": "rocm"
                                }
                                gpu_info["devices"].append(device_info)
                                
                            gpu_info["count"] = len(gpu_info["devices"])
                    except Exception:
                        pass
            
            elif platform.system() == "Darwin":
                # On macOS, check for Apple Silicon
                if platform.processor() == "arm":
                    device_info = {
                        "id": 0,
                        "brand": "Apple Silicon",
                        "type": "mps"
                    }
                    gpu_info["devices"].append(device_info)
                    gpu_info["count"] = 1
        
        return gpu_info
    
    def _detect_platform(self) -> Dict[str, Any]:
        """Detect platform information."""
        platform_info = {
            "system": platform.system(),
            "version": platform.version(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
            "hostname": socket.gethostname()
        }
        
        if PSUTIL_AVAILABLE:
            try:
                boot_time = datetime.fromtimestamp(psutil.boot_time())
                platform_info["boot_time"] = boot_time.isoformat()
                platform_info["uptime_seconds"] = (datetime.now() - boot_time).total_seconds()
            except Exception as e:
                logger.warning(f"Error getting boot time: {e}")
        
        return platform_info
    
    def _detect_browsers(self) -> List[str]:
        """Detect available browsers."""
        browsers = []
        
        if not SELENIUM_AVAILABLE:
            logger.warning("Selenium not available, skipping browser detection")
            return browsers
        
        # Check for Chrome
        try:
            from selenium.webdriver.chrome.service import Service as ChromeService
            from selenium.webdriver.chrome.options import Options as ChromeOptions
            
            # Just check if the class is available
            chrome_options = ChromeOptions()
            browsers.append("chrome")
        except Exception:
            pass
        
        # Check for Firefox
        try:
            from selenium.webdriver.firefox.service import Service as FirefoxService
            from selenium.webdriver.firefox.options import Options as FirefoxOptions
            
            # Just check if the class is available
            firefox_options = FirefoxOptions()
            browsers.append("firefox")
        except Exception:
            pass
        
        # Check for Edge
        try:
            from selenium.webdriver.edge.service import Service as EdgeService
            from selenium.webdriver.edge.options import Options as EdgeOptions
            
            # Just check if the class is available
            edge_options = EdgeOptions()
            browsers.append("edge")
        except Exception:
            pass
        
        # Check for Safari
        if platform.system() == "Darwin":  # macOS only
            try:
                from selenium.webdriver.safari.options import Options as SafariOptions
                
                # Just check if the class is available
                safari_options = SafariOptions()
                browsers.append("safari")
            except Exception:
                pass
        
        return browsers
    
    def _detect_network(self) -> Dict[str, Any]:
        """Detect network capabilities."""
        network_info = {
            "hostname": socket.gethostname(),
            "interfaces": []
        }
        
        if PSUTIL_AVAILABLE:
            try:
                network_addrs = psutil.net_if_addrs()
                for interface, addrs in network_addrs.items():
                    interface_info = {"name": interface, "addresses": []}
                    for addr in addrs:
                        if addr.family == socket.AF_INET:  # IPv4
                            interface_info["addresses"].append({
                                "address": addr.address,
                                "netmask": addr.netmask,
                                "type": "ipv4"
                            })
                        elif addr.family == socket.AF_INET6:  # IPv6
                            interface_info["addresses"].append({
                                "address": addr.address,
                                "netmask": addr.netmask,
                                "type": "ipv6"
                            })
                    
                    if interface_info["addresses"]:
                        network_info["interfaces"].append(interface_info)
            except Exception as e:
                logger.warning(f"Error detecting network interfaces: {e}")
        
        return network_info
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get hardware capabilities."""
        return self.capabilities


class TaskRunner:
    """Runs tasks assigned by the coordinator."""
    
    def __init__(self, work_dir: Optional[str] = None):
        """Initialize task runner.
        
        Args:
            work_dir: Working directory for tasks
        """
        self.work_dir = work_dir or os.path.abspath("./worker_tasks")
        os.makedirs(self.work_dir, exist_ok=True)
        
        self.current_task = None
        self.current_task_state = None
        self.task_lock = threading.Lock()
        self.task_result = None
        self.task_exception = None
        self.task_thread = None
        self.task_stop_event = threading.Event()
        
        self.hardware_detector = HardwareDetector()
        self.capabilities = self.hardware_detector.get_capabilities()
        
        logger.info(f"Task runner initialized with work directory: {self.work_dir}")
    
    def run_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Run a task.
        
        Args:
            task: Task configuration
            
        Returns:
            Dict containing task results
        """
        with self.task_lock:
            if self.current_task is not None:
                raise RuntimeError("Task already running")
                
            self.current_task = task
            self.current_task_state = TASK_STATE_RECEIVED
            self.task_result = None
            self.task_exception = None
            self.task_stop_event.clear()
        
        # Determine task type
        task_type = task.get("type", "benchmark")
        task_id = task.get("task_id", "unknown")
        
        logger.info(f"Running task {task_id} of type {task_type}")
        
        try:
            start_time = time.time()
            
            # Update task state
            with self.task_lock:
                self.current_task_state = TASK_STATE_RUNNING
            
            # Run task based on type
            if task_type == "benchmark":
                result = self._run_benchmark_task(task)
            elif task_type == "test":
                result = self._run_test_task(task)
            elif task_type == "command":
                result = self._run_command_task(task)
            else:
                raise ValueError(f"Unknown task type: {task_type}")
                
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Prepare result with metrics
            task_result = {
                "task_id": task_id,
                "success": True,
                "execution_time": execution_time,
                "results": result,
                "metadata": {
                    "start_time": datetime.fromtimestamp(start_time).isoformat(),
                    "end_time": datetime.fromtimestamp(end_time).isoformat(),
                    "execution_time": execution_time,
                    "hardware_metrics": self._get_hardware_metrics(),
                    "attempt": task.get("attempts", 1)
                }
            }
            
            # Update task state and result
            with self.task_lock:
                self.current_task_state = TASK_STATE_COMPLETED
                self.task_result = task_result
                self.current_task = None
                
            logger.info(f"Task {task_id} completed in {execution_time:.2f}s")
            return task_result
            
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            
            error_message = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Error running task {task_id}: {error_message}")
            traceback.print_exc()
            
            # Prepare error result
            task_result = {
                "task_id": task_id,
                "success": False,
                "error": error_message,
                "execution_time": execution_time,
                "metadata": {
                    "start_time": datetime.fromtimestamp(start_time).isoformat(),
                    "end_time": datetime.fromtimestamp(end_time).isoformat(),
                    "execution_time": execution_time,
                    "hardware_metrics": self._get_hardware_metrics(),
                    "attempt": task.get("attempts", 1),
                    "traceback": traceback.format_exc(),
                    "max_retries": task.get("config", {}).get("max_retries", 3)
                }
            }
            
            # Update task state and result
            with self.task_lock:
                self.current_task_state = TASK_STATE_FAILED
                self.task_result = task_result
                self.task_exception = e
                self.current_task = None
                
            return task_result
    
    def _run_benchmark_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Run a benchmark task.
        
        Args:
            task: Task configuration
            
        Returns:
            Dict containing benchmark results
        """
        config = task.get("config", {})
        model_name = config.get("model")
        
        if not model_name:
            raise ValueError("Model name not specified in benchmark task")
            
        batch_sizes = config.get("batch_sizes", [1])
        precision = config.get("precision", "fp16")
        iterations = config.get("iterations", 10)
        
        logger.info(f"Running benchmark for model {model_name} with {len(batch_sizes)} batch sizes")
        
        # Prepare results
        results = {
            "model": model_name,
            "precision": precision,
            "iterations": iterations,
            "batch_sizes": {}
        }
        
        # Run benchmark for each batch size
        for batch_size in batch_sizes:
            logger.info(f"Benchmarking with batch size {batch_size}")
            
            # Simulate benchmark execution
            batch_result = self._simulate_benchmark(model_name, batch_size, precision, iterations)
            results["batch_sizes"][str(batch_size)] = batch_result
            
            # Check if task should be stopped
            if self.task_stop_event.is_set():
                logger.warning("Benchmark task stopped")
                break
        
        return results
    
    def _run_test_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Run a test task.
        
        Args:
            task: Task configuration
            
        Returns:
            Dict containing test results
        """
        config = task.get("config", {})
        test_file = config.get("test_file")
        test_args = config.get("test_args", [])
        
        if not test_file:
            raise ValueError("Test file not specified in test task")
            
        logger.info(f"Running test {test_file} with args {test_args}")
        
        # Determine if test file is a Python module or a script
        if test_file.endswith(".py"):
            # Run as Python script
            cmd = [sys.executable, test_file] + test_args
            return self._run_command(cmd)
        else:
            # Try to import as module
            try:
                module_name = test_file.replace("/", ".").rstrip(".py")
                module = importlib.import_module(module_name)
                
                # Look for test functions
                test_results = {}
                
                for name in dir(module):
                    if name.startswith("test_"):
                        func = getattr(module, name)
                        if callable(func):
                            logger.info(f"Running test function {name}")
                            try:
                                result = func()
                                test_results[name] = {
                                    "success": True,
                                    "result": result
                                }
                            except Exception as e:
                                test_results[name] = {
                                    "success": False,
                                    "error": str(e)
                                }
                
                return {
                    "test_file": test_file,
                    "test_results": test_results,
                    "success": all(r.get("success", False) for r in test_results.values())
                }
            except Exception as e:
                raise RuntimeError(f"Failed to import test module {test_file}: {e}")
    
    def _run_command_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Run a command task.
        
        Args:
            task: Task configuration
            
        Returns:
            Dict containing command results
        """
        config = task.get("config", {})
        command = config.get("command")
        
        if not command:
            raise ValueError("Command not specified in command task")
            
        logger.info(f"Running command: {command}")
        
        if isinstance(command, list):
            return self._run_command(command)
        else:
            # Split command into args
            import shlex
            args = shlex.split(command)
            return self._run_command(args)
    
    def _run_command(self, command: List[str]) -> Dict[str, Any]:
        """Run a command.
        
        Args:
            command: Command to run
            
        Returns:
            Dict containing command results
        """
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.work_dir
            )
            
            stdout, stderr = process.communicate()
            
            return {
                "command": " ".join(command),
                "return_code": process.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "success": process.returncode == 0
            }
        except Exception as e:
            return {
                "command": " ".join(command),
                "error": str(e),
                "success": False
            }
    
    def _simulate_benchmark(self, model_name: str, batch_size: int, 
                          precision: str, iterations: int) -> Dict[str, Any]:
        """Simulate a benchmark run.
        
        This is a placeholder for actual benchmark implementation.
        
        Args:
            model_name: Name of the model to benchmark
            batch_size: Batch size to use
            precision: Precision to use
            iterations: Number of iterations to run
            
        Returns:
            Dict containing benchmark results
        """
        # Get a baseline latency based on the model name and batch size
        if "bert" in model_name.lower():
            base_latency = 10.0
        elif "t5" in model_name.lower():
            base_latency = 20.0
        elif "gpt" in model_name.lower() or "llama" in model_name.lower():
            base_latency = 50.0
        elif "clip" in model_name.lower() or "vit" in model_name.lower():
            base_latency = 15.0
        else:
            base_latency = 25.0
            
        # Adjust latency based on batch size (linear scaling for simplicity)
        latency = base_latency * batch_size
        
        # Adjust latency based on precision
        if precision == "fp16":
            latency *= 0.7
        elif precision == "int8":
            latency *= 0.5
        elif precision == "int4":
            latency *= 0.4
            
        # Add some random variation
        import random
        latency_variance = latency * 0.1
        latencies = [
            max(1.0, latency + random.uniform(-latency_variance, latency_variance))
            for _ in range(iterations)
        ]
        
        # Calculate throughput
        throughput = batch_size / (sum(latencies) / len(latencies)) * 1000
        
        # Simulate memory usage
        if "bert" in model_name.lower():
            memory_base = 500
        elif "t5" in model_name.lower():
            memory_base = 800
        elif "gpt" in model_name.lower() or "llama" in model_name.lower():
            memory_base = 1500
        elif "clip" in model_name.lower() or "vit" in model_name.lower():
            memory_base = 700
        else:
            memory_base = 600
            
        memory_usage = memory_base * batch_size * (1.0 if precision == "fp32" else 
                                               0.5 if precision == "fp16" else 
                                               0.25 if precision == "int8" else 
                                               0.125)
        
        # Simulate run with brief pauses
        for i in range(iterations):
            # Brief pause to simulate work
            time.sleep(latencies[i] / 1000)
            
            # Check if task should be stopped
            if self.task_stop_event.is_set():
                logger.warning("Benchmark iteration stopped")
                break
        
        return {
            "latency_ms": sum(latencies) / len(latencies),
            "latencies_ms": latencies,
            "throughput_items_per_second": throughput,
            "memory_mb": memory_usage,
            "iterations_completed": i + 1
        }
    
    def _get_hardware_metrics(self) -> Dict[str, Any]:
        """Get current hardware metrics.
        
        Returns:
            Dict containing hardware metrics and resource information for dynamic resource management
        """
        metrics = {
            "timestamp": datetime.now().isoformat()
        }
        
        # Add resource metrics for dynamic resource management
        resources = {
            "cpu": {},
            "memory": {},
            "gpu": {}
        }
        
        if PSUTIL_AVAILABLE:
            try:
                # CPU usage and resource information
                metrics["cpu_percent"] = psutil.cpu_percent(interval=0.1)
                metrics["cpu_per_core"] = psutil.cpu_percent(interval=0.1, percpu=True)
                
                # Add CPU resource metrics
                cpu_count = psutil.cpu_count(logical=True)
                cpu_physical = psutil.cpu_count(logical=False)
                cpu_load = [x / 100.0 for x in psutil.getloadavg()] if hasattr(psutil, 'getloadavg') else [0.0, 0.0, 0.0]
                
                resources["cpu"]["cores"] = cpu_count
                resources["cpu"]["physical_cores"] = cpu_physical
                resources["cpu"]["available_cores"] = max(0.1, cpu_count - cpu_load[0])  # Estimate available cores
                resources["cpu"]["load_average"] = cpu_load
                resources["cpu"]["percent_used"] = metrics["cpu_percent"]
                
                # Memory usage and resource information
                memory = psutil.virtual_memory()
                metrics["memory_percent"] = memory.percent
                metrics["memory_used_gb"] = round(memory.used / (1024 ** 3), 2)
                metrics["memory_available_gb"] = round(memory.available / (1024 ** 3), 2)
                
                # Add memory resource metrics
                resources["memory"]["total_mb"] = int(memory.total / (1024 * 1024))
                resources["memory"]["used_mb"] = int(memory.used / (1024 * 1024))
                resources["memory"]["available_mb"] = int(memory.available / (1024 * 1024))
                resources["memory"]["percent_used"] = memory.percent
                
                # Disk usage
                disk = psutil.disk_usage(self.work_dir)
                metrics["disk_percent"] = disk.percent
                metrics["disk_used_gb"] = round(disk.used / (1024 ** 3), 2)
                metrics["disk_free_gb"] = round(disk.free / (1024 ** 3), 2)
            except Exception as e:
                logger.warning(f"Error getting hardware metrics with psutil: {e}")
        
        # GPU metrics
        if GPUTIL_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                metrics["gpu_metrics"] = []
                
                # Track GPU resource info for dynamic resource management
                resources["gpu"]["devices"] = len(gpus)
                resources["gpu"]["available_devices"] = 0  # Will increment for available GPUs
                resources["gpu"]["total_memory_mb"] = 0
                resources["gpu"]["available_memory_mb"] = 0
                
                # Process individual GPUs
                for gpu in gpus:
                    gpu_metrics = {
                        "id": gpu.id,
                        "name": gpu.name,
                        "load_percent": gpu.load * 100,
                        "memory_used_percent": gpu.memoryUtil * 100,
                        "memory_used_mb": gpu.memoryUsed,
                        "memory_total_mb": gpu.memoryTotal,
                        "temperature": gpu.temperature
                    }
                    metrics["gpu_metrics"].append(gpu_metrics)
                    
                    # Update resource tracking for this GPU
                    resources["gpu"]["total_memory_mb"] += gpu.memoryTotal
                    
                    # Calculate available memory
                    available_memory_mb = gpu.memoryTotal - gpu.memoryUsed
                    resources["gpu"]["available_memory_mb"] += available_memory_mb
                    
                    # Consider a GPU "available" if it has less than 70% memory utilization
                    if gpu.memoryUtil < 0.7:
                        resources["gpu"]["available_devices"] += 1
                        
                    # Add per-device info
                    resources["gpu"][f"device_{gpu.id}"] = {
                        "name": gpu.name,
                        "total_memory_mb": gpu.memoryTotal,
                        "available_memory_mb": available_memory_mb,
                        "load_percent": gpu.load * 100,
                        "temperature": gpu.temperature
                    }
            except Exception as e:
                logger.warning(f"Error getting GPU metrics with GPUtil: {e}")
                
        # PyTorch GPU metrics
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                metrics["torch_gpu_metrics"] = []
                cuda_device_count = torch.cuda.device_count()
                
                # Set device count in resources if not already set
                if resources["gpu"].get("devices", 0) == 0:
                    resources["gpu"]["devices"] = cuda_device_count
                    resources["gpu"]["available_devices"] = 0
                    resources["gpu"]["total_memory_mb"] = 0
                    resources["gpu"]["available_memory_mb"] = 0
                
                for i in range(cuda_device_count):
                    torch_gpu_metrics = {
                        "id": i,
                        "name": torch.cuda.get_device_name(i)
                    }
                    
                    # Get memory usage
                    reserved_bytes = 0
                    allocated_bytes = 0
                    
                    if hasattr(torch.cuda, "memory_reserved"):
                        reserved_bytes = torch.cuda.memory_reserved(i)
                        torch_gpu_metrics["memory_reserved_bytes"] = reserved_bytes
                    
                    if hasattr(torch.cuda, "memory_allocated"):
                        allocated_bytes = torch.cuda.memory_allocated(i)
                        torch_gpu_metrics["memory_allocated_bytes"] = allocated_bytes
                    
                    # Get total memory if possible
                    total_memory_bytes = 0
                    try:
                        props = torch.cuda.get_device_properties(i)
                        total_memory_bytes = props.total_memory
                        torch_gpu_metrics["total_memory_bytes"] = total_memory_bytes
                    except Exception:
                        pass
                    
                    # Convert to MB for resource tracking if not already tracked by GPUtil
                    if f"device_{i}" not in resources["gpu"] and total_memory_bytes > 0:
                        total_memory_mb = total_memory_bytes / (1024 * 1024)
                        available_memory_mb = (total_memory_bytes - allocated_bytes) / (1024 * 1024)
                        
                        resources["gpu"]["total_memory_mb"] += total_memory_mb
                        resources["gpu"]["available_memory_mb"] += available_memory_mb
                        
                        # Consider a GPU "available" if it has less than 70% memory utilization
                        if allocated_bytes / total_memory_bytes < 0.7:
                            resources["gpu"]["available_devices"] += 1
                        
                        resources["gpu"][f"device_{i}"] = {
                            "name": torch.cuda.get_device_name(i),
                            "total_memory_mb": total_memory_mb,
                            "available_memory_mb": available_memory_mb,
                            "utilization_percent": (allocated_bytes / total_memory_bytes * 100) if total_memory_bytes > 0 else 0
                        }
                    
                    metrics["torch_gpu_metrics"].append(torch_gpu_metrics)
            except Exception as e:
                logger.warning(f"Error getting PyTorch GPU metrics: {e}")
        
        # Add resources info to metrics
        metrics["resources"] = resources
        
        return metrics
    
    def stop_task(self):
        """Stop the current task."""
        logger.info("Stopping current task")
        self.task_stop_event.set()
        
        if self.task_thread is not None and self.task_thread.is_alive():
            # Wait for thread to finish with timeout
            self.task_thread.join(timeout=5.0)
            if self.task_thread.is_alive():
                logger.warning("Task thread did not stop gracefully")
                
            self.task_thread = None
    
    def is_task_running(self) -> bool:
        """Check if a task is currently running.
        
        Returns:
            True if a task is running, False otherwise
        """
        with self.task_lock:
            return self.current_task is not None
    
    def get_task_status(self) -> Tuple[Optional[Dict[str, Any]], str, Optional[Dict[str, Any]]]:
        """Get the status of the current task.
        
        Returns:
            Tuple containing (task, state, result)
        """
        with self.task_lock:
            return (self.current_task, self.current_task_state, self.task_result)


class WorkerClient:
    """Client for communicating with the coordinator."""
    
    def __init__(self, coordinator_url: str, api_key: str, worker_id: Optional[str] = None,
                reconnect_interval: int = 5, heartbeat_interval: int = 30):
        """Initialize the worker client.
        
        Args:
            coordinator_url: URL of the coordinator server
            api_key: API key for authentication
            worker_id: Worker ID (generated if not provided)
            reconnect_interval: Interval in seconds between reconnection attempts
            heartbeat_interval: Interval in seconds between heartbeats
        """
        if not WEBSOCKETS_AVAILABLE:
            raise RuntimeError("websockets not available, worker cannot function")
            
        self.coordinator_url = coordinator_url
        self.api_key = api_key
        self.worker_id = worker_id or f"worker_{uuid.uuid4()}"
        self.reconnect_interval = reconnect_interval
        self.heartbeat_interval = heartbeat_interval
        
        self.state = WORKER_STATE_INITIALIZING
        self.connected = False
        self.authenticated = False
        self.token = None
        self.websocket = None
        
        self.hardware_detector = HardwareDetector()
        self.capabilities = self.hardware_detector.get_capabilities()
        
        # Initialize task runner
        self.task_runner = TaskRunner()
        
        # Control flags
        self.running = True
        self.should_reconnect = True
        
        # Heartbeat thread
        self.heartbeat_thread = None
        self.heartbeat_stop_event = threading.Event()
        
        # Statistics
        self.stats = {
            "connection_attempts": 0,
            "successful_connections": 0,
            "tasks_received": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_task_time": 0.0,
            "last_connection_time": None,
            "last_heartbeat_time": None
        }
        
        logger.info(f"Worker client initialized with ID: {self.worker_id}")
    
    async def connect(self):
        """Connect to the coordinator and authenticate."""
        if self.websocket:
            logger.warning("Already connected, closing existing connection")
            await self.websocket.close()
            self.websocket = None
            
        self.state = WORKER_STATE_CONNECTING
        self.connected = False
        self.authenticated = False
        
        self.stats["connection_attempts"] += 1
        
        try:
            logger.info(f"Connecting to coordinator at {self.coordinator_url}")
            self.websocket = await websockets.connect(self.coordinator_url)
            self.connected = True
            self.stats["last_connection_time"] = datetime.now()
            
            # Authenticate
            authenticated = await self._authenticate()
            if not authenticated:
                logger.error("Authentication failed")
                await self.websocket.close()
                self.websocket = None
                self.connected = False
                return False
                
            self.authenticated = True
            self.stats["successful_connections"] += 1
            
            # Register worker
            registered = await self._register()
            if not registered:
                logger.error("Registration failed")
                await self.websocket.close()
                self.websocket = None
                self.connected = False
                self.authenticated = False
                return False
                
            self.state = WORKER_STATE_ACTIVE
            logger.info("Connected and registered with coordinator")
            
            # Start heartbeat thread
            self._start_heartbeat_thread()
            
            return True
        except Exception as e:
            logger.error(f"Connection error: {e}")
            if self.websocket:
                await self.websocket.close()
                self.websocket = None
            self.connected = False
            self.authenticated = False
            self.state = WORKER_STATE_ERROR
            return False
    
    async def _authenticate(self) -> bool:
        """Authenticate with the coordinator.
        
        Returns:
            True if authentication is successful, False otherwise
        """
        try:
            # Wait for authentication challenge
            response = await self.websocket.recv()
            data = json.loads(response)
            
            if data.get("type") != "auth_challenge":
                logger.error(f"Expected auth_challenge, got {data.get('type')}")
                return False
                
            challenge_id = data.get("challenge_id")
            
            # Send authentication response
            auth_response = {
                "type": "auth_response",
                "challenge_id": challenge_id,
                "api_key": self.api_key,
                "worker_id": self.worker_id
            }
            
            await self.websocket.send(json.dumps(auth_response))
            
            # Wait for authentication result
            response = await self.websocket.recv()
            data = json.loads(response)
            
            if data.get("type") != "auth_result":
                logger.error(f"Expected auth_result, got {data.get('type')}")
                return False
                
            if not data.get("success", False):
                logger.error(f"Authentication failed: {data.get('error')}")
                return False
                
            # Store token
            self.token = data.get("token")
            
            # Check if worker_id was assigned by the server
            if data.get("worker_id") and data["worker_id"] != self.worker_id:
                logger.info(f"Worker ID assigned by server: {data['worker_id']}")
                self.worker_id = data["worker_id"]
                
            logger.info("Authentication successful")
            return True
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False
    
    async def _register(self) -> bool:
        """Register with the coordinator.
        
        Returns:
            True if registration is successful, False otherwise
        """
        try:
            # Prepare hostname
            hostname = socket.gethostname()
            
            # Get current hardware metrics including resource information
            hardware_metrics = self.task_runner._get_hardware_metrics()
            
            # Send registration request with resource information
            register_request = {
                "type": "register",
                "worker_id": self.worker_id,
                "hostname": hostname,
                "capabilities": self.capabilities,
                "resources": hardware_metrics.get("resources", {}),  # Include resource information
                "tags": {
                    "version": "0.1.0",
                    "py_version": platform.python_version(),
                    "psutil_available": PSUTIL_AVAILABLE,
                    "gputil_available": GPUTIL_AVAILABLE,
                    "torch_available": TORCH_AVAILABLE,
                    "cuda_available": TORCH_AVAILABLE and torch.cuda.is_available()
                }
            }
            
            await self.websocket.send(json.dumps(register_request))
            
            # Wait for registration result
            response = await self.websocket.recv()
            data = json.loads(response)
            
            if data.get("type") != "register_result":
                logger.error(f"Expected register_result, got {data.get('type')}")
                return False
                
            if not data.get("success", False):
                logger.error(f"Registration failed: {data.get('error')}")
                return False
                
            logger.info("Registration successful")
            return True
        except Exception as e:
            logger.error(f"Registration error: {e}")
            return False
    
    async def _send_heartbeat(self) -> bool:
        """Send a heartbeat to the coordinator.
        
        Returns:
            True if heartbeat is successful, False otherwise
        """
        if not self.connected or not self.authenticated:
            logger.warning("Cannot send heartbeat: not connected or authenticated")
            return False
            
        try:
            # Get updated hardware metrics
            hardware_metrics = self.task_runner._get_hardware_metrics()
            
            # Send heartbeat request with resource updates
            heartbeat_request = {
                "type": "heartbeat",
                "worker_id": self.worker_id,
                "timestamp": datetime.now().isoformat(),
                "resources": hardware_metrics.get("resources", {}),  # Include latest resource info
                "hardware_metrics": {
                    "cpu_percent": hardware_metrics.get("cpu_percent", 0),
                    "memory_percent": hardware_metrics.get("memory_percent", 0),
                    "gpu_utilization": next((gpu.get("load_percent", 0) for gpu in hardware_metrics.get("gpu_metrics", [])), 0) if hardware_metrics.get("gpu_metrics") else 0
                }
            }
            
            await self.websocket.send(json.dumps(heartbeat_request))
            
            # Update statistics
            self.stats["last_heartbeat_time"] = datetime.now()
            
            # Wait for heartbeat result
            response = await self.websocket.recv()
            data = json.loads(response)
            
            if data.get("type") != "heartbeat_result":
                logger.warning(f"Expected heartbeat_result, got {data.get('type')}")
                return False
                
            if not data.get("success", False):
                logger.warning(f"Heartbeat failed: {data.get('error')}")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Heartbeat error: {e}")
            return False
    
    def _start_heartbeat_thread(self):
        """Start the heartbeat thread."""
        if self.heartbeat_thread is not None and self.heartbeat_thread.is_alive():
            logger.warning("Heartbeat thread already running")
            return
            
        self.heartbeat_stop_event.clear()
        self.heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True
        )
        self.heartbeat_thread.start()
        logger.info("Heartbeat thread started")
    
    def _heartbeat_loop(self):
        """Heartbeat thread function."""
        while not self.heartbeat_stop_event.is_set():
            if self.connected and self.authenticated:
                try:
                    # Create event loop for async calls
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    # Send heartbeat
                    heartbeat_success = loop.run_until_complete(self._send_heartbeat())
                    if not heartbeat_success:
                        logger.warning("Heartbeat failed")
                except Exception as e:
                    logger.error(f"Error in heartbeat loop: {e}")
                    
            # Wait for next heartbeat interval
            self.heartbeat_stop_event.wait(self.heartbeat_interval)
            
        logger.info("Heartbeat thread stopped")
    
    async def run(self):
        """Run the worker client."""
        while self.running:
            if not self.connected or not self.authenticated:
                # Try to connect
                connected = await self.connect()
                if not connected:
                    # Wait before retrying
                    logger.info(f"Connection failed, retrying in {self.reconnect_interval}s...")
                    await asyncio.sleep(self.reconnect_interval)
                    continue
            
            try:
                # Process messages
                await self._process_messages()
            except websockets.exceptions.ConnectionClosed:
                logger.warning("Connection closed, reconnecting...")
                self.connected = False
                self.authenticated = False
                self.state = WORKER_STATE_DISCONNECTED
                
                if self.should_reconnect:
                    await asyncio.sleep(self.reconnect_interval)
            except Exception as e:
                logger.error(f"Error in message processing: {e}")
                traceback.print_exc()
                
                self.connected = False
                self.authenticated = False
                self.state = WORKER_STATE_ERROR
                
                if self.should_reconnect:
                    await asyncio.sleep(self.reconnect_interval)
        
        # Cleanup
        await self._cleanup()
    
    async def _process_messages(self):
        """Process messages from the coordinator."""
        while self.connected and self.authenticated:
            # Wait for messages
            message = await self.websocket.recv()
            
            try:
                data = json.loads(message)
                message_type = data.get("type")
                
                if message_type == "get_task_result":
                    # Task assignment
                    await self._handle_task_assignment(data)
                elif message_type == "heartbeat_result":
                    # Heartbeat response
                    pass  # Already handled in _send_heartbeat
                elif message_type == "status_update_result":
                    # Status update response
                    pass  # Ignore
                elif message_type == "task_result_result":
                    # Task result response
                    pass  # Ignore
                elif message_type == "error":
                    # Error message
                    logger.error(f"Error from coordinator: {data.get('error')}")
                else:
                    logger.warning(f"Unknown message type: {message_type}")
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON message: {message}")
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                traceback.print_exc()
    
    async def _handle_task_assignment(self, data: Dict[str, Any]):
        """Handle a task assignment from the coordinator.
        
        Args:
            data: Task assignment data
        """
        if not data.get("success", False):
            logger.error(f"Task assignment failed: {data.get('error')}")
            return
            
        task = data.get("task")
        if task is None:
            # No task available
            logger.debug("No task available")
            
            # Request a new task after a short delay
            await asyncio.sleep(5.0)
            await self._request_task()
            return
            
        # Update statistics
        self.stats["tasks_received"] += 1
        
        # Update worker state
        self.state = WORKER_STATE_BUSY
        await self._update_status(WORKER_STATE_BUSY)
        
        # Extract task info
        task_id = task.get("task_id", "unknown")
        task_type = task.get("type", "unknown")
        
        logger.info(f"Received task {task_id} of type {task_type}")
        
        # Run task in a separate thread
        task_thread = threading.Thread(
            target=self._run_task_thread,
            args=(task,),
            daemon=True
        )
        task_thread.start()
    
    def _run_task_thread(self, task: Dict[str, Any]):
        """Run a task in a separate thread.
        
        Args:
            task: Task configuration
        """
        task_id = task.get("task_id", "unknown")
        
        try:
            start_time = time.time()
            
            # Run the task
            result = self.task_runner.run_task(task)
            
            end_time = time.time()
            task_time = end_time - start_time
            
            # Update statistics
            if result.get("success", False):
                self.stats["tasks_completed"] += 1
            else:
                self.stats["tasks_failed"] += 1
                
            self.stats["total_task_time"] += task_time
            
            # Update worker state
            self.state = WORKER_STATE_ACTIVE
            
            # Create event loop for async calls
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Report result
            loop.run_until_complete(self._report_task_result(result))
            
            # Update status
            loop.run_until_complete(self._update_status(WORKER_STATE_ACTIVE))
            
            # Request new task
            loop.run_until_complete(self._request_task())
            
        except Exception as e:
            logger.error(f"Error in task thread for task {task_id}: {e}")
            traceback.print_exc()
            
            # Update worker state
            self.state = WORKER_STATE_ACTIVE
            
            # Create event loop for async calls
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Report error
            loop.run_until_complete(self._report_task_error(task_id, str(e)))
            
            # Update status
            loop.run_until_complete(self._update_status(WORKER_STATE_ACTIVE))
            
            # Request new task
            loop.run_until_complete(self._request_task())
    
    async def _report_task_result(self, result: Dict[str, Any]) -> bool:
        """Report task result to the coordinator.
        
        Args:
            result: Task result
            
        Returns:
            True if reporting is successful, False otherwise
        """
        if not self.connected or not self.authenticated:
            logger.warning("Cannot report result: not connected or authenticated")
            return False
            
        try:
            # Send task result
            task_result = {
                "type": "task_result",
                "worker_id": self.worker_id,
                "task_id": result.get("task_id", "unknown"),
                "success": result.get("success", False),
                "results": result.get("results", {}),
                "metadata": result.get("metadata", {}),
                "error": result.get("error", "")
            }
            
            await self.websocket.send(json.dumps(task_result))
            
            # Wait for response
            response = await self.websocket.recv()
            data = json.loads(response)
            
            if data.get("type") != "task_result_result":
                logger.warning(f"Expected task_result_result, got {data.get('type')}")
                return False
                
            if not data.get("success", False):
                logger.warning(f"Task result reporting failed: {data.get('error')}")
                return False
                
            logger.info(f"Task result reported for task {result.get('task_id', 'unknown')}")
            return True
        except Exception as e:
            logger.error(f"Error reporting task result: {e}")
            return False
    
    async def _report_task_error(self, task_id: str, error: str) -> bool:
        """Report task error to the coordinator.
        
        Args:
            task_id: ID of the task
            error: Error message
            
        Returns:
            True if reporting is successful, False otherwise
        """
        if not self.connected or not self.authenticated:
            logger.warning("Cannot report error: not connected or authenticated")
            return False
            
        try:
            # Send task result with error
            task_result = {
                "type": "task_result",
                "worker_id": self.worker_id,
                "task_id": task_id,
                "success": False,
                "error": error,
                "results": {},
                "metadata": {
                    "start_time": datetime.now().isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "execution_time": 0.0,
                    "hardware_metrics": self.task_runner._get_hardware_metrics()
                }
            }
            
            await self.websocket.send(json.dumps(task_result))
            
            # Wait for response
            response = await self.websocket.recv()
            data = json.loads(response)
            
            if data.get("type") != "task_result_result":
                logger.warning(f"Expected task_result_result, got {data.get('type')}")
                return False
                
            if not data.get("success", False):
                logger.warning(f"Task error reporting failed: {data.get('error')}")
                return False
                
            logger.info(f"Task error reported for task {task_id}")
            return True
        except Exception as e:
            logger.error(f"Error reporting task error: {e}")
            return False
    
    async def _update_status(self, status: str) -> bool:
        """Update worker status with the coordinator.
        
        Args:
            status: New status
            
        Returns:
            True if update is successful, False otherwise
        """
        if not self.connected or not self.authenticated:
            logger.warning("Cannot update status: not connected or authenticated")
            return False
            
        try:
            # Send status update
            status_update = {
                "type": "status_update",
                "worker_id": self.worker_id,
                "status": status
            }
            
            await self.websocket.send(json.dumps(status_update))
            
            # Wait for response
            response = await self.websocket.recv()
            data = json.loads(response)
            
            if data.get("type") != "status_update_result":
                logger.warning(f"Expected status_update_result, got {data.get('type')}")
                return False
                
            if not data.get("success", False):
                logger.warning(f"Status update failed: {data.get('error')}")
                return False
                
            logger.debug(f"Status updated to {status}")
            return True
        except Exception as e:
            logger.error(f"Error updating status: {e}")
            return False
    
    async def _request_task(self) -> bool:
        """Request a task from the coordinator.
        
        Returns:
            True if request is successful, False otherwise
        """
        if not self.connected or not self.authenticated:
            logger.warning("Cannot request task: not connected or authenticated")
            return False
            
        try:
            # Send task request
            task_request = {
                "type": "get_task",
                "worker_id": self.worker_id,
                "capabilities": self.capabilities
            }
            
            await self.websocket.send(json.dumps(task_request))
            return True
        except Exception as e:
            logger.error(f"Error requesting task: {e}")
            return False
    
    async def _cleanup(self):
        """Clean up resources."""
        # Stop heartbeat thread
        if self.heartbeat_thread is not None and self.heartbeat_thread.is_alive():
            self.heartbeat_stop_event.set()
            self.heartbeat_thread.join(timeout=5.0)
            
        # Close WebSocket connection
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception:
                pass
            self.websocket = None
            
        self.connected = False
        self.authenticated = False
        self.state = WORKER_STATE_DISCONNECTED
        
        logger.info("Worker client cleaned up")
    
    async def stop(self):
        """Stop the worker client."""
        logger.info("Stopping worker client")
        self.running = False
        self.should_reconnect = False
        
        # Stop any running task
        if self.task_runner.is_task_running():
            self.task_runner.stop_task()
            
        await self._cleanup()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Distributed Testing Framework Worker")
    
    parser.add_argument("--coordinator", required=True,
                      help="URL of the coordinator server")
    parser.add_argument("--api-key", required=True,
                      help="API key for authentication")
    parser.add_argument("--worker-id", default=None,
                      help="Worker ID (generated if not provided)")
    parser.add_argument("--work-dir", default=None,
                      help="Working directory for tasks")
    parser.add_argument("--reconnect-interval", type=int, default=5,
                      help="Interval in seconds between reconnection attempts")
    parser.add_argument("--heartbeat-interval", type=int, default=30,
                      help="Interval in seconds between heartbeats")
    parser.add_argument("--verbose", action="store_true",
                      help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled")
    
    if not WEBSOCKETS_AVAILABLE:
        logger.error("websockets not available, worker cannot function")
        return 1
    
    # Create worker client
    worker = WorkerClient(
        coordinator_url=args.coordinator,
        api_key=args.api_key,
        worker_id=args.worker_id,
        reconnect_interval=args.reconnect_interval,
        heartbeat_interval=args.heartbeat_interval
    )
    
    # Set up signal handlers
    loop = asyncio.get_event_loop()
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(
            sig,
            lambda: asyncio.create_task(worker.stop())
        )
    
    # Run worker
    try:
        logger.info(f"Starting worker with ID: {worker.worker_id}")
        loop.run_until_complete(worker.run())
        return 0
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        loop.run_until_complete(worker.stop())
        return 130


if __name__ == "__main__":
    sys.exit(main())