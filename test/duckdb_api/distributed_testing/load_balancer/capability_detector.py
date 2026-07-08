#!/usr/bin/env python3
"""
Distributed Testing Framework - Worker Capability Detector

This module implements the capability detection system for worker nodes
in the distributed testing framework.
"""

import os
import sys
import platform
import socket
import json
import logging
import subprocess
import shutil
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import multiprocessing
import importlib
try:
    from importlib import metadata as importlib_metadata
except ImportError:  # pragma: no cover
    import importlib_metadata  # type: ignore
try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover
    psutil = None
try:
    import torch
    import torch.cuda
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

from .models import WorkerCapabilities

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("capability_detector")


class WorkerCapabilityDetector:
    """Detects and maintains information about worker capabilities."""
    
    def __init__(self, worker_id: Optional[str] = None):
        """Initialize the capability detector.
        
        Args:
            worker_id: Unique identifier for this worker, or None to generate one
        """
        self.worker_id = worker_id or self._generate_worker_id()
        self.capabilities: Optional[WorkerCapabilities] = None
        
    def _generate_worker_id(self) -> str:
        """Generate a unique worker ID based on hostname and timestamp."""
        hostname = socket.gethostname()
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"{hostname}_{timestamp}"
    
    def detect_capabilities(self) -> WorkerCapabilities:
        """Detect worker hardware and software capabilities."""
        hostname = socket.gethostname()
        
        # Hardware specs
        hardware_specs = self._detect_hardware_specs()
        
        # Software versions
        software_versions = self._detect_software_versions()
        
        # Supported backends
        supported_backends = self._detect_supported_backends()
        
        # Network bandwidth (estimate)
        network_bandwidth = self._estimate_network_bandwidth()
        
        # Storage capacity
        storage_capacity = self._detect_storage_capacity()
        
        # Accelerators
        available_accelerators = self._detect_available_accelerators()
        
        # System resources
        available_memory = self._detect_available_memory()
        available_disk = self._detect_available_disk()
        cpu_cores = multiprocessing.cpu_count()
        cpu_threads = psutil.cpu_count(logical=True) if psutil else cpu_cores
        
        # Create capabilities object
        self.capabilities = WorkerCapabilities(
            worker_id=self.worker_id,
            hostname=hostname,
            hardware_specs=hardware_specs,
            software_versions=software_versions,
            supported_backends=supported_backends,
            network_bandwidth=network_bandwidth,
            storage_capacity=storage_capacity,
            available_accelerators=available_accelerators,
            available_memory=available_memory,
            available_disk=available_disk,
            cpu_cores=cpu_cores,
            cpu_threads=cpu_threads
        )
        
        logger.info(f"Detected capabilities for worker {self.worker_id}")
        return self.capabilities
    
    def _detect_hardware_specs(self) -> Dict[str, Any]:
        """Detect detailed hardware specifications."""
        specs = {}
        
        # Platform information
        specs["platform"] = platform.platform()
        specs["architecture"] = platform.machine()
        specs["processor"] = platform.processor()
        
        # CPU information
        if psutil:
            cpu_freq = psutil.cpu_freq().current if psutil.cpu_freq() else None
            cpu_physical = psutil.cpu_count(logical=False)
            cpu_logical = psutil.cpu_count(logical=True)
        else:
            cpu_freq = None
            cpu_physical = multiprocessing.cpu_count()
            cpu_logical = multiprocessing.cpu_count()

        specs["cpu"] = {
            "cores_physical": cpu_physical,
            "cores_logical": cpu_logical,
            "frequency_mhz": cpu_freq,
        }
        
        # Memory information
        virtual_memory = psutil.virtual_memory() if psutil else None
        specs["memory"] = {
            "total_gb": (virtual_memory.total / (1024 ** 3)) if virtual_memory else 0.0,
            "available_gb": (virtual_memory.available / (1024 ** 3)) if virtual_memory else 0.0,
        }
        
        # GPU information
        specs["gpu"] = self._detect_gpu_info()
        
        return specs
    
    def _detect_gpu_info(self) -> Dict[str, Any]:
        """Detect GPU information."""
        gpu_info = {}
        
        # PyTorch CUDA information
        if HAS_TORCH and torch.cuda.is_available():
            gpu_info["cuda_available"] = True
            gpu_info["cuda_version"] = torch.version.cuda
            gpu_info["device_count"] = torch.cuda.device_count()
            gpu_info["devices"] = []
            
            for i in range(torch.cuda.device_count()):
                device_props = torch.cuda.get_device_properties(i)
                gpu_info["devices"].append({
                    "name": device_props.name,
                    "compute_capability": f"{device_props.major}.{device_props.minor}",
                    "total_memory_gb": device_props.total_memory / (1024 ** 3),
                    "multi_processor_count": device_props.multi_processor_count
                })
        else:
            gpu_info["cuda_available"] = False
        
        # Check for ROCm (AMD) GPU support
        try:
            has_rocm = False
            if HAS_TORCH:
                has_rocm = hasattr(torch, 'version') and hasattr(torch.version, 'hip') and torch.version.hip is not None
            gpu_info["rocm_available"] = has_rocm
        except Exception as e:
            logger.warning(f"Error detecting ROCm: {e}")
            gpu_info["rocm_available"] = False
        
        # Check for MPS (Apple) support
        try:
            has_mps = False
            if HAS_TORCH and hasattr(torch, 'backends') and hasattr(torch.backends, 'mps'):
                has_mps = torch.backends.mps.is_available()
            gpu_info["mps_available"] = has_mps
        except Exception as e:
            logger.warning(f"Error detecting MPS: {e}")
            gpu_info["mps_available"] = False
            
        return gpu_info
    
    def _detect_software_versions(self) -> Dict[str, str]:
        """Detect installed software versions."""
        versions = {}
        
        # Python version
        versions["python"] = platform.python_version()
        
        # Check common libraries
        libraries = [
            "numpy", "pandas", "scipy", "torch", "tensorflow", 
            "onnx", "onnxruntime", "transformers", "diffusers",
            "matplotlib", "sklearn", "duckdb", "sqlalchemy", "psutil"
        ]
        
        for lib in libraries:
            try:
                versions[lib] = importlib_metadata.version(lib)
            except (importlib_metadata.PackageNotFoundError, ImportError):
                pass
        
        return versions
    
    def _detect_supported_backends(self) -> List[str]:
        """Detect supported inference backends."""
        backends = ["cpu"]
        
        # PyTorch backends
        if HAS_TORCH:
            if torch.cuda.is_available():
                backends.append("cuda")
            if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                backends.append("mps")
        
        # TensorFlow backends
        if HAS_TF:
            if tf.test.is_gpu_available(cuda_only=True):
                backends.append("tf_gpu")
        
        # ONNX Runtime providers
        if HAS_ORT:
            backends.extend([provider.split('ExecutionProvider')[0].lower() 
                           for provider in ort.get_available_providers()])
        
        return backends
    
    def _estimate_network_bandwidth(self) -> float:
        """Estimate network bandwidth in Mbps, based on system information."""
        # This is a simplified estimation based on network interface speed
        try:
            if not psutil:
                return 100.0

            # Get network stats for all interfaces
            net_io = psutil.net_io_counters(pernic=True)
            
            # Find the fastest interface (excluding loopback)
            max_speed = 0.0
            for interface, stats in net_io.items():
                if interface.startswith(('lo', 'veth', 'docker')):
                    continue
                    
                # Get max of bytes sent and received as a rough estimate
                speed = max(stats.bytes_sent, stats.bytes_recv)
                max_speed = max(max_speed, speed)
            
            # Convert to Mbps (very rough approximation)
            # In a real implementation, you would measure this properly
            estimated_mbps = max_speed / (1024 * 1024) * 8
            
            # Set reasonable bounds for the estimate
            if estimated_mbps < 10:
                return 100.0  # Assume at least 100 Mbps Ethernet
            if estimated_mbps > 10000:
                return 10000.0  # Cap at 10 Gbps
                
            return float(estimated_mbps)
        except Exception as e:
            logger.warning(f"Error estimating network bandwidth: {e}")
            return 100.0  # Default to 100 Mbps as a safe assumption
    
    def _detect_storage_capacity(self) -> float:
        """Detect total storage capacity in GB."""
        try:
            # Get disk usage for the root file system
            disk_usage = psutil.disk_usage('/') if psutil else shutil.disk_usage('/')
            total_gb = disk_usage.total / (1024 ** 3)
            return float(total_gb)
        except Exception as e:
            logger.warning(f"Error detecting storage capacity: {e}")
            return 0.0
    
    def _detect_available_accelerators(self) -> Dict[str, int]:
        """Detect available accelerators (GPUs, TPUs, etc.) and their count."""
        accelerators = {}
        
        # CUDA GPUs
        if HAS_TORCH and torch.cuda.is_available():
            accelerators["cuda"] = torch.cuda.device_count()
        
        # Apple MPS
        if HAS_TORCH and hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            accelerators["mps"] = 1  # Apple silicon has one unified GPU
        
        # ROCm GPUs
        if HAS_TORCH and hasattr(torch, 'version') and hasattr(torch.version, 'hip') and torch.version.hip is not None:
            # For ROCm, we need a different approach to count devices
            try:
                import subprocess
                result = subprocess.run(['rocm-smi', '--showcount'], 
                                      capture_output=True, text=True, check=False)
                if result.returncode == 0:
                    count_line = [line for line in result.stdout.splitlines() 
                                if "GPU count" in line]
                    if count_line:
                        count = int(count_line[0].split(":")[1].strip())
                        accelerators["rocm"] = count
            except Exception as e:
                logger.warning(f"Error detecting ROCm GPU count: {e}")
        
        # Intel oneAPI / OpenVINO
        if "openvino" in sys.modules or "oneapi" in sys.modules:
            try:
                if "openvino" in sys.modules:
                    import openvino as ov
                    core = ov.Core()
                    if "GPU" in core.available_devices:
                        accelerators["openvino_gpu"] = 1
                    if "CPU" in core.available_devices:
                        accelerators["openvino_cpu"] = 1
            except Exception as e:
                logger.warning(f"Error detecting OpenVINO devices: {e}")
        
        # Qualcomm Accelerators (Hexagon DSP)
        if HAS_ORT and "qnn" in [provider.lower() for provider in ort.get_available_providers()]:
            accelerators["hexagon"] = 1  # Standard assumption for mobile devices
        
        return accelerators
    
    def _detect_available_memory(self) -> float:
        """Detect available system memory in GB."""
        try:
            if psutil:
                memory = psutil.virtual_memory()
                return float(memory.available / (1024 ** 3))

            # Best-effort Linux fallback.
            if os.path.exists("/proc/meminfo"):
                meminfo = {}
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        if ":" in line:
                            key, value = line.split(":", 1)
                            meminfo[key.strip()] = value.strip()

                # Values are typically in kB.
                available_kb = meminfo.get("MemAvailable") or meminfo.get("MemFree")
                if available_kb:
                    available_kb_int = int(available_kb.split()[0])
                    return float((available_kb_int * 1024) / (1024 ** 3))

            return 0.0
        except Exception as e:
            logger.warning(f"Error detecting available memory: {e}")
            return 0.0
    
    def _detect_available_disk(self) -> float:
        """Detect available disk space in GB."""
        try:
            disk = psutil.disk_usage('/') if psutil else shutil.disk_usage('/')
            return float(disk.free / (1024 ** 3))
        except Exception as e:
            logger.warning(f"Error detecting available disk: {e}")
            return 0.0
    
    def get_capabilities(self) -> Optional[WorkerCapabilities]:
        """Get the current detected capabilities, or None if not yet detected."""
        return self.capabilities
    
    def update_capabilities(self) -> WorkerCapabilities:
        """Update the capabilities by re-detecting them."""
        return self.detect_capabilities()
    
    def to_json(self) -> str:
        """Serialize capabilities to JSON."""
        if not self.capabilities:
            self.detect_capabilities()
            
        if self.capabilities:
            return json.dumps(self.capabilities.to_dict(), indent=2)
        else:
            return "{}"
    
    def from_json(self, json_data: str) -> None:
        """Deserialize capabilities from JSON."""
        data = json.loads(json_data)
        self.capabilities = WorkerCapabilities.from_dict(data)
    
    def save_to_file(self, file_path: str) -> None:
        """Save capabilities to a file in JSON format."""
        if not self.capabilities:
            self.detect_capabilities()
            
        with open(file_path, 'w') as f:
            f.write(self.to_json())
            
    def load_from_file(self, file_path: str) -> None:
        """Load capabilities from a JSON file."""
        with open(file_path, 'r') as f:
            self.from_json(f.read())


def detect_capabilities_cli() -> None:
    """CLI entry point for detecting worker capabilities."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Detect worker capabilities")
    parser.add_argument('--worker-id', type=str, help="Worker ID (default: auto-generated)")
    parser.add_argument('--output', type=str, help="Output file (default: stdout)")
    args = parser.parse_args()
    
    detector = WorkerCapabilityDetector(worker_id=args.worker_id)
    capabilities = detector.detect_capabilities()
    
    if args.output:
        detector.save_to_file(args.output)
        print(f"Capabilities saved to {args.output}")
    else:
        print(detector.to_json())


if __name__ == "__main__":
    detect_capabilities_cli()