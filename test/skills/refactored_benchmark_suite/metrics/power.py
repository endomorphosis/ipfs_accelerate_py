"""
Power efficiency metrics for model benchmarking.

This module provides metrics for measuring power consumption and efficiency
during model inference across different hardware platforms.
"""

import time
import logging
import platform
import subprocess
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

import torch

logger = logging.getLogger("benchmark.metrics.power")

class PowerMetric:
    """
    Metric for measuring power consumption during inference.
    
    Supports various hardware platforms with platform-specific power monitoring.
    Provides efficiency metrics like performance per watt.
    """
    
    def __init__(self, device_type: str = "cpu"):
        """
        Initialize the power metric.
        
        Args:
            device_type: Type of device being benchmarked
        """
        self.device_type = device_type
        self.power_samples = []
        self.start_time = 0
        self.end_time = 0
        self.has_nvidia_smi = self._check_nvidia_smi()
        self.has_powermetrics = self._check_powermetrics()
        self.has_intel_rapl = self._check_intel_rapl()
        self.has_amd_rocm_smi = self._check_rocm_smi()
        self.sampling_rate_ms = 100  # Power sampling rate in ms
        self.is_sampling = False
        self.sampling_thread = None
        
        # Performance metrics to calculate efficiency
        self.operations_count = 0
        self.throughput = 0
        
    def _check_nvidia_smi(self) -> bool:
        """Check if NVIDIA SMI is available for GPU power monitoring."""
        if self.device_type != "cuda":
            return False
            
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            return len(result.stdout.strip()) > 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    def _check_powermetrics(self) -> bool:
        """Check if powermetrics is available (macOS)."""
        if platform.system() != "Darwin" or self.device_type != "mps":
            return False
            
        try:
            # Check if we have permission to run powermetrics (requires sudo)
            result = subprocess.run(
                ["powermetrics", "-h"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    def _check_intel_rapl(self) -> bool:
        """Check if Intel RAPL is available for CPU power monitoring."""
        if self.device_type != "cpu" or platform.system() != "Linux":
            return False
            
        try:
            # Check if RAPL sysfs entries exist
            result = subprocess.run(
                ["ls", "/sys/class/powercap/intel-rapl"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    def _check_rocm_smi(self) -> bool:
        """Check if ROCm SMI is available for AMD GPU monitoring."""
        if self.device_type != "rocm":
            return False
            
        try:
            result = subprocess.run(
                ["rocm-smi", "--showpower"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    def _get_nvidia_power(self) -> float:
        """Get current power draw from NVIDIA GPU in watts."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            return float(result.stdout.strip())
        except (subprocess.SubprocessError, ValueError):
            return 0.0
    
    def _get_intel_rapl_power(self) -> float:
        """Get current power draw from Intel CPU using RAPL in watts."""
        try:
            # Read package energy
            with open("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj", "r") as f:
                energy_uj = int(f.read().strip())
            
            # Convert from microjoules to watts (requires two readings)
            time.sleep(0.1)  # 100ms
            
            with open("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj", "r") as f:
                new_energy_uj = int(f.read().strip())
            
            # Calculate power in watts (energy difference / time)
            power_watts = (new_energy_uj - energy_uj) / 0.1 / 1000000
            return power_watts
        except (FileNotFoundError, ValueError, PermissionError):
            return 0.0
    
    def _get_rocm_power(self) -> float:
        """Get current power draw from AMD GPU using ROCm SMI in watts."""
        try:
            result = subprocess.run(
                ["rocm-smi", "--showpower"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            
            # Parse the output to extract power value
            for line in result.stdout.strip().split('\n'):
                if 'W' in line and '|' in line:
                    parts = line.split('|')
                    if len(parts) >= 3:
                        power_str = parts[2].strip().replace('W', '')
                        return float(power_str)
            return 0.0
        except (subprocess.SubprocessError, ValueError):
            return 0.0
    
    def _sample_power(self):
        """Sample power consumption periodically."""
        import threading
        
        def sampling_loop():
            while self.is_sampling:
                power_reading = self._get_current_power()
                if power_reading > 0:
                    self.power_samples.append((time.time(), power_reading))
                time.sleep(self.sampling_rate_ms / 1000)
        
        self.is_sampling = True
        self.sampling_thread = threading.Thread(target=sampling_loop)
        self.sampling_thread.daemon = True
        self.sampling_thread.start()
    
    def _get_current_power(self) -> float:
        """Get current power reading based on device type."""
        if self.device_type == "cuda" and self.has_nvidia_smi:
            return self._get_nvidia_power()
        elif self.device_type == "cpu" and self.has_intel_rapl:
            return self._get_intel_rapl_power()
        elif self.device_type == "rocm" and self.has_amd_rocm_smi:
            return self._get_rocm_power()
        # Add more platform-specific power monitoring here
        return 0.0
    
    def start(self):
        """Start measuring power consumption."""
        self.start_time = time.time()
        self.power_samples = []
        
        # Start power sampling in a separate thread if available
        if (self.has_nvidia_smi or self.has_intel_rapl or 
            self.has_powermetrics or self.has_amd_rocm_smi):
            self._sample_power()
    
    def stop(self):
        """Stop measuring power consumption."""
        self.end_time = time.time()
        
        # Stop sampling thread
        if self.is_sampling:
            self.is_sampling = False
            if self.sampling_thread:
                self.sampling_thread.join(timeout=1.0)
    
    def set_operations_count(self, count: int):
        """
        Set the number of operations performed during the benchmark.
        
        This is used to calculate operations per watt.
        
        Args:
            count: Number of operations (e.g., FLOPs)
        """
        self.operations_count = count
    
    def set_throughput(self, throughput: float):
        """
        Set the throughput achieved during the benchmark.
        
        This is used to calculate throughput per watt.
        
        Args:
            throughput: Throughput in items/second
        """
        self.throughput = throughput
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get power-related metrics.
        
        Returns:
            Dictionary of power metrics
        """
        if not self.power_samples:
            return {
                "power_supported": False,
                "power_avg_watts": 0.0
            }
        
        # Calculate average power
        power_values = [sample[1] for sample in self.power_samples]
        avg_power = sum(power_values) / len(power_values) if power_values else 0
        max_power = max(power_values) if power_values else 0
        
        # Calculate duration
        duration = self.end_time - self.start_time
        
        # Calculate energy
        energy_joules = avg_power * duration
        
        # Calculate efficiency metrics
        ops_per_watt = self.operations_count / avg_power if avg_power > 0 else 0
        throughput_per_watt = self.throughput / avg_power if avg_power > 0 else 0
        
        return {
            "power_supported": True,
            "power_avg_watts": avg_power,
            "power_max_watts": max_power,
            "power_samples_count": len(self.power_samples),
            "energy_joules": energy_joules,
            "ops_per_watt": ops_per_watt,
            "gflops_per_watt": ops_per_watt / 1e9 if ops_per_watt > 0 else 0,
            "throughput_per_watt": throughput_per_watt
        }


class PowerMetricFactory:
    """Factory class for creating appropriate power metrics based on hardware."""
    
    @staticmethod
    def create(device: Any) -> PowerMetric:
        """
        Create a power metric for the specified device.
        
        Args:
            device: PyTorch device or device object
            
        Returns:
            PowerMetric instance configured for the device
        """
        device_type = PowerMetricFactory._get_device_type(device)
        return PowerMetric(device_type)
    
    @staticmethod
    def _get_device_type(device: Any) -> str:
        """
        Extract device type from the device object.
        
        Args:
            device: PyTorch device or device object
            
        Returns:
            Device type string
        """
        device_type = "cpu"
        
        # Handle PyTorch devices
        if isinstance(device, torch.device):
            device_type = device.type
        # Handle hardware backend devices
        elif isinstance(device, dict) and "device" in device:
            device_type = device["device"]
        # Handle strings
        elif isinstance(device, str):
            device_type = device.split(":")[0]  # Handle "cuda:0" format
        
        return device_type