"""
Memory bandwidth metrics for model benchmarking.

This module provides metrics for measuring memory bandwidth utilization
during model inference across different hardware platforms.
"""

import time
import logging
import platform
import subprocess
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

import torch

logger = logging.getLogger("benchmark.metrics.bandwidth")

class BandwidthMetric:
    """
    Metric for measuring memory bandwidth during inference.
    
    Supports various hardware platforms with platform-specific bandwidth monitoring.
    Provides efficiency metrics like utilization percentage and roofline model data.
    """
    
    def __init__(self, device_type: str = "cpu"):
        """
        Initialize the bandwidth metric.
        
        Args:
            device_type: Type of device being benchmarked
        """
        self.device_type = device_type
        self.bandwidth_samples = []
        self.start_time = 0
        self.end_time = 0
        self.sampling_rate_ms = 100  # Bandwidth sampling rate in ms
        self.is_sampling = False
        self.sampling_thread = None
        
        # Memory access patterns
        self.memory_accesses = 0
        self.memory_transfers_bytes = 0
        self.compute_operations = 0
        
        # Theoretical peak bandwidth (GB/s)
        self.peak_bandwidth = self._get_theoretical_peak_bandwidth()
        
    def _get_theoretical_peak_bandwidth(self) -> float:
        """Get theoretical peak memory bandwidth for the device in GB/s."""
        if self.device_type == "cpu":
            return self._get_cpu_peak_bandwidth()
        elif self.device_type == "cuda":
            return self._get_cuda_peak_bandwidth()
        elif self.device_type == "rocm":
            return self._get_rocm_peak_bandwidth()
        elif self.device_type == "mps":
            return self._get_mps_peak_bandwidth()
        return 0.0
    
    def _get_cpu_peak_bandwidth(self) -> float:
        """Get theoretical peak memory bandwidth for CPU in GB/s."""
        try:
            # Try to get CPU info using lscpu on Linux
            if platform.system() == "Linux":
                result = subprocess.run(
                    ["lscpu"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # Example of parsing lscpu output to extract memory bandwidth
                # This is a simplified approach; real implementation would be more detailed
                mem_speed = 0.0
                for line in result.stdout.split('\n'):
                    if "Memory" in line and "MHz" in line:
                        parts = line.split(':')
                        if len(parts) >= 2:
                            try:
                                # Extract memory frequency and estimate bandwidth
                                mem_mhz = float(parts[1].strip().split()[0])
                                # Rough estimate: DDR4 has 8 bytes per transfer
                                # Convert MHz to GB/s (MHz * 8 bytes / 1000)
                                mem_speed = mem_mhz * 8 / 1000
                            except (ValueError, IndexError):
                                pass
                
                if mem_speed > 0:
                    return mem_speed
                
                # Default values if detection fails
                return 50.0  # Default for modern CPU (DDR4-3200)
            
            elif platform.system() == "Darwin":  # macOS
                return 60.0  # Estimated value for modern Mac
            
            else:  # Windows or other
                return 50.0  # Default estimate
                
        except Exception as e:
            logger.warning(f"Error detecting CPU peak memory bandwidth: {e}")
            return 50.0  # Default fallback
    
    def _get_cuda_peak_bandwidth(self) -> float:
        """Get theoretical peak memory bandwidth for NVIDIA GPU in GB/s."""
        if not torch.cuda.is_available():
            return 0.0
            
        try:
            # Query NVIDIA GPU properties
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total,memory.clock", "--format=csv,noheader,nounits"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if result.returncode == 0:
                parts = result.stdout.strip().split(',')
                if len(parts) >= 2:
                    # Extract memory size and clock
                    try:
                        mem_size_mb = float(parts[0].strip())
                        mem_clock_mhz = float(parts[1].strip())
                        
                        # Rough estimate based on memory type (GDDR6/HBM2)
                        # More accurate would be to use more device-specific properties
                        if mem_size_mb > 16000:  # Likely HBM memory with wider bus
                            bus_width = 4096  # bits
                        elif mem_size_mb > 8000:
                            bus_width = 384  # bits
                        else:
                            bus_width = 256  # bits
                        
                        # Calculate bandwidth: Clock * 2 (DDR) * bus_width / 8 (bits to bytes) / 1000 (MB to GB)
                        bandwidth = mem_clock_mhz * 2 * bus_width / 8 / 1000
                        return bandwidth
                    except (ValueError, IndexError):
                        pass
            
            # Fallback to getting device properties via CUDA
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            
            # Theoretical bandwidth based on memory clock and bus width
            # Note: This is a simplified calculation
            mem_clock_rate = getattr(props, 'memory_clock_rate', 0) / 1e6  # Convert to MHz
            mem_bus_width = getattr(props, 'memory_bus_width', 0)
            
            if mem_clock_rate > 0 and mem_bus_width > 0:
                # Calculate bandwidth: Clock * 2 (DDR) * bus_width / 8 (bits to bytes) / 1000 (MB to GB)
                bandwidth = mem_clock_rate * 2 * mem_bus_width / 8 / 1000
                return bandwidth
            
            # Default values if detection fails
            # Common values: RTX 3090 (936 GB/s), A100 (1,555 GB/s)
            return 800.0
            
        except Exception as e:
            logger.warning(f"Error detecting CUDA peak memory bandwidth: {e}")
            return 800.0  # Default fallback
    
    def _get_rocm_peak_bandwidth(self) -> float:
        """Get theoretical peak memory bandwidth for AMD GPU in GB/s."""
        try:
            # Query AMD GPU properties via rocm-smi
            result = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if result.returncode == 0:
                # Parse the output (this is simplified)
                mem_size_mb = 0
                for line in result.stdout.split('\n'):
                    if "vram" in line.lower() and "total" in line.lower():
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part.lower() == "total":
                                try:
                                    mem_size_mb = float(parts[i+1])
                                    break
                                except (ValueError, IndexError):
                                    pass
                
                # Rough estimate based on memory size
                if mem_size_mb >= 16000:  # MI100/MI200 class
                    return 1200.0
                elif mem_size_mb >= 8000:  # MI50/MI60 class
                    return 800.0
                else:  # Smaller GPUs
                    return 400.0
            
            # Default values if detection fails
            return 800.0  # Default for modern AMD GPUs
            
        except Exception as e:
            logger.warning(f"Error detecting ROCm peak memory bandwidth: {e}")
            return 800.0  # Default fallback
    
    def _get_mps_peak_bandwidth(self) -> float:
        """Get theoretical peak memory bandwidth for Apple Silicon in GB/s."""
        try:
            # macOS doesn't have a standard way to query this programmatically
            # We'll use known values based on the model
            if platform.processor() == "arm":  # Apple Silicon
                model = platform.machine()
                
                # Use known values for different Apple Silicon chips
                if "M1 Ultra" in model:
                    return 800.0
                elif "M1 Max" in model:
                    return 400.0
                elif "M1 Pro" in model:
                    return 200.0
                elif "M1" in model:
                    return 100.0
                elif "M2 Ultra" in model:
                    return 900.0
                elif "M2 Max" in model:
                    return 450.0
                elif "M2 Pro" in model:
                    return 225.0
                elif "M2" in model:
                    return 120.0
                elif "M3 Ultra" in model:
                    return 1000.0
                elif "M3 Max" in model:
                    return 500.0
                elif "M3 Pro" in model:
                    return 250.0
                elif "M3" in model:
                    return 150.0
                else:
                    return 100.0  # Default for Apple Silicon
            
            return 60.0  # Default for Intel Mac
            
        except Exception as e:
            logger.warning(f"Error detecting MPS peak memory bandwidth: {e}")
            return 100.0  # Default fallback for M-series
    
    def _sample_bandwidth(self):
        """Sample memory bandwidth periodically."""
        import threading
        
        def sampling_loop():
            while self.is_sampling:
                bandwidth_reading = self._get_current_bandwidth()
                if bandwidth_reading > 0:
                    self.bandwidth_samples.append((time.time(), bandwidth_reading))
                time.sleep(self.sampling_rate_ms / 1000)
        
        self.is_sampling = True
        self.sampling_thread = threading.Thread(target=sampling_loop)
        self.sampling_thread.daemon = True
        self.sampling_thread.start()
    
    def _get_current_bandwidth(self) -> float:
        """Get current memory bandwidth in GB/s."""
        if self.device_type == "cuda" and torch.cuda.is_available():
            return self._get_cuda_bandwidth()
        elif self.device_type == "rocm":
            return self._get_rocm_bandwidth()
        elif self.device_type == "cpu":
            return self._get_cpu_bandwidth()
        return 0.0
    
    def _get_cuda_bandwidth(self) -> float:
        """Get current memory bandwidth for NVIDIA GPU in GB/s."""
        try:
            # Use nvidia-smi to get current memory utilization
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.memory", "--format=csv,noheader,nounits"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            
            # Parse utilization percentage
            mem_util_percent = float(result.stdout.strip()) / 100.0
            
            # Calculate approximate bandwidth based on peak
            return self.peak_bandwidth * mem_util_percent
        except Exception:
            return 0.0
    
    def _get_rocm_bandwidth(self) -> float:
        """Get current memory bandwidth for AMD GPU in GB/s."""
        try:
            # Use rocm-smi to get current memory utilization
            result = subprocess.run(
                ["rocm-smi", "--showmemuse"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            
            # Parse memory utilization
            mem_util_percent = 0.0
            for line in result.stdout.strip().split('\n'):
                if '%' in line and 'mem' in line.lower():
                    parts = line.split()
                    for part in parts:
                        if '%' in part:
                            try:
                                mem_util_percent = float(part.replace('%', '')) / 100.0
                                break
                            except ValueError:
                                pass
            
            # Calculate approximate bandwidth based on peak
            return self.peak_bandwidth * mem_util_percent
        except Exception:
            return 0.0
    
    def _get_cpu_bandwidth(self) -> float:
        """Get current memory bandwidth for CPU in GB/s."""
        # This is challenging to measure directly, we'll use a rough estimate
        # based on memory allocation patterns observed through the benchmark
        
        # On Linux, we could potentially use perf counters, but that needs root
        # For now, we return a statistical estimate
        memory_transfer_rate = 0.0
        
        if self.end_time > self.start_time:
            duration = self.end_time - self.start_time
            if duration > 0 and self.memory_transfers_bytes > 0:
                # Convert bytes to GB
                memory_transfer_gb = self.memory_transfers_bytes / 1e9
                # Calculate GB/s
                memory_transfer_rate = memory_transfer_gb / duration
        
        return memory_transfer_rate
    
    def set_memory_accesses(self, count: int):
        """
        Set the number of memory accesses performed during the benchmark.
        
        Args:
            count: Number of memory accesses
        """
        self.memory_accesses = count
    
    def set_memory_transfers(self, bytes_transferred: int):
        """
        Set the total bytes transferred during the benchmark.
        
        Args:
            bytes_transferred: Number of bytes transferred
        """
        self.memory_transfers_bytes = bytes_transferred
    
    def set_compute_operations(self, count: int):
        """
        Set the number of compute operations performed during the benchmark.
        
        Used to calculate arithmetic intensity for roofline model.
        
        Args:
            count: Number of operations (e.g., FLOPs)
        """
        self.compute_operations = count
    
    def start(self):
        """Start measuring memory bandwidth."""
        self.start_time = time.time()
        self.bandwidth_samples = []
        
        # Start bandwidth sampling in a separate thread if available
        if self.device_type in ["cuda", "rocm"]:
            self._sample_bandwidth()
    
    def stop(self):
        """Stop measuring memory bandwidth."""
        self.end_time = time.time()
        
        # Stop sampling thread
        if self.is_sampling:
            self.is_sampling = False
            if self.sampling_thread:
                self.sampling_thread.join(timeout=1.0)
    
    def estimate_memory_transfers(self, model_size_bytes: int, batch_size: int, inference_count: int) -> int:
        """
        Estimate memory transfers based on model size and batch size.
        
        A simplified heuristic for estimating memory transfers when direct measurement isn't available.
        
        Args:
            model_size_bytes: Size of the model in bytes
            batch_size: Size of the batch
            inference_count: Number of inferences performed
            
        Returns:
            Estimated bytes transferred
        """
        # Basic heuristic: each parameter is read at least once
        base_transfers = model_size_bytes
        
        # Intermediate activations depend on batch size
        # This is a simplified model; actual memory transfers will vary by architecture
        activation_multiplier = 2.5  # Typical multiplier for activations vs parameters
        
        # Total estimated transfers
        estimated_transfers = base_transfers * (1 + batch_size * activation_multiplier) * inference_count
        
        # Update the stored value
        self.memory_transfers_bytes = estimated_transfers
        return estimated_transfers
    
    def get_arithmetic_intensity(self) -> float:
        """
        Calculate arithmetic intensity (FLOPs per byte) for roofline model.
        
        Returns:
            Arithmetic intensity (operations per byte)
        """
        if self.memory_transfers_bytes > 0:
            return self.compute_operations / self.memory_transfers_bytes
        return 0.0
    
    def is_compute_bound(self) -> bool:
        """
        Determine if workload is compute bound or memory bound.
        
        Returns:
            True if compute bound, False if memory bound
        """
        # Get arithmetic intensity
        arithmetic_intensity = self.get_arithmetic_intensity()
        
        # Get ridge point (where memory bandwidth stops being the bottleneck)
        # For example, on V100 with 900 GB/s and 7 TFLOPS, ridge point is 7.8 FLOP/byte
        if self.peak_bandwidth > 0:
            peak_flops = self._get_peak_compute()
            ridge_point = peak_flops / (self.peak_bandwidth * 1e9)
            
            # Compare with the workload's arithmetic intensity
            return arithmetic_intensity > ridge_point
        
        # Default to compute bound if we can't determine
        return True
    
    def _get_peak_compute(self) -> float:
        """Get theoretical peak compute throughput for the device in FLOP/s."""
        if self.device_type == "cuda" and torch.cuda.is_available():
            try:
                # Get GPU properties
                device_props = torch.cuda.get_device_properties(torch.cuda.current_device())
                
                # Calculate peak FLOPS
                # CUDA cores * 2 (FMA) * clock speed
                cuda_cores = device_props.multi_processor_count * 128  # Approximation
                clock_rate = device_props.clock_rate / 1e6  # MHz
                
                return cuda_cores * 2 * clock_rate * 1e6  # FLOP/s
            except Exception:
                # Fallback for modern GPUs (approximation)
                return 10e12  # 10 TFLOPS
                
        elif self.device_type == "cpu":
            # Crude estimation for modern CPUs
            return 1e12  # 1 TFLOPS
            
        elif self.device_type == "rocm":
            # Fallback for modern AMD GPUs
            return 10e12  # 10 TFLOPS
            
        elif self.device_type == "mps":
            # Fallback for modern Apple Silicon
            return 5e12  # 5 TFLOPS
            
        return 1e12  # Default fallback
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get bandwidth-related metrics.
        
        Returns:
            Dictionary of bandwidth metrics
        """
        if not self.bandwidth_samples and self.memory_transfers_bytes == 0:
            return {
                "bandwidth_supported": False,
                "avg_bandwidth_gbps": 0.0
            }
        
        # Calculate duration
        duration = self.end_time - self.start_time
        
        # Calculate average bandwidth
        avg_bandwidth = 0.0
        peak_bandwidth_measured = 0.0
        
        if self.bandwidth_samples:
            # Use sampled bandwidth if available
            bandwidth_values = [sample[1] for sample in self.bandwidth_samples]
            avg_bandwidth = sum(bandwidth_values) / len(bandwidth_values) if bandwidth_values else 0
            peak_bandwidth_measured = max(bandwidth_values) if bandwidth_values else 0
        elif self.memory_transfers_bytes > 0 and duration > 0:
            # Alternatively, calculate from total transfers
            avg_bandwidth = (self.memory_transfers_bytes / 1e9) / duration  # GB/s
            peak_bandwidth_measured = avg_bandwidth  # Without samples, peak = avg
        
        # Calculate utilization percentage
        utilization_percent = 0.0
        if self.peak_bandwidth > 0:
            utilization_percent = (avg_bandwidth / self.peak_bandwidth) * 100
        
        # Calculate arithmetic intensity for roofline model
        arithmetic_intensity = self.get_arithmetic_intensity()
        compute_bound = self.is_compute_bound()
        
        # Create metrics dictionary
        return {
            "bandwidth_supported": True,
            "avg_bandwidth_gbps": avg_bandwidth,
            "peak_bandwidth_gbps": peak_bandwidth_measured,
            "peak_theoretical_bandwidth_gbps": self.peak_bandwidth,
            "bandwidth_utilization_percent": utilization_percent,
            "memory_transfers_gb": self.memory_transfers_bytes / 1e9 if self.memory_transfers_bytes > 0 else 0.0,
            "arithmetic_intensity_flops_per_byte": arithmetic_intensity,
            "compute_bound": compute_bound,
            "memory_bound": not compute_bound,
            "bandwidth_samples_count": len(self.bandwidth_samples)
        }
    
    def get_roofline_data(self) -> Dict[str, Any]:
        """
        Get data for roofline performance model visualization.
        
        Returns:
            Dictionary with roofline model parameters
        """
        peak_compute = self._get_peak_compute()
        ridge_point = peak_compute / (self.peak_bandwidth * 1e9) if self.peak_bandwidth > 0 else 1.0
        
        # Calculate actual performance
        duration = self.end_time - self.start_time
        actual_performance = 0.0
        if duration > 0 and self.compute_operations > 0:
            actual_performance = self.compute_operations / duration
        
        # Get arithmetic intensity
        arithmetic_intensity = self.get_arithmetic_intensity()
        
        return {
            "peak_compute_flops": peak_compute,
            "peak_memory_bandwidth_bytes_per_sec": self.peak_bandwidth * 1e9,
            "ridge_point_flops_per_byte": ridge_point,
            "arithmetic_intensity_flops_per_byte": arithmetic_intensity,
            "actual_performance_flops": actual_performance,
            "compute_ceiling_flops": peak_compute,
            "memory_ceiling_flops": self.peak_bandwidth * 1e9 * arithmetic_intensity,
            "is_compute_bound": self.is_compute_bound()
        }


class BandwidthMetricFactory:
    """Factory class for creating appropriate bandwidth metrics based on hardware."""
    
    @staticmethod
    def create(device: Any) -> BandwidthMetric:
        """
        Create a bandwidth metric for the specified device.
        
        Args:
            device: PyTorch device or device object
            
        Returns:
            BandwidthMetric instance configured for the device
        """
        device_type = BandwidthMetricFactory._get_device_type(device)
        return BandwidthMetric(device_type)
    
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