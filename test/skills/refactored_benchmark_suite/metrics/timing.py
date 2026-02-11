"""
Timing metrics for model benchmarking.

This module provides metrics for measuring latency, throughput, and other timing-related
metrics for model inference across different hardware platforms.
"""

import time
import statistics
import logging
from typing import Dict, List, Any, Optional, Union

import torch

logger = logging.getLogger("benchmark.metrics.timing")

class LatencyMetric:
    """
    Metric for measuring model inference latency.
    
    Measures the time taken for each forward pass and calculates statistics like
    mean, min, max, and standard deviation.
    """
    
    def __init__(self, device_type: str = "cpu"):
        """
        Initialize the latency metric.
        
        Args:
            device_type: Type of device being benchmarked
        """
        self.device_type = device_type
        self.start_time = None
        self.timestamps = []
        self.latencies = []
        self.can_synchronize = self._check_synchronization_capabilities()
    
    def _check_synchronization_capabilities(self) -> bool:
        """
        Check if the device supports synchronization for accurate timing.
        
        Returns:
            True if synchronization is supported, False otherwise
        """
        if self.device_type == "cuda":
            return torch.cuda.is_available()
        elif self.device_type == "mps":
            return hasattr(torch, "mps") and hasattr(torch.mps, "synchronize")
        elif self.device_type == "xla":
            try:
                import torch_xla.core.xla_model as xm
                return True
            except ImportError:
                return False
        return False
    
    def _synchronize(self):
        """Synchronize the device if needed for accurate timing."""
        if not self.can_synchronize:
            return
            
        if self.device_type == "cuda":
            torch.cuda.synchronize()
        elif self.device_type == "mps" and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()
        elif self.device_type == "xla":
            try:
                import torch_xla.core.xla_model as xm
                xm.mark_step()
            except ImportError:
                pass
    
    def start(self):
        """Start measuring latency."""
        # Reset timestamp lists
        self.timestamps = []
        self.latencies = []
        
        # Synchronize before timing if supported
        self._synchronize()
        
        # Record initial timestamp
        self.start_time = time.perf_counter()
        self.timestamps.append(self.start_time)
    
    def stop(self):
        """Stop measuring latency."""
        # Synchronize before recording final timestamp
        self._synchronize()
        
        # Calculate latencies
        for i in range(1, len(self.timestamps)):
            self.latencies.append(self.timestamps[i] - self.timestamps[i-1])
    
    def record_step(self):
        """Record a timestamp for a single step."""
        # Synchronize before recording timestamp
        self._synchronize()
            
        self.timestamps.append(time.perf_counter())
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get the latency metrics.
        
        Returns:
            Dictionary of latency metrics (in milliseconds)
        """
        if not self.latencies:
            return {
                "latency_ms": 0.0,
                "latency_min_ms": 0.0,
                "latency_max_ms": 0.0,
                "latency_std_ms": 0.0,
                "latency_median_ms": 0.0,
                "latency_p90_ms": 0.0,
                "latency_p95_ms": 0.0,
                "latency_p99_ms": 0.0
            }
        
        # Convert to milliseconds
        latencies_ms = [latency * 1000 for latency in self.latencies]
        
        # Calculate statistics
        mean_latency = statistics.mean(latencies_ms)
        median_latency = statistics.median(latencies_ms)
        min_latency = min(latencies_ms)
        max_latency = max(latencies_ms)
        std_latency = statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0
        
        # Calculate percentiles
        sorted_latencies = sorted(latencies_ms)
        p90_index = int(len(sorted_latencies) * 0.9)
        p95_index = int(len(sorted_latencies) * 0.95)
        p99_index = int(len(sorted_latencies) * 0.99)
        
        p90_latency = sorted_latencies[p90_index] if p90_index < len(sorted_latencies) else max_latency
        p95_latency = sorted_latencies[p95_index] if p95_index < len(sorted_latencies) else max_latency
        p99_latency = sorted_latencies[p99_index] if p99_index < len(sorted_latencies) else max_latency
        
        return {
            "latency_ms": mean_latency,
            "latency_median_ms": median_latency,
            "latency_min_ms": min_latency,
            "latency_max_ms": max_latency,
            "latency_std_ms": std_latency,
            "latency_p90_ms": p90_latency,
            "latency_p95_ms": p95_latency,
            "latency_p99_ms": p99_latency
        }
    
    def get_latency_distribution(self) -> Dict[str, List[float]]:
        """
        Get the raw latency distribution.
        
        Returns:
            Dictionary with raw latency data for analysis and visualization
        """
        # Convert to milliseconds
        latencies_ms = [latency * 1000 for latency in self.latencies]
        
        return {
            "latencies_ms": latencies_ms,
            "timestamps": [t - self.start_time for t in self.timestamps[1:]]
        }


class ThroughputMetric:
    """
    Metric for measuring model inference throughput.
    
    Measures the number of items (e.g., tokens, images) processed per second.
    """
    
    def __init__(self, batch_size: int = 1, items_per_batch: Optional[int] = None, device_type: str = "cpu"):
        """
        Initialize the throughput metric.
        
        Args:
            batch_size: Number of items in each batch
            items_per_batch: Custom items per batch (e.g., for NLP tasks where items might be tokens)
            device_type: Type of device being benchmarked
        """
        self.batch_size = batch_size
        self.items_per_batch = items_per_batch or batch_size
        self.device_type = device_type
        self.start_time = None
        self.end_time = None
        self.num_iterations = 0
        self.custom_items_processed = 0
        self.batch_times = []
        self.can_synchronize = self._check_synchronization_capabilities()
    
    def _check_synchronization_capabilities(self) -> bool:
        """
        Check if the device supports synchronization for accurate timing.
        
        Returns:
            True if synchronization is supported, False otherwise
        """
        if self.device_type == "cuda":
            return torch.cuda.is_available()
        elif self.device_type == "mps":
            return hasattr(torch, "mps") and hasattr(torch.mps, "synchronize")
        elif self.device_type == "xla":
            try:
                import torch_xla.core.xla_model as xm
                return True
            except ImportError:
                return False
        return False
    
    def _synchronize(self):
        """Synchronize the device if needed for accurate timing."""
        if not self.can_synchronize:
            return
            
        if self.device_type == "cuda":
            torch.cuda.synchronize()
        elif self.device_type == "mps" and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()
        elif self.device_type == "xla":
            try:
                import torch_xla.core.xla_model as xm
                xm.mark_step()
            except ImportError:
                pass
    
    def start(self):
        """Start measuring throughput."""
        # Synchronize before timing if supported
        self._synchronize()
            
        self.start_time = time.perf_counter()
        self.num_iterations = 0
        self.custom_items_processed = 0
        self.batch_times = []
    
    def stop(self):
        """Stop measuring throughput."""
        # Synchronize before final timing if supported
        self._synchronize()
            
        self.end_time = time.perf_counter()
    
    def update(self, items_processed: Optional[int] = None):
        """
        Update the throughput measurement.
        
        Args:
            items_processed: Number of items processed in this update
        """
        # Synchronize before recording timestamp
        self._synchronize()
        
        # Record batch completion time
        self.batch_times.append(time.perf_counter())
        
        self.num_iterations += 1
        if items_processed is not None:
            self.custom_items_processed += items_processed
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get the throughput metrics.
        
        Returns:
            Dictionary of throughput metrics
        """
        if self.start_time is None or self.end_time is None:
            return {
                "throughput_items_per_sec": 0.0,
                "throughput_batches_per_sec": 0.0,
                "total_items_processed": 0,
                "time_per_batch_ms": 0.0
            }
        
        # Calculate elapsed time
        elapsed_time = self.end_time - self.start_time
        
        if elapsed_time <= 0 or self.num_iterations <= 0:
            return {
                "throughput_items_per_sec": 0.0,
                "throughput_batches_per_sec": 0.0,
                "total_items_processed": 0,
                "time_per_batch_ms": 0.0
            }
        
        # Calculate throughput
        if self.custom_items_processed > 0:
            total_items = self.custom_items_processed
        else:
            total_items = self.num_iterations * self.items_per_batch
            
        items_per_sec = total_items / elapsed_time
        batches_per_sec = self.num_iterations / elapsed_time
        
        # Calculate time per batch (in ms)
        if len(self.batch_times) > 1:
            batch_durations = []
            for i in range(1, len(self.batch_times)):
                batch_durations.append(self.batch_times[i] - self.batch_times[i-1])
            time_per_batch_ms = statistics.mean(batch_durations) * 1000
        else:
            time_per_batch_ms = (elapsed_time / self.num_iterations) * 1000 if self.num_iterations > 0 else 0.0
        
        return {
            "throughput_items_per_sec": items_per_sec,
            "throughput_batches_per_sec": batches_per_sec,
            "total_items_processed": total_items,
            "time_per_batch_ms": time_per_batch_ms
        }
    
    def get_batch_times(self) -> List[float]:
        """
        Get the raw batch completion times.
        
        Returns:
            List of batch completion times relative to start time
        """
        if not self.batch_times or self.start_time is None:
            return []
            
        return [t - self.start_time for t in self.batch_times]


class TimingMetricFactory:
    """Factory class for creating appropriate timing metrics based on hardware."""
    
    @staticmethod
    def create_latency_metric(device: Any) -> LatencyMetric:
        """
        Create a latency metric for the specified device.
        
        Args:
            device: PyTorch device or device object
            
        Returns:
            LatencyMetric instance configured for the device
        """
        device_type = TimingMetricFactory._get_device_type(device)
        return LatencyMetric(device_type)
    
    @staticmethod
    def create_throughput_metric(device: Any, batch_size: int = 1, items_per_batch: Optional[int] = None) -> ThroughputMetric:
        """
        Create a throughput metric for the specified device.
        
        Args:
            device: PyTorch device or device object
            batch_size: Number of items in each batch
            items_per_batch: Custom items per batch
            
        Returns:
            ThroughputMetric instance configured for the device
        """
        device_type = TimingMetricFactory._get_device_type(device)
        return ThroughputMetric(batch_size, items_per_batch, device_type)
    
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