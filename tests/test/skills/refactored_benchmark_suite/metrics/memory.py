"""
Memory usage metrics for model benchmarking.

This module provides metrics for measuring memory usage of PyTorch models during inference.
"""

import gc
import time
import logging
from typing import Dict, List, Any, Optional

import torch

logger = logging.getLogger("benchmark.metrics.memory")

class MemoryMetric:
    """
    Metric for measuring model memory usage.
    
    Tracks peak memory usage, memory allocated, and reserved by PyTorch during inference.
    Also tracks CPU memory usage if psutil is available.
    """
    
    def __init__(self, device_type: str = "cpu"):
        """
        Initialize the memory metric.
        
        Args:
            device_type: Type of device to track memory for ("cpu", "cuda", "mps", etc.)
        """
        self.device_type = device_type
        self.peak_memory = 0
        self.start_allocated = 0
        self.end_allocated = 0
        self.start_reserved = 0
        self.end_reserved = 0
        self.start_cpu_memory = 0
        self.end_cpu_memory = 0
        self.memory_timeline = []
        
        # Check device capabilities for memory tracking
        self.can_track_device_memory = False
        self.can_track_peak_memory = False
        
        if device_type == "cpu":
            # Only CPU memory tracking via psutil
            pass
        elif device_type == "cuda":
            self.can_track_device_memory = torch.cuda.is_available()
            self.can_track_peak_memory = torch.cuda.is_available()
        elif device_type == "mps":
            # MPS can track allocated memory but not peak in all PyTorch versions
            self.can_track_device_memory = (hasattr(torch, "mps") and 
                                           hasattr(torch.mps, "is_available") and 
                                           torch.mps.is_available())
            # Check if memory stats are available
            if self.can_track_device_memory:
                self.can_track_peak_memory = hasattr(torch.mps, "max_memory_allocated")
        else:
            # Other devices don't have built-in memory tracking
            pass
        
        # Check if psutil is available for CPU memory tracking
        try:
            import psutil
            self.psutil = psutil
            self.has_psutil = True
        except ImportError:
            self.has_psutil = False
            logger.warning("psutil not available, CPU memory usage will not be tracked")
    
    def start(self):
        """Start measuring memory usage."""
        # Force garbage collection to get more accurate measurements
        gc.collect()
        
        if self.device_type == "cuda" and self.can_track_device_memory:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self.start_allocated = torch.cuda.memory_allocated()
            self.start_reserved = torch.cuda.memory_reserved()
        
        elif self.device_type == "mps" and self.can_track_device_memory:
            if hasattr(torch.mps, "reset_peak_memory_stats"):
                torch.mps.reset_peak_memory_stats()
            self.start_allocated = torch.mps.memory_allocated()
            self.start_reserved = torch.mps.memory_reserved() if hasattr(torch.mps, "memory_reserved") else 0
        
        if self.has_psutil:
            process = self.psutil.Process()
            self.start_cpu_memory = process.memory_info().rss
        
        # Initialize peak memory
        self.peak_memory = self.start_allocated
        
        # Clear memory timeline
        self.memory_timeline = [(time.time(), self.start_allocated, self.start_reserved, self.start_cpu_memory)]
    
    def record_memory(self):
        """Record current memory usage."""
        current_allocated = 0
        current_reserved = 0
        current_cpu_memory = 0
        
        if self.device_type == "cuda" and self.can_track_device_memory:
            current_allocated = torch.cuda.memory_allocated()
            current_reserved = torch.cuda.memory_reserved()
            self.peak_memory = max(self.peak_memory, current_allocated)
        
        elif self.device_type == "mps" and self.can_track_device_memory:
            current_allocated = torch.mps.memory_allocated()
            current_reserved = torch.mps.memory_reserved() if hasattr(torch.mps, "memory_reserved") else 0
            if self.can_track_peak_memory:
                self.peak_memory = max(self.peak_memory, current_allocated)
        
        if self.has_psutil:
            process = self.psutil.Process()
            current_cpu_memory = process.memory_info().rss
        
        self.memory_timeline.append((time.time(), current_allocated, current_reserved, current_cpu_memory))
    
    def stop(self):
        """Stop measuring memory usage."""
        if self.device_type == "cuda" and self.can_track_device_memory:
            torch.cuda.synchronize()
            self.end_allocated = torch.cuda.memory_allocated()
            self.end_reserved = torch.cuda.memory_reserved()
            if self.can_track_peak_memory:
                self.peak_memory = max(self.peak_memory, torch.cuda.max_memory_allocated())
        
        elif self.device_type == "mps" and self.can_track_device_memory:
            self.end_allocated = torch.mps.memory_allocated()
            self.end_reserved = torch.mps.memory_reserved() if hasattr(torch.mps, "memory_reserved") else 0
            if self.can_track_peak_memory and hasattr(torch.mps, "max_memory_allocated"):
                self.peak_memory = max(self.peak_memory, torch.mps.max_memory_allocated())
        
        if self.has_psutil:
            process = self.psutil.Process()
            self.end_cpu_memory = process.memory_info().rss
        
        # Final memory recording
        self.memory_timeline.append((time.time(), self.end_allocated, self.end_reserved, self.end_cpu_memory))
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get the memory usage metrics.
        
        Returns:
            Dictionary of memory metrics (in MB)
        """
        # Convert from bytes to MB
        bytes_to_mb = lambda x: x / (1024 * 1024)
        
        metrics = {}
        
        # Device memory metrics
        if self.can_track_device_memory:
            metrics.update({
                "memory_peak_mb": bytes_to_mb(self.peak_memory),
                "memory_allocated_start_mb": bytes_to_mb(self.start_allocated),
                "memory_allocated_end_mb": bytes_to_mb(self.end_allocated),
                "memory_reserved_start_mb": bytes_to_mb(self.start_reserved),
                "memory_reserved_end_mb": bytes_to_mb(self.end_reserved),
                "memory_growth_mb": bytes_to_mb(self.end_allocated - self.start_allocated)
            })
        
        # CPU memory metrics
        if self.has_psutil:
            metrics.update({
                "cpu_memory_start_mb": bytes_to_mb(self.start_cpu_memory),
                "cpu_memory_end_mb": bytes_to_mb(self.end_cpu_memory),
                "cpu_memory_growth_mb": bytes_to_mb(self.end_cpu_memory - self.start_cpu_memory)
            })
        
        # Add memory_usage_mb for compatibility with existing tools
        metrics["memory_usage_mb"] = metrics.get("memory_peak_mb", 0) if self.can_track_device_memory else bytes_to_mb(self.end_cpu_memory)
        
        return metrics
    
    def get_memory_timeline(self) -> List[Dict[str, Any]]:
        """
        Get the memory usage timeline.
        
        Returns:
            List of memory usage measurements over time
        """
        # Convert from bytes to MB
        bytes_to_mb = lambda x: x / (1024 * 1024)
        
        timeline = []
        for timestamp, allocated, reserved, cpu_memory in self.memory_timeline:
            entry = {
                "timestamp": timestamp
            }
            
            if self.can_track_device_memory:
                entry.update({
                    "allocated_mb": bytes_to_mb(allocated),
                    "reserved_mb": bytes_to_mb(reserved)
                })
            
            if self.has_psutil:
                entry["cpu_memory_mb"] = bytes_to_mb(cpu_memory)
            
            timeline.append(entry)
        
        return timeline

class MemoryMetricFactory:
    """Factory class for creating appropriate memory metrics based on hardware."""
    
    @staticmethod
    def create(device: Any) -> MemoryMetric:
        """
        Create a memory metric for the specified device.
        
        Args:
            device: PyTorch device or device object
            
        Returns:
            MemoryMetric instance configured for the device
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
        
        return MemoryMetric(device_type)