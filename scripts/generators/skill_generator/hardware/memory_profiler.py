#!/usr/bin/env python3
"""
Memory Profiler for HuggingFace Skill Generator

This module provides comprehensive memory profiling capabilities across
different hardware backends (CPU, CUDA, ROCm, MPS, OpenVINO, QNN).

Features:
- Peak memory usage tracking
- Per-operation memory profiling
- Memory leak detection
- Cross-hardware support
- Integration with performance baselines
"""

import os
import sys
import time
import psutil
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Snapshot of memory usage at a specific point in time."""
    timestamp: float
    cpu_ram_used_mb: float
    cpu_ram_available_mb: float
    cpu_ram_percent: float
    gpu_vram_used_mb: Optional[float] = None
    gpu_vram_total_mb: Optional[float] = None
    gpu_vram_percent: Optional[float] = None
    device_name: str = "cpu"
    operation_name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class MemoryProfile:
    """Complete memory profile for an operation or model."""
    operation_name: str
    hardware_type: str
    start_snapshot: MemorySnapshot
    end_snapshot: MemorySnapshot
    peak_snapshot: MemorySnapshot
    duration_seconds: float
    memory_delta_mb: float
    peak_memory_mb: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "operation_name": self.operation_name,
            "hardware_type": self.hardware_type,
            "start_memory_mb": self.start_snapshot.cpu_ram_used_mb,
            "end_memory_mb": self.end_snapshot.cpu_ram_used_mb,
            "peak_memory_mb": self.peak_memory_mb,
            "memory_delta_mb": self.memory_delta_mb,
            "duration_seconds": self.duration_seconds,
            "start_timestamp": self.start_snapshot.timestamp,
            "end_timestamp": self.end_snapshot.timestamp
        }


class MemoryProfiler:
    """
    Cross-platform memory profiler for HuggingFace models.
    
    Supports CPU RAM, CUDA VRAM, ROCm VRAM, and MPS memory tracking.
    """
    
    def __init__(self, hardware_type: str = "cpu", device_id: int = 0):
        """
        Initialize the memory profiler.
        
        Args:
            hardware_type: Target hardware (cpu, cuda, rocm, mps, openvino, qnn)
            device_id: GPU device ID (for multi-GPU systems)
        """
        self.hardware_type = hardware_type.lower()
        self.device_id = device_id
        self.snapshots: List[MemorySnapshot] = []
        self.profiles: List[MemoryProfile] = []
        self._monitoring = False
        self._monitor_thread = None
        self._peak_memory = 0.0
        
        # Detect available memory tracking libraries
        self.has_torch = self._check_torch()
        self.has_pynvml = self._check_pynvml()
        
        logger.info(f"Memory Profiler initialized for {hardware_type} (device {device_id})")
    
    def _check_torch(self) -> bool:
        """Check if PyTorch is available."""
        try:
            import torch
            return True
        except ImportError:
            return False
    
    def _check_pynvml(self) -> bool:
        """Check if pynvml (NVIDIA Management Library) is available."""
        try:
            import pynvml
            pynvml.nvmlInit()
            return True
        except (ImportError, Exception):
            return False
    
    def get_current_snapshot(self, operation_name: Optional[str] = None) -> MemorySnapshot:
        """
        Get a snapshot of current memory usage.
        
        Args:
            operation_name: Optional name of the operation being profiled
        
        Returns:
            MemorySnapshot object with current memory state
        """
        # CPU RAM (always available)
        memory = psutil.virtual_memory()
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            cpu_ram_used_mb=memory.used / (1024 ** 2),
            cpu_ram_available_mb=memory.available / (1024 ** 2),
            cpu_ram_percent=memory.percent,
            device_name=self.hardware_type,
            operation_name=operation_name
        )
        
        # GPU VRAM (if available)
        if self.hardware_type == "cuda" and self.has_torch:
            try:
                import torch
                if torch.cuda.is_available():
                    snapshot.gpu_vram_used_mb = torch.cuda.memory_allocated(self.device_id) / (1024 ** 2)
                    snapshot.gpu_vram_total_mb = torch.cuda.get_device_properties(self.device_id).total_memory / (1024 ** 2)
                    snapshot.gpu_vram_percent = (snapshot.gpu_vram_used_mb / snapshot.gpu_vram_total_mb) * 100
            except Exception as e:
                logger.warning(f"Failed to get CUDA memory stats: {e}")
        
        elif self.hardware_type == "rocm" and self.has_torch:
            try:
                import torch
                if hasattr(torch, 'hip') and torch.hip.is_available():
                    snapshot.gpu_vram_used_mb = torch.hip.memory_allocated(self.device_id) / (1024 ** 2)
                    # ROCm doesn't have get_device_properties, use alternative
                    snapshot.gpu_vram_total_mb = None  # Not easily available
                    snapshot.gpu_vram_percent = None
            except Exception as e:
                logger.warning(f"Failed to get ROCm memory stats: {e}")
        
        elif self.hardware_type == "mps" and self.has_torch:
            try:
                import torch
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    # MPS memory tracking is limited
                    snapshot.gpu_vram_used_mb = torch.mps.current_allocated_memory() / (1024 ** 2) if hasattr(torch, 'mps') else None
            except Exception as e:
                logger.warning(f"Failed to get MPS memory stats: {e}")
        
        return snapshot
    
    @contextmanager
    def profile_operation(self, operation_name: str, enable_monitoring: bool = False):
        """
        Context manager to profile memory usage of an operation.
        
        Args:
            operation_name: Name of the operation being profiled
            enable_monitoring: Whether to continuously monitor memory during operation
        
        Yields:
            None
        
        Example:
            with profiler.profile_operation("model_inference"):
                output = model(inputs)
        """
        # Take start snapshot
        start_time = time.time()
        start_snapshot = self.get_current_snapshot(operation_name)
        self.snapshots.append(start_snapshot)
        
        # Start continuous monitoring if requested
        if enable_monitoring:
            self._start_monitoring(operation_name)
        
        try:
            yield
        finally:
            # Stop monitoring
            if enable_monitoring:
                self._stop_monitoring()
            
            # Take end snapshot
            end_time = time.time()
            end_snapshot = self.get_current_snapshot(operation_name)
            self.snapshots.append(end_snapshot)
            
            # Find peak memory
            peak_snapshot = self._find_peak_snapshot(start_time, end_time)
            
            # Calculate statistics
            memory_delta = end_snapshot.cpu_ram_used_mb - start_snapshot.cpu_ram_used_mb
            peak_memory = peak_snapshot.cpu_ram_used_mb if peak_snapshot else end_snapshot.cpu_ram_used_mb
            
            # Create profile
            profile = MemoryProfile(
                operation_name=operation_name,
                hardware_type=self.hardware_type,
                start_snapshot=start_snapshot,
                end_snapshot=end_snapshot,
                peak_snapshot=peak_snapshot if peak_snapshot else end_snapshot,
                duration_seconds=end_time - start_time,
                memory_delta_mb=memory_delta,
                peak_memory_mb=peak_memory
            )
            
            self.profiles.append(profile)
            
            # Log summary
            logger.info(f"Memory profile for '{operation_name}':")
            logger.info(f"  Duration: {profile.duration_seconds:.2f}s")
            logger.info(f"  Memory delta: {memory_delta:+.2f} MB")
            logger.info(f"  Peak memory: {peak_memory:.2f} MB")
            
            if end_snapshot.gpu_vram_used_mb is not None:
                gpu_delta = end_snapshot.gpu_vram_used_mb - start_snapshot.gpu_vram_used_mb
                logger.info(f"  GPU memory delta: {gpu_delta:+.2f} MB")
    
    def _start_monitoring(self, operation_name: str):
        """Start continuous memory monitoring in a background thread."""
        self._monitoring = True
        self._peak_memory = 0.0
        
        def monitor():
            while self._monitoring:
                snapshot = self.get_current_snapshot(operation_name)
                self.snapshots.append(snapshot)
                self._peak_memory = max(self._peak_memory, snapshot.cpu_ram_used_mb)
                time.sleep(0.1)  # Sample every 100ms
        
        self._monitor_thread = threading.Thread(target=monitor, daemon=True)
        self._monitor_thread.start()
    
    def _stop_monitoring(self):
        """Stop continuous memory monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
            self._monitor_thread = None
    
    def _find_peak_snapshot(self, start_time: float, end_time: float) -> Optional[MemorySnapshot]:
        """Find the snapshot with peak memory usage within a time range."""
        relevant_snapshots = [
            s for s in self.snapshots
            if start_time <= s.timestamp <= end_time
        ]
        
        if not relevant_snapshots:
            return None
        
        return max(relevant_snapshots, key=lambda s: s.cpu_ram_used_mb)
    
    def detect_memory_leaks(self, threshold_mb: float = 10.0) -> List[MemoryProfile]:
        """
        Detect potential memory leaks by analyzing profiles.
        
        Args:
            threshold_mb: Memory delta threshold to consider as potential leak
        
        Returns:
            List of MemoryProfile objects with suspected leaks
        """
        suspected_leaks = []
        
        for profile in self.profiles:
            # A leak is suspected if memory increased significantly and wasn't released
            if profile.memory_delta_mb > threshold_mb:
                suspected_leaks.append(profile)
                logger.warning(
                    f"Potential memory leak detected in '{profile.operation_name}': "
                    f"+{profile.memory_delta_mb:.2f} MB"
                )
        
        return suspected_leaks
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all memory profiling data.
        
        Returns:
            Dictionary with memory statistics
        """
        if not self.profiles:
            return {"message": "No profiles available"}
        
        total_delta = sum(p.memory_delta_mb for p in self.profiles)
        peak_memory = max(p.peak_memory_mb for p in self.profiles)
        avg_duration = sum(p.duration_seconds for p in self.profiles) / len(self.profiles)
        
        return {
            "hardware_type": self.hardware_type,
            "device_id": self.device_id,
            "num_profiles": len(self.profiles),
            "total_memory_delta_mb": total_delta,
            "peak_memory_mb": peak_memory,
            "average_duration_seconds": avg_duration,
            "profiles": [p.to_dict() for p in self.profiles]
        }
    
    def reset(self):
        """Reset all profiling data."""
        self.snapshots.clear()
        self.profiles.clear()
        self._peak_memory = 0.0
        logger.info("Memory profiler reset")
    
    def export_profile(self, filepath: str):
        """
        Export profiling data to a JSON file.
        
        Args:
            filepath: Path to output file
        """
        import json
        
        summary = self.get_memory_summary()
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Memory profile exported to {filepath}")
    
    def get_current_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage as a simple dictionary.
        
        Returns:
            Dictionary with current memory usage
        """
        snapshot = self.get_current_snapshot()
        
        result = {
            "cpu_ram_used_mb": snapshot.cpu_ram_used_mb,
            "cpu_ram_percent": snapshot.cpu_ram_percent,
            "timestamp": snapshot.timestamp
        }
        
        if snapshot.gpu_vram_used_mb is not None:
            result["gpu_vram_used_mb"] = snapshot.gpu_vram_used_mb
            result["gpu_vram_percent"] = snapshot.gpu_vram_percent
        
        return result


class MemoryBudgetManager:
    """
    Manager for handling memory budgets and constraints.
    
    Helps determine if operations can fit within available memory.
    """
    
    def __init__(self, hardware_type: str = "cpu", safety_margin: float = 0.2):
        """
        Initialize memory budget manager.
        
        Args:
            hardware_type: Target hardware
            safety_margin: Safety margin as fraction (0.2 = 20% buffer)
        """
        self.hardware_type = hardware_type
        self.safety_margin = safety_margin
        self.profiler = MemoryProfiler(hardware_type)
    
    def get_available_memory_mb(self) -> float:
        """
        Get available memory considering safety margin.
        
        Returns:
            Available memory in MB
        """
        snapshot = self.profiler.get_current_snapshot()
        available = snapshot.cpu_ram_available_mb
        
        # Apply safety margin
        usable = available * (1.0 - self.safety_margin)
        
        if snapshot.gpu_vram_total_mb is not None and snapshot.gpu_vram_used_mb is not None:
            gpu_available = snapshot.gpu_vram_total_mb - snapshot.gpu_vram_used_mb
            gpu_usable = gpu_available * (1.0 - self.safety_margin)
            return min(usable, gpu_usable)  # Return the more constrained one
        
        return usable
    
    def can_fit_model(self, estimated_model_size_mb: float) -> bool:
        """
        Check if a model can fit in available memory.
        
        Args:
            estimated_model_size_mb: Estimated model size in MB
        
        Returns:
            True if model can fit, False otherwise
        """
        available = self.get_available_memory_mb()
        can_fit = estimated_model_size_mb <= available
        
        if can_fit:
            logger.info(f"Model ({estimated_model_size_mb:.0f} MB) can fit in available memory ({available:.0f} MB)")
        else:
            logger.warning(f"Model ({estimated_model_size_mb:.0f} MB) may not fit in available memory ({available:.0f} MB)")
        
        return can_fit
    
    def recommend_batch_size(
        self,
        per_sample_memory_mb: float,
        max_batch_size: int = 128
    ) -> int:
        """
        Recommend a batch size based on available memory.
        
        Args:
            per_sample_memory_mb: Memory usage per sample in MB
            max_batch_size: Maximum desired batch size
        
        Returns:
            Recommended batch size
        """
        available = self.get_available_memory_mb()
        
        # Calculate how many samples can fit
        max_samples = int(available / per_sample_memory_mb)
        
        # Use smaller of calculated and desired max
        recommended = min(max_samples, max_batch_size)
        
        # Ensure at least 1
        recommended = max(1, recommended)
        
        logger.info(f"Recommended batch size: {recommended} (based on {available:.0f} MB available)")
        
        return recommended


def profile_model_loading(model_loader_func, *args, **kwargs):
    """
    Decorator function to profile memory usage of model loading.
    
    Args:
        model_loader_func: Function that loads the model
        *args, **kwargs: Arguments to pass to the loader function
    
    Returns:
        Tuple of (loaded_model, memory_profile)
    """
    profiler = MemoryProfiler()
    
    with profiler.profile_operation("model_loading", enable_monitoring=True):
        model = model_loader_func(*args, **kwargs)
    
    profile = profiler.profiles[-1] if profiler.profiles else None
    
    return model, profile


if __name__ == "__main__":
    # Example usage
    print("=== Memory Profiler Examples ===\n")
    
    # CPU profiling
    print("CPU Memory Profiling:")
    cpu_profiler = MemoryProfiler("cpu")
    
    with cpu_profiler.profile_operation("test_operation"):
        # Simulate some work
        data = [i for i in range(1000000)]
        time.sleep(0.1)
    
    print(f"Memory summary: {cpu_profiler.get_memory_summary()}\n")
    
    # Memory budget management
    print("Memory Budget Management:")
    budget_manager = MemoryBudgetManager("cpu")
    available = budget_manager.get_available_memory_mb()
    print(f"  Available memory: {available:.0f} MB")
    
    can_fit = budget_manager.can_fit_model(1024)
    print(f"  Can fit 1GB model: {can_fit}")
    
    batch_size = budget_manager.recommend_batch_size(per_sample_memory_mb=10)
    print(f"  Recommended batch size: {batch_size}")
