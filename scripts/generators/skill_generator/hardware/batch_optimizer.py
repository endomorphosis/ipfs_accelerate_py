#!/usr/bin/env python3
"""
Batch Size Optimizer for HuggingFace Skill Generator

This module provides automatic batch size optimization based on:
- Available memory (CPU RAM / GPU VRAM)
- Hardware capabilities
- Model characteristics
- Performance profiling

Features:
- Auto-tuning with binary search
- Memory-based calculation
- Hardware-specific profiling
- Dynamic batch size adjustment
- Benchmark database storage
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BatchSizeProfile:
    """Profile for a specific batch size."""
    batch_size: int
    throughput_samples_per_sec: float
    latency_ms: float
    memory_used_mb: float
    memory_peak_mb: float
    success: bool
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class OptimalBatchSize:
    """Optimal batch size result."""
    batch_size: int
    throughput: float
    latency_ms: float
    memory_used_mb: float
    utilization_percent: float
    recommendation: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class BatchOptimizer:
    """
    Automatic batch size optimizer for model inference.
    
    Finds the optimal batch size that maximizes throughput while
    staying within memory constraints.
    """
    
    def __init__(
        self,
        hardware_type: str = "cpu",
        device_id: int = 0,
        safety_margin: float = 0.15
    ):
        """
        Initialize the batch optimizer.
        
        Args:
            hardware_type: Target hardware (cpu, cuda, rocm, mps, etc.)
            device_id: GPU device ID
            safety_margin: Memory safety margin (0.15 = 15% buffer)
        """
        self.hardware_type = hardware_type.lower()
        self.device_id = device_id
        self.safety_margin = safety_margin
        self.profiles: List[BatchSizeProfile] = []
        self.cache_file = f".batch_cache_{hardware_type}_{device_id}.json"
        self.cache = self._load_cache()
        
        logger.info(f"Batch Optimizer initialized for {hardware_type} (device {device_id})")
    
    def _load_cache(self) -> Dict[str, Any]:
        """Load cached batch size recommendations."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    cache = json.load(f)
                logger.info(f"Loaded batch cache with {len(cache)} entries")
                return cache
            except Exception as e:
                logger.warning(f"Failed to load batch cache: {e}")
        return {}
    
    def _save_cache(self):
        """Save batch size recommendations to cache."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
            logger.info(f"Saved batch cache with {len(self.cache)} entries")
        except Exception as e:
            logger.warning(f"Failed to save batch cache: {e}")
    
    def get_available_memory_mb(self) -> float:
        """
        Get available memory in MB.
        
        Returns:
            Available memory in MB
        """
        import psutil
        
        if self.hardware_type == "cpu":
            memory = psutil.virtual_memory()
            available = memory.available / (1024 ** 2)
            return available * (1.0 - self.safety_margin)
        
        elif self.hardware_type == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    free, total = torch.cuda.mem_get_info(self.device_id)
                    available = free / (1024 ** 2)
                    return available * (1.0 - self.safety_margin)
            except Exception as e:
                logger.warning(f"Failed to get CUDA memory: {e}")
        
        elif self.hardware_type == "rocm":
            try:
                import torch
                if hasattr(torch, 'hip') and torch.hip.is_available():
                    # ROCm memory info
                    free = torch.hip.memory_allocated(self.device_id)
                    # Estimate available (ROCm doesn't have mem_get_info)
                    return 8000 * (1.0 - self.safety_margin)  # Conservative estimate
            except Exception as e:
                logger.warning(f"Failed to get ROCm memory: {e}")
        
        # Fallback
        memory = psutil.virtual_memory()
        return memory.available / (1024 ** 2) * (1.0 - self.safety_margin)
    
    def estimate_memory_per_sample(
        self,
        model_size_mb: float,
        input_size_mb: float,
        output_size_mb: float,
        activation_multiplier: float = 2.0
    ) -> float:
        """
        Estimate memory usage per sample.
        
        Args:
            model_size_mb: Model size in MB
            input_size_mb: Input tensor size in MB
            output_size_mb: Output tensor size in MB
            activation_multiplier: Multiplier for activation memory
        
        Returns:
            Estimated memory per sample in MB
        """
        # Memory = inputs + outputs + activations
        per_sample = input_size_mb + output_size_mb
        
        # Activations typically 2x input/output size
        per_sample *= activation_multiplier
        
        return per_sample
    
    def calculate_initial_batch_size(
        self,
        available_memory_mb: float,
        model_size_mb: float,
        per_sample_memory_mb: float
    ) -> int:
        """
        Calculate initial batch size estimate.
        
        Args:
            available_memory_mb: Available memory
            model_size_mb: Model size
            per_sample_memory_mb: Memory per sample
        
        Returns:
            Estimated batch size
        """
        # Memory budget = available - model size
        memory_budget = available_memory_mb - model_size_mb
        
        if memory_budget <= 0:
            logger.warning("Model size exceeds available memory!")
            return 1
        
        # Calculate batch size
        batch_size = int(memory_budget / per_sample_memory_mb)
        
        # Ensure minimum of 1
        batch_size = max(1, batch_size)
        
        # Round to nice numbers (powers of 2)
        batch_size = 2 ** int(math.log2(batch_size)) if batch_size > 1 else 1
        
        logger.info(f"Initial batch size estimate: {batch_size}")
        return batch_size
    
    def profile_batch_size(
        self,
        batch_size: int,
        inference_func: Callable,
        num_iterations: int = 3
    ) -> BatchSizeProfile:
        """
        Profile inference with a specific batch size.
        
        Args:
            batch_size: Batch size to test
            inference_func: Function that runs inference (takes batch_size arg)
            num_iterations: Number of iterations to average
        
        Returns:
            BatchSizeProfile with results
        """
        try:
            # Import memory profiler for tracking
            from .memory_profiler import MemoryProfiler
            
            profiler = MemoryProfiler(self.hardware_type, self.device_id)
            
            latencies = []
            memory_used = []
            memory_peak = []
            
            for i in range(num_iterations):
                with profiler.profile_operation(f"batch_{batch_size}_iter_{i}"):
                    start_time = time.time()
                    inference_func(batch_size)
                    latency = (time.time() - start_time) * 1000  # ms
                    latencies.append(latency)
                
                # Get memory stats from last profile
                if profiler.profiles:
                    profile = profiler.profiles[-1]
                    memory_used.append(profile.end_snapshot.cpu_ram_used_mb)
                    memory_peak.append(profile.peak_memory_mb)
            
            # Calculate averages
            avg_latency = sum(latencies) / len(latencies)
            avg_memory = sum(memory_used) / len(memory_used) if memory_used else 0
            avg_peak = sum(memory_peak) / len(memory_peak) if memory_peak else 0
            
            # Calculate throughput
            throughput = (batch_size * 1000) / avg_latency if avg_latency > 0 else 0
            
            return BatchSizeProfile(
                batch_size=batch_size,
                throughput_samples_per_sec=throughput,
                latency_ms=avg_latency,
                memory_used_mb=avg_memory,
                memory_peak_mb=avg_peak,
                success=True
            )
        
        except Exception as e:
            logger.warning(f"Failed to profile batch size {batch_size}: {e}")
            return BatchSizeProfile(
                batch_size=batch_size,
                throughput_samples_per_sec=0.0,
                latency_ms=0.0,
                memory_used_mb=0.0,
                memory_peak_mb=0.0,
                success=False,
                error=str(e)
            )
    
    def find_optimal_batch_size(
        self,
        inference_func: Callable,
        model_size_mb: float,
        per_sample_memory_mb: float,
        max_batch_size: int = 256,
        min_batch_size: int = 1
    ) -> OptimalBatchSize:
        """
        Find optimal batch size using binary search.
        
        Args:
            inference_func: Function to run inference (takes batch_size)
            model_size_mb: Model size in MB
            per_sample_memory_mb: Memory per sample in MB
            max_batch_size: Maximum batch size to try
            min_batch_size: Minimum batch size
        
        Returns:
            OptimalBatchSize with recommendation
        """
        available_memory = self.get_available_memory_mb()
        
        # Calculate initial estimate
        initial_batch = self.calculate_initial_batch_size(
            available_memory,
            model_size_mb,
            per_sample_memory_mb
        )
        
        # Cap at max
        initial_batch = min(initial_batch, max_batch_size)
        
        # Binary search for optimal batch size
        left, right = min_batch_size, initial_batch
        best_batch_size = min_batch_size
        best_throughput = 0.0
        
        logger.info(f"Searching for optimal batch size between {left} and {right}")
        
        while left <= right:
            mid = (left + right) // 2
            
            # Test this batch size
            profile = self.profile_batch_size(mid, inference_func)
            self.profiles.append(profile)
            
            if not profile.success:
                # OOM or error, try smaller batch
                right = mid - 1
                logger.info(f"Batch {mid} failed, trying smaller")
            else:
                # Success, this is a candidate
                if profile.throughput_samples_per_sec > best_throughput:
                    best_batch_size = mid
                    best_throughput = profile.throughput_samples_per_sec
                
                # Try larger batch
                left = mid + 1
                logger.info(f"Batch {mid} succeeded (throughput: {profile.throughput_samples_per_sec:.1f}), trying larger")
        
        # Get best profile
        best_profile = next(
            (p for p in self.profiles if p.batch_size == best_batch_size and p.success),
            None
        )
        
        if not best_profile:
            # Fallback to minimum
            logger.warning("No successful profile found, using minimum batch size")
            best_profile = self.profile_batch_size(min_batch_size, inference_func)
            best_batch_size = min_batch_size
        
        # Calculate utilization
        utilization = (best_profile.memory_peak_mb / available_memory) * 100
        
        # Generate recommendation
        if utilization < 50:
            recommendation = "Memory under-utilized, could use larger batch"
        elif utilization > 90:
            recommendation = "High memory utilization, near optimal"
        else:
            recommendation = "Good balance of throughput and memory"
        
        result = OptimalBatchSize(
            batch_size=best_batch_size,
            throughput=best_profile.throughput_samples_per_sec,
            latency_ms=best_profile.latency_ms,
            memory_used_mb=best_profile.memory_peak_mb,
            utilization_percent=utilization,
            recommendation=recommendation
        )
        
        logger.info(f"Optimal batch size: {best_batch_size}")
        logger.info(f"Throughput: {result.throughput:.1f} samples/sec")
        logger.info(f"Latency: {result.latency_ms:.1f} ms")
        logger.info(f"Memory utilization: {utilization:.1f}%")
        
        return result
    
    def get_cached_batch_size(self, model_id: str) -> Optional[int]:
        """
        Get cached batch size for a model.
        
        Args:
            model_id: Model identifier
        
        Returns:
            Cached batch size or None
        """
        cache_key = f"{model_id}_{self.hardware_type}_{self.device_id}"
        cached = self.cache.get(cache_key)
        
        if cached:
            logger.info(f"Using cached batch size {cached['batch_size']} for {model_id}")
            return cached['batch_size']
        
        return None
    
    def cache_batch_size(self, model_id: str, optimal: OptimalBatchSize):
        """
        Cache optimal batch size for a model.
        
        Args:
            model_id: Model identifier
            optimal: Optimal batch size result
        """
        cache_key = f"{model_id}_{self.hardware_type}_{self.device_id}"
        self.cache[cache_key] = optimal.to_dict()
        self._save_cache()
        logger.info(f"Cached batch size {optimal.batch_size} for {model_id}")
    
    def get_profiles_summary(self) -> Dict[str, Any]:
        """
        Get summary of all batch size profiles.
        
        Returns:
            Summary dictionary
        """
        successful = [p for p in self.profiles if p.success]
        failed = [p for p in self.profiles if not p.success]
        
        if not successful:
            return {"message": "No successful profiles"}
        
        return {
            "total_profiles": len(self.profiles),
            "successful": len(successful),
            "failed": len(failed),
            "best_throughput": max(p.throughput_samples_per_sec for p in successful),
            "profiles": [p.to_dict() for p in self.profiles]
        }
    
    def recommend_for_workload(
        self,
        workload_type: str,
        available_memory_mb: Optional[float] = None
    ) -> int:
        """
        Get batch size recommendation for a workload type.
        
        Args:
            workload_type: Type of workload (realtime, throughput, batch)
            available_memory_mb: Available memory (auto-detected if None)
        
        Returns:
            Recommended batch size
        """
        if available_memory_mb is None:
            available_memory_mb = self.get_available_memory_mb()
        
        # Workload-specific recommendations
        if workload_type == "realtime":
            # Prioritize low latency
            return 1
        
        elif workload_type == "throughput":
            # Maximize throughput, use larger batches
            # Rough estimate: 1 sample per 100MB available
            batch = int(available_memory_mb / 100)
            return max(8, min(batch, 128))
        
        elif workload_type == "batch":
            # Large batch processing
            # Use 80% of available memory
            batch = int(available_memory_mb / 50)
            return max(16, min(batch, 256))
        
        else:
            # Default: balanced
            batch = int(available_memory_mb / 75)
            return max(4, min(batch, 64))


def auto_tune_batch_size(
    model_inference_func: Callable,
    hardware_type: str = "cuda",
    model_size_mb: float = 1000,
    per_sample_memory_mb: float = 10
) -> int:
    """
    Convenience function to auto-tune batch size.
    
    Args:
        model_inference_func: Function that runs model inference
        hardware_type: Target hardware
        model_size_mb: Model size in MB
        per_sample_memory_mb: Memory per sample in MB
    
    Returns:
        Optimal batch size
    """
    optimizer = BatchOptimizer(hardware_type)
    
    optimal = optimizer.find_optimal_batch_size(
        model_inference_func,
        model_size_mb,
        per_sample_memory_mb
    )
    
    return optimal.batch_size


if __name__ == "__main__":
    # Example usage
    print("=== Batch Optimizer Examples ===\n")
    
    # Create optimizer
    optimizer = BatchOptimizer("cpu")
    
    # Get available memory
    available = optimizer.get_available_memory_mb()
    print(f"Available memory: {available:.0f} MB\n")
    
    # Estimate initial batch size
    batch_size = optimizer.calculate_initial_batch_size(
        available_memory_mb=available,
        model_size_mb=500,
        per_sample_memory_mb=10
    )
    print(f"Estimated batch size: {batch_size}\n")
    
    # Workload recommendations
    for workload in ["realtime", "throughput", "batch"]:
        batch = optimizer.recommend_for_workload(workload)
        print(f"Recommended for {workload}: {batch}")
