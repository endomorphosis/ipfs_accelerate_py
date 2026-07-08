#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full integration example demonstrating the integration between hardware abstraction layer,
metrics system, and benchmark runner components.

This script shows how the components of the refactored benchmark suite work together
to provide comprehensive model benchmarking.
"""

import os
import time
import logging
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("integration_example")

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import components
from hardware import (
    initialize_hardware, 
    get_available_hardware,
    get_hardware_info,
    CPUBackend,
    CUDABackend
)

from metrics.timing import (
    LatencyMetric,
    ThroughputMetric,
    TimingMetricFactory
)

from metrics.memory import (
    MemoryMetric,
    MemoryMetricFactory
)

from metrics.flops import (
    FLOPsMetric,
    FLOPsMetricFactory
)

# Define a simple model for testing
class SimpleModel(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)


def run_component_integration_test():
    """
    Test integration between hardware abstraction and metrics components directly.
    """
    logger.info("=== Testing component integration ===")
    
    # Get available hardware
    hardware_types = get_available_hardware()
    logger.info(f"Available hardware: {hardware_types}")
    
    # Create temp directory for results
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test each available hardware
        for hw_type in hardware_types:
            if hw_type not in ["cpu", "cuda"]:
                # Skip specialized hardware for this example
                continue
                
            logger.info(f"Testing on {hw_type.upper()}")
            
            try:
                # 1. Initialize hardware
                device = initialize_hardware(hw_type)
                logger.info(f"Initialized {hw_type} device: {device}")
                
                # 2. Create model
                model = SimpleModel()
                if hw_type == "cuda":
                    model = model.cuda()
                model.eval()
                
                # 3. Create inputs
                batch_size = 4
                x = torch.randn(batch_size, 64)
                if hw_type == "cuda":
                    x = x.cuda()
                
                # 4. Create metrics using factories
                logger.info("Creating hardware-aware metrics")
                latency_metric = TimingMetricFactory.create_latency_metric(device)
                throughput_metric = TimingMetricFactory.create_throughput_metric(device, batch_size=batch_size)
                memory_metric = MemoryMetricFactory.create(device)
                flops_metric = FLOPsMetricFactory.create(device)
                flops_metric.set_model_and_inputs(model, x)
                
                # 5. Start metrics
                logger.info("Starting metrics collection")
                latency_metric.start()
                throughput_metric.start()
                memory_metric.start()
                flops_metric.start()
                
                # 6. Run inference
                logger.info("Running inference")
                with torch.no_grad():
                    # Warmup
                    for _ in range(3):
                        _ = model(x)
                    
                    # Benchmark
                    for i in range(10):
                        # Record step start for latency
                        latency_metric.record_step()
                        
                        # Run model
                        output = model(x)
                        
                        # Update throughput
                        throughput_metric.update()
                        
                        # Record memory every few steps
                        if i % 2 == 0:
                            memory_metric.record_memory()
                
                # 7. Stop metrics
                logger.info("Stopping metrics collection")
                latency_metric.stop()
                throughput_metric.stop()
                memory_metric.stop()
                flops_metric.stop()
                
                # 8. Get metrics
                latency_metrics = latency_metric.get_metrics()
                throughput_metrics = throughput_metric.get_metrics()
                memory_metrics = memory_metric.get_metrics()
                flops_metrics = flops_metric.get_metrics()
                
                # 9. Print results
                logger.info(f"Latency metrics: {latency_metrics}")
                logger.info(f"Throughput metrics: {throughput_metrics}")
                logger.info(f"Memory metrics: {memory_metrics}")
                logger.info(f"FLOPs metrics: {flops_metrics}")
                
                # 10. Get detailed metrics for visualization (just as an example)
                latency_distribution = latency_metric.get_latency_distribution()
                memory_timeline = memory_metric.get_memory_timeline()
                
                logger.info(f"Number of latency measurements: {len(latency_distribution['latencies_ms'])}")
                logger.info(f"Number of memory timeline points: {len(memory_timeline)}")
                
            except Exception as e:
                logger.error(f"Error testing {hw_type}: {e}")


if __name__ == "__main__":
    # Don't run with CUDA if not available
    if not torch.cuda.is_available():
        logger.info("CUDA not available, testing only on CPU")
    
    # Run component integration test
    run_component_integration_test()