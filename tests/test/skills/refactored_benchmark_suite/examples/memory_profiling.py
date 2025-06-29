#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory profiling example.

This script demonstrates how to use the hardware abstraction layer
and memory metrics to profile model memory usage on different hardware.
"""

import os
import sys
import logging
import json
from pathlib import Path

# Add parent directory to Python path
parent_dir = str(Path(__file__).parent.parent.absolute())
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
import hardware
from metrics.memory import MemoryMetric, MemoryMetricFactory
from models import get_model_adapter

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def profile_model_memory(model_id, task, hardware_type, batch_size=1, sequence_length=128):
    """
    Profile memory usage of a model on the specified hardware.
    
    Args:
        model_id: HuggingFace model ID
        task: Model task
        hardware_type: Hardware platform to use
        batch_size: Batch size for inference
        sequence_length: Sequence length for inputs
    
    Returns:
        Memory metrics dictionary
    """
    logger.info(f"Profiling {model_id} on {hardware_type}...")
    
    # Initialize hardware
    device = hardware.initialize_hardware(hardware_type)
    if device is None:
        logger.error(f"Failed to initialize {hardware_type}")
        return None
    
    # Create memory metric using factory
    memory_metric = MemoryMetricFactory.create(device)
    
    # Create model adapter
    adapter = get_model_adapter(model_id, task)
    
    try:
        # Start measuring memory
        memory_metric.start()
        
        # Load model
        logger.info(f"Loading model {model_id}...")
        model = adapter.load_model(device)
        
        # Record memory after model loading
        memory_metric.record_memory()
        
        # Prepare inputs
        logger.info("Preparing inputs...")
        inputs = adapter.prepare_inputs(batch_size, sequence_length)
        
        # Record memory after input preparation
        memory_metric.record_memory()
        
        # Run inference
        logger.info("Running inference...")
        with torch.no_grad():
            # Move inputs to device if necessary
            if isinstance(inputs, dict):
                device_inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            else:
                device_inputs = inputs.to(device) if hasattr(inputs, 'to') else inputs
            
            # Run inference
            model(**device_inputs) if isinstance(device_inputs, dict) else model(device_inputs)
        
        # Record memory after inference
        memory_metric.record_memory()
        
        # Stop measuring memory
        memory_metric.stop()
        
        # Get metrics
        metrics = memory_metric.get_metrics()
        timeline = memory_metric.get_memory_timeline()
        
        # Add additional info to metrics
        metrics["model_id"] = model_id
        metrics["task"] = task
        metrics["hardware"] = hardware_type
        metrics["batch_size"] = batch_size
        metrics["sequence_length"] = sequence_length
        
        # Get memory timeline labels
        timeline_labels = ["start", "after_load", "after_inputs", "after_inference", "end"]
        if len(timeline) == len(timeline_labels):
            labeled_timeline = []
            for i, entry in enumerate(timeline):
                labeled_entry = entry.copy()
                labeled_entry["stage"] = timeline_labels[i]
                labeled_timeline.append(labeled_entry)
            metrics["memory_timeline"] = labeled_timeline
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error profiling model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
    finally:
        # Clean up
        if "model" in locals():
            del model
        if hardware_type == "cuda":
            torch.cuda.empty_cache()
        elif hardware_type == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        import gc
        gc.collect()

def profile_models_on_available_hardware(models, batch_sizes=None, sequence_lengths=None):
    """
    Profile multiple models on all available hardware.
    
    Args:
        models: List of (model_id, task) tuples
        batch_sizes: List of batch sizes to test
        sequence_lengths: List of sequence lengths to test
    
    Returns:
        Dictionary of profiling results
    """
    results = []
    
    # Default parameters
    if batch_sizes is None:
        batch_sizes = [1]
    if sequence_lengths is None:
        sequence_lengths = [128]
    
    # Get available hardware
    available_hardware = hardware.get_available_hardware()
    logger.info(f"Available hardware: {available_hardware}")
    
    # Profile each model on each hardware
    for model_id, task in models:
        for hardware_type in available_hardware:
            for batch_size in batch_sizes:
                for sequence_length in sequence_lengths:
                    # Skip OpenVINO, WebNN, and WebGPU for now (separate optimizations needed)
                    if hardware_type in ["openvino", "webnn", "webgpu"]:
                        continue
                    
                    metrics = profile_model_memory(
                        model_id, 
                        task, 
                        hardware_type, 
                        batch_size, 
                        sequence_length
                    )
                    
                    if metrics:
                        results.append(metrics)
    
    return results

def main():
    """Main entry point."""
    logger.info("Memory Profiling Example")
    logger.info("----------------------")
    
    # Define models to profile
    models = [
        ("bert-base-uncased", "fill-mask"),
        ("gpt2", "text-generation"),
        ("google/vit-base-patch16-224", "image-classification")
    ]
    
    # Define batch sizes and sequence lengths
    batch_sizes = [1, 2]
    sequence_lengths = [16, 32]
    
    # Profile models
    results = profile_models_on_available_hardware(
        models,
        batch_sizes,
        sequence_lengths
    )
    
    # Print summary
    logger.info("\nMemory Profiling Results Summary:")
    for result in results:
        logger.info(f"Model: {result['model_id']}, Hardware: {result['hardware']}, "
                    f"Batch: {result['batch_size']}, Seq: {result['sequence_length']}, "
                    f"Peak Memory: {result['memory_peak_mb']:.2f} MB")
    
    # Save results to file
    with open("memory_profiling_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to memory_profiling_results.json")

if __name__ == "__main__":
    main()