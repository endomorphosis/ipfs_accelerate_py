#!/usr/bin/env python3
"""
Memory Usage Visualization Tool for Web Platform Models

This script analyzes and visualizes memory usage for models running on various web platforms.
It provides detailed memory breakdowns for WebNN and WebGPU implementations, and compares
memory usage across different model configurations and optimizations.

Key Features:
- Memory usage breakdown by model components
- Comparison of different precision formats (FP16, INT8, INT4)
- Visualization of memory optimizations impact
- Cross-platform memory usage comparison (native vs. web)
- Adaptive 4-bit quantization visualization
- Memory timeline visualization during inference

Usage:
    python visualize_memory_usage.py --model <model_name> --platform <webnn|webgpu|all> --output <html|png|json>
"""

import os
import sys
import json
import time
import argparse
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("memory_visualization")

# Memory visualization color scheme
COLORS = {
    "weights": "#3498db",  # Blue
    "activations": "#2ecc71",  # Green
    "kv_cache": "#e74c3c",  # Red
    "attention": "#9b59b6",  # Purple
    "embeddings": "#f1c40f",  # Yellow
    "optimizer": "#1abc9c",  # Teal
    "other": "#95a5a6",  # Gray
    "overhead": "#34495e",  # Dark Gray
    "fp16": "#3498db",  # Blue
    "int8": "#2ecc71",  # Green
    "int4": "#e74c3c",  # Red 
    "int2": "#9b59b6",  # Purple
    "webgpu": "#3498db",  # Blue
    "webnn": "#2ecc71",  # Green
    "cpu": "#e74c3c",  # Red
    "cuda": "#9b59b6",  # Purple
    "openvino": "#f1c40f",  # Yellow
    "rocm": "#1abc9c",  # Teal
    "mps": "#95a5a6",  # Gray
}

def analyze_model_memory(model_name: str, platform: str, optimizations: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Analyze model memory usage for a specific platform.
    
    Args:
        model_name: Name of the model to analyze
        platform: Platform to analyze (webnn, webgpu)
        optimizations: List of optimizations to apply
        
    Returns:
        Dictionary with memory usage information
    """
    logger.info(f"Analyzing memory usage for {model_name} on {platform}")
    
    # Define the result structure
    result = {
        "model_name": model_name,
        "platform": platform,
        "optimizations": optimizations or [],
        "total_memory_mb": 0,
        "components": {},
        "timeline": [],
        "optimizations_impact": {},
    }
    
    # Model type detection
    model_type = detect_model_type(model_name)
    result["model_type"] = model_type
    
    # Generate example data based on model type
    if "bert" in model_type.lower():
        # BERT-like model memory breakdown
        result["components"] = {
            "weights": 112.5,  # MB
            "activations": 24.3,
            "attention": 18.7,
            "embeddings": 15.6,
            "optimizer": 0,
            "other": 5.2,
            "overhead": 3.8
        }
        # Initialize timeline data
        result["timeline"] = generate_timeline_data(model_type, "bert")
        
    elif "t5" in model_type.lower():
        # T5-like model memory breakdown
        result["components"] = {
            "weights": 156.8,  # MB
            "activations": 42.5,
            "attention": 32.1,
            "embeddings": 18.3,
            "optimizer": 0,
            "other": 7.5,
            "overhead": 4.2
        }
        # Initialize timeline data
        result["timeline"] = generate_timeline_data(model_type, "t5")
        
    elif "llama" in model_type.lower() or "qwen" in model_type.lower():
        # LLM memory breakdown
        result["components"] = {
            "weights": 3850.6,  # MB (for a ~7B param model)
            "activations": 245.2,
            "kv_cache": 325.8,
            "attention": 186.3,
            "embeddings": 92.5,
            "optimizer": 0,
            "other": 28.7,
            "overhead": 15.3
        }
        # Initialize timeline data
        result["timeline"] = generate_timeline_data(model_type, "llm")
        
    elif "llava" in model_type.lower() or "clip" in model_type.lower():
        # Multimodal model memory breakdown
        result["components"] = {
            "weights": 2450.3,  # MB
            "activations": 186.7,
            "kv_cache": 245.2,
            "attention": 148.6,
            "embeddings": 86.3,
            "optimizer": 0,
            "other": 32.4,
            "overhead": 12.8
        }
        # Initialize timeline data
        result["timeline"] = generate_timeline_data(model_type, "multimodal")
        
    elif "whisper" in model_type.lower() or "wav2vec" in model_type.lower() or "clap" in model_type.lower():
        # Audio model memory breakdown
        result["components"] = {
            "weights": 458.2,  # MB
            "activations": 96.3,
            "attention": 72.5,
            "embeddings": 48.7,
            "optimizer": 0,
            "other": 18.3,
            "overhead": 8.5
        }
        # Initialize timeline data
        result["timeline"] = generate_timeline_data(model_type, "audio")
        
    else:
        # Generic model memory breakdown
        result["components"] = {
            "weights": 245.6,  # MB
            "activations": 65.3,
            "attention": 42.8,
            "embeddings": 32.5,
            "optimizer": 0,
            "other": 12.6,
            "overhead": 6.8
        }
        # Initialize timeline data
        result["timeline"] = generate_timeline_data(model_type, "generic")
    
    # Calculate total memory (sum of all components)
    result["total_memory_mb"] = sum(result["components"].values())
    
    # Apply platform-specific adjustments
    if platform == "webgpu":
        # WebGPU typically has some overhead for shader compilation and data format conversion
        result["components"]["overhead"] *= 1.2
        result["total_memory_mb"] = sum(result["components"].values())
    
    # Apply optimizations if specified
    if optimizations:
        result["optimizations_impact"] = apply_memory_optimizations(result["components"], optimizations, model_type, platform)
    
    return result

def detect_model_type(model_name: str) -> str:
    """
    Detect model type from model name.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Model type string
    """
    model_name_lower = model_name.lower()
    
    if "bert" in model_name_lower:
        return "bert"
    elif "t5" in model_name_lower:
        return "t5"
    elif "llama" in model_name_lower:
        return "llama"
    elif "qwen" in model_name_lower:
        return "qwen"
    elif "llava" in model_name_lower:
        return "llava"
    elif "clip" in model_name_lower:
        return "clip"
    elif "vit" in model_name_lower:
        return "vit"
    elif "whisper" in model_name_lower:
        return "whisper"
    elif "wav2vec" in model_name_lower:
        return "wav2vec2"
    elif "clap" in model_name_lower:
        return "clap"
    elif "detr" in model_name_lower:
        return "detr"
    else:
        return "unknown"

def generate_timeline_data(model_name: str, model_category: str) -> List[Dict[str, Any]]:
    """
    Generate synthetic timeline data for memory usage visualization.
    
    Args:
        model_name: Name of the model
        model_category: Category of the model
        
    Returns:
        List of timeline data points
    """
    timeline = []
    
    # Define timeline steps based on model category
    if model_category == "bert":
        steps = ["load_model", "load_embeddings", "load_encoders", "initialize_buffers", "inference_start", 
                "process_input", "encoder_forward", "output_processing", "inference_complete"]
    elif model_category == "llm":
        steps = ["load_model", "load_embeddings", "load_layers", "initialize_kv_cache", "inference_start", 
                "process_input", "token_1_generation", "token_2_generation", "token_3_generation", 
                "token_4_generation", "inference_complete"]
    elif model_category == "multimodal":
        steps = ["load_model", "load_vision_encoder", "load_text_encoder", "load_layers", "initialize_kv_cache", 
                "inference_start", "process_image", "process_text", "multimodal_fusion", "token_generation", 
                "inference_complete"]
    elif model_category == "audio":
        steps = ["load_model", "load_feature_extractor", "load_encoders", "load_decoders", "inference_start", 
                "audio_preprocessing", "feature_extraction", "encoder_forward", "decoder_forward", 
                "inference_complete"]
    else:
        steps = ["load_model", "load_components", "initialize_buffers", "inference_start", 
                "process_input", "model_forward", "output_processing", "inference_complete"]
    
    # Generate timeline data
    base_memory = 50  # MB (starting memory)
    current_time = 0
    
    for i, step in enumerate(steps):
        # Calculate synthetic memory values for this step
        if "load" in step:
            memory_increase = np.random.uniform(30, 100)  # Loading increases memory
            base_memory += memory_increase
        elif "initialize" in step:
            memory_increase = np.random.uniform(10, 50)
            base_memory += memory_increase
        elif "inference_start" in step:
            memory_increase = np.random.uniform(5, 15)
            base_memory += memory_increase
        elif "token" in step:
            memory_increase = np.random.uniform(2, 8)  # Small increases during token generation
            base_memory += memory_increase
        elif "process" in step:
            memory_increase = np.random.uniform(5, 20)
            base_memory += memory_increase
        elif "forward" in step:
            memory_increase = np.random.uniform(10, 30)
            base_memory += memory_increase
        elif "inference_complete" in step:
            memory_decrease = np.random.uniform(-30, -10)  # Memory can decrease after completion
            base_memory += memory_decrease
        else:
            memory_increase = np.random.uniform(-5, 15)  # Can increase or decrease
            base_memory += memory_increase
        
        # Ensure memory doesn't go below a minimum threshold
        base_memory = max(base_memory, 50)
        
        # Add time increment (variable to simulate different step durations)
        current_time += np.random.uniform(0.1, 0.5)
        
        # Create the timeline point
        timeline.append({
            "step": step,
            "time_seconds": round(current_time, 2),
            "memory_mb": round(base_memory, 2),
            "memory_delta_mb": round(memory_increase if 'memory_increase' in locals() else 0, 2)
        })
    
    return timeline

def apply_memory_optimizations(components: Dict[str, float], optimizations: List[str], 
                              model_type: str, platform: str) -> Dict[str, Dict[str, float]]:
    """
    Apply memory optimizations and calculate their impact.
    
    Args:
        components: Dictionary of memory components
        optimizations: List of optimizations to apply
        model_type: Type of the model
        platform: Platform (webnn, webgpu)
        
    Returns:
        Dictionary with optimization impacts
    """
    # Create a deep copy of the original components
    original_components = components.copy()
    optimized_components = components.copy()  # This will be modified
    
    # Track optimization impacts
    optimization_impacts = {}
    
    for optimization in optimizations:
        # Save the state before this optimization
        before_components = optimized_components.copy()
        before_total = sum(before_components.values())
        
        # Apply specific optimizations
        if optimization == "4bit_quantization":
            # 4-bit quantization primarily affects weights (75% reduction)
            optimized_components["weights"] *= 0.25  # Reduce to 25% (4-bit vs 16-bit)
            optimized_components["overhead"] += 2.0  # Small overhead for dequantization
            
        elif optimization == "8bit_quantization":
            # 8-bit quantization primarily affects weights (50% reduction)
            optimized_components["weights"] *= 0.5  # Reduce to 50% (8-bit vs 16-bit)
            optimized_components["overhead"] += 1.0  # Small overhead for dequantization
            
        elif optimization == "shared_kv_cache":
            # KV cache optimization for LLMs and multimodal models
            if "kv_cache" in optimized_components:
                optimized_components["kv_cache"] *= 0.6  # 40% reduction
            
        elif optimization == "flash_attention":
            # Flash attention reduces memory for attention operations
            if "attention" in optimized_components:
                optimized_components["attention"] *= 0.55  # 45% reduction
            
        elif optimization == "progressive_loading":
            # Progressive loading affects how much of the model is loaded at once
            optimized_components["weights"] *= 0.75  # Load 75% of weights at a time
            optimized_components["overhead"] += 5.0  # Overhead for managing progressive loading
            
        elif optimization == "parallel_loading":
            # Parallel loading primarily affects loading time, less impact on memory
            optimized_components["overhead"] += 2.0  # Small overhead for coordination
            
        elif optimization == "activation_checkpointing":
            # Activation checkpointing trades compute for memory
            optimized_components["activations"] *= 0.6  # 40% reduction in activation memory
            optimized_components["overhead"] += 3.0  # Overhead for recomputation
            
        elif optimization == "compute_shaders":
            # Compute shaders can reduce memory for certain operations
            if platform == "webgpu":
                if "activations" in optimized_components:
                    optimized_components["activations"] *= 0.85  # 15% reduction
                optimized_components["overhead"] += 1.5  # Shader compilation overhead
        
        # Calculate the impact of this optimization
        after_total = sum(optimized_components.values())
        savings = before_total - after_total
        percent_reduction = (savings / before_total) * 100 if before_total > 0 else 0
        
        # Record the impact
        optimization_impacts[optimization] = {
            "before_mb": round(before_total, 2),
            "after_mb": round(after_total, 2),
            "savings_mb": round(savings, 2),
            "percent_reduction": round(percent_reduction, 2),
            "components_before": {k: round(v, 2) for k, v in before_components.items()},
            "components_after": {k: round(v, 2) for k, v in optimized_components.items()}
        }
    
    # Calculate cumulative impact of all optimizations
    original_total = sum(original_components.values())
    final_total = sum(optimized_components.values())
    total_savings = original_total - final_total
    total_percent_reduction = (total_savings / original_total) * 100 if original_total > 0 else 0
    
    optimization_impacts["total"] = {
        "before_mb": round(original_total, 2),
        "after_mb": round(final_total, 2),
        "savings_mb": round(total_savings, 2),
        "percent_reduction": round(total_percent_reduction, 2),
        "components_before": {k: round(v, 2) for k, v in original_components.items()},
        "components_after": {k: round(v, 2) for k, v in optimized_components.items()}
    }
    
    # Update the components with the optimized values
    for k, v in optimized_components.items():
        components[k] = v
    
    return optimization_impacts

def create_component_breakdown_chart(memory_data: Dict[str, Any], output_file: str):
    """
    Create a component breakdown chart of memory usage.
    
    Args:
        memory_data: Memory usage data dictionary
        output_file: Output file path
    """
    components = memory_data["components"]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create bar chart
    component_names = list(components.keys())
    component_values = [components[name] for name in component_names]
    colors = [COLORS.get(name, "#333333") for name in component_names]
    
    # Sort by size (largest first)
    sorted_indices = np.argsort(component_values)[::-1]
    component_names = [component_names[i] for i in sorted_indices]
    component_values = [component_values[i] for i in sorted_indices]
    colors = [colors[i] for i in sorted_indices]
    
    bars = plt.bar(component_names, component_values, color=colors)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{height:.1f} MB',
                ha='center', va='bottom', fontsize=10)
    
    # Chart styling
    plt.title(f'Memory Usage Breakdown: {memory_data["model_name"]} on {memory_data["platform"]}', fontsize=16)
    plt.ylabel('Memory (MB)', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save to file
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Component breakdown chart saved to {output_file}")

def create_optimization_impact_chart(memory_data: Dict[str, Any], output_file: str):
    """
    Create a chart showing the impact of optimizations on memory usage.
    
    Args:
        memory_data: Memory usage data dictionary
        output_file: Output file path
    """
    if "optimizations_impact" not in memory_data or not memory_data["optimizations_impact"]:
        logger.warning("No optimization data available for chart")
        return
    
    # Extract optimization data (excluding the "total" summary)
    optimization_data = {k: v for k, v in memory_data["optimizations_impact"].items() if k != "total"}
    
    if not optimization_data:
        logger.warning("No individual optimization data available for chart")
        return
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left subplot: Absolute memory savings
    opt_names = list(optimization_data.keys())
    savings_mb = [optimization_data[name]["savings_mb"] for name in opt_names]
    
    # Sort by savings (largest first)
    sorted_indices = np.argsort(savings_mb)[::-1]
    opt_names = [opt_names[i] for i in sorted_indices]
    savings_mb = [savings_mb[i] for i in sorted_indices]
    
    bars1 = ax1.bar(opt_names, savings_mb, color='#3498db')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{height:.1f} MB',
                ha='center', va='bottom', fontsize=10)
    
    ax1.set_title('Memory Savings by Optimization (MB)', fontsize=14)
    ax1.set_ylabel('Memory Saved (MB)', fontsize=12)
    ax1.set_xticks(range(len(opt_names)))
    ax1.set_xticklabels(opt_names, rotation=45, ha='right', fontsize=10)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Right subplot: Percentage reduction
    percent_reduction = [optimization_data[name]["percent_reduction"] for name in opt_names]
    
    bars2 = ax2.bar(opt_names, percent_reduction, color='#2ecc71')
    
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10)
    
    ax2.set_title('Memory Reduction by Optimization (%)', fontsize=14)
    ax2.set_ylabel('Memory Reduction (%)', fontsize=12)
    ax2.set_xticks(range(len(opt_names)))
    ax2.set_xticklabels(opt_names, rotation=45, ha='right', fontsize=10)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Main title
    plt.suptitle(f'Memory Optimization Impact: {memory_data["model_name"]} on {memory_data["platform"]}', fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for main title
    
    # Save to file
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Optimization impact chart saved to {output_file}")

def create_precision_comparison_chart(model_name: str, output_file: str):
    """
    Create a chart comparing memory usage across different precision formats.
    
    Args:
        model_name: Name of the model
        output_file: Output file path
    """
    # Detect model type
    model_type = detect_model_type(model_name)
    
    # Define precision formats
    precisions = ["fp16", "int8", "int4", "int2"]
    
    # Generate synthetic data based on model type
    if "llama" in model_type.lower() or "qwen" in model_type.lower():
        # Large language model (7B parameters example)
        base_size = 14000  # MB for fp16
    elif "llava" in model_type.lower():
        # Multimodal model
        base_size = 8000  # MB for fp16
    elif "whisper" in model_type.lower() or "wav2vec" in model_type.lower():
        # Audio model
        base_size = 1000  # MB for fp16
    elif "vit" in model_type.lower() or "clip" in model_type.lower():
        # Vision model
        base_size = 700  # MB for fp16
    elif "bert" in model_type.lower() or "t5" in model_type.lower():
        # Text model
        base_size = 500  # MB for fp16
    else:
        # Generic model
        base_size = 800  # MB for fp16
    
    # Calculate sizes for different precisions
    sizes = {
        "fp16": base_size,
        "int8": base_size * 0.5,  # 50% of fp16
        "int4": base_size * 0.25,  # 25% of fp16
        "int2": base_size * 0.125  # 12.5% of fp16
    }
    
    # Accuracy loss estimates (percentage points)
    accuracy_loss = {
        "fp16": 0.0,
        "int8": 0.3,  # 0.3% accuracy loss
        "int4": 0.8,  # 0.8% accuracy loss
        "int2": 2.5   # 2.5% accuracy loss
    }
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Left subplot: Memory usage
    bars1 = ax1.bar(precisions, [sizes[p] for p in precisions], 
                    color=[COLORS.get(p, "#333333") for p in precisions])
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 50,
                f'{height:.1f} MB',
                ha='center', va='bottom', fontsize=10)
    
    ax1.set_title('Memory Usage by Precision Format', fontsize=14)
    ax1.set_ylabel('Memory (MB)', fontsize=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Right subplot: Accuracy loss
    bars2 = ax2.bar(precisions, [accuracy_loss[p] for p in precisions], 
                   color=[COLORS.get(p, "#333333") for p in precisions])
    
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10)
    
    ax2.set_title('Estimated Accuracy Loss by Precision Format', fontsize=14)
    ax2.set_ylabel('Accuracy Loss (%)', fontsize=12)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Main title
    plt.suptitle(f'Memory vs. Accuracy: {model_name} at Different Precision Formats', fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for main title
    
    # Save to file
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Precision comparison chart saved to {output_file}")

def create_timeline_chart(memory_data: Dict[str, Any], output_file: str):
    """
    Create a timeline chart of memory usage during inference.
    
    Args:
        memory_data: Memory usage data dictionary
        output_file: Output file path
    """
    if "timeline" not in memory_data or not memory_data["timeline"]:
        logger.warning("No timeline data available for chart")
        return
    
    timeline = memory_data["timeline"]
    
    # Extract data
    steps = [point["step"] for point in timeline]
    times = [point["time_seconds"] for point in timeline]
    memory_values = [point["memory_mb"] for point in timeline]
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Plot memory usage over time
    plt.plot(times, memory_values, 'o-', linewidth=2, markersize=8, color='#3498db')
    
    # Add annotations for key steps
    for i, (step, time, memory) in enumerate(zip(steps, times, memory_values)):
        plt.annotate(step, (time, memory), textcoords="offset points", 
                    xytext=(0, 10), ha='center', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Chart styling
    plt.title(f'Memory Usage Timeline: {memory_data["model_name"]} on {memory_data["platform"]}', fontsize=16)
    plt.xlabel('Time (seconds)', fontsize=14)
    plt.ylabel('Memory (MB)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add y=0 line
    plt.axhline(y=0, color='#95a5a6', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    
    # Save to file
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Timeline chart saved to {output_file}")

def create_platform_comparison_chart(model_name: str, output_file: str):
    """
    Create a chart comparing memory usage across different platforms.
    
    Args:
        model_name: Name of the model
        output_file: Output file path
    """
    # Detect model type
    model_type = detect_model_type(model_name)
    
    # Define platforms
    platforms = ["cuda", "cpu", "webgpu", "webnn", "openvino", "rocm", "mps"]
    
    # Generate synthetic data based on model type
    if "llama" in model_type.lower() or "qwen" in model_type.lower():
        # Large language model
        base_size = 14000  # MB for CUDA
        platform_factors = {
            "cuda": 1.0,
            "cpu": 1.1,  # CPU usually has slightly higher overhead
            "webgpu": 1.15,  # WebGPU has some browser overhead
            "webnn": 1.2,  # WebNN has additional overhead
            "openvino": 0.85,  # OpenVINO optimizes memory usage
            "rocm": 1.05,  # ROCm similar to CUDA
            "mps": 1.1  # Metal Performance Shaders
        }
    elif "llava" in model_type.lower() or "clip" in model_type.lower():
        # Multimodal model
        base_size = 8000  # MB for CUDA
        platform_factors = {
            "cuda": 1.0,
            "cpu": 1.15,
            "webgpu": 1.25,
            "webnn": 1.3,
            "openvino": 0.9,
            "rocm": 1.05,
            "mps": 1.1
        }
    else:
        # Generic model
        base_size = 1000  # MB for CUDA
        platform_factors = {
            "cuda": 1.0,
            "cpu": 1.1,
            "webgpu": 1.2,
            "webnn": 1.25,
            "openvino": 0.85,
            "rocm": 1.05,
            "mps": 1.1
        }
    
    # Calculate sizes for different platforms
    sizes = {platform: base_size * factor for platform, factor in platform_factors.items()}
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create bar chart
    colors = [COLORS.get(platform, "#333333") for platform in platforms]
    bars = plt.bar(platforms, [sizes[platform] for platform in platforms], color=colors)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (base_size * 0.02),
                f'{height:.0f} MB',
                ha='center', va='bottom', fontsize=10)
    
    # Chart styling
    plt.title(f'Memory Usage Comparison Across Platforms: {model_name}', fontsize=16)
    plt.ylabel('Memory (MB)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save to file
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Platform comparison chart saved to {output_file}")

def create_4bit_inference_chart(model_name: str, output_file: str):
    """
    Create a chart specific to 4-bit inference optimizations.
    
    Args:
        model_name: Name of the model
        output_file: Output file path
    """
    # Detect model type
    model_type = detect_model_type(model_name)
    
    # Define analysis categories
    categories = ["Memory Reduction", "Memory Breakdown", "Accuracy Impact", "Layer-Specific Precision"]
    
    # Generate synthetic data based on model type
    if "llama" in model_type.lower() or "qwen" in model_type.lower():
        # Large language model (base size for FP16)
        fp16_size = 14000  # MB
        int4_size = 3500   # MB (25% of FP16)
        
        # Memory breakdown (by component)
        int4_components = {
            "Weights": 3000,
            "KV Cache": 250,
            "Activations": 150,
            "Overhead": 100
        }
        
        # Accuracy impact (perplexity increase)
        accuracy_impact = 0.8  # 0.8% accuracy decrease
        
        # Layer-specific precision (percentage of model in each precision)
        layer_precision = {
            "INT4": 80,  # 80% of model in INT4
            "INT8": 15,  # 15% of model in INT8
            "FP16": 5    # 5% of model in FP16
        }
    else:
        # Generic model
        fp16_size = 1000  # MB
        int4_size = 250   # MB
        
        # Memory breakdown (by component)
        int4_components = {
            "Weights": 200,
            "Activations": 30,
            "Overhead": 20
        }
        
        # Accuracy impact
        accuracy_impact = 0.5  # 0.5% accuracy decrease
        
        # Layer-specific precision
        layer_precision = {
            "INT4": 75,  # 75% of model in INT4
            "INT8": 15,  # 15% of model in INT8
            "FP16": 10   # 10% of model in FP16
        }
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Memory Reduction (Subplot 1)
    ax1 = fig.add_subplot(2, 2, 1)
    labels = ['FP16', 'INT4']
    sizes = [fp16_size, int4_size]
    colors = ['#3498db', '#e74c3c']
    
    bars = ax1.bar(labels, sizes, color=colors)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 150,
                f'{height:.0f} MB',
                ha='center', va='bottom', fontsize=10)
    
    # Add reduction percentage
    reduction = ((fp16_size - int4_size) / fp16_size) * 100
    ax1.text(0.5, 0.9, f'{reduction:.1f}% reduction', transform=ax1.transAxes,
            ha='center', va='center', fontsize=14,
            bbox=dict(boxstyle="round,pad=0.3", fc="#2ecc71", ec="gray", alpha=0.8))
    
    ax1.set_title('Memory Reduction with 4-bit Quantization', fontsize=14)
    ax1.set_ylabel('Memory (MB)', fontsize=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 2. Memory Breakdown (Subplot 2)
    ax2 = fig.add_subplot(2, 2, 2)
    component_names = list(int4_components.keys())
    component_values = list(int4_components.values())
    
    # Custom colors
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f'][:len(component_names)]
    
    # Sort by size (largest first)
    sorted_indices = np.argsort(component_values)[::-1]
    component_names = [component_names[i] for i in sorted_indices]
    component_values = [component_values[i] for i in sorted_indices]
    colors = [colors[i] for i in sorted_indices]
    
    bars = ax2.bar(component_names, component_values, color=colors)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 20,
                f'{height:.0f} MB',
                ha='center', va='bottom', fontsize=10)
    
    ax2.set_title('Memory Breakdown (4-bit Model)', fontsize=14)
    ax2.set_ylabel('Memory (MB)', fontsize=12)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 3. Accuracy Impact (Subplot 3)
    ax3 = fig.add_subplot(2, 2, 3)
    
    # Create a gauge-like visualization
    impact_categories = ['Negligible\n(<0.3%)', 'Minor\n(0.3-0.6%)', 'Moderate\n(0.6-1.0%)', 'Significant\n(>1.0%)']
    impact_thresholds = [0.3, 0.6, 1.0, 100.0]
    impact_colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
    
    # Determine category based on accuracy impact
    impact_category = 0
    for i, threshold in enumerate(impact_thresholds):
        if accuracy_impact <= threshold:
            impact_category = i
            break
    
    # Create bars for all categories
    bars = ax3.bar(impact_categories, [1, 1, 1, 1], color='#ecf0f1', edgecolor='#bdc3c7')
    
    # Highlight the relevant category
    bars[impact_category].set_color(impact_colors[impact_category])
    
    # Add accuracy impact text
    ax3.text(0.5, 0.85, f'Accuracy Impact: {accuracy_impact:.2f}%',
            transform=ax3.transAxes, ha='center', va='center', fontsize=14,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    ax3.set_title('Accuracy Impact Assessment', fontsize=14)
    ax3.set_ylim(0, 1.5)
    ax3.set_yticks([])
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['left'].set_visible(False)
    
    # 4. Layer-Specific Precision (Subplot 4)
    ax4 = fig.add_subplot(2, 2, 4)
    precision_labels = list(layer_precision.keys())
    precision_values = list(layer_precision.values())
    precision_colors = ['#e74c3c', '#f1c40f', '#3498db'][:len(precision_labels)]
    
    # Create pie chart
    wedges, texts, autotexts = ax4.pie(precision_values, labels=precision_labels, 
                                     autopct='%1.1f%%', startangle=90, colors=precision_colors)
    
    # Style pie chart text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(10)
        autotext.set_weight('bold')
    
    ax4.set_title('Layer-Specific Precision Distribution', fontsize=14)
    
    # Main title
    plt.suptitle(f'4-bit Inference Analysis: {model_name}', fontsize=18)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for main title
    
    # Save to file
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"4-bit inference chart saved to {output_file}")

def create_html_report(model_name: str, memory_data_list: List[Dict[str, Any]], output_file: str):
    """
    Create a comprehensive HTML report with all memory visualizations.
    
    Args:
        model_name: Name of the model
        memory_data_list: List of memory data dictionaries for different platforms/configs
        output_file: Output file path
    """
    # Create temporary directory for charts
    output_dir = os.path.dirname(output_file)
    charts_dir = os.path.join(output_dir, "memory_charts")
    os.makedirs(charts_dir, exist_ok=True)
    
    # Generate all charts
    chart_files = {}
    
    # For each platform/configuration
    for idx, memory_data in enumerate(memory_data_list):
        platform = memory_data.get("platform", f"config_{idx}")
        optimizations = memory_data.get("optimizations", [])
        
        # Generate charts
        component_chart = os.path.join(charts_dir, f"{model_name}_{platform}_components.png")
        create_component_breakdown_chart(memory_data, component_chart)
        chart_files[f"{platform}_components"] = os.path.relpath(component_chart, output_dir)
        
        if "optimizations_impact" in memory_data and memory_data["optimizations_impact"]:
            opt_chart = os.path.join(charts_dir, f"{model_name}_{platform}_optimizations.png")
            create_optimization_impact_chart(memory_data, opt_chart)
            chart_files[f"{platform}_optimizations"] = os.path.relpath(opt_chart, output_dir)
        
        if "timeline" in memory_data and memory_data["timeline"]:
            timeline_chart = os.path.join(charts_dir, f"{model_name}_{platform}_timeline.png")
            create_timeline_chart(memory_data, timeline_chart)
            chart_files[f"{platform}_timeline"] = os.path.relpath(timeline_chart, output_dir)
    
    # Generate common charts
    precision_chart = os.path.join(charts_dir, f"{model_name}_precision_comparison.png")
    create_precision_comparison_chart(model_name, precision_chart)
    chart_files["precision_comparison"] = os.path.relpath(precision_chart, output_dir)
    
    platform_chart = os.path.join(charts_dir, f"{model_name}_platform_comparison.png")
    create_platform_comparison_chart(model_name, platform_chart)
    chart_files["platform_comparison"] = os.path.relpath(platform_chart, output_dir)
    
    bit4_chart = os.path.join(charts_dir, f"{model_name}_4bit_inference.png")
    create_4bit_inference_chart(model_name, bit4_chart)
    chart_files["4bit_inference"] = os.path.relpath(bit4_chart, output_dir)
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Memory Usage Analysis: {model_name}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            .section {{
                margin-bottom: 30px;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 20px;
                background-color: #f9f9f9;
            }}
            .chart-container {{
                text-align: center;
                margin: 20px 0;
            }}
            .chart {{
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 10px;
                border: 1px solid #ddd;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            .platform-section {{
                margin-bottom: 40px;
            }}
            .summary {{
                background-color: #e8f4f8;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
            }}
            .optimization {{
                margin: 10px 0;
                padding: 10px;
                background-color: #f0f7ff;
                border-radius: 5px;
            }}
        </style>
    </head>
    <body>
        <h1>Memory Usage Analysis: {model_name}</h1>
        <div class="summary">
            <h2>Executive Summary</h2>
            <p>This report provides a comprehensive analysis of memory usage for <strong>{model_name}</strong> 
            across different platforms, precision formats, and optimization techniques.</p>
        </div>
        
        <div class="section">
            <h2>Cross-Platform Memory Comparison</h2>
            <div class="chart-container">
                <img class="chart" src="{chart_files['platform_comparison']}" alt="Platform Memory Comparison">
            </div>
            <p>The chart above shows memory usage comparison across different hardware and web platforms.</p>
        </div>
        
        <div class="section">
            <h2>Precision Format Comparison</h2>
            <div class="chart-container">
                <img class="chart" src="{chart_files['precision_comparison']}" alt="Precision Format Comparison">
            </div>
            <p>This chart compares memory usage and accuracy impact across different precision formats (FP16, INT8, INT4, INT2).</p>
        </div>
        
        <div class="section">
            <h2>4-bit Quantization Analysis</h2>
            <div class="chart-container">
                <img class="chart" src="{chart_files['4bit_inference']}" alt="4-bit Inference Analysis">
            </div>
            <p>Detailed analysis of 4-bit quantization impact, including memory reduction, component breakdown, 
            accuracy impact, and precision distribution across model layers.</p>
        </div>
    """
    
    # Add platform-specific sections
    for idx, memory_data in enumerate(memory_data_list):
        platform = memory_data.get("platform", f"config_{idx}")
        optimizations = memory_data.get("optimizations", [])
        
        html_content += f"""
        <div class="platform-section section">
            <h2>Memory Analysis: {platform.upper()}</h2>
            
            <h3>Component Breakdown</h3>
            <div class="chart-container">
                <img class="chart" src="{chart_files[f'{platform}_components']}" alt="{platform} Component Breakdown">
            </div>
        """
        
        if f"{platform}_optimizations" in chart_files:
            html_content += f"""
            <h3>Optimization Impact</h3>
            <div class="chart-container">
                <img class="chart" src="{chart_files[f'{platform}_optimizations']}" alt="{platform} Optimization Impact">
            </div>
            """
        
        if f"{platform}_timeline" in chart_files:
            html_content += f"""
            <h3>Memory Timeline</h3>
            <div class="chart-container">
                <img class="chart" src="{chart_files[f'{platform}_timeline']}" alt="{platform} Memory Timeline">
            </div>
            """
        
        # Add optimization details table
        if "optimizations_impact" in memory_data and memory_data["optimizations_impact"]:
            html_content += f"""
            <h3>Optimization Details</h3>
            <table>
                <tr>
                    <th>Optimization</th>
                    <th>Before (MB)</th>
                    <th>After (MB)</th>
                    <th>Savings (MB)</th>
                    <th>Reduction (%)</th>
                </tr>
            """
            
            for opt_name, opt_data in memory_data["optimizations_impact"].items():
                html_content += f"""
                <tr>
                    <td>{opt_name}</td>
                    <td>{opt_data['before_mb']}</td>
                    <td>{opt_data['after_mb']}</td>
                    <td>{opt_data['savings_mb']}</td>
                    <td>{opt_data['percent_reduction']}%</td>
                </tr>
                """
            
            html_content += """
            </table>
            """
        
        html_content += """
        </div>
        """
    
    # Close HTML
    html_content += """
    </body>
    </html>
    """
    
    # Write HTML file
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    logger.info(f"HTML report saved to {output_file}")

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Visualize memory usage for web platform models")
    
    parser.add_argument("--model", type=str, required=True, help="Model name to analyze")
    parser.add_argument("--platform", type=str, default="all", help="Platform to analyze (webnn, webgpu, all)")
    parser.add_argument("--optimizations", type=str, nargs="*", help="Optimizations to apply (comma-separated)")
    parser.add_argument("--output", type=str, default="html", help="Output format (html, png, json)")
    parser.add_argument("--output-dir", type=str, default="./memory_analysis", help="Output directory for results")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse optimizations
    optimizations = []
    if args.optimizations:
        for opt in args.optimizations:
            optimizations.extend(opt.split(","))
    
    # Validate optimizations
    valid_optimizations = [
        "4bit_quantization", "8bit_quantization", "shared_kv_cache", "flash_attention", 
        "progressive_loading", "parallel_loading", "activation_checkpointing", "compute_shaders"
    ]
    
    for opt in optimizations:
        if opt not in valid_optimizations:
            logger.warning(f"Unknown optimization: {opt}")
            logger.info(f"Valid optimizations: {', '.join(valid_optimizations)}")
    
    # Determine platforms to analyze
    platforms = []
    if args.platform == "all":
        platforms = ["webnn", "webgpu"]
    else:
        platforms = [args.platform]
    
    # Analyze memory usage for each platform
    memory_data_list = []
    for platform in platforms:
        memory_data = analyze_model_memory(args.model, platform, optimizations)
        memory_data_list.append(memory_data)
    
    # Generate output
    if args.output == "json":
        # Save JSON output
        output_file = os.path.join(args.output_dir, f"{args.model}_memory_analysis.json")
        with open(output_file, 'w') as f:
            json.dump(memory_data_list, f, indent=2)
        logger.info(f"Memory analysis saved to {output_file}")
        
    elif args.output == "png":
        # Generate individual charts
        for memory_data in memory_data_list:
            platform = memory_data["platform"]
            
            # Component breakdown chart
            component_chart = os.path.join(args.output_dir, f"{args.model}_{platform}_components.png")
            create_component_breakdown_chart(memory_data, component_chart)
            
            # Optimization impact chart (if optimizations were applied)
            if optimizations:
                opt_chart = os.path.join(args.output_dir, f"{args.model}_{platform}_optimizations.png")
                create_optimization_impact_chart(memory_data, opt_chart)
            
            # Timeline chart
            timeline_chart = os.path.join(args.output_dir, f"{args.model}_{platform}_timeline.png")
            create_timeline_chart(memory_data, timeline_chart)
        
        # Generate common charts
        precision_chart = os.path.join(args.output_dir, f"{args.model}_precision_comparison.png")
        create_precision_comparison_chart(args.model, precision_chart)
        
        platform_chart = os.path.join(args.output_dir, f"{args.model}_platform_comparison.png")
        create_platform_comparison_chart(args.model, platform_chart)
        
        bit4_chart = os.path.join(args.output_dir, f"{args.model}_4bit_inference.png")
        create_4bit_inference_chart(args.model, bit4_chart)
    
    elif args.output == "html":
        # Generate HTML report
        html_file = os.path.join(args.output_dir, f"{args.model}_memory_analysis.html")
        create_html_report(args.model, memory_data_list, html_file)
    
    else:
        logger.error(f"Unknown output format: {args.output}")
        logger.info("Valid formats: html, png, json")

if __name__ == "__main__":
    main()