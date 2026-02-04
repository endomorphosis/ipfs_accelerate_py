#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-Platform Performance Analysis Tool

This script analyzes and compares benchmark results across different platforms
(Android, desktop) from the benchmark database. It generates reports showing
performance comparisons, battery impact, thermal characteristics, and
optimization recommendations.

Features:
    - Cross-platform performance comparison
    - Model optimization recommendations
    - Battery impact analysis
    - Thermal impact analysis
    - Hardware compatibility scoring
    - Report generation (markdown, HTML)

Date: April 2025
"""

import os
import sys
import json
import logging
import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Local imports
try:
    from .database_integration import AndroidDatabaseAPI
    ANDROID_DB_AVAILABLE = True
except ImportError:
    logger.warning("Could not import Android database integration. Some functionality will be limited.")
    ANDROID_DB_AVAILABLE = False


def get_cross_platform_comparison(db_path: str, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get cross-platform performance comparison data.
    
    Args:
        db_path: Path to benchmark database
        model_name: Optional model name to filter by
        
    Returns:
        List of comparison results
    """
    if not ANDROID_DB_AVAILABLE:
        logger.error("Android database integration not available")
        return []
    
    try:
        # Connect to database
        db_api = AndroidDatabaseAPI(db_path)
        
        # Get comparison data
        comparison = db_api.get_cross_platform_comparison(model_name)
        
        # Return results
        return comparison
    
    except Exception as e:
        logger.error(f"Error getting cross-platform comparison: {e}")
        return []


def get_device_performance(db_path: str) -> List[Dict[str, Any]]:
    """
    Get Android device performance summary.
    
    Args:
        db_path: Path to benchmark database
        
    Returns:
        List of device performance summaries
    """
    if not ANDROID_DB_AVAILABLE:
        logger.error("Android database integration not available")
        return []
    
    try:
        # Connect to database
        db_api = AndroidDatabaseAPI(db_path)
        
        # Get device performance
        performance = db_api.get_device_performance_summary()
        
        # Return results
        return performance
    
    except Exception as e:
        logger.error(f"Error getting device performance: {e}")
        return []


def get_model_summary(db_path: str) -> List[Dict[str, Any]]:
    """
    Get Android model performance summary.
    
    Args:
        db_path: Path to benchmark database
        
    Returns:
        List of model performance summaries
    """
    if not ANDROID_DB_AVAILABLE:
        logger.error("Android database integration not available")
        return []
    
    try:
        # Connect to database
        db_api = AndroidDatabaseAPI(db_path)
        
        # Get model summary
        summary = db_api.get_model_summary()
        
        # Return results
        return summary
    
    except Exception as e:
        logger.error(f"Error getting model summary: {e}")
        return []


def get_benchmark_results(db_path: str, model_name: Optional[str] = None, device_model: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get Android benchmark results.
    
    Args:
        db_path: Path to benchmark database
        model_name: Optional model name to filter by
        device_model: Optional device model to filter by
        
    Returns:
        List of benchmark results
    """
    if not ANDROID_DB_AVAILABLE:
        logger.error("Android database integration not available")
        return []
    
    try:
        # Connect to database
        db_api = AndroidDatabaseAPI(db_path)
        
        # Get benchmark results
        results = db_api.get_benchmark_results(model_name, device_model)
        
        # Return results
        return results
    
    except Exception as e:
        logger.error(f"Error getting benchmark results: {e}")
        return []


def generate_cross_platform_report(db_path: str, output_path: Optional[str] = None, model_name: Optional[str] = None) -> str:
    """
    Generate a cross-platform performance comparison report.
    
    Args:
        db_path: Path to benchmark database
        output_path: Optional path to save the report
        model_name: Optional model name to filter by
        
    Returns:
        Generated report
    """
    # Get comparison data
    comparison = get_cross_platform_comparison(db_path, model_name)
    
    if not comparison:
        return "No cross-platform comparison data available."
    
    # Generate report
    report = "# Cross-Platform Performance Comparison\n\n"
    report += f"Generated: {datetime.datetime.now().isoformat()}\n\n"
    
    # Summary table
    report += "## Performance Summary\n\n"
    report += "| Model | Android Throughput | Desktop Throughput | Ratio (Desktop/Android) | Android Latency | Desktop Latency | Ratio (Android/Desktop) |\n"
    report += "|-------|-------------------|-------------------|------------------------|----------------|-----------------|-------------------------|\n"
    
    for item in comparison:
        model_name = item.get("model_name", "Unknown")
        android_throughput = item.get("android_throughput", 0)
        desktop_throughput = item.get("desktop_throughput", 0)
        throughput_ratio = item.get("throughput_ratio", 0)
        android_latency = item.get("android_latency_ms", 0)
        desktop_latency = item.get("desktop_latency_ms", 0)
        latency_ratio = item.get("latency_ratio", 0)
        
        report += f"| {model_name} | {android_throughput:.2f} items/s | {desktop_throughput:.2f} items/s | {throughput_ratio:.2f}x | "
        report += f"{android_latency:.2f} ms | {desktop_latency:.2f} ms | {latency_ratio:.2f}x |\n"
    
    # Analysis
    report += "\n## Analysis\n\n"
    
    # Find average ratios
    avg_throughput_ratio = sum(item.get("throughput_ratio", 0) for item in comparison) / len(comparison)
    avg_latency_ratio = sum(item.get("latency_ratio", 0) for item in comparison) / len(comparison)
    
    report += f"- **Average Desktop/Android Throughput Ratio**: {avg_throughput_ratio:.2f}x\n"
    report += f"- **Average Android/Desktop Latency Ratio**: {avg_latency_ratio:.2f}x\n\n"
    
    # Find best and worst performing models on mobile
    best_model = min(comparison, key=lambda x: x.get("throughput_ratio", float("inf")))
    worst_model = max(comparison, key=lambda x: x.get("throughput_ratio", 0))
    
    report += f"- **Best Mobile Performance**: {best_model.get('model_name', 'Unknown')} "
    report += f"(Desktop only {best_model.get('throughput_ratio', 0):.2f}x faster)\n"
    
    report += f"- **Worst Mobile Performance**: {worst_model.get('model_name', 'Unknown')} "
    report += f"(Desktop {worst_model.get('throughput_ratio', 0):.2f}x faster)\n\n"
    
    # Recommendations
    report += "## Recommendations\n\n"
    
    # General recommendations
    report += "### General Recommendations\n\n"
    report += "- **Quantization**: Use INT8 quantization for all mobile deployments\n"
    report += "- **Batch Size**: Use small batch sizes (1-4) on mobile devices\n"
    report += "- **Model Size**: Consider smaller model variants for mobile deployment\n"
    report += "- **Hardware Acceleration**: Use NPU/DSP acceleration when available\n\n"
    
    # Model-specific recommendations
    report += "### Model-Specific Recommendations\n\n"
    
    for item in comparison:
        model_name = item.get("model_name", "Unknown")
        throughput_ratio = item.get("throughput_ratio", 0)
        
        report += f"**{model_name}**:\n"
        
        if throughput_ratio > 10:
            # Very large performance gap
            report += "- **Critical**: Consider model pruning and special optimizations\n"
            report += "- Use knowledge distillation to create a mobile-specific variant\n"
            report += "- Consider mobile-specific architecture changes\n"
        elif throughput_ratio > 5:
            # Large performance gap
            report += "- **High Priority**: Apply model optimization techniques\n"
            report += "- Use INT8 or INT4 quantization\n"
            report += "- Apply layer fusion optimizations\n"
        elif throughput_ratio > 2:
            # Moderate performance gap
            report += "- **Medium Priority**: Apply standard mobile optimizations\n"
            report += "- Use INT8 quantization\n"
            report += "- Consider operator fusion\n"
        else:
            # Small performance gap
            report += "- **Low Priority**: Model performs well on mobile\n"
            report += "- Consider standard optimizations for battery and thermal impact\n"
        
        report += "\n"
    
    # Save report if output path provided
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
        logger.info(f"Cross-platform report saved to: {output_path}")
    
    return report


def generate_device_comparison_report(db_path: str, output_path: Optional[str] = None) -> str:
    """
    Generate a device comparison report.
    
    Args:
        db_path: Path to benchmark database
        output_path: Optional path to save the report
        
    Returns:
        Generated report
    """
    # Get device performance data
    devices = get_device_performance(db_path)
    
    if not devices:
        return "No device performance data available."
    
    # Generate report
    report = "# Android Device Performance Comparison\n\n"
    report += f"Generated: {datetime.datetime.now().isoformat()}\n\n"
    
    # Summary table
    report += "## Performance Summary\n\n"
    report += "| Device | Chipset | Accelerator | Benchmarks | Avg Throughput | Avg Latency | Battery Impact | Throttling |\n"
    report += "|--------|---------|-------------|------------|----------------|-------------|----------------|------------|\n"
    
    for device in devices:
        device_model = device.get("device_model", "Unknown")
        chipset = device.get("chipset", "Unknown")
        accelerator = device.get("accelerator", "Unknown")
        benchmark_count = device.get("benchmark_count", 0)
        avg_throughput = device.get("avg_throughput", 0)
        avg_latency = device.get("avg_latency_ms", 0)
        avg_battery_impact = device.get("avg_battery_impact", 0)
        throttling_frequency = device.get("throttling_frequency", 0) * 100  # Convert to percentage
        
        report += f"| {device_model} | {chipset} | {accelerator} | {benchmark_count} | "
        report += f"{avg_throughput:.2f} items/s | {avg_latency:.2f} ms | "
        report += f"{avg_battery_impact:.1f}% | {throttling_frequency:.1f}% |\n"
    
    # Analysis
    report += "\n## Analysis\n\n"
    
    # Find best and worst devices
    best_device = max(devices, key=lambda x: x.get("avg_throughput", 0))
    worst_device = min(devices, key=lambda x: x.get("avg_throughput", 0))
    
    best_efficiency = min(devices, key=lambda x: x.get("avg_battery_impact", float("inf")))
    worst_efficiency = max(devices, key=lambda x: x.get("avg_battery_impact", 0))
    
    report += f"- **Best Performance**: {best_device.get('device_model', 'Unknown')} "
    report += f"({best_device.get('avg_throughput', 0):.2f} items/s)\n"
    
    report += f"- **Worst Performance**: {worst_device.get('device_model', 'Unknown')} "
    report += f"({worst_device.get('avg_throughput', 0):.2f} items/s)\n\n"
    
    report += f"- **Best Efficiency**: {best_efficiency.get('device_model', 'Unknown')} "
    report += f"({best_efficiency.get('avg_battery_impact', 0):.1f}% battery impact)\n"
    
    report += f"- **Worst Efficiency**: {worst_efficiency.get('device_model', 'Unknown')} "
    report += f"({worst_efficiency.get('avg_battery_impact', 0):.1f}% battery impact)\n\n"
    
    # Accelerator comparison
    report += "## Accelerator Comparison\n\n"
    
    # Group by accelerator
    accelerators = {}
    for device in devices:
        accelerator = device.get("accelerator", "Unknown")
        if accelerator not in accelerators:
            accelerators[accelerator] = []
        accelerators[accelerator].append(device)
    
    # Compare accelerators
    report += "| Accelerator | Avg Throughput | Avg Latency | Avg Battery Impact | Throttling |\n"
    report += "|-------------|----------------|-------------|-------------------|------------|\n"
    
    for accelerator, accel_devices in accelerators.items():
        avg_throughput = sum(d.get("avg_throughput", 0) for d in accel_devices) / len(accel_devices)
        avg_latency = sum(d.get("avg_latency_ms", 0) for d in accel_devices) / len(accel_devices)
        avg_battery_impact = sum(d.get("avg_battery_impact", 0) for d in accel_devices) / len(accel_devices)
        avg_throttling = sum(d.get("throttling_frequency", 0) for d in accel_devices) / len(accel_devices) * 100
        
        report += f"| {accelerator} | {avg_throughput:.2f} items/s | {avg_latency:.2f} ms | "
        report += f"{avg_battery_impact:.1f}% | {avg_throttling:.1f}% |\n"
    
    # Recommendations
    report += "\n## Recommendations\n\n"
    
    report += "### General Recommendations\n\n"
    report += "- **Hardware Selection**: Choose devices with low throttling frequency\n"
    report += "- **Accelerator Selection**: Select accelerator based on model type and size\n"
    report += "- **Thermal Management**: Implement cooling breaks for devices with high throttling\n"
    report += "- **Battery Optimization**: Apply model quantization for better battery efficiency\n\n"
    
    # Device-specific recommendations
    report += "### Device-Specific Recommendations\n\n"
    
    for device in devices:
        device_model = device.get("device_model", "Unknown")
        throttling_frequency = device.get("throttling_frequency", 0)
        battery_impact = device.get("avg_battery_impact", 0)
        
        report += f"**{device_model}**:\n"
        
        if throttling_frequency > 0.5:
            # High throttling
            report += "- **Thermal Management Critical**: Implement cooling breaks\n"
            report += "- Consider reducing batch size and thread count\n"
        
        if battery_impact > 5:
            # High battery impact
            report += "- **Battery Optimization Critical**: Use more aggressive quantization\n"
            report += "- Implement power-efficient scheduling\n"
        
        report += "\n"
    
    # Save report if output path provided
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
        logger.info(f"Device comparison report saved to: {output_path}")
    
    return report


def generate_model_optimization_report(db_path: str, output_path: Optional[str] = None) -> str:
    """
    Generate a model optimization report.
    
    Args:
        db_path: Path to benchmark database
        output_path: Optional path to save the report
        
    Returns:
        Generated report
    """
    # Get model summary data
    models = get_model_summary(db_path)
    
    if not models:
        return "No model summary data available."
    
    # Get cross-platform comparison
    comparison = get_cross_platform_comparison(db_path)
    
    # Create model map for easier lookup
    model_map = {}
    for model in models:
        model_map[model.get("model_name", "")] = model
    
    # Create comparison map for easier lookup
    comparison_map = {}
    for item in comparison:
        comparison_map[item.get("model_name", "")] = item
    
    # Generate report
    report = "# Model Optimization Recommendations\n\n"
    report += f"Generated: {datetime.datetime.now().isoformat()}\n\n"
    
    # Optimization priority table
    report += "## Optimization Priority\n\n"
    report += "| Model | Family | Performance Gap | Battery Impact | Priority | Key Optimizations |\n"
    report += "|-------|--------|----------------|----------------|----------|-------------------|\n"
    
    for model in models:
        model_name = model.get("model_name", "Unknown")
        model_family = model.get("model_family", "Unknown")
        battery_impact = model.get("avg_battery_impact", 0)
        
        # Get performance gap from comparison if available
        throughput_ratio = 1
        if model_name in comparison_map:
            throughput_ratio = comparison_map[model_name].get("throughput_ratio", 1)
        
        # Determine priority
        if throughput_ratio > 10 or battery_impact > 10:
            priority = "Critical"
        elif throughput_ratio > 5 or battery_impact > 5:
            priority = "High"
        elif throughput_ratio > 2 or battery_impact > 2:
            priority = "Medium"
        else:
            priority = "Low"
        
        # Determine key optimizations
        optimizations = []
        
        if throughput_ratio > 5:
            optimizations.append("Pruning")
            optimizations.append("Knowledge Distillation")
        
        if throughput_ratio > 2:
            optimizations.append("INT8 Quantization")
            optimizations.append("Operator Fusion")
        
        if battery_impact > 5:
            optimizations.append("Power Scheduling")
        
        if not optimizations:
            optimizations.append("Standard Optimizations")
        
        # Add to table
        report += f"| {model_name} | {model_family} | {throughput_ratio:.2f}x | "
        report += f"{battery_impact:.1f}% | {priority} | {', '.join(optimizations)} |\n"
    
    # Family-specific optimizations
    report += "\n## Family-Specific Optimizations\n\n"
    
    # Group by family
    families = {}
    for model in models:
        family = model.get("model_family", "Unknown")
        if family not in families:
            families[family] = []
        families[family].append(model)
    
    # Generate recommendations for each family
    for family, family_models in families.items():
        report += f"### {family}\n\n"
        
        # Skip unknown family
        if family == "unknown":
            continue
        
        # Generate family-specific recommendations
        if family.lower() in ["bert", "distilbert", "albert"]:
            report += "- **Attention Optimization**: Fuse attention operations\n"
            report += "- **Embedding Optimization**: Use shared embeddings\n"
            report += "- **Quantization**: Use INT8 quantization\n"
        
        elif family.lower() in ["llama", "gpt", "t5", "flan-t5"]:
            report += "- **KV Cache Optimization**: Implement efficient KV cache\n"
            report += "- **Weight-Only Quantization**: Use INT4/INT8 for weights\n"
            report += "- **Batched Inference**: Implement efficient batching\n"
        
        elif family.lower() in ["vit", "resnet", "clip", "efficientnet"]:
            report += "- **Convolution Optimization**: Use winograd algorithm\n"
            report += "- **Channel Pruning**: Reduce channel dimensions\n"
            report += "- **Quantization**: Use symmetric INT8 quantization\n"
        
        elif family.lower() in ["whisper", "wav2vec2", "hubert"]:
            report += "- **Feature Extraction Optimization**: Optimize mel spectrogram computation\n"
            report += "- **DSP Acceleration**: Use DSP acceleration when available\n"
            report += "- **Streaming Inference**: Implement streaming interfaces\n"
        
        else:
            report += "- **Standard Optimizations**: Apply general model optimization techniques\n"
            report += "- **Quantization**: Use INT8 quantization\n"
            report += "- **Operator Fusion**: Fuse consecutive operations\n"
        
        report += "\n"
    
    # Implementation plan
    report += "## Implementation Plan\n\n"
    
    # Sort models by priority
    priority_map = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    sorted_models = sorted(models, key=lambda x: (
        priority_map.get(
            "Critical" if x.get("model_name") in comparison_map and comparison_map[x.get("model_name")].get("throughput_ratio", 1) > 10 else
            "High" if x.get("model_name") in comparison_map and comparison_map[x.get("model_name")].get("throughput_ratio", 1) > 5 else
            "Medium" if x.get("model_name") in comparison_map and comparison_map[x.get("model_name")].get("throughput_ratio", 1) > 2 else
            "Low", 
            3
        )
    ))
    
    # Generate implementation plan
    report += "### Phase 1: Critical Optimizations\n\n"
    critical_models = [model for model in sorted_models if 
                     model.get("model_name") in comparison_map and 
                     comparison_map[model.get("model_name")].get("throughput_ratio", 1) > 10]
    
    for model in critical_models:
        model_name = model.get("model_name", "Unknown")
        report += f"- **{model_name}**: Apply pruning, knowledge distillation, and INT4 quantization\n"
    
    report += "\n### Phase 2: High Priority Optimizations\n\n"
    high_models = [model for model in sorted_models if 
                 model.get("model_name") in comparison_map and 
                 5 < comparison_map[model.get("model_name")].get("throughput_ratio", 1) <= 10]
    
    for model in high_models:
        model_name = model.get("model_name", "Unknown")
        report += f"- **{model_name}**: Apply INT8 quantization and operator fusion\n"
    
    report += "\n### Phase 3: Medium Priority Optimizations\n\n"
    medium_models = [model for model in sorted_models if 
                   model.get("model_name") in comparison_map and 
                   2 < comparison_map[model.get("model_name")].get("throughput_ratio", 1) <= 5]
    
    for model in medium_models:
        model_name = model.get("model_name", "Unknown")
        report += f"- **{model_name}**: Apply standard optimizations\n"
    
    # Save report if output path provided
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
        logger.info(f"Model optimization report saved to: {output_path}")
    
    return report


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Cross-Platform Performance Analysis Tool")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Cross-platform comparison report command
    cross_parser = subparsers.add_parser("cross-platform", help="Generate cross-platform comparison report")
    cross_parser.add_argument("--db-path", required=True, help="Path to benchmark database")
    cross_parser.add_argument("--output", help="Path to save the report")
    cross_parser.add_argument("--model", help="Filter by model name")
    
    # Device comparison report command
    device_parser = subparsers.add_parser("device-comparison", help="Generate device comparison report")
    device_parser.add_argument("--db-path", required=True, help="Path to benchmark database")
    device_parser.add_argument("--output", help="Path to save the report")
    
    # Model optimization report command
    model_parser = subparsers.add_parser("model-optimization", help="Generate model optimization report")
    model_parser.add_argument("--db-path", required=True, help="Path to benchmark database")
    model_parser.add_argument("--output", help="Path to save the report")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == "cross-platform":
            report = generate_cross_platform_report(args.db_path, args.output, args.model)
            
            if not args.output:
                print(report)
            
            return 0
        
        elif args.command == "device-comparison":
            report = generate_device_comparison_report(args.db_path, args.output)
            
            if not args.output:
                print(report)
            
            return 0
        
        elif args.command == "model-optimization":
            report = generate_model_optimization_report(args.db_path, args.output)
            
            if not args.output:
                print(report)
            
            return 0
        
        else:
            parser.print_help()
            return 1
    
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())