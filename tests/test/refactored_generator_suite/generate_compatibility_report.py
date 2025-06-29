#!/usr/bin/env python3
"""
Generate a human-readable hardware compatibility report.

This script reads the hardware compatibility matrix JSON file and generates
a markdown report with tables showing compatibility between model architectures
and hardware backends.
"""

import json
import sys
import os
import argparse
from pathlib import Path
from datetime import datetime


def load_compatibility_matrix(file_path):
    """Load the compatibility matrix from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def generate_compatibility_table(matrix_data, architecture_types=None, hardware_types=None):
    """
    Generate a markdown table showing architecture-hardware compatibility.
    
    Args:
        matrix_data: The compatibility matrix data
        architecture_types: List of architecture types to include (default: all)
        hardware_types: List of hardware types to include (default: all)
    
    Returns:
        Markdown string with the compatibility table
    """
    # Get architecture and hardware types
    all_architectures = list(matrix_data["architecture_hardware_compatibility"].keys())
    architectures = architecture_types or all_architectures
    
    # Sort architectures in a logical order
    sorted_architectures = []
    architecture_groups = [
        ["encoder-only", "decoder-only", "encoder-decoder"],  # Text models
        ["vision", "vision-encoder-text-decoder"],  # Vision models
        ["speech"],  # Audio models
        ["multimodal", "diffusion"],  # Complex models
        ["mixture-of-experts", "state-space", "rag"]  # Special architectures
    ]
    
    for group in architecture_groups:
        for arch in group:
            if arch in architectures:
                sorted_architectures.append(arch)
    
    # Add any remaining architectures
    for arch in architectures:
        if arch not in sorted_architectures:
            sorted_architectures.append(arch)
    
    # Get all hardware types from the first architecture
    first_arch = list(matrix_data["architecture_hardware_compatibility"].keys())[0]
    all_hardware = list(matrix_data["architecture_hardware_compatibility"][first_arch].keys())
    hardware = hardware_types or all_hardware
    
    # Sort hardware in a logical order
    hw_order = ["cpu", "cuda", "rocm", "mps", "openvino", "qnn"]
    sorted_hardware = sorted(hardware, key=lambda x: hw_order.index(x) if x in hw_order else 999)
    
    # Create the table header
    table = "| Architecture | " + " | ".join(sorted_hardware) + " |\n"
    table += "| --- | " + " | ".join(["---"] * len(sorted_hardware)) + " |\n"
    
    # Add rows for each architecture
    for arch in sorted_architectures:
        row = f"| **{arch}** | "
        
        for hw in sorted_hardware:
            hw_data = matrix_data["architecture_hardware_compatibility"][arch][hw]
            compatible = hw_data.get("compatible", False)
            performance = hw_data.get("performance", "n/a")
            
            if compatible:
                if performance == "excellent":
                    cell = "✅ **Excellent**"
                elif performance == "good":
                    cell = "✅ Good"
                elif performance == "moderate":
                    cell = "✓ Moderate"
                elif performance == "poor":
                    cell = "⚠️ Poor"
                elif performance == "baseline":
                    cell = "✓ Baseline"
                else:
                    cell = "✓ Compatible"
            else:
                cell = "❌ No"
                
            row += f"{cell} | "
            
        table += row + "\n"
    
    return table


def generate_detailed_report(matrix_data, architecture_types=None, hardware_types=None):
    """
    Generate a detailed markdown report with architecture-hardware compatibility details.
    
    Args:
        matrix_data: The compatibility matrix data
        architecture_types: List of architecture types to include (default: all)
        hardware_types: List of hardware types to include (default: all)
    
    Returns:
        Markdown string with the detailed report
    """
    # Get architecture and hardware types
    all_architectures = list(matrix_data["architecture_hardware_compatibility"].keys())
    architectures = architecture_types or all_architectures
    
    all_hardware = list(matrix_data["architecture_hardware_compatibility"][all_architectures[0]].keys())
    hardware = hardware_types or all_hardware
    
    # Create the report
    report = f"# Hardware Compatibility Report\n\n"
    report += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # Add metadata
    metadata = matrix_data.get("metadata", {})
    report += f"**Version:** {metadata.get('version', 'N/A')}\n"
    report += f"**Last Updated:** {metadata.get('last_updated', 'N/A')}\n"
    report += f"**Description:** {metadata.get('description', 'N/A')}\n\n"
    
    # Add compatibility table
    report += "## Compatibility Overview\n\n"
    report += generate_compatibility_table(matrix_data, architectures, hardware)
    report += "\n\n"
    
    # Add detailed sections for each architecture
    report += "## Detailed Architecture Compatibility\n\n"
    
    for arch in architectures:
        report += f"### {arch.capitalize()} Architecture\n\n"
        
        if arch in matrix_data["architecture_hardware_compatibility"]:
            arch_data = matrix_data["architecture_hardware_compatibility"][arch]
            
            # Add examples
            example_models = []
            for hw in hardware:
                if hw in arch_data:
                    hw_examples = arch_data[hw].get("examples", [])
                    example_models.extend(hw_examples)
            
            # Deduplicate examples
            example_models = list(set(example_models))
            
            if example_models:
                report += "**Example Models:** " + ", ".join(f"`{model}`" for model in example_models) + "\n\n"
            
            # Add hardware support details
            report += "| Hardware | Compatibility | Performance | Notes | Optimizations |\n"
            report += "| --- | --- | --- | --- | --- |\n"
            
            for hw in hardware:
                if hw in arch_data:
                    hw_data = arch_data[hw]
                    compatible = "✅ Yes" if hw_data.get("compatible", False) else "❌ No"
                    performance = hw_data.get("performance", "n/a")
                    notes = hw_data.get("notes", "")
                    optimizations = ", ".join(hw_data.get("optimizations", []))
                    
                    report += f"| **{hw.upper()}** | {compatible} | {performance} | {notes} | {optimizations} |\n"
            
            report += "\n"
        else:
            report += "No compatibility data available for this architecture.\n\n"
    
    # Add hardware backend details
    report += "## Hardware Backend Details\n\n"
    
    # CPU
    if "cpu" in hardware:
        report += "### CPU\n\n"
        report += "CPU is the default fallback for all models. It offers universal compatibility but typically with baseline performance.\n\n"
        report += "**Key Benefits:**\n"
        report += "- Universal compatibility\n"
        report += "- No special hardware required\n"
        report += "- Predictable behavior\n\n"
        report += "**Limitations:**\n"
        report += "- Slower inference compared to accelerated hardware\n"
        report += "- Memory constraints for larger models\n\n"
    
    # CUDA
    if "cuda" in hardware:
        report += "### CUDA (NVIDIA GPUs)\n\n"
        report += "CUDA provides excellent performance for all model types on NVIDIA GPUs.\n\n"
        report += "**Key Benefits:**\n"
        report += "- Excellent performance across all model types\n"
        report += "- Mature ecosystem with optimized kernels\n"
        report += "- Wide range of supporting libraries\n\n"
        report += "**Limitations:**\n"
        report += "- Requires NVIDIA GPU hardware\n"
        report += "- Memory constraints for very large models\n\n"
    
    # ROCm
    if "rocm" in hardware:
        report += "### ROCm (AMD GPUs)\n\n"
        report += "ROCm provides excellent performance for all model types on AMD GPUs through the HIP API or CUDA compatibility layer.\n\n"
        report += "**Key Benefits:**\n"
        report += "- Excellent performance on AMD GPU hardware\n"
        report += "- CUDA compatibility layer for broad support\n"
        report += "- Growing ecosystem of optimized kernels\n\n"
        report += "**Limitations:**\n"
        report += "- Requires AMD GPU hardware\n"
        report += "- Some advanced optimizations may lag behind CUDA\n\n"
    
    # MPS
    if "mps" in hardware:
        report += "### MPS (Apple Silicon)\n\n"
        report += "Apple's Metal Performance Shaders (MPS) provides good to excellent performance on Apple Silicon hardware.\n\n"
        report += "**Key Benefits:**\n"
        report += "- Excellent performance on Apple Silicon\n"
        report += "- Unified memory architecture\n"
        report += "- Power-efficient inference\n\n"
        report += "**Limitations:**\n"
        report += "- Limited to Apple hardware\n"
        report += "- Memory constraints for large models\n"
        report += "- Limited support for very large models\n\n"
    
    # OpenVINO
    if "openvino" in hardware:
        report += "### OpenVINO (Intel)\n\n"
        report += "Intel's OpenVINO provides optimized inference on Intel CPUs, GPUs, and specialized hardware.\n\n"
        report += "**Key Benefits:**\n"
        report += "- Optimized for Intel hardware\n"
        report += "- INT8 quantization support\n"
        report += "- Model optimization capabilities\n\n"
        report += "**Limitations:**\n"
        report += "- Limited support for specialized architectures\n"
        report += "- Best performance on Intel hardware\n\n"
    
    # QNN
    if "qnn" in hardware:
        report += "### QNN (Qualcomm)\n\n"
        report += "Qualcomm Neural Network provides acceleration on Qualcomm NPUs, primarily for vision and simpler text models.\n\n"
        report += "**Key Benefits:**\n"
        report += "- Power-efficient inference on mobile\n"
        report += "- Specialized for edge devices\n"
        report += "- Good performance for vision models\n\n"
        report += "**Limitations:**\n"
        report += "- Limited support for large and complex models\n"
        report += "- Requires fixed input shapes\n"
        report += "- Limited availability outside Qualcomm hardware\n\n"
    
    return report


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Generate hardware compatibility report")
    parser.add_argument("--input", default="hardware_compatibility_matrix.json", help="Input JSON file")
    parser.add_argument("--output", default="hardware_compatibility_report.md", help="Output markdown file")
    parser.add_argument("--architecture", action="append", help="Filter by architecture type")
    parser.add_argument("--hardware", action="append", help="Filter by hardware type")
    args = parser.parse_args()
    
    # Load compatibility matrix
    matrix_data = load_compatibility_matrix(args.input)
    
    # Generate report
    report = generate_detailed_report(matrix_data, args.architecture, args.hardware)
    
    # Write report to file
    with open(args.output, 'w') as f:
        f.write(report)
    
    print(f"Report generated and saved to {args.output}")


if __name__ == "__main__":
    main()