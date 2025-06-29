#!/usr/bin/env python3
"""
Visualization script for HuggingFace test coverage.
This script generates visual reports of the current test coverage status.
"""

import os
import sys
import json
import glob
import re
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter, defaultdict

# Constants
OUTPUT_DIR = "coverage_visualizations"
MODEL_TYPES_JSON = "huggingface_model_types.json"
ARCHITECTURES = {
    "encoder_only": ["bert", "vit", "roberta", "albert", "electra", "convbert"],
    "decoder_only": ["gpt2", "gptj", "gpt_neo", "llama", "falcon", "mistral"],
    "encoder_decoder": ["t5", "bart", "pegasus", "mbart", "m2m_100", "led"],
    "vision": ["vit", "swin", "beit", "deit", "convnext", "sam"],
    "multimodal": ["clip", "blip", "llava", "flava", "idefics", "paligemma"],
    "audio": ["wav2vec2", "hubert", "whisper", "unispeech", "clap", "encodec"]
}

def load_model_types():
    """Load model types from JSON file."""
    if os.path.exists(MODEL_TYPES_JSON):
        with open(MODEL_TYPES_JSON, 'r') as f:
            return json.load(f)
    
    # If model types file doesn't exist, create a simulated list
    # based on existing test files
    model_types = []
    # Check both main directory and fixed_tests directory
    test_files = glob.glob("test_hf_*.py") + glob.glob("fixed_tests/test_hf_*.py")
    for test_file in test_files:
        model_name = os.path.basename(test_file).replace('test_hf_', '').replace('.py', '')
        if model_name not in model_types:
            model_types.append(model_name)
    
    # Add some known model types that might not have tests yet
    additional_types = [
        "gpt2", "bert", "t5", "vit", "llama", "falcon", "mistral", "roberta", 
        "albert", "bart", "pegasus", "clip", "blip", "wav2vec2", "whisper"
    ]
    for model in additional_types:
        if model not in model_types:
            model_types.append(model)
    
    return model_types

def get_implemented_models():
    """Get list of implemented models from test files."""
    # Check both main directory and fixed_tests directory
    test_files = glob.glob("test_hf_*.py") + glob.glob("fixed_tests/test_hf_*.py")
    implemented = []
    
    for test_file in test_files:
        model_name = os.path.basename(test_file).replace('test_hf_', '').replace('.py', '')
        if model_name not in implemented:
            implemented.append(model_name)
    
    return implemented

def categorize_models(models):
    """Categorize models by architecture type."""
    categorized = defaultdict(list)
    uncategorized = []
    
    for model in models:
        categorized_flag = False
        for arch, arch_models in ARCHITECTURES.items():
            for pattern in arch_models:
                if model == pattern or model.startswith(pattern + "_"):
                    categorized[arch].append(model)
                    categorized_flag = True
                    break
            if categorized_flag:
                break
        
        if not categorized_flag:
            uncategorized.append(model)
    
    # Add uncategorized as its own category
    if uncategorized:
        categorized["uncategorized"] = uncategorized
    
    return categorized

def generate_coverage_chart(model_types, implemented, output_dir):
    """Generate a bar chart of test coverage."""
    os.makedirs(output_dir, exist_ok=True)
    
    total = len(model_types)
    implemented_count = len(implemented)
    missing_count = total - implemented_count
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(
        ['Implemented', 'Missing'], 
        [implemented_count, missing_count],
        color=['#4CAF50', '#F44336']
    )
    
    # Add percentage labels on the bars
    for bar in bars:
        height = bar.get_height()
        percentage = 100 * height / total
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 5,
            f'{height} ({percentage:.1f}%)',
            ha='center', 
            va='bottom'
        )
    
    # Add title and labels
    ax.set_title('HuggingFace Model Test Coverage', fontsize=16)
    ax.set_ylabel('Number of Models', fontsize=12)
    ax.set_ylim(0, total * 1.1)  # Add some space for the labels
    
    # Add timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    fig.text(0.95, 0.05, f'Generated: {timestamp}', ha='right', va='bottom', fontsize=8)
    
    # Save chart
    output_path = os.path.join(output_dir, 'coverage_summary.png')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Coverage summary chart saved to: {output_path}")
    
    # Return for showing
    return fig

def generate_architecture_chart(implemented, output_dir):
    """Generate a chart of test coverage by architecture."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Categorize implemented models
    categorized = categorize_models(implemented)
    
    # Prepare data for chart
    architectures = list(categorized.keys())
    counts = [len(categorized[arch]) for arch in architectures]
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bars = ax.bar(
        architectures, 
        counts,
        color=[
            '#4CAF50',  # Green
            '#2196F3',  # Blue
            '#FFC107',  # Amber
            '#9C27B0',  # Purple
            '#FF5722',  # Deep Orange
            '#795548',  # Brown
            '#607D8B'   # Blue Grey
        ][:len(architectures)]
    )
    
    # Add count labels on the bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 0.5,
            f'{height}',
            ha='center', 
            va='bottom'
        )
    
    # Add title and labels
    ax.set_title('HuggingFace Model Test Coverage by Architecture', fontsize=16)
    ax.set_ylabel('Number of Models', fontsize=12)
    ax.set_xlabel('Architecture Type', fontsize=12)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    fig.text(0.95, 0.05, f'Generated: {timestamp}', ha='right', va='bottom', fontsize=8)
    
    # Save chart
    output_path = os.path.join(output_dir, 'coverage_by_architecture.png')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Architecture coverage chart saved to: {output_path}")
    
    # Return for showing
    return fig

def generate_coverage_report(model_types, implemented, output_dir):
    """Generate a detailed coverage report in markdown format."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure we don't have duplicate models
    unique_model_types = list(set(model_types))
    unique_implemented = list(set(implemented))
    
    # Calculate coverage statistics
    total = len(unique_model_types)
    implemented_count = len([m for m in unique_implemented if m in unique_model_types])
    additional_models = len([m for m in unique_implemented if m not in unique_model_types])
    missing_count = total - implemented_count
    coverage_percentage = 100 * implemented_count / total if total > 0 else 0
    
    # Get missing models
    missing = [model for model in unique_model_types if model not in unique_implemented]
    
    # Categorize implemented models
    categorized_implemented = categorize_models(unique_implemented)
    
    # Categorize missing models
    categorized_missing = categorize_models(missing)
    
    # Generate report
    report = [
        "# HuggingFace Model Test Coverage Report",
        f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"\n## Coverage Summary",
        f"\n- **Total models from HF:** {total}",
        f"- **Implemented (from HF list):** {implemented_count} ({coverage_percentage:.1f}%)",
        f"- **Additional models implemented:** {additional_models}",
        f"- **Missing models:** {missing_count}",
        
        f"\n## Coverage by Architecture",
        "\n| Architecture | Implemented | Missing |",
        "| --- | --- | --- |"
    ]
    
    # Add architecture-specific coverage
    all_architectures = set(list(categorized_implemented.keys()) + list(categorized_missing.keys()))
    for arch in sorted(all_architectures):
        implemented_in_arch = len(categorized_implemented.get(arch, []))
        missing_in_arch = len(categorized_missing.get(arch, []))
        total_in_arch = implemented_in_arch + missing_in_arch
        percentage = 100 * implemented_in_arch / total_in_arch if total_in_arch > 0 else 0
        
        report.append(f"| {arch} | {implemented_in_arch} ({percentage:.1f}%) | {missing_in_arch} |")
    
    # Add detailed lists of models
    report.extend([
        f"\n## Implemented Models",
        "\nThe following models have been implemented and tested:"
    ])
    
    for arch in sorted(categorized_implemented.keys()):
        report.append(f"\n### {arch.replace('_', ' ').title()}")
        for model in sorted(categorized_implemented[arch]):
            report.append(f"- {model}")
    
    report.extend([
        f"\n## Missing Models",
        "\nThe following models still need to be implemented:"
    ])
    
    for arch in sorted(categorized_missing.keys()):
        report.append(f"\n### {arch.replace('_', ' ').title()}")
        for model in sorted(categorized_missing[arch]):
            report.append(f"- {model}")
    
    # Write report to file
    output_path = os.path.join(output_dir, 'detailed_coverage_report.md')
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Detailed coverage report saved to: {output_path}")
    
    # Create a shorter summary for quick reference
    summary_path = os.path.join(output_dir, 'coverage_summary.md')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(report[:15]))  # Just the summary sections
    
    print(f"Coverage summary saved to: {summary_path}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Visualize HuggingFace model test coverage")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help=f"Output directory for visualizations (default: {OUTPUT_DIR})")
    parser.add_argument("--show", action="store_true",
                        help="Show visualizations (requires display)")
    parser.add_argument("--report-only", action="store_true",
                        help="Generate only text reports (no charts)")
    args = parser.parse_args()
    
    print("Loading model types...")
    model_types = load_model_types()
    
    print("Analyzing implemented models...")
    implemented = get_implemented_models()
    
    if not args.report_only:
        print("Generating coverage chart...")
        fig1 = generate_coverage_chart(model_types, implemented, args.output_dir)
        
        print("Generating architecture chart...")
        fig2 = generate_architecture_chart(implemented, args.output_dir)
    
    print("Generating coverage report...")
    generate_coverage_report(model_types, implemented, args.output_dir)
    
    print("Coverage analysis complete!")
    
    if args.show and not args.report_only:
        plt.show()

if __name__ == "__main__":
    main()