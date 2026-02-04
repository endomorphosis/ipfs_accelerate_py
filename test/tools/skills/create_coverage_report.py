#!/usr/bin/env python3
"""
Create a coverage report for HuggingFace model tests.

This script analyzes the test files in the fixed_tests directory
and creates a coverage report showing which model architectures have tests.

Usage:
    python create_coverage_report.py [--output MARKDOWN_FILE]
"""

import os
import sys
import logging
import argparse
import datetime
from pathlib import Path
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Model architecture categories
ARCHITECTURE_TYPES = {
    "encoder-only": ["bert", "distilbert", "roberta", "electra", "camembert", "xlm-roberta", "deberta", "albert", "ernie", "rembert"],
    "decoder-only": ["gpt2", "gpt-j", "gpt-neo", "gpt-neox", "bloom", "llama", "mistral", "falcon", "phi", "mixtral", "mpt", "gemma", "opt"],
    "encoder-decoder": ["t5", "bart", "pegasus", "mbart", "mt5", "longt5", "led", "marian", "prophetnet"],
    "vision": ["vit", "swin", "deit", "beit", "convnext", "poolformer", "dinov2", "segformer", "detr", "yolos", "mask2former", "sam"],
    "vision-text": ["vision-encoder-decoder", "vision-text-dual-encoder", "clip", "blip", "git", "flava", "paligemma"],
    "speech": ["wav2vec2", "hubert", "whisper", "bark", "speecht5", "sew", "encodec", "musicgen", "audio", "clap"],
    "multimodal": ["llava", "clip", "blip", "git", "pix2struct", "paligemma", "video-llava", "idefics", "imagebind"]
}

def check_file_validity(file_path):
    """Check if a Python file is syntactically valid."""
    try:
        subprocess.run(
            [sys.executable, "-m", "py_compile", file_path], 
            check=True, capture_output=True
        )
        return True
    except subprocess.CalledProcessError:
        return False

def get_architecture_type(model_type):
    """Determine architecture type based on model type."""
    model_type_lower = model_type.lower()
    for arch_type, models in ARCHITECTURE_TYPES.items():
        if any(model in model_type_lower for model in models):
            return arch_type
    return "unknown"  # Default if unknown

def run_test(test_file):
    """Run a test file with --list-models to see if it works."""
    try:
        result = subprocess.run(
            [sys.executable, test_file, "--list-models"], 
            check=True, capture_output=True, text=True
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def extract_model_info(file_path):
    """Extract model type from a test file path."""
    file_name = os.path.basename(file_path)
    if not file_name.startswith("test_hf_"):
        return None
    
    model_type = file_name[len("test_hf_"):-len(".py")]
    return model_type.replace("_", "-")

def create_report(test_dir="fixed_tests", output_file=None):
    """Create a coverage report for the test files."""
    # Get all test files
    test_files = []
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.startswith("test_hf_") and file.endswith(".py"):
                test_files.append(os.path.join(root, file))
    
    # Group by architecture type
    by_architecture = {}
    for test_file in test_files:
        model_type = extract_model_info(test_file)
        if not model_type:
            continue
        
        arch_type = get_architecture_type(model_type)
        if arch_type not in by_architecture:
            by_architecture[arch_type] = []
        
        valid = check_file_validity(test_file)
        runnable = False
        models = None
        
        if valid:
            runnable, output = run_test(test_file)
            if runnable:
                models = output
        
        by_architecture[arch_type].append({
            "file": os.path.basename(test_file),
            "model_type": model_type,
            "valid": valid,
            "runnable": runnable,
            "models": models
        })
    
    # Sort entries in each architecture
    for arch_type in by_architecture:
        by_architecture[arch_type] = sorted(
            by_architecture[arch_type], 
            key=lambda x: x["model_type"]
        )
    
    # Create the report text
    report = []
    report.append("# HuggingFace Model Test Coverage")
    report.append("")
    report.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append("## Summary")
    report.append("")
    
    total_tests = sum(len(tests) for tests in by_architecture.values())
    valid_tests = sum(sum(1 for test in tests if test["valid"]) for tests in by_architecture.values())
    runnable_tests = sum(sum(1 for test in tests if test["runnable"]) for tests in by_architecture.values())
    
    report.append(f"- Total test files: {total_tests}")
    report.append(f"- Syntactically valid files: {valid_tests} ({valid_tests/total_tests*100:.1f}%)")
    report.append(f"- Runnable files: {runnable_tests} ({runnable_tests/total_tests*100:.1f}%)")
    report.append("")
    
    # Count potential models vs. tested
    potential_models = 0
    for arch_type, models in ARCHITECTURE_TYPES.items():
        potential_models += len(models)
    
    tested_models = len({test["model_type"] for tests in by_architecture.values() for test in tests})
    report.append(f"- Known model types: {potential_models}")
    report.append(f"- Tested model types: {tested_models} ({tested_models/potential_models*100:.1f}%)")
    report.append("")
    
    # List of hyphenated models with tests
    hyphenated_models = [test["model_type"] for tests in by_architecture.values() 
                         for test in tests if "-" in test["model_type"]]
    report.append("## Hyphenated Models")
    report.append("")
    if hyphenated_models:
        for model in sorted(hyphenated_models):
            report.append(f"- {model}")
    else:
        report.append("No tests for hyphenated models found.")
    report.append("")
    
    # Coverage by architecture
    report.append("## Coverage by Architecture")
    report.append("")
    
    for arch_type, arch_models in sorted(ARCHITECTURE_TYPES.items()):
        report.append(f"### {arch_type}")
        report.append("")
        
        # Get tested models for this architecture
        if arch_type in by_architecture:
            tested = set(test["model_type"] for test in by_architecture[arch_type])
        else:
            tested = set()
            
        # Count coverage
        covered = len(tested)
        total = len(arch_models)
        report.append(f"- Models in this architecture: {total}")
        report.append(f"- Models with tests: {covered}")
        report.append(f"- Coverage: {covered/total*100:.1f}%")
        report.append("")
        
        # List tested models
        report.append("#### Tested Models")
        report.append("")
        if tested:
            for model in sorted(tested):
                report.append(f"- {model}")
        else:
            report.append("No tests found for this architecture.")
        report.append("")
        
        # List untested models
        report.append("#### Untested Models")
        report.append("")
        untested = set(arch_models) - tested
        if untested:
            for model in sorted(untested):
                report.append(f"- {model}")
        else:
            report.append("All models in this architecture have tests.")
        report.append("")
    
    # List all test files
    report.append("## Test Files")
    report.append("")
    report.append("| File | Model Type | Architecture | Valid | Runnable |")
    report.append("|------|------------|--------------|-------|----------|")
    
    all_tests = []
    for arch_type, tests in by_architecture.items():
        for test in tests:
            all_tests.append({
                "file": test["file"],
                "model_type": test["model_type"],
                "architecture": arch_type,
                "valid": test["valid"],
                "runnable": test["runnable"]
            })
    
    for test in sorted(all_tests, key=lambda x: x["file"]):
        valid_mark = "✅" if test["valid"] else "❌"
        runnable_mark = "✅" if test["runnable"] else "❌"
        report.append(f"| {test['file']} | {test['model_type']} | {test['architecture']} | {valid_mark} | {runnable_mark} |")
    
    report_text = "\n".join(report)
    
    # Output to file or stdout
    if output_file:
        with open(output_file, "w") as f:
            f.write(report_text)
        logger.info(f"Report saved to {output_file}")
    else:
        print(report_text)
    
    return report_text

def main():
    parser = argparse.ArgumentParser(description="Create coverage report for HuggingFace model tests")
    parser.add_argument("--test-dir", type=str, default="fixed_tests",
                        help="Directory containing test files")
    parser.add_argument("--output", type=str, help="Output markdown file")
    
    args = parser.parse_args()
    
    create_report(args.test_dir, args.output)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())