#!/usr/bin/env python3
"""
Validate skillset pattern consistency.

This script analyzes the skillset files to ensure they follow consistent patterns.
"""

import os
import sys
import ast
import logging
import importlib
from typing import Dict, List, Any, Set, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_class_methods(file_path: str) -> Dict[str, Set[str]]:
    """
    Get the methods defined in each class in a file.
    
    Args:
        file_path: Path to the file to analyze
        
    Returns:
        Dictionary mapping class names to sets of method names
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Parse the file
        tree = ast.parse(content)
        
        # Find all classes
        classes = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_name = node.name
                method_names = {
                    n.name for n in node.body 
                    if isinstance(n, ast.FunctionDef) and not n.name.startswith('__')
                }
                classes[class_name] = method_names
        
        return classes
    
    except Exception as e:
        logger.error(f"Error analyzing {file_path}: {e}")
        return {}

def get_required_methods() -> Dict[str, Set[str]]:
    """
    Get the required methods for skillset classes.
    
    Returns:
        Dictionary mapping class types to sets of required method names
    """
    return {
        "common": {
            "get_default_model_id",
            "get_optimal_device",
            "load_model",
            "run_inference",
            "benchmark"
        },
        "encoder_only": {
            "init_cpu",
            "init_cuda",
            "init_openvino",
            "init_mps",
            "init_rocm",
            "init_qualcomm",
            "init_webnn",
            "init_webgpu",
            "create_cpu_handler",
            "create_cuda_handler",
            "create_openvino_handler",
            "create_mps_handler",
            "create_rocm_handler", 
            "create_qualcomm_handler",
            "create_webnn_handler",
            "create_webgpu_handler",
            "run"
        },
        "encoder_decoder": {
            "init_cpu",
            "init_cuda",
            "init_openvino",
            "init_mps",
            "init_rocm",
            "init_qualcomm",
            "init_webnn",
            "init_webgpu",
            "create_cpu_handler",
            "create_cuda_handler",
            "create_openvino_handler",
            "create_mps_handler",
            "create_rocm_handler",
            "create_qualcomm_handler", 
            "create_webnn_handler",
            "create_webgpu_handler",
            "run"
        }
    }

def analyze_skillset_file(file_path: str) -> Dict[str, Any]:
    """
    Analyze a skillset file for pattern compliance.
    
    Args:
        file_path: Path to the file to analyze
        
    Returns:
        Analysis results
    """
    # Extract model name from filename
    filename = os.path.basename(file_path)
    parts = filename.split("_")
    if len(parts) < 3:
        return {"error": "Invalid filename format"}
    
    model_name = parts[0]
    device = parts[1]
    
    # Get required methods
    required_methods = get_required_methods()
    
    # Analyze file
    classes = get_class_methods(file_path)
    
    # Find the skillset class
    skillset_classes = [
        name for name in classes.keys()
        if "Skillset" in name
    ]
    
    if not skillset_classes:
        return {"error": "No skillset class found"}
    
    # Analyze the first skillset class
    skillset_class = skillset_classes[0]
    methods = classes[skillset_class]
    
    # Check for required methods
    common_missing = required_methods["common"] - methods
    
    # Determine architecture based on class name or methods
    architecture = "unknown"
    if "AutoModelForMaskedLM" in skillset_class or "Encoder" in skillset_class:
        architecture = "encoder_only"
    elif "AutoModelForSeq2SeqLM" in skillset_class or "EncoderDecoder" in skillset_class:
        architecture = "encoder_decoder"
    elif "AutoModelForCausalLM" in skillset_class or "Decoder" in skillset_class:
        architecture = "decoder_only"
    
    # Check specific requirements based on architecture
    arch_missing = set()
    if architecture in required_methods:
        arch_missing = required_methods[architecture] - methods
    
    # Compile results
    return {
        "model_name": model_name,
        "device": device,
        "skillset_class": skillset_class,
        "architecture": architecture,
        "method_count": len(methods),
        "missing_common_methods": list(common_missing),
        "missing_architecture_methods": list(arch_missing),
        "has_all_required_methods": len(common_missing) == 0 and len(arch_missing) == 0
    }

def analyze_all_skillsets() -> Dict[str, Dict[str, Any]]:
    """
    Analyze all skillset files.
    
    Returns:
        Dictionary mapping filenames to analysis results
    """
    # Get all skillset files
    skillset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skillsets")
    skillset_files = [f for f in os.listdir(skillset_dir) if f.endswith("_skillset.py")]
    
    # Analyze each file
    results = {}
    
    for skillset_file in skillset_files:
        file_path = os.path.join(skillset_dir, skillset_file)
        results[skillset_file] = analyze_skillset_file(file_path)
    
    return results

if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Validate skillset patterns")
    parser.add_argument("--file", "-f", type=str, help="Specific file to analyze")
    parser.add_argument("--model", "-m", type=str, help="Analyze files for a specific model")
    parser.add_argument("--json", "-j", action="store_true", help="Output in JSON format")
    
    args = parser.parse_args()
    
    if args.file:
        # Analyze specific file
        result = analyze_skillset_file(args.file)
        
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Analysis of {args.file}:")
            print(f"  Model: {result.get('model_name')}")
            print(f"  Architecture: {result.get('architecture')}")
            print(f"  Skillset class: {result.get('skillset_class')}")
            print(f"  Method count: {result.get('method_count')}")
            print(f"  Missing common methods: {', '.join(result.get('missing_common_methods', []))}")
            print(f"  Missing architecture methods: {', '.join(result.get('missing_architecture_methods', []))}")
            print(f"  Compliance: {'Complete' if result.get('has_all_required_methods', False) else 'Incomplete'}")
    
    elif args.model:
        # Analyze files for a specific model
        skillset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skillsets")
        skillset_files = [
            f for f in os.listdir(skillset_dir) 
            if f.startswith(f"{args.model}_") and f.endswith("_skillset.py")
        ]
        
        results = {}
        for file in skillset_files:
            file_path = os.path.join(skillset_dir, file)
            results[file] = analyze_skillset_file(file_path)
        
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            for file, result in results.items():
                print(f"Analysis of {file}:")
                print(f"  Architecture: {result.get('architecture')}")
                print(f"  Compliance: {'Complete' if result.get('has_all_required_methods', False) else 'Incomplete'}")
    
    else:
        # Analyze all skillsets
        results = analyze_all_skillsets()
        
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            complete_count = sum(1 for r in results.values() if r.get('has_all_required_methods', False))
            print(f"Analyzed {len(results)} skillset files")
            print(f"Compliant: {complete_count}/{len(results)}")
            
            # Print non-compliant files
            non_compliant = [
                (file, result) for file, result in results.items() 
                if not result.get('has_all_required_methods', False)
            ]
            
            if non_compliant:
                print("\nNon-compliant skillsets:")
                for file, result in non_compliant:
                    print(f"  - {file} ({result.get('architecture')})")
                    if result.get('missing_common_methods'):
                        print(f"    Missing common methods: {', '.join(result.get('missing_common_methods', []))}")
                    if result.get('missing_architecture_methods'):
                        print(f"    Missing architecture methods: {', '.join(result.get('missing_architecture_methods', []))}")