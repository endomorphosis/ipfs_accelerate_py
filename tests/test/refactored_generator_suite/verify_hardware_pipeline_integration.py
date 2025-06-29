#!/usr/bin/env python3
"""
Verify hardware backend integration with pipeline templates.

This script tests the integration between all pipeline templates and hardware backends
to ensure they work correctly together.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# Import from local modules
from generator_core.config import config_manager
from hardware.hardware_detection import detect_available_hardware, SUPPORTED_BACKENDS
from templates.base_pipeline import BasePipelineTemplate
from templates.base_hardware import BaseHardwareTemplate


# Define all pipeline types
PIPELINE_TYPES = [
    "text",
    "vision",
    "vision-text",
    "audio",
    "multimodal",
    "diffusion",
    "moe",  # Mixture of Experts
    "state-space",
    "rag"
]

# Define representative models for each pipeline type
REPRESENTATIVE_MODELS = {
    "text": "bert-base-uncased",
    "vision": "google/vit-base-patch16-224",
    "vision-text": "openai/clip-vit-base-patch32",
    "audio": "openai/whisper-tiny",
    "multimodal": "facebook/flava-full",
    "diffusion": "stabilityai/stable-diffusion-2-1-base",
    "moe": "mistralai/Mixtral-8x7B-v0.1",
    "state-space": "state-spaces/mamba-2.8b",
    "rag": "facebook/rag-token-nq"
}

# Define architecture types for each pipeline type
ARCHITECTURE_TYPES = {
    "text": ["encoder-only", "decoder-only", "encoder-decoder"],
    "vision": ["vision"],
    "vision-text": ["vision-encoder-text-decoder"],
    "audio": ["speech"],
    "multimodal": ["multimodal"],
    "diffusion": ["diffusion"],
    "moe": ["mixture-of-experts"],
    "state-space": ["state-space"],
    "rag": ["rag"]
}

# Define task types for each pipeline type
TASK_TYPES = {
    "text": ["text_embedding", "text_generation", "text2text_generation"],
    "vision": ["image_classification", "image_segmentation"],
    "vision-text": ["image_to_text", "visual_question_answering"],
    "audio": ["speech_recognition", "audio_classification"],
    "multimodal": ["multimodal_classification", "multimodal_generation"],
    "diffusion": ["image_generation", "image_to_image"],
    "moe": ["text_generation", "text_classification"],
    "state-space": ["text_generation", "sequence_modeling"],
    "rag": ["retrieval_generation", "document_qa"]
}


def import_pipeline_template(pipeline_type: str) -> BasePipelineTemplate:
    """Import a pipeline template class based on its type."""
    module_name = f"templates.{pipeline_type.replace('-', '_')}_pipeline"
    try:
        module = __import__(module_name, fromlist=["*"])
        
        # Handle special cases for class naming
        if pipeline_type == "moe":
            class_name = "MixOfExpertsPipelineTemplate"
        elif pipeline_type == "rag":
            class_name = "RAGPipelineTemplate"
        elif pipeline_type == "vision-text":
            class_name = "VisionTextPipelineTemplate"
        else:
            class_name = "".join(word.capitalize() for word in pipeline_type.replace("-", "_").split("_")) + "PipelineTemplate"
        
        template_class = getattr(module, class_name)
        return template_class()
    except (ImportError, AttributeError) as e:
        print(f"Error importing {pipeline_type} pipeline template: {e}")
        return None


def import_hardware_template(hardware_type: str) -> BaseHardwareTemplate:
    """Import a hardware template class based on its type."""
    module_name = f"templates.{hardware_type}_hardware"
    try:
        module = __import__(module_name, fromlist=["*"])
        
        # Handle special cases for class naming
        if hardware_type == "cpu":
            class_name = "CPUHardwareTemplate"
        elif hardware_type == "mps":
            class_name = "MPSHardwareTemplate"
        elif hardware_type == "qnn":
            class_name = "QNNHardwareTemplate"
        else:
            class_name = "".join(word.capitalize() for word in hardware_type.split("_")) + "HardwareTemplate"
        
        template_class = getattr(module, class_name)
        return template_class()
    except (ImportError, AttributeError) as e:
        print(f"Error importing {hardware_type} hardware template: {e}")
        return None


def verify_pipeline_hardware_integration(
    pipeline_type: str,
    hardware_type: str,
    arch_type: str,
    task_type: str
) -> Dict[str, Any]:
    """
    Verify integration between a pipeline template and hardware backend.
    
    Args:
        pipeline_type: The pipeline type (text, vision, etc.)
        hardware_type: The hardware backend (cpu, cuda, rocm, etc.)
        arch_type: The architecture type (encoder-only, etc.)
        task_type: The task type (text_embedding, etc.)
        
    Returns:
        Dictionary with verification results
    """
    result = {
        "pipeline_type": pipeline_type,
        "hardware_type": hardware_type,
        "arch_type": arch_type,
        "task_type": task_type,
        "success": False,
        "errors": [],
        "warnings": []
    }
    
    # Import templates
    pipeline_template = import_pipeline_template(pipeline_type)
    hardware_template = import_hardware_template(hardware_type)
    
    if pipeline_template is None:
        result["errors"].append(f"Failed to import {pipeline_type} pipeline template")
        return result
    
    if hardware_template is None:
        result["errors"].append(f"Failed to import {hardware_type} hardware template")
        return result
    
    # Verify architecture compatibility
    if not pipeline_template.is_compatible_with_architecture(arch_type):
        result["warnings"].append(f"{pipeline_type} pipeline is not compatible with {arch_type} architecture")
    
    if not hardware_template.is_compatible_with_architecture(arch_type):
        result["warnings"].append(f"{hardware_type} hardware is not compatible with {arch_type} architecture")
    
    # Verify task compatibility
    if not pipeline_template.is_compatible_with_task(task_type):
        result["warnings"].append(f"{pipeline_type} pipeline is not compatible with {task_type} task")
    
    # Verify template methods
    required_pipeline_methods = [
        "get_preprocessing_code",
        "get_postprocessing_code",
        "get_result_formatting_code"
    ]
    
    required_hardware_methods = [
        "get_hardware_init_code",
        "get_handler_creation_code",
        "get_inference_code",
        "get_cleanup_code"
    ]
    
    for method in required_pipeline_methods:
        if not hasattr(pipeline_template, method):
            result["errors"].append(f"{pipeline_type} pipeline is missing required method: {method}")
        else:
            try:
                # Try calling the method
                getattr(pipeline_template, method)(task_type)
            except Exception as e:
                result["errors"].append(f"Error calling {method} on {pipeline_type} pipeline: {e}")
    
    for method in required_hardware_methods:
        if not hasattr(hardware_template, method):
            result["errors"].append(f"{hardware_type} hardware is missing required method: {method}")
        else:
            try:
                # Try calling the method with appropriate arguments
                if method == "get_hardware_init_code" or method == "get_handler_creation_code":
                    getattr(hardware_template, method)("AutoModel", task_type)
                elif method == "get_cleanup_code":
                    # get_cleanup_code doesn't require any parameters
                    getattr(hardware_template, method)()
                else:
                    getattr(hardware_template, method)(task_type)
            except Exception as e:
                result["errors"].append(f"Error calling {method} on {hardware_type} hardware: {e}")
    
    # Verify integration by generating code snippets
    try:
        preprocessing_code = pipeline_template.get_preprocessing_code(task_type)
        hardware_init_code = hardware_template.get_hardware_init_code("AutoModel", task_type)
        handler_creation_code = hardware_template.get_handler_creation_code("AutoModel", task_type)
        inference_code = hardware_template.get_inference_code(task_type)
        postprocessing_code = pipeline_template.get_postprocessing_code(task_type)
        result_formatting_code = pipeline_template.get_result_formatting_code(task_type)
        cleanup_code = hardware_template.get_cleanup_code()
        
        # Check for compatibility issues
        if "CUDA" in hardware_init_code and hardware_type != "cuda" and hardware_type != "rocm":
            result["warnings"].append(f"{hardware_type} hardware template contains CUDA-specific code")
        
        if "MPS" in hardware_init_code and hardware_type != "apple":
            result["warnings"].append(f"{hardware_type} hardware template contains MPS-specific code")
        
        if "OpenVINO" in hardware_init_code and hardware_type != "openvino":
            result["warnings"].append(f"{hardware_type} hardware template contains OpenVINO-specific code")
        
        if "QNN" in hardware_init_code and hardware_type != "qnn":
            result["warnings"].append(f"{hardware_type} hardware template contains QNN-specific code")
        
        # Success if no errors were found
        if not result["errors"]:
            result["success"] = True
    
    except Exception as e:
        result["errors"].append(f"Error verifying integration: {e}")
    
    return result


def run_verification(args) -> Dict[str, Any]:
    """
    Run verification tests for pipeline-hardware integration.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Dictionary with verification results
    """
    results = {
        "timestamp": time.time(),
        "available_hardware": detect_available_hardware(),
        "results": [],
        "summary": {
            "total": 0,
            "success": 0,
            "failure": 0,
            "warnings": 0
        }
    }
    
    # Filter pipeline types if specified
    pipeline_types = [args.pipeline_type] if args.pipeline_type else PIPELINE_TYPES
    
    # Filter hardware types if specified
    hardware_types = [args.hardware_type] if args.hardware_type else SUPPORTED_BACKENDS
    
    # Run verification for each combination
    for pipeline_type in pipeline_types:
        for hardware_type in hardware_types:
            # Skip hardware that isn't available if requested
            if args.available_only and not results["available_hardware"][hardware_type]:
                continue
                
            # Get an architecture type for this pipeline
            arch_types = ARCHITECTURE_TYPES.get(pipeline_type, ["encoder-only"])
            arch_type = arch_types[0]  # Use the first architecture type
            
            # Get a task type for this pipeline
            task_types = TASK_TYPES.get(pipeline_type, ["text_embedding"])
            task_type = task_types[0]  # Use the first task type
            
            # Verify integration
            result = verify_pipeline_hardware_integration(
                pipeline_type=pipeline_type,
                hardware_type=hardware_type,
                arch_type=arch_type,
                task_type=task_type
            )
            
            results["results"].append(result)
            
            # Update summary
            results["summary"]["total"] += 1
            if result["success"]:
                results["summary"]["success"] += 1
            else:
                results["summary"]["failure"] += 1
            
            if result["warnings"]:
                results["summary"]["warnings"] += 1
    
    return results


def print_results(results: Dict[str, Any], verbose: bool = False) -> None:
    """Print verification results to the console."""
    print("\n=== Hardware-Pipeline Integration Verification Results ===\n")
    
    # Print available hardware
    print("Available Hardware:")
    for hw, available in results["available_hardware"].items():
        status = "✅" if available else "❌"
        print(f"  {hw}: {status}")
    
    print("\nIntegration Results:")
    for result in results["results"]:
        pipeline_type = result["pipeline_type"]
        hardware_type = result["hardware_type"]
        status = "✅" if result["success"] else "❌"
        
        print(f"  {pipeline_type} + {hardware_type}: {status}")
        
        if verbose or not result["success"]:
            if result["errors"]:
                print("    Errors:")
                for error in result["errors"]:
                    print(f"      - {error}")
            
            if result["warnings"]:
                print("    Warnings:")
                for warning in result["warnings"]:
                    print(f"      - {warning}")
    
    # Print summary
    print("\nSummary:")
    print(f"  Total Tests: {results['summary']['total']}")
    print(f"  Successful: {results['summary']['success']}")
    print(f"  Failed: {results['summary']['failure']}")
    print(f"  With Warnings: {results['summary']['warnings']}")
    
    # Print overall status
    success_pct = (results['summary']['success'] / results['summary']['total']) * 100 if results['summary']['total'] > 0 else 0
    if success_pct == 100:
        print("\n✅ All integration tests passed!")
    elif success_pct >= 80:
        print(f"\n⚠️ {success_pct:.1f}% of integration tests passed. Some issues need to be fixed.")
    else:
        print(f"\n❌ Only {success_pct:.1f}% of integration tests passed. Significant issues need to be fixed.")


def save_results(results: Dict[str, Any], output_file: str) -> None:
    """Save verification results to a JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Verify hardware backend integration with pipeline templates")
    parser.add_argument("--pipeline-type", help="Specific pipeline type to test")
    parser.add_argument("--hardware-type", help="Specific hardware type to test")
    parser.add_argument("--available-only", action="store_true", help="Only test available hardware")
    parser.add_argument("--output", help="Output file for results (JSON)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed results")
    args = parser.parse_args()
    
    # Run verification
    results = run_verification(args)
    
    # Print results
    print_results(results, args.verbose)
    
    # Save results if requested
    if args.output:
        save_results(results, args.output)
    
    # Return non-zero exit code if any failures
    if results["summary"]["failure"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()