#!/usr/bin/env python
"""
Test generator interactive script for IPFS Accelerate.

This script provides an interactive way to generate test files
using the template system.
"""

import os
import sys
import logging
import argparse
from typing import Dict, List, Any, Optional

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import templates
from template_system.templates.model_test_template import ModelTestTemplate
from template_system.templates.hardware_test_template import HardwareTestTemplate
from template_system.templates.api_test_template import APITestTemplate


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def interactive_model_test() -> Dict[str, Any]:
    """Interactive prompts for model test parameters."""
    params = {}
    
    print("\n=== Model Test Generator ===\n")
    
    params['model_name'] = input("Model Name (e.g., bert-base-uncased): ")
    
    model_type = ""
    while model_type not in ['text', 'vision', 'audio', 'multimodal']:
        model_type = input("Model Type [text/vision/audio/multimodal]: ").lower()
        if model_type not in ['text', 'vision', 'audio', 'multimodal']:
            print("Invalid model type. Please choose from text, vision, audio, or multimodal.")
    
    params['model_type'] = model_type
    
    test_name = input("Test Name (default: auto-generate from model name): ")
    if not test_name:
        test_name = params['model_name'].replace('-', '_').lower()
    params['test_name'] = test_name
    
    return params


def interactive_hardware_test() -> Dict[str, Any]:
    """Interactive prompts for hardware test parameters."""
    params = {}
    
    print("\n=== Hardware Test Generator ===\n")
    
    platform = ""
    while platform not in ['webgpu', 'webnn', 'cuda', 'rocm', 'cpu']:
        platform = input("Hardware Platform [webgpu/webnn/cuda/rocm/cpu]: ").lower()
        if platform not in ['webgpu', 'webnn', 'cuda', 'rocm', 'cpu']:
            print("Invalid platform. Please choose from webgpu, webnn, cuda, rocm, or cpu.")
    
    params['hardware_platform'] = platform
    
    test_name = input("Test Name (default: auto-generate from platform): ")
    if not test_name:
        test_name = f"{platform}_test"
    params['test_name'] = test_name
    
    test_category = input("Test Category (e.g., compute_shaders, inference): ")
    if test_category:
        params['test_category'] = test_category
    
    test_operation = input("Test Operation (e.g., matmul, conv): ")
    if test_operation:
        params['test_operation'] = test_operation
    
    return params


def interactive_api_test() -> Dict[str, Any]:
    """Interactive prompts for API test parameters."""
    params = {}
    
    print("\n=== API Test Generator ===\n")
    
    params['api_name'] = input("API Name (e.g., openai, huggingface-tgi): ")
    
    api_type = ""
    valid_types = ['openai', 'hf_tei', 'hf_tgi', 'ollama', 'vllm', 'claude', 'internal']
    while api_type not in valid_types:
        api_type = input("API Type [openai/hf_tei/hf_tgi/ollama/vllm/claude/internal]: ").lower()
        if api_type not in valid_types:
            print(f"Invalid API type. Please choose from {', '.join(valid_types)}.")
    
    params['api_type'] = api_type
    
    test_name = input("Test Name (default: auto-generate from API name): ")
    if not test_name:
        test_name = f"{params['api_name'].replace('-', '_').lower()}_api"
    params['test_name'] = test_name
    
    return params


def generate_test(test_type: str, output_dir: str):
    """
    Generate a test file interactively.
    
    Args:
        test_type: Type of test to generate
        output_dir: Output directory for the test file
    """
    if test_type == 'model':
        params = interactive_model_test()
        template = ModelTestTemplate(
            template_name=f"{params['model_name']}_test",
            output_dir=output_dir,
            parameters=params
        )
    elif test_type == 'hardware':
        params = interactive_hardware_test()
        template = HardwareTestTemplate(
            template_name=f"{params['hardware_platform']}_test",
            output_dir=output_dir,
            parameters=params
        )
    elif test_type == 'api':
        params = interactive_api_test()
        template = APITestTemplate(
            template_name=f"{params['api_name']}_test",
            output_dir=output_dir,
            parameters=params
        )
    else:
        print(f"Unknown test type: {test_type}")
        return
    
    # Write the test file
    try:
        file_path = template.write()
        print(f"\nTest file generated successfully: {file_path}")
    except Exception as e:
        print(f"\nError generating test file: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate a test file interactively")
    parser.add_argument('--type', choices=['model', 'hardware', 'api'], default='model',
                        help="Type of test to generate")
    parser.add_argument('--output-dir', type=str, default=os.path.dirname(os.path.abspath(__file__)),
                        help="Directory to write the test file")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
    
    # Generate the test file
    generate_test(args.type, args.output_dir)


if __name__ == "__main__":
    main()