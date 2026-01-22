#!/usr/bin/env python3
"""
Test generator script for IPFS Accelerate.

This script generates test files using the template system.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import templates
from template_system.templates.model_test_template import ModelTestTemplate
from template_system.templates.hardware_test_template import HardwareTestTemplate
from template_system.templates.api_test_template import APITestTemplate


def generate_model_test(args: Dict[str, Any]) -> str:
    """
    Generate a model test file.
    
    Args:
        args: Template arguments
        
    Returns:
        Path to the generated file
    """
    required_args = ['model_name', 'model_type']
    for arg in required_args:
        if arg not in args:
            raise ValueError(f"Missing required argument: {arg}")
    
    # Create the template
    template = ModelTestTemplate(
        model_name=args['model_name'],
        model_type=args['model_type'],
        **{k: v for k, v in args.items() if k not in ['model_name', 'model_type']}
    )
    
    # Generate the test file
    output_path = template.generate()
    
    logger.info(f"Generated model test: {output_path}")
    
    return output_path


def generate_hardware_test(args: Dict[str, Any]) -> str:
    """
    Generate a hardware test file.
    
    Args:
        args: Template arguments
        
    Returns:
        Path to the generated file
    """
    required_args = ['hardware_platform', 'test_name']
    for arg in required_args:
        if arg not in args:
            raise ValueError(f"Missing required argument: {arg}")
    
    # Create the template
    template = HardwareTestTemplate(
        parameters=args,
        output_dir=args.get('output_dir', 'test')
    )
    
    # Generate the test file
    output_path = template.write()
    
    logger.info(f"Generated hardware test: {output_path}")
    
    return output_path


def generate_api_test(args: Dict[str, Any]) -> str:
    """
    Generate an API test file.
    
    Args:
        args: Template arguments
        
    Returns:
        Path to the generated file
    """
    required_args = ['api_name', 'test_name']
    for arg in required_args:
        if arg not in args:
            raise ValueError(f"Missing required argument: {arg}")
    
    # Create the template
    template = APITestTemplate(
        parameters=args,
        output_dir=args.get('output_dir', 'test')
    )
    
    # Generate the test file
    output_path = template.write()
    
    logger.info(f"Generated API test: {output_path}")
    
    return output_path


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Generate test files for IPFS Accelerate')
    
    # Common arguments
    parser.add_argument('--output-dir', help='Output directory for generated files')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files')
    
    # Test type subparsers
    subparsers = parser.add_subparsers(dest='test_type', required=True, help='Type of test to generate')
    
    # Model test arguments
    model_parser = subparsers.add_parser('model', help='Generate a model test')
    model_parser.add_argument('--model-name', required=True, help='Name of the model (e.g., bert-base-uncased)')
    model_parser.add_argument('--model-type', required=True, choices=['text', 'vision', 'audio', 'multimodal'], 
                             help='Type of model')
    model_parser.add_argument('--framework', default='transformers', 
                             choices=['transformers', 'torch', 'tensorflow', 'onnx'],
                             help='Framework used for the model')
    model_parser.add_argument('--batch-size', type=int, default=1, help='Batch size for testing')
    
    # Hardware test arguments
    hw_parser = subparsers.add_parser('hardware', help='Generate a hardware test')
    hw_parser.add_argument('--hardware-platform', required=True, 
                          choices=['webgpu', 'webnn', 'cuda', 'rocm', 'cpu'],
                          help='Hardware platform to test')
    hw_parser.add_argument('--test-name', required=True, help='Name for the test')
    hw_parser.add_argument('--test-operation', default='matmul', 
                          choices=['matmul', 'conv', 'inference'],
                          help='Operation to test')
    hw_parser.add_argument('--test-category', default='compute', 
                          choices=['compute', 'memory', 'throughput', 'latency'],
                          help='Category of test')
    
    # API test arguments
    api_parser = subparsers.add_parser('api', help='Generate an API test')
    api_parser.add_argument('--api-name', required=True, help='Name of the API')
    api_parser.add_argument('--test-name', required=True, help='Name for the test')
    api_parser.add_argument('--api-type', default='internal', 
                           choices=['openai', 'hf_tei', 'hf_tgi', 'ollama', 'vllm', 'claude', 'internal'],
                           help='Type of API')
    
    return parser.parse_args()


def main() -> None:
    """
    Main function to generate test files.
    """
    args = parse_arguments()
    
    # Extract common arguments
    common_args = {
        'output_dir': args.output_dir,
        'overwrite': args.overwrite
    }
    
    # Clean up None values
    common_args = {k: v for k, v in common_args.items() if v is not None}
    
    # Generate the test file based on type
    try:
        if args.test_type == 'model':
            model_args = {
                'model_name': args.model_name,
                'model_type': args.model_type,
                'framework': args.framework,
                'batch_size': args.batch_size,
                **common_args
            }
            output_path = generate_model_test(model_args)
        elif args.test_type == 'hardware':
            hw_args = {
                'hardware_platform': args.hardware_platform,
                'test_name': args.test_name,
                'test_operation': args.test_operation,
                'test_category': args.test_category,
                **common_args
            }
            output_path = generate_hardware_test(hw_args)
        elif args.test_type == 'api':
            api_args = {
                'api_name': args.api_name,
                'test_name': args.test_name,
                'api_type': args.api_type,
                **common_args
            }
            output_path = generate_api_test(api_args)
        else:
            logger.error(f"Unsupported test type: {args.test_type}")
            return
        
        logger.info(f"Generated test file: {output_path}")
    except Exception as e:
        logger.error(f"Error generating test file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()