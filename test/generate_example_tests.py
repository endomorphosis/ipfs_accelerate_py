#!/usr/bin/env python3
"""
Example test generation script for IPFS Accelerate.

This script generates example tests for different model types, hardware platforms, and APIs.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import templates
from test.template_system.templates.model_test_template import ModelTestTemplate
from test.template_system.templates.hardware_test_template import HardwareTestTemplate
from test.template_system.templates.api_test_template import APITestTemplate


def generate_model_tests(output_dir: str) -> List[str]:
    """
    Generate example model tests.
    
    Args:
        output_dir: Output directory for generated tests
        
    Returns:
        List of generated file paths
    """
    # Define test cases
    model_tests = [
        # Text models
        {'model_name': 'bert-base-uncased', 'model_type': 'text', 'framework': 'transformers'},
        {'model_name': 't5-small', 'model_type': 'text', 'framework': 'transformers'},
        {'model_name': 'gpt2', 'model_type': 'text', 'framework': 'transformers'},
        
        # Vision models
        {'model_name': 'google/vit-base-patch16-224', 'model_type': 'vision', 'framework': 'transformers'},
        {'model_name': 'facebook/detr-resnet-50', 'model_type': 'vision', 'framework': 'transformers'},
        
        # Audio models
        {'model_name': 'openai/whisper-tiny', 'model_type': 'audio', 'framework': 'transformers'},
        {'model_name': 'facebook/wav2vec2-base', 'model_type': 'audio', 'framework': 'transformers'},
        
        # Multimodal models
        {'model_name': 'openai/clip-vit-base-patch32', 'model_type': 'multimodal', 'framework': 'transformers'},
    ]
    
    # Generate tests
    generated_files = []
    
    for test_params in model_tests:
        # Create output directory
        model_dir = os.path.join(output_dir, 'models', test_params['model_type'], test_params['model_name'].split('/')[-1])
        os.makedirs(model_dir, exist_ok=True)
        
        # Create the template
        template = ModelTestTemplate(
            model_name=test_params['model_name'],
            model_type=test_params['model_type'],
            framework=test_params.get('framework', 'transformers'),
            output_dir=model_dir,
            overwrite=True
        )
        
        # Generate the test file
        output_path = template.generate()
        generated_files.append(output_path)
        
        logger.info(f"Generated model test: {output_path}")
    
    return generated_files


def generate_hardware_tests(output_dir: str) -> List[str]:
    """
    Generate example hardware tests.
    
    Args:
        output_dir: Output directory for generated tests
        
    Returns:
        List of generated file paths
    """
    # Define test cases
    hardware_tests = [
        # WebGPU tests
        {'hardware_platform': 'webgpu', 'test_name': 'matmul_performance', 'test_operation': 'matmul', 'test_category': 'compute'},
        {'hardware_platform': 'webgpu', 'test_name': 'conv_performance', 'test_operation': 'conv', 'test_category': 'compute'},
        {'hardware_platform': 'webgpu', 'test_name': 'memory_transfer', 'test_operation': 'memory', 'test_category': 'memory'},
        
        # WebNN tests
        {'hardware_platform': 'webnn', 'test_name': 'graph_execution', 'test_operation': 'inference', 'test_category': 'compute'},
        {'hardware_platform': 'webnn', 'test_name': 'operator_fusion', 'test_operation': 'inference', 'test_category': 'optimization'},
        
        # CUDA tests
        {'hardware_platform': 'cuda', 'test_name': 'tensor_operations', 'test_operation': 'matmul', 'test_category': 'compute'},
        {'hardware_platform': 'cuda', 'test_name': 'memory_bandwidth', 'test_operation': 'memory', 'test_category': 'memory'},
        
        # ROCm tests
        {'hardware_platform': 'rocm', 'test_name': 'tensor_performance', 'test_operation': 'matmul', 'test_category': 'compute'},
        
        # MPS tests
        {'hardware_platform': 'mps', 'test_name': 'apple_silicon_performance', 'test_operation': 'inference', 'test_category': 'compute'},
    ]
    
    # Generate tests
    generated_files = []
    
    for test_params in hardware_tests:
        # Create output directory
        hardware_dir = os.path.join(output_dir, 'hardware', test_params['hardware_platform'], test_params['test_category'])
        os.makedirs(hardware_dir, exist_ok=True)
        
        # Create the template
        template = HardwareTestTemplate(
            parameters=test_params,
            output_dir=hardware_dir
        )
        
        # Generate the test file
        output_path = template.write()
        generated_files.append(output_path)
        
        logger.info(f"Generated hardware test: {output_path}")
    
    return generated_files


def generate_api_tests(output_dir: str) -> List[str]:
    """
    Generate example API tests.
    
    Args:
        output_dir: Output directory for generated tests
        
    Returns:
        List of generated file paths
    """
    # Define test cases
    api_tests = [
        # OpenAI tests
        {'api_name': 'openai', 'test_name': 'chat_completion', 'api_type': 'openai'},
        {'api_name': 'openai', 'test_name': 'embedding', 'api_type': 'openai'},
        
        # HuggingFace tests
        {'api_name': 'hf_tei', 'test_name': 'text_embedding', 'api_type': 'hf_tei'},
        {'api_name': 'hf_tgi', 'test_name': 'text_generation', 'api_type': 'hf_tgi'},
        
        # Ollama tests
        {'api_name': 'ollama', 'test_name': 'llm_inference', 'api_type': 'ollama'},
        
        # vLLM tests
        {'api_name': 'vllm', 'test_name': 'batch_inference', 'api_type': 'vllm'},
        
        # Claude tests
        {'api_name': 'claude', 'test_name': 'claude_chat', 'api_type': 'claude'},
    ]
    
    # Generate tests
    generated_files = []
    
    for test_params in api_tests:
        # Create output directory
        api_dir = os.path.join(output_dir, 'api', test_params['api_name'])
        os.makedirs(api_dir, exist_ok=True)
        
        # Create the template
        template = APITestTemplate(
            parameters=test_params,
            output_dir=api_dir
        )
        
        # Generate the test file
        output_path = template.write()
        generated_files.append(output_path)
        
        logger.info(f"Generated API test: {output_path}")
    
    return generated_files


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Generate example tests for IPFS Accelerate')
    parser.add_argument('--output-dir', default='.', help='Output directory for generated tests')
    parser.add_argument('--types', choices=['model', 'hardware', 'api', 'all'], default='all', help='Types of tests to generate')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    generated_files = []
    
    # Generate tests
    if args.types in ['model', 'all']:
        logger.info("Generating model tests...")
        generated_files.extend(generate_model_tests(args.output_dir))
    
    if args.types in ['hardware', 'all']:
        logger.info("Generating hardware tests...")
        generated_files.extend(generate_hardware_tests(args.output_dir))
    
    if args.types in ['api', 'all']:
        logger.info("Generating API tests...")
        generated_files.extend(generate_api_tests(args.output_dir))
    
    logger.info(f"Generated {len(generated_files)} test files")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())