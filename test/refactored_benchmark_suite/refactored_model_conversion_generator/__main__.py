"""
Command-line interface for model conversion.
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from .core import ModelConverterRegistry
from .utils import HardwareDetector, ModelFileManager, setup_logger, ModelVerifier

def main():
    """Command-line entry point for model conversion generator."""
    parser = argparse.ArgumentParser(description="Model Conversion Generator")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert a model")
    convert_parser.add_argument("--source", required=True, help="Path to source model")
    convert_parser.add_argument("--output", help="Path to save converted model")
    convert_parser.add_argument("--source-format", required=True, help="Source model format (e.g., onnx, pytorch)")
    convert_parser.add_argument("--target-format", required=True, help="Target model format (e.g., onnx, openvino)")
    convert_parser.add_argument("--model-type", help="Type of model (e.g., bert, vit)")
    convert_parser.add_argument("--cache-dir", help="Directory to store cached models")
    convert_parser.add_argument("--log-file", help="Path to log file")
    convert_parser.add_argument("--log-level", default="INFO", help="Logging level")
    convert_parser.add_argument("--force", action="store_true", help="Force conversion even if output exists")
    convert_parser.add_argument("--precision", default="default", 
                        choices=["default", "float16", "int8", "8bit", "4bit", "3bit", "2bit"],
                        help="Model precision level (lower values = smaller size but less accuracy)")
    convert_parser.add_argument("--mixed-precision-config", help="Path to mixed precision configuration JSON file")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available converters")
    list_parser.add_argument("--format", default="table", choices=["table", "json"], help="Output format")
    
    # Detect command
    detect_parser = subparsers.add_parser("detect", help="Detect available hardware")
    detect_parser.add_argument("--format", default="table", choices=["table", "json"], help="Output format")
    
    # Find command
    find_parser = subparsers.add_parser("find", help="Find model files")
    find_parser.add_argument("--directory", required=True, help="Directory to search in")
    find_parser.add_argument("--formats", help="Comma-separated list of formats to look for")
    find_parser.add_argument("--recursive", action="store_true", help="Search recursively")
    find_parser.add_argument("--format", default="table", choices=["table", "json"], help="Output format")
    
    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify model files")
    verify_parser.add_argument("--model", required=True, help="Path to model file")
    verify_parser.add_argument("--format", required=True, help="Model format")
    
    # Quantize command
    quantize_parser = subparsers.add_parser("quantize", help="Create quantization configuration")
    quantize_parser.add_argument("--model", required=True, help="Path to model file")
    quantize_parser.add_argument("--format", required=True, help="Model format (e.g., onnx)")
    quantize_parser.add_argument("--output", required=True, help="Path to save quantization config file")
    quantize_parser.add_argument("--precision", default="mixed", 
                               choices=["mixed", "4bit", "3bit", "2bit", "8bit", "int8"],
                               help="Target precision level or 'mixed' for optimal mixed precision")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up logging
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logger = setup_logger(level=log_level, log_file=getattr(args, "log_file", None))
    
    # Run command
    if args.command == "convert":
        run_convert_command(args, logger)
    elif args.command == "list":
        run_list_command(args)
    elif args.command == "detect":
        run_detect_command(args)
    elif args.command == "find":
        run_find_command(args)
    elif args.command == "verify":
        run_verify_command(args)
    elif args.command == "quantize":
        run_quantize_command(args, logger)
    else:
        parser.print_help()
        return 1
        
    return 0

def run_convert_command(args, logger):
    """Run convert command."""
    source_path = os.path.abspath(args.source)
    source_format = args.source_format.lower()
    target_format = args.target_format.lower()
    model_type = args.model_type
    cache_dir = args.cache_dir
    precision = args.precision
    
    # Determine output path
    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        # Use default output path
        model_name = os.path.splitext(os.path.basename(source_path))[0]
        output_dir = os.path.dirname(source_path)
        
        if target_format == 'onnx':
            ext = '.onnx'
        elif target_format == 'openvino':
            ext = '.xml'
        elif target_format == 'webnn':
            ext = '.js'
        elif target_format == 'webgpu':
            ext = '.js'
        else:
            ext = f'.{target_format}'
            
        output_path = os.path.join(output_dir, f"{model_name}_{target_format}{ext}")
    
    # Check if model is already cached
    if cache_dir and not args.force:
        is_cached, cache_path = ModelFileManager.check_cached_model(
            source_path, source_format, target_format, cache_dir
        )
        
        if is_cached:
            logger.info(f"Using cached model: {cache_path}")
            
            # Copy cached model to output path if different
            if cache_path != output_path:
                import shutil
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                shutil.copy2(cache_path, output_path)
                
                # Copy metadata if available
                metadata_path = cache_path + '.json'
                if os.path.exists(metadata_path):
                    shutil.copy2(metadata_path, output_path + '.json')
                    
                logger.info(f"Copied cached model to: {output_path}")
                
            return
    
    # Load mixed precision configuration if provided
    mixed_precision_config = None
    if args.mixed_precision_config:
        if os.path.exists(args.mixed_precision_config):
            try:
                with open(args.mixed_precision_config, 'r') as f:
                    mixed_precision_config = json.load(f)
                logger.info(f"Loaded mixed precision configuration from {args.mixed_precision_config}")
            except Exception as e:
                logger.warning(f"Error loading mixed precision configuration: {e}")
        else:
            logger.warning(f"Mixed precision configuration file not found: {args.mixed_precision_config}")
    
    # Get converter
    converter_class = ModelConverterRegistry.get_converter(source_format, target_format, model_type)
    
    if converter_class is None:
        logger.error(f"No converter found for {source_format} -> {target_format} ({model_type})")
        sys.exit(1)
        
    # Create converter
    converter = converter_class()
    
    # Convert model
    logger.info(f"Converting {source_format} model to {target_format}")
    logger.info(f"Source: {source_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Precision: {precision}")
    
    # Prepare additional conversion parameters
    conversion_params = {
        'model_path': source_path,
        'output_path': output_path,
        'model_type': model_type,
        'force': args.force,
        'precision': precision
    }
    
    # Add mixed precision configuration if available
    if mixed_precision_config:
        conversion_params['mixed_precision_config'] = mixed_precision_config
    
    # Perform conversion
    result = converter.convert(**conversion_params)
    
    if result.success:
        logger.info(f"Conversion successful: {result.output_path}")
        
        # Cache result if cache_dir specified
        if cache_dir:
            cache_path = ModelFileManager.get_cache_path(
                source_path, source_format, target_format, cache_dir
            )
            
            if cache_path != output_path:
                import shutil
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                shutil.copy2(output_path, cache_path)
                
                # Copy metadata if available
                metadata_path = output_path + '.json'
                if os.path.exists(metadata_path):
                    shutil.copy2(metadata_path, cache_path + '.json')
                    
                logger.info(f"Cached converted model to: {cache_path}")
    else:
        logger.error(f"Conversion failed: {result.error}")
        sys.exit(1)

def run_list_command(args):
    """Run list command."""
    # Get all registered converters
    converters = ModelConverterRegistry.list_converters()
    
    if args.format == "json":
        print(json.dumps(converters, indent=2))
    else:
        # Print as table
        print("Available Converters:")
        print("---------------------")
        print("Source Format | Target Format | Model Type | Converter Class")
        print("-------------|--------------|------------|----------------")
        
        for converter in converters:
            source_format = converter['source_format']
            target_format = converter['target_format']
            model_type = converter['model_type']
            converter_class = converter['converter_class']
            
            print(f"{source_format:<13} | {target_format:<14} | {model_type:<10} | {converter_class}")

def run_detect_command(args):
    """Run detect command."""
    # Detect available hardware
    hardware_info = HardwareDetector.get_available_hardware()
    
    if args.format == "json":
        print(json.dumps(hardware_info, indent=2))
    else:
        # Print as table
        print("Available Hardware:")
        print("------------------")
        print("Hardware | Available | Details")
        print("---------|-----------|--------")
        
        for name, info in hardware_info.items():
            available = info['available']
            details = ""
            
            if available:
                if name == 'cpu':
                    details = f"Cores: {info['info'].get('cores', 'unknown')}, Arch: {info['info'].get('architecture', 'unknown')}"
                elif name == 'cuda':
                    details = f"Devices: {info['info'].get('device_count', 0)}, Version: {info['info'].get('version', 'unknown')}"
                elif name == 'rocm':
                    details = f"Devices: {info['info'].get('device_count', 0)}"
                elif name == 'openvino':
                    devices = info['info'].get('devices', [])
                    details = f"Devices: {', '.join(devices)}"
                elif name == 'mps':
                    details = f"Architecture: {info['info'].get('machine', 'unknown')}"
                elif name == 'qnn':
                    details = f"Platform: {info['info'].get('platform', 'unknown')}"
                elif name == 'webgpu':
                    details = "Hardware capable of WebGPU"
                elif name == 'webnn':
                    details = "Hardware capable of WebNN"
            
            avail_str = "Yes" if available else "No"
            print(f"{name:<9} | {avail_str:<9} | {details}")

def run_find_command(args):
    """Run find command."""
    directory = os.path.abspath(args.directory)
    formats = args.formats.split(',') if args.formats else None
    recursive = args.recursive
    
    # Find models
    found_models = ModelFileManager.find_models(directory, formats, recursive)
    
    if args.format == "json":
        print(json.dumps(found_models, indent=2))
    else:
        # Print as table
        print(f"Models found in {directory}:")
        print("---------------------------")
        
        total_count = 0
        for fmt, model_paths in found_models.items():
            count = len(model_paths)
            total_count += count
            
            print(f"\n{fmt.upper()} Models ({count}):")
            print("-" * (len(fmt) + 10))
            
            for path in model_paths:
                rel_path = os.path.relpath(path, directory)
                print(f"  {rel_path}")
                
        print(f"\nTotal models found: {total_count}")

def run_verify_command(args):
    """Run verify command."""
    model_path = os.path.abspath(args.model)
    model_format = args.format.lower()
    
    # Verify model
    valid, error = ModelVerifier.verify_model(model_path, model_format)
    
    if valid:
        print(f"✅ Model {model_path} is valid")
    else:
        print(f"❌ Model {model_path} is invalid: {error}")
        sys.exit(1)

def run_quantize_command(args, logger):
    """Run quantize command to create a quantization configuration."""
    model_path = os.path.abspath(args.model)
    model_format = args.format.lower()
    output_path = os.path.abspath(args.output)
    precision = args.precision
    
    # Verify model first
    valid, error = ModelVerifier.verify_model(model_path, model_format)
    if not valid:
        logger.error(f"Invalid model: {error}")
        sys.exit(1)
        
    logger.info(f"Creating quantization configuration for {model_path}")
    
    try:
        # Load model to analyze structure
        model_info = None
        if model_format == 'onnx':
            import onnx
            model = onnx.load(model_path)
            model_info = {
                'num_layers': len(model.graph.node),
                'inputs': [input.name for input in model.graph.input],
                'outputs': [output.name for output in model.graph.output]
            }
            
            # Try to determine model type from ops
            model_type = "unknown"
            ops = [node.op_type for node in model.graph.node]
            op_counts = {}
            for op in ops:
                op_counts[op] = op_counts.get(op, 0) + 1
                
            # Heuristic for model type
            if "MatMul" in op_counts and "Attention" in op_counts:
                model_type = "transformer"
            elif "Conv" in op_counts and op_counts.get("Conv", 0) > 5:
                model_type = "vision"
                
            model_info['model_type'] = model_type
        
        # Create configuration based on precision and model info
        if precision == "mixed":
            # Create a mixed precision configuration
            config = generate_mixed_precision_config(model_info)
        else:
            # Create uniform precision configuration
            config = generate_uniform_precision_config(precision, model_info)
            
        # Save configuration
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        logger.info(f"Quantization configuration saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error creating quantization configuration: {e}")
        sys.exit(1)

def generate_mixed_precision_config(model_info):
    """Generate an optimal mixed precision configuration."""
    if not model_info:
        # Default configuration if no model info available
        return {
            "mixed_precision": {
                "2bit_layers": [],
                "3bit_layers": [],
                "4bit_layers": [0, 1, 2, 3, 4, 5, 6, 7],
                "8bit_layers": [8, 9, 10, 11],
                "fp16_layers": []
            }
        }
        
    # Get number of layers
    num_layers = model_info.get('num_layers', 12)
    model_type = model_info.get('model_type', 'unknown')
    
    # Create a mixed precision configuration tailored to model type
    if model_type == 'transformer':
        # For transformer models, use higher precision for early and final layers
        return {
            "model_type": model_type,
            "mixed_precision": {
                # 2-bit for most feed-forward layers (highest compression)
                "2bit_layers": list(range(2, num_layers - 4)),
                # 3-bit for middle attention layers
                "3bit_layers": [],
                # 4-bit for some key layers
                "4bit_layers": [num_layers - 4, num_layers - 3, num_layers - 2],
                # 8-bit for first and last layers (better accuracy)
                "8bit_layers": [0, 1, num_layers - 1],
                # fp16 not used by default
                "fp16_layers": []
            }
        }
    elif model_type == 'vision':
        # For vision models, use higher precision for early layers
        return {
            "model_type": model_type,
            "mixed_precision": {
                # 2-bit for later layers 
                "2bit_layers": list(range(num_layers // 2, num_layers - 1)),
                # 3-bit generally not used
                "3bit_layers": [],
                # 4-bit for middle layers
                "4bit_layers": list(range(2, num_layers // 2)),
                # 8-bit for first layers and final classifier
                "8bit_layers": [0, 1, num_layers - 1],
                # fp16 not used by default
                "fp16_layers": []
            }
        }
    else:
        # Generic configuration
        return {
            "model_type": model_type,
            "mixed_precision": {
                "2bit_layers": [],  # Conservative approach: no 2-bit
                "3bit_layers": [],  # Conservative approach: no 3-bit
                "4bit_layers": list(range(1, num_layers - 1)),  # Most layers use 4-bit
                "8bit_layers": [0, num_layers - 1],  # First and last layer use 8-bit
                "fp16_layers": []  # No fp16 by default
            }
        }

def generate_uniform_precision_config(precision, model_info):
    """Generate a uniform precision configuration."""
    if not model_info:
        # Default configuration if no model info available
        return {
            "uniform_precision": precision,
            "mixed_precision": {
                "2bit_layers": [],
                "3bit_layers": [],
                "4bit_layers": [],
                "8bit_layers": [],
                "fp16_layers": []
            }
        }
        
    # Get number of layers
    num_layers = model_info.get('num_layers', 12)
    model_type = model_info.get('model_type', 'unknown')
    
    # Fill the appropriate precision array with all layers
    config = {
        "model_type": model_type,
        "uniform_precision": precision,
        "mixed_precision": {
            "2bit_layers": [],
            "3bit_layers": [],
            "4bit_layers": [],
            "8bit_layers": [],
            "fp16_layers": []
        }
    }
    
    # Add all layers to the appropriate precision array
    if precision == "2bit":
        config["mixed_precision"]["2bit_layers"] = list(range(num_layers))
    elif precision == "3bit":
        config["mixed_precision"]["3bit_layers"] = list(range(num_layers))
    elif precision == "4bit":
        config["mixed_precision"]["4bit_layers"] = list(range(num_layers))
    elif precision in ["8bit", "int8"]:
        config["mixed_precision"]["8bit_layers"] = list(range(num_layers))
        
    return config

if __name__ == "__main__":
    sys.exit(main())