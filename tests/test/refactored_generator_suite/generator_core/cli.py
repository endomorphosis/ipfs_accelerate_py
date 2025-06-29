#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Command-line interface for the refactored generator suite.
Provides command-line access to generator functionality.
"""

import os
import sys
import json
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

from .config import ConfigManager, get_config
from .registry import ComponentRegistry
from .generator import GeneratorCore


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration.
    
    Args:
        verbose: Whether to use verbose logging.
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=log_level, format=log_format)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="HuggingFace Model Test Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Basic arguments
    parser.add_argument("--model", "-m", dest="model_type",
                        help="Model type to generate a test for (bert, gpt2, t5, etc.)")
    parser.add_argument("--output", "-o", dest="output_file",
                        help="Output file path")
    parser.add_argument("--config", "-c", dest="config_file",
                        help="Configuration file path")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")
    
    # Generator options
    parser.add_argument("--output-dir", dest="output_dir",
                        help="Output directory for generated files")
    parser.add_argument("--task", help="Task type (fill-mask, text-generation, etc.)")
    parser.add_argument("--fix-syntax", dest="fix_syntax", action="store_true",
                        help="Fix syntax errors in generated files")
    parser.add_argument("--no-fix-syntax", dest="fix_syntax", action="store_false",
                        help="Don't fix syntax errors in generated files")
    parser.set_defaults(fix_syntax=None)  # Use default from config
    
    # Model options
    parser.add_argument("--model-name", dest="model_name",
                        help="Specific model name/ID to use")
    parser.add_argument("--max-size", dest="max_size", type=int,
                        help="Maximum model size in MB")
    parser.add_argument("--framework", help="Framework (pt, tf, onnx, etc.)")
    
    # Batch generation
    parser.add_argument("--batch", action="store_true",
                        help="Enable batch generation mode")
    parser.add_argument("--batch-file", dest="batch_file",
                        help="File containing list of models to generate")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=10,
                        help="Number of models to generate in a batch")
    parser.add_argument("--start-index", dest="start_index", type=int, default=0,
                        help="Starting index for batch generation")
    parser.add_argument("--priority", choices=["high", "medium", "low"],
                        help="Priority level for batch generation")
    parser.add_argument("--architecture", choices=["encoder-only", "decoder-only", 
                                                  "encoder-decoder", "vision", 
                                                  "vision-text", "speech"],
                        help="Architecture for batch generation")
    
    # Output options
    parser.add_argument("--json", action="store_true",
                        help="Output results in JSON format")
    parser.add_argument("--report", action="store_true",
                        help="Generate a summary report")
    parser.add_argument("--report-file", dest="report_file",
                        help="Report file path")
    
    return parser.parse_args()


def load_batch_models(batch_file: Optional[str] = None, 
                      architecture: Optional[str] = None,
                      registry: ComponentRegistry = None,
                      priority: Optional[str] = None,
                      batch_size: int = 10,
                      start_index: int = 0) -> List[str]:
    """Load models for batch generation.
    
    Args:
        batch_file: Optional file containing list of models.
        architecture: Optional architecture to filter by.
        registry: Optional component registry for model info.
        priority: Optional priority level.
        batch_size: Number of models to include in the batch.
        start_index: Starting index for the batch.
        
    Returns:
        List of model types for batch generation.
    """
    models = []
    
    # If batch file is provided, load models from it
    if batch_file:
        with open(batch_file, 'r') as f:
            if batch_file.endswith('.json'):
                batch_data = json.load(f)
                if isinstance(batch_data, list):
                    models = batch_data
                elif isinstance(batch_data, dict) and 'models' in batch_data:
                    models = batch_data['models']
            else:
                # Assume one model per line
                models = [line.strip() for line in f if line.strip()]
    
    # If architecture is provided and registry is available, get models for that architecture
    elif architecture and registry:
        models = registry.get_models_by_architecture(architecture)
    
    # Otherwise, get all known models from registry
    elif registry:
        models = registry.get_model_ids()
    
    # If priority filtering is needed, apply it here
    # This is a placeholder - real implementation would require priority data
    if priority and registry:
        # In a real implementation, we would filter by priority
        pass
    
    # Apply batch slicing
    end_index = start_index + batch_size if batch_size > 0 else len(models)
    return models[start_index:end_index]


def create_options_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Create options dictionary from command-line arguments.
    
    Args:
        args: Parsed command-line arguments.
        
    Returns:
        Dictionary of generator options.
    """
    options = {}
    
    # Output options
    if args.output_file:
        options["output_file"] = args.output_file
    if args.output_dir:
        options["output_dir"] = args.output_dir
    
    # Model options
    if args.task:
        options["task"] = args.task
    if args.model_name:
        # Create a minimal model_info dict
        options["model_info"] = {"id": args.model_name, "name": args.model_name}
    if args.max_size:
        options["max_size"] = args.max_size
    if args.framework:
        options["framework"] = args.framework
    
    # Syntax options
    if args.fix_syntax is not None:  # Only if explicitly set
        options["fix_syntax"] = args.fix_syntax
    
    return options


def generate_model(generator: GeneratorCore, model_type: str, options: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a test for a single model.
    
    Args:
        generator: Generator instance.
        model_type: Model type to generate.
        options: Generator options.
        
    Returns:
        Generation result.
    """
    result = generator.generate(model_type, options)
    return result


def generate_batch(generator: GeneratorCore, models: List[str], options: Dict[str, Any]) -> Dict[str, Any]:
    """Generate tests for multiple models.
    
    Args:
        generator: Generator instance.
        models: List of model types to generate.
        options: Generator options.
        
    Returns:
        Batch generation result.
    """
    result = generator.generate_batch(models, options)
    return result


def print_result(result: Dict[str, Any], json_output: bool = False) -> None:
    """Print the generation result.
    
    Args:
        result: Generation result.
        json_output: Whether to output in JSON format.
    """
    if json_output:
        print(json.dumps(result, indent=2))
        return
    
    if result.get("success", False):
        print(f"Generation successful!")
        if "output_file" in result:
            print(f"Output written to: {result['output_file']}")
        print(f"Model type: {result.get('model_type', 'unknown')}")
        print(f"Architecture: {result.get('architecture', 'unknown')}")
        print(f"Duration: {result.get('duration', 0):.2f} seconds")
    else:
        print(f"Generation failed: {result.get('error', 'Unknown error')}")


def print_batch_result(result: Dict[str, Any], json_output: bool = False) -> None:
    """Print the batch generation result.
    
    Args:
        result: Batch generation result.
        json_output: Whether to output in JSON format.
    """
    if json_output:
        print(json.dumps(result, indent=2))
        return
    
    if result.get("success", False):
        print(f"Batch generation successful!")
    else:
        print(f"Batch generation completed with errors.")
    
    print(f"Total: {result.get('total_count', 0)}")
    print(f"Success: {result.get('success_count', 0)}")
    print(f"Errors: {result.get('error_count', 0)}")
    print(f"Duration: {result.get('duration', 0):.2f} seconds")
    
    if result.get("error_count", 0) > 0:
        print("\nFailed models:")
        for item in result.get("results", []):
            if not item.get("success", False):
                print(f"  - {item.get('model_type', 'unknown')}: {item.get('error', 'Unknown error')}")


def generate_report(result: Dict[str, Any], report_file: Optional[str] = None) -> None:
    """Generate a summary report.
    
    Args:
        result: Generation result.
        report_file: Optional file to write the report to.
    """
    # Create report content
    if "results" in result:  # Batch result
        total = result.get("total_count", 0)
        success = result.get("success_count", 0)
        errors = result.get("error_count", 0)
        duration = result.get("duration", 0)
        
        report = f"""# Batch Generation Report

## Summary
- Total: {total}
- Success: {success}
- Errors: {errors}
- Duration: {duration:.2f} seconds
- Success Rate: {(success / total * 100) if total > 0 else 0:.1f}%

## Details
"""
        
        # Add successful models
        report += "\n### Successful Models\n\n"
        for item in result.get("results", []):
            if item.get("success", False):
                model_type = item.get("model_type", "unknown")
                architecture = item.get("architecture", "unknown")
                output_file = item.get("output_file", "unknown")
                report += f"- **{model_type}** ({architecture}): {output_file}\n"
        
        # Add failed models
        report += "\n### Failed Models\n\n"
        for item in result.get("results", []):
            if not item.get("success", False):
                model_type = item.get("model_type", "unknown")
                error = item.get("error", "Unknown error")
                report += f"- **{model_type}**: {error}\n"
    
    else:  # Single model result
        success = result.get("success", False)
        model_type = result.get("model_type", "unknown")
        architecture = result.get("architecture", "unknown")
        duration = result.get("duration", 0)
        
        report = f"""# Generation Report for {model_type}

## Summary
- Model: {model_type}
- Architecture: {architecture}
- Success: {"Yes" if success else "No"}
- Duration: {duration:.2f} seconds

## Details
"""
        
        if success:
            output_file = result.get("output_file", "unknown")
            report += f"- Output file: {output_file}\n"
        else:
            error = result.get("error", "Unknown error")
            report += f"- Error: {error}\n"
    
    # Write report to file or print to stdout
    if report_file:
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"Report written to: {report_file}")
    else:
        print("\nReport:")
        print(report)


def main() -> int:
    """Main entry point.
    
    Returns:
        Exit code (0 for success, non-zero for error).
    """
    # Parse command-line arguments
    args = parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = ConfigManager(args.config_file)
    
    # Create component registry
    registry = ComponentRegistry()
    
    # Load templates
    registry.discover_templates()
    
    # Create options from arguments
    options = create_options_from_args(args)
    
    # Create generator
    generator = GeneratorCore(config, registry)
    
    # Generate tests
    if args.batch or args.batch_file:
        # Batch generation mode
        models = load_batch_models(
            batch_file=args.batch_file,
            architecture=args.architecture,
            registry=registry,
            priority=args.priority,
            batch_size=args.batch_size,
            start_index=args.start_index
        )
        
        if not models:
            logger.error("No models found for batch generation.")
            return 1
        
        logger.info(f"Generating tests for {len(models)} models...")
        result = generate_batch(generator, models, options)
        print_batch_result(result, args.json)
        
        if args.report or args.report_file:
            generate_report(result, args.report_file)
        
        return 0 if result.get("success", False) else 1
    
    elif args.model_type:
        # Single model generation mode
        logger.info(f"Generating test for model: {args.model_type}")
        result = generate_model(generator, args.model_type, options)
        print_result(result, args.json)
        
        if args.report or args.report_file:
            generate_report(result, args.report_file)
        
        return 0 if result.get("success", False) else 1
    
    else:
        # No model specified
        logger.error("No model type specified. Use --model or --batch.")
        return 1


if __name__ == "__main__":
    sys.exit(main())