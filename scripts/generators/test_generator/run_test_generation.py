#!/usr/bin/env python3
"""
Run test generation for HuggingFace models.

This script provides a command-line interface to generate test files for models
based on their architecture and priority.
"""

import os
import sys
import argparse
import logging
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"generate_tests_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required modules
try:
    from refactored_test_suite.generators.test_generator import ModelTestGenerator, PRIORITY_MODELS
except ImportError:
    logger.error("Failed to import required modules")
    sys.exit(1)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate HuggingFace model tests")
    
    # Model specification options
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--model",
        help="Specific model name to generate test for"
    )
    model_group.add_argument(
        "--priority",
        choices=["high", "medium", "low", "all"],
        default="high",
        help="Generate tests for models with this priority"
    )
    model_group.add_argument(
        "--architectures",
        nargs="+",
        choices=["encoder-only", "decoder-only", "encoder-decoder", "vision", 
                "vision-encoder-text-decoder", "speech", "multimodal", 
                "diffusion", "mixture-of-experts", "state-space", "rag", "all"],
        help="Generate tests for models of these architectures"
    )
    
    # Hardware backends
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "rocm", "mps", "openvino", "qnn", "auto"],
        default="auto",
        help="Target hardware device for tests (auto=use optimal available)"
    )
    
    # Hardware detection options
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List all available hardware devices and exit"
    )
    
    parser.add_argument(
        "--hardware-compatibility",
        action="store_true",
        help="Show hardware compatibility matrix for model architectures and exit"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir",
        default="./generated_tests",
        help="Directory to save generated files"
    )
    
    parser.add_argument(
        "--template-dir",
        help="Directory containing templates (default: use built-in templates)"
    )
    
    parser.add_argument(
        "--report-dir",
        default="./reports",
        help="Directory to save reports"
    )
    
    # Behavior options
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite of existing files"
    )
    
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify generated files"
    )
    
    parser.add_argument(
        "--coverage-report",
        action="store_true",
        help="Generate coverage report after generation"
    )
    
    return parser.parse_args()

def run_generation(args):
    """
    Run test generation based on arguments.
    
    Args:
        args: Command-line arguments
    """
    # Import hardware detection module for device listing and compatibility
    try:
        from refactored_test_suite.hardware.hardware_detection import (
            print_device_summary, detect_available_hardware, get_optimal_device,
            get_model_hardware_recommendations, is_device_compatible_with_model,
            SUPPORTED_BACKENDS
        )
        has_hardware_detection = True
    except ImportError:
        logger.warning("Hardware detection module not available")
        has_hardware_detection = False
        # Define fallback SUPPORTED_BACKENDS
        SUPPORTED_BACKENDS = ["cpu", "cuda", "rocm", "mps", "openvino", "qnn"]
    
    # Handle special hardware-related options
    if args.list_devices and has_hardware_detection:
        print_device_summary()
        return True
    
    if args.hardware_compatibility and has_hardware_detection:
        # Print hardware compatibility matrix for model architectures
        architectures = [
            "encoder-only", "decoder-only", "encoder-decoder", "vision",
            "vision-encoder-text-decoder", "speech", "multimodal",
            "diffusion", "mixture-of-experts", "state-space", "rag"
        ]
        
        hardware = detect_available_hardware()
        
        print("\n=== Hardware Compatibility Matrix ===")
        print("✅ = Compatible, ⚠️ = Limited compatibility, ❌ = Not recommended")
        print("\n| Architecture | " + " | ".join(hw.upper() for hw in SUPPORTED_BACKENDS) + " |")
        print("|" + "-"*13 + "|" + "".join("-"*11 + "|" for _ in SUPPORTED_BACKENDS))
        
        for arch in architectures:
            # Get recommendation for this architecture
            recommendations = get_model_hardware_recommendations(arch)
            
            row = f"| {arch.ljust(11)} |"
            for hw in SUPPORTED_BACKENDS:
                compat = is_device_compatible_with_model(hw, arch)
                avail = hardware.get(hw, False)
                
                if not avail:
                    # Device not available
                    symbol = "N/A"
                elif compat and hw in recommendations[:2]:
                    # Highly compatible and available
                    symbol = "✅"
                elif compat:
                    # Compatible but not optimal
                    symbol = "⚠️"
                else:
                    # Not compatible
                    symbol = "❌"
                
                row += f" {symbol.center(9)} |"
            
            print(row)
        
        print("\nNotes:")
        print("- CPU is always available but may be slow for large models")
        print("- CUDA and ROCm provide best performance for most model types")
        print("- OpenVINO works best with vision and audio models")
        print("- QNN (Qualcomm) is optimized for mobile/edge vision and audio models")
        print("- State Space and MoE models have specific hardware requirements")
        print("=======================================\n")
        
        return True
    
    # Create output and report directories if they don't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)
    
    # Process device argument
    selected_device = None
    if args.device:
        if args.device == "auto" and has_hardware_detection:
            # Use optimal device
            selected_device = get_optimal_device()
            logger.info(f"Auto-selected optimal device: {selected_device}")
        elif args.device != "auto":
            selected_device = args.device
            logger.info(f"Using specified device: {selected_device}")
    
    # Create generator
    generator = ModelTestGenerator(
        output_dir=args.output_dir,
        template_dir=args.template_dir
    )
    
    # Generate tests based on arguments
    if args.model:
        # Generate for specific model
        logger.info(f"Generating test for model: {args.model}")
        success, file_path = generator.generate_test_file(
            args.model,
            force=args.force,
            verify=args.verify
        )
        
        if success:
            logger.info(f"✅ Successfully generated test: {file_path}")
            
            # If device is specified, update the template to include the device
            if selected_device:
                try:
                    with open(file_path, "r") as f:
                        content = f.read()
                    
                    # Update the device choice in the generated file
                    updated_content = content.replace(
                        'parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"]',
                        f'parser.add_argument("--device", type=str, choices=["cpu", "cuda", "rocm", "mps", "openvino", "qnn"]'
                    )
                    
                    with open(file_path, "w") as f:
                        f.write(updated_content)
                    
                    logger.info(f"✅ Updated device choices in {file_path}")
                except Exception as e:
                    logger.error(f"❌ Failed to update device choices in {file_path}: {e}")
        else:
            logger.error(f"❌ Failed to generate test for {args.model}")
            return False
    elif args.architectures:
        # Generate for specific architectures
        all_models = []
        for arch in args.architectures:
            if arch == "all":
                for priority in PRIORITY_MODELS:
                    all_models.extend(PRIORITY_MODELS[priority])
                break
            
            # Find models of this architecture
            arch_models = []
            for priority in PRIORITY_MODELS:
                for model in PRIORITY_MODELS[priority]:
                    # This is a simplistic approach - in a real implementation
                    # we would use the architecture detector to check each model
                    # For now, we'll just use simple substring matching
                    if any(model_part in model for model_part in arch.split("-")):
                        arch_models.append(model)
            
            all_models.extend(arch_models)
        
        # Generate tests for all models
        generated = 0
        failed = 0
        for model in all_models:
            success, file_path = generator.generate_test_file(
                model,
                force=args.force,
                verify=args.verify
            )
            
            if success:
                generated += 1
                
                # If device is specified, update the template to include the device
                if selected_device:
                    try:
                        with open(file_path, "r") as f:
                            content = f.read()
                        
                        # Update the device choice in the generated file
                        updated_content = content.replace(
                            'parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"]',
                            f'parser.add_argument("--device", type=str, choices=["cpu", "cuda", "rocm", "mps", "openvino", "qnn"]'
                        )
                        
                        with open(file_path, "w") as f:
                            f.write(updated_content)
                        
                        logger.info(f"✅ Updated device choices in {file_path}")
                    except Exception as e:
                        logger.error(f"❌ Failed to update device choices in {file_path}: {e}")
            else:
                failed += 1
        
        logger.info(f"Generated {generated} tests, failed to generate {failed} tests")
        
        if failed > 0:
            logger.warning(f"❌ {failed} tests failed to generate")
            return False
    else:
        # Generate by priority
        logger.info(f"Generating tests for {args.priority} priority models")
        generated, failed, total = generator.generate_models_by_priority(
            args.priority,
            verify=args.verify,
            force=args.force
        )
        
        if failed > 0:
            logger.warning(f"❌ {failed} tests failed to generate")
            return generated > 0
        
        # Update device choices in all generated files if needed
        if selected_device and generated > 0:
            for f in os.listdir(args.output_dir):
                if f.endswith(".py"):
                    file_path = os.path.join(args.output_dir, f)
                    try:
                        with open(file_path, "r") as f:
                            content = f.read()
                        
                        # Update the device choice in the generated file
                        updated_content = content.replace(
                            'parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"]',
                            f'parser.add_argument("--device", type=str, choices=["cpu", "cuda", "rocm", "mps", "openvino", "qnn"]'
                        )
                        
                        with open(file_path, "w") as f:
                            f.write(updated_content)
                    except Exception as e:
                        logger.error(f"❌ Failed to update device choices in {file_path}: {e}")
    
    # Generate coverage report if requested
    if args.coverage_report:
        report_path = os.path.join(args.report_dir, "model_test_coverage.md")
        generator.generate_coverage_report(report_path)
    
    return True

def main():
    """Command-line entry point."""
    args = parse_args()
    success = run_generation(args)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())