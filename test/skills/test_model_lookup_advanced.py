#!/usr/bin/env python3

"""
Test script for the enhanced model lookup system with advanced selection.

Usage:
    python test_model_lookup_advanced.py --model-type MODEL_TYPE [--task TASK] [--hardware PROFILE]
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from advanced_model_selection import (
        select_model_advanced, 
        get_hardware_profile,
        TASK_TO_MODEL_TYPES,
        HARDWARE_PROFILES
    )
    HAS_ADVANCED_SELECTION = True
except ImportError:
    logger.error("Could not import advanced_model_selection.py")
    sys.exit(1)

try:
    from find_models import get_recommended_default_model, query_huggingface_api
    HAS_MODEL_LOOKUP = True
except ImportError:
    logger.error("Could not import find_models.py")
    sys.exit(1)

try:
    from test_generator_fixed import get_model_from_registry
    HAS_GENERATOR = True
except ImportError:
    logger.error("Could not import test_generator_fixed.py")
    sys.exit(1)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test enhanced model lookup system")
    parser.add_argument("--model-type", type=str, required=True, help="Model type to select")
    parser.add_argument("--task", type=str, help="Specific task for model selection")
    parser.add_argument("--hardware", type=str, choices=list(HARDWARE_PROFILES.keys()), 
                      help="Hardware profile for size constraints")
    parser.add_argument("--max-size", type=int, help="Maximum model size in MB")
    parser.add_argument("--framework", type=str, help="Framework compatibility")
    parser.add_argument("--detect-hardware", action="store_true", help="Detect hardware profile")
    parser.add_argument("--list-tasks", action="store_true", help="List available tasks for model type")
    
    args = parser.parse_args()
    
    # Detect hardware if requested
    if args.detect_hardware:
        profile = get_hardware_profile()
        print(f"\nDetected Hardware Profile:")
        print(f"  Max model size: {profile['max_size_mb']}MB")
        print(f"  Description: {profile['description']}")
        return 0
    
    # List tasks if requested
    if args.list_tasks:
        print("\nAvailable Tasks:")
        for task, model_types in sorted(TASK_TO_MODEL_TYPES.items()):
            if args.model_type in model_types:
                print(f"  - {task}")
        return 0
    
    # Get hardware profile if specified
    hardware_profile = None
    if args.hardware:
        hardware_profile = args.hardware
        print(f"Using hardware profile: {hardware_profile}")
        print(f"  Max size: {HARDWARE_PROFILES[hardware_profile]['max_size_mb']}MB")
        print(f"  Description: {HARDWARE_PROFILES[hardware_profile]['description']}")
    
    # Show the model selection process with all methods
    print(f"\nModel Lookup Results for '{args.model_type}':")
    
    # Method 1: Basic model lookup (find_models.py)
    try:
        basic_model = get_recommended_default_model(args.model_type)
        print(f"\n1. Basic Model Lookup (find_models.py):")
        print(f"   Selected model: {basic_model}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Method 2: Advanced model selection (advanced_model_selection.py)
    try:
        advanced_model = select_model_advanced(
            args.model_type,
            task=args.task,
            hardware_profile=args.hardware,
            max_size_mb=args.max_size,
            framework=args.framework
        )
        print(f"\n2. Advanced Model Selection (advanced_model_selection.py):")
        print(f"   Selected model: {advanced_model}")
        print(f"   Task: {args.task if args.task else 'Not specified'}")
        print(f"   Hardware profile: {args.hardware if args.hardware else 'Auto-detected'}")
        print(f"   Max size: {args.max_size if args.max_size else 'Not constrained'}")
        print(f"   Framework: {args.framework if args.framework else 'Any'}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Method 3: Integrated model selection (test_generator_fixed.py)
    try:
        integrated_model = get_model_from_registry(
            args.model_type,
            task=args.task,
            hardware_profile=args.hardware,
            max_size_mb=args.max_size,
            framework=args.framework
        )
        print(f"\n3. Integrated Model Selection (test_generator_fixed.py):")
        print(f"   Selected model: {integrated_model}")
    except Exception as e:
        print(f"   Error: {e}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
