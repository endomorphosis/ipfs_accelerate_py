#!/usr/bin/env python3
"""
Generate HuggingFace model tests with ModelTest pattern.

This script:
1. Detects model architecture using enhanced architecture detection
2. Generates standardized test files based on ModelTest base classes
3. Validates generated files for syntax and pattern compliance
4. Creates model coverage reports

Usage:
    python generate_model_tests.py --model MODEL_NAME
    python generate_model_tests.py --priority {high,medium,low,all}
    python generate_model_tests.py --validate
    python generate_model_tests.py --report
"""

import os
import sys
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"generate_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import components
try:
    from scripts.generators.test_generator import ModelTestGenerator
    from validation.test_validator import validate_test_files, generate_validation_report
except ImportError as e:
    logger.error(f"Error importing required modules: {e}")
    logger.error("Make sure you are running this script from the refactored_test_suite directory.")
    sys.exit(1)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate HuggingFace model tests")
    
    # Command groups
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str, help="Generate test for a specific model")
    group.add_argument("--priority", type=str, choices=["high", "medium", "low", "all"], 
                        help="Generate tests for models with this priority")
    group.add_argument("--validate", action="store_true", 
                        help="Validate existing test files")
    group.add_argument("--report", action="store_true", 
                        help="Generate coverage report for existing tests")
    
    # Options
    parser.add_argument("--output-dir", type=str, default="generated_tests", 
                        help="Directory to save generated files")
    parser.add_argument("--template-dir", type=str, 
                        help="Directory containing templates")
    parser.add_argument("--force", action="store_true", 
                        help="Overwrite existing files")
    parser.add_argument("--no-verify", action="store_true", 
                        help="Skip verification of generated files")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create generator
    generator = ModelTestGenerator(
        output_dir=args.output_dir,
        template_dir=args.template_dir
    )
    
    # Handle commands
    if args.model:
        # Generate test for a specific model
        logger.info(f"Generating test for model: {args.model}")
        success, file_path = generator.generate_test_file(
            args.model, 
            force=args.force, 
            verify=not args.no_verify
        )
        
        if success:
            logger.info(f"✅ Successfully generated test for {args.model}")
            print(f"Generated test file: {file_path}")
            return 0
        else:
            logger.error(f"❌ Failed to generate test for {args.model}")
            return 1
    
    elif args.priority:
        # Generate tests for models with given priority
        logger.info(f"Generating tests for {args.priority} priority models")
        generated, failed, total = generator.generate_models_by_priority(
            args.priority, 
            verify=not args.no_verify, 
            force=args.force
        )
        
        logger.info(f"Generated {generated} out of {total} test files")
        print(f"Generated {generated} out of {total} test files")
        
        return 0 if failed == 0 else 1
    
    elif args.validate:
        # Validate existing test files
        logger.info("Validating test files")
        results = validate_test_files(args.output_dir, "test_hf_*.py")
        
        # Generate validation report
        report_file = os.path.join(args.output_dir, "validation_report.md")
        generate_validation_report(results, report_file)
        
        # Print summary
        print(f"Validation complete: {results['valid']} out of {results['total']} files valid")
        print(f"Report written to: {report_file}")
        
        return 0 if results["invalid"] == 0 else 1
    
    elif args.report:
        # Generate coverage report
        logger.info("Generating coverage report")
        success = generator.generate_coverage_report(
            os.path.join(args.output_dir, "model_test_coverage.md")
        )
        
        if success:
            logger.info("✅ Successfully generated coverage report")
            print(f"Coverage report written to: {os.path.join(args.output_dir, 'model_test_coverage.md')}")
            return 0
        else:
            logger.error("❌ Failed to generate coverage report")
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())