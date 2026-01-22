#!/usr/bin/env python3
"""
Fixed Test Generator Runner

This is a wrapper around fixed_merged_test_generator.py that applies all
Phase 16 improvements including:
1. Database integration for templates and test results
2. Enhanced hardware detection and compatibility
3. Template inheritance support

Usage:
  python run_fixed_test_generator.py --model bert --use-db-templates --cross-platform
  python run_fixed_test_generator.py --family vision --hardware cuda,openvino,webgpu
  python run_fixed_test_generator.py --all-models --enable-all
"""

import os
import sys
import argparse
import logging
import importlib.util
import traceback
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path for module imports
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Import database integration
try:
    from integrated_improvements.database_integration import (
        DEPRECATE_JSON_OUTPUT, 
        get_db_connection,
        store_test_result,
        DUCKDB_AVAILABLE
    )
    HAS_DB = DUCKDB_AVAILABLE
except ImportError:
    logger.warning("Database integration not available")
    HAS_DB = False
    DEPRECATE_JSON_OUTPUT = os.environ.get("DEPRECATE_JSON_OUTPUT", "1") == "1"

# Import hardware detection
try:
    from integrated_improvements.improved_hardware_detection import (
        detect_all_hardware,
        HARDWARE_PLATFORMS,
        HAS_CUDA, HAS_ROCM, HAS_MPS, HAS_OPENVINO, HAS_WEBNN, HAS_WEBGPU
    )
    HAS_HARDWARE_DETECTION = True
except ImportError:
    logger.warning("Improved hardware detection not available")
    HAS_HARDWARE_DETECTION = False
    HARDWARE_PLATFORMS = ["cpu", "cuda", "openvino", "mps", "rocm", "webnn", "webgpu"]

# Import template database 
try:
    from hardware_test_templates.template_database import (
        TemplateDatabase,
        get_template,
        list_templates
    )
    HAS_TEMPLATE_DB = True
except ImportError:
    logger.warning("Template database not available")
    HAS_TEMPLATE_DB = False

# Run the fixed merged test generator
def main():
    parser = argparse.ArgumentParser(description="Run the fixed test generator")
    parser.add_argument("--model", type=str, help="Model to generate tests for")
    parser.add_argument("--family", type=str, help="Model family to generate tests for")
    parser.add_argument("--all-models", action="store_true", help="Generate tests for all models")
    parser.add_argument("--hardware", type=str, default="all",
                       help="Hardware platforms to target (comma-separated or 'all')")
    parser.add_argument("--cross-platform", action="store_true", help="Enable cross-platform compatibility")
    parser.add_argument("--use-db-templates", action="store_true", help="Use templates from database")
    parser.add_argument("--enable-all", action="store_true", 
                       help="Enable all features (DB templates, cross-platform, etc.)")
    parser.add_argument("--db-path", type=str, default=os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb"),
                       help="Path to database for storing results")
    parser.add_argument("--template-db-path", type=str, 
                       default=os.environ.get("TEMPLATE_DB_PATH", "./template_db.duckdb"),
                       help="Path to template database")
    
    args = parser.parse_args()
    
    # Set environment variables based on arguments
    if args.use_db_templates or args.enable_all:
        os.environ["USE_DB_TEMPLATES"] = "1"
    
    if args.cross_platform or args.enable_all:
        os.environ["CROSS_PLATFORM"] = "1"
    
    if args.db_path:
        os.environ["BENCHMARK_DB_PATH"] = args.db_path
    
    if args.template_db_path:
        os.environ["TEMPLATE_DB_PATH"] = args.template_db_path
    
    # Configure hardware platforms
    hardware_platforms = args.hardware
    if hardware_platforms != "all" and "," in hardware_platforms:
        hardware_platforms = hardware_platforms.split(",")
    
    # Check if at least one model selection option was specified
    if not args.model and not args.family and not args.all_models:
        parser.print_help()
        logger.error("Please specify a model (--model), family (--family), or --all-models")
        return 1
    
    # Import and execute the generator
    try:
        # Import the generator module directly
        from fixed_merged_test_generator import generate_test, generate_tests_for_family, generate_all_tests
        
        # Detect hardware
        if HAS_HARDWARE_DETECTION:
            logger.info("Detecting available hardware...")
            hardware_info = detect_all_hardware()
            logger.info(f"Detected hardware: {', '.join(k for k, v in hardware_info.items() if v.get('detected', False))}")
        
        # Log database status
        if HAS_DB:
            logger.info("Database integration is available")
            if DEPRECATE_JSON_OUTPUT:
                logger.info("JSON output is deprecated, results will be stored in database")
        
        # Log template database status
        if HAS_TEMPLATE_DB and (args.use_db_templates or args.enable_all):
            logger.info("Template database is available and enabled")
            # Count templates
            try:
                db = TemplateDatabase(db_path=args.template_db_path)
                templates = db.list_templates()
                logger.info(f"Found {len(templates)} templates in database")
            except Exception as e:
                logger.error(f"Error counting templates: {e}")
        
        # Generate tests based on the specified options
        if args.model:
            logger.info(f"Generating test for model: {args.model}")
            generate_test(
                model_name=args.model,
                hardware=hardware_platforms,
                cross_platform=args.cross_platform or args.enable_all,
                use_db_templates=args.use_db_templates or args.enable_all
            )
            logger.info(f"Successfully generated test for {args.model}")
        
        elif args.family:
            logger.info(f"Generating tests for family: {args.family}")
            generate_tests_for_family(
                family=args.family,
                hardware=hardware_platforms,
                cross_platform=args.cross_platform or args.enable_all,
                use_db_templates=args.use_db_templates or args.enable_all
            )
            logger.info(f"Successfully generated tests for family {args.family}")
        
        elif args.all_models:
            logger.info("Generating tests for all models")
            generate_all_tests(
                hardware=hardware_platforms,
                cross_platform=args.cross_platform or args.enable_all,
                use_db_templates=args.use_db_templates or args.enable_all
            )
            logger.info("Successfully generated tests for all models")
        
        return 0
    
    except ImportError as e:
        logger.error(f"Error importing fixed_merged_test_generator: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error generating tests: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())