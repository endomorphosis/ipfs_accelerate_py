#!/usr/bin/env python3
"""
Apply Improvements to All Generators

This script applies database integration, cross-platform hardware support, and 
web platform improvements to all test generators and benchmark scripts.

Usage:
  python apply_improvements.py --fix-all
  python apply_improvements.py --fix-tests-only
  python apply_improvements.py --fix-benchmarks-only
"""

import os
import sys
import argparse
import shutil
import logging
import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("apply_improvements")

# Directory paths
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = SCRIPT_DIR.parent
IMPROVEMENTS_DIR = TEST_DIR / "improvements"
BACKUP_DIR = TEST_DIR / "backups"
BACKUP_DIR.mkdir(exist_ok=True)

# Files to fix
GENERATORS = [
    TEST_DIR / "merged_test_generator.py",
    TEST_DIR / "fixed_merged_test_generator.py",
    TEST_DIR / "integrated_skillset_generator.py",
    TEST_DIR / "implementation_generator.py"
]

BENCHMARK_SCRIPTS = [
    TEST_DIR / "benchmark_all_key_models.py",
    TEST_DIR / "run_model_benchmarks.py",
    TEST_DIR / "benchmark_hardware_models.py",
    TEST_DIR / "run_benchmark_with_db.py"
]

def create_backup(file_path):
    """Create a backup of the file."""
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = BACKUP_DIR / f"{file_path.name}.bak_{timestamp}"
    try:
        shutil.copy2(file_path, backup_path)
        logger.info(f"Created backup of {file_path.name} at {backup_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create backup of {file_path}: {e}")
        return False

def apply_database_integration(file_path):
    """Apply database integration to file."""
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return False
    
    # Check if it's a test generator or benchmark script
    is_benchmark = "benchmark" in file_path.name.lower() or "run_" in file_path.name.lower()
    
    # Modify the file to add database integration
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if database integration already exists
    if "integrated_improvements.database_integration" in content:
        logger.info(f"Database integration already exists in {file_path.name}")
        return True
    
    # Add database imports
    db_imports = """
# Database integration
import os
try:
    from integrated_improvements.database_integration import (
        get_db_connection,
        store_test_result,
        store_performance_result,
        create_test_run,
        complete_test_run,
        get_or_create_model,
        get_or_create_hardware_platform,
        DEPRECATE_JSON_OUTPUT
    )
    HAS_DB_INTEGRATION = True
except ImportError:
    logger.warning("Database integration not available")
    HAS_DB_INTEGRATION = False
    DEPRECATE_JSON_OUTPUT = os.environ.get("DEPRECATE_JSON_OUTPUT", "1") == "1"
"""
    
    # Find a place to insert imports
    import_section_end = content.find("# Configure logging")
    if import_section_end == -1:
        import_section_end = content.find("import ")
        if import_section_end != -1:
            # Find the end of imports
            last_import = content.rfind("import ", 0, 1000)
            if last_import != -1:
                import_section_end = content.find("\n", last_import)
    
    if import_section_end != -1:
        content = content[:import_section_end] + db_imports + content[import_section_end:]
    
    # Add database storage function for test generators
    if not is_benchmark:
        db_store_function = """
def store_test_in_database(test_data, db_path=None):
    # Store test generation data in database
    if not HAS_DB_INTEGRATION:
        logger.warning("Database integration not available, cannot store test")
        return False
    
    try:
        # Get database connection
        conn = get_db_connection(db_path)
        if conn is None:
            logger.error("Failed to connect to database")
            return False
        
        # Create test run
        run_id = create_test_run(
            test_name=test_data.get("model_name", "unknown_model"),
            test_type="generator",
            metadata={"generator": os.path.basename(__file__)}
        )
        
        # Get or create model
        model_id = get_or_create_model(
            model_name=test_data.get("model_name", "unknown_model"),
            model_family=test_data.get("model_family"),
            model_type=test_data.get("model_type"),
            metadata=test_data
        )
        
        # Store test result for each hardware platform
        for hardware in test_data.get("hardware_support", []):
            hw_id = get_or_create_hardware_platform(
                hardware_type=hardware,
                metadata={"source": "generator"}
            )
            
            store_test_result(
                run_id=run_id,
                test_name=f"generate_{test_data.get('model_name')}_{hardware}",
                status="PASS",
                model_id=model_id,
                hardware_id=hw_id,
                metadata=test_data
            )
        
        # Complete test run
        complete_test_run(run_id)
        
        logger.info(f"Stored test generation data in database for {test_data.get('model_name', 'unknown')}")
        return True
    except Exception as e:
        logger.error(f"Error storing test in database: {e}")
        return False
"""
        # Add function to content
        function_section = content.find("def ")
        if function_section != -1:
            # Find first blank line after function declarations
            first_function_end = content.find("\ndef ", function_section)
            if first_function_end != -1:
                content = content[:first_function_end] + db_store_function + content[first_function_end:]
    
    # Add database storage function for benchmark scripts
    else:
        db_store_function = """
def store_benchmark_in_database(result, db_path=None):
    # Store benchmark results in database
    if not HAS_DB_INTEGRATION:
        logger.warning("Database integration not available, cannot store benchmark")
        return False
    
    try:
        # Get database connection
        conn = get_db_connection(db_path)
        if conn is None:
            logger.error("Failed to connect to database")
            return False
        
        # Create test run
        run_id = create_test_run(
            test_name=result.get("model_name", "unknown_model"),
            test_type="benchmark",
            metadata={"benchmark_script": os.path.basename(__file__)}
        )
        
        # Get or create model
        model_id = get_or_create_model(
            model_name=result.get("model_name", "unknown_model"),
            model_family=result.get("model_family"),
            model_type=result.get("model_type"),
            metadata=result
        )
        
        # Get or create hardware platform
        hw_id = get_or_create_hardware_platform(
            hardware_type=result.get("hardware", "unknown"),
            metadata={"source": "benchmark"}
        )
        
        # Store performance result
        store_performance_result(
            run_id=run_id,
            model_id=model_id,
            hardware_id=hw_id,
            batch_size=result.get("batch_size", 1),
            throughput=result.get("throughput_items_per_second"),
            latency=result.get("latency_ms"),
            memory=result.get("memory_mb"),
            metadata=result
        )
        
        # Complete test run
        complete_test_run(run_id)
        
        logger.info(f"Stored benchmark result in database for {result.get('model_name', 'unknown')}")
        return True
    except Exception as e:
        logger.error(f"Error storing benchmark in database: {e}")
        return False
"""
        # Add function to content
        function_section = content.find("def ")
        if function_section != -1:
            # Find first blank line after function declarations
            first_function_end = content.find("\ndef ", function_section)
            if first_function_end != -1:
                content = content[:first_function_end] + db_store_function + content[first_function_end:]
    
    # Find save results function and add database storage
    if is_benchmark:
        save_function = content.find("def save_results")
        if save_function != -1:
            save_function_end = content.find("def ", save_function + 10)
            if save_function_end != -1:
                # Extract save results function
                save_function_content = content[save_function:save_function_end]
                
                # Check if it already has database integration
                if "if not DEPRECATE_JSON_OUTPUT:" not in save_function_content and "if DEPRECATE_JSON_OUTPUT:" not in save_function_content:
                    # Modify the save_results function
                    modified_save_function = """def save_results(result, output_dir=None, db_path=None):
    # Save benchmark results to file or database
    # Check if JSON output is deprecated
    if not DEPRECATE_JSON_OUTPUT:
        # Legacy JSON output
        if output_dir is None:
            output_dir = "./benchmark_results"
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/{result['model_name']}_{result['hardware']}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Results saved to {filename}")
    else:
        # Database storage
        store_benchmark_in_database(result, db_path)
"""
                    # Replace old save_results function
                    content = content.replace(save_function_content, modified_save_function)
    
    # Add CLI argument for database path
    argparse_section = content.find("parser = argparse.ArgumentParser")
    if argparse_section != -1:
        args_section_end = content.find("args = parser.parse_args()", argparse_section)
        if args_section_end != -1:
            # Check if db-path argument already exists
            if "--db-path" not in content[argparse_section:args_section_end]:
                # Add db-path argument
                db_path_arg = """
    parser.add_argument("--db-path", type=str, 
                      help="Path to database for storing results",
                      default=os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb"))
"""
                content = content[:args_section_end] + db_path_arg + content[args_section_end:]
    
    # Write the modified content back
    with open(file_path, 'w') as f:
        f.write(content)
    
    logger.info(f"Applied database integration to {file_path.name}")
    return True

def apply_hardware_detection(file_path):
    """Apply improved hardware detection."""
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return False
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if hardware detection already exists
    if "integrated_improvements.improved_hardware_detection" in content:
        logger.info(f"Hardware detection already applied to {file_path.name}")
        return True
    
    # Add hardware detection imports
    hardware_imports = """
# Improved hardware detection
try:
    from integrated_improvements.improved_hardware_detection import (
        detect_available_hardware,
        check_web_optimizations,
        HARDWARE_PLATFORMS,
        HAS_CUDA,
        HAS_ROCM,
        HAS_MPS,
        HAS_OPENVINO,
        HAS_WEBNN,
        HAS_WEBGPU
    )
    HAS_HARDWARE_MODULE = True
except ImportError:
    logger.warning("Improved hardware detection not available")
    HAS_HARDWARE_MODULE = False
"""
    
    # Find a place to insert imports
    import_section_end = content.find("# Configure logging")
    if import_section_end == -1:
        import_section_end = content.find("import ")
        if import_section_end != -1:
            # Find the end of imports
            last_import = content.rfind("import ", 0, 1000)
            if last_import != -1:
                import_section_end = content.find("\n", last_import)
    
    if import_section_end != -1:
        content = content[:import_section_end] + hardware_imports + content[import_section_end:]
    
    # Replace any existing hardware detection
    if "def detect_available_hardware" in content:
        # Find the function
        hw_detect_start = content.find("def detect_available_hardware")
        if hw_detect_start != -1:
            hw_detect_end = content.find("def ", hw_detect_start + 10)
            if hw_detect_end != -1:
                # Extract old function
                old_hw_function = content[hw_detect_start:hw_detect_end]
                
                # Replace with improved function calling the imported version
                new_hw_function = """def detect_available_hardware():
    # Detect available hardware platforms on the current system
    if HAS_HARDWARE_MODULE:
        return detect_available_hardware()
    else:
        # Fallback to basic detection
        available_hardware = {
            "cpu": True  # CPU is always available
        }
        
        # Minimal hardware detection
        try:
            import torch
            available_hardware["cuda"] = torch.cuda.is_available()
        except ImportError:
            available_hardware["cuda"] = False
        
        return available_hardware
"""
                content = content.replace(old_hw_function, new_hw_function)
    
    # Write the modified content back
    with open(file_path, 'w') as f:
        f.write(content)
    
    logger.info(f"Applied hardware detection improvements to {file_path.name}")
    return True

def apply_web_platform_improvements(file_path):
    """Apply web platform improvements."""
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return False
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if web platform optimizations already exist
    if "check_web_optimizations" in content:
        logger.info(f"Web platform optimizations already applied to {file_path.name}")
        return True
    
    # Add web platform optimization handling
    web_platform_code = """
def apply_web_platform_optimizations(model_type, platform="webgpu"):
    # Apply web platform-specific optimizations based on model type
    if HAS_HARDWARE_MODULE:
        return check_web_optimizations(model_type, platform)
    else:
        # Fallback implementation
        optimizations = {
            "compute_shaders": False,
            "parallel_loading": False,
            "shader_precompile": False
        }
        
        # Check environment variables
        compute_shaders = os.environ.get("WEBGPU_COMPUTE_SHADERS_ENABLED", "0") == "1"
        parallel_loading = os.environ.get("WEB_PARALLEL_LOADING_ENABLED", "0") == "1"
        shader_precompile = os.environ.get("WEBGPU_SHADER_PRECOMPILE_ENABLED", "0") == "1"
        
        # Apply optimizations based on model type
        if model_type in ["audio"] and platform == "webgpu" and compute_shaders:
            optimizations["compute_shaders"] = True
        
        if model_type in ["multimodal", "vision_language"] and parallel_loading:
            optimizations["parallel_loading"] = True
        
        if platform == "webgpu" and shader_precompile:
            optimizations["shader_precompile"] = True
        
        return optimizations
"""
    
    # Find a place to add the function
    function_section = content.find("def ")
    if function_section != -1:
        # Find first blank line after function declarations
        first_function_end = content.find("\ndef ", function_section)
        if first_function_end != -1:
            content = content[:first_function_end] + web_platform_code + content[first_function_end:]
    
    # Write the modified content
    with open(file_path, 'w') as f:
        f.write(content)
    
    logger.info(f"Applied web platform improvements to {file_path.name}")
    return True

def fix_file(file_path):
    """Apply all fixes to a file."""
    if not os.path.exists(file_path):
        logger.warning(f"File not found: {file_path}")
        return False
    
    # Create backup
    if not create_backup(file_path):
        logger.error(f"Failed to create backup of {file_path}, skipping")
        return False
    
    # Apply all fixes
    success = True
    success = apply_database_integration(file_path) and success
    success = apply_hardware_detection(file_path) and success
    success = apply_web_platform_improvements(file_path) and success
    
    if success:
        logger.info(f"Successfully fixed {file_path}")
    else:
        logger.error(f"Failed to fully fix {file_path}")
    
    return success

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Apply improvements to generators and benchmarks")
    parser.add_argument("--fix-all", action="store_true", help="Fix all files")
    parser.add_argument("--fix-tests-only", action="store_true", help="Fix only test generators")
    parser.add_argument("--fix-benchmarks-only", action="store_true", help="Fix only benchmark scripts")
    args = parser.parse_args()
    
    # Determine what to fix
    fix_tests = args.fix_all or args.fix_tests_only
    fix_benchmarks = args.fix_all or args.fix_benchmarks_only
    
    if not (fix_tests or fix_benchmarks):
        # If no arguments provided, fix everything
        fix_tests = True
        fix_benchmarks = True
    
    success = True
    
    if fix_tests:
        logger.info("Fixing test generators...")
        for generator in GENERATORS:
            if os.path.exists(generator):
                success = fix_file(generator) and success
            else:
                logger.warning(f"Generator not found: {generator}")
    
    if fix_benchmarks:
        logger.info("Fixing benchmark scripts...")
        for benchmark in BENCHMARK_SCRIPTS:
            if os.path.exists(benchmark):
                success = fix_file(benchmark) and success
            else:
                logger.warning(f"Benchmark script not found: {benchmark}")
    
    if success:
        logger.info("Successfully applied all improvements")
        return 0
    else:
        logger.error("Failed to apply some improvements")
        return 1

if __name__ == "__main__":
    sys.exit(main())