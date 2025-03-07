#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix Benchmark Runner

This script addresses critical issues in the benchmark system:
1. Removes environment variable overrides that artificially flag hardware as available
2. Implements proper error reporting and categorization in benchmarks
3. Ensures clear delineation between real and simulated benchmark data
4. Adds verification steps to confirm fixes work properly

Usage:
    python fix_benchmark_runner.py --target [file_path] --fix-all
    python fix_benchmark_runner.py --target benchmark_all_key_models.py --fix-env-overrides
    python fix_benchmark_runner.py --target run_model_benchmarks.py --fix-error-reporting
"""

import os
import sys
import re
import logging
import argparse
from pathlib import Path
import shutil
import time
from typing import List, Dict, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def backup_file(file_path: str) -> str:
    """Create a backup of the file before making changes."""
    backup_path = f"{file_path}.bak_{int(time.time())}"
    logger.info(f"Creating backup of {file_path} to {backup_path}")
    try:
        shutil.copy(file_path, backup_path)
        logger.info(f"Backup created at {backup_path}")
        return backup_path
    except Exception as e:
        logger.error(f"Failed to create backup: {e}")
        return ""

def fix_environment_overrides(file_path: str) -> Tuple[bool, List[str]]:
    """
    Remove environment variable overrides that artificially flag hardware as available.
    
    Args:
        file_path: Path to the file to fix
        
    Returns:
        Tuple of (success flag, list of changes made)
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False, []
        
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Track changes made
    changes = []
    
    # Look for patterns that set environment variables to fake hardware availability
    patterns = [
        (r'os\.environ\["WEBNN_SIMULATION"\]\s*=\s*"1"', "WEBNN_SIMULATION environment override"),
        (r'os\.environ\["WEBNN_AVAILABLE"\]\s*=\s*"1"', "WEBNN_AVAILABLE environment override"),
        (r'os\.environ\["WEBGPU_SIMULATION"\]\s*=\s*"1"', "WEBGPU_SIMULATION environment override"),
        (r'os\.environ\["WEBGPU_AVAILABLE"\]\s*=\s*"1"', "WEBGPU_AVAILABLE environment override"),
        (r'os\.environ\["QNN_SIMULATION_MODE"\]\s*=\s*"1"', "QNN_SIMULATION_MODE environment override"),
        (r'os\.environ\["QUALCOMM_SDK"\]\s*=\s*"1"', "QUALCOMM_SDK environment override"),
        (r'os\.environ\["ROCM_HOME"\]\s*=\s*"1"', "ROCM_HOME environment override"),
    ]
    
    modified_content = content
    for pattern, description in patterns:
        if re.search(pattern, content):
            # Replace with commented-out line and add explanatory comment
            modified_content = re.sub(
                pattern,
                f'# REMOVED: {description} - Using actual hardware detection instead\n# \\g<0>',
                modified_content
            )
            changes.append(f"Removed {description}")
    
    # Check if hardware detection is properly used
    if not re.search(r'from\s+hardware_detection\s+import', modified_content) and \
       not re.search(r'from\s+centralized_hardware_detection\.hardware_detection\s+import', modified_content):
        
        # Add import for hardware detection
        import_section = re.search(r'^import.*?\n\n', modified_content, re.DOTALL)
        if import_section:
            # Add import after other imports
            new_import = 'from centralized_hardware_detection.hardware_detection import get_capabilities, get_hardware_manager\n'
            modified_content = modified_content[:import_section.end()] + new_import + modified_content[import_section.end():]
            changes.append("Added import for centralized hardware detection")
    
    # Check if a fix is needed to handle unavailable hardware properly
    if 'def _handle_unavailable_hardware(' not in modified_content:
        # Find a good place to insert the function - after imports but before other functions
        match = re.search(r'^\s*def\s+\w+\(', modified_content, re.MULTILINE)
        if match:
            # Insert before the first function
            handler_function = '''
def _handle_unavailable_hardware(hardware_type, model_name, batch_size, reason=None):
    """
    Handle the case where hardware is unavailable properly.
    
    Args:
        hardware_type: The hardware type that is unavailable
        model_name: The model being benchmarked
        batch_size: The batch size for the benchmark
        reason: Optional reason for hardware unavailability
        
    Returns:
        Dictionary with properly marked simulated results
    """
    logger.warning(f"Hardware {hardware_type} not available for model {model_name}. Reason: {reason or 'Not detected'}")
    
    # Return a result with clear simulation markers
    return {
        "model_name": model_name,
        "hardware_type": hardware_type,
        "batch_size": batch_size,
        "is_simulated": True,
        "simulation_reason": reason or f"Hardware {hardware_type} not available",
        "throughput_items_per_second": 0.0,  # Zero indicates simulation
        "latency_ms": 0.0,  # Zero indicates simulation
        "memory_mb": 0.0,  # Zero indicates simulation
        "error_category": "hardware_unavailable",
        "error_details": {"reason": reason or "Hardware not available"}
    }

'''
            modified_content = modified_content[:match.start()] + handler_function + modified_content[match.start():]
            changes.append("Added _handle_unavailable_hardware function for proper error handling")
    
    # Update hardware detection logic
    if 'hardware_type in available_hardware' in modified_content:
        # Replace the simple check with proper error handling
        modified_content = re.sub(
            r'if\s+hardware_type\s+in\s+available_hardware:',
            'if hardware_type in available_hardware and available_hardware[hardware_type]:',
            modified_content
        )
        
        # Add proper else clause if missing
        if not re.search(r'if\s+hardware_type\s+in\s+available_hardware.*?else', modified_content, re.DOTALL):
            modified_content = re.sub(
                r'(if\s+hardware_type\s+in\s+available_hardware.*?:.*?)(\s+[^\s]+)',
                '\\1\\2\n    else:\n        return _handle_unavailable_hardware(hardware_type, model_name, batch_size, "Not in available hardware list")',
                modified_content,
                flags=re.DOTALL
            )
        
        changes.append("Updated hardware detection logic with proper error handling")
    
    # Write changes back to file if any were made
    if content != modified_content:
        with open(file_path, 'w') as f:
            f.write(modified_content)
        logger.info(f"Fixed environment overrides in {file_path}")
        return True, changes
    else:
        logger.info(f"No environment overrides found in {file_path}")
        return False, []

def fix_error_reporting(file_path: str) -> Tuple[bool, List[str]]:
    """
    Implement proper error reporting and categorization in benchmarks.
    
    Args:
        file_path: Path to the file to fix
        
    Returns:
        Tuple of (success flag, list of changes made)
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False, []
        
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Track changes made
    changes = []
    
    # Look for error handling patterns
    modified_content = content
    
    # Check for try/except blocks without error categorization
    try_except_blocks = re.finditer(r'try:.*?except\s+(\w+)(?:\s+as\s+(\w+))?:(.*?)(?=try:|$)', modified_content, re.DOTALL)
    for match in try_except_blocks:
        exception_type = match.group(1)
        exception_var = match.group(2) or 'e'  # Default to 'e' if not specified
        exception_block = match.group(3)
        
        # Check if error categorization is missing
        if not re.search(r'error_category', exception_block) and 'logger.' in exception_block:
            # Add error categorization
            improved_block = exception_block.replace(
                f'logger.error',
                f'error_category = "runtime_error_{exception_type.lower()}"\n        error_details = {{"exception": str({exception_var}), "type": "{exception_type}"}}\n        logger.error'
            )
            modified_content = modified_content.replace(exception_block, improved_block)
            changes.append(f"Added error categorization for {exception_type} exceptions")
    
    # Check for functions that store benchmark results
    store_results_pattern = r'def\s+(\w+_results|store_benchmark|save_results|log_benchmark|record_benchmark).*?\(.*?\):(.*?)(?=def|\Z)'
    store_functions = re.finditer(store_results_pattern, modified_content, re.DOTALL)
    
    for match in store_functions:
        function_name = match.group(1)
        function_body = match.group(2)
        
        # Check if is_simulated flag handling is missing
        if 'is_simulated' not in function_body:
            # Find the place where results are stored/inserted
            if 'INSERT INTO' in function_body:
                # DuckDB direct insertion
                modified_body = function_body.replace(
                    'INSERT INTO',
                    '# Add simulation status if not present\n    if "is_simulated" not in result:\n        result["is_simulated"] = True\n        result["simulation_reason"] = "No simulation status provided"\n\n    INSERT INTO'
                )
                modified_content = modified_content.replace(function_body, modified_body)
                changes.append(f"Added simulation status check in {function_name}")
            elif 'conn.execute(' in function_body and 'INSERT' in function_body:
                # Using conn.execute for insertion
                insert_match = re.search(r'conn\.execute\([\'"]INSERT.*?[\'"]', function_body)
                if insert_match:
                    insert_stmt = insert_match.group(0)
                    if 'is_simulated' not in insert_stmt:
                        # Update insert to include is_simulated column
                        modified_body = function_body.replace(
                            insert_stmt,
                            insert_stmt.replace('INSERT INTO', 'INSERT INTO') + '\n    # Add simulation status if not present\n    if "is_simulated" not in result:\n        result["is_simulated"] = True\n        result["simulation_reason"] = "No simulation status provided"'
                        )
                        modified_content = modified_content.replace(function_body, modified_body)
                        changes.append(f"Added simulation status check in {function_name} database insertion")
    
    # Write changes back to file if any were made
    if content != modified_content:
        with open(file_path, 'w') as f:
            f.write(modified_content)
        logger.info(f"Fixed error reporting in {file_path}")
        return True, changes
    else:
        logger.info(f"No error reporting issues found in {file_path}")
        return False, []

def fix_simulation_tracking(file_path: str) -> Tuple[bool, List[str]]:
    """
    Ensure clear delineation between real and simulated benchmark data.
    
    Args:
        file_path: Path to the file to fix
        
    Returns:
        Tuple of (success flag, list of changes made)
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False, []
        
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Track changes made
    changes = []
    
    # Look for functions that generate sample/dummy data
    dummy_data_patterns = [
        r'def\s+generate_sample_data',
        r'def\s+create_dummy_data',
        r'def\s+get_sample_results',
    ]
    
    modified_content = content
    for pattern in dummy_data_patterns:
        match = re.search(f'{pattern}.*?\(.*?\):(.*?)(?=def|\Z)', modified_content, re.DOTALL)
        if match:
            function_body = match.group(1)
            
            # Check if is_simulated flag is missing
            if 'is_simulated' not in function_body:
                # Find return statement or dictionary creation
                if 'return {' in function_body:
                    # Add is_simulated flags to returned dictionary
                    modified_body = function_body.replace(
                        'return {',
                        'return {\n        "is_simulated": True,\n        "simulation_reason": "Generated sample data",'
                    )
                    modified_content = modified_content.replace(function_body, modified_body)
                    changes.append("Added simulation flags to sample data generator")
                elif 'results =' in function_body:
                    # Add is_simulated flags to results
                    modified_body = function_body.replace(
                        'results =',
                        '# Mark results as simulated\n    is_simulated = True\n    simulation_reason = "Generated sample data"\n    results ='
                    )
                    # Add flags to each result
                    modified_body = modified_body.replace(
                        'results.append(',
                        'result = {\n        "is_simulated": is_simulated,\n        "simulation_reason": simulation_reason\n    }\n    result.update('
                    )
                    modified_body = modified_body.replace('results.append(', 'results.append(result)')
                    modified_content = modified_content.replace(function_body, modified_body)
                    changes.append("Added simulation flags to sample data generator")
    
    # Check for fallback mechanisms that use simulated data
    fallback_patterns = [
        (r'except.*?logger\.warning.*?[Uu]sing sample data', "sample data fallback"),
        (r'if\s+len\(results\)\s*==\s*0.*?generate_sample', "empty results fallback"),
        (r'if not os\.path\.exists\(.*?\).*?sample_data', "missing file fallback"),
    ]
    
    for pattern, description in fallback_patterns:
        match = re.search(pattern, modified_content, re.DOTALL)
        if match:
            # Add explicit simulation warning
            fallback_code = match.group(0)
            if 'is_simulated' not in fallback_code:
                improved_code = fallback_code.replace(
                    'logger.warning',
                    'logger.warning("SIMULATION MODE: Using simulated data instead of actual hardware measurements")\n    logger.warning'
                )
                modified_content = modified_content.replace(fallback_code, improved_code)
                changes.append(f"Added explicit simulation warning for {description}")
    
    # Write changes back to file if any were made
    if content != modified_content:
        with open(file_path, 'w') as f:
            f.write(modified_content)
        logger.info(f"Fixed simulation tracking in {file_path}")
        return True, changes
    else:
        logger.info(f"No simulation tracking issues found in {file_path}")
        return False, []

def fix_verification_steps(file_path: str) -> Tuple[bool, List[str]]:
    """
    Add verification steps to confirm fixes work properly.
    
    Args:
        file_path: Path to the file to fix
        
    Returns:
        Tuple of (success flag, list of changes made)
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False, []
        
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Track changes made
    changes = []
    
    # Look for run_benchmark or main function
    main_function_pattern = r'def\s+(run_benchmark|main|run_benchmarks).*?\(.*?\):(.*?)(?=def|\Z|\s*if\s+__name__\s*==\s*[\'"]__main__[\'"])'
    main_match = re.search(main_function_pattern, content, re.DOTALL)
    
    modified_content = content
    if main_match:
        function_name = main_match.group(1)
        function_body = main_match.group(2)
        
        # Check if verification logic is missing
        if 'verify_hardware_availability' not in function_body and 'verify_benchmark_results' not in function_body:
            # Add verification logic at the end of the function
            last_line = function_body.rstrip()
            indent = re.search(r'^\s+', function_body)
            indent = indent.group(0) if indent else '    '
            
            verification_code = f'''
{indent}# Verify hardware detection is working properly
{indent}hardware_status = verify_hardware_availability()
{indent}logger.info(f"Hardware verification results: {{len([h for h, v in hardware_status.items() if v])}} platforms available")
{indent}for hw, available in hardware_status.items():
{indent}    availability = "AVAILABLE" if available else "NOT AVAILABLE"
{indent}    logger.info(f"  {{hw}}: {{availability}}")

{indent}# Verify benchmark results have proper simulation status
{indent}if hasattr(args, 'db_path') and args.db_path and os.path.exists(args.db_path):
{indent}    verify_benchmark_results(args.db_path)
{indent}elif os.environ.get("BENCHMARK_DB_PATH") and os.path.exists(os.environ.get("BENCHMARK_DB_PATH")):
{indent}    verify_benchmark_results(os.environ.get("BENCHMARK_DB_PATH"))
{indent}else:
{indent}    logger.warning("No database path provided, skipping benchmark result verification")
'''
            
            modified_body = function_body.rstrip() + verification_code
            modified_content = modified_content.replace(function_body, modified_body)
            changes.append(f"Added verification steps to {function_name} function")
            
            # Add verification functions
            if 'def verify_hardware_availability(' not in modified_content:
                # Find a good spot to add the functions - before the main function
                verification_functions = '''
def verify_hardware_availability():
    """
    Verify that hardware detection is working properly.
    
    Returns:
        Dictionary of hardware availability status
    """
    try:
        # Import hardware detection
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from centralized_hardware_detection.hardware_detection import get_capabilities, get_hardware_manager
        
        # Get hardware capabilities
        capabilities = get_capabilities()
        
        # Check if environment variables are overriding detection
        env_overrides = {
            "WEBNN_SIMULATION": os.environ.get("WEBNN_SIMULATION") == "1",
            "WEBGPU_SIMULATION": os.environ.get("WEBGPU_SIMULATION") == "1",
            "QNN_SIMULATION_MODE": os.environ.get("QNN_SIMULATION_MODE") == "1",
            "WEBNN_AVAILABLE": os.environ.get("WEBNN_AVAILABLE") == "1",
            "WEBGPU_AVAILABLE": os.environ.get("WEBGPU_AVAILABLE") == "1",
            "QUALCOMM_SDK": "QUALCOMM_SDK" in os.environ,
        }
        
        if any(env_overrides.values()):
            logger.warning("Environment variables are overriding hardware detection:")
            for var, value in env_overrides.items():
                if value:
                    logger.warning(f"  {var} is set - this may provide inaccurate hardware availability")
        
        return {
            "cpu": capabilities.get("cpu", True),  # CPU is always available
            "cuda": capabilities.get("cuda", False),
            "rocm": capabilities.get("rocm", False),
            "mps": capabilities.get("mps", False),
            "openvino": capabilities.get("openvino", False),
            "qualcomm": capabilities.get("qualcomm", False),
            "webnn": capabilities.get("webnn", False),
            "webgpu": capabilities.get("webgpu", False),
        }
    except Exception as e:
        logger.error(f"Error verifying hardware availability: {e}")
        return {
            "cpu": True,  # CPU is always available
            "cuda": False,
            "rocm": False,
            "mps": False,
            "openvino": False,
            "qualcomm": False,
            "webnn": False,
            "webgpu": False,
        }

def verify_benchmark_results(db_path):
    """
    Verify that benchmark results have proper simulation status.
    
    Args:
        db_path: Path to the benchmark database
    """
    try:
        import duckdb
        
        # Connect to the database
        conn = duckdb.connect(db_path)
        
        # Check if performance_results table exists
        tables = conn.execute("SHOW TABLES").fetchdf()
        if "performance_results" not in tables['name'].values:
            logger.warning("performance_results table not found in database")
            return
        
        # Check if is_simulated column exists
        columns = conn.execute("PRAGMA table_info(performance_results)").fetchdf()
        if "is_simulated" not in columns['name'].values:
            logger.warning("is_simulated column not found in performance_results table")
            return
        
        # Get simulation statistics
        simulation_stats = conn.execute("""
        SELECT 
            is_simulated, 
            COUNT(*) as count, 
            MIN(created_at) as earliest, 
            MAX(created_at) as latest
        FROM performance_results
        GROUP BY is_simulated
        """).fetchdf()
        
        logger.info("Benchmark result simulation statistics:")
        for _, row in simulation_stats.iterrows():
            is_simulated = row['is_simulated']
            count = row['count']
            status = "SIMULATED" if is_simulated else "REAL HARDWARE"
            logger.info(f"  {status}: {count} results (from {row['earliest']} to {row['latest']})")
        
        # Check recent results for simulation status
        recent_results = conn.execute("""
        SELECT 
            model_name, 
            hardware_type, 
            is_simulated, 
            simulation_reason,
            created_at
        FROM performance_results
        ORDER BY created_at DESC
        LIMIT 5
        """).fetchdf()
        
        logger.info("Most recent benchmark results:")
        for _, row in recent_results.iterrows():
            model = row['model_name']
            hardware = row['hardware_type']
            is_simulated = row['is_simulated']
            reason = row['simulation_reason'] if is_simulated else "REAL HARDWARE"
            logger.info(f"  {model} on {hardware}: {'SIMULATED' if is_simulated else 'REAL'} - {reason}")
        
        # Close the connection
        conn.close()
    except Exception as e:
        logger.error(f"Error verifying benchmark results: {e}")

'''
                modified_content = modified_content.replace(main_match.group(0), verification_functions + main_match.group(0))
                changes.append("Added verification functions")
    
    # Write changes back to file if any were made
    if content != modified_content:
        with open(file_path, 'w') as f:
            f.write(modified_content)
        logger.info(f"Added verification steps to {file_path}")
        return True, changes
    else:
        logger.info(f"No verification steps needed in {file_path}")
        return False, []

def fix_syntax_errors(file_path: str) -> Tuple[bool, List[str]]:
    """
    Fix syntax errors in the benchmark runner script (legacy functionality).
    
    Args:
        file_path: Path to the file to fix
        
    Returns:
        Tuple of (success flag, list of changes made)
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False, []
    
    changes = []
    
    # Read the current content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # The simplest approach is to completely rewrite the problematic function
    # First, let's identify where the main() function begins
    main_start_index = content.find("def main():")
    if main_start_index == -1:
        logger.error("Could not find main() function in the file.")
        return False, []
    
    # Get everything before main()
    content_before_main = content[:main_start_index]
    
    # Create a new, corrected main() function
    new_main_function = """def main():
    \"\"\"Main function for running model benchmarks from command line\"\"\"
    parser = argparse.ArgumentParser(description="Comprehensive Model Benchmark Runner")
    
    # Main options
    parser.add_argument("--output-dir", type=str, default="./benchmark_results", help="Output directory for benchmark results")
    parser.add_argument("--models-set", choices=["key", "small", "custom"], default="key", help="Which model set to use")
    parser.add_argument("--custom-models", type=str, help="JSON file with custom models configuration (required if models-set=custom)")
    parser.add_argument("--hardware", type=str, nargs="+", help="Hardware platforms to test (defaults to all available)")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=DEFAULT_BATCH_SIZES, help="Batch sizes to test")
    parser.add_argument("--verify-only", action="store_true", help="Only verify functionality without performance benchmarks")
    parser.add_argument("--benchmark-only", action="store_true", help="Only run performance benchmarks without verification")
    parser.add_argument("--no-plots", action="store_true", help="Disable plot generation")
    parser.add_argument("--no-compatibility-update", action="store_true", help="Disable compatibility matrix update")
    parser.add_argument("--no-resource-pool", action="store_true", help="Disable ResourcePool for model caching")
    parser.add_argument("--specific-models", type=str, nargs="+", help="Only benchmark specific models (by key) from the selected set")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--db-path", type=str, default="./benchmark_db.duckdb", help="Path to DuckDB database for storing results")
    parser.add_argument("--no-db-store", action="store_true", help="Disable storing results in the database")
    parser.add_argument("--visualize-from-db", action="store_true", help="Generate visualizations from database instead of current run results")
    parser.add_argument("--db-only", action="store_true", help="Store results only in the database, not in JSON")
    parser.add_argument("--models", type=str, nargs="+", help="Specific models to benchmark (alternative to --specific-models)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    # Process command line arguments
    args = parser.parse_args()
    
    # Check for deprecated JSON output
    if DEPRECATE_JSON_OUTPUT:
        logger.info("JSON output is deprecated. Results are stored directly in the database.")
    
    # Configure logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    # Handle custom models
    custom_models = None
    if args.models_set == "custom":
        if not args.custom_models:
            logger.error("--custom-models is required when using --models-set=custom")
            return 1
        
        try:
            # Try database first, fall back to JSON if necessary
            try:
                from benchmark_db_api import BenchmarkDBAPI
                db_api = BenchmarkDBAPI(db_path=os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb"))
                custom_models = db_api.get_benchmark_results()
                logger.info("Successfully loaded results from database")
            except Exception as e:
                logger.warning(f"Error reading from database, falling back to JSON: {e}")
                
            with open(args.custom_models, 'r') as f:
                custom_models = json.load(f)
        except Exception as e:
            logger.error(f"Error loading custom models: {e}")
            return 1
    
    # Handle specific models
    if args.models:
        args.specific_models = args.models
    
    if args.specific_models:
        if args.models_set == "key":
            model_set = {k: v for k, v in KEY_MODEL_SET.items() if k in args.specific_models}
        elif args.models_set == "small":
            model_set = {k: v for k, v in SMALL_MODEL_SET.items() if k in args.specific_models}
        elif args.models_set == "custom":
            model_set = {k: v for k, v in custom_models.items() if k in args.specific_models}
        
        # Check if we have any models after filtering
        if not model_set:
            logger.error(f"No models found matching the specified keys: {args.specific_models}")
            return 1
        
        custom_models = model_set
        args.models_set = "custom"
    
    # Create and run benchmarks
    runner = ModelBenchmarkRunner(
        output_dir=args.output_dir,
        models_set=args.models_set,
        custom_models=custom_models,
        hardware_types=args.hardware,
        batch_sizes=args.batch_sizes,
        verify_functionality=not args.benchmark_only,
        measure_performance=not args.verify_only,
        generate_plots=not args.no_plots,
        update_compatibility_matrix=not args.no_compatibility_update,
        use_resource_pool=not args.no_resource_pool,
        db_path=args.db_path,
        store_in_db=not args.no_db_store,
        db_only=args.db_only,
        verbose=args.verbose or args.debug
    )
    
    success = runner.run()
    
    # Generate visualizations if requested
    if args.visualize_from_db:
        logger.info("Generating visualizations from database...")
        if not runner.visualize_from_db():
            logger.error("Failed to generate visualizations from database")
            return 1
    
    # Verify hardware detection is working properly
    hardware_status = verify_hardware_availability()
    logger.info(f"Hardware verification results: {len([h for h, v in hardware_status.items() if v])} platforms available")
    for hw, available in hardware_status.items():
        availability = "AVAILABLE" if available else "NOT AVAILABLE"
        logger.info(f"  {hw}: {availability}")

    # Verify benchmark results have proper simulation status
    if args.db_path and os.path.exists(args.db_path):
        verify_benchmark_results(args.db_path)
    elif os.environ.get("BENCHMARK_DB_PATH") and os.path.exists(os.environ.get("BENCHMARK_DB_PATH")):
        verify_benchmark_results(os.environ.get("BENCHMARK_DB_PATH"))
    else:
        logger.warning("No database path provided, skipping benchmark result verification")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
"""
    
    # Create the new content
    modified_content = content_before_main + new_main_function
    
    # Write the fixed content back to the file if changes were made
    if content != modified_content:
        with open(file_path, 'w') as f:
            f.write(modified_content)
        
        changes.append("Replaced main() function with corrected version")
        logger.info(f"Successfully fixed syntax in {file_path}")
        return True, changes
    
    return False, ["No syntax errors found"]

def run_fixes(target_file: str, fixes: List[str]) -> Dict[str, List[str]]:
    """
    Run the specified fixes on the target file.
    
    Args:
        target_file: Path to the file to fix
        fixes: List of fixes to apply (environment, error, simulation, verification, or all)
        
    Returns:
        Dictionary of fix name to list of changes made
    """
    changes = {}
    
    # Create backup before making changes
    backup_path = backup_file(target_file)
    if not backup_path:
        logger.error(f"Failed to create backup of {target_file}, aborting")
        return changes
    
    # Apply fixes
    if "all" in fixes or "environment" in fixes:
        success, env_changes = fix_environment_overrides(target_file)
        changes["environment"] = env_changes
    
    if "all" in fixes or "error" in fixes:
        success, error_changes = fix_error_reporting(target_file)
        changes["error"] = error_changes
    
    if "all" in fixes or "simulation" in fixes:
        success, sim_changes = fix_simulation_tracking(target_file)
        changes["simulation"] = sim_changes
    
    if "all" in fixes or "verification" in fixes:
        success, ver_changes = fix_verification_steps(target_file)
        changes["verification"] = ver_changes
    
    if "all" in fixes or "syntax" in fixes:
        success, syntax_changes = fix_syntax_errors(target_file)
        changes["syntax"] = syntax_changes
    
    return changes

def main():
    """Main function to fix benchmark runner issues."""
    parser = argparse.ArgumentParser(description="Fix Benchmark Runner Issues")
    parser.add_argument("--target", type=str, required=True, help="Path to the target file to fix")
    
    # Fix options
    fix_group = parser.add_mutually_exclusive_group(required=True)
    fix_group.add_argument("--fix-all", action="store_true", help="Apply all fixes")
    fix_group.add_argument("--fix-env-overrides", action="store_true", help="Fix environment variable overrides")
    fix_group.add_argument("--fix-error-reporting", action="store_true", help="Fix error reporting and categorization")
    fix_group.add_argument("--fix-simulation-tracking", action="store_true", help="Fix simulation status tracking")
    fix_group.add_argument("--fix-verification-steps", action="store_true", help="Add verification steps")
    fix_group.add_argument("--fix-syntax-errors", action="store_true", help="Fix syntax errors (legacy functionality)")
    
    args = parser.parse_args()
    
    # Determine which fixes to apply
    fixes = []
    if args.fix_all:
        fixes.append("all")
    if args.fix_env_overrides:
        fixes.append("environment")
    if args.fix_error_reporting:
        fixes.append("error")
    if args.fix_simulation_tracking:
        fixes.append("simulation")
    if args.fix_verification_steps:
        fixes.append("verification")
    if args.fix_syntax_errors:
        fixes.append("syntax")
    
    # Run fixes
    target_file = args.target
    changes = run_fixes(target_file, fixes)
    
    # Report changes
    logger.info(f"Changes made to {target_file}:")
    for fix_type, fix_changes in changes.items():
        if fix_changes:
            logger.info(f"  {fix_type.upper()} fixes:")
            for change in fix_changes:
                logger.info(f"    - {change}")
        else:
            logger.info(f"  {fix_type.upper()} fixes: No changes needed")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())