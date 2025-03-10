#!/usr/bin/env python
"""
Benchmark Database Updater

This script provides a simple interface for test runners to write results directly to the
benchmark database. It handles creating necessary dimensions (hardware, models) and
inserting test results with proper foreign key relationships.
"""

import os
import sys
import json
import argparse
import logging
import datetime
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
import uuid

# Add parent directory to path for module imports
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("benchmark_updater")

def parse_args():
    parser = argparse.ArgumentParser(description="Update benchmark database with test results")
    
    parser.add_argument("--db", type=str, default="./benchmark_db.duckdb", 
                        help="Path to DuckDB database")
    
    # Input methods
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-file", type=str,
                            help="JSON file containing test results")
    input_group.add_argument("--result-type", type=str, choices=['performance', 'compatibility', 'integration'],
                            help="Type of result to add interactively")
    
    # Common test metadata
    parser.add_argument("--test-name", type=str,
                        help="Name of the test run")
    parser.add_argument("--git-commit", type=str,
                        help="Git commit hash for the test run")
    parser.add_argument("--git-branch", type=str,
                        help="Git branch for the test run")
    parser.add_argument("--command-line", type=str,
                        help="Command line used to run the test")
    parser.add_argument("--metadata", type=str,
                        help="JSON string with additional metadata")
    
    # Performance test specific args
    parser.add_argument("--model-name", type=str,
                        help="Model name for the test")
    parser.add_argument("--model-family", type=str,
                        help="Model family (bert, t5, etc.)")
    parser.add_argument("--hardware-type", type=str,
                        help="Hardware type (cpu, cuda, etc.)")
    parser.add_argument("--device-name", type=str,
                        help="Hardware device name")
    parser.add_argument("--test-case", type=str,
                        help="Specific test case name")
    parser.add_argument("--batch-size", type=int,
                        help="Batch size used in the test")
    parser.add_argument("--precision", type=str,
                        help="Precision used in the test (fp32, fp16, etc.)")
    parser.add_argument("--latency", type=float,
                        help="Average latency in milliseconds")
    parser.add_argument("--throughput", type=float,
                        help="Throughput in items per second")
    parser.add_argument("--memory-peak", type=float,
                        help="Peak memory usage in MB")
    
    # Compatibility test specific args
    parser.add_argument("--is-compatible", type=str, choices=['true', 'false'],
                        help="Whether the model is compatible with the hardware")
    parser.add_argument("--error-message", type=str,
                        help="Error message if incompatible")
    parser.add_argument("--compatibility-score", type=float,
                        help="Compatibility score (0.0-1.0)")
    
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    
    return parser.parse_args()

def connect_to_db(db_path):
    """Connect to the DuckDB database"""
    if not os.path.exists(db_path):
        logger.error(f"Database file not found: {db_path}")
        sys.exit(1)
        
    try:
        conn = duckdb.connect(db_path)
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        sys.exit(1)

def check_db_schema(conn):
    """Check if the database has the required schema"""
    tables = conn.execute("SHOW TABLES").fetchall()
    table_names = [t[0].lower() for t in tables]
    
    required_tables = ['hardware_platforms', 'models', 'test_runs', 
                       'performance_results', 'hardware_compatibility']
    
    missing_tables = [t for t in required_tables if t.lower() not in table_names]
    
    if missing_tables:
        logger.error(f"Required tables missing from database: {', '.join(missing_tables)}")
        logger.error("Please run create_benchmark_schema.py to initialize the database schema")
        sys.exit(1)

def find_or_create_model(conn, model_name, model_family=None, modality=None, 
                         source=None, version=None, parameters_million=None, metadata=None):
    """Find a model in the database or create it if it doesn't exist"""
    if not model_name:
        return None
    
    # Check if model exists
    existing_model = conn.execute("""
    SELECT model_id FROM models WHERE model_name = ?
    """, [model_name]).fetchone()
    
    if existing_model:
        return existing_model[0]
    
    # Try to extract model family from name if not provided
    if not model_family and model_name:
        if 'bert' in model_name.lower():
            model_family = 'bert'
        elif 't5' in model_name.lower():
            model_family = 't5'
        elif 'gpt' in model_name.lower():
            model_family = 'gpt'
        elif 'llama' in model_name.lower():
            model_family = 'llama'
        elif 'vit' in model_name.lower():
            model_family = 'vit'
        elif 'clip' in model_name.lower():
            model_family = 'clip'
        elif 'whisper' in model_name.lower():
            model_family = 'whisper'
        elif 'wav2vec' in model_name.lower():
            model_family = 'wav2vec'
    
    # Try to extract modality from family if not provided
    if not modality and model_family:
        if model_family in ['bert', 't5', 'gpt', 'llama']:
            modality = 'text'
        elif model_family in ['vit', 'clip']:
            modality = 'image'
        elif model_family in ['whisper', 'wav2vec']:
            modality = 'audio'
        elif model_family in ['llava']:
            modality = 'multimodal'
    
    # Create a new model
    metadata_json = json.dumps(metadata) if metadata else '{}'
    
    conn.execute("""
    INSERT INTO models (model_name, model_family, modality, source, version, parameters_million, metadata)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, [model_name, model_family, modality, source, version, parameters_million, metadata_json])
    
    # Get the inserted ID
    new_model_id = conn.execute("""
    SELECT model_id FROM models WHERE model_name = ?
    """, [model_name]).fetchone()[0]
    
    logger.info(f"Created new model entry: {model_name} (ID: {new_model_id})")
    
    return new_model_id

def find_or_create_hardware(conn, hardware_type, device_name=None, platform=None, 
                           platform_version=None, driver_version=None, memory_gb=None, 
                           compute_units=None, metadata=None):
    """Find a hardware platform in the database or create it if it doesn't exist"""
    if not hardware_type:
        return None
    
    # Build query based on available parameters
    query = "SELECT hardware_id FROM hardware_platforms WHERE hardware_type = ?"
    params = [hardware_type]
    
    if device_name:
        query += " AND device_name = ?"
        params.append(device_name)
    
    # Check if hardware exists
    existing_hardware = conn.execute(query, params).fetchone()
    
    if existing_hardware:
        return existing_hardware[0]
    
    # Create a new hardware entry
    metadata_json = json.dumps(metadata) if metadata else '{}'
    
    conn.execute("""
    INSERT INTO hardware_platforms (hardware_type, device_name, platform, platform_version, 
                                  driver_version, memory_gb, compute_units, metadata)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, [hardware_type, device_name, platform, platform_version, 
         driver_version, memory_gb, compute_units, metadata_json])
    
    # Get the inserted ID
    query = "SELECT hardware_id FROM hardware_platforms WHERE hardware_type = ?"
    params = [hardware_type]
    
    if device_name:
        query += " AND device_name = ?"
        params.append(device_name)
    
    new_hardware_id = conn.execute(query, params).fetchone()[0]
    
    logger.info(f"Created new hardware entry: {hardware_type} {device_name or ''} (ID: {new_hardware_id})")
    
    return new_hardware_id

def create_test_run(conn, test_name, test_type, git_commit=None, git_branch=None, 
                   command_line=None, execution_time_seconds=None, success=True, metadata=None):
    """Create a new test run entry in the database"""
    if not test_name:
        # Generate a default test name based on timestamp
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        test_name = f"{test_type}_test_{timestamp}"
    
    # Get current time
    now = datetime.datetime.now()
    
    # Create metadata JSON
    metadata_json = json.dumps(metadata) if metadata else '{}'
    
    # Insert the test run
    conn.execute("""
    INSERT INTO test_runs (test_name, test_type, started_at, completed_at, 
                         execution_time_seconds, success, git_commit, git_branch, 
                         command_line, metadata)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [test_name, test_type, now, now, execution_time_seconds, 
         success, git_commit, git_branch, command_line, metadata_json])
    
    # Get the inserted ID
    run_id = conn.execute("""
    SELECT run_id FROM test_runs WHERE test_name = ? AND started_at = ?
    """, [test_name, now]).fetchone()[0]
    
    logger.info(f"Created new test run: {test_name} (ID: {run_id})")
    
    return run_id

def add_performance_result(conn, run_id, model_id, hardware_id, test_case, batch_size=1, 
                          precision=None, total_time_seconds=None, average_latency_ms=None,
                          throughput_items_per_second=None, memory_peak_mb=None,
                          iterations=None, warmup_iterations=None, metrics=None):
    """Add a performance result to the database"""
    # Create metrics JSON
    metrics_json = json.dumps(metrics) if metrics else '{}'
    
    # Insert the performance result
    conn.execute("""
    INSERT INTO performance_results (run_id, model_id, hardware_id, test_case, batch_size,
                                   precision, total_time_seconds, average_latency_ms,
                                   throughput_items_per_second, memory_peak_mb,
                                   iterations, warmup_iterations, metrics)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [run_id, model_id, hardware_id, test_case, batch_size,
         precision, total_time_seconds, average_latency_ms,
         throughput_items_per_second, memory_peak_mb,
         iterations, warmup_iterations, metrics_json])
    
    # Get the inserted ID
    result_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    
    logger.info(f"Added performance result for model {model_id} on hardware {hardware_id} (ID: {result_id})")
    
    return result_id

def add_compatibility_result(conn, run_id, model_id, hardware_id, is_compatible, 
                            detection_success=True, initialization_success=None,
                            error_message=None, error_type=None, suggested_fix=None,
                            workaround_available=None, compatibility_score=None, metadata=None):
    """Add a hardware compatibility result to the database"""
    # Default initialization_success to is_compatible if not provided
    if initialization_success is None:
        initialization_success = is_compatible
    
    # Default workaround_available based on suggested_fix if not provided
    if workaround_available is None:
        workaround_available = bool(suggested_fix)
    
    # Default compatibility_score based on is_compatible if not provided
    if compatibility_score is None:
        compatibility_score = 1.0 if is_compatible else 0.0
    
    # Create metadata JSON
    metadata_json = json.dumps(metadata) if metadata else '{}'
    
    # Insert the compatibility result
    conn.execute("""
    INSERT INTO hardware_compatibility (run_id, model_id, hardware_id, is_compatible,
                                      detection_success, initialization_success,
                                      error_message, error_type, suggested_fix,
                                      workaround_available, compatibility_score, metadata)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [run_id, model_id, hardware_id, is_compatible,
         detection_success, initialization_success,
         error_message, error_type, suggested_fix,
         workaround_available, compatibility_score, metadata_json])
    
    # Get the inserted ID
    compatibility_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    
    logger.info(f"Added compatibility result for model {model_id} on hardware {hardware_id} (ID: {compatibility_id})")
    
    return compatibility_id

def add_integration_test_result(conn, run_id, test_module, test_name, status, 
                              model_id=None, hardware_id=None, test_class=None,
                              execution_time_seconds=None, error_message=None, 
                              error_traceback=None, metadata=None):
    """Add an integration test result to the database"""
    # Create metadata JSON
    metadata_json = json.dumps(metadata) if metadata else '{}'
    
    # Insert the test result
    conn.execute("""
    INSERT INTO integration_test_results (run_id, test_module, test_class, test_name,
                                       status, hardware_id, model_id, execution_time_seconds,
                                       error_message, error_traceback, metadata)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [run_id, test_module, test_class, test_name,
         status, hardware_id, model_id, execution_time_seconds,
         error_message, error_traceback, metadata_json])
    
    # Get the inserted ID
    test_result_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    
    logger.info(f"Added integration test result for {test_module}.{test_name} (ID: {test_result_id})")
    
    return test_result_id

def add_test_assertion(conn, test_result_id, assertion_name, passed, 
                      expected_value=None, actual_value=None, message=None):
    """Add a test assertion to the database"""
    # Insert the assertion
    conn.execute("""
    INSERT INTO integration_test_assertions (test_result_id, assertion_name, passed,
                                          expected_value, actual_value, message)
    VALUES (?, ?, ?, ?, ?, ?)
    """, [test_result_id, assertion_name, passed,
         expected_value, actual_value, message])
    
    # Get the inserted ID
    assertion_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    
    logger.debug(f"Added test assertion {assertion_name} for test result {test_result_id} (ID: {assertion_id})")
    
    return assertion_id

def process_input_file(conn, input_file):
    """Process a JSON input file and add results to the database"""
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Loaded input file: {input_file}")
        
        # Determine result type from file content
        result_type = None
        if 'throughput_items_per_second' in str(data) or 'average_latency_ms' in str(data) or 'test_case' in str(data):
            result_type = 'performance'
        elif 'is_compatible' in str(data) or 'compatibility_score' in str(data):
            result_type = 'compatibility'
        elif 'test_module' in str(data) or 'test_class' in str(data) or 'assertions' in str(data):
            result_type = 'integration'
        else:
            # Look for common keys by result type
            key_matches = {
                'performance': sum(1 for k in ['latency', 'throughput', 'memory_peak', 'batch_size'] if k in str(data)),
                'compatibility': sum(1 for k in ['compatible', 'compatibility', 'error_message'] if k in str(data)),
                'integration': sum(1 for k in ['test_name', 'status', 'assertions'] if k in str(data))
            }
            result_type = max(key_matches.items(), key=lambda x: x[1])[0]
        
        logger.info(f"Detected result type: {result_type}")
        
        # Extract common test metadata
        test_name = data.get('test_name', os.path.splitext(os.path.basename(input_file))[0])
        git_commit = data.get('git_commit')
        git_branch = data.get('git_branch')
        command_line = data.get('command_line')
        execution_time_seconds = data.get('execution_time_seconds')
        metadata = data.get('metadata', {})
        
        # Create test run
        run_id = create_test_run(conn, test_name, result_type, git_commit, git_branch,
                               command_line, execution_time_seconds, True, metadata)
        
        # Process results based on type
        if result_type == 'performance':
            process_performance_data(conn, data, run_id)
        elif result_type == 'compatibility':
            process_compatibility_data(conn, data, run_id)
        elif result_type == 'integration':
            process_integration_data(conn, data, run_id)
        
        logger.info(f"Successfully processed input file: {input_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing input file: {e}")
        return False

def process_performance_data(conn, data, run_id):
    """Process performance benchmark data and add to database"""
    # Check if results are nested or flat
    if 'results' in data and isinstance(data['results'], list):
        # Process multiple results
        for result in data['results']:
            process_single_performance_result(conn, result, run_id)
    elif 'results' in data and isinstance(data['results'], dict):
        # Process dictionary of model -> result mapping
        for model_name, result in data['results'].items():
            if isinstance(result, dict):
                result['model_name'] = model_name
                process_single_performance_result(conn, result, run_id)
    elif 'model_results' in data:
        # Process model-centric format
        for model_result in data['model_results']:
            process_single_performance_result(conn, model_result, run_id)
    elif 'benchmarks' in data:
        # Process benchmark-centric format
        for benchmark in data['benchmarks']:
            process_single_performance_result(conn, benchmark, run_id)
    else:
        # Try to process the entire data as a single result
        process_single_performance_result(conn, data, run_id)

def process_single_performance_result(conn, result, run_id):
    """Process a single performance result entry"""
    if not result or not isinstance(result, dict):
        return
    
    # Extract model information
    model_name = result.get('model_name', result.get('model'))
    if not model_name:
        logger.warning("Skipping performance result with no model name")
        return
    
    model_family = result.get('model_family')
    modality = result.get('modality')
    source = result.get('source')
    version = result.get('version')
    parameters_million = result.get('parameters_million')
    model_metadata = result.get('model_metadata', {})
    
    # Extract hardware information
    hardware_type = result.get('hardware_type', result.get('hardware'))
    if not hardware_type and 'hardware' in result and isinstance(result['hardware'], dict):
        hardware_type = result['hardware'].get('type')
    
    device_name = result.get('device_name')
    if not device_name and 'hardware' in result and isinstance(result['hardware'], dict):
        device_name = result['hardware'].get('name')
    
    platform = result.get('platform')
    platform_version = result.get('platform_version')
    driver_version = result.get('driver_version')
    memory_gb = result.get('memory_gb')
    compute_units = result.get('compute_units')
    hardware_metadata = result.get('hardware_metadata', {})
    
    # Extract performance metrics
    test_case = result.get('test_case', result.get('benchmark_type', 'default'))
    batch_size = result.get('batch_size', 1)
    precision = result.get('precision')
    total_time_seconds = result.get('total_time_seconds', result.get('total_time'))
    average_latency_ms = result.get('average_latency_ms', result.get('latency_ms'))
    throughput_items_per_second = result.get('throughput_items_per_second', result.get('throughput'))
    memory_peak_mb = result.get('memory_peak_mb', result.get('memory_mb'))
    iterations = result.get('iterations')
    warmup_iterations = result.get('warmup_iterations')
    metrics = result.get('metrics', {})
    
    # Find or create model and hardware entries
    model_id = find_or_create_model(conn, model_name, model_family, modality,
                                  source, version, parameters_million, model_metadata)
    
    hardware_id = find_or_create_hardware(conn, hardware_type, device_name, platform,
                                        platform_version, driver_version, memory_gb,
                                        compute_units, hardware_metadata)
    
    if not model_id or not hardware_id:
        logger.warning(f"Skipping performance result: missing model_id or hardware_id")
        return
    
    # Add performance result
    add_performance_result(conn, run_id, model_id, hardware_id, test_case, batch_size,
                          precision, total_time_seconds, average_latency_ms,
                          throughput_items_per_second, memory_peak_mb,
                          iterations, warmup_iterations, metrics)

def process_compatibility_data(conn, data, run_id):
    """Process hardware compatibility data and add to database"""
    # Check if results are nested or flat
    if 'compatibility_results' in data and isinstance(data['compatibility_results'], list):
        # Process multiple results
        for result in data['compatibility_results']:
            process_single_compatibility_result(conn, result, run_id)
    elif 'model_compatibility' in data and isinstance(data['model_compatibility'], list):
        # Process model-centric format
        for model_result in data['model_compatibility']:
            process_single_compatibility_result(conn, model_result, run_id)
    elif 'hardware_compatibility' in data and isinstance(data['hardware_compatibility'], list):
        # Process hardware-centric format
        for hw_result in data['hardware_compatibility']:
            process_single_compatibility_result(conn, hw_result, run_id)
    elif 'results' in data and isinstance(data['results'], list):
        # Process generic results array
        for result in data['results']:
            process_single_compatibility_result(conn, result, run_id)
    elif 'models' in data and isinstance(data['models'], list):
        # Process models-centric format with nested hardware compatibility
        for model_entry in data['models']:
            model_name = model_entry.get('name')
            if not model_name:
                continue
                
            if 'hardware_compatibility' in model_entry and isinstance(model_entry['hardware_compatibility'], list):
                for hw_compat in model_entry['hardware_compatibility']:
                    hw_compat['model_name'] = model_name
                    process_single_compatibility_result(conn, hw_compat, run_id)
    else:
        # Try to process the entire data as a single result
        process_single_compatibility_result(conn, data, run_id)

def process_single_compatibility_result(conn, result, run_id):
    """Process a single hardware compatibility result entry"""
    if not result or not isinstance(result, dict):
        return
    
    # Extract model information
    model_name = result.get('model_name', result.get('model'))
    if not model_name:
        logger.warning("Skipping compatibility result with no model name")
        return
    
    model_family = result.get('model_family')
    modality = result.get('modality')
    source = result.get('source')
    version = result.get('version')
    parameters_million = result.get('parameters_million')
    model_metadata = result.get('model_metadata', {})
    
    # Extract hardware information
    hardware_type = result.get('hardware_type', result.get('hardware'))
    if not hardware_type and 'hardware' in result and isinstance(result['hardware'], dict):
        hardware_type = result['hardware'].get('type')
    
    device_name = result.get('device_name')
    if not device_name and 'hardware' in result and isinstance(result['hardware'], dict):
        device_name = result['hardware'].get('name')
    
    platform = result.get('platform')
    platform_version = result.get('platform_version')
    driver_version = result.get('driver_version')
    memory_gb = result.get('memory_gb')
    compute_units = result.get('compute_units')
    hardware_metadata = result.get('hardware_metadata', {})
    
    # Extract compatibility information
    is_compatible = result.get('is_compatible')
    if is_compatible is None and 'compatibility' in result:
        # Some files use 'compatibility': true/false instead
        is_compatible = result['compatibility']
    
    if is_compatible is None and 'error' in result:
        # If there's an error entry, assume it's not compatible
        is_compatible = False
    
    # If still None, default to True if no error message
    if is_compatible is None:
        is_compatible = not bool(result.get('error_message', result.get('error')))
    
    detection_success = result.get('detection_success', True)
    initialization_success = result.get('initialization_success', is_compatible)
    error_message = result.get('error_message', result.get('error'))
    error_type = result.get('error_type')
    suggested_fix = result.get('suggested_fix', result.get('workaround'))
    workaround_available = result.get('workaround_available', bool(suggested_fix))
    compatibility_score = result.get('compatibility_score', 1.0 if is_compatible else 0.0)
    compatibility_metadata = result.get('compatibility_metadata', {})
    
    # Find or create model and hardware entries
    model_id = find_or_create_model(conn, model_name, model_family, modality,
                                  source, version, parameters_million, model_metadata)
    
    hardware_id = find_or_create_hardware(conn, hardware_type, device_name, platform,
                                        platform_version, driver_version, memory_gb,
                                        compute_units, hardware_metadata)
    
    if not model_id or not hardware_id:
        logger.warning(f"Skipping compatibility result: missing model_id or hardware_id")
        return
    
    # Add compatibility result
    add_compatibility_result(conn, run_id, model_id, hardware_id, is_compatible,
                           detection_success, initialization_success, error_message,
                           error_type, suggested_fix, workaround_available,
                           compatibility_score, compatibility_metadata)

def process_integration_data(conn, data, run_id):
    """Process integration test data and add to database"""
    # Check if results are nested or flat
    if 'test_results' in data and isinstance(data['test_results'], list):
        # Process multiple results
        for result in data['test_results']:
            process_single_integration_result(conn, result, run_id)
    elif 'modules' in data and isinstance(data['modules'], dict):
        # Process module-centric format
        for module_name, module_results in data['modules'].items():
            if isinstance(module_results, list):
                for result in module_results:
                    result['test_module'] = module_name
                    process_single_integration_result(conn, result, run_id)
            elif isinstance(module_results, dict):
                for test_name, test_result in module_results.items():
                    if isinstance(test_result, dict):
                        test_result['test_module'] = module_name
                        test_result['test_name'] = test_name
                        process_single_integration_result(conn, test_result, run_id)
    elif 'results' in data and isinstance(data['results'], list):
        # Process generic results array
        for result in data['results']:
            process_single_integration_result(conn, result, run_id)
    else:
        # Try to process the entire data as a single result
        process_single_integration_result(conn, data, run_id)

def process_single_integration_result(conn, result, run_id):
    """Process a single integration test result entry"""
    if not result or not isinstance(result, dict):
        return
    
    # Extract test information
    test_module = result.get('test_module', result.get('module'))
    test_class = result.get('test_class', result.get('class'))
    test_name = result.get('test_name', result.get('name'))
    
    if not test_name and not test_module:
        logger.warning("Skipping integration test result with no test name or module")
        return
    
    # Extract status information
    status = result.get('status', result.get('result'))
    if status is None:
        if result.get('passed', False):
            status = 'pass'
        elif result.get('failed', False):
            status = 'fail'
        elif result.get('error', False):
            status = 'error'
        elif result.get('skipped', False):
            status = 'skip'
        else:
            status = 'unknown'
    
    execution_time_seconds = result.get('execution_time_seconds', 
                                      result.get('time', result.get('duration')))
    error_message = result.get('error_message', result.get('error'))
    error_traceback = result.get('error_traceback', result.get('traceback'))
    test_metadata = result.get('metadata', {})
    
    # Extract hardware information if present
    hardware_type = result.get('hardware_type', result.get('hardware'))
    if not hardware_type and 'hardware' in result and isinstance(result['hardware'], dict):
        hardware_type = result['hardware'].get('type')
    
    device_name = result.get('device_name')
    if not device_name and 'hardware' in result and isinstance(result['hardware'], dict):
        device_name = result['hardware'].get('name')
    
    # Extract model information if present
    model_name = result.get('model_name', result.get('model'))
    model_family = result.get('model_family')
    
    # Find or create model and hardware entries if needed
    model_id = None
    hardware_id = None
    
    if model_name:
        model_id = find_or_create_model(conn, model_name, model_family)
    
    if hardware_type:
        hardware_id = find_or_create_hardware(conn, hardware_type, device_name)
    
    # Add integration test result
    test_result_id = add_integration_test_result(conn, run_id, test_module, test_name,
                                              status, model_id, hardware_id, test_class,
                                              execution_time_seconds, error_message,
                                              error_traceback, test_metadata)
    
    # Process assertions if present
    if 'assertions' in result and isinstance(result['assertions'], list):
        for assertion in result['assertions']:
            process_assertion(conn, assertion, test_result_id)
    elif 'tests' in result and isinstance(result['tests'], list):
        for test in result['tests']:
            process_assertion(conn, test, test_result_id)

def process_assertion(conn, assertion, test_result_id):
    """Process a single test assertion entry"""
    if not assertion or not isinstance(assertion, dict):
        return
    
    # Extract assertion information
    assertion_name = assertion.get('name', assertion.get('assertion_name', 'unnamed_assertion'))
    passed = assertion.get('passed', assertion.get('success'))
    
    if passed is None:
        passed = not bool(assertion.get('error'))
    
    expected_value = str(assertion.get('expected', assertion.get('expected_value', '')))
    actual_value = str(assertion.get('actual', assertion.get('actual_value', '')))
    message = assertion.get('message')
    
    # Add assertion
    add_test_assertion(conn, test_result_id, assertion_name, passed,
                      expected_value, actual_value, message)

def process_args_directly(conn, args):
    """Process command line arguments directly into database entries"""
    # Determine result type
    result_type = args.result_type
    
    # Check required parameters based on result type
    if result_type == 'performance':
        if not args.model_name or not args.hardware_type:
            logger.error("Performance results require --model-name and --hardware-type")
            return False
    elif result_type == 'compatibility':
        if not args.model_name or not args.hardware_type or args.is_compatible is None:
            logger.error("Compatibility results require --model-name, --hardware-type, and --is-compatible")
            return False
    elif result_type == 'integration':
        if not args.test_name:
            logger.error("Integration test results require --test-name")
            return False
    
    # Parse metadata if provided
    metadata = {}
    if args.metadata:
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError:
            logger.warning("Invalid metadata JSON, using empty metadata")
    
    # Create test run
    run_id = create_test_run(conn, args.test_name, result_type, args.git_commit, 
                           args.git_branch, args.command_line, None, True, metadata)
    
    if result_type == 'performance':
        # Find or create model and hardware
        model_id = find_or_create_model(conn, args.model_name, args.model_family)
        hardware_id = find_or_create_hardware(conn, args.hardware_type, args.device_name)
        
        # Add performance result
        add_performance_result(conn, run_id, model_id, hardware_id, 
                              args.test_case or 'default', args.batch_size or 1,
                              args.precision, None, args.latency,
                              args.throughput, args.memory_peak, None, None, {})
        
    elif result_type == 'compatibility':
        # Find or create model and hardware
        model_id = find_or_create_model(conn, args.model_name, args.model_family)
        hardware_id = find_or_create_hardware(conn, args.hardware_type, args.device_name)
        
        # Parse is_compatible
        is_compatible = args.is_compatible.lower() == 'true'
        
        # Add compatibility result
        add_compatibility_result(conn, run_id, model_id, hardware_id, is_compatible,
                               True, None, args.error_message, None, None, None,
                               args.compatibility_score, {})
        
    elif result_type == 'integration':
        # Add integration test result
        add_integration_test_result(conn, run_id, args.test_name, args.test_name,
                                  'pass', None, None, None, None, None, None, {})
    
    logger.info(f"Successfully added {result_type} result")
    return True

def main():
    args = parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Connect to the database
    conn = connect_to_db(args.db)
    
    # Check if the database has the required schema
    check_db_schema(conn)
    
    # Process input based on method
    if args.input_file:
        success = process_input_file(conn, args.input_file)
    else:
        success = process_args_directly(conn, args)
    
    # Commit changes if successful
    if success:
        conn.commit()
        logger.info("Changes committed to database")
    else:
        logger.error("Failed to process results, no changes committed")
    
    # Close the database connection
    conn.close()

if __name__ == "__main__":
    main()