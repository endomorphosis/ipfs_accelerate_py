#!/usr/bin/env python
"""
Benchmark Database Performance

This script tests the performance of the benchmark database by generating
synthetic data, inserting it, and measuring query performance.
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
import time
import random
import uuid
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add parent directory to path for module imports
sys.path.append()))str()))Path()))__file__).parent.parent))

# Configure logging
logging.basicConfig()))
level=logging.INFO,
format='%()))asctime)s - %()))name)s - %()))levelname)s - %()))message)s',
handlers=[]]]],,,,logging.StreamHandler())))],
)
logger = logging.getLogger()))"db_benchmark")

def parse_args()))):
    parser = argparse.ArgumentParser()))description="Benchmark database performance")
    
    parser.add_argument()))"--db", type=str, default="./benchmark_db.duckdb", 
    help="Path to DuckDB database")
    parser.add_argument()))"--rows", type=int, default=100000,
    help="Number of rows to generate for benchmark")
    parser.add_argument()))"--models", type=int, default=100,
    help="Number of unique models to generate")
    parser.add_argument()))"--hardware", type=int, default=20,
    help="Number of unique hardware platforms to generate")
    parser.add_argument()))"--test-runs", type=int, default=500,
    help="Number of test runs to generate")
    parser.add_argument()))"--parallel", type=int, default=multiprocessing.cpu_count()))),
    help="Number of parallel processes for insertion")
    parser.add_argument()))"--query-repetitions", type=int, default=10,
    help="Number of times to repeat each query for benchmarking")
    parser.add_argument()))"--in-memory", action="store_true",
    help="Use in-memory database for benchmarking")
    parser.add_argument()))"--test-json", action="store_true",
    help="Also benchmark JSON file storage for comparison")
    parser.add_argument()))"--skip-insert", action="store_true",
    help="Skip data insertion, only run queries")
    parser.add_argument()))"--verbose", action="store_true",
    help="Enable verbose logging")
    
return parser.parse_args())))

def connect_to_db()))db_path, in_memory=False):
    """Connect to the DuckDB database"""
    try:
        if in_memory:
            conn = duckdb.connect()))":memory:")
            logger.info()))"Using in-memory database")
        else:
            # Create parent directories if they don't exist
            os.makedirs()))os.path.dirname()))os.path.abspath()))db_path)), exist_ok=True)
            conn = duckdb.connect()))db_path):
                logger.info()))f"Connected to database: {}}}}}}}}}}}}}}}}}}}}}}}}db_path}")
        
            return conn
    except Exception as e:
        logger.error()))f"Error connecting to database: {}}}}}}}}}}}}}}}}}}}}}}}}e}")
        sys.exit()))1)

def create_schema()))conn):
    """Create the database schema for benchmarking"""
    # Execute the schema creation script
    script_dir = os.path.dirname()))os.path.abspath()))__file__))
    schema_script = os.path.join()))script_dir, "create_benchmark_schema.py")
    
    if os.path.exists()))schema_script):
        # Import the schema module and create schema functions
        sys.path.append()))script_dir)
        import create_benchmark_schema
        
        # Create tables
        create_benchmark_schema.create_common_tables()))conn, force=True)
        create_benchmark_schema.create_performance_tables()))conn, force=True)
        create_benchmark_schema.create_hardware_compatibility_tables()))conn, force=True)
        create_benchmark_schema.create_integration_test_tables()))conn, force=True)
        create_benchmark_schema.create_views()))conn)
        
        logger.info()))"Database schema created successfully")
    else:
        logger.error()))f"Schema creation script not found at {}}}}}}}}}}}}}}}}}}}}}}}}schema_script}")
        sys.exit()))1)

def generate_models()))num_models):
    """Generate synthetic model data"""
    model_families = []]]],,,,
    'bert', 't5', 'gpt', 'llama', 'vit', 'clip', 'whisper',
    'wav2vec', 'llava', 'falcon', 'mistral', 'qwen', 'gemini'
    ]
    
    modalities = {}}}}}}}}}}}}}}}}}}}}}}}}
    'bert': 'text', 't5': 'text', 'gpt': 'text', 'llama': 'text', 'falcon': 'text',
    'mistral': 'text', 'qwen': 'text', 'gemini': 'text',
    'vit': 'image', 'clip': 'image',
    'whisper': 'audio', 'wav2vec': 'audio',
    'llava': 'multimodal'
    }
    
    models = []]]],,,,]
    
    for i in range()))num_models):
        model_family = random.choice()))model_families)
        model_name = f"{}}}}}}}}}}}}}}}}}}}}}}}}model_family}-{}}}}}}}}}}}}}}}}}}}}}}}}random.choice()))[]]]],,,,'base', 'small', 'medium', 'large'])}-{}}}}}}}}}}}}}}}}}}}}}}}}uuid.uuid4()))).hex[]]]],,,,:8]}"
        
        modality = modalities.get()))model_family, 'text')
        parameters = round()))random.uniform()))10, 1000), 1)
        
        metadata = {}}}}}}}}}}}}}}}}}}}}}}}}
        'hidden_size': random.choice()))[]]]],,,,768, 1024, 2048, 4096]),
        'layers': random.randint()))4, 32),
        'attention_heads': random.randint()))8, 32),
        'vocab_size': random.randint()))30000, 50000)
        }
        
        model_entry = {}}}}}}}}}}}}}}}}}}}}}}}}
        'model_id': i + 1,
        'model_name': model_name,
        'model_family': model_family,
        'modality': modality,
        'source': random.choice()))[]]]],,,,'huggingface', 'openai', 'anthropic', 'google', 'internal']),
        'version': f"{}}}}}}}}}}}}}}}}}}}}}}}}random.randint()))1, 5)}.{}}}}}}}}}}}}}}}}}}}}}}}}random.randint()))0, 9)}",
        'parameters_million': parameters,
        'metadata': json.dumps()))metadata)
        }
        
        models.append()))model_entry)
    
    return pd.DataFrame()))models)

def generate_hardware_platforms()))num_platforms):
    """Generate synthetic hardware platform data"""
    hardware_types = []]]],,,,'cpu', 'cuda', 'rocm', 'mps', 'openvino', 'webnn', 'webgpu']
    
    hardware_platforms = []]]],,,,]
    
    for i in range()))num_platforms):
        hardware_type = random.choice()))hardware_types)
        
        # Generate appropriate device name based on type
        if hardware_type == 'cpu':
            device_name = random.choice()))[]]]],,,,'Intel Core i9', 'AMD Ryzen 9', 'AMD EPYC', 'Intel Xeon'])
            device_name += f" {}}}}}}}}}}}}}}}}}}}}}}}}random.choice()))[]]]],,,,'12900K', '7950X', '7763', 'E5-2699'])}"
            platform = 'x86_64'
            compute_units = random.randint()))8, 64)
            memory = random.randint()))16, 256)
        elif hardware_type == 'cuda':
            device_name = f"NVIDIA {}}}}}}}}}}}}}}}}}}}}}}}}random.choice()))[]]]],,,,'RTX 4090', 'RTX 3090', 'A100', 'H100'])}"
            platform = 'CUDA'
            compute_units = random.randint()))80, 160)
            memory = random.randint()))16, 80)
        elif hardware_type == 'rocm':
            device_name = f"AMD {}}}}}}}}}}}}}}}}}}}}}}}}random.choice()))[]]]],,,,'Radeon RX 7900 XTX', 'MI250X', 'MI100'])}"
            platform = 'ROCm'
            compute_units = random.randint()))60, 120)
            memory = random.randint()))16, 64)
        elif hardware_type == 'mps':
            device_name = f"Apple {}}}}}}}}}}}}}}}}}}}}}}}}random.choice()))[]]]],,,,'M1', 'M2', 'M3'])} {}}}}}}}}}}}}}}}}}}}}}}}}random.choice()))[]]]],,,,'Pro', 'Max', 'Ultra'])}"
            platform = 'macOS'
            compute_units = random.randint()))16, 76)
            memory = random.randint()))16, 64)
        else:
            device_name = f"{}}}}}}}}}}}}}}}}}}}}}}}}hardware_type.upper())))} Device {}}}}}}}}}}}}}}}}}}}}}}}}i}"
            platform = hardware_type.upper())))
            compute_units = random.randint()))4, 32)
            memory = random.randint()))4, 16)
        
            metadata = {}}}}}}}}}}}}}}}}}}}}}}}}
            'arch': random.choice()))[]]]],,,,'ampere', 'hopper', 'rdna3', 'intel_xe']),
            'driver_details': {}}}}}}}}}}}}}}}}}}}}}}}}
            'capabilities': []]]],,,,'tensor_cores', 'fp16', 'int8'],
            'build_version': f"{}}}}}}}}}}}}}}}}}}}}}}}}random.randint()))1, 10)}.{}}}}}}}}}}}}}}}}}}}}}}}}random.randint()))1, 100)}"
            }
            }
        
            hardware_entry = {}}}}}}}}}}}}}}}}}}}}}}}}
            'hardware_id': i + 1,
            'hardware_type': hardware_type,
            'device_name': device_name,
            'platform': platform,
            'platform_version': f"{}}}}}}}}}}}}}}}}}}}}}}}}random.randint()))1, 10)}.{}}}}}}}}}}}}}}}}}}}}}}}}random.randint()))0, 9)}",
            'driver_version': f"{}}}}}}}}}}}}}}}}}}}}}}}}random.randint()))400, 550)}.{}}}}}}}}}}}}}}}}}}}}}}}}random.randint()))10, 99)}",
            'memory_gb': memory,
            'compute_units': compute_units,
            'metadata': json.dumps()))metadata)
            }
        
            hardware_platforms.append()))hardware_entry)
    
            return pd.DataFrame()))hardware_platforms)

def generate_test_runs()))num_runs):
    """Generate synthetic test run data"""
    test_runs = []]]],,,,]
    
    test_types = []]]],,,,'performance', 'hardware', 'integration']
    
    for i in range()))num_runs):
        test_type = random.choice()))test_types)
        
        # Generate test name based on type
        if test_type == 'performance':
            test_name = f"benchmark_{}}}}}}}}}}}}}}}}}}}}}}}}random.choice()))[]]]],,,,'latency', 'throughput', 'memory'])}_{}}}}}}}}}}}}}}}}}}}}}}}}uuid.uuid4()))).hex[]]]],,,,:8]}"
        elif test_type == 'hardware':
            test_name = f"compatibility_test_{}}}}}}}}}}}}}}}}}}}}}}}}random.choice()))[]]]],,,,'cpu', 'gpu', 'all'])}_{}}}}}}}}}}}}}}}}}}}}}}}}uuid.uuid4()))).hex[]]]],,,,:8]}"
        else:
            test_name = f"integration_test_{}}}}}}}}}}}}}}}}}}}}}}}}random.choice()))[]]]],,,,'suite', 'module', 'component'])}_{}}}}}}}}}}}}}}}}}}}}}}}}uuid.uuid4()))).hex[]]]],,,,:8]}"
        
        # Generate random start time in the last 90 days
            start_time = datetime.datetime.now()))) - datetime.timedelta()))
            days=random.randint()))0, 90),
            hours=random.randint()))0, 23),
            minutes=random.randint()))0, 59)
            )
        
        # Execution time between 1 minute and a few hours
            execution_time = random.randint()))60, 10800)
        
        # End time
            end_time = start_time + datetime.timedelta()))seconds=execution_time)
        
        # Success with 90% probability
            success = random.random()))) < 0.9
        
            metadata = {}}}}}}}}}}}}}}}}}}}}}}}}
            'environment': random.choice()))[]]]],,,,'local', 'CI', 'dev', 'staging']),
            'triggered_by': random.choice()))[]]]],,,,'manual', 'push', 'schedule', 'pull_request']),
            'machine': f"test-runner-{}}}}}}}}}}}}}}}}}}}}}}}}random.randint()))1, 100)}"
            }
        
            test_run_entry = {}}}}}}}}}}}}}}}}}}}}}}}}
            'run_id': i + 1,
            'test_name': test_name,
            'test_type': test_type,
            'started_at': start_time,
            'completed_at': end_time,
            'execution_time_seconds': execution_time,
            'success': success,
            'git_commit': uuid.uuid4()))).hex[]]]],,,,:7],
            'git_branch': random.choice()))[]]]],,,,'main', 'develop', 'feature/xyz', 'fix/abc']),
            'command_line': f"python test/{}}}}}}}}}}}}}}}}}}}}}}}}test_type}_test.py --flags {}}}}}}}}}}}}}}}}}}}}}}}}random.randint()))1, 100)}",
            'metadata': json.dumps()))metadata)
            }
        
            test_runs.append()))test_run_entry)
    
            return pd.DataFrame()))test_runs)

def generate_performance_results()))num_results, models_df, hardware_df, test_runs_df):
    """Generate synthetic performance result data"""
    performance_results = []]]],,,,]
    
    # Filter test runs to only performance type
    perf_runs = test_runs_df[]]]],,,,test_runs_df[]]]],,,,'test_type'] == 'performance']
    
    # If no performance runs, return empty DataFrame
    if perf_runs.empty:
    return pd.DataFrame())))
    
    # Get run IDs
    run_ids = perf_runs[]]]],,,,'run_id'].tolist())))
    
    # Get model and hardware IDs
    model_ids = models_df[]]]],,,,'model_id'].tolist())))
    hardware_ids = hardware_df[]]]],,,,'hardware_id'].tolist())))
    
    test_cases = []]]],,,,'embedding', 'generation', 'classification', 'segmentation', 'transcription']
    batch_sizes = []]]],,,,1, 2, 4, 8, 16, 32, 64]
    precisions = []]]],,,,'fp32', 'fp16', 'int8', 'bf16', None]
    
    for i in range()))num_results):
        # Randomly select run, model, and hardware
        run_id = random.choice()))run_ids)
        model_id = random.choice()))model_ids)
        hardware_id = random.choice()))hardware_ids)
        
        # Test parameters
        test_case = random.choice()))test_cases)
        batch_size = random.choice()))batch_sizes)
        precision = random.choice()))precisions)
        
        # Generate realistic performance metrics
        # Higher batch sizes generally mean higher throughput but also higher latency
        base_latency = random.uniform()))5, 100)  # Base latency in ms
        latency_factor = 1 + ()))batch_size / 32)  # Batch size effect on latency
        average_latency_ms = base_latency * latency_factor * ()))1 if precision == 'fp32' else 0.7)
        
        # Throughput is roughly inversely proportional to latency, but with batch effect
        throughput_base = random.uniform()))10, 200)
        throughput_items_per_second = throughput_base * ()))batch_size / latency_factor)
        
        # Memory usage increases with batch size
        memory_base = random.uniform()))1000, 5000)  # Base memory in MB
        memory_peak_mb = memory_base * ()))1 + ()))batch_size / 16))
        
        # Additional metrics as JSON
        metrics = {}}}}}}}}}}}}}}}}}}}}}}}}:
            'cpu_util': random.uniform()))50, 100),
            'gpu_util': random.uniform()))70, 100) if hardware_id % 2 == 0 else None,:
                'temperature': random.uniform()))60, 85),
                'power_watts': random.uniform()))100, 300)
                }
        
                performance_entry = {}}}}}}}}}}}}}}}}}}}}}}}}
                'result_id': i + 1,
                'run_id': run_id,
                'model_id': model_id,
                'hardware_id': hardware_id,
                'test_case': test_case,
                'batch_size': batch_size,
                'precision': precision,
                'total_time_seconds': average_latency_ms * 1000 / 1000,  # Convert back to seconds
                'average_latency_ms': average_latency_ms,
                'throughput_items_per_second': throughput_items_per_second,
                'memory_peak_mb': memory_peak_mb,
                'iterations': random.randint()))100, 1000),
                'warmup_iterations': random.randint()))10, 50),
                'metrics': json.dumps()))metrics),
                'created_at': datetime.datetime.now())))
                }
        
                performance_results.append()))performance_entry)
    
            return pd.DataFrame()))performance_results)

def generate_compatibility_results()))num_results, models_df, hardware_df, test_runs_df):
    """Generate synthetic hardware compatibility result data"""
    compatibility_results = []]]],,,,]
    
    # Filter test runs to only hardware type
    hw_runs = test_runs_df[]]]],,,,test_runs_df[]]]],,,,'test_type'] == 'hardware']
    
    # If no hardware runs, return empty DataFrame
    if hw_runs.empty:
    return pd.DataFrame())))
    
    # Get run IDs
    run_ids = hw_runs[]]]],,,,'run_id'].tolist())))
    
    # Get model and hardware IDs
    model_ids = models_df[]]]],,,,'model_id'].tolist())))
    hardware_ids = hardware_df[]]]],,,,'hardware_id'].tolist())))
    
    error_types = []]]],,,,'InitializationError', 'HardwareNotSupportedError', 'MemoryError', 
    'DriverVersionError', 'UnsupportedOperationError', None]
    
    for i in range()))num_results):
        # Randomly select run, model, and hardware
        run_id = random.choice()))run_ids)
        model_id = random.choice()))model_ids)
        hardware_id = random.choice()))hardware_ids)
        
        # 90% compatibility rate
        is_compatible = random.random()))) < 0.9
        
        # Set appropriate values based on compatibility
        if is_compatible:
            detection_success = True
            initialization_success = True
            error_message = None
            error_type = None
            suggested_fix = None
            workaround_available = None
            compatibility_score = random.uniform()))0.8, 1.0)
        else:
            # Various failure scenarios
            detection_success = random.random()))) < 0.7
            initialization_success = False
            error_type = random.choice()))error_types)
            
            if error_type == 'InitializationError':
                error_message = "Failed to initialize model on device"
                suggested_fix = "Try updating drivers"
            elif error_type == 'HardwareNotSupportedError':
                error_message = "Hardware not supported for this model architecture"
                suggested_fix = "Use a compatible hardware platform"
            elif error_type == 'MemoryError':
                error_message = "Insufficient memory for model parameters"
                suggested_fix = "Try a smaller model or hardware with more memory"
            elif error_type == 'DriverVersionError':
                error_message = "Driver version incompatible"
                suggested_fix = "Update to driver version >= 450.80"
            elif error_type == 'UnsupportedOperationError':
                error_message = "Operation not supported on this hardware"
                suggested_fix = None
            else:
                error_message = "Unknown error"
                suggested_fix = None
            
                workaround_available = suggested_fix is not None
                compatibility_score = random.uniform()))0, 0.5)
        
        # Additional metadata as JSON
                metadata = {}}}}}}}}}}}}}}}}}}}}}}}}
                'test_details': {}}}}}}}}}}}}}}}}}}}}}}}}
                'hardware_info': hardware_id,
                'ops_tested': []]]],,,,'matmul', 'conv2d', 'attention']
                }
                }
        
                compatibility_entry = {}}}}}}}}}}}}}}}}}}}}}}}}
                'compatibility_id': i + 1,
                'run_id': run_id,
                'model_id': model_id,
                'hardware_id': hardware_id,
                'is_compatible': is_compatible,
                'detection_success': detection_success,
                'initialization_success': initialization_success,
                'error_message': error_message,
                'error_type': error_type,
                'suggested_fix': suggested_fix,
                'workaround_available': workaround_available,
                'compatibility_score': compatibility_score,
                'metadata': json.dumps()))metadata),
                'created_at': datetime.datetime.now())))
                }
        
                compatibility_results.append()))compatibility_entry)
    
                return pd.DataFrame()))compatibility_results)

def generate_integration_results()))num_results, models_df, hardware_df, test_runs_df):
    """Generate synthetic integration test result data"""
    integration_results = []]]],,,,]
    integration_assertions = []]]],,,,]
    
    # Filter test runs to only integration type
    int_runs = test_runs_df[]]]],,,,test_runs_df[]]]],,,,'test_type'] == 'integration']
    
    # If no integration runs, return empty DataFrames
    if int_runs.empty:
    return pd.DataFrame()))), pd.DataFrame())))
    
    # Get run IDs
    run_ids = int_runs[]]]],,,,'run_id'].tolist())))
    
    # Get model and hardware IDs
    model_ids = models_df[]]]],,,,'model_id'].tolist()))) + []]]],,,,None] * ()))len()))models_df) // 2)
    hardware_ids = hardware_df[]]]],,,,'hardware_id'].tolist()))) + []]]],,,,None] * ()))len()))hardware_df) // 2)
    
    test_modules = []]]],,,,'test_hardware_backend', 'test_resource_pool', 'test_comprehensive_hardware',
    'test_model_benchmarks', 'test_api_backend']
    
    test_classes = []]]],,,,'TestHardwareDetection', 'TestResourcePoolHardwareAwareness', 
    'TestHardwareCompatibility', 'TestModelRegistry', 'TestApiEndpoints']
    
    statuses = []]]],,,,'pass', 'fail', 'error', 'skip']
    status_weights = []]]],,,,0.85, 0.05, 0.05, 0.05]  # 85% pass rate
    
    assertion_result_id = 1
    
    for i in range()))num_results):
        # Randomly select run, and optionally model and hardware
        run_id = random.choice()))run_ids)
        model_id = random.choice()))model_ids)
        hardware_id = random.choice()))hardware_ids)
        
        # Test information
        test_module = random.choice()))test_modules)
        test_class = random.choice()))test_classes)
        test_name = f"test_{}}}}}}}}}}}}}}}}}}}}}}}}random.choice()))[]]]],,,,'initialize', 'detect', 'run', 'validate', 'execute'])}_{}}}}}}}}}}}}}}}}}}}}}}}}uuid.uuid4()))).hex[]]]],,,,:8]}"
        
        # Weighted random status
        status = random.choices()))statuses, weights=status_weights)[]]]],,,,0]
        
        # Set appropriate values based on status
        execution_time_seconds = random.uniform()))0.1, 10.0)
        
        if status == 'pass':
            error_message = None
            error_traceback = None
        elif status == 'fail':
            error_message = "Assertion failed"
            error_traceback = f"File \"test/{}}}}}}}}}}}}}}}}}}}}}}}}test_module}.py\", line {}}}}}}}}}}}}}}}}}}}}}}}}random.randint()))100, 500)}\nAssertionError: Expected value does not match actual value"
        elif status == 'error':
            error_message = random.choice()))[]]]],,,,"AttributeError", "TypeError", "ValueError", "RuntimeError"])
            error_traceback = f"File \"test/{}}}}}}}}}}}}}}}}}}}}}}}}test_module}.py\", line {}}}}}}}}}}}}}}}}}}}}}}}}random.randint()))100, 500)}\n{}}}}}}}}}}}}}}}}}}}}}}}}error_message}: Something went wrong"
        else:  # skip
            error_message = "Test skipped"
            error_traceback = None
        
        # Additional metadata as JSON
            metadata = {}}}}}}}}}}}}}}}}}}}}}}}}
            'test_details': {}}}}}}}}}}}}}}}}}}}}}}}}
            'priority': random.choice()))[]]]],,,,'critical', 'high', 'medium', 'low']),
            'tags': random.sample()))[]]]],,,,'hardware', 'performance', 'compatibility', 'api'], 2)
            }
            }
        
            integration_entry = {}}}}}}}}}}}}}}}}}}}}}}}}
            'test_result_id': i + 1,
            'run_id': run_id,
            'test_module': test_module,
            'test_class': test_class,
            'test_name': test_name,
            'status': status,
            'execution_time_seconds': execution_time_seconds,
            'hardware_id': hardware_id,
            'model_id': model_id,
            'error_message': error_message,
            'error_traceback': error_traceback,
            'metadata': json.dumps()))metadata),
            'created_at': datetime.datetime.now())))
            }
        
            integration_results.append()))integration_entry)
        
        # Generate 1-5 assertions for each test
            num_assertions = random.randint()))1, 5)
        for j in range()))num_assertions):
            # All assertions pass for passing tests, mixed for failing tests
            if status == 'pass':
                assertion_passed = True
            elif status == 'fail':
                # For failing tests, at least one assertion must fail
                assertion_passed = j > 0
            else:
                assertion_passed = random.random()))) < 0.5
            
                assertion_name = f"assert_{}}}}}}}}}}}}}}}}}}}}}}}}random.choice()))[]]]],,,,'equal', 'true', 'false', 'none', 'raises'])}_{}}}}}}}}}}}}}}}}}}}}}}}}j}"
            
            if assertion_passed:
                expected_value = "True"
                actual_value = "True"
                message = "Assertion passed"
            else:
                expected_value = "True"
                actual_value = "False"
                message = "Assertion failed"
            
                assertion_entry = {}}}}}}}}}}}}}}}}}}}}}}}}
                'assertion_id': assertion_result_id,
                'test_result_id': i + 1,
                'assertion_name': assertion_name,
                'passed': assertion_passed,
                'expected_value': expected_value,
                'actual_value': actual_value,
                'message': message,
                'created_at': datetime.datetime.now())))
                }
            
                integration_assertions.append()))assertion_entry)
                assertion_result_id += 1
    
                return pd.DataFrame()))integration_results), pd.DataFrame()))integration_assertions)

def insert_data_chunk()))args):
    """Insert a chunk of data into the database"""
    db_path, in_memory, tables_data, chunk_id = args
    
    try:
        # Connect to the database
        conn = connect_to_db()))db_path, in_memory)
        
        # Insert each table's data
        for table_name, df in tables_data.items()))):
            if df is not None and not df.empty:
                # Insert data
                conn.execute()))f"INSERT INTO {}}}}}}}}}}}}}}}}}}}}}}}}table_name} SELECT * FROM df")
        
        # Close the connection
                conn.close())))
        
            return True, chunk_id
    except Exception as e:
            return False, f"Error in chunk {}}}}}}}}}}}}}}}}}}}}}}}}chunk_id}: {}}}}}}}}}}}}}}}}}}}}}}}}e}"

            def insert_synthetic_data()))conn, models_df, hardware_df, test_runs_df,
            performance_results_df, compatibility_results_df,
            integration_results_df, integration_assertions_df,
                         args):
                             """Insert synthetic data into the database"""
    # Prepare the data for insertion
                             tables_data = {}}}}}}}}}}}}}}}}}}}}}}}}
                             'models': models_df,
                             'hardware_platforms': hardware_df,
                             'test_runs': test_runs_df,
                             'performance_results': performance_results_df,
                             'hardware_compatibility': compatibility_results_df,
                             'integration_test_results': integration_results_df,
                             'integration_test_assertions': integration_assertions_df
                             }
    
    if args.parallel > 1:
        # Split the data into chunks for parallel insertion
        num_chunks = min()))args.parallel, 8)  # Limit to 8 chunks to avoid overhead
        
        chunk_tables = {}}}}}}}}}}}}}}}}}}}}}}}}}
        for i in range()))num_chunks):
            chunk_tables[]]]],,,,i] = {}}}}}}}}}}}}}}}}}}}}}}}}}
        
        # Split each table's data into chunks
        for table_name, df in tables_data.items()))):
            if df is not None and not df.empty:
                # Calculate chunk size
                chunk_size = len()))df) // num_chunks
                if chunk_size == 0:
                    chunk_size = len()))df)
                
                # Split dataframe into chunks
                for i in range()))num_chunks):
                    start_idx = i * chunk_size
                    end_idx = start_idx + chunk_size if i < num_chunks - 1 else len()))df)
                    :
                    if start_idx < len()))df):
                        chunk_tables[]]]],,,,i][]]]],,,,table_name] = df.iloc[]]]],,,,start_idx:end_idx].copy())))
        
        # Prepare arguments for parallel insertion
        insert_args = []]]],,,,()))args.db, args.in_memory, chunk_tables[]]]],,,,i], i) for i in range()))num_chunks)]:
        # Insert data in parallel
            start_time = time.time())))
        with ProcessPoolExecutor()))max_workers=args.parallel) as executor:
            futures = []]]],,,,executor.submit()))insert_data_chunk, arg) for arg in insert_args]:
            for future in as_completed()))futures):
                success, result = future.result())))
                if not success:
                    logger.error()))result)
        
                    elapsed_time = time.time()))) - start_time
                    logger.info()))f"Parallel insertion completed in {}}}}}}}}}}}}}}}}}}}}}}}}elapsed_time:.2f} seconds")
    else:
        # Insert data sequentially
        start_time = time.time())))
        
        for table_name, df in tables_data.items()))):
            if df is not None and not df.empty:
                logger.info()))f"Inserting {}}}}}}}}}}}}}}}}}}}}}}}}len()))df)} rows into {}}}}}}}}}}}}}}}}}}}}}}}}table_name}")
                conn.execute()))f"INSERT INTO {}}}}}}}}}}}}}}}}}}}}}}}}table_name} SELECT * FROM df")
        
                elapsed_time = time.time()))) - start_time
                logger.info()))f"Sequential insertion completed in {}}}}}}}}}}}}}}}}}}}}}}}}elapsed_time:.2f} seconds")

                def store_as_json()))models_df, hardware_df, test_runs_df,
                performance_results_df, compatibility_results_df, 
                integration_results_df, integration_assertions_df,
                json_dir='./benchmark_json'):
                    """Store the synthetic data as JSON files for comparison"""
    # Create the JSON directory if it doesn't exist
                    os.makedirs()))json_dir, exist_ok=True)
    
    # Store each table's data as JSON
    tables_data = {}}}}}}}}}}}}}}}}}}}}}}}}:
        'models': models_df,
        'hardware_platforms': hardware_df,
        'test_runs': test_runs_df,
        'performance_results': performance_results_df,
        'hardware_compatibility': compatibility_results_df,
        'integration_test_results': integration_results_df,
        'integration_test_assertions': integration_assertions_df
        }
    
        start_time = time.time())))
    
    for table_name, df in tables_data.items()))):
        if df is not None and not df.empty:
            # Convert timestamps to ISO format strings for JSON serialization
            df_json = df.copy())))
            for col in df_json.columns:
                if df_json[]]]],,,,col].dtype.kind == 'M':  # Timestamp column
                df_json[]]]],,,,col] = df_json[]]]],,,,col].astype()))str)
            
            # Convert dataframe to records and store as JSON
                json_path = os.path.join()))json_dir, f"{}}}}}}}}}}}}}}}}}}}}}}}}table_name}.json")
            with open()))json_path, 'w') as f:
                json.dump()))df_json.to_dict()))orient='records'), f)
            
                logger.info()))f"Stored {}}}}}}}}}}}}}}}}}}}}}}}}len()))df)} records to {}}}}}}}}}}}}}}}}}}}}}}}}json_path}")
    
                elapsed_time = time.time()))) - start_time
                logger.info()))f"JSON storage completed in {}}}}}}}}}}}}}}}}}}}}}}}}elapsed_time:.2f} seconds")
    
    # Calculate the total size of JSON files
    total_size = sum()))os.path.getsize()))os.path.join()))json_dir, f)) for f in os.listdir()))json_dir)):
        logger.info()))f"Total JSON size: {}}}}}}}}}}}}}}}}}}}}}}}}total_size / ()))1024*1024):.2f} MB")
    
                return total_size

def benchmark_queries()))conn, args):
    """Benchmark various query patterns against the database"""
    queries = []]]],,,,
    ()))"Single model lookup",
    "SELECT * FROM models WHERE model_name LIKE 'bert%' LIMIT 1"),
        
    ()))"Simple performance query",
    "SELECT * FROM performance_results LIMIT 100"),
        
    ()))"Join with aggregation",
    """
    SELECT
    m.model_family,
    hp.hardware_type,
    AVG()))pr.throughput_items_per_second) as avg_throughput
    FROM
    performance_results pr
    JOIN
    models m ON pr.model_id = m.model_id
    JOIN
    hardware_platforms hp ON pr.hardware_id = hp.hardware_id
    GROUP BY
    m.model_family, hp.hardware_type
    """),
        
    ()))"Complex join with filtering",
    """
    SELECT
    m.model_name,
    m.model_family,
    hp.hardware_type,
    pr.test_case,
    pr.batch_size,
    pr.average_latency_ms,
    pr.throughput_items_per_second
    FROM
    performance_results pr
    JOIN
    models m ON pr.model_id = m.model_id
    JOIN
    hardware_platforms hp ON pr.hardware_id = hp.hardware_id
    WHERE
    m.model_family = 'bert'
    AND pr.batch_size > 1
    AND hp.hardware_type IN ()))'cpu', 'cuda')
    ORDER BY
    pr.throughput_items_per_second DESC
    LIMIT 20
    """),
        
    ()))"Compatibility matrix",
    """
    SELECT
    m.model_family,
    hp.hardware_type,
    COUNT()))*) as test_count,
    SUM()))CASE WHEN hc.is_compatible THEN 1 ELSE 0 END) as compatible_count,
    AVG()))hc.compatibility_score) as avg_compatibility_score
    FROM
    hardware_compatibility hc
    JOIN
    models m ON hc.model_id = m.model_id
    JOIN
    hardware_platforms hp ON hc.hardware_id = hp.hardware_id
    GROUP BY
    m.model_family, hp.hardware_type
    """),
        
    ()))"Temporal analysis",
    """
    SELECT
    DATE_TRUNC()))'day', pr.created_at) as day,
    m.model_family,
    hp.hardware_type,
    AVG()))pr.throughput_items_per_second) as avg_throughput
    FROM
    performance_results pr
    JOIN
    models m ON pr.model_id = m.model_id
    JOIN
    hardware_platforms hp ON pr.hardware_id = hp.hardware_id
    GROUP BY
    DATE_TRUNC()))'day', pr.created_at), m.model_family, hp.hardware_type
    ORDER BY
    day DESC
    LIMIT 100
    """),
        
    ()))"Integration test summary",
    """
    SELECT
    itr.test_module,
    COUNT()))*) as total_tests,
    SUM()))CASE WHEN itr.status = 'pass' THEN 1 ELSE 0 END) as passed,
    SUM()))CASE WHEN itr.status = 'fail' THEN 1 ELSE 0 END) as failed,
    SUM()))CASE WHEN itr.status = 'error' THEN 1 ELSE 0 END) as errors,
    SUM()))CASE WHEN itr.status = 'skip' THEN 1 ELSE 0 END) as skipped
    FROM
    integration_test_results itr
    GROUP BY
    itr.test_module
    """),
        
    ()))"Window function analysis",
    """
    SELECT
    m.model_name,
    hp.hardware_type,
    pr.batch_size,
    pr.throughput_items_per_second,
    AVG()))pr.throughput_items_per_second) OVER ()))
    PARTITION BY m.model_id, hp.hardware_id
    ORDER BY pr.batch_size
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) as running_avg_throughput
    FROM
    performance_results pr
    JOIN
    models m ON pr.model_id = m.model_id
    JOIN
    hardware_platforms hp ON pr.hardware_id = hp.hardware_id
    ORDER BY
    m.model_name, hp.hardware_type, pr.batch_size
    LIMIT 100
    """),
    ]
    
    logger.info()))"Running benchmark queries...")
    results = []]]],,,,]
    
    for query_name, query in queries:
        # Run the query multiple times to get average performance
        query_times = []]]],,,,]
        
        for i in range()))args.query_repetitions):
            # Clear caches between runs
            if i > 0:
                conn.execute()))"PRAGMA memory_limit='16GB'")  # Force some memory management
            
            # Run the query and measure time
                start_time = time.time())))
                result = conn.execute()))query).fetchdf())))
                elapsed_time = time.time()))) - start_time
            
                query_times.append()))elapsed_time)
            
            # Log result size for first run
            if i == 0:
                logger.info()))f"Query '{}}}}}}}}}}}}}}}}}}}}}}}}query_name}' returned {}}}}}}}}}}}}}}}}}}}}}}}}len()))result)} rows")
        
        # Calculate statistics
                min_time = min()))query_times)
                max_time = max()))query_times)
                avg_time = sum()))query_times) / len()))query_times)
        
        # Record results
                result = {}}}}}}}}}}}}}}}}}}}}}}}}
                'query_name': query_name,
                'min_time': min_time,
                'max_time': max_time,
                'avg_time': avg_time,
                'repetitions': args.query_repetitions
                }
        
                results.append()))result)
        
                logger.info()))f"Query '{}}}}}}}}}}}}}}}}}}}}}}}}query_name}': min={}}}}}}}}}}}}}}}}}}}}}}}}min_time:.4f}s, avg={}}}}}}}}}}}}}}}}}}}}}}}}avg_time:.4f}s, max={}}}}}}}}}}}}}}}}}}}}}}}}max_time:.4f}s")
    
                return results

def benchmark_json_queries()))json_dir, args):
    """Benchmark equivalent queries against JSON files for comparison"""
    # Load the JSON files
    json_files = {}}}}}}}}}}}}}}}}}}}}}}}}}
    for file_name in os.listdir()))json_dir):
        if file_name.endswith()))'.json'):
            table_name = os.path.splitext()))file_name)[]]]],,,,0]
            with open()))os.path.join()))json_dir, file_name), 'r') as f:
                json_files[]]]],,,,table_name] = json.load()))f)
    
                logger.info()))"Loaded JSON files for querying")
    
    # Define equivalent queries
                queries = []]]],,,,
                ()))"Single model lookup",
                lambda data: []]]],,,,m for m in data[]]]],,,,'models'] if m[]]]],,,,'model_name'].startswith()))'bert')][]]]],,,,:1]),
        
                ()))"Simple performance query",
                lambda data: data[]]]],,,,'performance_results'][]]]],,,,:100]),
        
                ()))"Join with aggregation",
                lambda data: pd.DataFrame()))data[]]]],,,,'performance_results'])
                .merge()))pd.DataFrame()))data[]]]],,,,'models']), left_on='model_id', right_on='model_id')
                .merge()))pd.DataFrame()))data[]]]],,,,'hardware_platforms']), left_on='hardware_id', right_on='hardware_id')
                .groupby()))[]]]],,,,'model_family', 'hardware_type'])
                .agg())){}}}}}}}}}}}}}}}}}}}}}}}}'throughput_items_per_second': 'mean'})
                .reset_index())))
                .to_dict()))'records')),
        
                ()))"Complex join with filtering",
                lambda data: pd.DataFrame()))data[]]]],,,,'performance_results'])
                .merge()))pd.DataFrame()))data[]]]],,,,'models']), left_on='model_id', right_on='model_id')
                .merge()))pd.DataFrame()))data[]]]],,,,'hardware_platforms']), left_on='hardware_id', right_on='hardware_id')
                .query()))"model_family == 'bert' and batch_size > 1 and hardware_type in ()))'cpu', 'cuda')")
                .sort_values()))'throughput_items_per_second', ascending=False)
                .head()))20)
                .to_dict()))'records')),
        
                ()))"Compatibility matrix",
                lambda data: pd.DataFrame()))data[]]]],,,,'hardware_compatibility'])
                .merge()))pd.DataFrame()))data[]]]],,,,'models']), left_on='model_id', right_on='model_id')
                .merge()))pd.DataFrame()))data[]]]],,,,'hardware_platforms']), left_on='hardware_id', right_on='hardware_id')
                .groupby()))[]]]],,,,'model_family', 'hardware_type'])
                .agg())){}}}}}}}}}}}}}}}}}}}}}}}}
                'compatibility_id': 'count',
                'is_compatible': lambda x: sum()))x),
                'compatibility_score': 'mean'
                })
                .reset_index())))
                .to_dict()))'records')),
        
                ()))"Integration test summary",
                lambda data: pd.DataFrame()))data[]]]],,,,'integration_test_results'])
                .groupby()))'test_module')
                .agg())){}}}}}}}}}}}}}}}}}}}}}}}}
                'test_result_id': 'count',
                'status': lambda x: []]]],,,,
                    sum()))s == 'pass' for s in x):,:::
                    sum()))s == 'fail' for s in x):,:::
                    sum()))s == 'error' for s in x):,:::
                    sum()))s == 'skip' for s in x):
                        ]
                        })
                        .reset_index())))
                        .to_dict()))'records')),
                        ]
    
                        logger.info()))"Running benchmark queries on JSON files...")
                        results = []]]],,,,]
    
    for query_name, query_func in queries:
        # Run the query multiple times to get average performance
        query_times = []]]],,,,]
        
        for i in range()))args.query_repetitions):
            # Run the query and measure time
            start_time = time.time())))
            result = query_func()))json_files)
            elapsed_time = time.time()))) - start_time
            
            query_times.append()))elapsed_time)
            
            # Log result size for first run
            if i == 0:
                logger.info()))f"JSON Query '{}}}}}}}}}}}}}}}}}}}}}}}}query_name}' returned {}}}}}}}}}}}}}}}}}}}}}}}}len()))result)} items")
        
        # Calculate statistics
                min_time = min()))query_times)
                max_time = max()))query_times)
                avg_time = sum()))query_times) / len()))query_times)
        
        # Record results
                result = {}}}}}}}}}}}}}}}}}}}}}}}}
                'query_name': query_name,
                'min_time': min_time,
                'max_time': max_time,
                'avg_time': avg_time,
                'repetitions': args.query_repetitions
                }
        
                results.append()))result)
        
                logger.info()))f"JSON Query '{}}}}}}}}}}}}}}}}}}}}}}}}query_name}': min={}}}}}}}}}}}}}}}}}}}}}}}}min_time:.4f}s, avg={}}}}}}}}}}}}}}}}}}}}}}}}avg_time:.4f}s, max={}}}}}}}}}}}}}}}}}}}}}}}}max_time:.4f}s")
    
            return results

def compare_query_results()))db_results, json_results):
    """Compare query performance between DuckDB and JSON"""
    # Create a combined dataframe for comparison
    comparison = []]]],,,,]
    
    for db_result in db_results:
        # Find matching JSON result
        json_result = next()))()))r for r in json_results if r[]]]],,,,'query_name'] == db_result[]]]],,,,'query_name']), None)
        :
        if json_result:
            comparison.append())){}}}}}}}}}}}}}}}}}}}}}}}}
            'query_name': db_result[]]]],,,,'query_name'],
            'duckdb_avg_time': db_result[]]]],,,,'avg_time'],
            'json_avg_time': json_result[]]]],,,,'avg_time'],
            'speedup_factor': json_result[]]]],,,,'avg_time'] / db_result[]]]],,,,'avg_time'] if db_result[]]]],,,,'avg_time'] > 0 else float()))'inf')
            })
    
    # Create a dataframe
            comparison_df = pd.DataFrame()))comparison)
    
    # Print the comparison:
            logger.info()))"\nPerformance Comparison ()))DuckDB vs JSON):")
            logger.info()))tabulate()))comparison_df, headers='keys', tablefmt='pipe', showindex=False))
    
    # Calculate overall average speedup
            avg_speedup = comparison_df[]]]],,,,'speedup_factor'].mean())))
            logger.info()))f"\nOverall average speedup: {}}}}}}}}}}}}}}}}}}}}}}}}avg_speedup:.2f}x")
    
            return comparison_df

def get_database_size()))db_path):
    """Get the size of the database file"""
    if os.path.exists()))db_path):
        size_bytes = os.path.getsize()))db_path)
        size_mb = size_bytes / ()))1024 * 1024)
    return size_bytes, size_mb
            return 0, 0

def main()))):
    args = parse_args())))
    
    # Set logging level
    if args.verbose:
        logger.setLevel()))logging.DEBUG)
    
    # Connect to the database
        conn = connect_to_db()))args.db, args.in_memory)
    
    # Create the schema
        create_schema()))conn)
    
    # Determine the number of each result type to generate
    # We want most results to be performance results
        num_models = args.models
        num_hardware = args.hardware
        num_test_runs = args.test_runs
    
    # Generate test runs split roughly into 60% performance, 20% hardware, 20% integration
        num_perf_runs = int()))num_test_runs * 0.6)
        num_hw_runs = int()))num_test_runs * 0.2)
        num_int_runs = num_test_runs - num_perf_runs - num_hw_runs
    
    # Calculate result counts based on desired total
        result_per_run = max()))1, args.rows // num_test_runs)
        num_perf_results = num_perf_runs * result_per_run
        num_compat_results = num_hw_runs * result_per_run
        num_int_results = num_int_runs * result_per_run
    
    # Total should be close to args.rows
        total_results = num_perf_results + num_compat_results + num_int_results
    if total_results < args.rows:
        # Add the remainder to performance results
        num_perf_results += args.rows - total_results
    
    # Insert the data if not skipping:
    if not args.skip_insert:
        logger.info()))f"Generating synthetic data for benchmark")
        logger.info()))f"  - {}}}}}}}}}}}}}}}}}}}}}}}}num_models} models")
        logger.info()))f"  - {}}}}}}}}}}}}}}}}}}}}}}}}num_hardware} hardware platforms")
        logger.info()))f"  - {}}}}}}}}}}}}}}}}}}}}}}}}num_test_runs} test runs")
        logger.info()))f"  - {}}}}}}}}}}}}}}}}}}}}}}}}num_perf_results} performance results")
        logger.info()))f"  - {}}}}}}}}}}}}}}}}}}}}}}}}num_compat_results} compatibility results")
        logger.info()))f"  - {}}}}}}}}}}}}}}}}}}}}}}}}num_int_results} integration test results")
        
        # Generate the synthetic data
        models_df = generate_models()))num_models)
        hardware_df = generate_hardware_platforms()))num_hardware)
        test_runs_df = generate_test_runs()))num_test_runs)
        
        # Generate result data
        perf_df = generate_performance_results()))num_perf_results, models_df, hardware_df, test_runs_df)
        compat_df = generate_compatibility_results()))num_compat_results, models_df, hardware_df, test_runs_df)
        int_df, assert_df = generate_integration_results()))num_int_results, models_df, hardware_df, test_runs_df)
        
        # Insert the synthetic data
        insert_synthetic_data()))conn, models_df, hardware_df, test_runs_df, 
        perf_df, compat_df, int_df, assert_df, args)
        
        # For comparison, also store as JSON if requested::
        if args.test_json:
            json_dir = './benchmark_json'
            json_size = store_as_json()))models_df, hardware_df, test_runs_df, 
            perf_df, compat_df, int_df, assert_df, json_dir)
    
    # Run query benchmarks
            logger.info()))"\nRunning query benchmarks...")
            db_query_results = benchmark_queries()))conn, args)
    
    # Run JSON query benchmarks if requested::
    if args.test_json:
        logger.info()))"\nRunning JSON query benchmarks for comparison...")
        json_query_results = benchmark_json_queries()))'./benchmark_json', args)
        
        # Compare the results
        comparison = compare_query_results()))db_query_results, json_query_results)
    
    # Get the database size
        db_size_bytes, db_size_mb = get_database_size()))args.db)
        logger.info()))f"\nDatabase size: {}}}}}}}}}}}}}}}}}}}}}}}}db_size_mb:.2f} MB")
    
    # Compare with JSON size if available:
    if args.test_json:
        json_size_bytes = sum()))os.path.getsize()))os.path.join()))'./benchmark_json', f)) 
        for f in os.listdir()))'./benchmark_json') if f.endswith()))'.json'))
        json_size_mb = json_size_bytes / ()))1024 * 1024)
        :
            logger.info()))f"JSON size: {}}}}}}}}}}}}}}}}}}}}}}}}json_size_mb:.2f} MB")
            logger.info()))f"Size reduction: {}}}}}}}}}}}}}}}}}}}}}}}}()))json_size_bytes - db_size_bytes) / json_size_bytes * 100:.2f}%")
    
    # Close the database connection
            conn.close())))
    
            logger.info()))"\nBenchmark completed successfully")

if __name__ == "__main__":
    main())))