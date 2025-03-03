#!/usr/bin/env python
"""
Create the benchmark database schema for test results storage.
This script defines the DuckDB schema for storing benchmark results and test outputs
in a structured format, replacing the current JSON file approach.
"""

import os
import sys
import argparse
import json
import datetime
import duckdb
import pandas as pd
from pathlib import Path

# Add parent directory to path for module imports
sys.path.append(str(Path(__file__).parent.parent))

def parse_args():
    parser = argparse.ArgumentParser(description="Create benchmark database schema")
    parser.add_argument("--output", type=str, default="./benchmark_db.duckdb", 
                       help="Path to create/update the DuckDB database")
    parser.add_argument("--sample-data", action="store_true", 
                       help="Generate sample data to test the schema")
    parser.add_argument("--force", action="store_true", 
                       help="Force recreate tables even if they exist")
    parser.add_argument("--verbose", action="store_true", 
                       help="Print detailed logging information")
    return parser.parse_args()

def connect_to_db(db_path):
    """Connect to DuckDB database"""
    # Create parent directories if they don't exist
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    
    # Connect to the database
    return duckdb.connect(db_path)

def create_common_tables(conn, force=False):
    """Create the common dimension tables used across schemas"""
    
    # Hardware platform dimension table
    if force:
        conn.execute("DROP TABLE IF EXISTS hardware_platforms")
    
    conn.execute("""
    CREATE TABLE IF NOT EXISTS hardware_platforms (
        hardware_id INTEGER PRIMARY KEY,
        hardware_type VARCHAR NOT NULL, -- 'cpu', 'cuda', 'rocm', 'mps', 'openvino', 'webnn', 'webgpu'
        device_name VARCHAR,
        platform VARCHAR,
        platform_version VARCHAR,
        driver_version VARCHAR,
        memory_gb FLOAT,
        compute_units INTEGER,
        metadata JSON,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Model dimension table
    if force:
        conn.execute("DROP TABLE IF EXISTS models")
    
    conn.execute("""
    CREATE TABLE IF NOT EXISTS models (
        model_id INTEGER PRIMARY KEY,
        model_name VARCHAR NOT NULL,
        model_family VARCHAR, -- 'bert', 't5', 'gpt', etc.
        modality VARCHAR, -- 'text', 'image', 'audio', 'multimodal'
        source VARCHAR, -- 'huggingface', 'openai', 'anthropic', etc.
        version VARCHAR,
        parameters_million FLOAT,
        metadata JSON,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Test runs dimension table to track individual test execution
    if force:
        conn.execute("DROP TABLE IF EXISTS test_runs")
    
    conn.execute("""
    CREATE TABLE IF NOT EXISTS test_runs (
        run_id INTEGER PRIMARY KEY,
        test_name VARCHAR NOT NULL,
        test_type VARCHAR NOT NULL, -- 'performance', 'hardware', 'compatibility', 'integration'
        started_at TIMESTAMP,
        completed_at TIMESTAMP,
        execution_time_seconds FLOAT,
        success BOOLEAN,
        git_commit VARCHAR,
        git_branch VARCHAR,
        command_line VARCHAR,
        metadata JSON,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

def create_performance_tables(conn, force=False):
    """Create tables for performance benchmark results"""
    
    # Drop tables if force flag is True
    if force:
        conn.execute("DROP TABLE IF EXISTS performance_results")
        conn.execute("DROP TABLE IF EXISTS performance_batch_results")
    
    # Main performance results table
    conn.execute("""
    CREATE TABLE IF NOT EXISTS performance_results (
        result_id INTEGER PRIMARY KEY,
        run_id INTEGER NOT NULL,
        model_id INTEGER NOT NULL,
        hardware_id INTEGER NOT NULL,
        test_case VARCHAR NOT NULL,
        batch_size INTEGER DEFAULT 1,
        precision VARCHAR, -- 'fp32', 'fp16', 'int8', etc.
        total_time_seconds FLOAT,
        average_latency_ms FLOAT,
        throughput_items_per_second FLOAT,
        memory_peak_mb FLOAT,
        iterations INTEGER,
        warmup_iterations INTEGER,
        metrics JSON, -- Additional metrics specific to test case
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (run_id) REFERENCES test_runs(run_id),
        FOREIGN KEY (model_id) REFERENCES models(model_id),
        FOREIGN KEY (hardware_id) REFERENCES hardware_platforms(hardware_id)
    )
    """)
    
    # Batch-level details for deeper analysis
    conn.execute("""
    CREATE TABLE IF NOT EXISTS performance_batch_results (
        batch_id INTEGER PRIMARY KEY,
        result_id INTEGER NOT NULL,
        batch_index INTEGER NOT NULL,
        batch_size INTEGER NOT NULL,
        latency_ms FLOAT,
        memory_usage_mb FLOAT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (result_id) REFERENCES performance_results(result_id)
    )
    """)

def create_hardware_compatibility_tables(conn, force=False):
    """Create tables for hardware compatibility test results"""
    
    if force:
        conn.execute("DROP TABLE IF EXISTS hardware_compatibility")
    
    conn.execute("""
    CREATE TABLE IF NOT EXISTS hardware_compatibility (
        compatibility_id INTEGER PRIMARY KEY,
        run_id INTEGER NOT NULL,
        model_id INTEGER NOT NULL,
        hardware_id INTEGER NOT NULL,
        is_compatible BOOLEAN NOT NULL,
        detection_success BOOLEAN NOT NULL,
        initialization_success BOOLEAN NOT NULL,
        error_message VARCHAR,
        error_type VARCHAR,
        suggested_fix VARCHAR,
        workaround_available BOOLEAN,
        compatibility_score FLOAT, -- 0-1 score if partially compatible
        metadata JSON,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (run_id) REFERENCES test_runs(run_id),
        FOREIGN KEY (model_id) REFERENCES models(model_id),
        FOREIGN KEY (hardware_id) REFERENCES hardware_platforms(hardware_id)
    )
    """)

def create_integration_test_tables(conn, force=False):
    """Create tables for integration test results"""
    
    if force:
        conn.execute("DROP TABLE IF EXISTS integration_test_results")
        conn.execute("DROP TABLE IF EXISTS integration_test_assertions")
    
    conn.execute("""
    CREATE TABLE IF NOT EXISTS integration_test_results (
        test_result_id INTEGER PRIMARY KEY,
        run_id INTEGER NOT NULL,
        test_module VARCHAR NOT NULL,
        test_class VARCHAR,
        test_name VARCHAR NOT NULL,
        status VARCHAR NOT NULL, -- 'pass', 'fail', 'error', 'skip'
        execution_time_seconds FLOAT,
        hardware_id INTEGER,
        model_id INTEGER,
        error_message VARCHAR,
        error_traceback VARCHAR,
        metadata JSON,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (run_id) REFERENCES test_runs(run_id),
        FOREIGN KEY (hardware_id) REFERENCES hardware_platforms(hardware_id),
        FOREIGN KEY (model_id) REFERENCES models(model_id)
    )
    """)
    
    conn.execute("""
    CREATE TABLE IF NOT EXISTS integration_test_assertions (
        assertion_id INTEGER PRIMARY KEY,
        test_result_id INTEGER NOT NULL,
        assertion_name VARCHAR NOT NULL,
        passed BOOLEAN NOT NULL,
        expected_value VARCHAR,
        actual_value VARCHAR,
        message VARCHAR,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (test_result_id) REFERENCES integration_test_results(test_result_id)
    )
    """)

def create_views(conn):
    """Create useful views across the tables"""
    
    # Model-Hardware compatibility view
    conn.execute("""
    CREATE OR REPLACE VIEW model_hardware_compatibility AS
    SELECT 
        m.model_name,
        m.model_family,
        hp.hardware_type,
        hp.device_name,
        COUNT(CASE WHEN hc.is_compatible THEN 1 END) AS compatible_count,
        COUNT(CASE WHEN NOT hc.is_compatible THEN 1 END) AS incompatible_count,
        AVG(CASE WHEN hc.compatibility_score IS NOT NULL THEN hc.compatibility_score ELSE 
            CASE WHEN hc.is_compatible THEN 1.0 ELSE 0.0 END END) AS avg_compatibility_score,
        MAX(hc.created_at) AS last_tested
    FROM 
        hardware_compatibility hc
    JOIN 
        models m ON hc.model_id = m.model_id
    JOIN 
        hardware_platforms hp ON hc.hardware_id = hp.hardware_id
    GROUP BY 
        m.model_name, m.model_family, hp.hardware_type, hp.device_name
    """)
    
    # Performance metrics view - latest results by model/hardware
    conn.execute("""
    CREATE OR REPLACE VIEW latest_performance_metrics AS
    SELECT 
        m.model_name,
        m.model_family,
        hp.hardware_type,
        hp.device_name,
        pr.batch_size,
        pr.precision,
        pr.average_latency_ms,
        pr.throughput_items_per_second,
        pr.memory_peak_mb,
        pr.created_at,
        ROW_NUMBER() OVER(PARTITION BY m.model_id, hp.hardware_id 
                        ORDER BY pr.created_at DESC) as rn
    FROM 
        performance_results pr
    JOIN 
        models m ON pr.model_id = m.model_id
    JOIN 
        hardware_platforms hp ON pr.hardware_id = hp.hardware_id
    QUALIFY rn = 1
    """)
    
    # Integration test status by component
    conn.execute("""
    CREATE OR REPLACE VIEW integration_test_status AS
    SELECT 
        test_module,
        COUNT(*) as total_tests,
        COUNT(CASE WHEN status = 'pass' THEN 1 END) as passed,
        COUNT(CASE WHEN status = 'fail' THEN 1 END) as failed,
        COUNT(CASE WHEN status = 'error' THEN 1 END) as errors,
        COUNT(CASE WHEN status = 'skip' THEN 1 END) as skipped,
        MAX(created_at) as last_run
    FROM 
        integration_test_results
    GROUP BY 
        test_module
    """)

def generate_sample_data(conn):
    """Generate sample data for testing the schema"""
    
    # Sample hardware platforms
    hardware_data = [
        (1, 'cpu', 'Intel Core i9-12900K', 'x86_64', '5.15.0-76-generic', 'N/A', 64.0, 16, 
         json.dumps({'cores': 16, 'threads': 24})),
        (2, 'cuda', 'NVIDIA RTX 4090', 'CUDA', '12.1', '535.54.03', 24.0, 128, 
         json.dumps({'cuda_cores': 16384, 'tensor_cores': 512})),
        (3, 'rocm', 'AMD Radeon RX 7900 XTX', 'ROCm', '5.5.0', '5.5.0', 24.0, 96, 
         json.dumps({'compute_units': 96, 'stream_processors': 12288})),
        (4, 'mps', 'Apple M2 Ultra', 'macOS', '14.1', 'N/A', 32.0, 76, 
         json.dumps({'neural_engine_cores': 16})),
        (5, 'openvino', 'Intel Neural Compute Stick 2', 'OpenVINO', '2023.0', '2023.0', 4.0, 16,
         json.dumps({'vpu_cores': 16})),
        (6, 'webnn', 'Chrome Browser', 'WebNN', '121.0', 'N/A', 0, 0,
         json.dumps({'user_agent': 'Mozilla/5.0 Chrome/121.0.0.0'})),
        (7, 'webgpu', 'Firefox Browser', 'WebGPU', '122.0', 'N/A', 0, 0,
         json.dumps({'user_agent': 'Mozilla/5.0 Firefox/122.0'}))
    ]
    
    hardware_df = pd.DataFrame(hardware_data, columns=[
        'hardware_id', 'hardware_type', 'device_name', 'platform', 'platform_version',
        'driver_version', 'memory_gb', 'compute_units', 'metadata'
    ])
    conn.execute("INSERT INTO hardware_platforms SELECT * FROM hardware_df")
    
    # Sample models
    model_data = [
        (1, 'bert-base-uncased', 'bert', 'text', 'huggingface', '1.0', 110.0,
         json.dumps({'vocab_size': 30522, 'hidden_size': 768})),
        (2, 't5-small', 't5', 'text', 'huggingface', '1.0', 60.0,
         json.dumps({'vocab_size': 32128, 'hidden_size': 512})),
        (3, 'whisper-tiny', 'whisper', 'audio', 'huggingface', '1.0', 39.0,
         json.dumps({'mel_filters': 80, 'hidden_size': 384})),
        (4, 'opt-125m', 'llama', 'text', 'huggingface', '1.0', 125.0,
         json.dumps({'vocab_size': 50272, 'hidden_size': 768})),
        (5, 'vit-base', 'vit', 'image', 'huggingface', '1.0', 86.0,
         json.dumps({'image_size': 224, 'patch_size': 16, 'hidden_size': 768})),
        (6, 'llava-onevision-base', 'llava', 'multimodal', 'huggingface', '1.0', 860.0,
         json.dumps({'image_size': 336, 'hidden_size': 4096}))
    ]
    
    model_df = pd.DataFrame(model_data, columns=[
        'model_id', 'model_name', 'model_family', 'modality', 'source', 'version',
        'parameters_million', 'metadata'
    ])
    conn.execute("INSERT INTO models SELECT * FROM model_df")
    
    # Sample test runs
    current_time = datetime.datetime.now()
    test_runs_data = [
        (1, 'performance_benchmark_bert', 'performance', 
         current_time - datetime.timedelta(hours=2),
         current_time - datetime.timedelta(hours=1),
         3600.0, True, 'a404c5a', 'main', 
         'python test/run_model_benchmarks.py --model bert-base-uncased',
         json.dumps({'environment': 'CI', 'triggered_by': 'schedule'})),
        (2, 'hardware_compatibility_test', 'hardware', 
         current_time - datetime.timedelta(days=1, hours=3),
         current_time - datetime.timedelta(days=1, hours=2),
         3600.0, True, '93af533', 'main', 
         'python test/test_hardware_backend.py --all',
         json.dumps({'environment': 'local', 'triggered_by': 'manual'})),
        (3, 'integration_test_suite', 'integration', 
         current_time - datetime.timedelta(hours=12),
         current_time - datetime.timedelta(hours=11, minutes=45),
         2700.0, True, 'f27af98', 'main', 
         './test/run_integration_ci_tests.sh --all',
         json.dumps({'environment': 'CI', 'triggered_by': 'push'}))
    ]
    
    test_runs_df = pd.DataFrame(test_runs_data, columns=[
        'run_id', 'test_name', 'test_type', 'started_at', 'completed_at',
        'execution_time_seconds', 'success', 'git_commit', 'git_branch', 
        'command_line', 'metadata'
    ])
    conn.execute("INSERT INTO test_runs SELECT * FROM test_runs_df")
    
    # Sample performance results
    perf_data = [
        (1, 1, 1, 1, 'embedding', 1, 'fp32', 120.5, 25.3, 39.5, 1200.0, 100, 10,
         json.dumps({'cpu_util': 78.5, 'memory_util': 45.2})),
        (2, 1, 1, 2, 'embedding', 1, 'fp32', 30.2, 6.1, 163.9, 2300.0, 100, 10,
         json.dumps({'gpu_util': 85.3, 'memory_util': 55.8})),
        (3, 1, 2, 1, 'text_generation', 1, 'fp32', 245.7, 50.1, 20.0, 1450.0, 100, 10,
         json.dumps({'cpu_util': 92.1, 'memory_util': 61.5})),
        (4, 1, 2, 2, 'text_generation', 1, 'fp32', 78.3, 15.9, 62.9, 3100.0, 100, 10,
         json.dumps({'gpu_util': 91.7, 'memory_util': 72.3})),
        (5, 1, 3, 2, 'audio_transcription', 1, 'fp16', 105.6, 21.3, 46.9, 2800.0, 100, 10,
         json.dumps({'gpu_util': 88.9, 'memory_util': 68.5}))
    ]
    
    perf_df = pd.DataFrame(perf_data, columns=[
        'result_id', 'run_id', 'model_id', 'hardware_id', 'test_case', 'batch_size',
        'precision', 'total_time_seconds', 'average_latency_ms', 'throughput_items_per_second',
        'memory_peak_mb', 'iterations', 'warmup_iterations', 'metrics'
    ])
    conn.execute("INSERT INTO performance_results SELECT * FROM perf_df")
    
    # Sample hardware compatibility
    compat_data = [
        (1, 2, 1, 1, True, True, True, None, None, None, True, 1.0, 
         json.dumps({'detected_features': ['avx2', 'fma']})),
        (2, 2, 1, 2, True, True, True, None, None, None, True, 1.0,
         json.dumps({'cuda_version_compatible': True})),
        (3, 2, 1, 3, True, True, True, None, None, None, True, 1.0,
         json.dumps({'rocm_compatible': True})),
        (4, 2, 2, 1, True, True, True, None, None, None, True, 1.0,
         json.dumps({'detected_features': ['avx2', 'fma']})),
        (5, 2, 3, 1, True, True, True, None, None, None, True, 1.0,
         json.dumps({'detected_features': ['avx2', 'fma']})),
        (6, 2, 3, 2, True, True, True, None, None, None, True, 1.0,
         json.dumps({'cuda_version_compatible': True})),
        (7, 2, 6, 1, True, True, True, None, None, None, True, 1.0,
         json.dumps({'detected_features': ['avx2', 'fma']})),
        (8, 2, 6, 2, True, True, True, None, None, None, True, 1.0,
         json.dumps({'cuda_version_compatible': True})),
        (9, 2, 6, 3, False, True, False, 'ROCm support not implemented for LLaVA models',
         'UnsupportedHardwareError', 'Use CUDA instead', False, 0.0,
         json.dumps({'error_code': 'ROCM_UNSUPPORTED'})),
        (10, 2, 6, 4, False, True, False, 'MPS support not implemented for LLaVA models',
         'UnsupportedHardwareError', 'Use CUDA instead', False, 0.0,
         json.dumps({'error_code': 'MPS_UNSUPPORTED'}))
    ]
    
    compat_df = pd.DataFrame(compat_data, columns=[
        'compatibility_id', 'run_id', 'model_id', 'hardware_id', 'is_compatible',
        'detection_success', 'initialization_success', 'error_message', 'error_type',
        'suggested_fix', 'workaround_available', 'compatibility_score', 'metadata'
    ])
    conn.execute("INSERT INTO hardware_compatibility SELECT * FROM compat_df")
    
    # Sample integration test results
    int_test_data = [
        (1, 3, 'test_hardware_backend', 'TestHardwareDetection', 'test_cpu_detection',
         'pass', 2.3, 1, None, None, None, json.dumps({'os': 'Linux'})),
        (2, 3, 'test_hardware_backend', 'TestHardwareDetection', 'test_cuda_detection',
         'pass', 3.5, 2, None, None, None, json.dumps({'cuda_version': '12.1'})),
        (3, 3, 'test_resource_pool', 'TestResourcePoolHardwareAwareness', 'test_cpu_allocation',
         'pass', 1.8, 1, None, None, None, json.dumps({'allocated_cores': 8})),
        (4, 3, 'test_resource_pool', 'TestResourcePoolHardwareAwareness', 'test_gpu_allocation',
         'pass', 2.1, 2, None, None, None, json.dumps({'allocated_memory': '8GB'})),
        (5, 3, 'test_comprehensive_hardware', 'TestHardwareCompatibility', 'test_t5_openvino',
         'fail', 4.2, 5, 1, 'OpenVINO backend failed to initialize T5 model',
         'File "/home/test/test_comprehensive_hardware.py", line 342\nAttributeError: \'NoneType\' object has no attribute \'initialize\'',
         json.dumps({'openvino_version': '2023.0'}))
    ]
    
    int_test_df = pd.DataFrame(int_test_data, columns=[
        'test_result_id', 'run_id', 'test_module', 'test_class', 'test_name', 'status',
        'execution_time_seconds', 'hardware_id', 'model_id', 'error_message', 'error_traceback', 'metadata'
    ])
    conn.execute("INSERT INTO integration_test_results SELECT * FROM int_test_df")
    
    # Sample test assertions
    assertion_data = [
        (1, 1, 'assert_cpu_features_detected', True, 'True', 'True', 'CPU features correctly detected'),
        (2, 2, 'assert_cuda_version_compatible', True, 'True', 'True', 'CUDA version is compatible'),
        (3, 3, 'assert_resource_allocation_success', True, 'True', 'True', 'Resource allocation successful'),
        (4, 4, 'assert_gpu_memory_allocated', True, '8GB', '8GB', 'GPU memory correctly allocated'),
        (5, 5, 'assert_openvino_initialized', False, 'True', 'False', 'OpenVINO failed to initialize')
    ]
    
    assertion_df = pd.DataFrame(assertion_data, columns=[
        'assertion_id', 'test_result_id', 'assertion_name', 'passed', 'expected_value',
        'actual_value', 'message'
    ])
    conn.execute("INSERT INTO integration_test_assertions SELECT * FROM assertion_df")

def main():
    args = parse_args()
    
    print(f"Creating benchmark database schema at: {args.output}")
    conn = connect_to_db(args.output)
    
    # Create the schema
    create_common_tables(conn, args.force)
    create_performance_tables(conn, args.force)
    create_hardware_compatibility_tables(conn, args.force)
    create_integration_test_tables(conn, args.force)
    create_views(conn)
    
    # Generate sample data if requested
    if args.sample_data:
        print("Generating sample data...")
        try:
            generate_sample_data(conn)
            print("Sample data generated successfully")
        except Exception as e:
            print(f"Error generating sample data: {e}")
            # If error is about duplicate data, inform the user
            if "duplicate key" in str(e).lower() or "unique constraint" in str(e).lower():
                print("It appears sample data already exists. Use --force to recreate tables.")
    
    # Display schema counts for verification
    tables = conn.execute("SHOW TABLES").fetchall()
    print(f"\nCreated {len(tables)} tables and views:")
    for table in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM {table[0]}").fetchone()[0]
        print(f"  - {table[0]}: {count} rows")
    
    conn.close()
    print("\nDatabase schema creation completed successfully.")

if __name__ == "__main__":
    main()