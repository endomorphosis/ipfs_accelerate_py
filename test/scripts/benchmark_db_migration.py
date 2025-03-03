#!/usr/bin/env python
"""
Benchmark Database Migration Tool

This script handles the migration of benchmark data from various sources into
the DuckDB database. It supports:

1. Migrating JSON test results from various directories
2. Migrating CI/CD benchmark results
3. Tracking migration status to avoid duplicate imports
4. Validating migrated data for consistency

Example usage:
    # Migrate JSON files from specific directories
    python benchmark_db_migration.py --input-dirs performance_results --db benchmark_db.duckdb
    
    # Migrate all known result directories
    python benchmark_db_migration.py --migrate-all --db benchmark_db.duckdb
    
    # Migrate CI results
    python benchmark_db_migration.py --migrate-ci --artifacts-dir ./artifacts --db benchmark_db.duckdb
    
    # Validate migrated data
    python benchmark_db_migration.py --validate --db benchmark_db.duckdb
"""

import os
import sys
import json
import glob
import argparse
import logging
import datetime
import hashlib
import concurrent.futures
from pathlib import Path
import shutil
import duckdb
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("db_migration")

# Default directories to scan for benchmark data
DEFAULT_DIRECTORIES = [
    "performance_results",
    "archived_test_results",
    "hardware_compatibility_reports",
    "collected_results",
    "integration_results",
    "critical_model_results",
    "new_model_results",
    "archived_test_files",
    "batch_inference_results"
]

def parse_args():
    parser = argparse.ArgumentParser(description="Migrate benchmark data to DuckDB database")
    
    parser.add_argument("--db", type=str, default="./benchmark_db.duckdb", 
                        help="Path to DuckDB database")
    
    # Source selection
    parser.add_argument("--input-dirs", type=str,
                        help="Comma-separated list of directories to scan for JSON files")
    parser.add_argument("--input-files", type=str,
                        help="Comma-separated list of specific JSON files to migrate")
    parser.add_argument("--migrate-all", action="store_true",
                        help="Migrate all known result directories")
    parser.add_argument("--migrate-ci", action="store_true",
                        help="Migrate CI/CD benchmark results")
    parser.add_argument("--artifacts-dir", type=str, default="./artifacts",
                        help="Directory containing CI artifacts (for --migrate-ci)")
    
    # Migration behavior
    parser.add_argument("--skip-if-exists", action="store_true",
                        help="Skip migration if file hash exists in tracking table")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing entries even if hash matches")
    parser.add_argument("--action", type=str, choices=['none', 'archive', 'remove'], default='none',
                        help="Action to take with processed files")
    parser.add_argument("--archive-dir", type=str, default="./archived_results",
                        help="Directory to archive processed files (for --action=archive)")
    
    # Validation
    parser.add_argument("--validate", action="store_true",
                        help="Validate migrated data for consistency")
    parser.add_argument("--fix-inconsistencies", action="store_true",
                        help="Attempt to fix inconsistencies found during validation")
    
    # Performance
    parser.add_argument("--parallel", type=int, default=4,
                        help="Number of parallel processes for migration (0 for sequential)")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Number of files to process in each batch")
    
    # Misc
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate migration without making changes")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--summary-only", action="store_true",
                        help="Only print summary, no details")
    
    return parser.parse_args()

def setup_database(db_path, create_if_missing=True):
    """Connect to the DuckDB database and ensure migration tracking table exists"""
    try:
        # Check if database file exists
        db_exists = os.path.exists(db_path)
        
        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        
        # Connect to the database
        conn = duckdb.connect(db_path)
        
        # Check if the database is initialized with our schema
        if db_exists:
            tables = conn.execute("SHOW TABLES").fetchall()
            table_names = [t[0].lower() for t in tables]
            
            required_tables = ['hardware_platforms', 'models', 'test_runs', 'performance_results']
            missing_tables = [t for t in required_tables if t.lower() not in table_names]
            
            if missing_tables:
                if create_if_missing:
                    logger.warning(f"Database exists but missing tables: {', '.join(missing_tables)}")
                    logger.warning("Initializing missing tables...")
                    
                    # Try to initialize the database schema
                    try:
                        script_dir = os.path.dirname(os.path.abspath(__file__))
                        schema_script = os.path.join(script_dir, "create_benchmark_schema.py")
                        
                        if os.path.exists(schema_script):
                            # Import and run the schema creation function
                            sys.path.insert(0, script_dir)
                            from create_benchmark_schema import create_schema
                            create_schema(conn, drop_existing=False)
                        else:
                            logger.error(f"Schema creation script not found at {schema_script}")
                            return None
                    except Exception as e:
                        logger.error(f"Error initializing database schema: {e}")
                        return None
                else:
                    logger.error(f"Database exists but missing required tables: {', '.join(missing_tables)}")
                    logger.error("Run scripts/create_benchmark_schema.py to initialize the database")
                    return None
        else:
            if create_if_missing:
                logger.info(f"Database {db_path} does not exist, initializing...")
                
                # Try to initialize the database schema
                try:
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    schema_script = os.path.join(script_dir, "create_benchmark_schema.py")
                    
                    if os.path.exists(schema_script):
                        # Import and run the schema creation function
                        sys.path.insert(0, script_dir)
                        from create_benchmark_schema import create_schema
                        create_schema(conn)
                    else:
                        logger.error(f"Schema creation script not found at {schema_script}")
                        return None
                except Exception as e:
                    logger.error(f"Error initializing database schema: {e}")
                    return None
            else:
                logger.error(f"Database {db_path} does not exist")
                return None
        
        # Create migration tracking table if it doesn't exist
        conn.execute("""
        CREATE TABLE IF NOT EXISTS migration_tracking (
            file_path VARCHAR,
            file_hash VARCHAR,
            file_size BIGINT,
            category VARCHAR,
            migrated_at TIMESTAMP,
            status VARCHAR,
            records_imported INTEGER,
            message VARCHAR,
            PRIMARY KEY (file_path)
        )
        """)
        
        return conn
    except Exception as e:
        logger.error(f"Error setting up database: {e}")
        return None

def calculate_file_hash(file_path):
    """Calculate a hash of the file contents for tracking"""
    try:
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            buf = f.read(65536)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating hash for {file_path}: {e}")
        return None

def categorize_json_file(file_path):
    """Determine the category of a JSON file based on naming and content patterns"""
    filename = os.path.basename(file_path)
    dirname = os.path.basename(os.path.dirname(file_path))
    
    # Performance results
    if any(pattern in filename for pattern in ['performance', 'benchmark']):
        return 'performance'
    
    # Hardware compatibility
    if any(pattern in filename for pattern in ['hardware', 'compatibility', 'detection']):
        return 'hardware'
    
    # Model test results
    if any(pattern in filename for pattern in ['model_test', 'critical_model', 'test_status']):
        return 'model'
    
    # Integration test results
    if any(pattern in filename for pattern in ['integration', 'test_results']):
        return 'integration'
    
    # Check file content as a fallback
    try:
        with open(file_path, 'r') as f:
            # Read just the first 1000 chars to detect file type
            content_start = f.read(1000)
            
            if any(key in content_start for key in ['"throughput":', '"latency":', '"memory_peak":']):
                return 'performance'
            
            if any(key in content_start for key in ['"is_compatible":', '"hardware_type":', '"device_name":']):
                return 'hardware'
            
            if any(key in content_start for key in ['"test_name":', '"test_module":', '"status":']):
                return 'integration'
            
            if any(key in content_start for key in ['"model_name":', '"model_family":', '"model_tests":']):
                return 'model'
    except:
        # If we can't read the file, use directory name as a hint
        pass
    
    # If still undetermined, use directory name as hint
    if dirname in ['performance_results', 'benchmark_results', 'batch_inference_results']:
        return 'performance'
    elif dirname in ['hardware_compatibility_reports', 'collected_results']:
        return 'hardware'
    elif dirname in ['integration_results']:
        return 'integration'
    elif dirname in ['critical_model_results', 'new_model_results']:
        return 'model'
    
    # Default to unknown
    return 'unknown'

def find_json_files(args):
    """Find JSON files to migrate based on command line args"""
    json_files = []
    
    # Process specific files if provided
    if args.input_files:
        file_paths = [f.strip() for f in args.input_files.split(',')]
        for file_path in file_paths:
            if os.path.exists(file_path) and file_path.endswith('.json'):
                file_stat = os.stat(file_path)
                json_files.append({
                    'path': file_path,
                    'category': categorize_json_file(file_path),
                    'size': file_stat.st_size,
                    'modified': datetime.datetime.fromtimestamp(file_stat.st_mtime)
                })
            else:
                logger.warning(f"File not found or not a JSON file: {file_path}")
        
        return json_files
    
    # Determine directories to scan
    dirs_to_scan = []
    if args.input_dirs:
        dirs_to_scan = [d.strip() for d in args.input_dirs.split(',')]
    elif args.migrate_all:
        dirs_to_scan = DEFAULT_DIRECTORIES
    else:
        logger.warning("No input files or directories specified, use --input-dirs, --input-files, or --migrate-all")
        return []
    
    # Scan directories for JSON files
    for directory in dirs_to_scan:
        directory_path = os.path.abspath(directory)
        if not os.path.exists(directory_path):
            logger.warning(f"Directory not found: {directory_path}")
            continue
            
        logger.info(f"Scanning for JSON files in: {directory}")
        
        # Find all JSON files in the directory and subdirectories
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    file_stat = os.stat(file_path)
                    
                    # Determine the category
                    category = categorize_json_file(file_path)
                    
                    json_files.append({
                        'path': file_path,
                        'category': category,
                        'size': file_stat.st_size,
                        'modified': datetime.datetime.fromtimestamp(file_stat.st_mtime)
                    })
    
    logger.info(f"Found {len(json_files)} JSON files to process")
    
    return json_files

def find_ci_artifacts(args):
    """Find CI artifacts to migrate"""
    ci_files = []
    
    artifacts_dir = os.path.abspath(args.artifacts_dir)
    if not os.path.exists(artifacts_dir):
        logger.warning(f"CI artifacts directory not found: {artifacts_dir}")
        return []
    
    logger.info(f"Scanning for CI artifacts in: {artifacts_dir}")
    
    # Look for benchmark databases
    db_files = glob.glob(os.path.join(artifacts_dir, "**/ci_benchmark.duckdb"), recursive=True)
    db_files += glob.glob(os.path.join(artifacts_dir, "**/benchmark.duckdb"), recursive=True)
    db_files += glob.glob(os.path.join(artifacts_dir, "**/benchmark_*.duckdb"), recursive=True)
    
    # Look for JSON files
    json_files = glob.glob(os.path.join(artifacts_dir, "**/*.json"), recursive=True)
    
    # Process database files
    for db_path in db_files:
        file_stat = os.stat(db_path)
        ci_files.append({
            'path': db_path,
            'category': 'ci_database',
            'size': file_stat.st_size,
            'modified': datetime.datetime.fromtimestamp(file_stat.st_mtime)
        })
    
    # Process JSON files
    for file_path in json_files:
        file_stat = os.stat(file_path)
        category = categorize_json_file(file_path)
        ci_files.append({
            'path': file_path,
            'category': f'ci_{category}',
            'size': file_stat.st_size,
            'modified': datetime.datetime.fromtimestamp(file_stat.st_mtime)
        })
    
    logger.info(f"Found {len(ci_files)} CI artifacts to process")
    
    return ci_files

def check_migration_status(conn, file_info, args):
    """Check if a file has already been migrated"""
    file_path = file_info['path']
    
    # Calculate file hash if we care about duplicate prevention
    file_hash = None
    if args.skip_if_exists or not args.overwrite:
        file_hash = calculate_file_hash(file_path)
        if not file_hash:
            return False, "Failed to calculate file hash"
    
    # Check if this file is already tracked
    try:
        if args.skip_if_exists:
            # Check by file path
            result = conn.execute(
                "SELECT file_hash, status FROM migration_tracking WHERE file_path = ?",
                [file_path]
            ).fetchone()
            
            if result:
                db_hash, status = result
                
                # If hash matches, skip this file
                if db_hash == file_hash and status == 'success':
                    return True, "File already migrated (same hash)"
                
                # If hash doesn't match but we're not overwriting, skip
                if not args.overwrite:
                    return True, f"File tracked with different hash (status: {status})"
        
        # We'll process this file
        return False, None
        
    except Exception as e:
        logger.error(f"Error checking migration status for {file_path}: {e}")
        return False, str(e)

def migrate_json_file(conn, file_info, args):
    """Migrate a single JSON file to the database"""
    file_path = file_info['path']
    category = file_info['category']
    
    logger.debug(f"Migrating {category} file: {file_path}")
    
    # If dry run, just return success
    if args.dry_run:
        logger.debug(f"Dry run: would migrate {file_path}")
        return True, 0, "Dry run"
    
    try:
        # Calculate file hash for tracking
        file_hash = calculate_file_hash(file_path)
        if not file_hash:
            return False, 0, "Failed to calculate file hash"
        
        # Try to read the JSON file
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                return False, 0, f"JSON decode error: {e}"
        
        # Process according to category
        records_imported = 0
        
        if category == 'performance':
            # Import performance benchmark results
            records_imported = import_performance_data(conn, data, file_path)
        elif category == 'hardware':
            # Import hardware compatibility results
            records_imported = import_hardware_data(conn, data, file_path)
        elif category == 'integration':
            # Import integration test results
            records_imported = import_integration_data(conn, data, file_path)
        elif category == 'model':
            # Import model test results
            records_imported = import_model_data(conn, data, file_path)
        else:
            return False, 0, f"Unknown category: {category}"
        
        # Record successful migration
        conn.execute("""
        INSERT OR REPLACE INTO migration_tracking
        (file_path, file_hash, file_size, category, migrated_at, status, records_imported, message)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            file_path, file_hash, file_info['size'], category,
            datetime.datetime.now(), 'success', records_imported, 'Successfully imported'
        ])
        
        # Handle file action (archive/remove)
        if args.action == 'archive':
            archive_file(file_path, args.archive_dir)
        elif args.action == 'remove':
            os.remove(file_path)
            logger.debug(f"Removed file: {file_path}")
        
        return True, records_imported, "Successfully imported"
    
    except Exception as e:
        # Record failed migration
        if file_hash:
            try:
                conn.execute("""
                INSERT OR REPLACE INTO migration_tracking
                (file_path, file_hash, file_size, category, migrated_at, status, records_imported, message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    file_path, file_hash, file_info['size'], category,
                    datetime.datetime.now(), 'error', 0, str(e)
                ])
            except:
                pass
        
        logger.error(f"Error migrating file {file_path}: {e}")
        return False, 0, str(e)

def migrate_ci_database(conn, file_info, args):
    """Migrate data from a CI database file"""
    db_path = file_info['path']
    
    logger.debug(f"Migrating CI database: {db_path}")
    
    # If dry run, just return success
    if args.dry_run:
        logger.debug(f"Dry run: would migrate CI database {db_path}")
        return True, 0, "Dry run"
    
    try:
        # Calculate file hash for tracking
        file_hash = calculate_file_hash(db_path)
        if not file_hash:
            return False, 0, "Failed to calculate file hash"
        
        # Connect to the source database
        source_conn = duckdb.connect(db_path, read_only=True)
        
        # Check if required tables exist
        tables = source_conn.execute("SHOW TABLES").fetchall()
        table_names = [t[0].lower() for t in tables]
        
        records_imported = 0
        
        # Migrate test runs
        if 'test_runs' in table_names:
            # Get last run_id in target database
            last_run_id = conn.execute("SELECT COALESCE(MAX(run_id), 0) FROM test_runs").fetchone()[0]
            
            # Get runs from source database
            source_runs = source_conn.execute("""
            SELECT * FROM test_runs WHERE run_id > ?
            """, [last_run_id]).fetchdf()
            
            if not source_runs.empty:
                # Prepare data for insert
                source_runs_dict = source_runs.to_dict('records')
                
                # Insert runs in target database
                for run in source_runs_dict:
                    conn.execute("""
                    INSERT INTO test_runs 
                    (run_id, test_name, test_type, started_at, completed_at, 
                     execution_time_seconds, success, git_commit, git_branch, 
                     command_line, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, [
                        run['run_id'], run['test_name'], run['test_type'], run['started_at'], 
                        run['completed_at'], run['execution_time_seconds'], run['success'], 
                        run['git_commit'], run['git_branch'], run['command_line'], run['metadata']
                    ])
                
                records_imported += len(source_runs)
                logger.info(f"Imported {len(source_runs)} test runs from {db_path}")
        
        # Migrate performance results
        if 'performance_results' in table_names:
            # Get last result_id in target database
            last_result_id = conn.execute("SELECT COALESCE(MAX(result_id), 0) FROM performance_results").fetchone()[0]
            
            # Get results from source database
            source_results = source_conn.execute("""
            SELECT * FROM performance_results WHERE result_id > ?
            """, [last_result_id]).fetchdf()
            
            if not source_results.empty:
                # Prepare data for insert
                source_results_dict = source_results.to_dict('records')
                
                # Insert results in target database
                for result in source_results_dict:
                    conn.execute("""
                    INSERT INTO performance_results 
                    (result_id, run_id, model_id, hardware_id, test_case, batch_size,
                     precision, total_time_seconds, average_latency_ms,
                     throughput_items_per_second, memory_peak_mb,
                     iterations, warmup_iterations, metrics)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, [
                        result['result_id'], result['run_id'], result['model_id'], result['hardware_id'], 
                        result['test_case'], result['batch_size'], result['precision'], 
                        result['total_time_seconds'], result['average_latency_ms'], 
                        result['throughput_items_per_second'], result['memory_peak_mb'],
                        result['iterations'], result['warmup_iterations'], result['metrics']
                    ])
                
                records_imported += len(source_results)
                logger.info(f"Imported {len(source_results)} performance results from {db_path}")
        
        # Migrate models and hardware if needed
        for table_name in ['models', 'hardware_platforms']:
            if table_name in table_names:
                # Get data from source
                source_data = source_conn.execute(f"SELECT * FROM {table_name}").fetchdf()
                
                if not source_data.empty:
                    # For each record in source, check if it exists in target
                    for _, row in source_data.iterrows():
                        if table_name == 'models':
                            # Check if model exists by name
                            exists = conn.execute(
                                "SELECT COUNT(*) FROM models WHERE model_name = ?", 
                                [row['model_name']]
                            ).fetchone()[0]
                            
                            if not exists:
                                # Insert model
                                conn.execute("""
                                INSERT INTO models 
                                (model_name, model_family, modality, source, metadata)
                                VALUES (?, ?, ?, ?, ?)
                                """, [
                                    row['model_name'], row['model_family'], row['modality'], 
                                    row['source'], row['metadata']
                                ])
                                records_imported += 1
                        
                        elif table_name == 'hardware_platforms':
                            # Check if hardware exists by type and name
                            exists = conn.execute(
                                "SELECT COUNT(*) FROM hardware_platforms WHERE hardware_type = ? AND device_name = ?", 
                                [row['hardware_type'], row['device_name']]
                            ).fetchone()[0]
                            
                            if not exists:
                                # Insert hardware
                                conn.execute("""
                                INSERT INTO hardware_platforms 
                                (hardware_type, device_name, platform, platform_version, 
                                 driver_version, memory_gb, compute_units, metadata)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                """, [
                                    row['hardware_type'], row['device_name'], row['platform'], 
                                    row['platform_version'], row['driver_version'], row['memory_gb'], 
                                    row['compute_units'], row['metadata']
                                ])
                                records_imported += 1
        
        # Close source connection
        source_conn.close()
        
        # Record successful migration
        conn.execute("""
        INSERT OR REPLACE INTO migration_tracking
        (file_path, file_hash, file_size, category, migrated_at, status, records_imported, message)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            db_path, file_hash, file_info['size'], 'ci_database',
            datetime.datetime.now(), 'success', records_imported, 'Successfully imported'
        ])
        
        return True, records_imported, "Successfully imported"
    
    except Exception as e:
        # Record failed migration
        if file_hash:
            try:
                conn.execute("""
                INSERT OR REPLACE INTO migration_tracking
                (file_path, file_hash, file_size, category, migrated_at, status, records_imported, message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    db_path, file_hash, file_info['size'], 'ci_database',
                    datetime.datetime.now(), 'error', 0, str(e)
                ])
            except:
                pass
        
        logger.error(f"Error migrating CI database {db_path}: {e}")
        return False, 0, str(e)

def import_performance_data(conn, data, file_path):
    """Import performance benchmark data from a JSON file"""
    records_imported = 0
    
    # Determine the structure of the JSON data
    if isinstance(data, dict) and 'test_runs' in data:
        # This is a test runs collection
        for run in data['test_runs']:
            records_imported += import_performance_run(conn, run)
    
    elif isinstance(data, dict) and 'model_name' in data:
        # This is a single performance test
        records_imported += import_performance_run(conn, data)
    
    elif isinstance(data, list):
        # This is a list of test results
        for item in data:
            if isinstance(item, dict) and ('model_name' in item or 'test_name' in item):
                records_imported += import_performance_run(conn, item)
    
    else:
        # Try to infer from filename and content
        filename = os.path.basename(file_path)
        if 'performance' in filename or 'benchmark' in filename:
            if isinstance(data, dict):
                # Try to import as a single performance test
                records_imported += import_performance_run(conn, data)
            elif isinstance(data, list):
                # Try each item in the list
                for item in data:
                    if isinstance(item, dict):
                        records_imported += import_performance_run(conn, item)
    
    return records_imported

def import_performance_run(conn, data):
    """Import a single performance benchmark run"""
    try:
        # Extract model information
        model_name = data.get('model_name')
        if not model_name:
            model_name = data.get('model', 'unknown')
        
        # Extract hardware information
        hardware_type = data.get('hardware_type')
        if not hardware_type:
            hardware_type = data.get('hardware', 'cpu')
        
        device_name = data.get('device_name')
        
        # Find or create model
        model_family = data.get('model_family')
        if not model_family:
            # Try to determine from model name
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
            else:
                # Default to the first part of the name
                model_family = model_name.split('-')[0].lower()
        
        # Determine modality
        modality = data.get('modality')
        if not modality:
            if model_family in ['vit', 'clip']:
                modality = 'image'
            elif model_family in ['whisper', 'wav2vec']:
                modality = 'audio'
            elif model_family in ['llava']:
                modality = 'multimodal'
            else:
                modality = 'text'
        
        # Find or create model
        model_result = conn.execute(
            "SELECT model_id FROM models WHERE model_name = ?", 
            [model_name]
        ).fetchone()
        
        if model_result:
            model_id = model_result[0]
        else:
            # Create model
            conn.execute("""
            INSERT INTO models (model_name, model_family, modality, source, metadata)
            VALUES (?, ?, ?, ?, ?)
            """, [model_name, model_family, modality, 'huggingface', '{}'])
            
            model_id = conn.execute(
                "SELECT model_id FROM models WHERE model_name = ?", 
                [model_name]
            ).fetchone()[0]
        
        # Find or create hardware
        hardware_query = "SELECT hardware_id FROM hardware_platforms WHERE hardware_type = ?"
        hardware_params = [hardware_type]
        
        if device_name:
            hardware_query += " AND device_name = ?"
            hardware_params.append(device_name)
        
        hardware_result = conn.execute(hardware_query, hardware_params).fetchone()
        
        if hardware_result:
            hardware_id = hardware_result[0]
        else:
            # If no exact match with device name, try just the type
            if device_name:
                hardware_result = conn.execute(
                    "SELECT hardware_id FROM hardware_platforms WHERE hardware_type = ?",
                    [hardware_type]
                ).fetchone()
                
                if hardware_result:
                    hardware_id = hardware_result[0]
                else:
                    # Create default hardware entry
                    conn.execute("""
                    INSERT INTO hardware_platforms 
                    (hardware_type, device_name, platform, platform_version, 
                     driver_version, memory_gb, compute_units, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, [
                        hardware_type, 
                        device_name or f"{hardware_type.upper()} Device", 
                        hardware_type.upper(), 
                        "Unknown", 
                        None, 
                        0, 
                        0, 
                        '{}'
                    ])
                    
                    hardware_id = conn.execute(hardware_query, hardware_params).fetchone()[0]
            else:
                # Create default hardware entry without device name
                conn.execute("""
                INSERT INTO hardware_platforms 
                (hardware_type, device_name, platform, platform_version, 
                 driver_version, memory_gb, compute_units, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    hardware_type, 
                    f"{hardware_type.upper()} Device", 
                    hardware_type.upper(), 
                    "Unknown", 
                    None, 
                    0, 
                    0, 
                    '{}'
                ])
                
                hardware_id = conn.execute(
                    "SELECT hardware_id FROM hardware_platforms WHERE hardware_type = ?",
                    [hardware_type]
                ).fetchone()[0]
        
        # Create test run
        test_name = data.get('test_name')
        if not test_name:
            # Generate a test name from model and timestamp
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            test_name = f"performance_{model_name}_{hardware_type}_{timestamp}"
        
        # Get test timestamp
        test_timestamp = data.get('timestamp')
        if test_timestamp:
            try:
                # Try to parse timestamp from string
                started_at = datetime.datetime.fromisoformat(test_timestamp)
            except:
                # Default to file timestamp or now
                started_at = datetime.datetime.now()
        else:
            started_at = datetime.datetime.now()
        
        # Default completed_at to started_at if not available
        completed_at = data.get('completed_at')
        if completed_at:
            try:
                completed_at = datetime.datetime.fromisoformat(completed_at)
            except:
                completed_at = started_at
        else:
            completed_at = started_at
        
        # Get execution time
        execution_time = data.get('execution_time_seconds')
        if not execution_time:
            # Try to calculate from timestamps
            if completed_at and started_at:
                execution_time = (completed_at - started_at).total_seconds()
            else:
                execution_time = 0
        
        # Create run
        conn.execute("""
        INSERT INTO test_runs 
        (test_name, test_type, started_at, completed_at, 
         execution_time_seconds, success, git_commit, git_branch, 
         command_line, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            test_name, 
            'performance', 
            started_at, 
            completed_at, 
            execution_time, 
            True, 
            data.get('git_commit'), 
            data.get('git_branch'), 
            data.get('command_line'), 
            json.dumps(data.get('metadata', {}))
        ])
        
        # Get the run ID
        run_id = conn.execute(
            "SELECT run_id FROM test_runs WHERE test_name = ? AND started_at = ?", 
            [test_name, started_at]
        ).fetchone()[0]
        
        # Get results data
        results = data.get('results', [])
        if not results and 'test_case' in data:
            # This is a single result directly in the data
            results = [data]
        elif not results and 'batch_size' in data:
            # This is a single result without a test_case field
            results = [data]
        
        # Import each result
        records_imported = 0
        
        for result in results:
            # Get required fields with defaults
            test_case = result.get('test_case', 'embedding')
            batch_size = result.get('batch_size', 1)
            precision = result.get('precision', 'fp32')
            
            # Performance metrics
            total_time_seconds = result.get('total_time_seconds', 0)
            average_latency_ms = result.get('average_latency_ms')
            if average_latency_ms is None:
                average_latency_ms = result.get('latency', 0)
            
            throughput = result.get('throughput_items_per_second')
            if throughput is None:
                throughput = result.get('throughput', 0)
            
            memory_peak_mb = result.get('memory_peak_mb')
            if memory_peak_mb is None:
                memory_peak_mb = result.get('memory_peak', 0)
            
            # Test parameters
            iterations = result.get('iterations', 1)
            warmup_iterations = result.get('warmup_iterations', 0)
            
            # Add performance result
            conn.execute("""
            INSERT INTO performance_results 
            (run_id, model_id, hardware_id, test_case, batch_size,
             precision, total_time_seconds, average_latency_ms,
             throughput_items_per_second, memory_peak_mb,
             iterations, warmup_iterations, metrics)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                run_id, 
                model_id, 
                hardware_id, 
                test_case, 
                batch_size, 
                precision, 
                total_time_seconds, 
                average_latency_ms, 
                throughput, 
                memory_peak_mb, 
                iterations, 
                warmup_iterations, 
                '{}'
            ])
            
            records_imported += 1
        
        return records_imported
    
    except Exception as e:
        logger.error(f"Error importing performance run: {e}")
        return 0

def import_hardware_data(conn, data, file_path):
    """Import hardware compatibility data from a JSON file"""
    # To be implemented
    return 0

def import_integration_data(conn, data, file_path):
    """Import integration test data from a JSON file"""
    # To be implemented
    return 0

def import_model_data(conn, data, file_path):
    """Import model test data from a JSON file"""
    # To be implemented
    return 0

def archive_file(file_path, archive_dir):
    """Archive a file by moving it to the archive directory"""
    try:
        # Create the archive directory if it doesn't exist
        os.makedirs(archive_dir, exist_ok=True)
        
        # Get the relative path within the repo
        try:
            rel_path = os.path.relpath(file_path, os.getcwd())
        except:
            # Fallback to just the basename if relpath fails
            rel_path = os.path.basename(file_path)
        
        # Create the archive path
        archive_path = os.path.join(archive_dir, rel_path)
        
        # Create parent directories if needed
        os.makedirs(os.path.dirname(archive_path), exist_ok=True)
        
        # Move the file
        shutil.move(file_path, archive_path)
        logger.debug(f"Archived {file_path} to {archive_path}")
        
        return archive_path
    except Exception as e:
        logger.error(f"Error archiving file {file_path}: {e}")
        return None

def validate_database(conn, args):
    """Validate the database for consistency and completeness"""
    logger.info("Validating database...")
    
    issues = []
    fixes_applied = 0
    
    # Check for orphaned performance results
    orphaned_count = conn.execute("""
    SELECT COUNT(*) FROM performance_results pr
    LEFT JOIN test_runs tr ON pr.run_id = tr.run_id
    WHERE tr.run_id IS NULL
    """).fetchone()[0]
    
    if orphaned_count > 0:
        issues.append(f"Found {orphaned_count} performance results with missing test runs")
        
        if args.fix_inconsistencies:
            logger.info(f"Fixing orphaned performance results...")
            # Create placeholder test runs for orphaned results
            conn.execute("""
            INSERT INTO test_runs (test_name, test_type, started_at, completed_at, success)
            SELECT 'restored_run_' || pr.run_id, 'performance', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, TRUE
            FROM performance_results pr
            LEFT JOIN test_runs tr ON pr.run_id = tr.run_id
            WHERE tr.run_id IS NULL
            GROUP BY pr.run_id
            """)
            fixes_applied += 1
    
    # Check for missing model or hardware references
    missing_refs = conn.execute("""
    SELECT 
        SUM(CASE WHEN m.model_id IS NULL THEN 1 ELSE 0 END) as missing_models,
        SUM(CASE WHEN h.hardware_id IS NULL THEN 1 ELSE 0 END) as missing_hardware
    FROM performance_results pr
    LEFT JOIN models m ON pr.model_id = m.model_id
    LEFT JOIN hardware_platforms h ON pr.hardware_id = h.hardware_id
    """).fetchone()
    
    missing_models, missing_hardware = missing_refs
    
    if missing_models > 0:
        issues.append(f"Found {missing_models} performance results with missing model references")
        
        if args.fix_inconsistencies:
            logger.info(f"Creating placeholder models for missing references...")
            # Add placeholder model entries
            conn.execute("""
            INSERT INTO models (model_id, model_name, model_family, modality)
            SELECT DISTINCT pr.model_id, 'unknown_model_' || pr.model_id, 'unknown', 'unknown'
            FROM performance_results pr
            LEFT JOIN models m ON pr.model_id = m.model_id
            WHERE m.model_id IS NULL
            """)
            fixes_applied += 1
    
    if missing_hardware > 0:
        issues.append(f"Found {missing_hardware} performance results with missing hardware references")
        
        if args.fix_inconsistencies:
            logger.info(f"Creating placeholder hardware for missing references...")
            # Add placeholder hardware entries
            conn.execute("""
            INSERT INTO hardware_platforms (hardware_id, hardware_type, device_name)
            SELECT DISTINCT pr.hardware_id, 'unknown', 'unknown_device_' || pr.hardware_id
            FROM performance_results pr
            LEFT JOIN hardware_platforms h ON pr.hardware_id = h.hardware_id
            WHERE h.hardware_id IS NULL
            """)
            fixes_applied += 1
    
    # Check for duplicate model entries
    duplicate_models = conn.execute("""
    SELECT model_name, COUNT(*) as count
    FROM models
    GROUP BY model_name
    HAVING COUNT(*) > 1
    """).fetchall()
    
    if duplicate_models:
        model_names = [f"{row[0]} ({row[1]} entries)" for row in duplicate_models]
        issues.append(f"Found duplicate model entries: {', '.join(model_names)}")
        
        if args.fix_inconsistencies:
            logger.info(f"Fixing duplicate model entries...")
            # For each duplicate model, keep the one with the lowest ID
            for model_name, _ in duplicate_models:
                # Get all IDs for this model
                model_ids = conn.execute("""
                SELECT model_id FROM models WHERE model_name = ?
                ORDER BY model_id
                """, [model_name]).fetchall()
                
                # Keep the first one, remove others
                keep_id = model_ids[0][0]
                remove_ids = [row[0] for row in model_ids[1:]]
                
                # Update references to removed models
                for old_id in remove_ids:
                    conn.execute("""
                    UPDATE performance_results 
                    SET model_id = ?
                    WHERE model_id = ?
                    """, [keep_id, old_id])
                
                # Delete the duplicate models
                for old_id in remove_ids:
                    conn.execute("DELETE FROM models WHERE model_id = ?", [old_id])
            
            fixes_applied += 1
    
    # Check for duplicate hardware entries
    duplicate_hardware = conn.execute("""
    SELECT hardware_type, device_name, COUNT(*) as count
    FROM hardware_platforms
    GROUP BY hardware_type, device_name
    HAVING COUNT(*) > 1
    """).fetchall()
    
    if duplicate_hardware:
        hw_names = [f"{row[0]}/{row[1]} ({row[2]} entries)" for row in duplicate_hardware]
        issues.append(f"Found duplicate hardware entries: {', '.join(hw_names)}")
        
        if args.fix_inconsistencies:
            logger.info(f"Fixing duplicate hardware entries...")
            # For each duplicate hardware, keep the one with the lowest ID
            for hw_type, device_name, _ in duplicate_hardware:
                # Get all IDs for this hardware
                hw_ids = conn.execute("""
                SELECT hardware_id FROM hardware_platforms 
                WHERE hardware_type = ? AND device_name = ?
                ORDER BY hardware_id
                """, [hw_type, device_name]).fetchall()
                
                # Keep the first one, remove others
                keep_id = hw_ids[0][0]
                remove_ids = [row[0] for row in hw_ids[1:]]
                
                # Update references to removed hardware
                for old_id in remove_ids:
                    conn.execute("""
                    UPDATE performance_results 
                    SET hardware_id = ?
                    WHERE hardware_id = ?
                    """, [keep_id, old_id])
                
                # Delete the duplicate hardware
                for old_id in remove_ids:
                    conn.execute("DELETE FROM hardware_platforms WHERE hardware_id = ?", [old_id])
            
            fixes_applied += 1
    
    # Report validation results
    if issues:
        logger.warning(f"Found {len(issues)} issues during validation:")
        for i, issue in enumerate(issues):
            logger.warning(f"  {i+1}. {issue}")
        
        if fixes_applied > 0:
            logger.info(f"Applied {fixes_applied} fixes to resolve issues")
        elif args.fix_inconsistencies:
            logger.info("No fixes were needed or could be applied automatically")
    else:
        logger.info("Database validation completed with no issues found")
    
    return len(issues) == 0

def main():
    args = parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Connect to the database
    conn = setup_database(args.db)
    if not conn:
        logger.error("Failed to connect to database")
        sys.exit(1)
    
    # Start a transaction
    conn.begin()
    
    try:
        # Run validation if requested
        if args.validate:
            validate_database(conn, args)
            conn.commit()
            return
        
        files_to_process = []
        
        # Find files to process
        if args.migrate_ci:
            files_to_process = find_ci_artifacts(args)
        else:
            files_to_process = find_json_files(args)
        
        if not files_to_process:
            logger.info("No files found to process")
            conn.commit()
            return
        
        # Process files
        success_count = 0
        error_count = 0
        records_imported = 0
        
        # Process in batches to commit periodically
        batch_size = args.batch_size
        for i in range(0, len(files_to_process), batch_size):
            batch = files_to_process[i:i+batch_size]
            
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(files_to_process) + batch_size - 1)//batch_size} ({len(batch)} files)")
            
            # Process batch
            if args.parallel > 1:
                # TODO: Implement parallel processing
                # This requires careful handling of database connections
                # For now, fall back to sequential processing
                logger.warning("Parallel processing not yet implemented, falling back to sequential")
                process_sequentially = True
            else:
                process_sequentially = True
            
            if process_sequentially:
                for file_info in batch:
                    # Check if already migrated
                    already_migrated, reason = check_migration_status(conn, file_info, args)
                    if already_migrated:
                        logger.info(f"Skipping {file_info['path']}: {reason}")
                        continue
                    
                    # Process the file
                    if file_info['category'] == 'ci_database':
                        success, imported, message = migrate_ci_database(conn, file_info, args)
                    else:
                        success, imported, message = migrate_json_file(conn, file_info, args)
                    
                    if success:
                        success_count += 1
                        records_imported += imported
                        if not args.summary_only:
                            logger.info(f"Successfully processed: {file_info['path']} ({imported} records)")
                    else:
                        error_count += 1
                        if not args.summary_only:
                            logger.error(f"Failed to process: {file_info['path']} - {message}")
            
            # Commit after each batch
            conn.commit()
            conn.begin()
        
        # Commit final changes
        conn.commit()
        
        # Print summary
        logger.info(f"\nMigration complete:")
        logger.info(f"  - Total files: {len(files_to_process)}")
        logger.info(f"  - Successfully processed: {success_count}")
        logger.info(f"  - Records imported: {records_imported}")
        logger.info(f"  - Errors: {error_count}")
        
        if args.action == 'archive':
            logger.info(f"  - Files archived to: {args.archive_dir}")
        elif args.action == 'remove':
            logger.info(f"  - Files removed: {success_count}")
        
    except Exception as e:
        # Rollback on error
        logger.error(f"Error during migration: {e}")
        conn.rollback()
        sys.exit(1)
    finally:
        # Close database connection
        conn.close()

if __name__ == "__main__":
    main()