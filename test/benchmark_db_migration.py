#!/usr/bin/env python
"""
Benchmark Database Migration Tool for the IPFS Accelerate Python Framework.

This tool implements a comprehensive data migration pipeline for moving JSON-based test results
into the structured DuckDB/Parquet database system created for Phase 16.

The migration process handles:
1. Parsing and extracting data from diverse JSON formats
2. Normalizing data to fit the new database schema
3. Deduplicating entries while preserving history
4. Validating data integrity before insertion
5. Managing incremental migrations with change tracking

Usage:
    python benchmark_db_migration.py --input-dirs archived_test_results performance_results --output-db ./benchmark_db.duckdb
    python benchmark_db_migration.py --reindex-models --output-db ./benchmark_db.duckdb
    python benchmark_db_migration.py --input-file performance_results/latest_benchmark.json --incremental
"""

import os
import sys
import json
import glob
import logging
import argparse
import datetime
import hashlib
import shutil
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from pathlib import Path
from collections import defaultdict

try:
    import duckdb
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    print("Error: Required packages not installed. Please install with:")
    print("pip install duckdb pandas pyarrow")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path for importing modules
parent_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

class BenchmarkDBMigration:
    """
    Implements a comprehensive data migration pipeline for moving JSON-based test results
    into the structured DuckDB/Parquet database system.
    """
    
    def __init__(self, output_db: str = "./benchmark_db.duckdb", debug: bool = False):
        """
        Initialize the benchmark database migration tool.
        
        Args:
            output_db: Path to the output DuckDB database
            debug: Enable debug logging
        """
        self.output_db = output_db
        self.migration_log_dir = os.path.join(os.path.dirname(output_db), "migration_logs")
        self.processed_files = set()
        self.migrated_files_log = os.path.join(self.migration_log_dir, "migrated_files.json")
        
        # Set up logging
        if debug:
            logger.setLevel(logging.DEBUG)
        
        # Ensure the migration log directory exists
        os.makedirs(self.migration_log_dir, exist_ok=True)
        
        # Load previously migrated files if available
        if os.path.exists(self.migrated_files_log):
            try:
                with open(self.migrated_files_log, 'r') as f:
                    self.processed_files = set(json.load(f))
                logger.info(f"Loaded {len(self.processed_files)} previously migrated files")
            except Exception as e:
                logger.warning(f"Error loading previously migrated files: {e}")
        
        # Mappings for model and hardware data
        self.model_lookup = {}
        self.hardware_lookup = {}
        self.run_id_counter = 0
        
        # Connect to the database
        self._init_db_connection()
    
    def _init_db_connection(self):
        """Initialize the database connection and load existing mappings"""
        try:
            # Check if the database exists
            db_exists = os.path.exists(self.output_db)
            
            # Connect to the database
            self.conn = duckdb.connect(self.output_db)
            
            # Initialize database with schema if it doesn't exist
            if not db_exists:
                logger.info(f"Database doesn't exist. Creating schema at {self.output_db}")
                self._create_schema()
            
            # Load existing model and hardware mappings
            self._load_mappings()
            
            logger.info(f"Connected to database: {self.output_db}")
            logger.info(f"Loaded {len(self.model_lookup)} models and {len(self.hardware_lookup)} hardware platforms")
            
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            sys.exit(1)
    
    def _create_schema(self):
        """Create the database schema if it doesn't exist"""
        try:
            # Attempt to execute the create_benchmark_schema.py script
            scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
            schema_script = os.path.join(scripts_dir, "create_benchmark_schema.py")
            
            if os.path.exists(schema_script):
                # Import and use the schema creation function
                sys.path.append(scripts_dir)
                from create_benchmark_schema import create_common_tables, create_performance_tables
                from create_benchmark_schema import create_hardware_compatibility_tables
                from create_benchmark_schema import create_integration_test_tables
                from create_benchmark_schema import create_views
                
                # Create the schema
                create_common_tables(self.conn)
                create_performance_tables(self.conn)
                create_hardware_compatibility_tables(self.conn)
                create_integration_test_tables(self.conn)
                create_views(self.conn)
                
                logger.info("Created database schema using create_benchmark_schema.py")
            else:
                # Fallback to creating tables directly
                logger.warning("create_benchmark_schema.py not found, creating basic schema manually")
                self._create_basic_schema()
        
        except Exception as e:
            logger.error(f"Error creating schema: {e}")
            # Fallback to basic schema
            self._create_basic_schema()
    
    def _create_basic_schema(self):
        """Create a basic database schema if the schema script is not available"""
        # Create common dimension tables
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS hardware_platforms (
            hardware_id INTEGER PRIMARY KEY,
            hardware_type VARCHAR NOT NULL,
            device_name VARCHAR,
            platform VARCHAR,
            driver_version VARCHAR,
            memory_gb FLOAT,
            compute_units INTEGER,
            metadata JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS models (
            model_id INTEGER PRIMARY KEY,
            model_name VARCHAR NOT NULL,
            model_family VARCHAR,
            modality VARCHAR,
            source VARCHAR,
            version VARCHAR,
            parameters_million FLOAT,
            metadata JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS test_runs (
            run_id INTEGER PRIMARY KEY,
            test_name VARCHAR NOT NULL,
            test_type VARCHAR NOT NULL,
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
        
        # Create performance results table
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS performance_results (
            result_id INTEGER PRIMARY KEY,
            run_id INTEGER NOT NULL,
            model_id INTEGER NOT NULL,
            hardware_id INTEGER NOT NULL,
            test_case VARCHAR NOT NULL,
            batch_size INTEGER DEFAULT 1,
            precision VARCHAR,
            total_time_seconds FLOAT,
            average_latency_ms FLOAT,
            throughput_items_per_second FLOAT,
            memory_peak_mb FLOAT,
            iterations INTEGER,
            warmup_iterations INTEGER,
            metrics JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (run_id) REFERENCES test_runs(run_id),
            FOREIGN KEY (model_id) REFERENCES models(model_id),
            FOREIGN KEY (hardware_id) REFERENCES hardware_platforms(hardware_id)
        )
        """)
        
        # Create hardware compatibility table
        self.conn.execute("""
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
            compatibility_score FLOAT,
            metadata JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (run_id) REFERENCES test_runs(run_id),
            FOREIGN KEY (model_id) REFERENCES models(model_id),
            FOREIGN KEY (hardware_id) REFERENCES hardware_platforms(hardware_id)
        )
        """)
        
        logger.info("Created basic database schema")
    
    def _load_mappings(self):
        """Load existing model and hardware mappings from the database"""
        try:
            # Load models
            models_df = self.conn.execute("SELECT model_id, model_name FROM models").fetchdf()
            for _, row in models_df.iterrows():
                self.model_lookup[row['model_name']] = row['model_id']
            
            # Load hardware platforms
            hardware_df = self.conn.execute(
                "SELECT hardware_id, hardware_type, device_name FROM hardware_platforms").fetchdf()
            for _, row in hardware_df.iterrows():
                key = f"{row['hardware_type']}|{row['device_name']}"
                self.hardware_lookup[key] = row['hardware_id']
            
            # Get the max run_id to continue from there
            max_run_id = self.conn.execute("SELECT MAX(run_id) FROM test_runs").fetchone()[0]
            self.run_id_counter = max_run_id if max_run_id is not None else 0
            
        except Exception as e:
            logger.error(f"Error loading mappings: {e}")
    
    def add_model(self, model_data: Dict[str, Any]) -> int:
        """
        Add a model to the database or get its ID if it already exists.
        
        Args:
            model_data: Dictionary of model information
            
        Returns:
            The model_id
        """
        model_name = model_data.get('model_name', '').strip()
        if not model_name:
            logger.warning("Attempted to add model with empty name")
            model_name = "unknown_model"
        
        # Check if model already exists
        if model_name in self.model_lookup:
            return self.model_lookup[model_name]
        
        # Get the next model_id
        try:
            max_id = self.conn.execute("SELECT MAX(model_id) FROM models").fetchone()[0]
            model_id = max_id + 1 if max_id is not None else 1
        except Exception:
            # Table might be empty
            model_id = 1
        
        # Prepare the model data
        model_family = model_data.get('model_family', self._infer_model_family(model_name))
        modality = model_data.get('modality', self._infer_modality(model_name, model_family))
        source = model_data.get('source', 'huggingface' if 'huggingface' in model_name or 'hf' in model_name else 'unknown')
        version = model_data.get('version', '1.0')
        parameters = model_data.get('parameters_million', 0.0)
        metadata = model_data.get('metadata', {})
        
        # Insert the model
        self.conn.execute("""
        INSERT INTO models (model_id, model_name, model_family, modality, source, version, parameters_million, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [model_id, model_name, model_family, modality, source, version, parameters, json.dumps(metadata)])
        
        # Add to lookup
        self.model_lookup[model_name] = model_id
        
        logger.debug(f"Added model: {model_name} (ID: {model_id})")
        return model_id
    
    def get_or_add_model(self, model_name: str, model_family: str = None) -> int:
        """
        Get the model ID or add it if it doesn't exist.
        
        Args:
            model_name: Name of the model
            model_family: Optional model family
            
        Returns:
            The model_id
        """
        if not model_name:
            logger.warning("Attempted to get/add model with empty name")
            model_name = "unknown_model"
        
        # Check if model already exists
        if model_name in self.model_lookup:
            return self.model_lookup[model_name]
        
        # Prepare model data
        model_data = {
            'model_name': model_name,
            'model_family': model_family or self._infer_model_family(model_name)
        }
        
        # Add the model
        return self.add_model(model_data)
    
    def add_hardware_platform(self, hardware_data: Dict[str, Any]) -> int:
        """
        Add a hardware platform to the database or get its ID if it already exists.
        
        Args:
            hardware_data: Dictionary of hardware information
            
        Returns:
            The hardware_id
        """
        hardware_type = hardware_data.get('hardware_type', '').lower()
        device_name = hardware_data.get('device_name', 'unknown')
        
        # Create a lookup key
        key = f"{hardware_type}|{device_name}"
        
        # Check if hardware already exists
        if key in self.hardware_lookup:
            return self.hardware_lookup[key]
        
        # Get the next hardware_id
        try:
            max_id = self.conn.execute("SELECT MAX(hardware_id) FROM hardware_platforms").fetchone()[0]
            hardware_id = max_id + 1 if max_id is not None else 1
        except Exception:
            # Table might be empty
            hardware_id = 1
        
        # Prepare the hardware data
        platform = hardware_data.get('platform', '')
        driver_version = hardware_data.get('driver_version', '')
        memory_gb = hardware_data.get('memory_gb', 0.0)
        compute_units = hardware_data.get('compute_units', 0)
        metadata = hardware_data.get('metadata', {})
        
        # Insert the hardware platform
        self.conn.execute("""
        INSERT INTO hardware_platforms 
        (hardware_id, hardware_type, device_name, platform, driver_version, memory_gb, compute_units, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [hardware_id, hardware_type, device_name, platform, driver_version, 
             memory_gb, compute_units, json.dumps(metadata)])
        
        # Add to lookup
        self.hardware_lookup[key] = hardware_id
        
        logger.debug(f"Added hardware platform: {hardware_type} - {device_name} (ID: {hardware_id})")
        return hardware_id
    
    def get_or_add_hardware(self, hardware_type: str, device_name: str = None) -> int:
        """
        Get the hardware ID or add it if it doesn't exist.
        
        Args:
            hardware_type: Type of hardware ('cpu', 'cuda', etc.)
            device_name: Optional device name
            
        Returns:
            The hardware_id
        """
        hardware_type = hardware_type.lower() if hardware_type else 'unknown'
        device_name = device_name or self._default_device_name(hardware_type)
        
        # Create a lookup key
        key = f"{hardware_type}|{device_name}"
        
        # Check if hardware already exists
        if key in self.hardware_lookup:
            return self.hardware_lookup[key]
        
        # Prepare hardware data
        hardware_data = {
            'hardware_type': hardware_type,
            'device_name': device_name
        }
        
        # Add the hardware platform
        return self.add_hardware_platform(hardware_data)
    
    def add_test_run(self, run_data: Dict[str, Any]) -> int:
        """
        Add a test run to the database.
        
        Args:
            run_data: Dictionary of test run information
            
        Returns:
            The run_id
        """
        # Increment the run_id counter
        self.run_id_counter += 1
        run_id = self.run_id_counter
        
        # Prepare the test run data
        test_name = run_data.get('test_name', 'unknown_test')
        test_type = run_data.get('test_type', 'unknown')
        started_at = run_data.get('started_at')
        completed_at = run_data.get('completed_at')
        execution_time = run_data.get('execution_time_seconds', 0.0)
        success = run_data.get('success', True)
        git_commit = run_data.get('git_commit', '')
        git_branch = run_data.get('git_branch', '')
        command_line = run_data.get('command_line', '')
        metadata = run_data.get('metadata', {})
        
        # Parse timestamps
        if isinstance(started_at, str):
            started_at = self._parse_timestamp(started_at)
        if isinstance(completed_at, str):
            completed_at = self._parse_timestamp(completed_at)
        
        # Insert the test run
        self.conn.execute("""
        INSERT INTO test_runs 
        (run_id, test_name, test_type, started_at, completed_at, execution_time_seconds, 
         success, git_commit, git_branch, command_line, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [run_id, test_name, test_type, started_at, completed_at, execution_time, 
              success, git_commit, git_branch, command_line, json.dumps(metadata)])
        
        logger.debug(f"Added test run: {test_name} (ID: {run_id})")
        return run_id
    
    def migrate_file(self, file_path: str, incremental: bool = False) -> Dict[str, int]:
        """
        Migrate a single JSON file to the database.
        
        Args:
            file_path: Path to the JSON file
            incremental: If True, only migrate if file hasn't been processed before
            
        Returns:
            Dictionary with counts of migrated items by type
        """
        # Check if file has been processed before
        file_path = os.path.abspath(file_path)
        if incremental and file_path in self.processed_files:
            logger.info(f"Skipping already migrated file: {file_path}")
            return {'skipped': 1}
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Detect the file type
            file_type = self._detect_file_type(data, file_path)
            
            if file_type == 'unknown':
                logger.warning(f"Unknown file type: {file_path}")
                return {'unknown': 1}
            
            # Process based on file type
            counts = {}
            
            if file_type == 'performance':
                counts = self._migrate_performance_data(data, file_path)
            elif file_type == 'hardware':
                counts = self._migrate_hardware_data(data, file_path)
            elif file_type == 'compatibility':
                counts = self._migrate_compatibility_data(data, file_path)
            elif file_type == 'integration':
                counts = self._migrate_integration_data(data, file_path)
            
            # Mark file as processed
            self.processed_files.add(file_path)
            self._save_processed_files()
            
            # Generate a summary log
            summary = {
                'file_path': file_path,
                'file_type': file_type,
                'migrated_at': datetime.datetime.now().isoformat(),
                'counts': counts
            }
            
            # Save summary to log file
            log_file = os.path.join(
                self.migration_log_dir, 
                f"migration_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.path.basename(file_path)}.json"
            )
            with open(log_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            return counts
            
        except Exception as e:
            logger.error(f"Error migrating file {file_path}: {e}")
            return {'error': 1}
    
    def migrate_directory(self, directory: str, recursive: bool = True, 
                        incremental: bool = True) -> Dict[str, int]:
        """
        Migrate all JSON files in a directory to the database.
        
        Args:
            directory: Directory containing JSON files
            recursive: If True, search subdirectories
            incremental: If True, only migrate files that haven't been processed before
            
        Returns:
            Dictionary with counts of migrated items by type
        """
        # Find all JSON files
        pattern = os.path.join(directory, "**/*.json") if recursive else os.path.join(directory, "*.json")
        json_files = glob.glob(pattern, recursive=recursive)
        
        logger.info(f"Found {len(json_files)} JSON files in {directory}")
        
        # Process each file
        total_counts = defaultdict(int)
        for file_path in json_files:
            counts = self.migrate_file(file_path, incremental)
            for key, count in counts.items():
                total_counts[key] += count
        
        # Log the result
        log_message = f"Migrated directory {directory}: "
        log_message += ", ".join([f"{count} {item}" for item, count in total_counts.items()])
        logger.info(log_message)
        
        return dict(total_counts)
    
    def cleanup_json_files(self, older_than_days: int = None, move_to: str = None, 
                         delete: bool = False) -> int:
        """
        Clean up JSON files that have been migrated.
        
        Args:
            older_than_days: Only process files older than this many days
            move_to: Directory to move files to (None to leave in place)
            delete: If True, delete files instead of moving them
            
        Returns:
            Number of files processed
        """
        if not self.processed_files:
            logger.info("No files have been migrated yet")
            return 0
        
        # Calculate cutoff date if needed
        cutoff_date = None
        if older_than_days is not None:
            cutoff_date = datetime.datetime.now() - datetime.timedelta(days=older_than_days)
        
        count = 0
        for file_path in self.processed_files:
            # Skip files that don't exist
            if not os.path.exists(file_path):
                continue
            
            # Check age if needed
            if cutoff_date is not None:
                mtime = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
                if mtime > cutoff_date:
                    continue
            
            # Process the file
            if delete:
                try:
                    os.remove(file_path)
                    logger.debug(f"Deleted migrated file: {file_path}")
                    count += 1
                except Exception as e:
                    logger.error(f"Error deleting file {file_path}: {e}")
            elif move_to:
                try:
                    # Create target directory if it doesn't exist
                    os.makedirs(move_to, exist_ok=True)
                    
                    # Determine target path
                    rel_path = os.path.basename(file_path)
                    target_path = os.path.join(move_to, rel_path)
                    
                    # Move the file
                    shutil.move(file_path, target_path)
                    logger.debug(f"Moved migrated file: {file_path} -> {target_path}")
                    count += 1
                except Exception as e:
                    logger.error(f"Error moving file {file_path}: {e}")
        
        if delete:
            logger.info(f"Deleted {count} migrated files")
        elif move_to:
            logger.info(f"Moved {count} migrated files to {move_to}")
        else:
            logger.info(f"Found {count} files that could be processed")
        
        return count
    
    def reindex_models(self) -> Dict[str, int]:
        """
        Reindex models by analyzing compatible names and families.
        
        Returns:
            Dictionary with counts of updated items
        """
        # Get all models
        models_df = self.conn.execute("""
        SELECT model_id, model_name, model_family, modality FROM models
        """).fetchdf()
        
        updates = 0
        family_updates = 0
        modality_updates = 0
        
        # Update model families and modalities
        for _, row in models_df.iterrows():
            model_id = row['model_id']
            model_name = row['model_name']
            current_family = row['model_family']
            current_modality = row['modality']
            
            # Infer model family if not set
            if not current_family or current_family == 'unknown':
                new_family = self._infer_model_family(model_name)
                if new_family and new_family != 'unknown':
                    self.conn.execute("""
                    UPDATE models SET model_family = ? WHERE model_id = ?
                    """, [new_family, model_id])
                    family_updates += 1
            
            # Infer modality if not set
            if not current_modality or current_modality == 'unknown':
                new_modality = self._infer_modality(model_name, current_family or self._infer_model_family(model_name))
                if new_modality and new_modality != 'unknown':
                    self.conn.execute("""
                    UPDATE models SET modality = ? WHERE model_id = ?
                    """, [new_modality, model_id])
                    modality_updates += 1
        
        # Handle special cases for popular model families
        family_mapping = {
            'bert': ['bert-base', 'bert-large', 'distilbert', 'roberta'],
            't5': ['t5-small', 't5-base', 't5-large', 't5-efficient'],
            'llama': ['llama', 'llama2', 'llama3', 'opt'],
            'gpt': ['gpt2', 'gpt-neo', 'gpt-j'],
            'clip': ['clip', 'chinese-clip'],
            'vit': ['vit', 'deit'],
            'whisper': ['whisper'],
            'wav2vec2': ['wav2vec2']
        }
        
        # Update models whose family can be inferred from name patterns
        for family, patterns in family_mapping.items():
            for pattern in patterns:
                self.conn.execute("""
                UPDATE models SET model_family = ? 
                WHERE model_family != ? AND model_name LIKE ?
                """, [family, family, f'%{pattern}%'])
            
            # Get number of updates
            count = self.conn.execute("""
            SELECT COUNT(*) FROM models WHERE model_family = ?
            """, [family]).fetchone()[0]
            
            logger.debug(f"Model family '{family}': {count} models")
            updates += count
        
        return {
            'total_models': len(models_df),
            'family_updates': family_updates,
            'modality_updates': modality_updates,
            'total_updates': updates
        }
    
    def _detect_file_type(self, data: Dict, file_path: str) -> str:
        """
        Detect the type of a JSON file based on its content and filename.
        
        Args:
            data: The loaded JSON data
            file_path: Path to the JSON file
            
        Returns:
            File type ('performance', 'hardware', 'compatibility', 'integration', or 'unknown')
        """
        filename = os.path.basename(file_path).lower()
        
        # Check for performance data
        if ('throughput' in data or 'latency' in data or 'performance' in data or 'benchmark' in data or
            'throughput_items_per_second' in data):
            return 'performance'
        elif any(x in filename for x in ['performance', 'benchmark', 'throughput', 'latency']):
            return 'performance'
        
        # Check for hardware data
        if any(k in data for k in ['cuda', 'rocm', 'mps', 'openvino', 'hardware_detection']):
            return 'hardware'
        elif any(x in filename for x in ['hardware', 'device', 'platform']):
            return 'hardware'
        
        # Check for compatibility data
        if any(k in data for k in ['compatibility', 'is_compatible']):
            return 'compatibility'
        elif any(x in filename for x in ['compatibility', 'matrix']):
            return 'compatibility'
        
        # Check for integration test data
        if any(k in data for k in ['test_results', 'assertions', 'tests']):
            return 'integration'
        elif any(x in filename for x in ['test', 'integration']):
            return 'integration'
        
        # Default to unknown
        return 'unknown'
    
    def _migrate_performance_data(self, data: Dict, file_path: str) -> Dict[str, int]:
        """
        Migrate performance benchmark data to the database.
        
        Args:
            data: The loaded JSON data
            file_path: Path to the source file
            
        Returns:
            Dictionary with counts of migrated items
        """
        test_name = os.path.basename(file_path).replace('.json', '')
        timestamp = data.get('timestamp', self._extract_timestamp_from_filename(file_path))
        
        # Create a test run
        run_data = {
            'test_name': test_name,
            'test_type': 'performance',
            'started_at': timestamp,
            'completed_at': timestamp,
            'success': True,
            'metadata': {'source_file': file_path}
        }
        run_id = self.add_test_run(run_data)
        
        # Process results
        results_count = 0
        
        # Handle different file formats
        if 'results' in data and isinstance(data['results'], list):
            # Multiple results format
            for result in data['results']:
                self._add_performance_result(result, data, run_id, file_path)
                results_count += 1
        else:
            # Single result format
            self._add_performance_result(data, {}, run_id, file_path)
            results_count += 1
        
        return {'run': 1, 'results': results_count}
    
    def _add_performance_result(self, result: Dict, parent_data: Dict, run_id: int, file_path: str) -> None:
        """
        Add a single performance result to the database.
        
        Args:
            result: The result data
            parent_data: Parent data for defaults
            run_id: The test run ID
            file_path: Path to the source file
        """
        # Extract model and hardware info
        model_name = result.get('model', parent_data.get('model', 'unknown'))
        hardware_type = result.get('hardware', parent_data.get('hardware', 'cpu'))
        device_name = result.get('device', parent_data.get('device', self._default_device_name(hardware_type)))
        
        # Get or add model and hardware
        model_id = self.get_or_add_model(model_name)
        hardware_id = self.get_or_add_hardware(hardware_type, device_name)
        
        # Extract metrics
        test_case = result.get('test_case', parent_data.get('test_case', self._infer_test_case(model_name)))
        batch_size = int(result.get('batch_size', parent_data.get('batch_size', 1)))
        precision = result.get('precision', parent_data.get('precision', 'fp32'))
        
        # Extract performance metrics
        total_time_seconds = float(result.get('total_time', parent_data.get('total_time', 0.0)))
        avg_latency = float(result.get('latency_avg', result.get('latency', parent_data.get('latency', 0.0))))
        throughput = float(result.get('throughput', parent_data.get('throughput', 0.0)))
        memory_peak = float(result.get('memory_peak', result.get('memory', parent_data.get('memory', 0.0))))
        iterations = int(result.get('iterations', parent_data.get('iterations', 0)))
        warmup_iterations = int(result.get('warmup_iterations', parent_data.get('warmup_iterations', 0)))
        
        # Extract additional metrics
        metrics = {}
        for k, v in result.items():
            if k not in ['model', 'hardware', 'device', 'test_case', 'batch_size', 'precision',
                        'total_time', 'latency_avg', 'latency', 'throughput', 'memory_peak',
                        'memory', 'iterations', 'warmup_iterations']:
                metrics[k] = v
        
        # Add metrics from parent data if not in result
        for k, v in parent_data.items():
            if k not in result and k not in ['model', 'hardware', 'device', 'test_case', 
                                            'batch_size', 'precision', 'total_time', 
                                            'latency_avg', 'latency', 'throughput', 
                                            'memory_peak', 'memory', 'iterations', 
                                            'warmup_iterations', 'results', 'timestamp']:
                metrics[k] = v
        
        # Insert performance result
        try:
            result_id = self.conn.execute("""
            SELECT MAX(result_id) FROM performance_results
            """).fetchone()[0]
            result_id = result_id + 1 if result_id is not None else 1
            
            self.conn.execute("""
            INSERT INTO performance_results 
            (result_id, run_id, model_id, hardware_id, test_case, batch_size, precision,
             total_time_seconds, average_latency_ms, throughput_items_per_second,
             memory_peak_mb, iterations, warmup_iterations, metrics)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [result_id, run_id, model_id, hardware_id, test_case, batch_size, precision,
                 total_time_seconds, avg_latency, throughput, memory_peak, iterations,
                 warmup_iterations, json.dumps(metrics)])
            
        except Exception as e:
            logger.error(f"Error adding performance result: {e}")
    
    def _migrate_hardware_data(self, data: Dict, file_path: str) -> Dict[str, int]:
        """
        Migrate hardware detection data to the database.
        
        Args:
            data: The loaded JSON data
            file_path: Path to the source file
            
        Returns:
            Dictionary with counts of migrated items
        """
        test_name = os.path.basename(file_path).replace('.json', '')
        timestamp = data.get('timestamp', self._extract_timestamp_from_filename(file_path))
        
        # Create a test run
        run_data = {
            'test_name': test_name,
            'test_type': 'hardware',
            'started_at': timestamp,
            'completed_at': timestamp,
            'success': True,
            'metadata': {'source_file': file_path}
        }
        run_id = self.add_test_run(run_data)
        
        # Add hardware platforms
        hardware_count = 0
        
        # Process hardware data
        self._add_hardware_from_data(data, run_id, file_path)
        hardware_count += 1
        
        return {'run': 1, 'hardware': hardware_count}
    
    def _add_hardware_from_data(self, data: Dict, run_id: int, file_path: str) -> None:
        """
        Add hardware platforms from detection data.
        
        Args:
            data: The hardware detection data
            run_id: The test run ID
            file_path: Path to the source file
        """
        # Extract system info
        system_info = data.get('system', {})
        platform = system_info.get('platform', 'unknown')
        
        # Add CPU
        cpu_info = system_info.get('cpu_info', 'Unknown CPU')
        memory_total = float(system_info.get('memory_total', 0.0))
        memory_free = float(system_info.get('memory_free', 0.0))
        
        # Create hardware platform entry for CPU
        cpu_data = {
            'hardware_type': 'cpu',
            'device_name': cpu_info,
            'platform': platform,
            'driver_version': 'n/a',
            'memory_gb': memory_total / 1024 if memory_total > 1024 else memory_total,
            'compute_units': system_info.get('cpu_count', 0),
            'metadata': {
                'memory_free_gb': memory_free / 1024 if memory_free > 1024 else memory_free,
                'architecture': system_info.get('architecture', ''),
                'machine': system_info.get('machine', '')
            }
        }
        self.add_hardware_platform(cpu_data)
        
        # Add CUDA devices
        if 'cuda' in data and data['cuda'] is True and 'cuda_devices' in data:
            for device in data['cuda_devices']:
                device_name = device.get('name', 'Unknown CUDA Device')
                total_memory = float(device.get('total_memory', 0.0))
                free_memory = float(device.get('free_memory', 0.0))
                
                cuda_data = {
                    'hardware_type': 'cuda',
                    'device_name': device_name,
                    'platform': platform,
                    'driver_version': data.get('cuda_driver_version', 'unknown'),
                    'memory_gb': total_memory / 1024 if total_memory > 1024 else total_memory,
                    'compute_units': 0,  # Not directly available
                    'metadata': {
                        'compute_capability': device.get('compute_capability', ''),
                        'memory_free_gb': free_memory / 1024 if free_memory > 1024 else free_memory,
                        'cuda_version': data.get('cuda_version', '')
                    }
                }
                self.add_hardware_platform(cuda_data)
        
        # Add ROCm devices
        if 'rocm' in data and data['rocm'] is True and 'rocm_devices' in data:
            for device in data['rocm_devices']:
                device_name = device.get('name', 'Unknown ROCm Device')
                total_memory = float(device.get('total_memory', 0.0))
                free_memory = float(device.get('free_memory', 0.0))
                
                rocm_data = {
                    'hardware_type': 'rocm',
                    'device_name': device_name,
                    'platform': platform,
                    'driver_version': data.get('rocm_version', 'unknown'),
                    'memory_gb': total_memory / 1024 if total_memory > 1024 else total_memory,
                    'compute_units': 0,  # Not directly available
                    'metadata': {
                        'compute_capability': device.get('compute_capability', ''),
                        'memory_free_gb': free_memory / 1024 if free_memory > 1024 else free_memory,
                        'rocm_version': data.get('rocm_version', '')
                    }
                }
                self.add_hardware_platform(rocm_data)
        
        # Add MPS
        if 'mps' in data and data['mps'] is True:
            mps_data = {
                'hardware_type': 'mps',
                'device_name': 'Apple Silicon',
                'platform': platform,
                'driver_version': 'n/a',
                'memory_gb': 0.0,  # Not directly available
                'compute_units': 0,  # Not directly available
                'metadata': {
                    'mps_version': data.get('mps_version', 'unknown')
                }
            }
            self.add_hardware_platform(mps_data)
        
        # Add OpenVINO
        if 'openvino' in data and data['openvino'] is True:
            openvino_data = {
                'hardware_type': 'openvino',
                'device_name': 'OpenVINO',
                'platform': platform,
                'driver_version': data.get('openvino_version', 'unknown'),
                'memory_gb': 0.0,  # Not directly available
                'compute_units': 0,  # Not directly available
                'metadata': {
                    'openvino_version': data.get('openvino_version', 'unknown')
                }
            }
            self.add_hardware_platform(openvino_data)
        
        # Add WebNN
        if 'webnn' in data and data['webnn'] is True:
            webnn_data = {
                'hardware_type': 'webnn',
                'device_name': 'WebNN',
                'platform': platform,
                'driver_version': 'n/a',
                'memory_gb': 0.0,  # Not directly available
                'compute_units': 0,  # Not directly available
                'metadata': {
                    'browser': data.get('webnn_browser', 'unknown'),
                    'user_agent': data.get('webnn_user_agent', '')
                }
            }
            self.add_hardware_platform(webnn_data)
        
        # Add WebGPU
        if 'webgpu' in data and data['webgpu'] is True:
            webgpu_data = {
                'hardware_type': 'webgpu',
                'device_name': 'WebGPU',
                'platform': platform,
                'driver_version': 'n/a',
                'memory_gb': 0.0,  # Not directly available
                'compute_units': 0,  # Not directly available
                'metadata': {
                    'browser': data.get('webgpu_browser', 'unknown'),
                    'user_agent': data.get('webgpu_user_agent', '')
                }
            }
            self.add_hardware_platform(webgpu_data)
    
    def _migrate_compatibility_data(self, data: Dict, file_path: str) -> Dict[str, int]:
        """
        Migrate hardware compatibility data to the database.
        
        Args:
            data: The loaded JSON data
            file_path: Path to the source file
            
        Returns:
            Dictionary with counts of migrated items
        """
        test_name = os.path.basename(file_path).replace('.json', '')
        timestamp = data.get('timestamp', self._extract_timestamp_from_filename(file_path))
        
        # Create a test run
        run_data = {
            'test_name': test_name,
            'test_type': 'compatibility',
            'started_at': timestamp,
            'completed_at': timestamp,
            'success': True,
            'metadata': {'source_file': file_path}
        }
        run_id = self.add_test_run(run_data)
        
        # Process compatibility data
        compat_count = 0
        
        # Handle different file formats
        if 'tests' in data and isinstance(data['tests'], list):
            # Multiple tests format
            for test in data['tests']:
                compat_count += self._add_compatibility_results(test, run_id, file_path)
        elif 'compatibility' in data and isinstance(data['compatibility'], dict):
            # Single model with multiple hardware compatibility
            compat_count += self._add_compatibility_results(data, run_id, file_path)
        else:
            # Try to extract compatibility from structure
            compat_count += self._add_compatibility_results(data, run_id, file_path)
        
        return {'run': 1, 'compatibility': compat_count}
    
    def _add_compatibility_results(self, data: Dict, run_id: int, file_path: str) -> int:
        """
        Add hardware compatibility results to the database.
        
        Args:
            data: The compatibility data
            run_id: The test run ID
            file_path: Path to the source file
            
        Returns:
            Number of compatibility records added
        """
        model_name = data.get('model', os.path.basename(file_path).split('_')[0])
        model_id = self.get_or_add_model(model_name)
        
        count = 0
        
        # Get compatibility data
        compat_data = data.get('compatibility', {})
        if not compat_data and 'hardware_types' in data:
            # Convert list of hardware types to compatibility dict
            compat_data = {}
            for hw_type in data.get('hardware_types', []):
                is_compatible = data.get(hw_type, False)
                error = data.get(f"{hw_type}_error", '')
                compat_data[hw_type] = {
                    'is_compatible': is_compatible,
                    'error': error
                }
        
        # Process each hardware type
        for hw_type, hw_data in compat_data.items():
            # Skip if not a dict
            if not isinstance(hw_data, dict):
                continue
            
            # Get hardware ID
            device_name = hw_data.get('device_name', self._default_device_name(hw_type))
            hardware_id = self.get_or_add_hardware(hw_type, device_name)
            
            # Extract compatibility info
            is_compatible = hw_data.get('is_compatible', hw_data.get('compatible', False))
            detection_success = hw_data.get('detection_success', True)
            initialization_success = hw_data.get('initialization_success', is_compatible)
            error_message = hw_data.get('error', hw_data.get('error_message', ''))
            error_type = hw_data.get('error_type', '')
            suggested_fix = hw_data.get('suggested_fix', hw_data.get('fix', ''))
            workaround_available = hw_data.get('workaround_available', False)
            compatibility_score = hw_data.get('compatibility_score', 1.0 if is_compatible else 0.0)
            
            # Collect additional metadata
            metadata = {}
            for k, v in hw_data.items():
                if k not in ['is_compatible', 'compatible', 'detection_success', 'initialization_success',
                           'error', 'error_message', 'error_type', 'suggested_fix', 'fix',
                           'workaround_available', 'compatibility_score', 'device_name']:
                    metadata[k] = v
            
            # Add compatibility record
            try:
                compat_id = self.conn.execute("""
                SELECT MAX(compatibility_id) FROM hardware_compatibility
                """).fetchone()[0]
                compat_id = compat_id + 1 if compat_id is not None else 1
                
                self.conn.execute("""
                INSERT INTO hardware_compatibility
                (compatibility_id, run_id, model_id, hardware_id, is_compatible, detection_success,
                 initialization_success, error_message, error_type, suggested_fix,
                 workaround_available, compatibility_score, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [compat_id, run_id, model_id, hardware_id, is_compatible, detection_success,
                      initialization_success, error_message, error_type, suggested_fix,
                      workaround_available, compatibility_score, json.dumps(metadata)])
                
                count += 1
            except Exception as e:
                logger.error(f"Error adding compatibility record: {e}")
        
        return count
    
    def _migrate_integration_data(self, data: Dict, file_path: str) -> Dict[str, int]:
        """
        Migrate integration test data to the database.
        
        Args:
            data: The loaded JSON data
            file_path: Path to the source file
            
        Returns:
            Dictionary with counts of migrated items
        """
        # This is a placeholder for future integration test migration
        # Currently, we don't have a specific structure for integration test results
        return {'skipped_integration': 1}
    
    def _save_processed_files(self) -> None:
        """Save the list of processed files to disk"""
        try:
            with open(self.migrated_files_log, 'w') as f:
                json.dump(list(self.processed_files), f)
        except Exception as e:
            logger.warning(f"Error saving processed files: {e}")
    
    def _parse_timestamp(self, timestamp_str: str) -> datetime.datetime:
        """Parse a timestamp string into a datetime object"""
        if not timestamp_str:
            return datetime.datetime.now()
        
        # Try various formats
        formats = [
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%S.%f',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d %H:%M:%S.%f',
            '%Y%m%d_%H%M%S'
        ]
        
        for fmt in formats:
            try:
                return datetime.datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue
        
        # If all formats fail, return current time
        logger.warning(f"Could not parse timestamp: {timestamp_str}")
        return datetime.datetime.now()
    
    def _extract_timestamp_from_filename(self, file_path: str) -> str:
        """Extract a timestamp from a filename if possible"""
        filename = os.path.basename(file_path)
        
        # Look for patterns like 20250301_173742
        import re
        timestamp_match = re.search(r'(\d{8}_\d{6})', filename)
        if timestamp_match:
            return timestamp_match.group(1)
        
        # Use file modification time as fallback
        try:
            mtime = os.path.getmtime(file_path)
            return datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%dT%H:%M:%S')
        except Exception:
            return datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    
    def _infer_model_family(self, model_name: str) -> str:
        """Infer the model family from the model name"""
        model_name = model_name.lower()
        
        # Common model families
        if 'bert' in model_name:
            return 'bert'
        elif 't5' in model_name:
            return 't5'
        elif 'gpt' in model_name:
            return 'gpt'
        elif 'llama' in model_name or 'opt' in model_name:
            return 'llama'
        elif 'clip' in model_name:
            return 'clip'
        elif 'vit' in model_name or 'vision' in model_name:
            return 'vit'
        elif 'whisper' in model_name:
            return 'whisper'
        elif 'wav2vec' in model_name:
            return 'wav2vec2'
        elif 'llava' in model_name:
            return 'llava'
        elif 'qwen' in model_name:
            return 'qwen'
        elif 'detr' in model_name:
            return 'detr'
        elif 'clap' in model_name:
            return 'clap'
        elif 'xclip' in model_name:
            return 'xclip'
        
        # Default
        return 'unknown'
    
    def _infer_modality(self, model_name: str, model_family: str) -> str:
        """Infer the modality from the model name and family"""
        model_name = model_name.lower()
        model_family = model_family.lower()
        
        # Text models
        if model_family in ['bert', 't5', 'gpt', 'llama']:
            return 'text'
        
        # Vision models
        if model_family in ['vit', 'detr']:
            return 'image'
        
        # Audio models
        if model_family in ['whisper', 'wav2vec2', 'clap']:
            return 'audio'
        
        # Vision-language models
        if model_family in ['clip', 'xclip']:
            return 'image_text'
        
        # Multimodal models
        if model_family in ['llava']:
            return 'multimodal'
        
        # Check for keywords in name
        if 'text' in model_name or 'bert' in model_name or 't5' in model_name:
            return 'text'
        elif 'vision' in model_name or 'image' in model_name or 'vit' in model_name:
            return 'image'
        elif 'audio' in model_name or 'speech' in model_name or 'whisper' in model_name:
            return 'audio'
        elif 'clip' in model_name:
            return 'image_text'
        elif 'multi' in model_name or 'llava' in model_name:
            return 'multimodal'
        
        # Default
        return 'unknown'
    
    def _infer_test_case(self, model_name: str) -> str:
        """Infer the test case from the model name"""
        model_name = model_name.lower()
        
        # Embedding models
        if 'bert' in model_name or 'embed' in model_name:
            return 'embedding'
        
        # Text generation
        if 'gpt' in model_name or 'llama' in model_name or 't5' in model_name:
            return 'text_generation'
        
        # Vision
        if 'vit' in model_name or 'vision' in model_name:
            return 'image_classification'
        
        # Audio
        if 'whisper' in model_name:
            return 'audio_transcription'
        if 'wav2vec' in model_name:
            return 'speech_recognition'
        
        # Multimodal
        if 'clip' in model_name:
            return 'image_text_matching'
        if 'llava' in model_name:
            return 'multimodal_generation'
        
        # Default
        return 'general'
    
    def _default_device_name(self, hardware_type: str) -> str:
        """Get a default device name for the hardware type"""
        hardware_type = hardware_type.lower()
        
        if hardware_type == 'cpu':
            return 'CPU'
        elif hardware_type == 'cuda':
            return 'NVIDIA GPU'
        elif hardware_type == 'rocm':
            return 'AMD GPU'
        elif hardware_type == 'mps':
            return 'Apple Silicon'
        elif hardware_type == 'openvino':
            return 'OpenVINO'
        elif hardware_type == 'webnn':
            return 'WebNN'
        elif hardware_type == 'webgpu':
            return 'WebGPU'
        else:
            return hardware_type.upper()

def main():
    """Command-line interface for the benchmark database migration tool."""
    parser = argparse.ArgumentParser(description="Benchmark Database Migration Tool")
    parser.add_argument("--input-dirs", nargs="+", 
                      help="Directories containing JSON benchmark files to migrate")
    parser.add_argument("--input-file", 
                      help="Single JSON file to migrate")
    parser.add_argument("--output-db", default="./benchmark_db.duckdb",
                      help="Output DuckDB database path")
    parser.add_argument("--incremental", action="store_true",
                      help="Only migrate files that haven't been processed before")
    parser.add_argument("--reindex-models", action="store_true",
                      help="Reindex and update model families and modalities")
    parser.add_argument("--cleanup", action="store_true",
                      help="Clean up JSON files after migration")
    parser.add_argument("--cleanup-days", type=int, default=30,
                      help="Only clean up files older than this many days")
    parser.add_argument("--move-to", 
                      help="Directory to move processed files to (instead of deleting)")
    parser.add_argument("--delete", action="store_true",
                      help="Delete processed files instead of moving them")
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug logging")
    args = parser.parse_args()
    
    # Create migration tool
    migration = BenchmarkDBMigration(output_db=args.output_db, debug=args.debug)
    
    if args.reindex_models:
        # Reindex models
        logger.info("Reindexing models...")
        results = migration.reindex_models()
        logger.info(f"Reindexed {results['total_models']} models:")
        logger.info(f"  Updated {results['family_updates']} model families")
        logger.info(f"  Updated {results['modality_updates']} model modalities")
        logger.info(f"  Total updates: {results['total_updates']}")
    
    elif args.input_file:
        # Migrate single file
        logger.info(f"Migrating file: {args.input_file}")
        counts = migration.migrate_file(args.input_file, args.incremental)
        logger.info(f"Migration complete: {counts}")
    
    elif args.input_dirs:
        # Migrate directories
        for directory in args.input_dirs:
            logger.info(f"Migrating directory: {directory}")
            counts = migration.migrate_directory(directory, True, args.incremental)
            logger.info(f"Migration complete for {directory}: {counts}")
    
    elif args.cleanup:
        # Clean up processed files
        logger.info("Cleaning up processed files...")
        if args.delete:
            count = migration.cleanup_json_files(args.cleanup_days, None, True)
        else:
            count = migration.cleanup_json_files(args.cleanup_days, args.move_to, False)
        logger.info(f"Cleanup complete: processed {count} files")
    
    else:
        # No action specified
        parser.print_help()

if __name__ == "__main__":
    main()