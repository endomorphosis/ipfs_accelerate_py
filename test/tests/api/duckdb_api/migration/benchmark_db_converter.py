#!/usr/bin/env python
"""
Benchmark Database Converter for the IPFS Accelerate Python Framework.

This module converts benchmark and test output JSON files to DuckDB/Parquet format
for efficient storage and querying.

Usage:
    python benchmark_db_converter.py --input-dir ./archived_test_results --output-db ./benchmark_db.duckdb
    python benchmark_db_converter.py --consolidate --categories performance hardware compatibility
"""

import os
import sys
import json
import glob
import logging
import argparse
import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

# Try to import the dependency management framework
try:
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from test.web_platform.unified_framework.dependency_management import (
        global_dependency_manager, require_dependencies
    )
    
    # Check core dependencies using the dependency manager
    for dep in ["duckdb", "pandas", "pyarrow"]:
        if not global_dependency_manager.check_optional_dependency(dep):
            # Get installation instructions
            install_instructions = global_dependency_manager.get_installation_instructions()
            print(f"Error: Required packages not installed.\n{install_instructions}")
            sys.exit(1)
    
    # Import the dependencies now that we've verified they are available
    import duckdb
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    HAS_DEPENDENCY_MANAGER = True
except ImportError:
    # Fallback to direct import checking
    try:
        import duckdb
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        print("Error: Required packages not installed. Please install with:")
        print("pip install duckdb pandas pyarrow")
        sys.exit(1)
    
    HAS_DEPENDENCY_MANAGER = False

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BenchmarkDBConverter:
    """
    Converts JSON benchmark results to DuckDB/Parquet format for efficient
    storage and querying.
    """
    
    def __init__(self, output_db: str = "./benchmark_db.duckdb", debug: bool = False):
        """
        Initialize the benchmark database converter.
        
        Args:
            output_db: Path to the output DuckDB database
            debug: Enable debug logging
        """
        self.output_db = output_db
        
        # Set up logging
        if debug:
            logger.setLevel(logging.DEBUG)
        
        # Schema definitions for different data types
        self.schemas = {
            "performance": self._get_performance_schema(),
            "hardware": self._get_hardware_schema(),
            "compatibility": self._get_compatibility_schema()
        }
        
        logger.info(f"Initialized BenchmarkDBConverter with output DB: {output_db}")
    
    def _get_performance_schema(self):
        """
        Define the schema for performance benchmark data.
        """
        return pa.schema([
            ('model', pa.string()),
            ('hardware', pa.string()),
            ('device', pa.string()),
            ('batch_size', pa.int32()),
            ('precision', pa.string()),
            ('throughput', pa.float32()),
            ('latency_avg', pa.float32()),
            ('latency_p90', pa.float32()),
            ('latency_p95', pa.float32()),
            ('latency_p99', pa.float32()),
            ('memory_peak', pa.float32()),
            ('timestamp', pa.timestamp('ms')),
            ('source_file', pa.string()),
            ('notes', pa.string())
        ])
    
    def _get_hardware_schema(self):
        """
        Define the schema for hardware detection data.
        """
        return pa.schema([
            ('hardware_type', pa.string()),
            ('device_name', pa.string()),
            ('is_available', pa.bool_()),
            ('platform', pa.string()),
            ('driver_version', pa.string()),
            ('memory_total', pa.float32()),
            ('memory_free', pa.float32()),
            ('compute_capability', pa.string()),
            ('error', pa.string()),
            ('timestamp', pa.timestamp('ms')),
            ('source_file', pa.string())
        ])
    
    def _get_compatibility_schema(self):
        """
        Define the schema for compatibility test data.
        """
        return pa.schema([
            ('model', pa.string()),
            ('hardware_type', pa.string()),
            ('is_compatible', pa.bool_()),
            ('compatibility_level', pa.string()),
            ('error_message', pa.string()),
            ('error_type', pa.string()),
            ('memory_required', pa.float32()),
            ('memory_available', pa.float32()),
            ('timestamp', pa.timestamp('ms')),
            ('source_file', pa.string())
        ])
    
    def _detect_file_category(self, file_path: str) -> str:
        """
        Detect the category of a JSON file based on its content.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Category string ('performance', 'hardware', or 'compatibility')
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Check for performance data
            if any(k in data for k in ['throughput', 'latency', 'performance', 'benchmark']):
                return 'performance'
            
            # Check for hardware detection data
            if any(k in data for k in ['cuda', 'rocm', 'mps', 'openvino', 'hardware_detection']):
                return 'hardware'
            
            # Check for compatibility data
            if any(k in data for k in ['compatibility', 'error', 'is_compatible']):
                return 'compatibility'
            
            # Default to performance if we can't determine
            return 'performance'
            
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Error detecting category for {file_path}: {e}")
            return 'unknown'
    
    def _normalize_performance_data(self, data: Dict, source_file: str) -> List[Dict]:
        """
        Normalize performance benchmark data to a standardized format.
        
        Args:
            data: Input data dictionary from JSON
            source_file: Source file path
            
        Returns:
            List of normalized data dictionaries
        """
        normalized = []
        timestamp = data.get('timestamp', datetime.datetime.now().isoformat())
        
        # Parse timestamp if it's a string
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.datetime.fromisoformat(timestamp)
            except ValueError:
                # Try another common format
                try:
                    timestamp = datetime.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    # Default to now if parsing fails
                    timestamp = datetime.datetime.now()
        
        # Handle different file formats
        if 'results' in data and isinstance(data['results'], list):
            # Multiple results format
            for result in data['results']:
                entry = {
                    'model': result.get('model', data.get('model', 'unknown')),
                    'hardware': result.get('hardware', data.get('hardware', 'unknown')),
                    'device': result.get('device', data.get('device', 'unknown')),
                    'batch_size': int(result.get('batch_size', data.get('batch_size', 1))),
                    'precision': result.get('precision', data.get('precision', 'fp32')),
                    'throughput': float(result.get('throughput', 0.0)),
                    'latency_avg': float(result.get('latency_avg', result.get('latency', 0.0))),
                    'latency_p90': float(result.get('latency_p90', 0.0)),
                    'latency_p95': float(result.get('latency_p95', 0.0)),
                    'latency_p99': float(result.get('latency_p99', 0.0)),
                    'memory_peak': float(result.get('memory_peak', result.get('memory', 0.0))),
                    'timestamp': timestamp,
                    'source_file': source_file,
                    'notes': result.get('notes', data.get('notes', ''))
                }
                normalized.append(entry)
        else:
            # Single result format
            entry = {
                'model': data.get('model', 'unknown'),
                'hardware': data.get('hardware', 'unknown'),
                'device': data.get('device', 'unknown'),
                'batch_size': int(data.get('batch_size', 1)),
                'precision': data.get('precision', 'fp32'),
                'throughput': float(data.get('throughput', 0.0)),
                'latency_avg': float(data.get('latency_avg', data.get('latency', 0.0))),
                'latency_p90': float(data.get('latency_p90', 0.0)),
                'latency_p95': float(data.get('latency_p95', 0.0)),
                'latency_p99': float(data.get('latency_p99', 0.0)),
                'memory_peak': float(data.get('memory_peak', data.get('memory', 0.0))),
                'timestamp': timestamp,
                'source_file': source_file,
                'notes': data.get('notes', '')
            }
            normalized.append(entry)
        
        return normalized
    
    def _normalize_hardware_data(self, data: Dict, source_file: str) -> List[Dict]:
        """
        Normalize hardware detection data to a standardized format.
        
        Args:
            data: Input data dictionary from JSON
            source_file: Source file path
            
        Returns:
            List of normalized data dictionaries
        """
        normalized = []
        timestamp = data.get('timestamp', datetime.datetime.now().isoformat())
        
        # Parse timestamp if it's a string
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.datetime.fromisoformat(timestamp)
            except ValueError:
                # Try another common format
                try:
                    timestamp = datetime.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    # Default to now if parsing fails
                    timestamp = datetime.datetime.now()
        
        # Handle CUDA devices
        if 'cuda' in data and data['cuda'] is True and 'cuda_devices' in data:
            for device in data['cuda_devices']:
                entry = {
                    'hardware_type': 'cuda',
                    'device_name': device.get('name', 'unknown'),
                    'is_available': True,
                    'platform': data.get('system', {}).get('platform', 'unknown'),
                    'driver_version': data.get('cuda_driver_version', 'unknown'),
                    'memory_total': float(device.get('total_memory', 0.0)),
                    'memory_free': float(device.get('free_memory', 0.0)),
                    'compute_capability': device.get('compute_capability', 'unknown'),
                    'error': '',
                    'timestamp': timestamp,
                    'source_file': source_file
                }
                normalized.append(entry)
        elif 'cuda' in data:
            # CUDA not available or no devices
            entry = {
                'hardware_type': 'cuda',
                'device_name': 'none',
                'is_available': data['cuda'] is True,
                'platform': data.get('system', {}).get('platform', 'unknown'),
                'driver_version': data.get('cuda_driver_version', 'unknown'),
                'memory_total': 0.0,
                'memory_free': 0.0,
                'compute_capability': 'unknown',
                'error': data.get('cuda_error', ''),
                'timestamp': timestamp,
                'source_file': source_file
            }
            normalized.append(entry)
        
        # Handle ROCm devices
        if 'rocm' in data and data['rocm'] is True and 'rocm_devices' in data:
            for device in data['rocm_devices']:
                entry = {
                    'hardware_type': 'rocm',
                    'device_name': device.get('name', 'unknown'),
                    'is_available': True,
                    'platform': data.get('system', {}).get('platform', 'unknown'),
                    'driver_version': data.get('rocm_version', 'unknown'),
                    'memory_total': float(device.get('total_memory', 0.0)),
                    'memory_free': float(device.get('free_memory', 0.0)),
                    'compute_capability': device.get('compute_capability', 'unknown'),
                    'error': '',
                    'timestamp': timestamp,
                    'source_file': source_file
                }
                normalized.append(entry)
        elif 'rocm' in data:
            # ROCm not available or no devices
            entry = {
                'hardware_type': 'rocm',
                'device_name': 'none',
                'is_available': data['rocm'] is True,
                'platform': data.get('system', {}).get('platform', 'unknown'),
                'driver_version': data.get('rocm_version', 'unknown'),
                'memory_total': 0.0,
                'memory_free': 0.0,
                'compute_capability': 'unknown',
                'error': data.get('rocm_error', ''),
                'timestamp': timestamp,
                'source_file': source_file
            }
            normalized.append(entry)
        
        # Handle MPS
        if 'mps' in data:
            entry = {
                'hardware_type': 'mps',
                'device_name': 'Apple Silicon',
                'is_available': data['mps'] is True,
                'platform': data.get('system', {}).get('platform', 'unknown'),
                'driver_version': 'n/a',
                'memory_total': 0.0,  # MPS typically doesn't report memory
                'memory_free': 0.0,
                'compute_capability': 'n/a',
                'error': data.get('mps_error', ''),
                'timestamp': timestamp,
                'source_file': source_file
            }
            normalized.append(entry)
        
        # Handle OpenVINO
        if 'openvino' in data:
            entry = {
                'hardware_type': 'openvino',
                'device_name': 'OpenVINO',
                'is_available': data['openvino'] is True,
                'platform': data.get('system', {}).get('platform', 'unknown'),
                'driver_version': data.get('openvino_version', 'unknown'),
                'memory_total': 0.0,
                'memory_free': 0.0,
                'compute_capability': 'n/a',
                'error': data.get('openvino_error', ''),
                'timestamp': timestamp,
                'source_file': source_file
            }
            normalized.append(entry)
        
        # Handle WebNN
        if 'webnn' in data:
            entry = {
                'hardware_type': 'webnn',
                'device_name': 'WebNN',
                'is_available': data['webnn'] is True,
                'platform': data.get('system', {}).get('platform', 'unknown'),
                'driver_version': 'n/a',
                'memory_total': 0.0,
                'memory_free': 0.0,
                'compute_capability': 'n/a',
                'error': data.get('webnn_error', ''),
                'timestamp': timestamp,
                'source_file': source_file
            }
            normalized.append(entry)
        
        # Handle WebGPU
        if 'webgpu' in data:
            entry = {
                'hardware_type': 'webgpu',
                'device_name': 'WebGPU',
                'is_available': data['webgpu'] is True,
                'platform': data.get('system', {}).get('platform', 'unknown'),
                'driver_version': 'n/a',
                'memory_total': 0.0,
                'memory_free': 0.0,
                'compute_capability': 'n/a',
                'error': data.get('webgpu_error', ''),
                'timestamp': timestamp,
                'source_file': source_file
            }
            normalized.append(entry)
        
        # Handle CPU
        entry = {
            'hardware_type': 'cpu',
            'device_name': data.get('system', {}).get('cpu_info', 'Unknown CPU'),
            'is_available': True,  # CPU is always available
            'platform': data.get('system', {}).get('platform', 'unknown'),
            'driver_version': 'n/a',
            'memory_total': float(data.get('system', {}).get('memory_total', 0.0)),
            'memory_free': float(data.get('system', {}).get('memory_free', 0.0)),
            'compute_capability': 'n/a',
            'error': '',
            'timestamp': timestamp,
            'source_file': source_file
        }
        normalized.append(entry)
        
        return normalized
    
    def _normalize_compatibility_data(self, data: Dict, source_file: str) -> List[Dict]:
        """
        Normalize compatibility test data to a standardized format.
        
        Args:
            data: Input data dictionary from JSON
            source_file: Source file path
            
        Returns:
            List of normalized data dictionaries
        """
        normalized = []
        timestamp = data.get('timestamp', datetime.datetime.now().isoformat())
        
        # Parse timestamp if it's a string
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.datetime.fromisoformat(timestamp)
            except ValueError:
                # Try another common format
                try:
                    timestamp = datetime.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    # Default to now if parsing fails
                    timestamp = datetime.datetime.now()
        
        # Handle different file formats
        if 'tests' in data and isinstance(data['tests'], list):
            # Multiple test results format
            for test in data['tests']:
                model = test.get('model', data.get('model', 'unknown'))
                for hw_type, hw_data in test.get('compatibility', {}).items():
                    entry = {
                        'model': model,
                        'hardware_type': hw_type,
                        'is_compatible': hw_data.get('is_compatible', hw_data.get('compatible', False)),
                        'compatibility_level': hw_data.get('level', 'unknown'),
                        'error_message': hw_data.get('error', ''),
                        'error_type': hw_data.get('error_type', ''),
                        'memory_required': float(hw_data.get('memory_required', 0.0)),
                        'memory_available': float(hw_data.get('memory_available', 0.0)),
                        'timestamp': timestamp,
                        'source_file': source_file
                    }
                    normalized.append(entry)
        elif 'compatibility' in data and isinstance(data['compatibility'], dict):
            # Compatibility matrix format
            model = data.get('model', 'unknown')
            for hw_type, hw_data in data['compatibility'].items():
                entry = {
                    'model': model,
                    'hardware_type': hw_type,
                    'is_compatible': hw_data.get('is_compatible', hw_data.get('compatible', False)),
                    'compatibility_level': hw_data.get('level', 'unknown'),
                    'error_message': hw_data.get('error', ''),
                    'error_type': hw_data.get('error_type', ''),
                    'memory_required': float(hw_data.get('memory_required', 0.0)),
                    'memory_available': float(hw_data.get('memory_available', 0.0)),
                    'timestamp': timestamp,
                    'source_file': source_file
                }
                normalized.append(entry)
        elif 'errors' in data and isinstance(data['errors'], list):
            # Error list format
            model = data.get('model', 'unknown')
            for error in data['errors']:
                hw_type = error.get('hardware_type', 'unknown')
                entry = {
                    'model': model,
                    'hardware_type': hw_type,
                    'is_compatible': False,  # Errors indicate incompatibility
                    'compatibility_level': 'incompatible',
                    'error_message': error.get('message', ''),
                    'error_type': error.get('error_type', ''),
                    'memory_required': float(error.get('memory_required', 0.0)),
                    'memory_available': float(error.get('memory_available', 0.0)),
                    'timestamp': timestamp,
                    'source_file': source_file
                }
                normalized.append(entry)
        else:
            # Simple format or unknown
            # Try to extract some basic info
            model = data.get('model', 'unknown')
            hardware_types = ['cuda', 'rocm', 'mps', 'openvino', 'webnn', 'webgpu', 'cpu']
            
            for hw_type in hardware_types:
                if hw_type in data:
                    is_compatible = data.get(hw_type, False)
                    error = data.get(f"{hw_type}_error", '')
                    
                    entry = {
                        'model': model,
                        'hardware_type': hw_type,
                        'is_compatible': is_compatible,
                        'compatibility_level': 'compatible' if is_compatible else 'incompatible',
                        'error_message': error,
                        'error_type': 'unknown',
                        'memory_required': 0.0,
                        'memory_available': 0.0,
                        'timestamp': timestamp,
                        'source_file': source_file
                    }
                    normalized.append(entry)
        
        return normalized
    
    def convert_file(self, file_path: str, category: str = None) -> Tuple[str, pd.DataFrame]:
        """
        Convert a single JSON file to a pandas DataFrame with a standardized schema.
        
        Args:
            file_path: Path to the JSON file
            category: Data category (if known, otherwise auto-detected)
            
        Returns:
            Tuple of (category, DataFrame)
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Auto-detect category if not provided
            if category is None:
                category = self._detect_file_category(file_path)
                logger.debug(f"Auto-detected category for {file_path}: {category}")
            
            # Skip unknown categories
            if category == 'unknown':
                logger.warning(f"Skipping file with unknown category: {file_path}")
                return category, pd.DataFrame()
            
            # Normalize data based on category
            source_file = os.path.basename(file_path)
            if category == 'performance':
                normalized = self._normalize_performance_data(data, source_file)
            elif category == 'hardware':
                normalized = self._normalize_hardware_data(data, source_file)
            elif category == 'compatibility':
                normalized = self._normalize_compatibility_data(data, source_file)
            else:
                logger.warning(f"Unsupported category: {category}")
                return 'unknown', pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(normalized)
            logger.debug(f"Converted {file_path} to DataFrame with {len(df)} rows")
            return category, df
            
        except Exception as e:
            logger.error(f"Error converting file {file_path}: {e}")
            return 'error', pd.DataFrame()
    
    def convert_directory(self, input_dir: str, categories: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Convert all JSON files in a directory to pandas DataFrames.
        
        Args:
            input_dir: Path to the directory containing JSON files
            categories: List of categories to include (or None for all)
            
        Returns:
            Dictionary of DataFrames by category
        """
        # Validate input directory
        if not os.path.isdir(input_dir):
            logger.error(f"Input directory not found: {input_dir}")
            return {}
        
        # Find all JSON files
        json_files = glob.glob(os.path.join(input_dir, "**/*.json"), recursive=True)
        logger.info(f"Found {len(json_files)} JSON files in {input_dir}")
        
        # Initialize result DataFrames
        result_dfs = {}
        
        # Process each file
        for file_path in json_files:
            category, df = self.convert_file(file_path)
            
            # Skip empty DataFrames or unwanted categories
            if df.empty or (categories is not None and category not in categories):
                continue
                
            # Add to result DataFrames
            if category not in result_dfs:
                result_dfs[category] = df
            else:
                result_dfs[category] = pd.concat([result_dfs[category], df], ignore_index=True)
        
        # Log the result
        for category, df in result_dfs.items():
            logger.info(f"Converted {len(df)} rows for category: {category}")
        
        return result_dfs
    
    def save_to_duckdb(self, dataframes: Dict[str, pd.DataFrame]) -> bool:
        """
        Save DataFrames to a DuckDB database.
        
        Args:
            dataframes: Dictionary of DataFrames by category
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Connect to the database
            con = duckdb.connect(self.output_db)
            
            # Create tables for each category
            for category, df in dataframes.items():
                if df.empty:
                    continue
                
                # Create table if not exists
                table_name = f"benchmark_{category}"
                create_table_sql = f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM df LIMIT 0;"
                con.execute(create_table_sql)
                
                # Insert data
                con.execute(f"INSERT INTO {table_name} SELECT * FROM df", {"df": df})
                logger.info(f"Inserted {len(df)} rows into table {table_name}")
            
            # Create views for common queries
            self._create_views(con)
            
            # Close connection
            con.close()
            
            logger.info(f"Successfully saved to DuckDB database: {self.output_db}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving to DuckDB: {e}")
            return False
    
    def save_to_parquet(self, dataframes: Dict[str, pd.DataFrame], output_dir: str = "./benchmark_parquet") -> bool:
        """
        Save DataFrames to Parquet files.
        
        Args:
            dataframes: Dictionary of DataFrames by category
            output_dir: Directory for Parquet files
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Save each DataFrame to a Parquet file
            for category, df in dataframes.items():
                if df.empty:
                    continue
                
                # Convert DataFrame to PyArrow Table with schema
                schema = self.schemas.get(category)
                if schema:
                    table = pa.Table.from_pandas(df, schema=schema)
                else:
                    table = pa.Table.from_pandas(df)
                
                # Save to Parquet file
                output_file = os.path.join(output_dir, f"benchmark_{category}.parquet")
                pq.write_table(table, output_file)
                logger.info(f"Saved {len(df)} rows to Parquet file: {output_file}")
            
            logger.info(f"Successfully saved to Parquet files in directory: {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving to Parquet: {e}")
            return False
    
    def _create_views(self, con: duckdb.DuckDBPyConnection) -> None:
        """
        Create views for common queries.
        
        Args:
            con: DuckDB connection
        """
        try:
            # View for latest performance results by model and hardware
            con.execute("""
            CREATE OR REPLACE VIEW latest_performance AS
            SELECT
                model,
                hardware,
                batch_size,
                precision,
                throughput,
                latency_avg,
                memory_peak,
                timestamp
            FROM (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY model, hardware, batch_size, precision
                        ORDER BY timestamp DESC
                    ) as row_num
                FROM benchmark_performance
            ) WHERE row_num = 1
            """)
            
            # View for hardware comparison
            con.execute("""
            CREATE OR REPLACE VIEW hardware_comparison AS
            SELECT
                model,
                hardware,
                AVG(throughput) as avg_throughput,
                MIN(latency_avg) as min_latency,
                MAX(memory_peak) as max_memory,
                COUNT(*) as num_runs
            FROM benchmark_performance
            GROUP BY model, hardware
            """)
            
            # View for the latest hardware detection
            con.execute("""
            CREATE OR REPLACE VIEW latest_hardware AS
            SELECT
                hardware_type,
                device_name,
                is_available,
                memory_total,
                memory_free,
                timestamp
            FROM (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY hardware_type, device_name
                        ORDER BY timestamp DESC
                    ) as row_num
                FROM benchmark_hardware
            ) WHERE row_num = 1
            """)
            
            # View for model compatibility matrix
            con.execute("""
            CREATE OR REPLACE VIEW compatibility_matrix AS
            SELECT
                model,
                hardware_type,
                is_compatible,
                compatibility_level,
                error_message
            FROM (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY model, hardware_type
                        ORDER BY timestamp DESC
                    ) as row_num
                FROM benchmark_compatibility
            ) WHERE row_num = 1
            """)
            
            logger.info("Created views in DuckDB database")
            
        except Exception as e:
            logger.error(f"Error creating views: {e}")
    
    def consolidate_directories(self, directories: List[str], categories: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Consolidate JSON files from multiple directories.
        
        Args:
            directories: List of directories to process
            categories: List of categories to include (or None for all)
            
        Returns:
            Dictionary of consolidated DataFrames by category
        """
        # Initialize result DataFrames
        result_dfs = {}
        
        # Process each directory
        for directory in directories:
            logger.info(f"Processing directory: {directory}")
            dfs = self.convert_directory(directory, categories)
            
            # Merge with existing DataFrames
            for category, df in dfs.items():
                if category not in result_dfs:
                    result_dfs[category] = df
                else:
                    result_dfs[category] = pd.concat([result_dfs[category], df], ignore_index=True)
        
        # Log the result
        for category, df in result_dfs.items():
            logger.info(f"Consolidated {len(df)} rows for category: {category}")
        
        return result_dfs
    
    def deduplicate_data(self, dataframes: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Deduplicate data by keeping the latest version of each unique entry.
        
        Args:
            dataframes: Dictionary of DataFrames by category
            
        Returns:
            Dictionary of deduplicated DataFrames
        """
        result_dfs = {}
        
        for category, df in dataframes.items():
            if df.empty:
                result_dfs[category] = df
                continue
            
            # Define deduplication keys based on category
            if category == 'performance':
                keys = ['model', 'hardware', 'batch_size', 'precision']
            elif category == 'hardware':
                keys = ['hardware_type', 'device_name']
            elif category == 'compatibility':
                keys = ['model', 'hardware_type']
            else:
                # If we don't know how to deduplicate, just copy
                result_dfs[category] = df
                continue
            
            # Sort by timestamp (descending) and keep first occurrence
            df = df.sort_values('timestamp', ascending=False)
            df = df.drop_duplicates(subset=keys, keep='first')
            
            logger.info(f"Deduplicated {category} data from {len(dataframes[category])} to {len(df)} rows")
            result_dfs[category] = df
        
        return result_dfs

def main():
    """Command-line interface for the benchmark database converter."""
    parser = argparse.ArgumentParser(description="Benchmark Database Converter")
    parser.add_argument("--input-dir", 
                       help="Directory containing JSON benchmark files")
    parser.add_argument("--output-db", default="./benchmark_db.duckdb",
                       help="Output DuckDB database path")
    parser.add_argument("--output-parquet-dir", default="./benchmark_parquet",
                       help="Output directory for Parquet files")
    parser.add_argument("--categories", nargs="+", 
                       choices=["performance", "hardware", "compatibility"],
                       help="Categories to include (default: all)")
    parser.add_argument("--consolidate", action="store_true",
                       help="Consolidate data from multiple directories")
    parser.add_argument("--deduplicate", action="store_true",
                       help="Deduplicate data, keeping the latest version")
    parser.add_argument("--directories", nargs="+",
                       help="Directories to consolidate when using --consolidate")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    args = parser.parse_args()
    
    # Create converter
    converter = BenchmarkDBConverter(output_db=args.output_db, debug=args.debug)
    
    # Perform requested actions
    if args.consolidate:
        directories = args.directories or [
            "./archived_test_results",
            "./performance_results",
            "./hardware_compatibility_reports"
        ]
        dataframes = converter.consolidate_directories(directories, args.categories)
    elif args.input_dir:
        dataframes = converter.convert_directory(args.input_dir, args.categories)
    else:
        # No source specified
        parser.print_help()
        return
    
    # Deduplicate if requested
    if args.deduplicate:
        dataframes = converter.deduplicate_data(dataframes)
    
    # Save to DuckDB
    success_duckdb = converter.save_to_duckdb(dataframes)
    
    # Save to Parquet
    success_parquet = converter.save_to_parquet(dataframes, args.output_parquet_dir)
    
    if success_duckdb and success_parquet:
        logger.info("Conversion completed successfully")
    else:
        logger.error("Conversion completed with errors")

if __name__ == "__main__":
    main()