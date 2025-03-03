#!/usr/bin/env python3
"""
Benchmark Database for IPFS Accelerate Framework

This module provides a comprehensive database system for storing, retrieving, and
analyzing benchmark results across different hardware platforms and model types.
The database uses a combination of Parquet files for efficient storage and pandas
for analysis capabilities.

Features:
- Schema for storing comprehensive benchmark metadata
- Support for multiple hardware platforms and model types
- Version tracking for performance changes over time
- Query interface for flexible data retrieval
- Integration with existing benchmark tooling
- Support for historical data analysis and trends
"""

import os
import json
import time
import uuid
import logging
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default schema definitions
DEFAULT_SCHEMA = {
    "benchmark_id": str,           # Unique benchmark ID
    "timestamp": str,              # ISO format timestamp
    "version": str,                # Version of benchmark runner
    
    # Model information
    "model_name": str,             # Model name or path
    "model_family": str,           # Model family (e.g., embedding, text_generation)
    "model_size": str,             # Model size category (tiny, small, base, large)
    "model_modality": str,         # Model modality (text, vision, audio, multimodal)
    "model_parameters": int,       # Number of parameters (if available)
    
    # Hardware information
    "hardware_type": str,          # Hardware type (cpu, cuda, mps, openvino, rocm, webnn, webgpu)
    "hardware_name": str,          # Hardware name/model
    "hardware_memory": int,        # Available hardware memory in bytes
    "hardware_compute_units": int, # Number of compute units (cores, SMs, etc.)
    
    # Test configuration
    "batch_size": int,             # Batch size used
    "sequence_length": int,        # Sequence length (for text models)
    "precision": str,              # Precision used (fp32, fp16, bf16, int8)
    "test_type": str,              # Test type (inference, training)
    "warmup_iterations": int,      # Number of warmup iterations
    "benchmark_iterations": int,   # Number of benchmark iterations
    
    # Performance metrics
    "avg_latency": float,          # Average latency in seconds
    "p50_latency": float,          # 50th percentile latency
    "p90_latency": float,          # 90th percentile latency
    "p99_latency": float,          # 99th percentile latency
    "min_latency": float,          # Minimum latency
    "max_latency": float,          # Maximum latency
    "std_latency": float,          # Standard deviation of latency
    "throughput": float,           # Throughput in items/second
    
    # Memory metrics
    "peak_memory_usage": int,      # Peak memory usage in bytes
    "avg_memory_usage": int,       # Average memory usage in bytes
    "memory_utilization": float,   # Memory utilization percentage
    
    # Training metrics (if applicable)
    "samples_per_second": float,   # Training samples processed per second
    "gradient_computation_time": float,  # Time for gradient computation
    
    # Status information
    "status": str,                 # Benchmark status (completed, failed, timed_out)
    "error": str,                  # Error message (if failed)
    
    # Additional metadata
    "metadata": Dict               # Additional metadata (JSON serializable)
}

class BenchmarkDatabase:
    """
    Database for storing and retrieving benchmark results across different
    hardware platforms and model types.
    """
    
    def __init__(
        self, 
        database_dir: str = "./benchmark_database",
        schema: Optional[Dict] = None,
        auto_initialize: bool = True
    ):
        """
        Initialize the benchmark database.
        
        Args:
            database_dir: Directory to store benchmark data
            schema: Optional custom schema for benchmark data
            auto_initialize: Automatically initialize database if not existing
        """
        self.database_dir = Path(database_dir)
        self.schema = schema or DEFAULT_SCHEMA
        self.data_file = self.database_dir / "benchmark_data.parquet"
        self.index_file = self.database_dir / "benchmark_index.parquet"
        self.metadata_file = self.database_dir / "database_metadata.json"
        
        # Initialize database
        if auto_initialize and not self.database_dir.exists():
            self.initialize_database()
        elif not self.database_dir.exists():
            logger.warning(f"Database directory {database_dir} does not exist. Call initialize_database() to create it.")
        
        # Load dataframes
        self._load_dataframes()
    
    def initialize_database(self) -> bool:
        """
        Initialize the benchmark database by creating necessary directories
        and database files.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create database directory
            self.database_dir.mkdir(exist_ok=True, parents=True)
            
            # Create empty dataframes with schema
            empty_df = pd.DataFrame(columns=list(self.schema.keys()))
            
            # Save empty dataframes
            empty_df.to_parquet(self.data_file)
            
            # Create index dataframe (subset of columns for quick access)
            index_columns = [
                "benchmark_id", "timestamp", "model_name", "model_family", 
                "hardware_type", "batch_size", "test_type", "status"
            ]
            empty_index_df = empty_df[index_columns] if len(empty_df) > 0 else pd.DataFrame(columns=index_columns)
            empty_index_df.to_parquet(self.index_file)
            
            # Create metadata
            metadata = {
                "created": datetime.datetime.now().isoformat(),
                "schema_version": "1.0",
                "schema": self.schema,
                "num_entries": 0,
                "last_updated": datetime.datetime.now().isoformat()
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Initialized benchmark database at {self.database_dir}")
            return True
        
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            return False
    
    def _load_dataframes(self):
        """Load dataframes from database files"""
        try:
            if self.data_file.exists():
                self.df = pd.read_parquet(self.data_file)
                logger.info(f"Loaded {len(self.df)} benchmark entries")
            else:
                self.df = pd.DataFrame(columns=list(self.schema.keys()))
                logger.warning(f"Data file {self.data_file} does not exist. Using empty dataframe.")
            
            if self.index_file.exists():
                self.index_df = pd.read_parquet(self.index_file)
            else:
                index_columns = [
                    "benchmark_id", "timestamp", "model_name", "model_family", 
                    "hardware_type", "batch_size", "test_type", "status"
                ]
                self.index_df = self.df[index_columns] if len(self.df) > 0 else pd.DataFrame(columns=index_columns)
                logger.warning(f"Index file {self.index_file} does not exist. Using derived index.")
        
        except Exception as e:
            logger.error(f"Error loading dataframes: {e}")
            # Create empty dataframes as fallback
            self.df = pd.DataFrame(columns=list(self.schema.keys()))
            index_columns = [
                "benchmark_id", "timestamp", "model_name", "model_family", 
                "hardware_type", "batch_size", "test_type", "status"
            ]
            self.index_df = pd.DataFrame(columns=index_columns)
    
    def add_benchmark_result(self, benchmark_data: Dict[str, Any]) -> str:
        """
        Add a new benchmark result to the database.
        
        Args:
            benchmark_data: Dictionary containing benchmark data
            
        Returns:
            str: Benchmark ID
        """
        # Generate benchmark ID if not provided
        if "benchmark_id" not in benchmark_data:
            benchmark_data["benchmark_id"] = str(uuid.uuid4())
        
        # Add timestamp if not provided
        if "timestamp" not in benchmark_data:
            benchmark_data["timestamp"] = datetime.datetime.now().isoformat()
        
        # Validate schema
        missing_required = []
        for key in self.schema:
            if key not in benchmark_data and key != "metadata":
                missing_required.append(key)
        
        if missing_required:
            logger.warning(f"Missing required fields: {missing_required}")
            # Fill missing with None
            for key in missing_required:
                benchmark_data[key] = None
        
        # Add as new row to dataframe
        new_row = pd.DataFrame([benchmark_data])
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        
        # Update index
        index_columns = [
            "benchmark_id", "timestamp", "model_name", "model_family", 
            "hardware_type", "batch_size", "test_type", "status"
        ]
        new_index_row = new_row[index_columns]
        self.index_df = pd.concat([self.index_df, new_index_row], ignore_index=True)
        
        # Save changes
        self._save_dataframes()
        
        # Update metadata
        self._update_metadata()
        
        logger.info(f"Added benchmark result with ID: {benchmark_data['benchmark_id']}")
        return benchmark_data["benchmark_id"]
    
    def add_benchmark_results(self, benchmark_data_list: List[Dict[str, Any]]) -> List[str]:
        """
        Add multiple benchmark results to the database.
        
        Args:
            benchmark_data_list: List of dictionaries containing benchmark data
            
        Returns:
            List[str]: List of benchmark IDs
        """
        benchmark_ids = []
        
        # Prepare all data first
        for i, benchmark_data in enumerate(benchmark_data_list):
            # Generate benchmark ID if not provided
            if "benchmark_id" not in benchmark_data:
                benchmark_data["benchmark_id"] = str(uuid.uuid4())
            
            # Add timestamp if not provided
            if "timestamp" not in benchmark_data:
                benchmark_data["timestamp"] = datetime.datetime.now().isoformat()
            
            # Validate schema
            missing_required = []
            for key in self.schema:
                if key not in benchmark_data and key != "metadata":
                    missing_required.append(key)
            
            if missing_required:
                logger.warning(f"Entry {i}: Missing required fields: {missing_required}")
                # Fill missing with None
                for key in missing_required:
                    benchmark_data[key] = None
            
            benchmark_ids.append(benchmark_data["benchmark_id"])
        
        # Add all as new rows to dataframe
        new_rows = pd.DataFrame(benchmark_data_list)
        self.df = pd.concat([self.df, new_rows], ignore_index=True)
        
        # Update index
        index_columns = [
            "benchmark_id", "timestamp", "model_name", "model_family", 
            "hardware_type", "batch_size", "test_type", "status"
        ]
        new_index_rows = new_rows[index_columns]
        self.index_df = pd.concat([self.index_df, new_index_rows], ignore_index=True)
        
        # Save changes
        self._save_dataframes()
        
        # Update metadata
        self._update_metadata()
        
        logger.info(f"Added {len(benchmark_data_list)} benchmark results")
        return benchmark_ids
    
    def get_benchmark(self, benchmark_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific benchmark result by ID.
        
        Args:
            benchmark_id: Benchmark ID
            
        Returns:
            Optional[Dict]: Benchmark data or None if not found
        """
        if benchmark_id not in self.df["benchmark_id"].values:
            logger.warning(f"Benchmark with ID {benchmark_id} not found")
            return None
        
        # Get the row as a dictionary
        benchmark = self.df[self.df["benchmark_id"] == benchmark_id].iloc[0].to_dict()
        return benchmark
    
    def query_benchmarks(self, 
                         model_name: Optional[str] = None, 
                         model_family: Optional[str] = None,
                         hardware_type: Optional[str] = None,
                         test_type: Optional[str] = None,
                         batch_size: Optional[int] = None,
                         status: Optional[str] = None,
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Query benchmark results based on filters.
        
        Args:
            model_name: Filter by model name
            model_family: Filter by model family
            hardware_type: Filter by hardware type
            test_type: Filter by test type
            batch_size: Filter by batch size
            status: Filter by status
            start_date: Filter by start date (ISO format)
            end_date: Filter by end date (ISO format)
            
        Returns:
            pd.DataFrame: Filtered benchmark results
        """
        # Start with index for faster filtering
        filtered_index = self.index_df.copy()
        
        # Apply filters
        if model_name is not None:
            filtered_index = filtered_index[filtered_index["model_name"] == model_name]
        
        if model_family is not None:
            filtered_index = filtered_index[filtered_index["model_family"] == model_family]
        
        if hardware_type is not None:
            filtered_index = filtered_index[filtered_index["hardware_type"] == hardware_type]
        
        if test_type is not None:
            filtered_index = filtered_index[filtered_index["test_type"] == test_type]
        
        if batch_size is not None:
            filtered_index = filtered_index[filtered_index["batch_size"] == batch_size]
        
        if status is not None:
            filtered_index = filtered_index[filtered_index["status"] == status]
        
        if start_date is not None:
            filtered_index = filtered_index[filtered_index["timestamp"] >= start_date]
        
        if end_date is not None:
            filtered_index = filtered_index[filtered_index["timestamp"] <= end_date]
        
        # Get benchmark IDs from filtered index
        benchmark_ids = filtered_index["benchmark_id"].tolist()
        
        # Get full data for these IDs
        results = self.df[self.df["benchmark_id"].isin(benchmark_ids)]
        
        logger.info(f"Query returned {len(results)} results")
        return results
    
    def get_latest_benchmarks(self, 
                             model_name: Optional[str] = None, 
                             model_family: Optional[str] = None,
                             hardware_type: Optional[str] = None,
                             test_type: Optional[str] = None) -> pd.DataFrame:
        """
        Get latest benchmark results for each model-hardware combination.
        
        Args:
            model_name: Filter by model name
            model_family: Filter by model family
            hardware_type: Filter by hardware type
            test_type: Filter by test type
            
        Returns:
            pd.DataFrame: Latest benchmark results
        """
        # Apply basic filters first
        filtered_df = self.query_benchmarks(
            model_name=model_name,
            model_family=model_family,
            hardware_type=hardware_type,
            test_type=test_type,
            status="completed"
        )
        
        if len(filtered_df) == 0:
            return filtered_df
        
        # Group by model and hardware, then get the latest timestamp
        grouped = filtered_df.groupby(["model_name", "hardware_type", "batch_size"])
        latest_timestamps = grouped["timestamp"].max().reset_index()
        
        # Merge back to get the full rows
        result = pd.merge(
            filtered_df,
            latest_timestamps,
            on=["model_name", "hardware_type", "batch_size", "timestamp"]
        )
        
        logger.info(f"Retrieved {len(result)} latest benchmark results")
        return result
    
    def get_hardware_comparison(self, 
                               model_name: Optional[str] = None,
                               model_family: Optional[str] = None,
                               batch_size: Optional[int] = None,
                               metric: str = "throughput") -> pd.DataFrame:
        """
        Get hardware comparison for a specific model or model family.
        
        Args:
            model_name: Model name (if None, uses model_family)
            model_family: Model family (if model_name is None)
            batch_size: Batch size
            metric: Metric to compare (throughput, avg_latency, etc.)
            
        Returns:
            pd.DataFrame: Hardware comparison data
        """
        if model_name is None and model_family is None:
            logger.error("Either model_name or model_family must be provided")
            return pd.DataFrame()
        
        # Get latest benchmark results
        if model_name is not None:
            results = self.get_latest_benchmarks(model_name=model_name, batch_size=batch_size)
        else:
            results = self.get_latest_benchmarks(model_family=model_family, batch_size=batch_size)
        
        if len(results) == 0:
            logger.warning(f"No results found for comparison")
            return pd.DataFrame()
        
        # Extract hardware types and metric values
        comparison = results[["model_name", "hardware_type", metric]].copy()
        
        # Pivot table with model_name as index and hardware_type as columns
        pivot_table = comparison.pivot_table(
            index="model_name",
            columns="hardware_type",
            values=metric
        )
        
        return pivot_table
    
    def get_model_comparison(self,
                            model_family: str,
                            hardware_type: str,
                            batch_size: Optional[int] = None,
                            metric: str = "throughput") -> pd.DataFrame:
        """
        Get model comparison within a model family on specific hardware.
        
        Args:
            model_family: Model family
            hardware_type: Hardware type
            batch_size: Batch size
            metric: Metric to compare (throughput, avg_latency, etc.)
            
        Returns:
            pd.DataFrame: Model comparison data
        """
        # Get latest benchmark results
        results = self.get_latest_benchmarks(
            model_family=model_family,
            hardware_type=hardware_type,
            batch_size=batch_size
        )
        
        if len(results) == 0:
            logger.warning(f"No results found for comparison")
            return pd.DataFrame()
        
        # Extract model names and metric values
        comparison = results[["model_name", metric]].copy()
        
        # Sort by metric (ascending or descending based on metric)
        if metric in ["throughput", "samples_per_second"]:
            # Higher is better
            comparison = comparison.sort_values(by=metric, ascending=False)
        else:
            # Lower is better (latency, memory usage)
            comparison = comparison.sort_values(by=metric, ascending=True)
        
        return comparison
    
    def get_batch_size_scaling(self,
                              model_name: str,
                              hardware_type: str,
                              metric: str = "throughput") -> pd.DataFrame:
        """
        Get batch size scaling data for a specific model on specific hardware.
        
        Args:
            model_name: Model name
            hardware_type: Hardware type
            metric: Metric to analyze (throughput, avg_latency, etc.)
            
        Returns:
            pd.DataFrame: Batch size scaling data
        """
        # Get latest benchmark results for different batch sizes
        results = self.get_latest_benchmarks(
            model_name=model_name,
            hardware_type=hardware_type
        )
        
        if len(results) == 0:
            logger.warning(f"No results found for batch size scaling")
            return pd.DataFrame()
        
        # Extract batch sizes and metric values
        scaling = results[["batch_size", metric]].copy()
        
        # Sort by batch size
        scaling = scaling.sort_values(by="batch_size")
        
        return scaling
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dict: Statistics about the database
        """
        stats = {
            "total_entries": len(self.df),
            "model_families": self.df["model_family"].nunique(),
            "unique_models": self.df["model_name"].nunique(),
            "hardware_types": self.df["hardware_type"].nunique(),
            "test_types": self.df["test_type"].nunique(),
            "completed_benchmarks": len(self.df[self.df["status"] == "completed"]),
            "failed_benchmarks": len(self.df[self.df["status"] == "failed"]),
            "latest_benchmark": self.df["timestamp"].max() if len(self.df) > 0 else None,
            "oldest_benchmark": self.df["timestamp"].min() if len(self.df) > 0 else None,
        }
        
        # Add model family breakdown
        family_counts = self.df["model_family"].value_counts().to_dict()
        stats["model_family_counts"] = family_counts
        
        # Add hardware type breakdown
        hardware_counts = self.df["hardware_type"].value_counts().to_dict()
        stats["hardware_type_counts"] = hardware_counts
        
        return stats
    
    def update_benchmark(self, benchmark_id: str, update_data: Dict[str, Any]) -> bool:
        """
        Update an existing benchmark result.
        
        Args:
            benchmark_id: Benchmark ID
            update_data: Dictionary containing updated benchmark data
            
        Returns:
            bool: True if successful, False otherwise
        """
        if benchmark_id not in self.df["benchmark_id"].values:
            logger.warning(f"Benchmark with ID {benchmark_id} not found")
            return False
        
        try:
            # Get index of benchmark
            idx = self.df[self.df["benchmark_id"] == benchmark_id].index[0]
            
            # Update fields
            for key, value in update_data.items():
                if key in self.schema:
                    self.df.at[idx, key] = value
                else:
                    logger.warning(f"Field {key} not in schema, skipping")
            
            # Update index if needed
            index_columns = [
                "benchmark_id", "timestamp", "model_name", "model_family", 
                "hardware_type", "batch_size", "test_type", "status"
            ]
            needs_index_update = any(key in index_columns for key in update_data.keys())
            
            if needs_index_update:
                # Rebuild index
                idx_idx = self.index_df[self.index_df["benchmark_id"] == benchmark_id].index[0]
                for key in index_columns:
                    if key in update_data:
                        self.index_df.at[idx_idx, key] = update_data[key]
            
            # Save changes
            self._save_dataframes()
            
            # Update metadata
            self._update_metadata()
            
            logger.info(f"Updated benchmark with ID: {benchmark_id}")
            return True
        
        except Exception as e:
            logger.error(f"Error updating benchmark: {e}")
            return False
    
    def delete_benchmark(self, benchmark_id: str) -> bool:
        """
        Delete a benchmark result.
        
        Args:
            benchmark_id: Benchmark ID
            
        Returns:
            bool: True if successful, False otherwise
        """
        if benchmark_id not in self.df["benchmark_id"].values:
            logger.warning(f"Benchmark with ID {benchmark_id} not found")
            return False
        
        try:
            # Delete from main dataframe
            self.df = self.df[self.df["benchmark_id"] != benchmark_id]
            
            # Delete from index
            self.index_df = self.index_df[self.index_df["benchmark_id"] != benchmark_id]
            
            # Save changes
            self._save_dataframes()
            
            # Update metadata
            self._update_metadata()
            
            logger.info(f"Deleted benchmark with ID: {benchmark_id}")
            return True
        
        except Exception as e:
            logger.error(f"Error deleting benchmark: {e}")
            return False
    
    def _save_dataframes(self):
        """Save dataframes to database files"""
        try:
            self.df.to_parquet(self.data_file)
            self.index_df.to_parquet(self.index_file)
            logger.debug("Saved dataframes to database files")
        except Exception as e:
            logger.error(f"Error saving dataframes: {e}")
    
    def _update_metadata(self):
        """Update database metadata"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {
                    "created": datetime.datetime.now().isoformat(),
                    "schema_version": "1.0",
                    "schema": self.schema
                }
            
            # Update metadata
            metadata["num_entries"] = len(self.df)
            metadata["last_updated"] = datetime.datetime.now().isoformat()
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.debug("Updated database metadata")
        except Exception as e:
            logger.error(f"Error updating metadata: {e}")
    
    def create_backup(self, backup_dir: Optional[str] = None) -> str:
        """
        Create a backup of the database.
        
        Args:
            backup_dir: Directory to store backup (if None, uses database_dir/backups)
            
        Returns:
            str: Path to backup directory
        """
        # Create backup directory
        if backup_dir is None:
            backup_dir = self.database_dir / "backups"
        else:
            backup_dir = Path(backup_dir)
        
        backup_dir.mkdir(exist_ok=True, parents=True)
        
        # Create timestamp for backup
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"benchmark_db_backup_{timestamp}"
        backup_path.mkdir(exist_ok=True)
        
        try:
            # Copy database files
            if self.data_file.exists():
                self.df.to_parquet(backup_path / "benchmark_data.parquet")
            
            if self.index_file.exists():
                self.index_df.to_parquet(backup_path / "benchmark_index.parquet")
            
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                with open(backup_path / "database_metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            logger.info(f"Created backup at {backup_path}")
            return str(backup_path)
        
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return ""
    
    def restore_from_backup(self, backup_path: str) -> bool:
        """
        Restore database from backup.
        
        Args:
            backup_path: Path to backup directory
            
        Returns:
            bool: True if successful, False otherwise
        """
        backup_path = Path(backup_path)
        
        if not backup_path.exists():
            logger.error(f"Backup path {backup_path} does not exist")
            return False
        
        try:
            # Check if backup files exist
            data_file = backup_path / "benchmark_data.parquet"
            index_file = backup_path / "benchmark_index.parquet"
            metadata_file = backup_path / "database_metadata.json"
            
            if not data_file.exists() or not index_file.exists() or not metadata_file.exists():
                logger.error(f"Backup at {backup_path} is missing required files")
                return False
            
            # Create backup of current database
            self.create_backup()
            
            # Restore from backup
            self.df = pd.read_parquet(data_file)
            self.index_df = pd.read_parquet(index_file)
            
            # Copy metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Save restored data
            self._save_dataframes()
            
            logger.info(f"Restored database from backup at {backup_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error restoring from backup: {e}")
            return False

    def compact_database(self) -> bool:
        """
        Compact the database by removing duplicates and reorganizing data.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create backup before compacting
            self.create_backup()
            
            # Remove duplicates (keep latest version of each benchmark_id)
            self.df = self.df.drop_duplicates(subset=["benchmark_id"], keep="last")
            
            # Rebuild index
            index_columns = [
                "benchmark_id", "timestamp", "model_name", "model_family", 
                "hardware_type", "batch_size", "test_type", "status"
            ]
            self.index_df = self.df[index_columns].copy()
            
            # Save compacted dataframes
            self._save_dataframes()
            
            # Update metadata
            self._update_metadata()
            
            logger.info(f"Compacted database, removed duplicates")
            return True
        
        except Exception as e:
            logger.error(f"Error compacting database: {e}")
            return False
    
    def export_to_csv(self, export_path: Optional[str] = None) -> str:
        """
        Export database to CSV file.
        
        Args:
            export_path: Path to export CSV (if None, uses database_dir/exports)
            
        Returns:
            str: Path to exported CSV file
        """
        # Create export directory
        if export_path is None:
            export_dir = self.database_dir / "exports"
            export_dir.mkdir(exist_ok=True, parents=True)
            
            # Create timestamp for export
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = export_dir / f"benchmark_data_{timestamp}.csv"
        else:
            export_path = Path(export_path)
            export_path.parent.mkdir(exist_ok=True, parents=True)
        
        try:
            # Export to CSV
            self.df.to_csv(export_path, index=False)
            logger.info(f"Exported database to {export_path}")
            return str(export_path)
        
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return ""

    def import_from_csv(self, import_path: str) -> int:
        """
        Import database from CSV file.
        
        Args:
            import_path: Path to CSV file
            
        Returns:
            int: Number of imported entries
        """
        import_path = Path(import_path)
        
        if not import_path.exists():
            logger.error(f"Import path {import_path} does not exist")
            return 0
        
        try:
            # Create backup before import
            self.create_backup()
            
            # Read CSV
            imported_df = pd.read_csv(import_path)
            
            # Validate schema
            missing_cols = [col for col in self.schema if col not in imported_df.columns and col != "metadata"]
            if missing_cols:
                logger.warning(f"Missing columns in imported data: {missing_cols}")
                # Add missing columns
                for col in missing_cols:
                    imported_df[col] = None
            
            # Get benchmark IDs already in database
            existing_ids = set(self.df["benchmark_id"].values)
            
            # Filter out entries that already exist
            new_entries = imported_df[~imported_df["benchmark_id"].isin(existing_ids)]
            
            if len(new_entries) == 0:
                logger.warning("No new entries to import")
                return 0
            
            # Add new entries
            self.df = pd.concat([self.df, new_entries], ignore_index=True)
            
            # Rebuild index
            index_columns = [
                "benchmark_id", "timestamp", "model_name", "model_family", 
                "hardware_type", "batch_size", "test_type", "status"
            ]
            self.index_df = self.df[index_columns].copy()
            
            # Save updated dataframes
            self._save_dataframes()
            
            # Update metadata
            self._update_metadata()
            
            logger.info(f"Imported {len(new_entries)} entries from {import_path}")
            return len(new_entries)
        
        except Exception as e:
            logger.error(f"Error importing from CSV: {e}")
            return 0

def create_sample_benchmark_data() -> Dict[str, Any]:
    """
    Create sample benchmark data for testing.
    
    Returns:
        Dict: Sample benchmark data
    """
    model_families = ["embedding", "text_generation", "vision", "audio", "multimodal"]
    hardware_types = ["cpu", "cuda", "mps", "openvino", "rocm"]
    
    sample_data = {
        "benchmark_id": str(uuid.uuid4()),
        "timestamp": datetime.datetime.now().isoformat(),
        "version": "1.0.0",
        
        # Model information
        "model_name": f"sample-model-{random.randint(1, 100)}",
        "model_family": random.choice(model_families),
        "model_size": random.choice(["tiny", "small", "base", "large"]),
        "model_modality": random.choice(["text", "vision", "audio", "multimodal"]),
        "model_parameters": random.randint(1000000, 1000000000),
        
        # Hardware information
        "hardware_type": random.choice(hardware_types),
        "hardware_name": f"sample-hardware-{random.randint(1, 100)}",
        "hardware_memory": random.randint(1000000000, 100000000000),
        "hardware_compute_units": random.randint(1, 128),
        
        # Test configuration
        "batch_size": random.choice([1, 2, 4, 8, 16, 32]),
        "sequence_length": random.choice([32, 64, 128, 256, 512]),
        "precision": random.choice(["fp32", "fp16", "bf16", "int8"]),
        "test_type": random.choice(["inference", "training"]),
        "warmup_iterations": random.randint(1, 10),
        "benchmark_iterations": random.randint(10, 100),
        
        # Performance metrics
        "avg_latency": random.uniform(0.001, 1.0),
        "p50_latency": random.uniform(0.001, 1.0),
        "p90_latency": random.uniform(0.001, 1.0),
        "p99_latency": random.uniform(0.001, 1.0),
        "min_latency": random.uniform(0.001, 1.0),
        "max_latency": random.uniform(0.001, 1.0),
        "std_latency": random.uniform(0.0001, 0.1),
        "throughput": random.uniform(1.0, 1000.0),
        
        # Memory metrics
        "peak_memory_usage": random.randint(1000000, 10000000000),
        "avg_memory_usage": random.randint(1000000, 10000000000),
        "memory_utilization": random.uniform(0.1, 1.0),
        
        # Training metrics
        "samples_per_second": random.uniform(1.0, 1000.0) if random.choice([True, False]) else None,
        "gradient_computation_time": random.uniform(0.001, 1.0) if random.choice([True, False]) else None,
        
        # Status information
        "status": random.choice(["completed", "failed", "timed_out"]),
        "error": "Sample error message" if random.choice([True, False]) else None,
        
        # Additional metadata
        "metadata": {
            "tags": ["sample", "test"],
            "notes": "Sample benchmark data for testing"
        }
    }
    
    return sample_data

if __name__ == "__main__":
    import random
    
    # Example usage
    print("Benchmark Database Example")
    print("-------------------------\n")
    
    # Initialize database
    db = BenchmarkDatabase(database_dir="./test_benchmark_database")
    
    # Generate and add sample data
    num_samples = 50
    print(f"Generating {num_samples} sample benchmark entries...")
    
    sample_data_list = []
    for _ in range(num_samples):
        sample_data = create_sample_benchmark_data()
        sample_data_list.append(sample_data)
    
    # Add to database
    benchmark_ids = db.add_benchmark_results(sample_data_list)
    print(f"Added {len(benchmark_ids)} benchmark entries")
    
    # Query examples
    print("\nQuery Examples:")
    
    # Get latest benchmarks for embedding models
    embedding_results = db.get_latest_benchmarks(model_family="embedding")
    print(f"Latest embedding benchmarks: {len(embedding_results)} results")
    
    # Hardware comparison for a model family
    hw_comparison = db.get_hardware_comparison(model_family="text_generation", metric="throughput")
    print("\nHardware comparison for text_generation models (throughput):")
    print(hw_comparison)
    
    # Model comparison on specific hardware
    model_comparison = db.get_model_comparison(model_family="vision", hardware_type="cuda", metric="avg_latency")
    print("\nModel comparison for vision models on CUDA (latency):")
    print(model_comparison.head())
    
    # Batch size scaling
    if len(db.df) > 0:
        random_model = db.df["model_name"].iloc[0]
        random_hw = db.df["hardware_type"].iloc[0]
        batch_scaling = db.get_batch_size_scaling(model_name=random_model, hardware_type=random_hw)
        print(f"\nBatch size scaling for {random_model} on {random_hw}:")
        print(batch_scaling)
    
    # Database statistics
    stats = db.get_statistics()
    print("\nDatabase Statistics:")
    for key, value in stats.items():
        if not isinstance(value, dict):
            print(f"  {key}: {value}")
    
    # Export to CSV
    export_path = db.export_to_csv()
    print(f"\nExported database to {export_path}")
    
    print("\nExample completed successfully!")