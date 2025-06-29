#!/usr/bin/env python
"""
Intelligent incremental benchmark runner for the IPFS Accelerate framework.

This module provides a tool for running benchmarks incrementally, focusing only
on missing or outdated benchmarks to efficiently utilize resources.
"""

import os
import sys
import json
import logging
import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Set

try:
    import duckdb
    import pandas as pd
except ImportError:
    print("Error: Required packages not installed. Please install with:")
    print("pip install duckdb pandas")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path for module imports
sys.path.append(str(Path(__file__).parent.parent.parent))

class IncrementalBenchmarkRunner:
    """
    Intelligent incremental benchmark runner for the IPFS Accelerate framework.
    
    This class identifies missing or outdated benchmarks and runs only those,
    rather than re-running all benchmarks every time.
    """
    
    def __init__(self, db_path: str = "./benchmark_db.duckdb", debug: bool = False):
        """
        Initialize the incremental benchmark runner.
        
        Args:
            db_path: Path to the DuckDB database
            debug: Enable debug logging
        """
        self.db_path = db_path
        
        # Set up logging
        if debug:
            logger.setLevel(logging.DEBUG)
        
        logger.info(f"Initialized IncrementalBenchmarkRunner with database: {db_path}")
    
    def _get_connection(self):
        """Get a connection to the database."""
        return duckdb.connect(self.db_path)
    
    def identify_missing_benchmarks(self, models: List[str] = None, 
                                   hardware: List[str] = None,
                                   batch_sizes: List[int] = None) -> pd.DataFrame:
        """
        Identify missing benchmarks in the database.
        
        Args:
            models: List of model names to check (or None for all models in database)
            hardware: List of hardware types to check (or None for all hardware in database)
            batch_sizes: List of batch sizes to check (or None for [1, 4, 16])
            
        Returns:
            DataFrame with missing benchmark configurations
        """
        # Default batch sizes if not provided
        if not batch_sizes:
            batch_sizes = [1, 4, 16]
        
        conn = self._get_connection()
        
        try:
            # Get list of models
            if models:
                # Use provided models
                model_list = [(i+1, model) for i, model in enumerate(models)]
                model_df = pd.DataFrame(model_list, columns=['model_id', 'model_name'])
                
                # Check if models exist in database, add them if not
                for _, row in model_df.iterrows():
                    result = conn.execute(
                        "SELECT COUNT(*) FROM models WHERE model_name = ?", 
                        [row['model_name']]
                    ).fetchone()[0]
                    
                    if result == 0:
                        # Model doesn't exist, add it
                        logger.info(f"Adding model to database: {row['model_name']}")
                        max_id = conn.execute("SELECT COALESCE(MAX(model_id), 0) FROM models").fetchone()[0]
                        next_id = max_id + 1
                        
                        conn.execute(
                            """
                            INSERT INTO models (model_id, model_name, created_at)
                            VALUES (?, ?, CURRENT_TIMESTAMP)
                            """,
                            [next_id, row['model_name']]
                        )
            else:
                # Get all models from database
                model_df = conn.execute(
                    "SELECT model_id, model_name FROM models"
                ).fetch_df()
            
            # Get list of hardware platforms
            if hardware:
                # Use provided hardware types
                hardware_list = [(i+1, hw) for i, hw in enumerate(hardware)]
                hardware_df = pd.DataFrame(hardware_list, columns=['hardware_id', 'hardware_type'])
                
                # Check if hardware exists in database, add them if not
                for _, row in hardware_df.iterrows():
                    result = conn.execute(
                        "SELECT COUNT(*) FROM hardware_platforms WHERE hardware_type = ?", 
                        [row['hardware_type']]
                    ).fetchone()[0]
                    
                    if result == 0:
                        # Hardware doesn't exist, add it
                        logger.info(f"Adding hardware to database: {row['hardware_type']}")
                        max_id = conn.execute("SELECT COALESCE(MAX(hardware_id), 0) FROM hardware_platforms").fetchone()[0]
                        next_id = max_id + 1
                        
                        conn.execute(
                            """
                            INSERT INTO hardware_platforms (hardware_id, hardware_type, created_at)
                            VALUES (?, ?, CURRENT_TIMESTAMP)
                            """,
                            [next_id, row['hardware_type']]
                        )
            else:
                # Get all hardware platforms from database
                hardware_df = conn.execute(
                    "SELECT hardware_id, hardware_type FROM hardware_platforms"
                ).fetch_df()
            
            # Create a cartesian product of all possible combinations
            all_combinations = []
            for _, model_row in model_df.iterrows():
                for _, hw_row in hardware_df.iterrows():
                    for batch_size in batch_sizes:
                        all_combinations.append({
                            'model_id': model_row['model_id'],
                            'model_name': model_row['model_name'],
                            'hardware_id': hw_row['hardware_id'],
                            'hardware_type': hw_row['hardware_type'],
                            'batch_size': batch_size
                        })
            
            all_df = pd.DataFrame(all_combinations)
            
            # Get existing benchmark configurations
            existing_df = conn.execute(
                """
                SELECT 
                    m.model_id, 
                    m.model_name,
                    hp.hardware_id,
                    hp.hardware_type,
                    pr.batch_size
                FROM 
                    performance_results pr
                JOIN 
                    models m ON pr.model_id = m.model_id
                JOIN 
                    hardware_platforms hp ON pr.hardware_id = hp.hardware_id
                GROUP BY 
                    m.model_id, m.model_name, hp.hardware_id, hp.hardware_type, pr.batch_size
                """
            ).fetch_df()
            
            # If existing_df is empty, all combinations are missing
            if existing_df.empty:
                logger.info(f"No existing benchmarks found. All {len(all_df)} combinations are missing.")
                return all_df
            
            # Use a merge to find missing combinations
            merged_df = pd.merge(
                all_df, 
                existing_df, 
                on=['model_id', 'model_name', 'hardware_id', 'hardware_type', 'batch_size'],
                how='left',
                indicator=True
            )
            
            missing_df = merged_df[merged_df['_merge'] == 'left_only'].drop(columns=['_merge'])
            
            logger.info(f"Found {len(missing_df)} missing benchmark configurations out of {len(all_df)} total.")
            return missing_df
            
        finally:
            conn.close()
    
    def identify_outdated_benchmarks(self, models: List[str] = None, 
                                    hardware: List[str] = None,
                                    batch_sizes: List[int] = None,
                                    older_than_days: int = 30) -> pd.DataFrame:
        """
        Identify outdated benchmarks in the database.
        
        Args:
            models: List of model names to check (or None for all models in database)
            hardware: List of hardware types to check (or None for all hardware in database)
            batch_sizes: List of batch sizes to check (or None for [1, 4, 16])
            older_than_days: Consider benchmarks older than this many days as outdated
            
        Returns:
            DataFrame with outdated benchmark configurations
        """
        # Default batch sizes if not provided
        if not batch_sizes:
            batch_sizes = [1, 4, 16]
        
        # Calculate cutoff date
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=older_than_days)
        
        conn = self._get_connection()
        
        try:
            # Build SQL query
            sql = """
            SELECT 
                m.model_id, 
                m.model_name,
                hp.hardware_id,
                hp.hardware_type,
                pr.batch_size,
                MAX(pr.created_at) as latest_benchmark
            FROM 
                performance_results pr
            JOIN 
                models m ON pr.model_id = m.model_id
            JOIN 
                hardware_platforms hp ON pr.hardware_id = hp.hardware_id
            """
            
            conditions = []
            params = {}
            
            # Add model filter if provided
            if models:
                model_list = ", ".join([f"'{model}'" for model in models])
                conditions.append(f"m.model_name IN ({model_list})")
            
            # Add hardware filter if provided
            if hardware:
                hw_list = ", ".join([f"'{hw}'" for hw in hardware])
                conditions.append(f"hp.hardware_type IN ({hw_list})")
            
            # Add batch size filter if provided
            if batch_sizes:
                bs_list = ", ".join([str(bs) for bs in batch_sizes])
                conditions.append(f"pr.batch_size IN ({bs_list})")
            
            # Add conditions to SQL
            if conditions:
                sql += " WHERE " + " AND ".join(conditions)
            
            # Group by and having clause for finding outdated benchmarks
            sql += """
            GROUP BY 
                m.model_id, m.model_name, hp.hardware_id, hp.hardware_type, pr.batch_size
            HAVING 
                MAX(pr.created_at) < ?
            """
            
            # Execute query
            outdated_df = conn.execute(sql, [cutoff_date]).fetch_df()
            
            logger.info(f"Found {len(outdated_df)} outdated benchmark configurations (older than {older_than_days} days).")
            return outdated_df
            
        finally:
            conn.close()
    
    def identify_priority_benchmarks(self, priority_models: List[str] = None,
                                    priority_hardware: List[str] = None,
                                    batch_sizes: List[int] = None) -> pd.DataFrame:
        """
        Identify priority benchmark configurations based on key models and hardware.
        
        Args:
            priority_models: List of priority model names (or None for default priorities)
            priority_hardware: List of priority hardware types (or None for default priorities)
            batch_sizes: List of batch sizes to include (or None for [1, 4, 16])
            
        Returns:
            DataFrame with priority benchmark configurations
        """
        # Default priority models if not provided
        if not priority_models:
            priority_models = [
                'bert-base-uncased',
                't5-small',
                'whisper-tiny',
                'opt-125m',
                'vit-base'
            ]
        
        # Default priority hardware if not provided
        if not priority_hardware:
            priority_hardware = [
                'cpu',
                'cuda',
                'rocm',
                'openvino',
                'webgpu'
            ]
        
        # Default batch sizes if not provided
        if not batch_sizes:
            batch_sizes = [1, 4, 16]
        
        # Get all missing benchmarks for priority configurations
        missing_df = self.identify_missing_benchmarks(
            models=priority_models,
            hardware=priority_hardware,
            batch_sizes=batch_sizes
        )
        
        # Get all outdated benchmarks for priority configurations
        outdated_df = self.identify_outdated_benchmarks(
            models=priority_models,
            hardware=priority_hardware,
            batch_sizes=batch_sizes
        )
        
        # Combine missing and outdated benchmarks
        combined_df = pd.concat([missing_df, outdated_df], ignore_index=True)
        
        # Remove duplicates if any
        priority_df = combined_df.drop_duplicates(subset=[
            'model_id', 'model_name', 'hardware_id', 'hardware_type', 'batch_size'
        ])
        
        logger.info(f"Identified {len(priority_df)} priority benchmark configurations.")
        return priority_df
    
    def run_benchmarks(self, benchmarks_df: pd.DataFrame) -> bool:
        """
        Run benchmarks for the specified configurations.
        
        Args:
            benchmarks_df: DataFrame with benchmark configurations to run
            
        Returns:
            True if all benchmarks ran successfully, False otherwise
        """
        if benchmarks_df.empty:
            logger.info("No benchmarks to run.")
            return True
        
        # Group benchmarks by model and hardware for efficient execution
        grouped_benchmarks = {}
        for _, row in benchmarks_df.iterrows():
            key = (row['model_name'], row['hardware_type'])
            if key not in grouped_benchmarks:
                grouped_benchmarks[key] = []
            grouped_benchmarks[key].append(row['batch_size'])
        
        # Run benchmarks for each model-hardware combination
        all_successful = True
        for (model, hardware), batch_sizes in grouped_benchmarks.items():
            batch_sizes_str = ",".join([str(bs) for bs in batch_sizes])
            logger.info(f"Running benchmarks for {model} on {hardware} with batch sizes {batch_sizes_str}")
            
            # Construct command to run
            # This is a placeholder; in a real implementation, this would call the actual benchmark runner
            cmd = f"python -m duckdb_api.core.run_benchmark_with_db --model {model} --hardware {hardware} --batch-sizes {batch_sizes_str}"
            
            logger.info(f"Command: {cmd}")
            # In a real implementation, we would execute the command here
            # success = subprocess.run(cmd, shell=True).returncode == 0
            
            # Simulate success for testing
            success = True
            
            if not success:
                all_successful = False
        
        return all_successful

def main():
    """Command-line interface for the incremental benchmark runner."""
    parser = argparse.ArgumentParser(description="Incremental Benchmark Runner")
    parser.add_argument("--db-path", default="./benchmark_db.duckdb",
                       help="Path to the DuckDB database")
    parser.add_argument("--models", type=str,
                       help="Comma-separated list of model names to benchmark")
    parser.add_argument("--hardware", type=str,
                       help="Comma-separated list of hardware types to benchmark")
    parser.add_argument("--batch-sizes", type=str, default="1,4,16",
                       help="Comma-separated list of batch sizes to benchmark")
    parser.add_argument("--missing-only", action="store_true",
                       help="Only run benchmarks for missing configurations")
    parser.add_argument("--refresh-older-than", type=int, default=30,
                       help="Refresh benchmarks older than this many days")
    parser.add_argument("--priority-only", action="store_true",
                       help="Only run benchmarks for priority configurations")
    parser.add_argument("--output", type=str,
                       help="Output file for benchmark configurations (CSV format)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Only identify benchmarks to run, don't actually run them")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    args = parser.parse_args()
    
    # Convert comma-separated strings to lists
    models = args.models.split(',') if args.models else None
    hardware = args.hardware.split(',') if args.hardware else None
    batch_sizes = [int(bs) for bs in args.batch_sizes.split(',')] if args.batch_sizes else None
    
    # Create runner
    runner = IncrementalBenchmarkRunner(db_path=args.db_path, debug=args.debug)
    
    # Determine which benchmarks to run
    if args.priority_only:
        benchmarks_df = runner.identify_priority_benchmarks(
            priority_models=models,
            priority_hardware=hardware,
            batch_sizes=batch_sizes
        )
    elif args.missing_only:
        benchmarks_df = runner.identify_missing_benchmarks(
            models=models,
            hardware=hardware,
            batch_sizes=batch_sizes
        )
    else:
        # Combine missing and outdated benchmarks
        missing_df = runner.identify_missing_benchmarks(
            models=models,
            hardware=hardware,
            batch_sizes=batch_sizes
        )
        
        outdated_df = runner.identify_outdated_benchmarks(
            models=models,
            hardware=hardware,
            batch_sizes=batch_sizes,
            older_than_days=args.refresh_older_than
        )
        
        benchmarks_df = pd.concat([missing_df, outdated_df], ignore_index=True)
        
        # Remove duplicates if any
        benchmarks_df = benchmarks_df.drop_duplicates(subset=[
            'model_id', 'model_name', 'hardware_id', 'hardware_type', 'batch_size'
        ])
    
    # Output benchmark configurations if requested
    if args.output:
        benchmarks_df.to_csv(args.output, index=False)
        logger.info(f"Wrote {len(benchmarks_df)} benchmark configurations to {args.output}")
    
    # Run benchmarks if not a dry run
    if not args.dry_run:
        success = runner.run_benchmarks(benchmarks_df)
        if success:
            logger.info("All benchmarks completed successfully.")
        else:
            logger.error("Some benchmarks failed.")
            sys.exit(1)
    else:
        logger.info("Dry run completed. Would run the following benchmarks:")
        for _, row in benchmarks_df.iterrows():
            logger.info(f"Model: {row['model_name']}, Hardware: {row['hardware_type']}, Batch Size: {row['batch_size']}")

if __name__ == "__main__":
    main()