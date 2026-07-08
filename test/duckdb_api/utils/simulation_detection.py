#!/usr/bin/env python
"""
Simulation detection and validation for benchmark results.

This module provides tools for detecting, marking, and validating simulated
benchmark results to ensure transparency in benchmark reporting.
"""

import os
import sys
import json
import logging
import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

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

class SimulationDetection:
    """
    Simulation detection and validation for benchmark results.
    
    This class provides tools for detecting, marking, and validating simulated
    benchmark results to ensure transparency in benchmark reporting.
    """
    
    def __init__(self, db_path: str = "./benchmark_db.duckdb", debug: bool = False):
        """
        Initialize the simulation detection utility.
        
        Args:
            db_path: Path to the DuckDB database
            debug: Enable debug logging
        """
        self.db_path = db_path
        
        # Set up logging
        if debug:
            logger.setLevel(logging.DEBUG)
        
        logger.info(f"Initialized SimulationDetection with database: {db_path}")
    
    def _get_connection(self):
        """Get a connection to the database."""
        return duckdb.connect(self.db_path)
    
    def check_simulation_columns_exist(self) -> bool:
        """
        Check if the simulation columns exist in the database.
        
        Returns:
            True if simulation columns exist, False otherwise
        """
        conn = self._get_connection()
        
        try:
            # Check for is_simulated column in performance_results table
            result = conn.execute(
                """
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'performance_results' 
                AND column_name = 'is_simulated'
                """
            ).fetchone()
            
            if result:
                logger.info("Simulation columns already exist in the database.")
                return True
            else:
                logger.info("Simulation columns do not exist in the database.")
                return False
                
        except Exception as e:
            logger.error(f"Error checking simulation columns: {e}")
            return False
        finally:
            conn.close()
    
    def add_simulation_columns(self) -> bool:
        """
        Add simulation columns to the database tables.
        
        Returns:
            True if columns were added successfully, False otherwise
        """
        # Check if columns already exist
        if self.check_simulation_columns_exist():
            logger.info("Simulation columns already exist. No changes made.")
            return True
        
        conn = self._get_connection()
        
        try:
            # Add is_simulated and simulation_reason columns to performance_results
            conn.execute(
                """
                ALTER TABLE performance_results 
                ADD COLUMN is_simulated BOOLEAN DEFAULT FALSE,
                ADD COLUMN simulation_reason VARCHAR
                """
            )
            
            # Add is_simulated and simulation_reason columns to hardware_compatibility
            conn.execute(
                """
                ALTER TABLE hardware_compatibility 
                ADD COLUMN is_simulated BOOLEAN DEFAULT FALSE,
                ADD COLUMN simulation_reason VARCHAR
                """
            )
            
            # Create hardware_availability_log table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS hardware_availability_log (
                    log_id INTEGER PRIMARY KEY,
                    hardware_id INTEGER NOT NULL,
                    is_available BOOLEAN NOT NULL,
                    detection_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    reason VARCHAR,
                    detected_on_host VARCHAR,
                    FOREIGN KEY (hardware_id) REFERENCES hardware_platforms(hardware_id)
                )
                """
            )
            
            logger.info("Successfully added simulation columns and hardware_availability_log table.")
            return True
            
        except Exception as e:
            logger.error(f"Error adding simulation columns: {e}")
            return False
        finally:
            conn.close()
    
    def get_simulated_results(self) -> pd.DataFrame:
        """
        Get all simulated benchmark results.
        
        Returns:
            DataFrame with simulated benchmark results
        """
        # Check if simulation columns exist
        if not self.check_simulation_columns_exist():
            logger.error("Simulation columns do not exist in the database.")
            return pd.DataFrame()
        
        conn = self._get_connection()
        
        try:
            # Query simulated performance results
            df = conn.execute(
                """
                SELECT 
                    m.model_name,
                    hp.hardware_type,
                    pr.batch_size,
                    pr.precision,
                    pr.throughput_items_per_second,
                    pr.average_latency_ms,
                    pr.is_simulated,
                    pr.simulation_reason,
                    pr.created_at
                FROM 
                    performance_results pr
                JOIN 
                    models m ON pr.model_id = m.model_id
                JOIN 
                    hardware_platforms hp ON pr.hardware_id = hp.hardware_id
                WHERE 
                    pr.is_simulated = TRUE
                ORDER BY 
                    pr.created_at DESC
                """
            ).fetch_df()
            
            logger.info(f"Found {len(df)} simulated benchmark results.")
            return df
            
        except Exception as e:
            logger.error(f"Error getting simulated results: {e}")
            return pd.DataFrame()
        finally:
            conn.close()
    
    def get_hardware_availability(self) -> pd.DataFrame:
        """
        Get hardware availability log.
        
        Returns:
            DataFrame with hardware availability log
        """
        # Check if simulation columns exist
        if not self.check_simulation_columns_exist():
            logger.error("Simulation columns do not exist in the database.")
            return pd.DataFrame()
        
        conn = self._get_connection()
        
        try:
            # Query hardware availability log
            df = conn.execute(
                """
                SELECT 
                    hp.hardware_type,
                    hp.device_name,
                    hal.is_available,
                    hal.detection_timestamp,
                    hal.reason,
                    hal.detected_on_host
                FROM 
                    hardware_availability_log hal
                JOIN 
                    hardware_platforms hp ON hal.hardware_id = hp.hardware_id
                ORDER BY 
                    hal.detection_timestamp DESC
                """
            ).fetch_df()
            
            logger.info(f"Found {len(df)} hardware availability log entries.")
            return df
            
        except Exception as e:
            logger.error(f"Error getting hardware availability: {e}")
            return pd.DataFrame()
        finally:
            conn.close()
    
    def mark_result_as_simulated(self, result_id: int, reason: str) -> bool:
        """
        Mark a benchmark result as simulated.
        
        Args:
            result_id: ID of the benchmark result
            reason: Reason for simulation
            
        Returns:
            True if marked successfully, False otherwise
        """
        # Check if simulation columns exist
        if not self.check_simulation_columns_exist():
            logger.error("Simulation columns do not exist in the database.")
            return False
        
        conn = self._get_connection()
        
        try:
            # Mark result as simulated
            conn.execute(
                """
                UPDATE performance_results 
                SET is_simulated = TRUE, simulation_reason = ? 
                WHERE result_id = ?
                """,
                [reason, result_id]
            )
            
            logger.info(f"Marked result_id {result_id} as simulated: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Error marking result as simulated: {e}")
            return False
        finally:
            conn.close()
    
    def mark_hardware_as_simulated(self, hardware_type: str, reason: str) -> int:
        """
        Mark all benchmark results for a hardware type as simulated.
        
        Args:
            hardware_type: Type of hardware
            reason: Reason for simulation
            
        Returns:
            Number of results marked as simulated
        """
        # Check if simulation columns exist
        if not self.check_simulation_columns_exist():
            logger.error("Simulation columns do not exist in the database.")
            return 0
        
        conn = self._get_connection()
        
        try:
            # Get hardware ID
            hardware_id_result = conn.execute(
                "SELECT hardware_id FROM hardware_platforms WHERE hardware_type = ?",
                [hardware_type]
            ).fetchone()
            
            if not hardware_id_result:
                logger.error(f"Hardware type not found: {hardware_type}")
                return 0
            
            hardware_id = hardware_id_result[0]
            
            # Log hardware unavailability
            max_log_id = conn.execute(
                "SELECT COALESCE(MAX(log_id), 0) FROM hardware_availability_log"
            ).fetchone()[0]
            
            next_log_id = max_log_id + 1
            
            conn.execute(
                """
                INSERT INTO hardware_availability_log (
                    log_id, hardware_id, is_available, reason, detected_on_host
                )
                VALUES (?, ?, FALSE, ?, ?)
                """,
                [next_log_id, hardware_id, reason, os.uname().nodename]
            )
            
            # Mark performance results as simulated
            perf_result = conn.execute(
                """
                UPDATE performance_results 
                SET is_simulated = TRUE, simulation_reason = ? 
                WHERE hardware_id = ? AND (is_simulated IS NULL OR is_simulated = FALSE)
                """,
                [reason, hardware_id]
            )
            
            # Mark compatibility results as simulated
            compat_result = conn.execute(
                """
                UPDATE hardware_compatibility 
                SET is_simulated = TRUE, simulation_reason = ? 
                WHERE hardware_id = ? AND (is_simulated IS NULL OR is_simulated = FALSE)
                """,
                [reason, hardware_id]
            )
            
            # Get count of updated rows (not reliable in DuckDB, so we'll query)
            count_result = conn.execute(
                """
                SELECT COUNT(*) FROM performance_results 
                WHERE hardware_id = ? AND is_simulated = TRUE
                """,
                [hardware_id]
            ).fetchone()[0]
            
            logger.info(f"Marked {count_result} benchmark results for {hardware_type} as simulated: {reason}")
            return count_result
            
        except Exception as e:
            logger.error(f"Error marking hardware as simulated: {e}")
            return 0
        finally:
            conn.close()
    
    def mark_hardware_as_available(self, hardware_type: str) -> bool:
        """
        Mark a hardware type as available.
        
        Args:
            hardware_type: Type of hardware
            
        Returns:
            True if marked successfully, False otherwise
        """
        # Check if simulation columns exist
        if not self.check_simulation_columns_exist():
            logger.error("Simulation columns do not exist in the database.")
            return False
        
        conn = self._get_connection()
        
        try:
            # Get hardware ID
            hardware_id_result = conn.execute(
                "SELECT hardware_id FROM hardware_platforms WHERE hardware_type = ?",
                [hardware_type]
            ).fetchone()
            
            if not hardware_id_result:
                logger.error(f"Hardware type not found: {hardware_type}")
                return False
            
            hardware_id = hardware_id_result[0]
            
            # Log hardware availability
            max_log_id = conn.execute(
                "SELECT COALESCE(MAX(log_id), 0) FROM hardware_availability_log"
            ).fetchone()[0]
            
            next_log_id = max_log_id + 1
            
            conn.execute(
                """
                INSERT INTO hardware_availability_log (
                    log_id, hardware_id, is_available, detected_on_host
                )
                VALUES (?, ?, TRUE, ?)
                """,
                [next_log_id, hardware_id, os.uname().nodename]
            )
            
            logger.info(f"Marked hardware {hardware_type} as available.")
            return True
            
        except Exception as e:
            logger.error(f"Error marking hardware as available: {e}")
            return False
        finally:
            conn.close()
    
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
        
        # Try to import the benchmark runner
        try:
            sys.path.append(str(Path(__file__).parent.parent))
            from core.run_benchmark_with_db import BenchmarkRunner
            runner = BenchmarkRunner(db_path=self.db_path)
        except ImportError:
            logger.error("Failed to import BenchmarkRunner. Using subprocess instead.")
            runner = None
        
        # Run benchmarks for each model-hardware combination
        all_successful = True
        for (model, hardware), batch_sizes in grouped_benchmarks.items():
            batch_sizes_str = ",".join([str(bs) for bs in batch_sizes])
            logger.info(f"Running benchmarks for {model} on {hardware} with batch sizes {batch_sizes_str}")
            
            if runner:
                # Use the benchmark runner directly
                try:
                    batch_sizes_list = [int(bs) for bs in batch_sizes]
                    summary = runner.run_benchmarks(
                        model_names=[model],
                        hardware_types=[hardware],
                        batch_sizes=batch_sizes_list
                    )
                    
                    success = summary.get('successful', 0) > 0
                    total = summary.get('total', 0)
                    failed = summary.get('failed', 0)
                    
                    logger.info(f"Benchmark completion: {success}")
                    logger.info(f"Total benchmarks: {total}, Successful: {summary.get('successful', 0)}, Failed: {failed}")
                    
                    if not success or failed > 0:
                        all_successful = False
                        
                except Exception as e:
                    logger.error(f"Error running benchmarks: {e}")
                    all_successful = False
            else:
                # Use subprocess as fallback
                import subprocess
                
                # Construct command to run
                cmd = [
                    sys.executable,
                    "-m", "duckdb_api.core.run_benchmark_with_db",
                    "--model", model,
                    "--hardware", hardware,
                    "--batch-sizes", batch_sizes_str,
                    "--db-path", self.db_path
                ]
                
                logger.info(f"Running command: {' '.join(cmd)}")
                
                try:
                    result = subprocess.run(cmd, check=False)
                    success = result.returncode == 0
                    
                    if not success:
                        all_successful = False
                        logger.error(f"Benchmark failed with return code: {result.returncode}")
                    else:
                        logger.info("Benchmark completed successfully")
                        
                except Exception as e:
                    logger.error(f"Error running benchmark command: {e}")
                    all_successful = False
        
        return all_successful
    
    def generate_simulation_report(self, output_file: Optional[str] = None) -> Dict:
        """
        Generate a report on simulated benchmark results.
        
        Args:
            output_file: Path to output file, or None to not write to file
            
        Returns:
            Dictionary with simulation report data
        """
        # Check if simulation columns exist
        if not self.check_simulation_columns_exist():
            logger.error("Simulation columns do not exist in the database.")
            return {}
        
        conn = self._get_connection()
        
        try:
            # Get count of simulated results by hardware type
            simulation_by_hardware = conn.execute(
                """
                SELECT 
                    hp.hardware_type,
                    COUNT(*) as count,
                    MAX(pr.created_at) as latest_simulation
                FROM 
                    performance_results pr
                JOIN 
                    hardware_platforms hp ON pr.hardware_id = hp.hardware_id
                WHERE 
                    pr.is_simulated = TRUE
                GROUP BY 
                    hp.hardware_type
                ORDER BY 
                    count DESC
                """
            ).fetch_df()
            
            # Get count of simulated results by model
            simulation_by_model = conn.execute(
                """
                SELECT 
                    m.model_name,
                    COUNT(*) as count,
                    MAX(pr.created_at) as latest_simulation
                FROM 
                    performance_results pr
                JOIN 
                    models m ON pr.model_id = m.model_id
                WHERE 
                    pr.is_simulated = TRUE
                GROUP BY 
                    m.model_name
                ORDER BY 
                    count DESC
                """
            ).fetch_df()
            
            # Get count of simulated results by reason
            simulation_by_reason = conn.execute(
                """
                SELECT 
                    simulation_reason,
                    COUNT(*) as count,
                    MAX(created_at) as latest_simulation
                FROM 
                    performance_results
                WHERE 
                    is_simulated = TRUE
                GROUP BY 
                    simulation_reason
                ORDER BY 
                    count DESC
                """
            ).fetch_df()
            
            # Get total counts
            total_results = conn.execute(
                "SELECT COUNT(*) FROM performance_results"
            ).fetchone()[0]
            
            total_simulated = conn.execute(
                "SELECT COUNT(*) FROM performance_results WHERE is_simulated = TRUE"
            ).fetchone()[0]
            
            # Convert DataFrames to dictionaries
            report = {
                "total_results": total_results,
                "total_simulated": total_simulated,
                "simulation_percentage": (total_simulated / total_results * 100) if total_results > 0 else 0,
                "by_hardware": simulation_by_hardware.to_dict(orient="records"),
                "by_model": simulation_by_model.to_dict(orient="records"),
                "by_reason": simulation_by_reason.to_dict(orient="records"),
                "generated_at": datetime.datetime.now().isoformat(),
            }
            
            # Write to file if requested
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(report, f, indent=2)
                logger.info(f"Wrote simulation report to: {output_file}")
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating simulation report: {e}")
            return {}
        finally:
            conn.close()
    
    def update_db_schema_for_simulation(self) -> bool:
        """
        Update the database schema to add simulation tracking.
        
        Returns:
            True if updated successfully, False otherwise
        """
        # Check if simulation columns already exist
        if self.check_simulation_columns_exist():
            logger.info("Simulation columns already exist. No changes needed.")
            return True
        
        # Add simulation columns
        return self.add_simulation_columns()

def main():
    """Command-line interface for the simulation detection utility."""
    parser = argparse.ArgumentParser(description="Simulation Detection Utility")
    parser.add_argument("--db-path", default="./benchmark_db.duckdb",
                       help="Path to the DuckDB database")
    parser.add_argument("--update-schema", action="store_true",
                       help="Update database schema to add simulation tracking")
    parser.add_argument("--check-schema", action="store_true",
                       help="Check if simulation columns exist in the database")
    parser.add_argument("--get-simulated", action="store_true",
                       help="Get all simulated benchmark results")
    parser.add_argument("--get-availability", action="store_true",
                       help="Get hardware availability log")
    parser.add_argument("--mark-result", type=int,
                       help="Mark a benchmark result as simulated by result_id")
    parser.add_argument("--mark-hardware", type=str,
                       help="Mark all benchmark results for a hardware type as simulated")
    parser.add_argument("--mark-available", type=str,
                       help="Mark a hardware type as available")
    parser.add_argument("--reason", type=str,
                       help="Reason for simulation")
    parser.add_argument("--report", action="store_true",
                       help="Generate a report on simulated benchmark results")
    parser.add_argument("--run-benchmarks", type=str,
                       help="Path to CSV file with benchmark configurations to run")
    parser.add_argument("--output", type=str,
                       help="Output file for report or simulated results")
    parser.add_argument("--format", choices=['csv', 'json', 'markdown', 'html'], default='csv',
                       help="Output format for simulated results")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    args = parser.parse_args()
    
    # Create simulation detection utility
    simulation = SimulationDetection(db_path=args.db_path, debug=args.debug)
    
    # Process commands
    if args.update_schema:
        success = simulation.update_db_schema_for_simulation()
        if not success:
            sys.exit(1)
    
    elif args.check_schema:
        exists = simulation.check_simulation_columns_exist()
        print(f"Simulation columns exist: {exists}")
    
    elif args.get_simulated:
        simulated_df = simulation.get_simulated_results()
        if not simulated_df.empty:
            if args.output:
                if args.format == 'csv':
                    simulated_df.to_csv(args.output, index=False)
                elif args.format == 'json':
                    simulated_df.to_json(args.output, orient='records', indent=2)
                elif args.format == 'markdown':
                    with open(args.output, 'w') as f:
                        f.write(simulated_df.to_markdown(index=False))
                elif args.format == 'html':
                    with open(args.output, 'w') as f:
                        f.write(simulated_df.to_html(index=False))
                logger.info(f"Wrote {len(simulated_df)} simulated results to {args.output}")
            else:
                print(simulated_df.to_string(index=False))
    
    elif args.get_availability:
        availability_df = simulation.get_hardware_availability()
        if not availability_df.empty:
            if args.output:
                if args.format == 'csv':
                    availability_df.to_csv(args.output, index=False)
                elif args.format == 'json':
                    availability_df.to_json(args.output, orient='records', indent=2)
                elif args.format == 'markdown':
                    with open(args.output, 'w') as f:
                        f.write(availability_df.to_markdown(index=False))
                elif args.format == 'html':
                    with open(args.output, 'w') as f:
                        f.write(availability_df.to_html(index=False))
                logger.info(f"Wrote {len(availability_df)} availability log entries to {args.output}")
            else:
                print(availability_df.to_string(index=False))
    
    elif args.mark_result:
        if not args.reason:
            logger.error("--reason is required when marking a result as simulated")
            sys.exit(1)
        
        success = simulation.mark_result_as_simulated(args.mark_result, args.reason)
        if not success:
            sys.exit(1)
    
    elif args.mark_hardware:
        if not args.reason:
            logger.error("--reason is required when marking hardware as simulated")
            sys.exit(1)
        
        count = simulation.mark_hardware_as_simulated(args.mark_hardware, args.reason)
        print(f"Marked {count} benchmark results for {args.mark_hardware} as simulated")
    
    elif args.mark_available:
        success = simulation.mark_hardware_as_available(args.mark_available)
        if not success:
            sys.exit(1)
    
    elif args.report:
        report = simulation.generate_simulation_report(args.output)
        if not report and not args.output:
            print("No simulation data found.")
        elif not args.output:
            print("Simulation Report:")
            print(f"Total Results: {report['total_results']}")
            print(f"Simulated Results: {report['total_simulated']} ({report['simulation_percentage']:.2f}%)")
            
            print("\nSimulation by Hardware:")
            for item in report.get('by_hardware', []):
                print(f"  {item['hardware_type']}: {item['count']} results")
            
            print("\nSimulation by Model:")
            for item in report.get('by_model', [])[:5]:  # Top 5
                print(f"  {item['model_name']}: {item['count']} results")
            
            print("\nSimulation by Reason:")
            for item in report.get('by_reason', []):
                print(f"  {item['simulation_reason']}: {item['count']} results")
    
    elif args.run_benchmarks:
        if not os.path.exists(args.run_benchmarks):
            logger.error(f"Benchmark configuration file not found: {args.run_benchmarks}")
            sys.exit(1)
            
        try:
            # Load benchmark configurations from CSV
            benchmarks_df = pd.read_csv(args.run_benchmarks)
            
            required_columns = ['model_name', 'hardware_type', 'batch_size']
            missing_columns = [col for col in required_columns if col not in benchmarks_df.columns]
            
            if missing_columns:
                logger.error(f"Missing required columns in benchmark configuration file: {', '.join(missing_columns)}")
                sys.exit(1)
                
            # Run benchmarks
            success = simulation.run_benchmarks(benchmarks_df)
            
            if not success:
                logger.error("Some benchmarks failed. Check logs for details.")
                sys.exit(1)
                
            logger.info("All benchmarks completed successfully.")
            
        except Exception as e:
            logger.error(f"Error running benchmarks: {e}")
            sys.exit(1)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()