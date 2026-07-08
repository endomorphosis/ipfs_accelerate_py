#!/usr/bin/env python3
"""
Tool for viewing benchmark results with simulation status information.

This script provides a CLI tool for querying and viewing benchmark results
with a focus on simulation tracking and transparency.
"""

import os
import sys
import json
import logging
import argparse
import datetime
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

try:
    import duckdb
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


class BenchmarkResultViewer:
    """
    Tool for viewing benchmark results with simulation status information.
    """
    
    def __init__(self, db_path: str = "./benchmark_db.duckdb", debug: bool = False):
        """
        Initialize the benchmark result viewer.
        
        Args:
            db_path: Path to the DuckDB database
            debug: Enable debug logging
        """
        self.db_path = db_path
        
        # Set up logging
        if debug:
            logger.setLevel(logging.DEBUG)
        
        logger.info(f"Initialized BenchmarkResultViewer with database: {db_path}")
    
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
            result = conn.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'performance_results' 
                AND column_name = 'is_simulated'
            """).fetchone()
            
            if result:
                logger.info("Simulation columns exist in the database.")
                return True
            else:
                logger.info("Simulation columns do not exist in the database.")
                return False
                
        except Exception as e:
            logger.error(f"Error checking simulation columns: {e}")
            return False
        finally:
            conn.close()
    
    def get_simulation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of simulated benchmark results.
        
        Returns:
            Dictionary with simulation summary data
        """
        # Check if simulation columns exist
        if not self.check_simulation_columns_exist():
            logger.error("Simulation columns do not exist in the database.")
            return {}
        
        conn = self._get_connection()
        
        try:
            # Query total results
            total_results = conn.execute(
                "SELECT COUNT(*) FROM performance_results"
            ).fetchone()[0]
            
            # Query simulated results
            simulated_results = conn.execute(
                "SELECT COUNT(*) FROM performance_results WHERE is_simulated = TRUE"
            ).fetchone()[0]
            
            # Query simulation by hardware type
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
            
            # Query simulation by model
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
            
            # Query simulation by reason
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
            
            # Create summary
            summary = {
                "total_results": total_results,
                "total_simulated": simulated_results,
                "simulation_percentage": (simulated_results / total_results * 100) if total_results > 0 else 0,
                "by_hardware": simulation_by_hardware.to_dict(orient="records") if not simulation_by_hardware.empty else [],
                "by_model": simulation_by_model.to_dict(orient="records") if not simulation_by_model.empty else [],
                "by_reason": simulation_by_reason.to_dict(orient="records") if not simulation_by_reason.empty else [],
                "generated_at": datetime.datetime.now().isoformat(),
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting simulation summary: {e}")
            return {}
        finally:
            conn.close()
    
    def get_benchmark_results(self, model_name: Optional[str] = None, 
                            hardware_type: Optional[str] = None,
                            batch_size: Optional[int] = None,
                            simulated_only: bool = False,
                            real_only: bool = False,
                            latest_only: bool = True) -> pd.DataFrame:
        """
        Get benchmark results with simulation information.
        
        Args:
            model_name: Filter by model name (optional)
            hardware_type: Filter by hardware type (optional)
            batch_size: Filter by batch size (optional)
            simulated_only: Only include simulated results
            real_only: Only include real results
            latest_only: Return only the latest results for each model-hardware combination
            
        Returns:
            DataFrame with benchmark results
        """
        conn = self._get_connection()
        
        try:
            # Check if simulation columns exist
            has_sim_columns = self.check_simulation_columns_exist()
            
            # Build SQL query
            if has_sim_columns:
                sql = """
                SELECT 
                    m.model_name,
                    hp.hardware_type,
                    pr.batch_size,
                    pr.sequence_length,
                    pr.average_latency_ms,
                    pr.throughput_items_per_second,
                    pr.memory_peak_mb as memory_mb,
                    pr.is_simulated,
                    pr.simulation_reason,
                    pr.created_at
                FROM 
                    performance_results pr
                JOIN 
                    models m ON pr.model_id = m.model_id
                JOIN 
                    hardware_platforms hp ON pr.hardware_id = hp.hardware_id
                """
            else:
                # No simulation columns, use basic query
                sql = """
                SELECT 
                    m.model_name,
                    hp.hardware_type,
                    pr.batch_size,
                    pr.sequence_length,
                    pr.average_latency_ms,
                    pr.throughput_items_per_second,
                    pr.memory_peak_mb as memory_mb,
                    FALSE as is_simulated,
                    NULL as simulation_reason,
                    pr.created_at
                FROM 
                    performance_results pr
                JOIN 
                    models m ON pr.model_id = m.model_id
                JOIN 
                    hardware_platforms hp ON pr.hardware_id = hp.hardware_id
                """
            
            # Add filter conditions
            conditions = []
            params = {}
            
            if model_name:
                conditions.append("m.model_name = :model_name")
                params['model_name'] = model_name
            
            if hardware_type:
                conditions.append("hp.hardware_type = :hardware_type")
                params['hardware_type'] = hardware_type
            
            if batch_size:
                conditions.append("pr.batch_size = :batch_size")
                params['batch_size'] = batch_size
            
            if simulated_only and has_sim_columns:
                conditions.append("pr.is_simulated = TRUE")
            
            if real_only and has_sim_columns:
                conditions.append("(pr.is_simulated = FALSE OR pr.is_simulated IS NULL)")
            
            if conditions:
                sql += " WHERE " + " AND ".join(conditions)
            
            # Add sorting
            sql += " ORDER BY pr.created_at DESC"
            
            # Execute query
            if params:
                df = conn.execute(sql, params).fetch_df()
            else:
                df = conn.execute(sql).fetch_df()
            
            # Filter to latest only if requested
            if latest_only and not df.empty:
                # Create a key for grouping
                df['group_key'] = df['model_name'] + '_' + df['hardware_type'] + '_' + df['batch_size'].astype(str)
                
                # Sort by latest and drop duplicates
                df = df.sort_values('created_at', ascending=False)
                df = df.drop_duplicates(subset=['group_key'])
                
                # Drop the grouping key
                df = df.drop(columns=['group_key'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting benchmark results: {e}")
            return pd.DataFrame()
        finally:
            conn.close()
    
    def get_hardware_availability(self) -> pd.DataFrame:
        """
        Get hardware availability log.
        
        Returns:
            DataFrame with hardware availability log
        """
        conn = self._get_connection()
        
        try:
            # Check if hardware_availability_log table exists
            result = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='hardware_availability_log'
            """).fetchall()
            
            if not result:
                logger.error("hardware_availability_log table does not exist.")
                return pd.DataFrame()
            
            # Query hardware availability log
            sql = """
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
            
            df = conn.execute(sql).fetch_df()
            return df
            
        except Exception as e:
            logger.error(f"Error getting hardware availability: {e}")
            return pd.DataFrame()
        finally:
            conn.close()
    
    def generate_markdown_report(self, model_name: Optional[str] = None, 
                               hardware_type: Optional[str] = None) -> str:
        """
        Generate a markdown report of benchmark results with simulation status.
        
        Args:
            model_name: Filter by model name (optional)
            hardware_type: Filter by hardware type (optional)
            
        Returns:
            Markdown formatted report
        """
        # Get simulation summary
        summary = self.get_simulation_summary()
        
        # Get benchmark results
        results = self.get_benchmark_results(
            model_name=model_name, 
            hardware_type=hardware_type,
            latest_only=True
        )
        
        # Start building markdown
        markdown = "# Benchmark Results Report\n\n"
        markdown += f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Add simulation summary if available
        if summary:
            markdown += "## Simulation Summary\n\n"
            markdown += f"Total Results: {summary['total_results']}\n\n"
            markdown += f"Simulated Results: {summary['total_simulated']} ({summary['simulation_percentage']:.2f}%)\n\n"
            
            if summary['by_hardware']:
                markdown += "### Simulation by Hardware\n\n"
                markdown += "| Hardware Type | Count | Latest Simulation |\n"
                markdown += "|--------------|------:|------------------:|\n"
                
                for item in summary['by_hardware']:
                    markdown += f"| {item['hardware_type']} | {item['count']} | {item['latest_simulation']} |\n"
                
                markdown += "\n"
            
            if summary['by_model']:
                markdown += "### Simulation by Model\n\n"
                markdown += "| Model | Count | Latest Simulation |\n"
                markdown += "|-------|------:|------------------:|\n"
                
                for item in summary['by_model'][:10]:  # Top 10
                    markdown += f"| {item['model_name']} | {item['count']} | {item['latest_simulation']} |\n"
                
                markdown += "\n"
            
            if summary['by_reason']:
                markdown += "### Simulation by Reason\n\n"
                markdown += "| Reason | Count | Latest Simulation |\n"
                markdown += "|--------|------:|------------------:|\n"
                
                for item in summary['by_reason']:
                    reason = item['simulation_reason'] or "Unknown"
                    markdown += f"| {reason} | {item['count']} | {item['latest_simulation']} |\n"
                
                markdown += "\n"
        
        # Add benchmark results
        if not results.empty:
            markdown += "## Benchmark Results\n\n"
            
            # Format the results DataFrame for markdown
            results_for_md = results.copy()
            
            # Format columns
            results_for_md['average_latency_ms'] = results_for_md['average_latency_ms'].round(2)
            results_for_md['throughput_items_per_second'] = results_for_md['throughput_items_per_second'].round(2)
            results_for_md['memory_mb'] = results_for_md['memory_mb'].round(2)
            
            # Add simulation indicator
            results_for_md['status'] = results_for_md.apply(
                lambda row: "⚠️ Simulated" if row['is_simulated'] else "✅ Real", axis=1
            )
            
            # Select columns for display
            display_columns = [
                'model_name', 'hardware_type', 'batch_size', 
                'average_latency_ms', 'throughput_items_per_second', 
                'memory_mb', 'status'
            ]
            
            # Convert to markdown table
            markdown += results_for_md[display_columns].to_markdown(index=False) + "\n\n"
            
            # Add simulation details for simulated results
            simulated_results = results[results['is_simulated'] == True]
            if not simulated_results.empty:
                markdown += "### Simulation Details\n\n"
                markdown += "| Model | Hardware | Batch Size | Simulation Reason |\n"
                markdown += "|-------|----------|------------|-------------------|\n"
                
                for _, row in simulated_results.iterrows():
                    reason = row['simulation_reason'] or "Unknown"
                    markdown += f"| {row['model_name']} | {row['hardware_type']} | {row['batch_size']} | {reason} |\n"
                
                markdown += "\n"
        else:
            markdown += "## No benchmark results found\n\n"
        
        # Add hardware availability if available
        availability = self.get_hardware_availability()
        if not availability.empty:
            markdown += "## Hardware Availability Log\n\n"
            
            # Convert to markdown table
            markdown += availability.to_markdown(index=False) + "\n\n"
        
        return markdown

def main():
    """Command-line interface for the benchmark result viewer."""
    parser = argparse.ArgumentParser(description="Benchmark Result Viewer")
    parser.add_argument("--db-path", default="./benchmark_db.duckdb",
                       help="Path to the DuckDB database")
    parser.add_argument("--check-simulation", action="store_true",
                       help="Check if simulation columns exist in the database")
    parser.add_argument("--model", type=str,
                       help="Filter by model name")
    parser.add_argument("--hardware", type=str,
                       help="Filter by hardware type")
    parser.add_argument("--batch-size", type=int,
                       help="Filter by batch size")
    parser.add_argument("--simulated-only", action="store_true",
                       help="Only include simulated results")
    parser.add_argument("--real-only", action="store_true",
                       help="Only include real results")
    parser.add_argument("--latest-only", action="store_true", default=True,
                       help="Only include latest results for each model-hardware combination")
    parser.add_argument("--format", choices=["csv", "json", "markdown", "html"], default="markdown",
                       help="Output format")
    parser.add_argument("--output", type=str,
                       help="Output file for results")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    args = parser.parse_args()
    
    # Set up environment variable for database path
    db_path = args.db_path
    if not db_path:
        db_path = os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")
    
    # Create viewer
    viewer = BenchmarkResultViewer(db_path=db_path, debug=args.debug)
    
    # Check simulation columns if requested
    if args.check_simulation:
        exists = viewer.check_simulation_columns_exist()
        print(f"Simulation columns exist: {exists}")
        if not exists:
            print("Run 'python -m duckdb_api.schema.update_db_schema_for_simulation' to add simulation tracking.")
        return
    
    # Get results
    results = viewer.get_benchmark_results(
        model_name=args.model,
        hardware_type=args.hardware,
        batch_size=args.batch_size,
        simulated_only=args.simulated_only,
        real_only=args.real_only,
        latest_only=args.latest_only
    )
    
    if results.empty:
        print("No benchmark results found.")
        return
    
    # Format output
    if args.format == "csv":
        output = results.to_csv(index=False)
    elif args.format == "json":
        output = results.to_json(orient="records", indent=2)
    elif args.format == "html":
        output = results.to_html(index=False)
    else:  # markdown
        output = viewer.generate_markdown_report(model_name=args.model, hardware_type=args.hardware)
    
    # Output
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Results written to {args.output}")
    else:
        print(output)

if __name__ == "__main__":
    main()