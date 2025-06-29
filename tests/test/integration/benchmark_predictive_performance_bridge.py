#!/usr/bin/env python3
"""
Benchmark to Predictive Performance Bridge

This module integrates the Benchmark API with the Predictive Performance API,
allowing benchmark results to be automatically recorded as measurements in the
predictive performance database. This enriches the dataset for hardware recommendations
and enables more accurate performance predictions.
"""

import os
import sys
import json
import logging
import argparse
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple

# Add parent directory to path to allow importing from project root
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(parent_dir))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("benchmark_pp_bridge")

# Import clients
try:
    from test.api_client.predictive_performance_client import (
        PredictivePerformanceClient,
        HardwarePlatform,
        PrecisionType,
        ModelMode
    )
    PREDICTIVE_CLIENT_AVAILABLE = True
except ImportError:
    logger.warning("Predictive Performance client not available")
    PREDICTIVE_CLIENT_AVAILABLE = False

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    logger.warning("DuckDB not available")
    DUCKDB_AVAILABLE = False

class BenchmarkPredictivePerformanceBridge:
    """
    Bridge between Benchmark API and Predictive Performance API.
    
    This class provides functionality to:
    1. Connect to both benchmark and predictive performance databases
    2. Query benchmark results based on various filters
    3. Convert benchmark results to predictive performance measurements
    4. Record measurements in the predictive performance database
    5. Track synchronization status
    6. Generate reports on the integration
    """
    
    def __init__(
        self,
        benchmark_db_path: str = "benchmark_db.duckdb",
        predictive_api_url: str = "http://localhost:8080",
        api_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the bridge.
        
        Args:
            benchmark_db_path: Path to the benchmark DuckDB database
            predictive_api_url: URL of the Predictive Performance API
            api_key: Optional API key for authenticated endpoints
            config: Optional configuration dictionary
        """
        self.benchmark_db_path = benchmark_db_path
        self.predictive_api_url = predictive_api_url
        self.api_key = api_key
        self.config = config or {}
        
        # Connect to benchmark database
        self.benchmark_conn = None
        self.benchmark_connected = False
        if DUCKDB_AVAILABLE:
            try:
                self.benchmark_conn = duckdb.connect(benchmark_db_path)
                self.benchmark_connected = True
                logger.info(f"Connected to benchmark database at {benchmark_db_path}")
            except Exception as e:
                logger.error(f"Error connecting to benchmark database: {e}")
        
        # Initialize predictive performance client
        self.predictive_client = None
        if PREDICTIVE_CLIENT_AVAILABLE:
            try:
                self.predictive_client = PredictivePerformanceClient(
                    base_url=predictive_api_url,
                    api_key=api_key
                )
                logger.info(f"Initialized Predictive Performance client with URL {predictive_api_url}")
            except Exception as e:
                logger.error(f"Error initializing Predictive Performance client: {e}")
        
        # Synchronization tracking
        self.sync_history = []
    
    def check_connections(self) -> Dict[str, bool]:
        """
        Check connections to both systems.
        
        Returns:
            Dictionary with connection status
        """
        status = {
            "benchmark_db": False,
            "predictive_api": False
        }
        
        # Check benchmark database connection
        if self.benchmark_conn:
            try:
                # Run simple query to test connection
                self.benchmark_conn.execute("SELECT 1")
                status["benchmark_db"] = True
            except Exception as e:
                logger.error(f"Benchmark database connection error: {e}")
        
        # Check predictive performance API connection
        if self.predictive_client:
            try:
                # Try to list measurements (minimal API call)
                response = self.predictive_client.list_measurements(limit=1)
                if "error" not in response:
                    status["predictive_api"] = True
                else:
                    logger.error(f"Predictive API connection error: {response['error']}")
            except Exception as e:
                logger.error(f"Predictive API connection error: {e}")
        
        return status
    
    def get_benchmark_schema(self) -> Dict[str, List[Dict[str, str]]]:
        """
        Get the schema of the benchmark database tables.
        
        Returns:
            Dictionary with table schemas
        """
        if not self.benchmark_conn:
            logger.error("No benchmark database connection")
            return {}
        
        tables = {}
        
        try:
            # Get list of tables
            result = self.benchmark_conn.execute("SHOW TABLES").fetchall()
            
            for table_name in [row[0] for row in result]:
                # Get schema for each table
                schema = self.benchmark_conn.execute(f"DESCRIBE {table_name}").fetchall()
                tables[table_name] = [
                    {"column": row[0], "type": row[1]} for row in schema
                ]
        except Exception as e:
            logger.error(f"Error getting benchmark schema: {e}")
        
        return tables
    
    def get_benchmark_results(
        self,
        model_name: Optional[str] = None,
        hardware_platform: Optional[str] = None,
        batch_size: Optional[int] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
        already_synced: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get benchmark results from the benchmark database.
        
        Args:
            model_name: Optional model name filter
            hardware_platform: Optional hardware platform filter
            batch_size: Optional batch size filter
            start_time: Optional start time filter
            end_time: Optional end time filter
            limit: Maximum number of results to return
            already_synced: Whether to include already synced results
            
        Returns:
            List of benchmark results
        """
        if not self.benchmark_conn:
            logger.error("No benchmark database connection")
            return []
        
        try:
            # Build query with filters
            query = """
                SELECT 
                    r.result_id,
                    r.run_id,
                    m.model_name,
                    m.model_family,
                    h.hardware_type AS hardware_platform,
                    r.batch_size,
                    r.precision,
                    r.throughput_items_per_second AS throughput,
                    r.average_latency_ms AS latency,
                    r.memory_peak_mb AS memory_usage,
                    tr.started_at AS timestamp,
                    r.test_case AS mode,
                    r.metrics,
                    COALESCE(r.sync_id, '') AS sync_id
                FROM 
                    performance_results r
                JOIN 
                    models m ON r.model_id = m.model_id
                JOIN 
                    hardware_platforms h ON r.hardware_id = h.hardware_id
                JOIN
                    test_runs tr ON r.run_id = tr.run_id
                WHERE 
                    1=1
            """
            
            # Add filters
            params = []
            
            if model_name:
                query += " AND m.model_name = ?"
                params.append(model_name)
            
            if hardware_platform:
                query += " AND h.hardware_type = ?"
                params.append(hardware_platform)
            
            if batch_size:
                query += " AND r.batch_size = ?"
                params.append(batch_size)
            
            if start_time:
                query += " AND tr.started_at >= ?"
                params.append(start_time)
            
            if end_time:
                query += " AND tr.started_at <= ?"
                params.append(end_time)
            
            if not already_synced:
                query += " AND (r.sync_id IS NULL OR r.sync_id = '')"
            
            # Add order and limit
            query += " ORDER BY tr.started_at DESC LIMIT ?"
            params.append(limit)
            
            # Execute query
            result = self.benchmark_conn.execute(query, params).fetchall()
            
            # Convert to dictionaries
            columns = [
                "result_id", "run_id", "model_name", "model_family", "hardware_platform",
                "batch_size", "precision", "throughput", "latency", "memory_usage",
                "timestamp", "mode", "metrics", "sync_id"
            ]
            
            results = []
            for row in result:
                # Convert row to dictionary
                row_dict = {columns[i]: row[i] for i in range(len(columns))}
                
                # Parse metrics JSON if present
                if row_dict.get("metrics"):
                    try:
                        row_dict["metrics"] = json.loads(row_dict["metrics"])
                    except:
                        row_dict["metrics"] = {}
                
                results.append(row_dict)
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting benchmark results: {e}")
            return []
    
    def convert_to_measurement(self, benchmark_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a benchmark result to a predictive performance measurement.
        
        Args:
            benchmark_result: Benchmark result dictionary
            
        Returns:
            Predictive performance measurement dictionary
        """
        # Extract sequence length from metrics if available
        sequence_length = 128  # Default
        metrics = benchmark_result.get("metrics", {})
        if isinstance(metrics, dict) and "sequence_length" in metrics:
            sequence_length = metrics["sequence_length"]
        
        # Map precision
        precision = benchmark_result.get("precision", "fp32")
        if precision in ["fp32", "fp16", "int8", "int4"]:
            precision_value = precision
        else:
            # Default to fp32 if unknown
            precision_value = "fp32"
        
        # Map mode
        mode = benchmark_result.get("mode", "inference")
        if mode in ["inference", "training"]:
            mode_value = mode
        else:
            # Default to inference if unknown
            mode_value = "inference"
        
        # Create measurement
        measurement = {
            "model_name": benchmark_result.get("model_name"),
            "model_family": benchmark_result.get("model_family"),
            "hardware_platform": benchmark_result.get("hardware_platform"),
            "batch_size": benchmark_result.get("batch_size", 1),
            "sequence_length": sequence_length,
            "precision": precision_value,
            "mode": mode_value,
            "throughput": benchmark_result.get("throughput"),
            "latency": benchmark_result.get("latency"),
            "memory_usage": benchmark_result.get("memory_usage"),
            "source": "benchmark"
        }
        
        return measurement
    
    def sync_result(self, benchmark_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synchronize a single benchmark result to the predictive performance system.
        
        Args:
            benchmark_result: Benchmark result dictionary
            
        Returns:
            Dictionary with synchronization result
        """
        if not self.predictive_client:
            return {"success": False, "error": "Predictive Performance client not available"}
        
        # Already synced?
        if benchmark_result.get("sync_id"):
            return {
                "success": True,
                "message": "Already synced",
                "sync_id": benchmark_result.get("sync_id"),
                "result_id": benchmark_result.get("result_id")
            }
        
        try:
            # Convert to measurement format
            measurement = self.convert_to_measurement(benchmark_result)
            
            # Additional validation
            if not measurement.get("model_name"):
                return {"success": False, "error": "Missing model_name", "result_id": benchmark_result.get("result_id")}
            
            if not measurement.get("hardware_platform"):
                return {"success": False, "error": "Missing hardware_platform", "result_id": benchmark_result.get("result_id")}
            
            # Map hardware platform to enum
            try:
                hardware_platform = HardwarePlatform(measurement["hardware_platform"])
            except ValueError:
                # Default to CPU if unknown
                hardware_platform = HardwarePlatform.CPU
                logger.warning(f"Unknown hardware platform {measurement['hardware_platform']}, defaulting to CPU")
            
            # Record measurement
            result = self.predictive_client.record_measurement(
                model_name=measurement["model_name"],
                model_family=measurement["model_family"],
                hardware_platform=hardware_platform,
                batch_size=measurement["batch_size"],
                sequence_length=measurement["sequence_length"],
                precision=measurement["precision"],
                mode=measurement["mode"],
                throughput=measurement["throughput"],
                latency=measurement["latency"],
                memory_usage=measurement["memory_usage"],
                source="benchmark",
                wait=True
            )
            
            if "error" in result:
                return {
                    "success": False, 
                    "error": result["error"],
                    "result_id": benchmark_result.get("result_id")
                }
            
            # Extract measurement ID
            if "result" in result and "measurement_id" in result["result"]:
                sync_id = result["result"]["measurement_id"]
            else:
                # Generate a fallback sync ID
                sync_id = f"sync-{int(time.time())}-{benchmark_result.get('result_id')}"
            
            # Update benchmark record with sync ID if DuckDB is available
            if self.benchmark_conn:
                try:
                    self.benchmark_conn.execute(
                        "UPDATE performance_results SET sync_id = ? WHERE result_id = ?",
                        [sync_id, benchmark_result.get("result_id")]
                    )
                except Exception as e:
                    logger.error(f"Error updating benchmark record: {e}")
            
            # Return success
            return {
                "success": True,
                "sync_id": sync_id,
                "measurement_id": result.get("result", {}).get("measurement_id"),
                "result_id": benchmark_result.get("result_id")
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "result_id": benchmark_result.get("result_id")
            }
    
    def sync_recent_results(self, limit: int = 50) -> Dict[str, Any]:
        """
        Synchronize recent benchmark results to the predictive performance system.
        
        Args:
            limit: Maximum number of results to synchronize
            
        Returns:
            Dictionary with synchronization results
        """
        # Get recent results
        results = self.get_benchmark_results(limit=limit, already_synced=False)
        
        if not results:
            return {
                "success": True,
                "message": "No results to synchronize",
                "total": 0,
                "synced": 0,
                "failed": 0,
                "details": []
            }
        
        # Synchronize each result
        synced = 0
        failed = 0
        details = []
        
        for result in results:
            sync_result = self.sync_result(result)
            details.append(sync_result)
            
            if sync_result.get("success"):
                synced += 1
            else:
                failed += 1
        
        # Add to sync history
        self.sync_history.append({
            "timestamp": datetime.now().isoformat(),
            "total": len(results),
            "synced": synced,
            "failed": failed
        })
        
        return {
            "success": failed == 0,
            "message": f"Synchronized {synced} results, {failed} failed",
            "total": len(results),
            "synced": synced,
            "failed": failed,
            "details": details
        }
    
    def sync_by_model(
        self,
        model_name: str,
        days: int = 30,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Synchronize benchmark results for a specific model.
        
        Args:
            model_name: Model name to synchronize
            days: Number of days to look back
            limit: Maximum number of results to synchronize
            
        Returns:
            Dictionary with synchronization results
        """
        # Calculate start time
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        # Get results for the model
        results = self.get_benchmark_results(
            model_name=model_name,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            already_synced=False
        )
        
        if not results:
            return {
                "success": True,
                "message": f"No results found for model {model_name}",
                "total": 0,
                "synced": 0,
                "failed": 0,
                "details": []
            }
        
        # Synchronize each result
        synced = 0
        failed = 0
        details = []
        
        for result in results:
            sync_result = self.sync_result(result)
            details.append(sync_result)
            
            if sync_result.get("success"):
                synced += 1
            else:
                failed += 1
        
        # Add to sync history
        self.sync_history.append({
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "total": len(results),
            "synced": synced,
            "failed": failed
        })
        
        return {
            "success": failed == 0,
            "message": f"Synchronized {synced} results for model {model_name}, {failed} failed",
            "total": len(results),
            "synced": synced,
            "failed": failed,
            "details": details
        }
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a report on the synchronization status.
        
        Returns:
            Dictionary with report data
        """
        if not self.benchmark_conn:
            return {
                "success": False,
                "error": "No benchmark database connection"
            }
        
        try:
            # Count total benchmark results
            total_results = self.benchmark_conn.execute(
                "SELECT COUNT(*) FROM performance_results"
            ).fetchone()[0]
            
            # Count synced results
            synced_results = self.benchmark_conn.execute(
                "SELECT COUNT(*) FROM performance_results WHERE sync_id IS NOT NULL AND sync_id <> ''"
            ).fetchone()[0]
            
            # Count unsynced results
            unsynced_results = total_results - synced_results
            
            # Get model coverage
            model_coverage = self.benchmark_conn.execute("""
                SELECT 
                    m.model_name,
                    COUNT(r.result_id) AS total_results,
                    SUM(CASE WHEN r.sync_id IS NOT NULL AND r.sync_id <> '' THEN 1 ELSE 0 END) AS synced_results
                FROM 
                    models m
                JOIN 
                    performance_results r ON r.model_id = m.model_id
                GROUP BY 
                    m.model_name
                ORDER BY 
                    total_results DESC
            """).fetchall()
            
            # Get hardware coverage
            hardware_coverage = self.benchmark_conn.execute("""
                SELECT 
                    h.hardware_type,
                    COUNT(r.result_id) AS total_results,
                    SUM(CASE WHEN r.sync_id IS NOT NULL AND r.sync_id <> '' THEN 1 ELSE 0 END) AS synced_results
                FROM 
                    hardware_platforms h
                JOIN 
                    performance_results r ON r.hardware_id = h.hardware_id
                GROUP BY 
                    h.hardware_type
                ORDER BY 
                    total_results DESC
            """).fetchall()
            
            # Format model coverage
            model_coverage_formatted = [
                {
                    "model_name": row[0],
                    "total_results": row[1],
                    "synced_results": row[2],
                    "sync_percentage": round(row[2] / row[1] * 100 if row[1] > 0 else 0, 2)
                }
                for row in model_coverage
            ]
            
            # Format hardware coverage
            hardware_coverage_formatted = [
                {
                    "hardware_type": row[0],
                    "total_results": row[1],
                    "synced_results": row[2],
                    "sync_percentage": round(row[2] / row[1] * 100 if row[1] > 0 else 0, 2)
                }
                for row in hardware_coverage
            ]
            
            # Generate report
            report = {
                "generated_at": datetime.now().isoformat(),
                "total_results": total_results,
                "synced_results": synced_results,
                "unsynced_results": unsynced_results,
                "sync_percentage": round(synced_results / total_results * 100 if total_results > 0 else 0, 2),
                "model_coverage": model_coverage_formatted,
                "hardware_coverage": hardware_coverage_formatted,
                "sync_history": self.sync_history
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def close(self):
        """Close database connections."""
        if self.benchmark_conn:
            self.benchmark_conn.close()
            self.benchmark_conn = None
            logger.info("Closed benchmark database connection")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Benchmark to Predictive Performance Bridge")
    parser.add_argument("--benchmark-db", type=str, default="benchmark_db.duckdb", 
                      help="Path to benchmark DuckDB database")
    parser.add_argument("--api-url", type=str, default="http://localhost:8080",
                      help="URL of the Predictive Performance API")
    parser.add_argument("--api-key", type=str, help="API key for authenticated endpoints")
    parser.add_argument("--model", type=str, help="Synchronize results for a specific model")
    parser.add_argument("--days", type=int, default=30, help="Number of days to look back")
    parser.add_argument("--limit", type=int, default=100, help="Maximum number of results to synchronize")
    parser.add_argument("--report", action="store_true", help="Generate synchronization report")
    parser.add_argument("--output", type=str, help="Path to write report JSON")
    
    args = parser.parse_args()
    
    # Create bridge
    bridge = BenchmarkPredictivePerformanceBridge(
        benchmark_db_path=args.benchmark_db,
        predictive_api_url=args.api_url,
        api_key=args.api_key
    )
    
    try:
        # Check connections
        status = bridge.check_connections()
        if not status["benchmark_db"]:
            print(f"ERROR: Could not connect to benchmark database at {args.benchmark_db}")
            return 1
        
        if not status["predictive_api"]:
            print(f"ERROR: Could not connect to Predictive Performance API at {args.api_url}")
            return 1
        
        # Synchronize results
        if args.model:
            print(f"Synchronizing results for model {args.model}...")
            result = bridge.sync_by_model(
                model_name=args.model,
                days=args.days,
                limit=args.limit
            )
        else:
            print(f"Synchronizing recent benchmark results (limit: {args.limit})...")
            result = bridge.sync_recent_results(limit=args.limit)
        
        # Print summary
        print(f"Synchronization complete: {result['message']}")
        print(f"Total: {result['total']}, Synced: {result['synced']}, Failed: {result['failed']}")
        
        # Generate report if requested
        if args.report:
            print("Generating synchronization report...")
            report = bridge.generate_report()
            
            # Print report summary
            print(f"Report generated at {report['generated_at']}")
            print(f"Total results: {report['total_results']}")
            print(f"Synced results: {report['synced_results']} ({report['sync_percentage']}%)")
            
            # Write report to file if requested
            if args.output:
                with open(args.output, "w") as f:
                    json.dump(report, f, indent=2)
                print(f"Report written to {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return 1
    
    finally:
        # Close connections
        bridge.close()

if __name__ == "__main__":
    sys.exit(main())