#!/usr/bin/env python3
"""
IPFS Accelerate Python - Database Handler

This module provides a database handler for storing IPFS acceleration results
in a DuckDB database. It supports storing test results, IPFS operations,
and hardware detection information.
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ipfs_accelerate_db")

# Determine if we should use DuckDB for result storage
ENABLE_DB_STORAGE = os.environ.get("ENABLE_DB_STORAGE", "1").lower() in ("1", "true", "yes")
DEPRECATE_JSON_OUTPUT = os.environ.get("DEPRECATE_JSON_OUTPUT", "1").lower() in ("1", "true", "yes")

# Try to import DuckDB
try:
    import duckdb
    HAVE_DUCKDB = True
    if ENABLE_DB_STORAGE:
        logger.info("DuckDB support enabled for IPFS Accelerate results")
except ImportError:
    HAVE_DUCKDB = False
    if ENABLE_DB_STORAGE:
        logger.warning("DuckDB not installed but ENABLE_DB_STORAGE=1. Will use JSON fallback.")
        logger.warning("To enable database storage, install duckdb: pip install duckdb pandas")


class DatabaseHandler:
    """
    Handler for storing IPFS Accelerate results in DuckDB database.
    This class abstracts away the database operations to store results.
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize the database handler.
        
        Args:
            db_path: Path to DuckDB database file. If None, uses BENCHMARK_DB_PATH
                    environment variable or default path ./benchmark_db.duckdb
        """
        # Skip initialization if DuckDB is not available
        if not HAVE_DUCKDB:
            self.db_path = None
            self.con = None
            logger.info("DuckDB not available - results will not be stored in database")
            return
            
        # Get database path from environment or argument
        if db_path is None:
            self.db_path = os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")
        else:
            self.db_path = db_path
            
        try:
            # Connect to DuckDB database
            self.con = duckdb.connect(self.db_path)
            logger.info(f"Connected to DuckDB database at: {self.db_path}")
            
            # Create necessary tables
            self._create_tables()
        except Exception as e:
            logger.warning(f"Failed to initialize database connection: {e}")
            self.con = None
            
    def _create_tables(self):
        """Create necessary tables if they don't exist."""
        if self.con is None:
            return
            
        try:
            # Create ipfs_acceleration_results table for storing IPFS acceleration test results
            self.con.execute("""
                CREATE TABLE IF NOT EXISTS ipfs_acceleration_results (
                    id INTEGER PRIMARY KEY,
                    run_id VARCHAR,
                    model_name VARCHAR,
                    endpoint_type VARCHAR,
                    acceleration_type VARCHAR,
                    status VARCHAR,
                    success BOOLEAN,
                    execution_time_ms FLOAT,
                    implementation_type VARCHAR,
                    error_message VARCHAR,
                    additional_data VARCHAR,
                    test_date TIMESTAMP
                )
            """)
            
            # Create ipfs_operations table for tracking IPFS operations
            self.con.execute("""
                CREATE TABLE IF NOT EXISTS ipfs_operations (
                    id INTEGER PRIMARY KEY,
                    operation_type VARCHAR,
                    cid VARCHAR,
                    success BOOLEAN,
                    execution_time_ms FLOAT,
                    file_size INTEGER,
                    error_message VARCHAR,
                    timestamp TIMESTAMP
                )
            """)
            
            # Create hardware_detection table for storing hardware detection results
            self.con.execute("""
                CREATE TABLE IF NOT EXISTS hardware_detection (
                    id INTEGER PRIMARY KEY,
                    hardware_type VARCHAR,
                    device_name VARCHAR,
                    available BOOLEAN,
                    details VARCHAR,
                    timestamp TIMESTAMP
                )
            """)
            
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
            
    def store_acceleration_result(self, result: Dict[str, Any], run_id: str = None) -> bool:
        """
        Store IPFS acceleration result in the database.
        
        Args:
            result: Result dictionary
            run_id: Optional run ID to group results
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not HAVE_DUCKDB or self.con is None:
            return False
            
        try:
            # Generate run_id if not provided
            if run_id is None:
                run_id = f"accel_{int(datetime.now().timestamp())}"
                
            # Extract values from the result dictionary
            model_name = result.get('model_name', 'unknown')
            endpoint_type = result.get('endpoint_type', 'unknown')
            acceleration_type = result.get('acceleration_type', 'unknown')
            status = result.get('status', 'unknown')
            success = result.get('success', False)
            execution_time_ms = result.get('execution_time_ms', None)
            implementation_type = result.get('implementation_type', 'unknown')
            error_message = result.get('error_message', None)
            
            # Serialize additional data
            additional_data = {}
            for key, value in result.items():
                if key not in ['model_name', 'endpoint_type', 'acceleration_type', 'status', 
                              'success', 'execution_time_ms', 'implementation_type', 'error_message']:
                    additional_data[key] = value
                    
            additional_data_json = json.dumps(additional_data)
            
            # Insert the result into the database
            self.con.execute("""
                INSERT INTO ipfs_acceleration_results (
                    run_id, model_name, endpoint_type, acceleration_type, status,
                    success, execution_time_ms, implementation_type, error_message,
                    additional_data, test_date
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                run_id, model_name, endpoint_type, acceleration_type, status,
                success, execution_time_ms, implementation_type, error_message,
                additional_data_json, datetime.now()
            ])
            
            logger.debug(f"Stored acceleration result for {model_name} in database")
            return True
        except Exception as e:
            logger.error(f"Error storing acceleration result: {e}")
            return False
            
    def store_ipfs_operation(self, operation_type: str, cid: str, success: bool, 
                           execution_time_ms: float = None, file_size: int = None,
                           error_message: str = None) -> bool:
        """
        Store IPFS operation in the database.
        
        Args:
            operation_type: Type of operation ('add', 'get', etc.)
            cid: Content identifier
            success: Whether the operation was successful
            execution_time_ms: Execution time in milliseconds
            file_size: Size of the file in bytes
            error_message: Error message if operation failed
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not HAVE_DUCKDB or self.con is None:
            return False
            
        try:
            # Insert the operation into the database
            self.con.execute("""
                INSERT INTO ipfs_operations (
                    operation_type, cid, success, execution_time_ms,
                    file_size, error_message, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [
                operation_type, cid, success, execution_time_ms,
                file_size, error_message, datetime.now()
            ])
            
            logger.debug(f"Stored IPFS operation {operation_type} for CID {cid} in database")
            return True
        except Exception as e:
            logger.error(f"Error storing IPFS operation: {e}")
            return False
            
    def store_hardware_detection(self, hardware_type: str, available: bool, 
                               device_name: str = None, details: Dict = None) -> bool:
        """
        Store hardware detection result in the database.
        
        Args:
            hardware_type: Type of hardware ('cpu', 'cuda', etc.)
            available: Whether the hardware is available
            device_name: Name of the device
            details: Additional details as dictionary
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not HAVE_DUCKDB or self.con is None:
            return False
            
        try:
            # Serialize details
            details_json = json.dumps(details) if details else None
            
            # Insert the hardware detection result into the database
            self.con.execute("""
                INSERT INTO hardware_detection (
                    hardware_type, device_name, available, details, timestamp
                ) VALUES (?, ?, ?, ?, ?)
            """, [
                hardware_type, device_name, available, details_json, datetime.now()
            ])
            
            logger.debug(f"Stored hardware detection for {hardware_type} in database")
            return True
        except Exception as e:
            logger.error(f"Error storing hardware detection: {e}")
            return False
            
    def is_available(self) -> bool:
        """Check if database storage is available."""
        return HAVE_DUCKDB and self.con is not None
    
    def generate_report(self, days: int = 7, format: str = 'markdown') -> str:
        """
        Generate a report of IPFS acceleration activity.
        
        Args:
            days: Number of days to include in the report
            format: Report format ('markdown', 'html', or 'json')
            
        Returns:
            str: Report text
        """
        if not self.is_available():
            return "Database not available, cannot generate report."
            
        try:
            # Get acceleration results
            accel_results = self.con.execute(f"""
                SELECT 
                    model_name, endpoint_type, acceleration_type,
                    COUNT(*) as total_runs,
                    SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful_runs,
                    AVG(execution_time_ms) as avg_time_ms
                FROM ipfs_acceleration_results
                WHERE test_date >= CURRENT_DATE - {days}
                GROUP BY model_name, endpoint_type, acceleration_type
                ORDER BY total_runs DESC
            """).fetchall()
            
            # Get IPFS operations
            ipfs_ops = self.con.execute(f"""
                SELECT 
                    operation_type,
                    COUNT(*) as total_ops,
                    SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful_ops,
                    AVG(execution_time_ms) as avg_time_ms,
                    AVG(file_size) as avg_file_size
                FROM ipfs_operations
                WHERE timestamp >= CURRENT_DATE - {days}
                GROUP BY operation_type
                ORDER BY total_ops DESC
            """).fetchall()
            
            # Get hardware detection
            hardware = self.con.execute(f"""
                SELECT 
                    hardware_type,
                    COUNT(*) as detection_count,
                    SUM(CASE WHEN available THEN 1 ELSE 0 END) as available_count
                FROM hardware_detection
                WHERE timestamp >= CURRENT_DATE - {days}
                GROUP BY hardware_type
                ORDER BY detection_count DESC
            """).fetchall()
            
            # Format the report
            if format.lower() == 'markdown':
                return self._format_markdown_report(accel_results, ipfs_ops, hardware, days)
            elif format.lower() == 'html':
                return self._format_html_report(accel_results, ipfs_ops, hardware, days)
            elif format.lower() == 'json':
                return self._format_json_report(accel_results, ipfs_ops, hardware, days)
            else:
                return f"Unsupported format: {format}"
                
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return f"Error generating report: {e}"
            
    def _format_markdown_report(self, accel_results, ipfs_ops, hardware, days):
        """Format report as markdown."""
        report = []
        
        # Header
        report.append("# IPFS Accelerate Report")
        report.append(f"\nReport generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Period: Last {days} days\n")
        
        # Acceleration results
        report.append("## Acceleration Results")
        if accel_results:
            report.append("\n| Model | Endpoint | Acceleration | Total Runs | Success Rate | Avg Time (ms) |")
            report.append("|-------|----------|--------------|------------|--------------|---------------|")
            
            for row in accel_results:
                model, endpoint, accel_type, total, success, avg_time = row
                success_rate = (success / total * 100) if total > 0 else 0
                avg_time_str = f"{avg_time:.2f}" if avg_time is not None else "N/A"
                
                report.append(f"| {model} | {endpoint} | {accel_type} | {total} | {success_rate:.1f}% | {avg_time_str} |")
        else:
            report.append("\nNo acceleration results found for this period.")
            
        # IPFS operations
        report.append("\n## IPFS Operations")
        if ipfs_ops:
            report.append("\n| Operation | Total | Success Rate | Avg Time (ms) | Avg Size (bytes) |")
            report.append("|-----------|-------|--------------|---------------|------------------|")
            
            for row in ipfs_ops:
                op_type, total, success, avg_time, avg_size = row
                success_rate = (success / total * 100) if total > 0 else 0
                avg_time_str = f"{avg_time:.2f}" if avg_time is not None else "N/A"
                avg_size_str = f"{avg_size:.0f}" if avg_size is not None else "N/A"
                
                report.append(f"| {op_type} | {total} | {success_rate:.1f}% | {avg_time_str} | {avg_size_str} |")
        else:
            report.append("\nNo IPFS operations found for this period.")
            
        # Hardware detection
        report.append("\n## Hardware Detection")
        if hardware:
            report.append("\n| Hardware | Detection Count | Availability Rate |")
            report.append("|----------|-----------------|-------------------|")
            
            for row in hardware:
                hw_type, count, available = row
                avail_rate = (available / count * 100) if count > 0 else 0
                
                report.append(f"| {hw_type} | {count} | {avail_rate:.1f}% |")
        else:
            report.append("\nNo hardware detection results found for this period.")
            
        return "\n".join(report)
        
    def _format_html_report(self, accel_results, ipfs_ops, hardware, days):
        """Format report as HTML."""
        html = []
        
        # Header
        html.append("<!DOCTYPE html>")
        html.append("<html>")
        html.append("<head>")
        html.append("    <title>IPFS Accelerate Report</title>")
        html.append("    <style>")
        html.append("        body { font-family: Arial, sans-serif; margin: 20px; }")
        html.append("        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }")
        html.append("        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
        html.append("        th { background-color: #f2f2f2; }")
        html.append("        tr:nth-child(even) { background-color: #f9f9f9; }")
        html.append("        h1, h2 { color: #333; }")
        html.append("    </style>")
        html.append("</head>")
        html.append("<body>")
        
        # Header
        html.append(f"<h1>IPFS Accelerate Report</h1>")
        html.append(f"<p>Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
        html.append(f"<p>Period: Last {days} days</p>")
        
        # Acceleration results
        html.append("<h2>Acceleration Results</h2>")
        if accel_results:
            html.append("<table>")
            html.append("    <tr><th>Model</th><th>Endpoint</th><th>Acceleration</th>" +
                       "<th>Total Runs</th><th>Success Rate</th><th>Avg Time (ms)</th></tr>")
            
            for row in accel_results:
                model, endpoint, accel_type, total, success, avg_time = row
                success_rate = (success / total * 100) if total > 0 else 0
                avg_time_str = f"{avg_time:.2f}" if avg_time is not None else "N/A"
                
                html.append(f"    <tr><td>{model}</td><td>{endpoint}</td><td>{accel_type}</td>" +
                          f"<td>{total}</td><td>{success_rate:.1f}%</td><td>{avg_time_str}</td></tr>")
                
            html.append("</table>")
        else:
            html.append("<p>No acceleration results found for this period.</p>")
            
        # IPFS operations
        html.append("<h2>IPFS Operations</h2>")
        if ipfs_ops:
            html.append("<table>")
            html.append("    <tr><th>Operation</th><th>Total</th><th>Success Rate</th>" +
                       "<th>Avg Time (ms)</th><th>Avg Size (bytes)</th></tr>")
            
            for row in ipfs_ops:
                op_type, total, success, avg_time, avg_size = row
                success_rate = (success / total * 100) if total > 0 else 0
                avg_time_str = f"{avg_time:.2f}" if avg_time is not None else "N/A"
                avg_size_str = f"{avg_size:.0f}" if avg_size is not None else "N/A"
                
                html.append(f"    <tr><td>{op_type}</td><td>{total}</td><td>{success_rate:.1f}%</td>" +
                          f"<td>{avg_time_str}</td><td>{avg_size_str}</td></tr>")
                
            html.append("</table>")
        else:
            html.append("<p>No IPFS operations found for this period.</p>")
            
        # Hardware detection
        html.append("<h2>Hardware Detection</h2>")
        if hardware:
            html.append("<table>")
            html.append("    <tr><th>Hardware</th><th>Detection Count</th><th>Availability Rate</th></tr>")
            
            for row in hardware:
                hw_type, count, available = row
                avail_rate = (available / count * 100) if count > 0 else 0
                
                html.append(f"    <tr><td>{hw_type}</td><td>{count}</td><td>{avail_rate:.1f}%</td></tr>")
                
            html.append("</table>")
        else:
            html.append("<p>No hardware detection results found for this period.</p>")
            
        html.append("</body>")
        html.append("</html>")
        
        return "\n".join(html)
        
    def _format_json_report(self, accel_results, ipfs_ops, hardware, days):
        """Format report as JSON."""
        # Convert results to JSON-friendly format
        accel_json = []
        for row in accel_results:
            model, endpoint, accel_type, total, success, avg_time = row
            success_rate = (success / total * 100) if total > 0 else 0
            
            accel_json.append({
                "model": model,
                "endpoint": endpoint,
                "acceleration_type": accel_type,
                "total_runs": total,
                "successful_runs": success,
                "success_rate": success_rate,
                "avg_time_ms": avg_time
            })
            
        ops_json = []
        for row in ipfs_ops:
            op_type, total, success, avg_time, avg_size = row
            success_rate = (success / total * 100) if total > 0 else 0
            
            ops_json.append({
                "operation_type": op_type,
                "total_operations": total,
                "successful_operations": success,
                "success_rate": success_rate,
                "avg_time_ms": avg_time,
                "avg_file_size": avg_size
            })
            
        hardware_json = []
        for row in hardware:
            hw_type, count, available = row
            avail_rate = (available / count * 100) if count > 0 else 0
            
            hardware_json.append({
                "hardware_type": hw_type,
                "detection_count": count,
                "available_count": available,
                "availability_rate": avail_rate
            })
            
        # Create the report structure
        report = {
            "report_type": "IPFS Accelerate Report",
            "generated_at": datetime.now().isoformat(),
            "period_days": days,
            "acceleration_results": accel_json,
            "ipfs_operations": ops_json,
            "hardware_detection": hardware_json
        }
        
        return json.dumps(report, indent=2)

# Create global database handler instance 
db_handler = DatabaseHandler() if ENABLE_DB_STORAGE else None