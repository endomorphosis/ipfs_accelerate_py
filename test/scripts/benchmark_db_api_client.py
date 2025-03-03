#!/usr/bin/env python
"""
Benchmark Database API Client

This module provides a Python client for the Benchmark Database API.
"""

import os
import sys
import json
import logging
import datetime
import requests
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import duckdb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("benchmark_db_client")

class BenchmarkDBAPI:
    """Client API for the benchmark database"""
    
    def __init__(self, database_path="./benchmark_db.duckdb", server_url=None):
        """
        Initialize the database API.
        
        Args:
            database_path: Path to the DuckDB database file
            server_url: URL of the database API server (if using server mode)
        """
        self.database_path = database_path
        self.server_url = server_url
        self.conn = None
        
        # If using direct database access, create required tables if they don't exist
        if server_url is None:
            # Ensure the database directory exists
            os.makedirs(os.path.dirname(os.path.abspath(database_path)), exist_ok=True)
            self.ensure_database_initialized()
    
    def get_connection(self):
        """Get a connection to the database"""
        if self.conn is None:
            try:
                # Check if database file exists, create it if it doesn't
                db_path = Path(self.database_path)
                if not db_path.exists():
                    # Create parent directory if it doesn't exist
                    db_path.parent.mkdir(parents=True, exist_ok=True)
                
                self.conn = duckdb.connect(self.database_path)
                
                # Initialize tables if they don't exist
                self.ensure_database_initialized()
            except Exception as e:
                logger.error(f"Error connecting to database: {e}")
                raise
        
        return self.conn
    
    def close(self):
        """Close the database connection"""
        if self.conn is not None:
            self.conn.close()
            self.conn = None
    
    def ensure_database_initialized(self):
        """Initialize the database if needed"""
        self.ensure_web_platform_table_exists()
    
    def ensure_web_platform_table_exists(self):
        """Ensure the web_platform_results table exists"""
        conn = self.get_connection()
        
        try:
            # Check if table exists
            table_exists = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='web_platform_results'
            """).fetchone()
            
            if not table_exists:
                # Create table if it doesn't exist
                conn.execute("""
                CREATE TABLE web_platform_results (
                    result_id INTEGER PRIMARY KEY,
                    model_name VARCHAR,
                    model_type VARCHAR,
                    browser VARCHAR,
                    platform VARCHAR,
                    status VARCHAR,
                    execution_time DOUBLE,
                    metrics JSON,
                    error_message VARCHAR,
                    source_file VARCHAR,
                    timestamp TIMESTAMP
                )
                """)
                logger.info("Created web_platform_results table")
                
                # Create indices
                conn.execute("CREATE INDEX idx_wpr_model_type ON web_platform_results(model_type)")
                conn.execute("CREATE INDEX idx_wpr_browser ON web_platform_results(browser)")
                conn.execute("CREATE INDEX idx_wpr_platform ON web_platform_results(platform)")
                conn.execute("CREATE INDEX idx_wpr_timestamp ON web_platform_results(timestamp)")
                
                return True
            
            return False
        except Exception as e:
            logger.error(f"Error ensuring web_platform_results table exists: {e}")
            raise
    
    def store_web_platform_result(self, 
                                model_name: str,
                                model_type: str,
                                browser: str,
                                platform: str,
                                status: str,
                                metrics: Optional[Dict[str, Any]] = None,
                                execution_time: Optional[float] = None,
                                error_message: Optional[str] = None,
                                source_file: Optional[str] = None,
                                timestamp: Optional[datetime.datetime] = None) -> int:
        """
        Store a web platform test result in the database.
        
        Args:
            model_name: Name of the model tested
            model_type: Type of model (whisper, wav2vec2, clap, etc.)
            browser: Browser used for testing (chrome, firefox, safari, edge)
            platform: Web platform tested (webnn, webgpu)
            status: Test status (successful, failed, etc.)
            metrics: Dictionary of test metrics (optional)
            execution_time: Execution time in seconds (optional)
            error_message: Error message if test failed (optional)
            source_file: Source file containing the test results (optional)
            timestamp: Timestamp for the test result (optional, defaults to now)
            
        Returns:
            ID of the stored result
        """
        # Use REST API if server URL is provided
        if self.server_url is not None:
            try:
                response = requests.post(
                    f"{self.server_url}/api/web-platform",
                    json={
                        "model_name": model_name,
                        "model_type": model_type,
                        "browser": browser,
                        "platform": platform,
                        "status": status,
                        "metrics": metrics,
                        "execution_time": execution_time,
                        "error_message": error_message,
                        "source_file": source_file,
                        "timestamp": timestamp.isoformat() if timestamp else datetime.datetime.now().isoformat()
                    }
                )
                response.raise_for_status()
                return response.json()["result_id"]
            except Exception as e:
                logger.error(f"Error storing web platform result via API: {e}")
                raise
        
        # Use direct database access
        conn = self.get_connection()
        
        # Ensure required table exists
        self.ensure_web_platform_table_exists()
        
        try:
            # Set default timestamp if not provided
            if timestamp is None:
                timestamp = datetime.datetime.now()
            
            # Convert metrics to JSON string if provided
            metrics_json = None
            if metrics is not None:
                metrics_json = json.dumps(metrics)
            
            # Insert into database
            result = conn.execute("""
                INSERT INTO web_platform_results (
                    model_name, model_type, browser, platform, status, 
                    execution_time, metrics, error_message, source_file, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                RETURNING result_id
            """, (
                model_name, model_type, browser, platform, status,
                execution_time, metrics_json, error_message, source_file, timestamp
            )).fetchone()
            
            # Get the inserted ID
            result_id = result[0]
            
            logger.debug(f"Stored web platform result with ID: {result_id}")
            return result_id
        except Exception as e:
            logger.error(f"Error storing web platform result: {e}")
            raise
    
    def query_web_platform_results(self,
                                 model_name: Optional[str] = None,
                                 model_type: Optional[str] = None,
                                 browser: Optional[str] = None,
                                 platform: Optional[str] = None,
                                 status: Optional[str] = None,
                                 start_date: Optional[str] = None,
                                 end_date: Optional[str] = None,
                                 limit: int = 100) -> List[Dict[str, Any]]:
        """
        Query web platform test results from the database.
        
        Args:
            model_name: Filter by model name (optional)
            model_type: Filter by model type (optional)
            browser: Filter by browser (optional)
            platform: Filter by platform (optional)
            status: Filter by status (optional)
            start_date: Filter by start date (optional)
            end_date: Filter by end date (optional)
            limit: Maximum number of results to return (optional)
            
        Returns:
            List of web platform test results
        """
        # Use REST API if server URL is provided
        if self.server_url is not None:
            try:
                params = {
                    "limit": limit
                }
                
                if model_name:
                    params["model_name"] = model_name
                
                if model_type:
                    params["model_type"] = model_type
                
                if browser:
                    params["browser"] = browser
                
                if platform:
                    params["platform"] = platform
                
                if status:
                    params["status"] = status
                
                if start_date:
                    params["start_date"] = start_date
                
                if end_date:
                    params["end_date"] = end_date
                
                response = requests.get(
                    f"{self.server_url}/api/web-platform",
                    params=params
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                logger.error(f"Error querying web platform results via API: {e}")
                raise
        
        # Use direct database access
        conn = self.get_connection()
        
        # Ensure table exists
        try:
            self.ensure_web_platform_table_exists()
        except Exception:
            # If table doesn't exist, return empty list
            return []
        
        try:
            # Build query with filters
            query = """
            SELECT 
                result_id,
                model_name,
                model_type,
                browser,
                platform,
                status,
                execution_time,
                metrics,
                error_message,
                source_file,
                timestamp
            FROM 
                web_platform_results
            """
            
            params = []
            where_clauses = []
            
            if model_name:
                where_clauses.append("model_name LIKE ?")
                params.append(f"%{model_name}%")
            
            if model_type:
                where_clauses.append("model_type = ?")
                params.append(model_type)
            
            if browser:
                where_clauses.append("browser = ?")
                params.append(browser)
            
            if platform:
                where_clauses.append("platform = ?")
                params.append(platform)
            
            if status:
                where_clauses.append("status = ?")
                params.append(status)
            
            if start_date:
                try:
                    start_dt = datetime.datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                    where_clauses.append("timestamp >= ?")
                    params.append(start_dt)
                except ValueError as e:
                    logger.error(f"Invalid date format for start_date: {e}")
                    raise ValueError(f"Invalid date format for start_date: {start_date}")
            
            if end_date:
                try:
                    end_dt = datetime.datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                    where_clauses.append("timestamp <= ?")
                    params.append(end_dt)
                except ValueError as e:
                    logger.error(f"Invalid date format for end_date: {e}")
                    raise ValueError(f"Invalid date format for end_date: {end_date}")
            
            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            # Execute query
            results = conn.execute(query, params).fetchdf()
            
            # Convert to list of dicts and parse metrics JSON
            results_list = []
            for _, row in results.iterrows():
                result = row.to_dict()
                
                # Parse metrics JSON
                if 'metrics' in result and result['metrics'] is not None:
                    try:
                        result['metrics'] = json.loads(result['metrics'])
                    except json.JSONDecodeError:
                        result['metrics'] = {}
                
                results_list.append(result)
            
            return results_list
        except Exception as e:
            logger.error(f"Error querying web platform results: {e}")
            raise