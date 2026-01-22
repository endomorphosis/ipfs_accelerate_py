#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Android Test Harness Database Integration

This module provides database integration for the Android Test Harness,
enabling storage and retrieval of benchmark results and thermal analysis data.
It integrates with the existing DuckDB benchmark database system used throughout
the IPFS Accelerate Python Framework.

Features:
    - Schema creation for Android-specific benchmark tables
    - Result storage for benchmark and thermal analysis data
    - Query utilities for retrieving and analyzing results
    - Comparison tools for cross-platform analysis
    - Integration with the mobile edge metrics system

Date: April 2025
"""

import os
import sys
import json
import logging
import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Local imports
try:
    from duckdb_api.core.benchmark_db_api import BenchmarkDBAPI, get_db_connection
    DUCKDB_AVAILABLE = True
except ImportError:
    logger.warning("Could not import benchmark_db_api. Database functionality will be limited.")
    DUCKDB_AVAILABLE = False


class AndroidDatabaseSchema:
    """
    Manages database schema for Android test harness.
    
    Creates and maintains database tables for storing Android benchmark
    results and thermal analysis data.
    """
    
    def __init__(self, db_path: str):
        """
        Initialize with database path.
        
        Args:
            db_path: Path to the DuckDB database
        """
        self.db_path = db_path
        
        # Verify that database is available
        if not DUCKDB_AVAILABLE:
            logger.error("DuckDB is not available. Please install it with 'pip install duckdb'")
        
        # Initialize database connection
        try:
            self.connection = get_db_connection(db_path)
            logger.info(f"Connected to database: {db_path}")
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            self.connection = None
    
    def create_schema(self, overwrite: bool = False) -> bool:
        """
        Create Android benchmark schema in the database.
        
        Args:
            overwrite: Whether to drop and recreate existing tables
            
        Returns:
            Success status
        """
        if not self.connection:
            logger.error("Database connection not available")
            return False
        
        try:
            # Create benchmark results table
            if overwrite:
                self.connection.execute("DROP TABLE IF EXISTS android_benchmark_results")
            
            self.connection.execute("""
            CREATE TABLE IF NOT EXISTS android_benchmark_results (
                id INTEGER PRIMARY KEY,
                model_id INTEGER,
                device_model VARCHAR,
                android_version VARCHAR,
                chipset VARCHAR,
                accelerator VARCHAR,
                batch_size INTEGER,
                thread_count INTEGER,
                average_latency_ms FLOAT,
                p90_latency_ms FLOAT,
                p99_latency_ms FLOAT,
                throughput_items_per_second FLOAT,
                battery_impact_percent FLOAT,
                temperature_max_celsius FLOAT,
                throttling_detected BOOLEAN,
                throttling_duration_seconds FLOAT,
                memory_peak_mb FLOAT,
                execution_time_seconds FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                device_info JSON,
                configuration JSON,
                full_results JSON,
                FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE CASCADE
            )
            """)
            logger.info("Created android_benchmark_results table")
            
            # Create thermal analysis table
            if overwrite:
                self.connection.execute("DROP TABLE IF EXISTS android_thermal_analysis")
            
            self.connection.execute("""
            CREATE TABLE IF NOT EXISTS android_thermal_analysis (
                id INTEGER PRIMARY KEY,
                benchmark_id INTEGER,
                model_id INTEGER,
                device_model VARCHAR,
                total_duration_seconds FLOAT,
                battery_impact_percent_per_hour FLOAT,
                throttling_percentage FLOAT,
                max_temperature_celsius FLOAT,
                temperature_delta_celsius FLOAT,
                impact_score FLOAT,
                recommendations JSON,
                time_series JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (benchmark_id) REFERENCES android_benchmark_results(id) ON DELETE CASCADE,
                FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE CASCADE
            )
            """)
            logger.info("Created android_thermal_analysis table")
            
            # Create model mapping view if it doesn't exist
            self.connection.execute("""
            CREATE VIEW IF NOT EXISTS android_model_mapping AS
            SELECT
                m.id AS model_id,
                m.model_name,
                m.model_family,
                COUNT(ab.id) AS benchmark_count,
                AVG(ab.average_latency_ms) AS avg_latency_ms,
                AVG(ab.throughput_items_per_second) AS avg_throughput,
                AVG(ab.battery_impact_percent) AS avg_battery_impact,
                MAX(ab.created_at) AS last_benchmark
            FROM
                models m
            LEFT JOIN
                android_benchmark_results ab ON m.id = ab.model_id
            GROUP BY
                m.id, m.model_name, m.model_family
            """)
            logger.info("Created android_model_mapping view")
            
            # Create device performance view
            self.connection.execute("""
            CREATE VIEW IF NOT EXISTS android_device_performance AS
            SELECT
                ab.device_model,
                ab.chipset,
                ab.accelerator,
                COUNT(ab.id) AS benchmark_count,
                AVG(ab.average_latency_ms) AS avg_latency_ms,
                AVG(ab.throughput_items_per_second) AS avg_throughput,
                AVG(ab.battery_impact_percent) AS avg_battery_impact,
                AVG(CASE WHEN ab.throttling_detected THEN 1 ELSE 0 END) AS throttling_frequency,
                MAX(ab.created_at) AS last_benchmark
            FROM
                android_benchmark_results ab
            GROUP BY
                ab.device_model, ab.chipset, ab.accelerator
            """)
            logger.info("Created android_device_performance view")
            
            # Create cross-platform comparison view
            self.connection.execute("""
            CREATE VIEW IF NOT EXISTS cross_platform_model_performance AS
            SELECT
                m.id AS model_id,
                m.model_name,
                m.model_family,
                
                -- Android performance
                (SELECT AVG(throughput_items_per_second) 
                 FROM android_benchmark_results 
                 WHERE model_id = m.id) AS android_throughput,
                
                (SELECT AVG(average_latency_ms) 
                 FROM android_benchmark_results 
                 WHERE model_id = m.id) AS android_latency_ms,
                
                -- Desktop performance
                (SELECT AVG(throughput_items_per_second) 
                 FROM performance_results 
                 WHERE model_id = m.id) AS desktop_throughput,
                
                (SELECT AVG(average_latency_ms) 
                 FROM performance_results 
                 WHERE model_id = m.id) AS desktop_latency_ms,
                
                -- Performance ratios (desktop to mobile)
                (SELECT 
                    CASE 
                        WHEN AVG(ar.throughput_items_per_second) > 0 AND AVG(pr.throughput_items_per_second) > 0
                        THEN AVG(pr.throughput_items_per_second) / AVG(ar.throughput_items_per_second)
                        ELSE NULL
                    END
                 FROM android_benchmark_results ar, performance_results pr
                 WHERE ar.model_id = m.id AND pr.model_id = m.id) AS throughput_ratio,
                
                (SELECT 
                    CASE 
                        WHEN AVG(ar.average_latency_ms) > 0 AND AVG(pr.average_latency_ms) > 0
                        THEN AVG(ar.average_latency_ms) / AVG(pr.average_latency_ms)
                        ELSE NULL
                    END
                 FROM android_benchmark_results ar, performance_results pr
                 WHERE ar.model_id = m.id AND pr.model_id = m.id) AS latency_ratio
                
            FROM
                models m
            """)
            logger.info("Created cross_platform_model_performance view")
            
            # Add indices for performance
            self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_android_benchmark_model_id 
            ON android_benchmark_results(model_id)
            """)
            
            self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_android_benchmark_device_model 
            ON android_benchmark_results(device_model)
            """)
            
            self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_android_thermal_model_id 
            ON android_thermal_analysis(model_id)
            """)
            
            logger.info("Created indices for Android tables")
            
            logger.info("Android database schema created successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error creating Android database schema: {e}")
            return False
    
    def verify_schema(self) -> bool:
        """
        Verify that the Android benchmark schema exists.
        
        Returns:
            True if all tables and views exist
        """
        if not self.connection:
            logger.error("Database connection not available")
            return False
        
        try:
            # Check if tables exist
            tables = [
                "android_benchmark_results",
                "android_thermal_analysis"
            ]
            
            views = [
                "android_model_mapping",
                "android_device_performance",
                "cross_platform_model_performance"
            ]
            
            missing_tables = []
            for table in tables:
                result = self.connection.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'").fetchone()
                if not result:
                    missing_tables.append(table)
                else:
                    logger.info(f"Table exists: {table}")
            
            missing_views = []
            for view in views:
                result = self.connection.execute(f"SELECT name FROM sqlite_master WHERE type='view' AND name='{view}'").fetchone()
                if not result:
                    missing_views.append(view)
                else:
                    logger.info(f"View exists: {view}")
            
            if missing_tables or missing_views:
                if missing_tables:
                    logger.error(f"Missing tables: {', '.join(missing_tables)}")
                if missing_views:
                    logger.error(f"Missing views: {', '.join(missing_views)}")
                return False
            
            logger.info("All Android database tables and views exist")
            return True
        
        except Exception as e:
            logger.error(f"Error verifying Android database schema: {e}")
            return False


class AndroidDatabaseAPI:
    """
    API for interacting with the Android benchmark database.
    
    Provides methods for storing and retrieving Android benchmark results
    and thermal analysis data.
    """
    
    def __init__(self, db_path: str):
        """
        Initialize with database path.
        
        Args:
            db_path: Path to the DuckDB database
        """
        self.db_path = db_path
        
        # Initialize database connection
        try:
            if DUCKDB_AVAILABLE:
                self.connection = get_db_connection(db_path)
                logger.info(f"Connected to database: {db_path}")
            else:
                logger.error("DuckDB is not available. Please install it with 'pip install duckdb'")
                self.connection = None
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            self.connection = None
        
        # Initialize schema
        self.schema = AndroidDatabaseSchema(db_path)
    
    def ensure_schema_exists(self) -> bool:
        """
        Ensure that the Android database schema exists.
        
        Returns:
            Success status
        """
        if not self.connection:
            return False
        
        # Verify schema, create if needed
        if not self.schema.verify_schema():
            logger.info("Creating Android database schema")
            return self.schema.create_schema()
        
        return True
    
    def get_model_id(self, model_name: str, model_family: Optional[str] = None) -> Optional[int]:
        """
        Get or create a model ID for a given model name.
        
        Args:
            model_name: Name of the model
            model_family: Optional model family
            
        Returns:
            Model ID or None if not found and cannot be created
        """
        if not self.connection:
            return None
        
        try:
            # Try to find existing model
            query = "SELECT id FROM models WHERE model_name = ?"
            result = self.connection.execute(query, [model_name]).fetchone()
            
            if result:
                model_id = result[0]
                logger.debug(f"Found existing model ID {model_id} for {model_name}")
                return model_id
            
            # If not found, create a new model
            logger.info(f"Creating new model entry for {model_name}")
            
            # Determine model family if not provided
            if not model_family:
                model_family = self._determine_model_family(model_name)
            
            # Insert new model
            query = """
            INSERT INTO models (model_name, model_family, created_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            RETURNING id
            """
            result = self.connection.execute(query, [model_name, model_family]).fetchone()
            
            if result:
                model_id = result[0]
                logger.info(f"Created new model with ID {model_id}")
                return model_id
            
            logger.error(f"Failed to create model ID for {model_name}")
            return None
        
        except Exception as e:
            logger.error(f"Error getting model ID: {e}")
            return None
    
    def _determine_model_family(self, model_name: str) -> str:
        """
        Determine model family from model name.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model family
        """
        # Simple heuristic to determine model family
        model_name_lower = model_name.lower()
        
        if "bert" in model_name_lower:
            return "bert"
        elif "t5" in model_name_lower:
            return "t5"
        elif "gpt" in model_name_lower:
            return "gpt"
        elif "llama" in model_name_lower:
            return "llama"
        elif "clip" in model_name_lower:
            return "clip"
        elif "vit" in model_name_lower:
            return "vit"
        elif "resnet" in model_name_lower:
            return "resnet"
        elif "whisper" in model_name_lower:
            return "whisper"
        else:
            # Try to extract family from name format
            parts = model_name_lower.split("-")
            if len(parts) > 1:
                return parts[0]
            
            return "unknown"
    
    def store_benchmark_result(self, result: Dict[str, Any]) -> Optional[int]:
        """
        Store an Android benchmark result in the database.
        
        Args:
            result: Benchmark result dictionary
            
        Returns:
            Benchmark ID or None if failed
        """
        if not self.connection:
            return None
        
        # Ensure schema exists
        if not self.ensure_schema_exists():
            return None
        
        try:
            # Get model ID
            model_name = result.get("model_name", "unknown")
            model_id = self.get_model_id(model_name)
            
            if not model_id:
                logger.error(f"Failed to get or create model ID for {model_name}")
                return None
            
            # Extract key benchmark data
            device_info = result.get("device_info", {})
            device_model = device_info.get("model", "unknown")
            android_version = device_info.get("android_version", "unknown")
            chipset = device_info.get("chipset", "unknown")
            
            # Get best configuration (highest throughput)
            best_config = None
            best_throughput = 0
            
            for config in result.get("configurations", []):
                throughput = config.get("throughput_items_per_second", 0)
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_config = config
            
            if not best_config:
                logger.error("No configuration data found in benchmark result")
                return None
            
            # Extract configuration data
            config_info = best_config.get("configuration", {})
            accelerator = config_info.get("accelerator", "auto")
            batch_size = config_info.get("batch_size", 1)
            thread_count = config_info.get("threads", 4)
            
            # Extract performance metrics
            latency_ms = best_config.get("latency_ms", {})
            avg_latency = latency_ms.get("mean", 0)
            p90_latency = latency_ms.get("p90", 0)
            p99_latency = latency_ms.get("p99", 0)
            throughput = best_config.get("throughput_items_per_second", 0)
            
            # Extract thermal and battery metrics
            battery_metrics = best_config.get("battery_metrics", {})
            battery_impact = battery_metrics.get("impact_percentage", 0)
            
            thermal_metrics = best_config.get("thermal_metrics", {})
            thermal_delta = thermal_metrics.get("delta", {})
            max_temp_delta = max(thermal_delta.values()) if thermal_delta else 0
            max_temp = max(thermal_metrics.get("post", {}).values()) if thermal_metrics.get("post") else 0
            
            # Extract throttling metrics
            throttling_detected = best_config.get("throttling_detected", False)
            throttling_duration = best_config.get("throttling_duration_seconds", 0)
            
            # Extract memory metrics
            memory_metrics = best_config.get("memory_metrics", {})
            memory_peak = memory_metrics.get("peak_mb", 0)
            
            # Extract execution time
            execution_time = best_config.get("execution_time_seconds", 0)
            
            # Insert benchmark result
            query = """
            INSERT INTO android_benchmark_results (
                model_id, device_model, android_version, chipset, accelerator, batch_size, thread_count,
                average_latency_ms, p90_latency_ms, p99_latency_ms, throughput_items_per_second,
                battery_impact_percent, temperature_max_celsius, throttling_detected,
                throttling_duration_seconds, memory_peak_mb, execution_time_seconds,
                created_at, device_info, configuration, full_results
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?, ?
            )
            RETURNING id
            """
            
            # Convert dictionaries to JSON strings
            device_info_json = json.dumps(device_info)
            config_json = json.dumps(config_info)
            full_results_json = json.dumps(result)
            
            result = self.connection.execute(
                query,
                [
                    model_id, device_model, android_version, chipset, accelerator, batch_size, thread_count,
                    avg_latency, p90_latency, p99_latency, throughput,
                    battery_impact, max_temp, throttling_detected,
                    throttling_duration, memory_peak, execution_time,
                    device_info_json, config_json, full_results_json
                ]
            ).fetchone()
            
            if result:
                benchmark_id = result[0]
                logger.info(f"Stored Android benchmark result with ID {benchmark_id}")
                return benchmark_id
            
            logger.error("Failed to store Android benchmark result")
            return None
        
        except Exception as e:
            logger.error(f"Error storing Android benchmark result: {e}")
            return None
    
    def store_thermal_analysis(self, analysis: Dict[str, Any], benchmark_id: Optional[int] = None) -> Optional[int]:
        """
        Store a thermal analysis result in the database.
        
        Args:
            analysis: Thermal analysis result dictionary
            benchmark_id: Optional benchmark ID to link with
            
        Returns:
            Analysis ID or None if failed
        """
        if not self.connection:
            return None
        
        # Ensure schema exists
        if not self.ensure_schema_exists():
            return None
        
        try:
            # Get model ID
            model_name = analysis.get("model_name", "unknown")
            model_id = self.get_model_id(model_name)
            
            if not model_id:
                logger.error(f"Failed to get or create model ID for {model_name}")
                return None
            
            # Extract key thermal data
            device_info = analysis.get("device_info", {})
            device_model = device_info.get("model", "unknown")
            
            # Extract duration
            duration_seconds = analysis.get("duration_seconds", 0)
            
            # Extract impact analysis
            impact_analysis = analysis.get("impact_analysis", {})
            battery_impact = impact_analysis.get("battery_impact", {}).get("percent_per_hour", 0)
            throttling_percentage = impact_analysis.get("throttling_percentage", 0)
            impact_score = impact_analysis.get("overall_impact_score", 0)
            
            # Extract temperature data
            final_temps = analysis.get("final", {}).get("temperatures", {})
            baseline_temps = analysis.get("baseline", {}).get("temperatures", {})
            max_temp = max(final_temps.values()) if final_temps else 0
            
            temp_deltas = {}
            for zone in final_temps:
                if zone in baseline_temps:
                    temp_deltas[zone] = final_temps[zone] - baseline_temps[zone]
            
            max_temp_delta = max(temp_deltas.values()) if temp_deltas else 0
            
            # Extract recommendations
            recommendations = analysis.get("recommendations", [])
            
            # Extract time series
            time_series = analysis.get("time_series", [])
            
            # Insert thermal analysis
            query = """
            INSERT INTO android_thermal_analysis (
                benchmark_id, model_id, device_model, total_duration_seconds,
                battery_impact_percent_per_hour, throttling_percentage,
                max_temperature_celsius, temperature_delta_celsius,
                impact_score, recommendations, time_series,
                created_at
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP
            )
            RETURNING id
            """
            
            # Convert arrays and objects to JSON strings
            recommendations_json = json.dumps(recommendations)
            time_series_json = json.dumps(time_series)
            
            result = self.connection.execute(
                query,
                [
                    benchmark_id, model_id, device_model, duration_seconds,
                    battery_impact, throttling_percentage,
                    max_temp, max_temp_delta,
                    impact_score, recommendations_json, time_series_json
                ]
            ).fetchone()
            
            if result:
                analysis_id = result[0]
                logger.info(f"Stored thermal analysis with ID {analysis_id}")
                return analysis_id
            
            logger.error("Failed to store thermal analysis")
            return None
        
        except Exception as e:
            logger.error(f"Error storing thermal analysis: {e}")
            return None
    
    def get_benchmark_results(self, 
                             model_name: Optional[str] = None,
                             device_model: Optional[str] = None,
                             limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get benchmark results from the database.
        
        Args:
            model_name: Optional model name to filter by
            device_model: Optional device model to filter by
            limit: Maximum number of results to return
            
        Returns:
            List of benchmark result dictionaries
        """
        if not self.connection:
            return []
        
        try:
            # Build query
            query = """
            SELECT
                ab.id,
                m.model_name,
                m.model_family,
                ab.device_model,
                ab.android_version,
                ab.chipset,
                ab.accelerator,
                ab.batch_size,
                ab.thread_count,
                ab.average_latency_ms,
                ab.throughput_items_per_second,
                ab.battery_impact_percent,
                ab.temperature_max_celsius,
                ab.throttling_detected,
                ab.throttling_duration_seconds,
                ab.memory_peak_mb,
                ab.execution_time_seconds,
                ab.created_at
            FROM
                android_benchmark_results ab
            JOIN
                models m ON ab.model_id = m.id
            WHERE 1=1
            """
            
            params = []
            
            if model_name:
                query += " AND m.model_name = ?"
                params.append(model_name)
            
            if device_model:
                query += " AND ab.device_model = ?"
                params.append(device_model)
            
            query += " ORDER BY ab.created_at DESC LIMIT ?"
            params.append(limit)
            
            # Execute query
            rows = self.connection.execute(query, params).fetchall()
            
            # Convert to dictionaries
            results = []
            for row in rows:
                result = {
                    "id": row[0],
                    "model_name": row[1],
                    "model_family": row[2],
                    "device_model": row[3],
                    "android_version": row[4],
                    "chipset": row[5],
                    "accelerator": row[6],
                    "batch_size": row[7],
                    "thread_count": row[8],
                    "average_latency_ms": row[9],
                    "throughput_items_per_second": row[10],
                    "battery_impact_percent": row[11],
                    "temperature_max_celsius": row[12],
                    "throttling_detected": row[13],
                    "throttling_duration_seconds": row[14],
                    "memory_peak_mb": row[15],
                    "execution_time_seconds": row[16],
                    "created_at": row[17]
                }
                results.append(result)
            
            return results
        
        except Exception as e:
            logger.error(f"Error retrieving benchmark results: {e}")
            return []
    
    def get_thermal_analyses(self, 
                            model_name: Optional[str] = None,
                            device_model: Optional[str] = None,
                            limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get thermal analyses from the database.
        
        Args:
            model_name: Optional model name to filter by
            device_model: Optional device model to filter by
            limit: Maximum number of results to return
            
        Returns:
            List of thermal analysis dictionaries
        """
        if not self.connection:
            return []
        
        try:
            # Build query
            query = """
            SELECT
                at.id,
                m.model_name,
                m.model_family,
                at.device_model,
                at.total_duration_seconds,
                at.battery_impact_percent_per_hour,
                at.throttling_percentage,
                at.max_temperature_celsius,
                at.temperature_delta_celsius,
                at.impact_score,
                at.recommendations,
                at.created_at
            FROM
                android_thermal_analysis at
            JOIN
                models m ON at.model_id = m.id
            WHERE 1=1
            """
            
            params = []
            
            if model_name:
                query += " AND m.model_name = ?"
                params.append(model_name)
            
            if device_model:
                query += " AND at.device_model = ?"
                params.append(device_model)
            
            query += " ORDER BY at.created_at DESC LIMIT ?"
            params.append(limit)
            
            # Execute query
            rows = self.connection.execute(query, params).fetchall()
            
            # Convert to dictionaries
            results = []
            for row in rows:
                # Parse JSON column
                try:
                    recommendations = json.loads(row[10])
                except (json.JSONDecodeError, TypeError):
                    recommendations = []
                
                result = {
                    "id": row[0],
                    "model_name": row[1],
                    "model_family": row[2],
                    "device_model": row[3],
                    "total_duration_seconds": row[4],
                    "battery_impact_percent_per_hour": row[5],
                    "throttling_percentage": row[6],
                    "max_temperature_celsius": row[7],
                    "temperature_delta_celsius": row[8],
                    "impact_score": row[9],
                    "recommendations": recommendations,
                    "created_at": row[11]
                }
                results.append(result)
            
            return results
        
        except Exception as e:
            logger.error(f"Error retrieving thermal analyses: {e}")
            return []
    
    def get_cross_platform_comparison(self, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get cross-platform performance comparison for models.
        
        Args:
            model_name: Optional model name to filter by
            
        Returns:
            List of comparison result dictionaries
        """
        if not self.connection:
            return []
        
        try:
            # Build query
            query = """
            SELECT
                cp.model_id,
                cp.model_name,
                cp.model_family,
                cp.android_throughput,
                cp.android_latency_ms,
                cp.desktop_throughput,
                cp.desktop_latency_ms,
                cp.throughput_ratio,
                cp.latency_ratio
            FROM
                cross_platform_model_performance cp
            WHERE 
                cp.android_throughput IS NOT NULL AND
                cp.desktop_throughput IS NOT NULL
            """
            
            params = []
            
            if model_name:
                query += " AND cp.model_name = ?"
                params.append(model_name)
            
            query += " ORDER BY cp.model_name"
            
            # Execute query
            rows = self.connection.execute(query, params).fetchall()
            
            # Convert to dictionaries
            results = []
            for row in rows:
                result = {
                    "model_id": row[0],
                    "model_name": row[1],
                    "model_family": row[2],
                    "android_throughput": row[3],
                    "android_latency_ms": row[4],
                    "desktop_throughput": row[5],
                    "desktop_latency_ms": row[6],
                    "throughput_ratio": row[7],  # desktop / android (higher means desktop is faster)
                    "latency_ratio": row[8]      # android / desktop (higher means android is slower)
                }
                results.append(result)
            
            return results
        
        except Exception as e:
            logger.error(f"Error retrieving cross-platform comparison: {e}")
            return []
    
    def get_device_performance_summary(self) -> List[Dict[str, Any]]:
        """
        Get a performance summary for all Android devices.
        
        Returns:
            List of device performance summary dictionaries
        """
        if not self.connection:
            return []
        
        try:
            # Build query
            query = """
            SELECT
                device_model,
                chipset,
                accelerator,
                benchmark_count,
                avg_latency_ms,
                avg_throughput,
                avg_battery_impact,
                throttling_frequency,
                last_benchmark
            FROM
                android_device_performance
            ORDER BY
                avg_throughput DESC
            """
            
            # Execute query
            rows = self.connection.execute(query).fetchall()
            
            # Convert to dictionaries
            results = []
            for row in rows:
                result = {
                    "device_model": row[0],
                    "chipset": row[1],
                    "accelerator": row[2],
                    "benchmark_count": row[3],
                    "avg_latency_ms": row[4],
                    "avg_throughput": row[5],
                    "avg_battery_impact": row[6],
                    "throttling_frequency": row[7],  # 0-1 scale
                    "last_benchmark": row[8]
                }
                results.append(result)
            
            return results
        
        except Exception as e:
            logger.error(f"Error retrieving device performance summary: {e}")
            return []
    
    def get_model_summary(self) -> List[Dict[str, Any]]:
        """
        Get a summary of all models with Android benchmarks.
        
        Returns:
            List of model summary dictionaries
        """
        if not self.connection:
            return []
        
        try:
            # Build query
            query = """
            SELECT
                model_id,
                model_name,
                model_family,
                benchmark_count,
                avg_latency_ms,
                avg_throughput,
                avg_battery_impact,
                last_benchmark
            FROM
                android_model_mapping
            WHERE
                benchmark_count > 0
            ORDER BY
                model_name
            """
            
            # Execute query
            rows = self.connection.execute(query).fetchall()
            
            # Convert to dictionaries
            results = []
            for row in rows:
                result = {
                    "model_id": row[0],
                    "model_name": row[1],
                    "model_family": row[2],
                    "benchmark_count": row[3],
                    "avg_latency_ms": row[4],
                    "avg_throughput": row[5],
                    "avg_battery_impact": row[6],
                    "last_benchmark": row[7]
                }
                results.append(result)
            
            return results
        
        except Exception as e:
            logger.error(f"Error retrieving model summary: {e}")
            return []


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Android Database Integration")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Create schema command
    create_parser = subparsers.add_parser("create-schema", help="Create Android database schema")
    create_parser.add_argument("--db-path", required=True, help="Path to DuckDB database")
    create_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing tables")
    
    # Verify schema command
    verify_parser = subparsers.add_parser("verify-schema", help="Verify Android database schema")
    verify_parser.add_argument("--db-path", required=True, help="Path to DuckDB database")
    
    # Store benchmark command
    store_parser = subparsers.add_parser("store-benchmark", help="Store Android benchmark result")
    store_parser.add_argument("--db-path", required=True, help="Path to DuckDB database")
    store_parser.add_argument("--result-file", required=True, help="Path to benchmark result JSON file")
    
    # Store thermal analysis command
    store_thermal_parser = subparsers.add_parser("store-thermal", help="Store thermal analysis result")
    store_thermal_parser.add_argument("--db-path", required=True, help="Path to DuckDB database")
    store_thermal_parser.add_argument("--analysis-file", required=True, help="Path to thermal analysis JSON file")
    store_thermal_parser.add_argument("--benchmark-id", type=int, help="Optional benchmark ID to link with")
    
    # Get benchmark results command
    get_benchmark_parser = subparsers.add_parser("get-benchmarks", help="Get Android benchmark results")
    get_benchmark_parser.add_argument("--db-path", required=True, help="Path to DuckDB database")
    get_benchmark_parser.add_argument("--model", help="Filter by model name")
    get_benchmark_parser.add_argument("--device", help="Filter by device model")
    get_benchmark_parser.add_argument("--limit", type=int, default=10, help="Maximum number of results")
    get_benchmark_parser.add_argument("--output", help="Path to output JSON file")
    
    # Get cross-platform comparison command
    get_comparison_parser = subparsers.add_parser("get-comparison", help="Get cross-platform performance comparison")
    get_comparison_parser.add_argument("--db-path", required=True, help="Path to DuckDB database")
    get_comparison_parser.add_argument("--model", help="Filter by model name")
    get_comparison_parser.add_argument("--output", help="Path to output JSON file")
    
    # Get device summary command
    get_device_parser = subparsers.add_parser("get-devices", help="Get Android device performance summary")
    get_device_parser.add_argument("--db-path", required=True, help="Path to DuckDB database")
    get_device_parser.add_argument("--output", help="Path to output JSON file")
    
    # Get model summary command
    get_model_parser = subparsers.add_parser("get-models", help="Get Android model summary")
    get_model_parser.add_argument("--db-path", required=True, help="Path to DuckDB database")
    get_model_parser.add_argument("--output", help="Path to output JSON file")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == "create-schema":
            schema = AndroidDatabaseSchema(args.db_path)
            success = schema.create_schema(args.overwrite)
            if success:
                print("Android database schema created successfully")
                return 0
            else:
                print("Failed to create Android database schema")
                return 1
        
        elif args.command == "verify-schema":
            schema = AndroidDatabaseSchema(args.db_path)
            success = schema.verify_schema()
            if success:
                print("Android database schema verified successfully")
                return 0
            else:
                print("Android database schema verification failed")
                return 1
        
        elif args.command == "store-benchmark":
            with open(args.result_file, "r") as f:
                result = json.load(f)
            
            api = AndroidDatabaseAPI(args.db_path)
            benchmark_id = api.store_benchmark_result(result)
            
            if benchmark_id:
                print(f"Benchmark stored with ID: {benchmark_id}")
                return 0
            else:
                print("Failed to store benchmark")
                return 1
        
        elif args.command == "store-thermal":
            with open(args.analysis_file, "r") as f:
                analysis = json.load(f)
            
            api = AndroidDatabaseAPI(args.db_path)
            analysis_id = api.store_thermal_analysis(analysis, args.benchmark_id)
            
            if analysis_id:
                print(f"Thermal analysis stored with ID: {analysis_id}")
                return 0
            else:
                print("Failed to store thermal analysis")
                return 1
        
        elif args.command == "get-benchmarks":
            api = AndroidDatabaseAPI(args.db_path)
            results = api.get_benchmark_results(args.model, args.device, args.limit)
            
            if args.output:
                with open(args.output, "w") as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"Benchmark results saved to: {args.output}")
            else:
                for result in results:
                    print(f"ID: {result['id']}")
                    print(f"Model: {result['model_name']}")
                    print(f"Device: {result['device_model']}")
                    print(f"Latency: {result['average_latency_ms']:.2f} ms")
                    print(f"Throughput: {result['throughput_items_per_second']:.2f} items/s")
                    print(f"Date: {result['created_at']}")
                    print()
            
            return 0
        
        elif args.command == "get-comparison":
            api = AndroidDatabaseAPI(args.db_path)
            results = api.get_cross_platform_comparison(args.model)
            
            if args.output:
                with open(args.output, "w") as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"Comparison results saved to: {args.output}")
            else:
                print("Cross-Platform Performance Comparison:")
                print("--------------------------------------")
                for result in results:
                    print(f"Model: {result['model_name']}")
                    print(f"  Android: {result['android_throughput']:.2f} items/s, {result['android_latency_ms']:.2f} ms")
                    print(f"  Desktop: {result['desktop_throughput']:.2f} items/s, {result['desktop_latency_ms']:.2f} ms")
                    print(f"  Desktop/Android Throughput Ratio: {result['throughput_ratio']:.2f}x")
                    print(f"  Android/Desktop Latency Ratio: {result['latency_ratio']:.2f}x")
                    print()
            
            return 0
        
        elif args.command == "get-devices":
            api = AndroidDatabaseAPI(args.db_path)
            results = api.get_device_performance_summary()
            
            if args.output:
                with open(args.output, "w") as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"Device summary saved to: {args.output}")
            else:
                print("Android Device Performance Summary:")
                print("-----------------------------------")
                for result in results:
                    print(f"Device: {result['device_model']}")
                    print(f"  Chipset: {result['chipset']}")
                    print(f"  Accelerator: {result['accelerator']}")
                    print(f"  Benchmarks: {result['benchmark_count']}")
                    print(f"  Avg Throughput: {result['avg_throughput']:.2f} items/s")
                    print(f"  Avg Latency: {result['avg_latency_ms']:.2f} ms")
                    print(f"  Avg Battery Impact: {result['avg_battery_impact']:.2f}%")
                    print(f"  Throttling Frequency: {result['throttling_frequency']*100:.1f}%")
                    print()
            
            return 0
        
        elif args.command == "get-models":
            api = AndroidDatabaseAPI(args.db_path)
            results = api.get_model_summary()
            
            if args.output:
                with open(args.output, "w") as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"Model summary saved to: {args.output}")
            else:
                print("Android Model Summary:")
                print("---------------------")
                for result in results:
                    print(f"Model: {result['model_name']} ({result['model_family']})")
                    print(f"  Benchmarks: {result['benchmark_count']}")
                    print(f"  Avg Throughput: {result['avg_throughput']:.2f} items/s")
                    print(f"  Avg Latency: {result['avg_latency_ms']:.2f} ms")
                    print(f"  Avg Battery Impact: {result['avg_battery_impact']:.2f}%")
                    print()
            
            return 0
        
        else:
            parser.print_help()
            return 1
    
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())