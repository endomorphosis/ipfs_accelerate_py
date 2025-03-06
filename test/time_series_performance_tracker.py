#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time Series Performance Tracker for IPFS Accelerate Python Framework

This module implements the time-series performance tracking system mentioned in NEXT_STEPS.md.
It provides components for versioned test results, regression detection, trend visualization,
and notification systems.

Date: March 2025
"""

import os
import sys
import json
import time
import datetime
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Union, Optional, Any
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent))

# Local imports
try:
    from benchmark_db_api import BenchmarkDBAPI, get_db_connection
    import benchmark_db_query
except ImportError:
    logger.warning("Warning: Some local modules could not be imported. Functionality may be limited.")


class TimeSeriesSchema:
    """Schema extension for versioned test results in DuckDB database."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize with optional database path."""
        self.db_path = db_path or os.environ.get('BENCHMARK_DB_PATH', './benchmark_db.duckdb')
        
        # Check if DB_API instance is needed
        self.db_api = None
    
    def _get_connection(self):
        """Get a connection to the database."""
        try:
            return get_db_connection(self.db_path)
        except NameError:
            # Fallback if get_db_connection is not available
            import duckdb
            return duckdb.connect(self.db_path)
    
    def create_schema_extensions(self) -> bool:
        """
        Create schema extensions for versioned test results.
        
        Returns:
            bool: Success status
        """
        logger.info("Creating schema extensions for time series data")
        
        conn = self._get_connection()
        
        try:
            # Start a transaction for consistency
            conn.execute("BEGIN TRANSACTION")
            
            # Create version history table if it doesn't exist
            conn.execute("""
            CREATE TABLE IF NOT EXISTS version_history (
                id INTEGER PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                version_tag VARCHAR,
                description VARCHAR,
                user_id VARCHAR,
                commit_hash VARCHAR,
                git_branch VARCHAR
            )
            """)
            
            # Create performance time series table if it doesn't exist
            conn.execute("""
            CREATE TABLE IF NOT EXISTS performance_time_series (
                id INTEGER PRIMARY KEY,
                version_id INTEGER,
                model_id INTEGER,
                hardware_id INTEGER,
                batch_size INTEGER,
                test_type VARCHAR,
                throughput FLOAT,
                latency FLOAT,
                memory_usage FLOAT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (version_id) REFERENCES version_history(id),
                FOREIGN KEY (model_id) REFERENCES models(id),
                FOREIGN KEY (hardware_id) REFERENCES hardware_platforms(id)
            )
            """)
            
            # Create regression alerts table if it doesn't exist
            conn.execute("""
            CREATE TABLE IF NOT EXISTS regression_alerts (
                id INTEGER PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                model_id INTEGER,
                hardware_id INTEGER,
                metric VARCHAR,
                previous_value FLOAT,
                current_value FLOAT,
                percent_change FLOAT,
                severity VARCHAR,
                status VARCHAR DEFAULT 'active',
                notification_sent BOOLEAN DEFAULT FALSE,
                resolved_at TIMESTAMP,
                FOREIGN KEY (model_id) REFERENCES models(id),
                FOREIGN KEY (hardware_id) REFERENCES hardware_platforms(id)
            )
            """)
            
            # Commit the transaction
            conn.execute("COMMIT")
            
            logger.info("Schema extensions created successfully!")
            return True
            
        except Exception as e:
            # Rollback in case of error
            try:
                conn.execute("ROLLBACK")
            except:
                pass
                
            logger.error(f"Error creating schema extensions: {e}")
            return False
            
        finally:
            conn.close()


class VersionManager:
    """Manages version entries for time series tracking."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize with optional database path."""
        self.db_path = db_path or os.environ.get('BENCHMARK_DB_PATH', './benchmark_db.duckdb')
    
    def _get_connection(self):
        """Get a connection to the database."""
        try:
            return get_db_connection(self.db_path)
        except NameError:
            # Fallback if get_db_connection is not available
            import duckdb
            return duckdb.connect(self.db_path)
    
    def create_version(self, 
                      version_tag: str, 
                      description: Optional[str] = None,
                      commit_hash: Optional[str] = None,
                      git_branch: Optional[str] = None,
                      user_id: Optional[str] = None) -> Optional[int]:
        """
        Create a new version entry in the version history.
        
        Args:
            version_tag: Tag for the version (e.g., 'v1.0.0')
            description: Optional description of the version
            commit_hash: Optional Git commit hash
            git_branch: Optional Git branch name
            user_id: Optional user ID
            
        Returns:
            version_id: ID of the created version, or None if creation failed
        """
        conn = self._get_connection()
        
        try:
            # Check if version already exists
            result = conn.execute(
                "SELECT id FROM version_history WHERE version_tag = ?",
                [version_tag]
            ).fetchone()
            
            if result:
                logger.warning(f"Version '{version_tag}' already exists")
                return result[0]
            
            # Get next version ID
            result = conn.execute("SELECT MAX(id) FROM version_history").fetchone()
            next_id = 1 if result[0] is None else result[0] + 1
            
            # Create version entry
            conn.execute(
                """
                INSERT INTO version_history 
                (id, version_tag, description, user_id, commit_hash, git_branch)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [next_id, version_tag, description, user_id, commit_hash, git_branch]
            )
            
            logger.info(f"Created version '{version_tag}' with ID {next_id}")
            return next_id
            
        except Exception as e:
            logger.error(f"Error creating version entry: {e}")
            return None
            
        finally:
            conn.close()
    
    def get_version(self, version_tag: str) -> Optional[Dict[str, Any]]:
        """
        Get a version entry by tag.
        
        Args:
            version_tag: Tag for the version
            
        Returns:
            version: Version information, or None if not found
        """
        conn = self._get_connection()
        
        try:
            result = conn.execute(
                """
                SELECT 
                    id, timestamp, version_tag, description, 
                    user_id, commit_hash, git_branch
                FROM 
                    version_history
                WHERE 
                    version_tag = ?
                """,
                [version_tag]
            ).fetchone()
            
            if not result:
                logger.warning(f"Version '{version_tag}' not found")
                return None
            
            return {
                'id': result[0],
                'timestamp': result[1],
                'version_tag': result[2],
                'description': result[3],
                'user_id': result[4],
                'commit_hash': result[5],
                'git_branch': result[6]
            }
            
        except Exception as e:
            logger.error(f"Error getting version entry: {e}")
            return None
            
        finally:
            conn.close()
    
    def list_versions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        List recent versions in the history.
        
        Args:
            limit: Maximum number of versions to return
            
        Returns:
            versions: List of version information
        """
        conn = self._get_connection()
        
        try:
            results = conn.execute(
                """
                SELECT 
                    id, timestamp, version_tag, description, 
                    user_id, commit_hash, git_branch
                FROM 
                    version_history
                ORDER BY 
                    timestamp DESC
                LIMIT ?
                """,
                [limit]
            ).fetchall()
            
            versions = []
            for result in results:
                versions.append({
                    'id': result[0],
                    'timestamp': result[1],
                    'version_tag': result[2],
                    'description': result[3],
                    'user_id': result[4],
                    'commit_hash': result[5],
                    'git_branch': result[6]
                })
            
            return versions
            
        except Exception as e:
            logger.error(f"Error listing versions: {e}")
            return []
            
        finally:
            conn.close()


class TimeSeriesManager:
    """Manages time series data for performance metrics."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize with optional database path."""
        self.db_path = db_path or os.environ.get('BENCHMARK_DB_PATH', './benchmark_db.duckdb')
        
        # Initialize version manager
        self.version_manager = VersionManager(self.db_path)
    
    def _get_connection(self):
        """Get a connection to the database."""
        try:
            return get_db_connection(self.db_path)
        except NameError:
            # Fallback if get_db_connection is not available
            import duckdb
            return duckdb.connect(self.db_path)
    
    def _get_model_id(self, conn, model_name: str) -> Optional[int]:
        """Get model ID from name."""
        result = conn.execute(
            "SELECT model_id FROM models WHERE model_name = ?",
            [model_name]
        ).fetchone()
        
        if result:
            return result[0]
        
        # Try to add the model if it doesn't exist
        try:
            # Get next model ID
            max_id = conn.execute("SELECT MAX(model_id) FROM models").fetchone()[0]
            model_id = 1 if max_id is None else max_id + 1
            
            # Add model
            conn.execute(
                """
                INSERT INTO models (model_id, model_name)
                VALUES (?, ?)
                """,
                [model_id, model_name]
            )
            
            logger.info(f"Added new model: {model_name} (ID: {model_id})")
            return model_id
            
        except Exception as e:
            logger.error(f"Error adding model: {e}")
            return None
    
    def _get_hardware_id(self, conn, hardware_type: str) -> Optional[int]:
        """Get hardware ID from type."""
        result = conn.execute(
            "SELECT hardware_id FROM hardware_platforms WHERE hardware_type = ?",
            [hardware_type]
        ).fetchone()
        
        if result:
            return result[0]
        
        # Try to add the hardware if it doesn't exist
        try:
            # Get next hardware ID
            max_id = conn.execute("SELECT MAX(hardware_id) FROM hardware_platforms").fetchone()[0]
            hardware_id = 1 if max_id is None else max_id + 1
            
            # Add hardware
            conn.execute(
                """
                INSERT INTO hardware_platforms (hardware_id, hardware_type)
                VALUES (?, ?)
                """,
                [hardware_id, hardware_type]
            )
            
            logger.info(f"Added new hardware: {hardware_type} (ID: {hardware_id})")
            return hardware_id
            
        except Exception as e:
            logger.error(f"Error adding hardware: {e}")
            return None
    
    def record_performance(self, 
                          model_name: str,
                          hardware_type: str,
                          batch_size: int,
                          test_type: str,
                          throughput: float,
                          latency: float,
                          memory_usage: Optional[float] = None,
                          version_tag: Optional[str] = None) -> bool:
        """
        Record a performance metric in the time series.
        
        Args:
            model_name: Name of the model
            hardware_type: Type of hardware
            batch_size: Batch size used
            test_type: Type of test
            throughput: Throughput value (items per second)
            latency: Latency value (milliseconds)
            memory_usage: Memory usage (MB)
            version_tag: Optional version tag
            
        Returns:
            success: True if the metric was recorded successfully
        """
        conn = self._get_connection()
        
        try:
            # Get model ID
            model_id = self._get_model_id(conn, model_name)
            if model_id is None:
                logger.error(f"Could not find or create model: {model_name}")
                return False
            
            # Get hardware ID
            hardware_id = self._get_hardware_id(conn, hardware_type)
            if hardware_id is None:
                logger.error(f"Could not find or create hardware: {hardware_type}")
                return False
            
            # Get version ID if provided
            version_id = None
            if version_tag:
                version = self.version_manager.get_version(version_tag)
                if version:
                    version_id = version['id']
                else:
                    # Create version if it doesn't exist
                    version_id = self.version_manager.create_version(version_tag)
            
            # Get next ID
            result = conn.execute("SELECT MAX(id) FROM performance_time_series").fetchone()
            next_id = 1 if result[0] is None else result[0] + 1
            
            # Record performance metric
            conn.execute(
                """
                INSERT INTO performance_time_series 
                (id, version_id, model_id, hardware_id, batch_size, test_type, 
                 throughput, latency, memory_usage)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [next_id, version_id, model_id, hardware_id, batch_size, test_type, 
                 throughput, latency, memory_usage]
            )
            
            logger.info(f"Recorded performance metric for {model_name} on {hardware_type}")
            return True
            
        except Exception as e:
            logger.error(f"Error recording performance metric: {e}")
            return False
            
        finally:
            conn.close()
    
    def get_performance_history(self, 
                              model_name: str,
                              hardware_type: str,
                              batch_size: Optional[int] = None,
                              test_type: Optional[str] = None,
                              metric: str = 'throughput',
                              days: int = 30) -> pd.DataFrame:
        """
        Get performance history for a model and hardware combination.
        
        Args:
            model_name: Name of the model
            hardware_type: Type of hardware
            batch_size: Optional batch size filter
            test_type: Optional test type filter
            metric: Metric to retrieve ('throughput', 'latency', 'memory_usage')
            days: Number of days of history to retrieve
            
        Returns:
            history: DataFrame with performance history
        """
        conn = self._get_connection()
        
        try:
            # Build query
            query = """
            SELECT 
                pts.timestamp,
                pts.{metric},
                vh.version_tag,
                pts.batch_size,
                pts.test_type
            FROM 
                performance_time_series pts
            JOIN 
                models m ON pts.model_id = m.model_id
            JOIN 
                hardware_platforms hp ON pts.hardware_id = hp.hardware_id
            LEFT JOIN
                version_history vh ON pts.version_id = vh.id
            WHERE 
                m.model_name = ?
                AND hp.hardware_type = ?
                AND pts.timestamp >= ?
            """.format(metric=metric)
            
            params = [model_name, hardware_type, 
                     (datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')]
            
            if batch_size is not None:
                query += " AND pts.batch_size = ?"
                params.append(batch_size)
            
            if test_type is not None:
                query += " AND pts.test_type = ?"
                params.append(test_type)
            
            query += " ORDER BY pts.timestamp"
            
            # Execute query
            result = conn.execute(query, params).fetchall()
            
            # Convert to DataFrame
            df = pd.DataFrame(result, columns=['timestamp', metric, 'version_tag', 'batch_size', 'test_type'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting performance history: {e}")
            return pd.DataFrame()
            
        finally:
            conn.close()


class RegressionDetector:
    """Detects performance regressions in time series data."""
    
    def __init__(self, db_path: Optional[str] = None, threshold: float = 0.10):
        """
        Initialize with database path and regression threshold.
        
        Args:
            db_path: Optional database path
            threshold: Threshold for regression detection (0.1 = 10%)
        """
        self.db_path = db_path or os.environ.get('BENCHMARK_DB_PATH', './benchmark_db.duckdb')
        self.threshold = threshold
    
    def _get_connection(self):
        """Get a connection to the database."""
        try:
            return get_db_connection(self.db_path)
        except NameError:
            # Fallback if get_db_connection is not available
            import duckdb
            return duckdb.connect(self.db_path)
    
    def detect_regressions(self, 
                          model_name: Optional[str] = None,
                          hardware_type: Optional[str] = None,
                          metric: str = 'throughput',
                          window_size: int = 5,
                          min_change: float = 0.05) -> List[Dict[str, Any]]:
        """
        Detect performance regressions for specified model/hardware.
        
        Args:
            model_name: Filter by model name
            hardware_type: Filter by hardware type
            metric: Performance metric to analyze (throughput, latency, memory_usage)
            window_size: Number of previous data points to use for baseline
            min_change: Minimum change percentage to consider (0.05 = 5%)
            
        Returns:
            List of regression events
        """
        conn = self._get_connection()
        
        try:
            # Build base query
            query = """
            SELECT
                m.model_id,
                m.model_name,
                hp.hardware_id,
                hp.hardware_type,
                pts.batch_size,
                pts.{metric},
                pts.timestamp,
                pts.id
            FROM 
                performance_time_series pts
            JOIN 
                models m ON pts.model_id = m.model_id
            JOIN 
                hardware_platforms hp ON pts.hardware_id = hp.hardware_id
            """.format(metric=metric)
            
            # Add filters
            params = []
            if model_name:
                query += " WHERE m.model_name = ?"
                params.append(model_name)
                
                if hardware_type:
                    query += " AND hp.hardware_type = ?"
                    params.append(hardware_type)
            elif hardware_type:
                query += " WHERE hp.hardware_type = ?"
                params.append(hardware_type)
            
            # Add ordering
            query += """
            ORDER BY 
                m.model_name, hp.hardware_type, pts.batch_size, pts.timestamp
            """
            
            # Execute query
            result = conn.execute(query, params).fetchall()
            
            if not result:
                logger.warning("No performance data found for regression detection")
                return []
            
            # Process results by group (model, hardware, batch_size)
            regressions = []
            current_group = None
            group_data = []
            
            for row in result:
                model_id, model_name, hardware_id, hardware_type, batch_size, value, timestamp, pts_id = row
                group_key = (model_name, hardware_type, batch_size)
                
                # Start new group if needed
                if current_group != group_key:
                    # Process previous group if it exists
                    if current_group and len(group_data) > window_size:
                        group_regressions = self._analyze_group(group_data, current_group, metric, window_size, min_change)
                        regressions.extend(group_regressions)
                    
                    # Start new group
                    current_group = group_key
                    group_data = []
                
                # Add data to current group
                group_data.append({
                    'model_id': model_id,
                    'model_name': model_name,
                    'hardware_id': hardware_id,
                    'hardware_type': hardware_type,
                    'batch_size': batch_size,
                    'value': value,
                    'timestamp': timestamp,
                    'pts_id': pts_id
                })
            
            # Process last group
            if current_group and len(group_data) > window_size:
                group_regressions = self._analyze_group(group_data, current_group, metric, window_size, min_change)
                regressions.extend(group_regressions)
            
            return regressions
            
        except Exception as e:
            logger.error(f"Error detecting regressions: {e}")
            return []
            
        finally:
            conn.close()
    
    def _analyze_group(self, 
                      data: List[Dict[str, Any]], 
                      group_key: Tuple[str, str, int],
                      metric: str,
                      window_size: int,
                      min_change: float) -> List[Dict[str, Any]]:
        """
        Analyze a group of data points for regressions.
        
        Args:
            data: List of data points for the group
            group_key: (model_name, hardware_type, batch_size)
            metric: Performance metric being analyzed
            window_size: Window size for baseline calculation
            min_change: Minimum change percentage to consider
            
        Returns:
            List of regression events for this group
        """
        model_name, hardware_type, batch_size = group_key
        regressions = []
        
        # Need at least window_size+1 points to detect regression
        if len(data) <= window_size:
            return []
        
        # Analyze each point after the initial window
        for i in range(window_size, len(data)):
            # Calculate baseline as mean of previous window_size points
            window = [point['value'] for point in data[i-window_size:i]]
            baseline = np.mean(window)
            
            # Get current value
            current = data[i]['value']
            
            # Calculate change percentage based on metric type
            # For latency, lower is better; for others, higher is better
            if metric == 'latency':
                percent_change = (current - baseline) / baseline
                is_regression = percent_change > self.threshold
            else:
                percent_change = (baseline - current) / baseline
                is_regression = percent_change > self.threshold
            
            # Check if this is a regression exceeding the threshold
            if is_regression and abs(percent_change) >= min_change:
                # Determine severity
                if percent_change > 0.25:
                    severity = 'critical'
                elif percent_change > 0.15:
                    severity = 'high'
                else:
                    severity = 'medium'
                
                # Add to regressions list
                regressions.append({
                    'model_id': data[i]['model_id'],
                    'model_name': model_name,
                    'hardware_id': data[i]['hardware_id'],
                    'hardware_type': hardware_type,
                    'batch_size': batch_size,
                    'metric': metric,
                    'previous_value': baseline,
                    'current_value': current,
                    'percent_change': percent_change * 100,
                    'timestamp': data[i]['timestamp'],
                    'severity': severity,
                    'data_point_id': data[i]['pts_id']
                })
        
        return regressions
    
    def record_regressions(self, regressions: List[Dict[str, Any]]) -> int:
        """
        Record detected regressions in the database.
        
        Args:
            regressions: List of regression events
            
        Returns:
            Number of regressions recorded
        """
        if not regressions:
            return 0
        
        conn = self._get_connection()
        recorded = 0
        
        try:
            for regression in regressions:
                # Get next ID
                result = conn.execute("SELECT MAX(id) FROM regression_alerts").fetchone()
                next_id = 1 if result[0] is None else result[0] + 1
                
                # Record regression alert
                conn.execute(
                    """
                    INSERT INTO regression_alerts 
                    (id, model_id, hardware_id, metric, previous_value, current_value, 
                     percent_change, severity, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        next_id,
                        regression['model_id'],
                        regression['hardware_id'],
                        regression['metric'],
                        regression['previous_value'],
                        regression['current_value'],
                        regression['percent_change'],
                        regression['severity'],
                        regression['timestamp']
                    ]
                )
                
                recorded += 1
            
            logger.info(f"Recorded {recorded} regression alerts")
            return recorded
            
        except Exception as e:
            logger.error(f"Error recording regressions: {e}")
            return 0
            
        finally:
            conn.close()
    
    def list_active_regressions(self, 
                               severity: Optional[str] = None, 
                               days: int = 30) -> List[Dict[str, Any]]:
        """
        List active regression alerts.
        
        Args:
            severity: Optional filter by severity
            days: Number of days to look back
            
        Returns:
            List of active regression alerts
        """
        conn = self._get_connection()
        
        try:
            # Build query
            query = """
            SELECT 
                ra.id,
                m.model_name,
                hp.hardware_type,
                ra.metric,
                ra.previous_value,
                ra.current_value,
                ra.percent_change,
                ra.severity,
                ra.timestamp,
                ra.notification_sent
            FROM 
                regression_alerts ra
            JOIN 
                models m ON ra.model_id = m.model_id
            JOIN 
                hardware_platforms hp ON ra.hardware_id = hp.hardware_id
            WHERE 
                ra.status = 'active'
                AND ra.timestamp >= ?
            """
            
            params = [(datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')]
            
            if severity:
                query += " AND ra.severity = ?"
                params.append(severity)
            
            query += " ORDER BY ra.severity, ra.timestamp DESC"
            
            # Execute query
            result = conn.execute(query, params).fetchall()
            
            # Format results
            alerts = []
            for row in result:
                alerts.append({
                    'id': row[0],
                    'model_name': row[1],
                    'hardware_type': row[2],
                    'metric': row[3],
                    'previous_value': row[4],
                    'current_value': row[5],
                    'percent_change': row[6],
                    'severity': row[7],
                    'timestamp': row[8],
                    'notification_sent': row[9]
                })
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error listing active regressions: {e}")
            return []
            
        finally:
            conn.close()
    
    def resolve_regression(self, regression_id: int, resolution_note: Optional[str] = None) -> bool:
        """
        Mark a regression alert as resolved.
        
        Args:
            regression_id: ID of the regression alert
            resolution_note: Optional note about resolution
            
        Returns:
            success: True if successful
        """
        conn = self._get_connection()
        
        try:
            # Check if regression exists
            result = conn.execute(
                "SELECT id FROM regression_alerts WHERE id = ?",
                [regression_id]
            ).fetchone()
            
            if not result:
                logger.warning(f"Regression alert with ID {regression_id} not found")
                return False
            
            # Update regression status
            conn.execute(
                """
                UPDATE regression_alerts
                SET status = 'resolved',
                    resolved_at = CURRENT_TIMESTAMP,
                    metadata = json_insert(coalesce(metadata, '{}'), '$.resolution_note', ?)
                WHERE id = ?
                """,
                [resolution_note, regression_id]
            )
            
            logger.info(f"Marked regression alert {regression_id} as resolved")
            return True
            
        except Exception as e:
            logger.error(f"Error resolving regression: {e}")
            return False
            
        finally:
            conn.close()


class TrendVisualizer:
    """Creates visualizations for performance trends over time."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize with database path."""
        self.db_path = db_path or os.environ.get('BENCHMARK_DB_PATH', './benchmark_db.duckdb')
        
        # Initialize time series manager
        self.ts_manager = TimeSeriesManager(self.db_path)
        
        # Set up Seaborn style
        sns.set_style('whitegrid')
    
    def visualize_metric_trend(self, 
                              model_name: str, 
                              hardware_type: str, 
                              metric: str = 'throughput',
                              batch_size: Optional[int] = None,
                              test_type: Optional[str] = None,
                              days: int = 30,
                              output_file: Optional[str] = None,
                              show_versions: bool = True) -> str:
        """
        Create visualization for a metric trend over time.
        
        Args:
            model_name: Name of the model
            hardware_type: Type of hardware
            metric: Performance metric to visualize
            batch_size: Optional filter for batch size
            test_type: Optional filter for test type
            days: Number of days of history
            output_file: Path to save the visualization
            show_versions: Whether to show version markers
            
        Returns:
            Path to the saved visualization
        """
        # Get performance history
        df = self.ts_manager.get_performance_history(
            model_name=model_name,
            hardware_type=hardware_type,
            batch_size=batch_size,
            test_type=test_type,
            metric=metric,
            days=days
        )
        
        if df.empty:
            logger.warning(f"No data found for {model_name} on {hardware_type}")
            return ""
        
        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # If batch_size is not specified and multiple batch sizes exist, group by batch_size
        if batch_size is None and len(df['batch_size'].unique()) > 1:
            for bs, group in df.groupby('batch_size'):
                plt.plot(group['timestamp'], group[metric], marker='o', linestyle='-', 
                         label=f'Batch Size {bs}')
        else:
            plt.plot(df['timestamp'], df[metric], marker='o', linestyle='-')
        
        # Add version markers if requested
        if show_versions and 'version_tag' in df.columns:
            for i, row in df.iterrows():
                if pd.notna(row['version_tag']):
                    plt.axvline(x=row['timestamp'], color='gray', linestyle='--', alpha=0.5)
                    plt.annotate(row['version_tag'], 
                               (row['timestamp'], df[metric].min()),
                               rotation=90, fontsize=9, alpha=0.8,
                               horizontalalignment='right', verticalalignment='bottom')
        
        # Set labels and title
        plt.xlabel('Date')
        plt.ylabel(metric.capitalize())
        metric_label = metric.replace('_', ' ').capitalize()
        title = f'{metric_label} Trend for {model_name} on {hardware_type}'
        if batch_size:
            title += f' (Batch Size {batch_size})'
        if test_type:
            title += f' - {test_type}'
        plt.title(title)
        
        # Add legend if multiple batch sizes
        if batch_size is None and len(df['batch_size'].unique()) > 1:
            plt.legend(title='Batch Size')
        
        # Format x-axis ticks
        plt.gcf().autofmt_xdate()
        plt.tight_layout()
        
        # Save the visualization
        if output_file:
            plt.savefig(output_file, dpi=300)
            logger.info(f"Saved visualization to {output_file}")
            return output_file
        
        # Generate default filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(os.getcwd(), "benchmark_visualizations")
        os.makedirs(output_dir, exist_ok=True)
        
        filename = os.path.join(output_dir, f"{model_name}_{hardware_type}_{metric}_{timestamp}.png")
        plt.savefig(filename, dpi=300)
        plt.close()
        
        logger.info(f"Saved visualization to {filename}")
        return filename
    
    def create_regression_dashboard(self,
                                  days: int = 30,
                                  limit: int = 10,
                                  output_file: Optional[str] = None) -> str:
        """
        Create a dashboard of recent regressions.
        
        Args:
            days: Number of days to include
            limit: Maximum number of regressions to show
            output_file: Path to save the dashboard
            
        Returns:
            Path to the saved dashboard
        """
        # Initialize regression detector
        detector = RegressionDetector(self.db_path)
        
        # Get active regressions
        regressions = detector.list_active_regressions(days=days)
        
        if not regressions:
            logger.warning(f"No regression alerts found in the last {days} days")
            return ""
        
        # Sort by severity and timestamp
        regressions.sort(key=lambda x: (
            {'critical': 0, 'high': 1, 'medium': 2}.get(x['severity'], 3),
            x['timestamp']
        ), reverse=True)
        
        # Limit number of regressions
        regressions = regressions[:limit]
        
        # Create dashboard figure
        fig_height = min(4 + len(regressions) * 0.5, 15)
        plt.figure(figsize=(12, fig_height))
        
        # Create summary bar chart
        severity_counts = {}
        for reg in regressions:
            severity = reg['severity']
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Set up colors
        colors = {'critical': 'red', 'high': 'orange', 'medium': 'yellow'}
        
        # Create figure
        plt.subplot(len(regressions) + 1, 1, 1)
        plt.bar(severity_counts.keys(), severity_counts.values(), 
               color=[colors.get(s, 'blue') for s in severity_counts.keys()])
        plt.title('Active Regression Alerts by Severity')
        plt.xlabel('Severity')
        plt.ylabel('Count')
        
        # Create individual regression info
        for i, regression in enumerate(regressions):
            plt.subplot(len(regressions) + 1, 1, i + 2)
            
            # Format information
            title = f"{regression['model_name']} on {regression['hardware_type']} - {regression['metric']}"
            details = (f"Change: {regression['percent_change']:.2f}% "
                      f"({'increase' if regression['metric'] == 'latency' else 'decrease'}) - "
                      f"Severity: {regression['severity']}")
            
            # Create text-only panel
            plt.text(0.5, 0.5, f"{title}\n{details}", 
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=plt.gca().transAxes,
                    fontsize=10,
                    bbox=dict(facecolor=colors.get(regression['severity'], 'white'), 
                             alpha=0.2, boxstyle='round'))
            
            plt.axis('off')
        
        plt.tight_layout()
        
        # Save the dashboard
        if output_file:
            plt.savefig(output_file, dpi=300)
            logger.info(f"Saved dashboard to {output_file}")
            return output_file
        
        # Generate default filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(os.getcwd(), "benchmark_visualizations")
        os.makedirs(output_dir, exist_ok=True)
        
        filename = os.path.join(output_dir, f"regression_dashboard_{timestamp}.png")
        plt.savefig(filename, dpi=300)
        plt.close()
        
        logger.info(f"Saved dashboard to {filename}")
        return filename
    
    def create_comparative_dashboard(self,
                                   model_names: List[str],
                                   hardware_types: List[str],
                                   metric: str = 'throughput',
                                   days: int = 30,
                                   output_file: Optional[str] = None) -> str:
        """
        Create a dashboard comparing multiple models across hardware platforms.
        
        Args:
            model_names: List of model names to compare
            hardware_types: List of hardware types to compare
            metric: Metric to compare
            days: Number of days of history
            output_file: Path to save the dashboard
            
        Returns:
            Path to the saved dashboard
        """
        if not model_names or not hardware_types:
            logger.warning("No models or hardware types specified for comparison")
            return ""
        
        # Set up figure size based on number of comparisons
        fig_width = 12
        fig_height = 4 * len(model_names)
        plt.figure(figsize=(fig_width, fig_height))
        
        # Create subplots for each model
        for i, model_name in enumerate(model_names):
            plt.subplot(len(model_names), 1, i + 1)
            
            # Collect data for each hardware type
            for hardware_type in hardware_types:
                df = self.ts_manager.get_performance_history(
                    model_name=model_name,
                    hardware_type=hardware_type,
                    metric=metric,
                    days=days
                )
                
                if not df.empty:
                    # Convert timestamp to datetime if needed
                    if not pd.api.types.is_datetime64_dtype(df['timestamp']):
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                    
                    # Plot data
                    plt.plot(df['timestamp'], df[metric], marker='o', linestyle='-', 
                            label=hardware_type)
            
            # Set up labels and title
            plt.xlabel('Date')
            plt.ylabel(metric.capitalize())
            plt.title(f'{model_name} - {metric.capitalize()} Comparison')
            plt.legend(title='Hardware')
            plt.grid(True, alpha=0.3)
            
            # Format x-axis ticks
            plt.gcf().autofmt_xdate()
        
        plt.tight_layout()
        
        # Save the dashboard
        if output_file:
            plt.savefig(output_file, dpi=300)
            logger.info(f"Saved comparison dashboard to {output_file}")
            return output_file
        
        # Generate default filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(os.getcwd(), "benchmark_visualizations")
        os.makedirs(output_dir, exist_ok=True)
        
        model_str = "_".join([m.split("/")[-1] for m in model_names])
        if len(model_str) > 50:
            model_str = model_str[:47] + "..."
        
        filename = os.path.join(output_dir, f"comparison_{metric}_{model_str}_{timestamp}.png")
        plt.savefig(filename, dpi=300)
        plt.close()
        
        logger.info(f"Saved comparison dashboard to {filename}")
        return filename


class NotificationSystem:
    """Notification system for performance regressions."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize with database path."""
        self.db_path = db_path or os.environ.get('BENCHMARK_DB_PATH', './benchmark_db.duckdb')
    
    def _get_connection(self):
        """Get a connection to the database."""
        try:
            return get_db_connection(self.db_path)
        except NameError:
            # Fallback if get_db_connection is not available
            import duckdb
            return duckdb.connect(self.db_path)
    
    def get_pending_notifications(self) -> List[Dict[str, Any]]:
        """
        Get regression alerts that need notifications.
        
        Returns:
            List of alerts that need notifications
        """
        conn = self._get_connection()
        
        try:
            query = """
            SELECT
                ra.id,
                m.model_name,
                hp.hardware_type,
                ra.metric,
                ra.previous_value,
                ra.current_value,
                ra.percent_change,
                ra.severity,
                ra.timestamp
            FROM 
                regression_alerts ra
            JOIN 
                models m ON ra.model_id = m.model_id
            JOIN 
                hardware_platforms hp ON ra.hardware_id = hp.hardware_id
            WHERE 
                ra.notification_sent = FALSE
                AND ra.status = 'active'
            ORDER BY 
                ra.severity DESC,
                ra.timestamp DESC
            """
            
            result = conn.execute(query).fetchall()
            
            if not result:
                return []
            
            # Format results
            notifications = []
            for row in result:
                notifications.append({
                    'id': row[0],
                    'model_name': row[1],
                    'hardware_type': row[2],
                    'metric': row[3],
                    'previous_value': row[4],
                    'current_value': row[5],
                    'percent_change': row[6],
                    'severity': row[7],
                    'timestamp': row[8]
                })
            
            return notifications
            
        except Exception as e:
            logger.error(f"Error getting pending notifications: {e}")
            return []
            
        finally:
            conn.close()
    
    def mark_notification_sent(self, alert_id: int) -> bool:
        """
        Mark a regression alert as notified.
        
        Args:
            alert_id: ID of the regression alert
            
        Returns:
            Success status
        """
        conn = self._get_connection()
        
        try:
            conn.execute(
                """
                UPDATE regression_alerts
                SET notification_sent = TRUE
                WHERE id = ?
                """,
                [alert_id]
            )
            
            logger.info(f"Marked regression alert {alert_id} as notified")
            return True
            
        except Exception as e:
            logger.error(f"Error marking notification as sent: {e}")
            return False
            
        finally:
            conn.close()
    
    def create_github_issue(self, regression: Dict[str, Any], repository: str) -> Optional[str]:
        """
        Create a GitHub issue for a regression alert.
        
        Args:
            regression: Regression alert information
            repository: GitHub repository (owner/repo)
            
        Returns:
            Issue URL if successful, None otherwise
        """
        try:
            # Check if GitHub CLI is available
            import subprocess
            result = subprocess.run(["gh", "--version"], capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("GitHub CLI not available")
                return None
            
            # Generate issue title
            severity_emoji = {
                'critical': '🔴',
                'high': '🟠',
                'medium': '🟡'
            }
            emoji = severity_emoji.get(regression['severity'], '⚪')
            
            title = f"{emoji} {regression['severity'].upper()} regression: {regression['model_name']} on {regression['hardware_type']}"
            
            # Generate issue body
            body = f"""
## Performance Regression Detected

- **Model**: {regression['model_name']}
- **Hardware**: {regression['hardware_type']}
- **Metric**: {regression['metric']}
- **Change**: {regression['percent_change']:.2f}% {'increase' if regression['metric'] == 'latency' else 'decrease'}
- **Previous**: {regression['previous_value']:.4f}
- **Current**: {regression['current_value']:.4f}
- **Detected**: {regression['timestamp']}
- **Severity**: {regression['severity']}

## Recommended Actions

1. Verify the regression with additional tests
2. Check recent code changes that might affect performance
3. Investigate potential hardware or environment issues
4. Update benchmark baselines if the change is expected

## Automatically Generated

This issue was automatically generated by the performance regression detection system.
            """
            
            # Create issue using GitHub CLI
            body_file = os.path.join(os.getcwd(), "temp_issue_body.md")
            with open(body_file, "w") as f:
                f.write(body)
            
            cmd = ["gh", "issue", "create", "--repo", repository, "--title", title, "--body-file", body_file]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Clean up temporary file
            try:
                os.remove(body_file)
            except:
                pass
            
            if result.returncode != 0:
                logger.error(f"Error creating GitHub issue: {result.stderr}")
                return None
            
            # Extract URL from response
            issue_url = result.stdout.strip()
            logger.info(f"Created GitHub issue: {issue_url}")
            return issue_url
            
        except Exception as e:
            logger.error(f"Error creating GitHub issue: {e}")
            return None
    
    def send_email_notification(self, 
                              regression: Dict[str, Any],
                              recipients: List[str],
                              smtp_server: str,
                              smtp_port: int = 587,
                              smtp_user: Optional[str] = None,
                              smtp_password: Optional[str] = None) -> bool:
        """
        Send email notification about a regression.
        
        Args:
            regression: Regression alert information
            recipients: List of email recipients
            smtp_server: SMTP server address
            smtp_port: SMTP server port
            smtp_user: Optional SMTP username
            smtp_password: Optional SMTP password
            
        Returns:
            True if sent successfully, False otherwise
        """
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            # Create message
            message = MIMEMultipart()
            message['Subject'] = f"[{regression['severity'].upper()}] Performance Regression: {regression['model_name']} on {regression['hardware_type']}"
            message['From'] = smtp_user or "performance-monitor@example.com"
            message['To'] = ", ".join(recipients)
            
            # Create email body
            body = f"""
<h2>Performance Regression Detected</h2>

<p><strong>Model</strong>: {regression['model_name']}<br/>
<strong>Hardware</strong>: {regression['hardware_type']}<br/>
<strong>Metric</strong>: {regression['metric']}<br/>
<strong>Change</strong>: {regression['percent_change']:.2f}% {'increase' if regression['metric'] == 'latency' else 'decrease'}<br/>
<strong>Previous</strong>: {regression['previous_value']:.4f}<br/>
<strong>Current</strong>: {regression['current_value']:.4f}<br/>
<strong>Detected</strong>: {regression['timestamp']}<br/>
<strong>Severity</strong>: {regression['severity']}</p>

<h2>Recommended Actions</h2>

<ol>
<li>Verify the regression with additional tests</li>
<li>Check recent code changes that might affect performance</li>
<li>Investigate potential hardware or environment issues</li>
<li>Update benchmark baselines if the change is expected</li>
</ol>

<p><em>This email was automatically generated by the performance regression detection system.</em></p>
            """
            
            message.attach(MIMEText(body, 'html'))
            
            # Connect to SMTP server and send
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                
                if smtp_user and smtp_password:
                    server.login(smtp_user, smtp_password)
                
                server.send_message(message)
            
            logger.info(f"Sent email notification to {len(recipients)} recipients")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
            return False
    
    def process_notifications(self, 
                            github_repo: Optional[str] = None,
                            email_recipients: Optional[List[str]] = None,
                            smtp_settings: Optional[Dict[str, Any]] = None) -> int:
        """
        Process all pending notifications.
        
        Args:
            github_repo: Optional GitHub repository for issues (owner/repo)
            email_recipients: Optional list of email recipients
            smtp_settings: Optional SMTP settings for email
            
        Returns:
            Number of notifications processed
        """
        # Get pending notifications
        notifications = self.get_pending_notifications()
        
        if not notifications:
            logger.info("No pending notifications to process")
            return 0
        
        logger.info(f"Processing {len(notifications)} notification(s)...")
        processed = 0
        
        for notification in notifications:
            # Determine notification method based on severity
            if notification['severity'] == 'critical':
                # For critical regressions, use all available notification methods
                success = False
                
                # Create GitHub issue if enabled
                if github_repo:
                    issue_url = self.create_github_issue(notification, github_repo)
                    if issue_url:
                        success = True
                
                # Send email if enabled
                if email_recipients and smtp_settings:
                    email_sent = self.send_email_notification(
                        notification,
                        email_recipients,
                        smtp_settings.get('server'),
                        smtp_settings.get('port', 587),
                        smtp_settings.get('user'),
                        smtp_settings.get('password')
                    )
                    if email_sent:
                        success = True
                
                if success:
                    self.mark_notification_sent(notification['id'])
                    processed += 1
                    
            elif notification['severity'] == 'high':
                # For high severity, create GitHub issue if enabled
                if github_repo:
                    issue_url = self.create_github_issue(notification, github_repo)
                    if issue_url:
                        self.mark_notification_sent(notification['id'])
                        processed += 1
                else:
                    # If GitHub not enabled, mark as sent anyway (will appear in dashboard)
                    self.mark_notification_sent(notification['id'])
                    processed += 1
                    
            else:
                # For medium severity, just mark as sent (will appear in dashboard)
                self.mark_notification_sent(notification['id'])
                processed += 1
        
        logger.info(f"Processed {processed} notification(s)")
        return processed


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Time Series Performance Tracker')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Schema creation command
    schema_parser = subparsers.add_parser('create-schema', help='Create schema extensions')
    schema_parser.add_argument('--db-path', help='Database path')
    
    # Record performance command
    record_parser = subparsers.add_parser('record', help='Record performance metric')
    record_parser.add_argument('--db-path', help='Database path')
    record_parser.add_argument('--model', required=True, help='Model name')
    record_parser.add_argument('--hardware', required=True, help='Hardware type')
    record_parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    record_parser.add_argument('--test-type', default='default', help='Test type')
    record_parser.add_argument('--throughput', type=float, required=True, help='Throughput value')
    record_parser.add_argument('--latency', type=float, required=True, help='Latency value')
    record_parser.add_argument('--memory', type=float, help='Memory usage value')
    record_parser.add_argument('--version', help='Version tag')
    
    # Create version command
    version_parser = subparsers.add_parser('create-version', help='Create version entry')
    version_parser.add_argument('--db-path', help='Database path')
    version_parser.add_argument('--tag', required=True, help='Version tag')
    version_parser.add_argument('--description', help='Version description')
    version_parser.add_argument('--commit', help='Git commit hash')
    version_parser.add_argument('--branch', help='Git branch')
    version_parser.add_argument('--user', help='User ID')
    
    # Regression detection command
    detect_parser = subparsers.add_parser('detect', help='Detect performance regressions')
    detect_parser.add_argument('--db-path', help='Database path')
    detect_parser.add_argument('--model', help='Model name filter')
    detect_parser.add_argument('--hardware', help='Hardware type filter')
    detect_parser.add_argument('--metric', default='throughput', 
                             choices=['throughput', 'latency', 'memory_usage'],
                             help='Metric to analyze')
    detect_parser.add_argument('--threshold', type=float, default=0.1,
                             help='Regression threshold (0.1 = 10%)')
    detect_parser.add_argument('--window', type=int, default=5,
                             help='Window size for baseline calculation')
    detect_parser.add_argument('--record', action='store_true',
                             help='Record detected regressions in the database')
    
    # Visualization command
    visualize_parser = subparsers.add_parser('visualize', help='Create trend visualization')
    visualize_parser.add_argument('--db-path', help='Database path')
    visualize_parser.add_argument('--model', required=True, help='Model name')
    visualize_parser.add_argument('--hardware', required=True, help='Hardware type')
    visualize_parser.add_argument('--metric', default='throughput',
                                choices=['throughput', 'latency', 'memory_usage'],
                                help='Metric to visualize')
    visualize_parser.add_argument('--batch-size', type=int, help='Batch size filter')
    visualize_parser.add_argument('--test-type', help='Test type filter')
    visualize_parser.add_argument('--days', type=int, default=30, help='Days of history')
    visualize_parser.add_argument('--output', help='Output file path')
    visualize_parser.add_argument('--hide-versions', action='store_true',
                                help='Hide version markers')
    
    # Regression dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Create regression dashboard')
    dashboard_parser.add_argument('--db-path', help='Database path')
    dashboard_parser.add_argument('--days', type=int, default=30, help='Days to include')
    dashboard_parser.add_argument('--limit', type=int, default=10, help='Maximum regressions')
    dashboard_parser.add_argument('--output', help='Output file path')
    
    # Comparison dashboard command
    compare_parser = subparsers.add_parser('compare', help='Create comparison dashboard')
    compare_parser.add_argument('--db-path', help='Database path')
    compare_parser.add_argument('--models', required=True, help='Comma-separated list of model names')
    compare_parser.add_argument('--hardware', required=True, help='Comma-separated list of hardware types')
    compare_parser.add_argument('--metric', default='throughput',
                               choices=['throughput', 'latency', 'memory_usage'],
                               help='Metric to compare')
    compare_parser.add_argument('--days', type=int, default=30, help='Days of history')
    compare_parser.add_argument('--output', help='Output file path')
    
    # Notification command
    notify_parser = subparsers.add_parser('notify', help='Process notifications')
    notify_parser.add_argument('--db-path', help='Database path')
    notify_parser.add_argument('--github-repo', help='GitHub repository (owner/repo)')
    notify_parser.add_argument('--email-to', help='Comma-separated list of email recipients')
    notify_parser.add_argument('--smtp-server', help='SMTP server address')
    notify_parser.add_argument('--smtp-port', type=int, default=587, help='SMTP server port')
    notify_parser.add_argument('--smtp-user', help='SMTP username')
    notify_parser.add_argument('--smtp-password', help='SMTP password')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if args.command == 'create-schema':
        schema = TimeSeriesSchema(args.db_path)
        success = schema.create_schema_extensions()
        if success:
            print("Schema extensions created successfully")
        else:
            print("Failed to create schema extensions")
            sys.exit(1)
    
    elif args.command == 'record':
        ts_manager = TimeSeriesManager(args.db_path)
        success = ts_manager.record_performance(
            model_name=args.model,
            hardware_type=args.hardware,
            batch_size=args.batch_size,
            test_type=args.test_type,
            throughput=args.throughput,
            latency=args.latency,
            memory_usage=args.memory,
            version_tag=args.version
        )
        if success:
            print(f"Performance metric recorded for {args.model} on {args.hardware}")
        else:
            print("Failed to record performance metric")
            sys.exit(1)
    
    elif args.command == 'create-version':
        version_manager = VersionManager(args.db_path)
        version_id = version_manager.create_version(
            version_tag=args.tag,
            description=args.description,
            commit_hash=args.commit,
            git_branch=args.branch,
            user_id=args.user
        )
        if version_id:
            print(f"Version '{args.tag}' created with ID {version_id}")
        else:
            print(f"Failed to create version '{args.tag}'")
            sys.exit(1)
    
    elif args.command == 'detect':
        detector = RegressionDetector(args.db_path, args.threshold)
        regressions = detector.detect_regressions(
            model_name=args.model,
            hardware_type=args.hardware,
            metric=args.metric,
            window_size=args.window
        )
        
        if regressions:
            print(f"Detected {len(regressions)} regression(s):")
            for reg in regressions:
                print(f"- {reg['model_name']} on {reg['hardware_type']}: "
                     f"{reg['percent_change']:.2f}% {args.metric} "
                     f"{'increase' if args.metric == 'latency' else 'decrease'} "
                     f"(severity: {reg['severity']})")
            
            if args.record:
                recorded = detector.record_regressions(regressions)
                print(f"Recorded {recorded} regression alert(s) in the database")
        else:
            print("No regressions detected")
    
    elif args.command == 'visualize':
        visualizer = TrendVisualizer(args.db_path)
        output_file = visualizer.visualize_metric_trend(
            model_name=args.model,
            hardware_type=args.hardware,
            metric=args.metric,
            batch_size=args.batch_size,
            test_type=args.test_type,
            days=args.days,
            output_file=args.output,
            show_versions=not args.hide_versions
        )
        
        if output_file:
            print(f"Visualization saved to: {output_file}")
        else:
            print("Failed to create visualization")
            sys.exit(1)
    
    elif args.command == 'dashboard':
        visualizer = TrendVisualizer(args.db_path)
        output_file = visualizer.create_regression_dashboard(
            days=args.days,
            limit=args.limit,
            output_file=args.output
        )
        
        if output_file:
            print(f"Dashboard saved to: {output_file}")
        else:
            print("Failed to create dashboard")
            sys.exit(1)
    
    elif args.command == 'compare':
        visualizer = TrendVisualizer(args.db_path)
        models = [m.strip() for m in args.models.split(',')]
        hardware = [h.strip() for h in args.hardware.split(',')]
        
        output_file = visualizer.create_comparative_dashboard(
            model_names=models,
            hardware_types=hardware,
            metric=args.metric,
            days=args.days,
            output_file=args.output
        )
        
        if output_file:
            print(f"Comparison dashboard saved to: {output_file}")
        else:
            print("Failed to create comparison dashboard")
            sys.exit(1)
    
    elif args.command == 'notify':
        notifier = NotificationSystem(args.db_path)
        
        # Process email settings if provided
        smtp_settings = None
        if args.smtp_server:
            smtp_settings = {
                'server': args.smtp_server,
                'port': args.smtp_port,
                'user': args.smtp_user,
                'password': args.smtp_password
            }
        
        # Process email recipients if provided
        email_recipients = None
        if args.email_to:
            email_recipients = [e.strip() for e in args.email_to.split(',')]
        
        processed = notifier.process_notifications(
            github_repo=args.github_repo,
            email_recipients=email_recipients,
            smtp_settings=smtp_settings
        )
        
        print(f"Processed {processed} notification(s)")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()