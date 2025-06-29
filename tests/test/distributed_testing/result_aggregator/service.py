#!/usr/bin/env python3
"""
Result Aggregator Service for Distributed Testing Framework

This module provides the core functionality for aggregating and analyzing test results
from the distributed testing framework. It includes statistical analysis, anomaly detection,
visualization, and machine learning integration.

Usage:
    # Create a result aggregator service with database integration
    aggregator = ResultAggregatorService(db_path="./test_db.duckdb")
    
    # Store a test result
    aggregator.store_result(test_result)
    
    # Get aggregated results
    results = aggregator.get_aggregated_results(filter_criteria={"model": "bert"})
    
    # Generate analysis report
    report = aggregator.generate_analysis_report(format="markdown")
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Union

import duckdb
import numpy as np
import pandas as pd
from pathlib import Path

# Import performance analyzer if available
try:
    from result_aggregator.performance_analyzer import PerformanceAnalyzer
    PERFORMANCE_ANALYZER_AVAILABLE = True
except ImportError:
    PERFORMANCE_ANALYZER_AVAILABLE = False
    logging.warning("Performance Analyzer not available. Advanced performance analysis features will be disabled.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("result_aggregator.log")
    ]
)
logger = logging.getLogger(__name__)

# Import optional dependencies if available
try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    logger.warning("Matplotlib not available. Visualization features will be disabled.")

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning("Scikit-learn not available. ML-based anomaly detection will be disabled.")

class ResultAggregatorService:
    """Service for aggregating and analyzing test results."""
    
    def __init__(self, db_path: str, enable_ml: bool = True, enable_visualization: bool = True):
        """
        Initialize the result aggregator service.
        
        Args:
            db_path: Path to DuckDB database
            enable_ml: Enable machine learning features
            enable_visualization: Enable visualization features
        """
        self.db_path = db_path
        self.enable_ml = enable_ml and ML_AVAILABLE
        self.enable_visualization = enable_visualization and VISUALIZATION_AVAILABLE
        
        # Connect to database
        self.db = None
        if db_path:
            try:
                # Ensure database directory exists
                db_dir = os.path.dirname(db_path)
                if db_dir and not os.path.exists(db_dir):
                    os.makedirs(db_dir)
                
                # Connect to database
                self.db = duckdb.connect(db_path)
                
                # Initialize database tables
                self._init_database_tables()
                
                logger.info(f"Connected to database at {db_path}")
            except Exception as e:
                logger.error(f"Error connecting to database: {e}")
        
        # Initialize ML components if enabled
        self.ml_models = {}
        if self.enable_ml:
            self._init_ml_components()
            
        # Initialize performance analyzer if available
        self.performance_analyzer = None
        if PERFORMANCE_ANALYZER_AVAILABLE:
            self.performance_analyzer = PerformanceAnalyzer(self)
    
    def _init_database_tables(self):
        """Initialize database tables."""
        try:
            # Test results table
            self.db.execute("""
            CREATE TABLE IF NOT EXISTS test_results (
                id INTEGER PRIMARY KEY,
                task_id VARCHAR,
                worker_id VARCHAR,
                timestamp TIMESTAMP,
                test_type VARCHAR,
                status VARCHAR,
                duration FLOAT,
                details JSON,
                metrics JSON
            )
            """)
            
            # Performance metrics table
            self.db.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY,
                result_id INTEGER,
                metric_name VARCHAR,
                metric_value FLOAT,
                metric_unit VARCHAR,
                FOREIGN KEY (result_id) REFERENCES test_results(id)
            )
            """)
            
            # Anomaly detection table
            self.db.execute("""
            CREATE TABLE IF NOT EXISTS anomaly_detections (
                id INTEGER PRIMARY KEY,
                result_id INTEGER,
                detection_time TIMESTAMP,
                anomaly_score FLOAT,
                anomaly_type VARCHAR,
                is_confirmed BOOLEAN,
                details JSON,
                FOREIGN KEY (result_id) REFERENCES test_results(id)
            )
            """)
            
            # Result aggregations table
            self.db.execute("""
            CREATE TABLE IF NOT EXISTS result_aggregations (
                id INTEGER PRIMARY KEY,
                aggregation_name VARCHAR,
                aggregation_type VARCHAR,
                filter_criteria JSON,
                aggregation_data JSON,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            )
            """)
            
            # Analysis reports table
            self.db.execute("""
            CREATE TABLE IF NOT EXISTS analysis_reports (
                id INTEGER PRIMARY KEY,
                report_name VARCHAR,
                report_type VARCHAR,
                filter_criteria JSON,
                report_data JSON,
                created_at TIMESTAMP
            )
            """)
            
            logger.info("Database tables initialized")
            
        except Exception as e:
            logger.error(f"Error initializing database tables: {e}")
    
    def _init_ml_components(self):
        """Initialize machine learning components."""
        if not self.enable_ml:
            return
        
        try:
            # Initialize isolation forest for anomaly detection
            self.ml_models["isolation_forest"] = {
                "model": IsolationForest(contamination=0.05, random_state=42),
                "scaler": StandardScaler(),
                "is_trained": False
            }
            
            logger.info("ML components initialized")
            
        except Exception as e:
            logger.error(f"Error initializing ML components: {e}")
            self.enable_ml = False
    
    def store_result(self, result: Dict[str, Any]) -> int:
        """
        Store a test result in the database.
        
        Args:
            result: Test result data
            
        Returns:
            Result ID
        """
        if not self.db:
            logger.warning("No database connection available. Result not stored.")
            return -1
        
        try:
            # Extract basic fields
            task_id = result.get("task_id", str(time.time()))
            worker_id = result.get("worker_id", "unknown")
            timestamp = result.get("timestamp", datetime.now().isoformat())
            test_type = result.get("type", "unknown")
            status = result.get("status", "unknown")
            duration = result.get("duration", 0.0)
            
            # Extract metrics and details
            metrics = result.get("metrics", {})
            details = {k: v for k, v in result.items() if k not in [
                "task_id", "worker_id", "timestamp", "type", "status", "duration", "metrics"
            ]}
            
            # Convert timestamp to datetime if it's a string
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            
            # Insert into test_results table
            self.db.execute(
                """
                INSERT INTO test_results 
                (task_id, worker_id, timestamp, test_type, status, duration, details, metrics)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                RETURNING id
                """,
                (task_id, worker_id, timestamp, test_type, status, duration, 
                 json.dumps(details), json.dumps(metrics))
            )
            
            result_id = self.db.fetchone()[0]
            
            # Insert metrics into performance_metrics table
            for metric_name, metric_data in metrics.items():
                if isinstance(metric_data, dict):
                    metric_value = metric_data.get("value", 0.0)
                    metric_unit = metric_data.get("unit", "")
                else:
                    metric_value = float(metric_data)
                    metric_unit = ""
                
                self.db.execute(
                    """
                    INSERT INTO performance_metrics
                    (result_id, metric_name, metric_value, metric_unit)
                    VALUES (?, ?, ?, ?)
                    """,
                    (result_id, metric_name, metric_value, metric_unit)
                )
            
            logger.info(f"Stored test result {result_id} for task {task_id}")
            
            # Perform anomaly detection if ML is enabled
            if self.enable_ml:
                self._detect_anomalies_for_result(result_id)
            
            return result_id
            
        except Exception as e:
            logger.error(f"Error storing test result: {e}")
            return -1
    
    def get_result(self, result_id: int) -> Dict[str, Any]:
        """
        Get a test result from the database.
        
        Args:
            result_id: Result ID to retrieve
            
        Returns:
            Test result data
        """
        if not self.db:
            logger.warning("No database connection available.")
            return {}
        
        try:
            # Query test_results table
            result = self.db.execute(
                """
                SELECT id, task_id, worker_id, timestamp, test_type, status, duration, details, metrics
                FROM test_results
                WHERE id = ?
                """,
                (result_id,)
            ).fetchone()
            
            if not result:
                logger.warning(f"No result found with ID {result_id}")
                return {}
            
            # Convert to dictionary
            result_dict = {
                "id": result[0],
                "task_id": result[1],
                "worker_id": result[2],
                "timestamp": result[3],
                "type": result[4],
                "status": result[5],
                "duration": result[6],
                "details": json.loads(result[7]),
                "metrics": json.loads(result[8])
            }
            
            # Query performance_metrics table for additional metrics
            metrics = self.db.execute(
                """
                SELECT metric_name, metric_value, metric_unit
                FROM performance_metrics
                WHERE result_id = ?
                """,
                (result_id,)
            ).fetchall()
            
            # Add metrics to result dictionary
            for metric in metrics:
                metric_name, metric_value, metric_unit = metric
                if metric_unit:
                    result_dict["metrics"][metric_name] = {
                        "value": metric_value,
                        "unit": metric_unit
                    }
                else:
                    result_dict["metrics"][metric_name] = metric_value
            
            # Query anomaly_detections table for anomalies
            anomalies = self.db.execute(
                """
                SELECT anomaly_score, anomaly_type, is_confirmed, details
                FROM anomaly_detections
                WHERE result_id = ?
                """,
                (result_id,)
            ).fetchall()
            
            # Add anomalies to result dictionary
            if anomalies:
                result_dict["anomalies"] = []
                for anomaly in anomalies:
                    anomaly_score, anomaly_type, is_confirmed, details = anomaly
                    result_dict["anomalies"].append({
                        "score": anomaly_score,
                        "type": anomaly_type,
                        "confirmed": is_confirmed,
                        "details": json.loads(details)
                    })
            
            return result_dict
            
        except Exception as e:
            logger.error(f"Error retrieving test result: {e}")
            return {}
    
    def get_results(self, filter_criteria: Dict[str, Any] = None, 
                   limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get test results from the database based on filter criteria.
        
        Args:
            filter_criteria: Filter criteria for results
            limit: Maximum number of results to return
            offset: Offset for pagination
            
        Returns:
            List of test results
        """
        if not self.db:
            logger.warning("No database connection available.")
            return []
        
        try:
            # Build query based on filter criteria
            query = """
            SELECT id, task_id, worker_id, timestamp, test_type, status, duration, details, metrics
            FROM test_results
            """
            
            params = []
            
            if filter_criteria:
                conditions = []
                
                for key, value in filter_criteria.items():
                    if key == "test_type":
                        conditions.append("test_type = ?")
                        params.append(value)
                    elif key == "status":
                        conditions.append("status = ?")
                        params.append(value)
                    elif key == "worker_id":
                        conditions.append("worker_id = ?")
                        params.append(value)
                    elif key == "task_id":
                        conditions.append("task_id = ?")
                        params.append(value)
                    elif key == "start_time":
                        conditions.append("timestamp >= ?")
                        params.append(value)
                    elif key == "end_time":
                        conditions.append("timestamp <= ?")
                        params.append(value)
                    elif key == "min_duration":
                        conditions.append("duration >= ?")
                        params.append(value)
                    elif key == "max_duration":
                        conditions.append("duration <= ?")
                        params.append(value)
                    elif key == "details":
                        for detail_key, detail_value in value.items():
                            conditions.append(f"json_extract(details, '$.{detail_key}') = ?")
                            params.append(str(detail_value))
                
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
            
            # Add order, limit, and offset
            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            # Execute query
            rows = self.db.execute(query, params).fetchall()
            
            # Convert to list of dictionaries
            results = []
            for row in rows:
                result_dict = {
                    "id": row[0],
                    "task_id": row[1],
                    "worker_id": row[2],
                    "timestamp": row[3],
                    "type": row[4],
                    "status": row[5],
                    "duration": row[6],
                    "details": json.loads(row[7]),
                    "metrics": json.loads(row[8])
                }
                results.append(result_dict)
            
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving test results: {e}")
            return []
    
    def get_aggregated_results(self, filter_criteria: Dict[str, Any] = None,
                              aggregation_type: str = "mean",
                              group_by: List[str] = None,
                              metrics: List[str] = None) -> Dict[str, Any]:
        """
        Get aggregated test results from the database.
        
        Args:
            filter_criteria: Filter criteria for results
            aggregation_type: Type of aggregation (mean, median, min, max, etc.)
            group_by: Fields to group by
            metrics: Metrics to aggregate
            
        Returns:
            Aggregated test results
        """
        if not self.db:
            logger.warning("No database connection available.")
            return {}
        
        try:
            # Start building the query
            base_query = """
            SELECT {select_clause}
            FROM test_results t
            JOIN performance_metrics m ON t.id = m.result_id
            """
            
            params = []
            
            # Apply filters
            if filter_criteria:
                conditions = []
                
                for key, value in filter_criteria.items():
                    if key == "test_type":
                        conditions.append("t.test_type = ?")
                        params.append(value)
                    elif key == "status":
                        conditions.append("t.status = ?")
                        params.append(value)
                    elif key == "worker_id":
                        conditions.append("t.worker_id = ?")
                        params.append(value)
                    elif key == "task_id":
                        conditions.append("t.task_id = ?")
                        params.append(value)
                    elif key == "start_time":
                        conditions.append("t.timestamp >= ?")
                        params.append(value)
                    elif key == "end_time":
                        conditions.append("t.timestamp <= ?")
                        params.append(value)
                    elif key == "metric_name":
                        conditions.append("m.metric_name = ?")
                        params.append(value)
                
                if conditions:
                    base_query += " WHERE " + " AND ".join(conditions)
            
            # Determine aggregation function
            if aggregation_type == "mean":
                agg_func = "AVG"
            elif aggregation_type == "median":
                agg_func = "MEDIAN"
            elif aggregation_type == "min":
                agg_func = "MIN"
            elif aggregation_type == "max":
                agg_func = "MAX"
            elif aggregation_type == "count":
                agg_func = "COUNT"
            elif aggregation_type == "sum":
                agg_func = "SUM"
            else:
                logger.warning(f"Unknown aggregation type: {aggregation_type}. Using mean.")
                agg_func = "AVG"
            
            # Build select clause
            select_parts = []
            
            # Add group by fields
            if group_by:
                for field in group_by:
                    if field == "test_type":
                        select_parts.append("t.test_type")
                    elif field == "status":
                        select_parts.append("t.status")
                    elif field == "worker_id":
                        select_parts.append("t.worker_id")
                    elif field == "task_id":
                        select_parts.append("t.task_id")
                    elif field == "day":
                        select_parts.append("DATE_TRUNC('day', t.timestamp) AS day")
                    elif field == "hour":
                        select_parts.append("DATE_TRUNC('hour', t.timestamp) AS hour")
                    elif field == "metric_name":
                        select_parts.append("m.metric_name")
            
            # Add metrics
            if metrics:
                for metric in metrics:
                    select_parts.append(f"{agg_func}(CASE WHEN m.metric_name = '{metric}' THEN m.metric_value ELSE NULL END) AS {metric}")
            else:
                select_parts.append("m.metric_name")
                select_parts.append(f"{agg_func}(m.metric_value) AS value")
            
            # Complete select clause
            select_clause = ", ".join(select_parts)
            
            # Add group by clause
            group_by_clause = ""
            if group_by:
                group_by_parts = []
                for field in group_by:
                    if field == "test_type":
                        group_by_parts.append("t.test_type")
                    elif field == "status":
                        group_by_parts.append("t.status")
                    elif field == "worker_id":
                        group_by_parts.append("t.worker_id")
                    elif field == "task_id":
                        group_by_parts.append("t.task_id")
                    elif field == "day":
                        group_by_parts.append("DATE_TRUNC('day', t.timestamp)")
                    elif field == "hour":
                        group_by_parts.append("DATE_TRUNC('hour', t.timestamp)")
                    elif field == "metric_name":
                        group_by_parts.append("m.metric_name")
                
                if group_by_parts:
                    group_by_clause = " GROUP BY " + ", ".join(group_by_parts)
            elif not metrics:
                # If no specific metrics requested, group by metric_name
                group_by_clause = " GROUP BY m.metric_name"
            
            # Complete query
            query = base_query.format(select_clause=select_clause) + group_by_clause
            
            # Execute query
            rows = self.db.execute(query, params).fetchall()
            
            # Process results
            if not group_by:
                # Simple aggregation
                results = {}
                for row in rows:
                    metric_name = row[0]
                    value = row[1]
                    results[metric_name] = value
                return results
            else:
                # Group by results
                results = []
                for row in rows:
                    result = {}
                    for i, field in enumerate(group_by):
                        result[field] = row[i]
                    
                    if metrics:
                        # Add specific metrics
                        for j, metric in enumerate(metrics):
                            result[metric] = row[len(group_by) + j]
                    else:
                        # Add generic value
                        result["metric_name"] = row[len(group_by)]
                        result["value"] = row[len(group_by) + 1]
                    
                    results.append(result)
                return results
            
        except Exception as e:
            logger.error(f"Error retrieving aggregated results: {e}")
            return {}
    
    def analyze_performance_trends(self, filter_criteria: Dict[str, Any] = None,
                                  metrics: List[str] = None, 
                                  window_size: int = 10) -> Dict[str, Any]:
        """
        Analyze performance trends over time.
        
        Args:
            filter_criteria: Filter criteria for results
            metrics: Metrics to analyze
            window_size: Window size for moving average
            
        Returns:
            Performance trend analysis results
        """
        if not self.db:
            logger.warning("No database connection available.")
            return {}
        
        try:
            # Get results with timestamps for trend analysis
            base_query = """
            SELECT t.timestamp, m.metric_name, m.metric_value
            FROM test_results t
            JOIN performance_metrics m ON t.id = m.result_id
            """
            
            params = []
            
            # Apply filters
            if filter_criteria:
                conditions = []
                
                for key, value in filter_criteria.items():
                    if key == "test_type":
                        conditions.append("t.test_type = ?")
                        params.append(value)
                    elif key == "status":
                        conditions.append("t.status = ?")
                        params.append(value)
                    elif key == "worker_id":
                        conditions.append("t.worker_id = ?")
                        params.append(value)
                    elif key == "task_id":
                        conditions.append("t.task_id = ?")
                        params.append(value)
                    elif key == "start_time":
                        conditions.append("t.timestamp >= ?")
                        params.append(value)
                    elif key == "end_time":
                        conditions.append("t.timestamp <= ?")
                        params.append(value)
                
                if conditions:
                    base_query += " WHERE " + " AND ".join(conditions)
                    
                # Add metric filter if specified
                if metrics:
                    if conditions:
                        base_query += " AND m.metric_name IN (" + ", ".join(["?"] * len(metrics)) + ")"
                    else:
                        base_query += " WHERE m.metric_name IN (" + ", ".join(["?"] * len(metrics)) + ")"
                    params.extend(metrics)
            elif metrics:
                # Add metric filter if specified
                base_query += " WHERE m.metric_name IN (" + ", ".join(["?"] * len(metrics)) + ")"
                params.extend(metrics)
            
            # Add order
            base_query += " ORDER BY t.timestamp ASC"
            
            # Execute query
            rows = self.db.execute(base_query, params).fetchall()
            
            # Convert to pandas DataFrame for trend analysis
            df = pd.DataFrame(rows, columns=["timestamp", "metric_name", "value"])
            
            # Convert timestamp to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            
            # Analyze trends for each metric
            results = {}
            
            for metric_name, group in df.groupby("metric_name"):
                # Sort by timestamp
                group = group.sort_values("timestamp")
                
                # Calculate moving average
                group["moving_avg"] = group["value"].rolling(window=min(window_size, len(group))).mean()
                
                # Calculate trend (simple linear regression)
                x = np.arange(len(group))
                if len(x) > 1:  # Need at least 2 points for linear regression
                    y = group["value"].values
                    A = np.vstack([x, np.ones(len(x))]).T
                    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
                    
                    # Calculate percent change over the period
                    if len(group) > 1 and group["value"].iloc[0] != 0:
                        first_value = group["value"].iloc[0]
                        last_value = group["value"].iloc[-1]
                        percent_change = ((last_value - first_value) / abs(first_value)) * 100
                    else:
                        percent_change = 0
                    
                    # Determine trend direction
                    if slope > 0.01:
                        trend = "increasing"
                    elif slope < -0.01:
                        trend = "decreasing"
                    else:
                        trend = "stable"
                    
                    # Calculate statistics
                    mean = group["value"].mean()
                    median = group["value"].median()
                    min_val = group["value"].min()
                    max_val = group["value"].max()
                    std_dev = group["value"].std()
                    
                    # Create a time series
                    time_series = []
                    for _, row in group.iterrows():
                        time_series.append({
                            "timestamp": row["timestamp"].isoformat(),
                            "value": row["value"],
                            "moving_avg": row["moving_avg"] if not pd.isna(row["moving_avg"]) else None
                        })
                    
                    # Store results
                    results[metric_name] = {
                        "trend": trend,
                        "slope": slope,
                        "percent_change": percent_change,
                        "statistics": {
                            "mean": mean,
                            "median": median,
                            "min": min_val,
                            "max": max_val,
                            "std_dev": std_dev
                        },
                        "time_series": time_series
                    }
                else:
                    # Not enough data points
                    results[metric_name] = {
                        "trend": "unknown",
                        "error": "Not enough data points for trend analysis"
                    }
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing performance trends: {e}")
            return {}
    
    def _detect_anomalies_for_result(self, result_id: int) -> List[Dict[str, Any]]:
        """
        Detect anomalies for a specific test result.
        
        Args:
            result_id: Result ID to analyze
            
        Returns:
            List of detected anomalies
        """
        if not self.enable_ml:
            return []
        
        try:
            # Get result details
            result = self.get_result(result_id)
            if not result:
                logger.warning(f"No result found with ID {result_id}")
                return []
            
            # Get historical data for same test type
            filter_criteria = {
                "test_type": result["type"],
                "status": "completed",  # Only consider completed tests
                "end_time": result["timestamp"]  # Only consider tests before this one
            }
            
            historical_results = self.get_results(filter_criteria=filter_criteria, limit=100)
            
            if len(historical_results) < 10:
                logger.info(f"Not enough historical data for anomaly detection (need at least 10, have {len(historical_results)})")
                return []
            
            # Extract features for anomaly detection
            features = []
            for hist_result in historical_results:
                feature_vector = []
                
                # Add duration
                feature_vector.append(hist_result["duration"])
                
                # Add key metrics
                metrics = hist_result["metrics"]
                for metric_name in sorted(metrics.keys()):
                    if isinstance(metrics[metric_name], dict):
                        feature_vector.append(metrics[metric_name].get("value", 0.0))
                    else:
                        feature_vector.append(float(metrics[metric_name]))
                
                features.append(feature_vector)
            
            # Ensure all feature vectors have the same length
            max_len = max(len(f) for f in features)
            features = [f + [0] * (max_len - len(f)) for f in features]
            
            # Extract features for current result
            current_feature_vector = []
            
            # Add duration
            current_feature_vector.append(result["duration"])
            
            # Add key metrics
            metrics = result["metrics"]
            for metric_name in sorted(metrics.keys()):
                if isinstance(metrics[metric_name], dict):
                    current_feature_vector.append(metrics[metric_name].get("value", 0.0))
                else:
                    current_feature_vector.append(float(metrics[metric_name]))
            
            # Pad current feature vector
            current_feature_vector = current_feature_vector + [0] * (max_len - len(current_feature_vector))
            
            # Get isolation forest model
            iso_forest = self.ml_models["isolation_forest"]
            
            # Scale features
            X = np.array(features)
            if not iso_forest["is_trained"]:
                iso_forest["scaler"].fit(X)
                X_scaled = iso_forest["scaler"].transform(X)
                
                # Train isolation forest
                iso_forest["model"].fit(X_scaled)
                iso_forest["is_trained"] = True
            else:
                X_scaled = iso_forest["scaler"].transform(X)
            
            # Scale current feature vector
            current_feature_scaled = iso_forest["scaler"].transform(np.array([current_feature_vector]))
            
            # Predict anomaly
            anomaly_score = -iso_forest["model"].score_samples(current_feature_scaled)[0]
            is_anomaly = anomaly_score > 0.7  # Threshold for anomaly
            
            if is_anomaly:
                logger.info(f"Anomaly detected for result {result_id} with score {anomaly_score}")
                
                # Determine anomaly type
                anomaly_type = "performance"
                
                # Calculate Z-scores to identify specific anomalous metrics
                anomaly_details = {}
                
                # Calculate mean and std for each feature
                mean = np.mean(X, axis=0)
                std = np.std(X, axis=0)
                
                # Calculate Z-scores for current feature vector
                z_scores = [(current_feature_vector[i] - mean[i]) / max(std[i], 0.0001) for i in range(len(current_feature_vector))]
                
                # Identify anomalous features (Z-score > 3 or < -3)
                anomalous_features = []
                
                feature_names = ["duration"] + sorted(metrics.keys())
                for i, z_score in enumerate(z_scores):
                    if abs(z_score) > 3 and i < len(feature_names):
                        anomalous_features.append({
                            "feature": feature_names[i],
                            "value": current_feature_vector[i],
                            "z_score": z_score,
                            "mean": mean[i],
                            "std": std[i]
                        })
                
                anomaly_details["anomalous_features"] = anomalous_features
                
                # Store anomaly in database
                self.db.execute(
                    """
                    INSERT INTO anomaly_detections
                    (result_id, detection_time, anomaly_score, anomaly_type, is_confirmed, details)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (result_id, datetime.now(), anomaly_score, anomaly_type, False, json.dumps(anomaly_details))
                )
                
                return [{
                    "score": anomaly_score,
                    "type": anomaly_type,
                    "confirmed": False,
                    "details": anomaly_details
                }]
            
            return []
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return []
    
    def detect_anomalies(self, filter_criteria: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Detect anomalies in test results.
        
        Args:
            filter_criteria: Filter criteria for results
            
        Returns:
            List of detected anomalies
        """
        if not self.enable_ml:
            logger.warning("ML-based anomaly detection is disabled.")
            return []
        
        try:
            # Get results for anomaly detection
            results = self.get_results(filter_criteria=filter_criteria, limit=1000)
            
            anomalies = []
            
            # Group results by test_type
            test_types = {}
            for result in results:
                test_type = result["type"]
                if test_type not in test_types:
                    test_types[test_type] = []
                test_types[test_type].append(result)
            
            # Process each test type separately
            for test_type, type_results in test_types.items():
                if len(type_results) < 10:
                    logger.info(f"Not enough results for test type {test_type} (need at least 10, have {len(type_results)})")
                    continue
                
                # Process results in chronological order
                sorted_results = sorted(type_results, key=lambda r: r["timestamp"])
                
                # Process each result
                for result in sorted_results:
                    anomalies_for_result = self._detect_anomalies_for_result(result["id"])
                    anomalies.extend(anomalies_for_result)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return []
    
    def generate_analysis_report(self, filter_criteria: Dict[str, Any] = None,
                               report_type: str = "performance", format: str = "json") -> str:
        """
        Generate an analysis report.
        
        Args:
            filter_criteria: Filter criteria for results
            report_type: Type of report (performance, anomaly, etc.)
            format: Report format (json, markdown, html)
            
        Returns:
            Analysis report
        """
        try:
            report_data = {}
            
            if report_type == "performance":
                # Get aggregated performance metrics
                report_data["aggregated_metrics"] = self.get_aggregated_results(
                    filter_criteria=filter_criteria,
                    aggregation_type="mean"
                )
                
                # Get performance trends
                report_data["performance_trends"] = self.analyze_performance_trends(
                    filter_criteria=filter_criteria
                )
                
                # Get recent results
                report_data["recent_results"] = self.get_results(
                    filter_criteria=filter_criteria,
                    limit=10
                )
                
            elif report_type == "anomaly":
                # Get recent anomalies
                anomalies = []
                
                # Query anomaly_detections table
                if self.db:
                    query = """
                    SELECT a.id, a.result_id, a.detection_time, a.anomaly_score, a.anomaly_type, a.is_confirmed, a.details,
                           t.task_id, t.worker_id, t.test_type, t.status, t.timestamp
                    FROM anomaly_detections a
                    JOIN test_results t ON a.result_id = t.id
                    """
                    
                    params = []
                    
                    # Apply filters
                    if filter_criteria:
                        conditions = []
                        
                        for key, value in filter_criteria.items():
                            if key == "test_type":
                                conditions.append("t.test_type = ?")
                                params.append(value)
                            elif key == "status":
                                conditions.append("t.status = ?")
                                params.append(value)
                            elif key == "worker_id":
                                conditions.append("t.worker_id = ?")
                                params.append(value)
                            elif key == "task_id":
                                conditions.append("t.task_id = ?")
                                params.append(value)
                            elif key == "start_time":
                                conditions.append("t.timestamp >= ?")
                                params.append(value)
                            elif key == "end_time":
                                conditions.append("t.timestamp <= ?")
                                params.append(value)
                            elif key == "anomaly_type":
                                conditions.append("a.anomaly_type = ?")
                                params.append(value)
                            elif key == "min_score":
                                conditions.append("a.anomaly_score >= ?")
                                params.append(value)
                            elif key == "is_confirmed":
                                conditions.append("a.is_confirmed = ?")
                                params.append(value)
                        
                        if conditions:
                            query += " WHERE " + " AND ".join(conditions)
                    
                    # Add order and limit
                    query += " ORDER BY a.detection_time DESC LIMIT 100"
                    
                    # Execute query
                    rows = self.db.execute(query, params).fetchall()
                    
                    # Process results
                    for row in rows:
                        anomaly = {
                            "id": row[0],
                            "result_id": row[1],
                            "detection_time": row[2],
                            "anomaly_score": row[3],
                            "anomaly_type": row[4],
                            "is_confirmed": row[5],
                            "details": json.loads(row[6]),
                            "task_id": row[7],
                            "worker_id": row[8],
                            "test_type": row[9],
                            "status": row[10],
                            "timestamp": row[11]
                        }
                        anomalies.append(anomaly)
                
                report_data["anomalies"] = anomalies
                
            elif report_type == "summary":
                # Get summary statistics
                
                # Get total number of results
                if self.db:
                    total_results = self.db.execute("SELECT COUNT(*) FROM test_results").fetchone()[0]
                    report_data["total_results"] = total_results
                    
                    # Get results by status
                    status_counts = self.db.execute("""
                        SELECT status, COUNT(*) as count
                        FROM test_results
                        GROUP BY status
                    """).fetchall()
                    
                    report_data["status_counts"] = {
                        status: count for status, count in status_counts
                    }
                    
                    # Get results by test type
                    type_counts = self.db.execute("""
                        SELECT test_type, COUNT(*) as count
                        FROM test_results
                        GROUP BY test_type
                    """).fetchall()
                    
                    report_data["type_counts"] = {
                        test_type: count for test_type, count in type_counts
                    }
                    
                    # Get results by worker
                    worker_counts = self.db.execute("""
                        SELECT worker_id, COUNT(*) as count
                        FROM test_results
                        GROUP BY worker_id
                    """).fetchall()
                    
                    report_data["worker_counts"] = {
                        worker_id: count for worker_id, count in worker_counts
                    }
                    
                    # Get recent anomalies count
                    anomaly_count = self.db.execute("""
                        SELECT COUNT(*) FROM anomaly_detections
                        WHERE detection_time >= DATEADD('day', -7, CURRENT_TIMESTAMP)
                    """).fetchone()[0]
                    
                    report_data["recent_anomaly_count"] = anomaly_count
            
            # Format the report
            if format == "json":
                # Return JSON string
                return json.dumps(report_data, indent=2)
                
            elif format == "markdown":
                # Generate Markdown report
                markdown = f"# Analysis Report - {report_type.capitalize()}\n\n"
                markdown += f"Generated: {datetime.now().isoformat()}\n\n"
                
                if report_type == "performance":
                    # Add aggregated metrics section
                    markdown += "## Aggregated Metrics\n\n"
                    
                    if isinstance(report_data["aggregated_metrics"], dict):
                        markdown += "| Metric | Value |\n"
                        markdown += "|--------|-------|\n"
                        
                        for metric, value in report_data["aggregated_metrics"].items():
                            markdown += f"| {metric} | {value:.4f} |\n"
                    else:
                        markdown += "No metrics available.\n"
                    
                    # Add performance trends section
                    markdown += "\n## Performance Trends\n\n"
                    
                    for metric, trend_data in report_data["performance_trends"].items():
                        markdown += f"### {metric}\n\n"
                        
                        if "error" in trend_data:
                            markdown += f"Error: {trend_data['error']}\n\n"
                            continue
                        
                        markdown += f"- Trend: {trend_data['trend']}\n"
                        markdown += f"- Slope: {trend_data['slope']:.4f}\n"
                        markdown += f"- Percent Change: {trend_data['percent_change']:.2f}%\n\n"
                        
                        # Add statistics
                        markdown += "**Statistics:**\n\n"
                        markdown += "| Statistic | Value |\n"
                        markdown += "|-----------|-------|\n"
                        
                        stats = trend_data["statistics"]
                        for stat, value in stats.items():
                            markdown += f"| {stat.capitalize()} | {value:.4f} |\n"
                        
                        markdown += "\n"
                    
                    # Add recent results section
                    markdown += "## Recent Results\n\n"
                    
                    if report_data["recent_results"]:
                        markdown += "| ID | Task ID | Worker ID | Type | Status | Duration |\n"
                        markdown += "|------|---------|-----------|------|--------|----------|\n"
                        
                        for result in report_data["recent_results"]:
                            markdown += f"| {result['id']} | {result['task_id']} | {result['worker_id']} | "
                            markdown += f"{result['type']} | {result['status']} | {result['duration']:.2f} |\n"
                    else:
                        markdown += "No recent results available.\n"
                        
                elif report_type == "anomaly":
                    # Add anomalies section
                    markdown += "## Detected Anomalies\n\n"
                    
                    if report_data["anomalies"]:
                        markdown += "| ID | Result ID | Test Type | Score | Type | Confirmed | Detection Time |\n"
                        markdown += "|------|-----------|-----------|-------|------|-----------|---------------|\n"
                        
                        for anomaly in report_data["anomalies"]:
                            markdown += f"| {anomaly['id']} | {anomaly['result_id']} | {anomaly['test_type']} | "
                            markdown += f"{anomaly['anomaly_score']:.4f} | {anomaly['anomaly_type']} | "
                            markdown += f"{anomaly['is_confirmed']} | {anomaly['detection_time']} |\n"
                        
                        # Add details for top anomalies
                        markdown += "\n### Anomaly Details\n\n"
                        
                        for i, anomaly in enumerate(report_data["anomalies"][:5]):
                            markdown += f"#### Anomaly {i+1} (ID: {anomaly['id']})\n\n"
                            markdown += f"- Result ID: {anomaly['result_id']}\n"
                            markdown += f"- Test Type: {anomaly['test_type']}\n"
                            markdown += f"- Score: {anomaly['anomaly_score']:.4f}\n"
                            markdown += f"- Type: {anomaly['anomaly_type']}\n"
                            markdown += f"- Confirmed: {anomaly['is_confirmed']}\n"
                            markdown += f"- Detection Time: {anomaly['detection_time']}\n\n"
                            
                            if "anomalous_features" in anomaly["details"]:
                                markdown += "**Anomalous Features:**\n\n"
                                markdown += "| Feature | Value | Z-Score | Mean | Std Dev |\n"
                                markdown += "|---------|-------|---------|------|--------|\n"
                                
                                for feature in anomaly["details"]["anomalous_features"]:
                                    markdown += f"| {feature['feature']} | {feature['value']:.4f} | "
                                    markdown += f"{feature['z_score']:.4f} | {feature['mean']:.4f} | "
                                    markdown += f"{feature['std']:.4f} |\n"
                            
                            markdown += "\n"
                    else:
                        markdown += "No anomalies detected.\n"
                        
                elif report_type == "summary":
                    # Add summary section
                    markdown += "## Summary Statistics\n\n"
                    
                    markdown += f"- Total Results: {report_data['total_results']}\n"
                    markdown += f"- Recent Anomalies: {report_data['recent_anomaly_count']}\n\n"
                    
                    # Add status counts section
                    markdown += "### Results by Status\n\n"
                    markdown += "| Status | Count |\n"
                    markdown += "|--------|-------|\n"
                    
                    for status, count in report_data["status_counts"].items():
                        markdown += f"| {status} | {count} |\n"
                    
                    # Add test type counts section
                    markdown += "\n### Results by Test Type\n\n"
                    markdown += "| Test Type | Count |\n"
                    markdown += "|-----------|-------|\n"
                    
                    for test_type, count in report_data["type_counts"].items():
                        markdown += f"| {test_type} | {count} |\n"
                    
                    # Add worker counts section
                    markdown += "\n### Results by Worker\n\n"
                    markdown += "| Worker ID | Count |\n"
                    markdown += "|-----------|-------|\n"
                    
                    for worker_id, count in report_data["worker_counts"].items():
                        markdown += f"| {worker_id} | {count} |\n"
                
                return markdown
                
            elif format == "html":
                # Generate HTML report
                html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Analysis Report - {report_type.capitalize()}</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        h1 {{ color: #333; }}
                        h2 {{ color: #555; margin-top: 30px; }}
                        h3 {{ color: #777; }}
                        table {{ border-collapse: collapse; width: 100%; }}
                        th, td {{ text-align: left; padding: 8px; }}
                        th {{ background-color: #f2f2f2; }}
                        tr:nth-child(even) {{ background-color: #f9f9f9; }}
                        .trend-increasing {{ color: green; }}
                        .trend-decreasing {{ color: red; }}
                        .trend-stable {{ color: blue; }}
                    </style>
                </head>
                <body>
                    <h1>Analysis Report - {report_type.capitalize()}</h1>
                    <p>Generated: {datetime.now().isoformat()}</p>
                """
                
                if report_type == "performance":
                    # Add aggregated metrics section
                    html += "<h2>Aggregated Metrics</h2>"
                    
                    if isinstance(report_data["aggregated_metrics"], dict):
                        html += "<table>"
                        html += "<tr><th>Metric</th><th>Value</th></tr>"
                        
                        for metric, value in report_data["aggregated_metrics"].items():
                            html += f"<tr><td>{metric}</td><td>{value:.4f}</td></tr>"
                        
                        html += "</table>"
                    else:
                        html += "<p>No metrics available.</p>"
                    
                    # Add performance trends section
                    html += "<h2>Performance Trends</h2>"
                    
                    for metric, trend_data in report_data["performance_trends"].items():
                        html += f"<h3>{metric}</h3>"
                        
                        if "error" in trend_data:
                            html += f"<p>Error: {trend_data['error']}</p>"
                            continue
                        
                        trend_class = f"trend-{trend_data['trend']}"
                        
                        html += "<ul>"
                        html += f"<li>Trend: <span class='{trend_class}'>{trend_data['trend']}</span></li>"
                        html += f"<li>Slope: {trend_data['slope']:.4f}</li>"
                        html += f"<li>Percent Change: {trend_data['percent_change']:.2f}%</li>"
                        html += "</ul>"
                        
                        # Add statistics
                        html += "<h4>Statistics</h4>"
                        html += "<table>"
                        html += "<tr><th>Statistic</th><th>Value</th></tr>"
                        
                        stats = trend_data["statistics"]
                        for stat, value in stats.items():
                            html += f"<tr><td>{stat.capitalize()}</td><td>{value:.4f}</td></tr>"
                        
                        html += "</table>"
                        
                        # Add time series plot if visualization is enabled
                        if self.enable_visualization and "time_series" in trend_data and len(trend_data["time_series"]) > 1:
                            # Generate a base64-encoded image
                            try:
                                fig, ax = plt.subplots(figsize=(10, 5))
                                
                                # Extract time series data
                                timestamps = [datetime.fromisoformat(point["timestamp"].replace('Z', '+00:00')) for point in trend_data["time_series"]]
                                values = [point["value"] for point in trend_data["time_series"]]
                                moving_avgs = [point["moving_avg"] if point["moving_avg"] is not None else None for point in trend_data["time_series"]]
                                
                                # Plot raw values
                                ax.plot(timestamps, values, 'o-', label='Value', alpha=0.6)
                                
                                # Plot moving average
                                valid_indices = [i for i, v in enumerate(moving_avgs) if v is not None]
                                if valid_indices:
                                    valid_timestamps = [timestamps[i] for i in valid_indices]
                                    valid_moving_avgs = [moving_avgs[i] for i in valid_indices]
                                    ax.plot(valid_timestamps, valid_moving_avgs, 'r-', label='Moving Avg', linewidth=2)
                                
                                # Add labels and legend
                                ax.set_title(f"{metric} Trend")
                                ax.set_xlabel('Time')
                                ax.set_ylabel('Value')
                                ax.legend()
                                
                                # Rotate x-axis labels for better readability
                                plt.xticks(rotation=45)
                                
                                # Adjust layout
                                plt.tight_layout()
                                
                                # Save as base64
                                import io
                                import base64
                                buf = io.BytesIO()
                                plt.savefig(buf, format='png')
                                buf.seek(0)
                                img_str = base64.b64encode(buf.read()).decode('utf-8')
                                
                                # Add image to HTML
                                html += f"<h4>Time Series Plot</h4>"
                                html += f"<img src='data:image/png;base64,{img_str}' alt='{metric} Trend' />"
                                
                                # Close the figure to free memory
                                plt.close(fig)
                                
                            except Exception as e:
                                logger.error(f"Error generating time series plot: {e}")
                    
                    # Add recent results section
                    html += "<h2>Recent Results</h2>"
                    
                    if report_data["recent_results"]:
                        html += "<table>"
                        html += "<tr><th>ID</th><th>Task ID</th><th>Worker ID</th><th>Type</th><th>Status</th><th>Duration</th></tr>"
                        
                        for result in report_data["recent_results"]:
                            html += "<tr>"
                            html += f"<td>{result['id']}</td>"
                            html += f"<td>{result['task_id']}</td>"
                            html += f"<td>{result['worker_id']}</td>"
                            html += f"<td>{result['type']}</td>"
                            html += f"<td>{result['status']}</td>"
                            html += f"<td>{result['duration']:.2f}</td>"
                            html += "</tr>"
                        
                        html += "</table>"
                    else:
                        html += "<p>No recent results available.</p>"
                        
                elif report_type == "anomaly":
                    # Add anomalies section
                    html += "<h2>Detected Anomalies</h2>"
                    
                    if report_data["anomalies"]:
                        html += "<table>"
                        html += "<tr><th>ID</th><th>Result ID</th><th>Test Type</th><th>Score</th><th>Type</th><th>Confirmed</th><th>Detection Time</th></tr>"
                        
                        for anomaly in report_data["anomalies"]:
                            html += "<tr>"
                            html += f"<td>{anomaly['id']}</td>"
                            html += f"<td>{anomaly['result_id']}</td>"
                            html += f"<td>{anomaly['test_type']}</td>"
                            html += f"<td>{anomaly['anomaly_score']:.4f}</td>"
                            html += f"<td>{anomaly['anomaly_type']}</td>"
                            html += f"<td>{anomaly['is_confirmed']}</td>"
                            html += f"<td>{anomaly['detection_time']}</td>"
                            html += "</tr>"
                        
                        html += "</table>"
                        
                        # Add details for top anomalies
                        html += "<h2>Anomaly Details</h2>"
                        
                        for i, anomaly in enumerate(report_data["anomalies"][:5]):
                            html += f"<h3>Anomaly {i+1} (ID: {anomaly['id']})</h3>"
                            html += "<ul>"
                            html += f"<li>Result ID: {anomaly['result_id']}</li>"
                            html += f"<li>Test Type: {anomaly['test_type']}</li>"
                            html += f"<li>Score: {anomaly['anomaly_score']:.4f}</li>"
                            html += f"<li>Type: {anomaly['anomaly_type']}</li>"
                            html += f"<li>Confirmed: {anomaly['is_confirmed']}</li>"
                            html += f"<li>Detection Time: {anomaly['detection_time']}</li>"
                            html += "</ul>"
                            
                            if "anomalous_features" in anomaly["details"]:
                                html += "<h4>Anomalous Features</h4>"
                                html += "<table>"
                                html += "<tr><th>Feature</th><th>Value</th><th>Z-Score</th><th>Mean</th><th>Std Dev</th></tr>"
                                
                                for feature in anomaly["details"]["anomalous_features"]:
                                    html += "<tr>"
                                    html += f"<td>{feature['feature']}</td>"
                                    html += f"<td>{feature['value']:.4f}</td>"
                                    html += f"<td>{feature['z_score']:.4f}</td>"
                                    html += f"<td>{feature['mean']:.4f}</td>"
                                    html += f"<td>{feature['std']:.4f}</td>"
                                    html += "</tr>"
                                
                                html += "</table>"
                    else:
                        html += "<p>No anomalies detected.</p>"
                        
                elif report_type == "summary":
                    # Add summary section
                    html += "<h2>Summary Statistics</h2>"
                    
                    html += "<ul>"
                    html += f"<li>Total Results: {report_data['total_results']}</li>"
                    html += f"<li>Recent Anomalies: {report_data['recent_anomaly_count']}</li>"
                    html += "</ul>"
                    
                    # Add status counts section
                    html += "<h3>Results by Status</h3>"
                    html += "<table>"
                    html += "<tr><th>Status</th><th>Count</th></tr>"
                    
                    for status, count in report_data["status_counts"].items():
                        html += f"<tr><td>{status}</td><td>{count}</td></tr>"
                    
                    html += "</table>"
                    
                    # Add test type counts section
                    html += "<h3>Results by Test Type</h3>"
                    html += "<table>"
                    html += "<tr><th>Test Type</th><th>Count</th></tr>"
                    
                    for test_type, count in report_data["type_counts"].items():
                        html += f"<tr><td>{test_type}</td><td>{count}</td></tr>"
                    
                    html += "</table>"
                    
                    # Add worker counts section
                    html += "<h3>Results by Worker</h3>"
                    html += "<table>"
                    html += "<tr><th>Worker ID</th><th>Count</th></tr>"
                    
                    for worker_id, count in report_data["worker_counts"].items():
                        html += f"<tr><td>{worker_id}</td><td>{count}</td></tr>"
                    
                    html += "</table>"
                    
                    # Add visualizations if enabled
                    if self.enable_visualization:
                        try:
                            # Create a pie chart for test types
                            fig, ax = plt.subplots(figsize=(8, 6))
                            types = list(report_data["type_counts"].keys())
                            counts = list(report_data["type_counts"].values())
                            ax.pie(counts, labels=types, autopct='%1.1f%%', startangle=90)
                            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                            ax.set_title('Results by Test Type')
                            
                            # Save as base64
                            import io
                            import base64
                            buf = io.BytesIO()
                            plt.savefig(buf, format='png')
                            buf.seek(0)
                            img_str = base64.b64encode(buf.read()).decode('utf-8')
                            
                            # Add image to HTML
                            html += "<h3>Test Type Distribution</h3>"
                            html += f"<img src='data:image/png;base64,{img_str}' alt='Test Type Distribution' />"
                            
                            # Close the figure to free memory
                            plt.close(fig)
                            
                        except Exception as e:
                            logger.error(f"Error generating visualization: {e}")
                
                html += """
                </body>
                </html>
                """
                
                return html
                
            else:
                logger.warning(f"Unknown report format: {format}")
                return json.dumps(report_data, indent=2)
                
        except Exception as e:
            logger.error(f"Error generating analysis report: {e}")
            return f"Error generating report: {str(e)}"
    
    def save_report(self, report_name: str, report_type: str, 
                   filter_criteria: Dict[str, Any] = None,
                   format: str = "json") -> int:
        """
        Generate and save an analysis report.
        
        Args:
            report_name: Name for the report
            report_type: Type of report (performance, anomaly, etc.)
            filter_criteria: Filter criteria for results
            format: Report format (json, markdown, html)
            
        Returns:
            Report ID
        """
        if not self.db:
            logger.warning("No database connection available. Report not saved.")
            return -1
        
        try:
            # Generate report
            report_data = self.generate_analysis_report(
                filter_criteria=filter_criteria,
                report_type=report_type,
                format="json"  # Always store JSON in the database
            )
            
            # Store in database
            self.db.execute(
                """
                INSERT INTO analysis_reports
                (report_name, report_type, filter_criteria, report_data, created_at)
                VALUES (?, ?, ?, ?, ?)
                RETURNING id
                """,
                (report_name, report_type, json.dumps(filter_criteria) if filter_criteria else None,
                 report_data, datetime.now())
            )
            
            report_id = self.db.fetchone()[0]
            
            logger.info(f"Saved analysis report {report_id}: {report_name}")
            
            return report_id
            
        except Exception as e:
            logger.error(f"Error saving analysis report: {e}")
            return -1
    
    def get_report(self, report_id: int, format: str = None) -> str:
        """
        Get a saved analysis report.
        
        Args:
            report_id: Report ID to retrieve
            format: Report format override (json, markdown, html)
            
        Returns:
            Analysis report
        """
        if not self.db:
            logger.warning("No database connection available.")
            return ""
        
        try:
            # Query analysis_reports table
            result = self.db.execute(
                """
                SELECT report_name, report_type, filter_criteria, report_data, created_at
                FROM analysis_reports
                WHERE id = ?
                """,
                (report_id,)
            ).fetchone()
            
            if not result:
                logger.warning(f"No report found with ID {report_id}")
                return ""
            
            report_name, report_type, filter_criteria, report_data, created_at = result
            
            # If format is specified, regenerate the report in the requested format
            if format and format != "json":
                try:
                    # Parse the stored JSON report data
                    report_dict = json.loads(report_data)
                    
                    # Parse filter criteria
                    filter_dict = json.loads(filter_criteria) if filter_criteria else None
                    
                    # Generate report in requested format
                    return self.generate_analysis_report(
                        filter_criteria=filter_dict,
                        report_type=report_type,
                        format=format
                    )
                except json.JSONDecodeError:
                    logger.error(f"Error parsing stored report data: {report_data}")
                    return report_data
            
            return report_data
            
        except Exception as e:
            logger.error(f"Error retrieving analysis report: {e}")
            return ""
    
    def cleanup_old_data(self, days: int = 30) -> int:
        """
        Clean up old data from the database.
        
        Args:
            days: Number of days to keep
            
        Returns:
            Number of deleted records
        """
        if not self.db:
            logger.warning("No database connection available.")
            return 0
        
        try:
            # Calculate cutoff date
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Delete old test results
            self.db.execute(
                """
                DELETE FROM test_results
                WHERE timestamp < ?
                """,
                (cutoff_date,)
            )
            
            # Get number of affected rows
            deleted_count = self.db.execute("SELECT changes()").fetchone()[0]
            
            logger.info(f"Cleaned up {deleted_count} old test results (older than {days} days)")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            return 0
    
    def analyze_performance_regression(self, metric_name: str = None, baseline_period: str = "7d",
                                 comparison_period: str = "1d", filter_criteria: Dict[str, Any] = None):
        """
        Detect performance regression for specified metrics.
        
        Args:
            metric_name: Name of the metric to analyze (None for all key metrics)
            baseline_period: Period for baseline (e.g., "7d" for 7 days)
            comparison_period: Period for comparison (e.g., "1d" for 1 day)
            filter_criteria: Additional filter criteria
            
        Returns:
            Performance regression analysis
        """
        if not hasattr(self, 'performance_analyzer') or not self.performance_analyzer:
            logger.warning("Performance Analyzer not available. Cannot analyze performance regression.")
            return {}
        
        return self.performance_analyzer.detect_performance_regression(
            metric_name=metric_name,
            baseline_period=baseline_period,
            comparison_period=comparison_period,
            filter_criteria=filter_criteria
        )
    
    def compare_hardware_performance(self, metrics: List[str] = None, test_type: str = None,
                                    time_period: str = "30d"):
        """
        Compare performance across different hardware profiles.
        
        Args:
            metrics: List of metrics to compare (None for all key metrics)
            test_type: Type of test to analyze
            time_period: Time period for analysis (e.g., "30d" for 30 days)
            
        Returns:
            Hardware performance comparison results
        """
        if not hasattr(self, 'performance_analyzer') or not self.performance_analyzer:
            logger.warning("Performance Analyzer not available. Cannot compare hardware performance.")
            return {}
        
        return self.performance_analyzer.compare_hardware_performance(
            metrics=metrics,
            test_type=test_type,
            time_period=time_period
        )
    
    def analyze_resource_efficiency(self, test_type: str = None, time_period: str = "30d"):
        """
        Analyze resource efficiency metrics.
        
        Args:
            test_type: Type of test to analyze
            time_period: Time period for analysis (e.g., "30d" for 30 days)
            
        Returns:
            Resource efficiency analysis results
        """
        if not hasattr(self, 'performance_analyzer') or not self.performance_analyzer:
            logger.warning("Performance Analyzer not available. Cannot analyze resource efficiency.")
            return {}
        
        return self.performance_analyzer.analyze_resource_efficiency(
            test_type=test_type,
            time_period=time_period
        )
    
    def analyze_performance_over_time(self, metric_name: str, grouping: str = "day",
                                     test_type: str = None, time_period: str = "90d"):
        """
        Analyze performance trends over time with advanced regression analysis.
        
        Args:
            metric_name: Metric to analyze
            grouping: Time grouping (day, week, month)
            test_type: Type of test to analyze
            time_period: Time period for analysis (e.g., "90d" for 90 days)
            
        Returns:
            Time-based performance analysis results
        """
        if not hasattr(self, 'performance_analyzer') or not self.performance_analyzer:
            logger.warning("Performance Analyzer not available. Cannot analyze performance over time.")
            return {}
        
        return self.performance_analyzer.analyze_performance_over_time(
            metric_name=metric_name,
            grouping=grouping,
            test_type=test_type,
            time_period=time_period
        )
    
    def generate_performance_report(self, report_type: str = "comprehensive",
                                  filter_criteria: Dict[str, Any] = None,
                                  format: str = "markdown", time_period: str = "30d"):
        """
        Generate a comprehensive performance report.
        
        Args:
            report_type: Type of report (comprehensive, regression, hardware_comparison, efficiency, time_analysis)
            filter_criteria: Filter criteria for the report
            format: Report format (markdown, html, json)
            time_period: Time period for analysis (e.g., "30d" for 30 days)
            
        Returns:
            Performance report in the specified format
        """
        if not hasattr(self, 'performance_analyzer') or not self.performance_analyzer:
            logger.warning("Performance Analyzer not available. Cannot generate performance report.")
            return f"# Performance Report\n\nPerformance Analyzer module is not available. Cannot generate report."
        
        return self.performance_analyzer.generate_performance_report(
            report_type=report_type,
            filter_criteria=filter_criteria,
            format=format,
            time_period=time_period
        )
    
    def close(self):
        """Close the database connection."""
        if self.db:
            self.db.close()
            self.db = None
            logger.info("Database connection closed")


if __name__ == "__main__":
    # Example usage
    aggregator = ResultAggregatorService(db_path="./test_db.duckdb")
    
    # Store a test result
    result_id = aggregator.store_result({
        "task_id": "example_task_1",
        "worker_id": "worker_1",
        "type": "benchmark",
        "status": "completed",
        "duration": 10.5,
        "metrics": {
            "throughput": 120.5,
            "latency": 5.2,
            "memory_usage": 1024.0
        },
        "details": {
            "model": "example_model",
            "batch_size": 8,
            "precision": "fp16"
        }
    })
    
    print(f"Stored result with ID: {result_id}")
    
    # Get the result
    result = aggregator.get_result(result_id)
    print(f"Retrieved result: {result}")
    
    # Generate a report
    report = aggregator.generate_analysis_report(format="markdown")
    print(f"Generated report: {report}")
    
    # Close the connection
    aggregator.close()