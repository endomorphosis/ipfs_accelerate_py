#!/usr/bin/env python3
"""
DuckDB API Metrics Repository

This module implements a repository for storing and retrieving API performance metrics
using DuckDB as the storage backend. It provides methods for handling historical data,
predictions, anomalies, and recommendations.
"""

import os
import sys
import logging
import json
import datetime
import uuid
from typing import Dict, List, Any, Optional, Union, Tuple, Set

try:
    import duckdb
    import pandas as pd
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("duckdb_api_metrics")


class DuckDBAPIMetricsRepository:
    """
    DuckDB-based repository for API performance metrics.
    
    This class provides methods for storing and retrieving API performance metrics
    including historical data, predictions, anomalies, and recommendations.
    """
    
    def __init__(
        self, 
        db_path: str = "api_metrics.duckdb",
        create_if_missing: bool = True,
        config: Dict[str, Any] = None
    ):
        """
        Initialize the DuckDBAPIMetricsRepository.
        
        Args:
            db_path: Path to the DuckDB database file
            create_if_missing: Whether to create the database if it doesn't exist
            config: Additional configuration for the repository
        """
        if not DUCKDB_AVAILABLE:
            raise ImportError("DuckDB is required for DuckDBAPIMetricsRepository. Install with: pip install duckdb pandas")
            
        self.db_path = db_path
        self.config = config or {}
        
        # Create parent directories if needed
        if create_if_missing:
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        
        # Connect to DuckDB
        try:
            self.conn = duckdb.connect(db_path)
            logger.info(f"Connected to DuckDB database at {db_path}")
            
            # Initialize schema
            self._initialize_tables()
        except Exception as e:
            logger.error(f"Error connecting to DuckDB database at {db_path}: {e}")
            raise
    
    def _initialize_tables(self) -> None:
        """Create tables for API metrics if they don't exist."""
        logger.info("Initializing API metrics tables")
        
        try:
            # Create historical_metrics table
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS historical_metrics (
                id VARCHAR PRIMARY KEY,
                api_name VARCHAR NOT NULL,
                metric_type VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                value DOUBLE NOT NULL,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            # Create predictions table
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id VARCHAR PRIMARY KEY,
                api_name VARCHAR NOT NULL,
                metric_type VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                value DOUBLE NOT NULL,
                lower_bound DOUBLE,
                upper_bound DOUBLE,
                confidence DOUBLE,
                model_type VARCHAR,
                prediction_timestamp TIMESTAMP,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            # Create anomalies table
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS anomalies (
                id VARCHAR PRIMARY KEY,
                api_name VARCHAR NOT NULL,
                metric_type VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                value DOUBLE NOT NULL,
                anomaly_type VARCHAR NOT NULL,
                confidence DOUBLE NOT NULL,
                severity VARCHAR NOT NULL,
                description VARCHAR,
                detection_timestamp TIMESTAMP,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            # Create recommendations table
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS recommendations (
                id VARCHAR PRIMARY KEY,
                api_name VARCHAR NOT NULL,
                title VARCHAR NOT NULL,
                description VARCHAR NOT NULL,
                impact DOUBLE NOT NULL,
                effort VARCHAR NOT NULL,
                implementation_time VARCHAR,
                roi_period VARCHAR,
                status VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            # Create comparative_metrics table for storing aggregate comparisons
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS comparative_metrics (
                id VARCHAR PRIMARY KEY,
                metric_type VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                api_values JSON NOT NULL,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            # Create indices for faster querying
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_historical_metrics_api_name ON historical_metrics(api_name)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_historical_metrics_metric_type ON historical_metrics(metric_type)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_historical_metrics_timestamp ON historical_metrics(timestamp)")
            
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_api_name ON predictions(api_name)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_metric_type ON predictions(metric_type)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp)")
            
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_anomalies_api_name ON anomalies(api_name)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_anomalies_metric_type ON anomalies(metric_type)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_anomalies_timestamp ON anomalies(timestamp)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_anomalies_severity ON anomalies(severity)")
            
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_recommendations_api_name ON recommendations(api_name)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_recommendations_impact ON recommendations(impact)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_recommendations_status ON recommendations(status)")
            
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_comparative_metrics_metric_type ON comparative_metrics(metric_type)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_comparative_metrics_timestamp ON comparative_metrics(timestamp)")
            
            logger.info("API metrics tables initialized")
            
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise
    
    # Historical metrics methods
    
    def save_historical_metric(
        self,
        api_name: str,
        metric_type: str,
        timestamp: Union[str, datetime.datetime],
        value: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a historical metric.
        
        Args:
            api_name: Name of the API provider
            metric_type: Type of metric (latency, cost, etc.)
            timestamp: Timestamp of the metric
            value: Metric value
            metadata: Additional metadata for the metric
            
        Returns:
            ID of the saved metric
        """
        try:
            # Convert timestamp to string if it's a datetime
            if isinstance(timestamp, datetime.datetime):
                timestamp = timestamp.isoformat()
            
            # Generate ID if not provided
            metric_id = str(uuid.uuid4())
            
            # Convert metadata to JSON
            metadata_json = json.dumps(metadata or {})
            
            # Insert historical metric
            self.conn.execute(
                """
                INSERT INTO historical_metrics (
                    id, api_name, metric_type, timestamp, value, metadata
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [metric_id, api_name, metric_type, timestamp, value, metadata_json]
            )
            
            return metric_id
            
        except Exception as e:
            logger.error(f"Error saving historical metric: {e}")
            raise
    
    def save_historical_metrics_batch(
        self,
        metrics: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Save multiple historical metrics in a batch.
        
        Args:
            metrics: List of metric dictionaries, each containing:
                    api_name, metric_type, timestamp, value, and optional metadata
                    
        Returns:
            List of IDs for the saved metrics
        """
        try:
            metric_ids = []
            
            self.conn.execute("BEGIN TRANSACTION")
            
            for metric in metrics:
                api_name = metric['api_name']
                metric_type = metric['metric_type']
                timestamp = metric['timestamp']
                value = metric['value']
                metadata = metric.get('metadata', {})
                
                # Convert timestamp to string if it's a datetime
                if isinstance(timestamp, datetime.datetime):
                    timestamp = timestamp.isoformat()
                
                # Generate ID if not provided
                metric_id = str(uuid.uuid4())
                metric_ids.append(metric_id)
                
                # Convert metadata to JSON
                metadata_json = json.dumps(metadata or {})
                
                # Insert historical metric
                self.conn.execute(
                    """
                    INSERT INTO historical_metrics (
                        id, api_name, metric_type, timestamp, value, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [metric_id, api_name, metric_type, timestamp, value, metadata_json]
                )
            
            self.conn.execute("COMMIT")
            
            return metric_ids
            
        except Exception as e:
            self.conn.execute("ROLLBACK")
            logger.error(f"Error saving historical metrics batch: {e}")
            raise
    
    def get_historical_metrics(
        self,
        api_name: Optional[str] = None,
        metric_type: Optional[str] = None,
        start_time: Optional[Union[str, datetime.datetime]] = None,
        end_time: Optional[Union[str, datetime.datetime]] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Get historical metrics with optional filtering.
        
        Args:
            api_name: Filter by API provider name
            metric_type: Filter by metric type
            start_time: Filter by start time
            end_time: Filter by end time
            limit: Maximum number of metrics to return
            
        Returns:
            List of historical metrics
        """
        try:
            # Build query conditions
            conditions = []
            params = []
            
            if api_name:
                conditions.append("api_name = ?")
                params.append(api_name)
            
            if metric_type:
                conditions.append("metric_type = ?")
                params.append(metric_type)
            
            if start_time:
                if isinstance(start_time, datetime.datetime):
                    start_time = start_time.isoformat()
                conditions.append("timestamp >= ?")
                params.append(start_time)
            
            if end_time:
                if isinstance(end_time, datetime.datetime):
                    end_time = end_time.isoformat()
                conditions.append("timestamp <= ?")
                params.append(end_time)
            
            # Build and execute query
            query = "SELECT * FROM historical_metrics"
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            result = self.conn.execute(query, params).fetchall()
            
            # Process results
            metrics = []
            for row in result:
                metrics.append({
                    'id': row['id'],
                    'api_name': row['api_name'],
                    'metric_type': row['metric_type'],
                    'timestamp': row['timestamp'],
                    'value': row['value'],
                    'metadata': json.loads(row['metadata']) if row['metadata'] else {},
                    'created_at': row['created_at']
                })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting historical metrics: {e}")
            return []
    
    def get_historical_metrics_formatted(
        self,
        api_name: Optional[str] = None,
        metric_type: Optional[str] = None,
        start_time: Optional[Union[str, datetime.datetime]] = None,
        end_time: Optional[Union[str, datetime.datetime]] = None,
        limit: int = 1000
    ) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """
        Get historical metrics formatted for the API Management UI.
        
        Returns data in the format:
        {
            "metric_type": {
                "api_name": [
                    {"timestamp": "iso-format", "value": numeric_value},
                    ...
                ]
            }
        }
        
        Args:
            api_name: Filter by API provider name
            metric_type: Filter by metric type
            start_time: Filter by start time
            end_time: Filter by end time
            limit: Maximum number of metrics to return
            
        Returns:
            Formatted historical metrics
        """
        try:
            # Get raw metrics
            raw_metrics = self.get_historical_metrics(
                api_name=api_name,
                metric_type=metric_type,
                start_time=start_time,
                end_time=end_time,
                limit=limit
            )
            
            # Format metrics
            formatted_metrics = {}
            
            for metric in raw_metrics:
                m_type = metric['metric_type']
                api = metric['api_name']
                
                if m_type not in formatted_metrics:
                    formatted_metrics[m_type] = {}
                
                if api not in formatted_metrics[m_type]:
                    formatted_metrics[m_type][api] = []
                
                formatted_metrics[m_type][api].append({
                    'timestamp': metric['timestamp'],
                    'value': metric['value']
                })
            
            # Sort by timestamp
            for m_type in formatted_metrics:
                for api in formatted_metrics[m_type]:
                    formatted_metrics[m_type][api].sort(key=lambda x: x['timestamp'])
            
            return formatted_metrics
            
        except Exception as e:
            logger.error(f"Error getting formatted historical metrics: {e}")
            return {}
    
    # Prediction methods
    
    def save_prediction(
        self,
        api_name: str,
        metric_type: str,
        timestamp: Union[str, datetime.datetime],
        value: float,
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
        confidence: Optional[float] = None,
        model_type: Optional[str] = None,
        prediction_timestamp: Optional[Union[str, datetime.datetime]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a prediction.
        
        Args:
            api_name: Name of the API provider
            metric_type: Type of metric (latency, cost, etc.)
            timestamp: Timestamp for the prediction
            value: Predicted value
            lower_bound: Lower bound of prediction interval
            upper_bound: Upper bound of prediction interval
            confidence: Confidence score for the prediction
            model_type: Type of model used for prediction
            prediction_timestamp: When the prediction was made
            metadata: Additional metadata for the prediction
            
        Returns:
            ID of the saved prediction
        """
        try:
            # Convert timestamps to strings if they're datetimes
            if isinstance(timestamp, datetime.datetime):
                timestamp = timestamp.isoformat()
            
            if isinstance(prediction_timestamp, datetime.datetime):
                prediction_timestamp = prediction_timestamp.isoformat()
            elif prediction_timestamp is None:
                prediction_timestamp = datetime.datetime.now().isoformat()
            
            # Generate ID if not provided
            prediction_id = str(uuid.uuid4())
            
            # Convert metadata to JSON
            metadata_json = json.dumps(metadata or {})
            
            # Insert prediction
            self.conn.execute(
                """
                INSERT INTO predictions (
                    id, api_name, metric_type, timestamp, value, 
                    lower_bound, upper_bound, confidence, model_type,
                    prediction_timestamp, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    prediction_id, api_name, metric_type, timestamp, value,
                    lower_bound, upper_bound, confidence, model_type,
                    prediction_timestamp, metadata_json
                ]
            )
            
            return prediction_id
            
        except Exception as e:
            logger.error(f"Error saving prediction: {e}")
            raise
    
    def save_predictions_batch(
        self,
        predictions: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Save multiple predictions in a batch.
        
        Args:
            predictions: List of prediction dictionaries
                    
        Returns:
            List of IDs for the saved predictions
        """
        try:
            prediction_ids = []
            
            self.conn.execute("BEGIN TRANSACTION")
            
            for pred in predictions:
                api_name = pred['api_name']
                metric_type = pred['metric_type']
                timestamp = pred['timestamp']
                value = pred['value']
                lower_bound = pred.get('lower_bound')
                upper_bound = pred.get('upper_bound')
                confidence = pred.get('confidence')
                model_type = pred.get('model_type')
                prediction_timestamp = pred.get('prediction_timestamp')
                metadata = pred.get('metadata', {})
                
                # Convert timestamps to strings if they're datetimes
                if isinstance(timestamp, datetime.datetime):
                    timestamp = timestamp.isoformat()
                
                if isinstance(prediction_timestamp, datetime.datetime):
                    prediction_timestamp = prediction_timestamp.isoformat()
                elif prediction_timestamp is None:
                    prediction_timestamp = datetime.datetime.now().isoformat()
                
                # Generate ID if not provided
                prediction_id = str(uuid.uuid4())
                prediction_ids.append(prediction_id)
                
                # Convert metadata to JSON
                metadata_json = json.dumps(metadata or {})
                
                # Insert prediction
                self.conn.execute(
                    """
                    INSERT INTO predictions (
                        id, api_name, metric_type, timestamp, value, 
                        lower_bound, upper_bound, confidence, model_type,
                        prediction_timestamp, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        prediction_id, api_name, metric_type, timestamp, value,
                        lower_bound, upper_bound, confidence, model_type,
                        prediction_timestamp, metadata_json
                    ]
                )
            
            self.conn.execute("COMMIT")
            
            return prediction_ids
            
        except Exception as e:
            self.conn.execute("ROLLBACK")
            logger.error(f"Error saving predictions batch: {e}")
            raise
    
    def get_predictions(
        self,
        api_name: Optional[str] = None,
        metric_type: Optional[str] = None,
        start_time: Optional[Union[str, datetime.datetime]] = None,
        end_time: Optional[Union[str, datetime.datetime]] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Get predictions with optional filtering.
        
        Args:
            api_name: Filter by API provider name
            metric_type: Filter by metric type
            start_time: Filter by start time
            end_time: Filter by end time
            limit: Maximum number of predictions to return
            
        Returns:
            List of predictions
        """
        try:
            # Build query conditions
            conditions = []
            params = []
            
            if api_name:
                conditions.append("api_name = ?")
                params.append(api_name)
            
            if metric_type:
                conditions.append("metric_type = ?")
                params.append(metric_type)
            
            if start_time:
                if isinstance(start_time, datetime.datetime):
                    start_time = start_time.isoformat()
                conditions.append("timestamp >= ?")
                params.append(start_time)
            
            if end_time:
                if isinstance(end_time, datetime.datetime):
                    end_time = end_time.isoformat()
                conditions.append("timestamp <= ?")
                params.append(end_time)
            
            # Build and execute query
            query = "SELECT * FROM predictions"
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            query += " ORDER BY timestamp ASC LIMIT ?"
            params.append(limit)
            
            result = self.conn.execute(query, params).fetchall()
            
            # Process results
            predictions = []
            for row in result:
                predictions.append({
                    'id': row['id'],
                    'api_name': row['api_name'],
                    'metric_type': row['metric_type'],
                    'timestamp': row['timestamp'],
                    'value': row['value'],
                    'lower_bound': row['lower_bound'],
                    'upper_bound': row['upper_bound'],
                    'confidence': row['confidence'],
                    'model_type': row['model_type'],
                    'prediction_timestamp': row['prediction_timestamp'],
                    'metadata': json.loads(row['metadata']) if row['metadata'] else {},
                    'created_at': row['created_at']
                })
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error getting predictions: {e}")
            return []
    
    def get_predictions_formatted(
        self,
        api_name: Optional[str] = None,
        metric_type: Optional[str] = None,
        start_time: Optional[Union[str, datetime.datetime]] = None,
        end_time: Optional[Union[str, datetime.datetime]] = None,
        limit: int = 1000
    ) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """
        Get predictions formatted for the API Management UI.
        
        Returns data in the format:
        {
            "metric_type": {
                "api_name": [
                    {
                        "timestamp": "iso-format", 
                        "value": numeric_value,
                        "lower_bound": numeric_value,
                        "upper_bound": numeric_value
                    },
                    ...
                ]
            }
        }
        
        Args:
            api_name: Filter by API provider name
            metric_type: Filter by metric type
            start_time: Filter by start time
            end_time: Filter by end time
            limit: Maximum number of predictions to return
            
        Returns:
            Formatted predictions
        """
        try:
            # Get raw predictions
            raw_predictions = self.get_predictions(
                api_name=api_name,
                metric_type=metric_type,
                start_time=start_time,
                end_time=end_time,
                limit=limit
            )
            
            # Format predictions
            formatted_predictions = {}
            
            for pred in raw_predictions:
                m_type = pred['metric_type']
                api = pred['api_name']
                
                if m_type not in formatted_predictions:
                    formatted_predictions[m_type] = {}
                
                if api not in formatted_predictions[m_type]:
                    formatted_predictions[m_type][api] = []
                
                formatted_predictions[m_type][api].append({
                    'timestamp': pred['timestamp'],
                    'value': pred['value'],
                    'lower_bound': pred['lower_bound'],
                    'upper_bound': pred['upper_bound']
                })
            
            # Sort by timestamp
            for m_type in formatted_predictions:
                for api in formatted_predictions[m_type]:
                    formatted_predictions[m_type][api].sort(key=lambda x: x['timestamp'])
            
            return formatted_predictions
            
        except Exception as e:
            logger.error(f"Error getting formatted predictions: {e}")
            return {}
    
    # Anomaly methods
    
    def save_anomaly(
        self,
        api_name: str,
        metric_type: str,
        timestamp: Union[str, datetime.datetime],
        value: float,
        anomaly_type: str,
        confidence: float,
        severity: str,
        description: Optional[str] = None,
        detection_timestamp: Optional[Union[str, datetime.datetime]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save an anomaly.
        
        Args:
            api_name: Name of the API provider
            metric_type: Type of metric (latency, cost, etc.)
            timestamp: Timestamp of the anomaly
            value: Value at the anomaly point
            anomaly_type: Type of anomaly (spike, trend_break, etc.)
            confidence: Confidence score for the anomaly detection
            severity: Severity of the anomaly (low, medium, high, critical)
            description: Description of the anomaly
            detection_timestamp: When the anomaly was detected
            metadata: Additional metadata for the anomaly
            
        Returns:
            ID of the saved anomaly
        """
        try:
            # Convert timestamps to strings if they're datetimes
            if isinstance(timestamp, datetime.datetime):
                timestamp = timestamp.isoformat()
            
            if isinstance(detection_timestamp, datetime.datetime):
                detection_timestamp = detection_timestamp.isoformat()
            elif detection_timestamp is None:
                detection_timestamp = datetime.datetime.now().isoformat()
            
            # Generate ID if not provided
            anomaly_id = str(uuid.uuid4())
            
            # Convert metadata to JSON
            metadata_json = json.dumps(metadata or {})
            
            # Insert anomaly
            self.conn.execute(
                """
                INSERT INTO anomalies (
                    id, api_name, metric_type, timestamp, value, 
                    anomaly_type, confidence, severity, description,
                    detection_timestamp, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    anomaly_id, api_name, metric_type, timestamp, value,
                    anomaly_type, confidence, severity, description,
                    detection_timestamp, metadata_json
                ]
            )
            
            return anomaly_id
            
        except Exception as e:
            logger.error(f"Error saving anomaly: {e}")
            raise
    
    def save_anomalies_batch(
        self,
        anomalies: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Save multiple anomalies in a batch.
        
        Args:
            anomalies: List of anomaly dictionaries
                    
        Returns:
            List of IDs for the saved anomalies
        """
        try:
            anomaly_ids = []
            
            self.conn.execute("BEGIN TRANSACTION")
            
            for anomaly in anomalies:
                api_name = anomaly['api_name']
                metric_type = anomaly['metric_type']
                timestamp = anomaly['timestamp']
                value = anomaly['value']
                anomaly_type = anomaly['type']
                confidence = anomaly['confidence']
                severity = anomaly.get('severity', 'medium')
                description = anomaly.get('description')
                detection_timestamp = anomaly.get('detection_timestamp')
                metadata = anomaly.get('metadata', {})
                
                # Convert timestamps to strings if they're datetimes
                if isinstance(timestamp, datetime.datetime):
                    timestamp = timestamp.isoformat()
                
                if isinstance(detection_timestamp, datetime.datetime):
                    detection_timestamp = detection_timestamp.isoformat()
                elif detection_timestamp is None:
                    detection_timestamp = datetime.datetime.now().isoformat()
                
                # Generate ID if not provided
                anomaly_id = str(uuid.uuid4())
                anomaly_ids.append(anomaly_id)
                
                # Convert metadata to JSON
                metadata_json = json.dumps(metadata or {})
                
                # Insert anomaly
                self.conn.execute(
                    """
                    INSERT INTO anomalies (
                        id, api_name, metric_type, timestamp, value, 
                        anomaly_type, confidence, severity, description,
                        detection_timestamp, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        anomaly_id, api_name, metric_type, timestamp, value,
                        anomaly_type, confidence, severity, description,
                        detection_timestamp, metadata_json
                    ]
                )
            
            self.conn.execute("COMMIT")
            
            return anomaly_ids
            
        except Exception as e:
            self.conn.execute("ROLLBACK")
            logger.error(f"Error saving anomalies batch: {e}")
            raise
    
    def get_anomalies(
        self,
        api_name: Optional[str] = None,
        metric_type: Optional[str] = None,
        anomaly_type: Optional[str] = None,
        severity: Optional[str] = None,
        min_confidence: Optional[float] = None,
        start_time: Optional[Union[str, datetime.datetime]] = None,
        end_time: Optional[Union[str, datetime.datetime]] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Get anomalies with optional filtering.
        
        Args:
            api_name: Filter by API provider name
            metric_type: Filter by metric type
            anomaly_type: Filter by anomaly type
            severity: Filter by severity level
            min_confidence: Minimum confidence score
            start_time: Filter by start time
            end_time: Filter by end time
            limit: Maximum number of anomalies to return
            
        Returns:
            List of anomalies
        """
        try:
            # Build query conditions
            conditions = []
            params = []
            
            if api_name:
                conditions.append("api_name = ?")
                params.append(api_name)
            
            if metric_type:
                conditions.append("metric_type = ?")
                params.append(metric_type)
            
            if anomaly_type:
                conditions.append("anomaly_type = ?")
                params.append(anomaly_type)
            
            if severity:
                conditions.append("severity = ?")
                params.append(severity)
            
            if min_confidence is not None:
                conditions.append("confidence >= ?")
                params.append(min_confidence)
            
            if start_time:
                if isinstance(start_time, datetime.datetime):
                    start_time = start_time.isoformat()
                conditions.append("timestamp >= ?")
                params.append(start_time)
            
            if end_time:
                if isinstance(end_time, datetime.datetime):
                    end_time = end_time.isoformat()
                conditions.append("timestamp <= ?")
                params.append(end_time)
            
            # Build and execute query
            query = "SELECT * FROM anomalies"
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            result = self.conn.execute(query, params).fetchall()
            
            # Process results
            anomalies = []
            for row in result:
                anomalies.append({
                    'id': row['id'],
                    'api_name': row['api_name'],
                    'metric_type': row['metric_type'],
                    'timestamp': row['timestamp'],
                    'value': row['value'],
                    'type': row['anomaly_type'],
                    'confidence': row['confidence'],
                    'severity': row['severity'],
                    'description': row['description'],
                    'detection_timestamp': row['detection_timestamp'],
                    'metadata': json.loads(row['metadata']) if row['metadata'] else {},
                    'created_at': row['created_at']
                })
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error getting anomalies: {e}")
            return []
    
    def get_anomalies_formatted(
        self,
        api_name: Optional[str] = None,
        metric_type: Optional[str] = None,
        anomaly_type: Optional[str] = None,
        severity: Optional[str] = None,
        min_confidence: Optional[float] = None,
        start_time: Optional[Union[str, datetime.datetime]] = None,
        end_time: Optional[Union[str, datetime.datetime]] = None,
        limit: int = 1000
    ) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """
        Get anomalies formatted for the API Management UI.
        
        Returns data in the format:
        {
            "metric_type": {
                "api_name": [
                    {
                        "timestamp": "iso-format",
                        "value": numeric_value,
                        "type": "spike|trend_break|oscillation|seasonal",
                        "confidence": numeric_value,
                        "description": "description_text",
                        "severity": "low|medium|high|critical"
                    },
                    ...
                ]
            }
        }
        
        Args:
            api_name: Filter by API provider name
            metric_type: Filter by metric type
            anomaly_type: Filter by anomaly type
            severity: Filter by severity level
            min_confidence: Minimum confidence score
            start_time: Filter by start time
            end_time: Filter by end time
            limit: Maximum number of anomalies to return
            
        Returns:
            Formatted anomalies
        """
        try:
            # Get raw anomalies
            raw_anomalies = self.get_anomalies(
                api_name=api_name,
                metric_type=metric_type,
                anomaly_type=anomaly_type,
                severity=severity,
                min_confidence=min_confidence,
                start_time=start_time,
                end_time=end_time,
                limit=limit
            )
            
            # Format anomalies
            formatted_anomalies = {}
            
            for anomaly in raw_anomalies:
                m_type = anomaly['metric_type']
                api = anomaly['api_name']
                
                if m_type not in formatted_anomalies:
                    formatted_anomalies[m_type] = {}
                
                if api not in formatted_anomalies[m_type]:
                    formatted_anomalies[m_type][api] = []
                
                formatted_anomalies[m_type][api].append({
                    'timestamp': anomaly['timestamp'],
                    'value': anomaly['value'],
                    'type': anomaly['type'],
                    'confidence': anomaly['confidence'],
                    'description': anomaly['description'],
                    'severity': anomaly['severity']
                })
            
            # Sort by timestamp
            for m_type in formatted_anomalies:
                for api in formatted_anomalies[m_type]:
                    formatted_anomalies[m_type][api].sort(key=lambda x: x['timestamp'])
            
            return formatted_anomalies
            
        except Exception as e:
            logger.error(f"Error getting formatted anomalies: {e}")
            return {}
    
    # Recommendation methods
    
    def save_recommendation(
        self,
        api_name: str,
        title: str,
        description: str,
        impact: float,
        effort: str,
        implementation_time: Optional[str] = None,
        roi_period: Optional[str] = None,
        status: Optional[str] = "New"
    ) -> str:
        """
        Save a recommendation.
        
        Args:
            api_name: Name of the API provider
            title: Title of the recommendation
            description: Description of the recommendation
            impact: Impact score (0-1) of the recommendation
            effort: Effort required (Low, Medium, High)
            implementation_time: Estimated time to implement (Hours, Days, Weeks, Months)
            roi_period: Estimated time to see ROI (Days, Weeks, Months)
            status: Status of the recommendation (New, In Progress, Implemented, Verified)
            
        Returns:
            ID of the saved recommendation
        """
        try:
            # Generate ID if not provided
            recommendation_id = str(uuid.uuid4())
            
            # Insert recommendation
            self.conn.execute(
                """
                INSERT INTO recommendations (
                    id, api_name, title, description, impact, 
                    effort, implementation_time, roi_period, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    recommendation_id, api_name, title, description, impact,
                    effort, implementation_time, roi_period, status
                ]
            )
            
            return recommendation_id
            
        except Exception as e:
            logger.error(f"Error saving recommendation: {e}")
            raise
    
    def save_recommendations_batch(
        self,
        recommendations: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Save multiple recommendations in a batch.
        
        Args:
            recommendations: List of recommendation dictionaries
                    
        Returns:
            List of IDs for the saved recommendations
        """
        try:
            recommendation_ids = []
            
            self.conn.execute("BEGIN TRANSACTION")
            
            for rec in recommendations:
                api_name = rec['api_name']
                title = rec['title']
                description = rec['description']
                impact = rec['impact']
                effort = rec['effort']
                implementation_time = rec.get('implementation_time')
                roi_period = rec.get('roi_period')
                status = rec.get('status', 'New')
                
                # Generate ID if not provided
                recommendation_id = str(uuid.uuid4())
                recommendation_ids.append(recommendation_id)
                
                # Insert recommendation
                self.conn.execute(
                    """
                    INSERT INTO recommendations (
                        id, api_name, title, description, impact, 
                        effort, implementation_time, roi_period, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        recommendation_id, api_name, title, description, impact,
                        effort, implementation_time, roi_period, status
                    ]
                )
            
            self.conn.execute("COMMIT")
            
            return recommendation_ids
            
        except Exception as e:
            self.conn.execute("ROLLBACK")
            logger.error(f"Error saving recommendations batch: {e}")
            raise
    
    def update_recommendation_status(
        self,
        recommendation_id: str,
        new_status: str
    ) -> bool:
        """
        Update the status of a recommendation.
        
        Args:
            recommendation_id: ID of the recommendation
            new_status: New status value
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.conn.execute(
                """
                UPDATE recommendations
                SET status = ?, updated_at = ?
                WHERE id = ?
                """,
                [new_status, datetime.datetime.now().isoformat(), recommendation_id]
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating recommendation status: {e}")
            return False
    
    def get_recommendations(
        self,
        api_name: Optional[str] = None,
        status: Optional[str] = None,
        min_impact: Optional[float] = None,
        effort: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get recommendations with optional filtering.
        
        Args:
            api_name: Filter by API provider name
            status: Filter by status
            min_impact: Minimum impact score
            effort: Filter by effort level
            limit: Maximum number of recommendations to return
            
        Returns:
            List of recommendations
        """
        try:
            # Build query conditions
            conditions = []
            params = []
            
            if api_name:
                conditions.append("api_name = ?")
                params.append(api_name)
            
            if status:
                conditions.append("status = ?")
                params.append(status)
            
            if min_impact is not None:
                conditions.append("impact >= ?")
                params.append(min_impact)
            
            if effort:
                conditions.append("effort = ?")
                params.append(effort)
            
            # Build and execute query
            query = "SELECT * FROM recommendations"
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            query += " ORDER BY impact DESC LIMIT ?"
            params.append(limit)
            
            result = self.conn.execute(query, params).fetchall()
            
            # Process results
            recommendations = []
            for row in result:
                recommendations.append({
                    'id': row['id'],
                    'api_name': row['api_name'],
                    'title': row['title'],
                    'description': row['description'],
                    'impact': row['impact'],
                    'effort': row['effort'],
                    'implementation_time': row['implementation_time'],
                    'roi_period': row['roi_period'],
                    'status': row['status'],
                    'created_at': row['created_at'],
                    'updated_at': row['updated_at']
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return []
    
    def get_recommendations_formatted(
        self,
        api_name: Optional[str] = None,
        status: Optional[str] = None,
        min_impact: Optional[float] = None,
        effort: Optional[str] = None,
        limit: int = 100
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get recommendations formatted for the API Management UI.
        
        Returns data in the format:
        {
            "api_name": [
                {
                    "title": "recommendation_title",
                    "description": "recommendation_text",
                    "impact": numeric_value,
                    "effort": "Low|Medium|High",
                    "implementation_time": "Hours|Days|Weeks",
                    "roi_period": "Days|Weeks|Months",
                    "status": "New|In Progress|Implemented|Verified"
                },
                ...
            ]
        }
        
        Args:
            api_name: Filter by API provider name
            status: Filter by status
            min_impact: Minimum impact score
            effort: Filter by effort level
            limit: Maximum number of recommendations to return
            
        Returns:
            Formatted recommendations
        """
        try:
            # Get raw recommendations
            raw_recommendations = self.get_recommendations(
                api_name=api_name,
                status=status,
                min_impact=min_impact,
                effort=effort,
                limit=limit
            )
            
            # Format recommendations
            formatted_recommendations = {}
            
            for rec in raw_recommendations:
                api = rec['api_name']
                
                if api not in formatted_recommendations:
                    formatted_recommendations[api] = []
                
                formatted_recommendations[api].append({
                    'title': rec['title'],
                    'description': rec['description'],
                    'impact': rec['impact'],
                    'effort': rec['effort'],
                    'implementation_time': rec['implementation_time'],
                    'roi_period': rec['roi_period'],
                    'status': rec['status']
                })
            
            # Sort by impact (descending)
            for api in formatted_recommendations:
                formatted_recommendations[api].sort(key=lambda x: x['impact'], reverse=True)
            
            return formatted_recommendations
            
        except Exception as e:
            logger.error(f"Error getting formatted recommendations: {e}")
            return {}
    
    # Comparative metrics methods
    
    def save_comparative_metric(
        self,
        metric_type: str,
        timestamp: Union[str, datetime.datetime],
        api_values: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a comparative metric for multiple APIs.
        
        Args:
            metric_type: Type of metric (latency, cost, etc.)
            timestamp: Timestamp of the metric
            api_values: Dictionary mapping API names to metric values
            metadata: Additional metadata for the metric
            
        Returns:
            ID of the saved comparative metric
        """
        try:
            # Convert timestamp to string if it's a datetime
            if isinstance(timestamp, datetime.datetime):
                timestamp = timestamp.isoformat()
            
            # Generate ID if not provided
            metric_id = str(uuid.uuid4())
            
            # Convert API values and metadata to JSON
            api_values_json = json.dumps(api_values)
            metadata_json = json.dumps(metadata or {})
            
            # Insert comparative metric
            self.conn.execute(
                """
                INSERT INTO comparative_metrics (
                    id, metric_type, timestamp, api_values, metadata
                ) VALUES (?, ?, ?, ?, ?)
                """,
                [metric_id, metric_type, timestamp, api_values_json, metadata_json]
            )
            
            return metric_id
            
        except Exception as e:
            logger.error(f"Error saving comparative metric: {e}")
            raise
    
    def save_comparative_metrics_batch(
        self,
        metrics: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Save multiple comparative metrics in a batch.
        
        Args:
            metrics: List of comparative metric dictionaries
                    
        Returns:
            List of IDs for the saved metrics
        """
        try:
            metric_ids = []
            
            self.conn.execute("BEGIN TRANSACTION")
            
            for metric in metrics:
                metric_type = metric['metric_type']
                timestamp = metric['timestamp']
                api_values = metric['values']
                metadata = metric.get('metadata', {})
                
                # Convert timestamp to string if it's a datetime
                if isinstance(timestamp, datetime.datetime):
                    timestamp = timestamp.isoformat()
                
                # Generate ID if not provided
                metric_id = str(uuid.uuid4())
                metric_ids.append(metric_id)
                
                # Convert API values and metadata to JSON
                api_values_json = json.dumps(api_values)
                metadata_json = json.dumps(metadata or {})
                
                # Insert comparative metric
                self.conn.execute(
                    """
                    INSERT INTO comparative_metrics (
                        id, metric_type, timestamp, api_values, metadata
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    [metric_id, metric_type, timestamp, api_values_json, metadata_json]
                )
            
            self.conn.execute("COMMIT")
            
            return metric_ids
            
        except Exception as e:
            self.conn.execute("ROLLBACK")
            logger.error(f"Error saving comparative metrics batch: {e}")
            raise
    
    def get_comparative_metrics(
        self,
        metric_type: Optional[str] = None,
        start_time: Optional[Union[str, datetime.datetime]] = None,
        end_time: Optional[Union[str, datetime.datetime]] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Get comparative metrics with optional filtering.
        
        Args:
            metric_type: Filter by metric type
            start_time: Filter by start time
            end_time: Filter by end time
            limit: Maximum number of metrics to return
            
        Returns:
            List of comparative metrics
        """
        try:
            # Build query conditions
            conditions = []
            params = []
            
            if metric_type:
                conditions.append("metric_type = ?")
                params.append(metric_type)
            
            if start_time:
                if isinstance(start_time, datetime.datetime):
                    start_time = start_time.isoformat()
                conditions.append("timestamp >= ?")
                params.append(start_time)
            
            if end_time:
                if isinstance(end_time, datetime.datetime):
                    end_time = end_time.isoformat()
                conditions.append("timestamp <= ?")
                params.append(end_time)
            
            # Build and execute query
            query = "SELECT * FROM comparative_metrics"
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            query += " ORDER BY timestamp ASC LIMIT ?"
            params.append(limit)
            
            result = self.conn.execute(query, params).fetchall()
            
            # Process results
            metrics = []
            for row in result:
                metrics.append({
                    'id': row['id'],
                    'metric_type': row['metric_type'],
                    'timestamp': row['timestamp'],
                    'values': json.loads(row['api_values']),
                    'metadata': json.loads(row['metadata']) if row['metadata'] else {},
                    'created_at': row['created_at']
                })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting comparative metrics: {e}")
            return []
    
    def get_comparative_metrics_formatted(
        self,
        metric_type: Optional[str] = None,
        start_time: Optional[Union[str, datetime.datetime]] = None,
        end_time: Optional[Union[str, datetime.datetime]] = None,
        limit: int = 1000
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get comparative metrics formatted for the API Management UI.
        
        Returns data in the format:
        {
            "metric_type": [
                {
                    "timestamp": "iso-format",
                    "values": {
                        "api_name_1": numeric_value,
                        "api_name_2": numeric_value,
                        ...
                    }
                },
                ...
            ]
        }
        
        Args:
            metric_type: Filter by metric type
            start_time: Filter by start time
            end_time: Filter by end time
            limit: Maximum number of metrics to return
            
        Returns:
            Formatted comparative metrics
        """
        try:
            # Get raw comparative metrics
            raw_metrics = self.get_comparative_metrics(
                metric_type=metric_type,
                start_time=start_time,
                end_time=end_time,
                limit=limit
            )
            
            # Format comparative metrics
            formatted_metrics = {}
            
            for metric in raw_metrics:
                m_type = metric['metric_type']
                
                if m_type not in formatted_metrics:
                    formatted_metrics[m_type] = []
                
                formatted_metrics[m_type].append({
                    'timestamp': metric['timestamp'],
                    'values': metric['values']
                })
            
            # Sort by timestamp
            for m_type in formatted_metrics:
                formatted_metrics[m_type].sort(key=lambda x: x['timestamp'])
            
            return formatted_metrics
            
        except Exception as e:
            logger.error(f"Error getting formatted comparative metrics: {e}")
            return {}
    
    # Complete data export for API Management UI
    
    def export_all_data_for_ui(
        self,
        api_names: Optional[List[str]] = None,
        metric_types: Optional[List[str]] = None,
        start_time: Optional[Union[str, datetime.datetime]] = None,
        end_time: Optional[Union[str, datetime.datetime]] = None,
        limit_per_section: int = 1000
    ) -> Dict[str, Any]:
        """
        Export all data formatted for the API Management UI.
        
        Returns complete data structure with historical data, predictions,
        anomalies, recommendations, and comparative metrics.
        
        Args:
            api_names: List of API names to include (all if None)
            metric_types: List of metric types to include (all if None)
            start_time: Filter by start time
            end_time: Filter by end time
            limit_per_section: Maximum items per section
            
        Returns:
            Complete data structure for API Management UI
        """
        try:
            result = {
                'historical_data': {},
                'predictions': {},
                'anomalies': {},
                'recommendations': {},
                'comparative_data': {}
            }
            
            # Get historical data
            historical_data = self.get_historical_metrics_formatted(
                api_name=api_names[0] if api_names and len(api_names) == 1 else None,
                metric_type=metric_types[0] if metric_types and len(metric_types) == 1 else None,
                start_time=start_time,
                end_time=end_time,
                limit=limit_per_section
            )
            
            # Filter by requested API names and metric types
            if api_names or metric_types:
                filtered_historical = {}
                for m_type, apis in historical_data.items():
                    if not metric_types or m_type in metric_types:
                        filtered_historical[m_type] = {}
                        for api, data in apis.items():
                            if not api_names or api in api_names:
                                filtered_historical[m_type][api] = data
                historical_data = filtered_historical
            
            result['historical_data'] = historical_data
            
            # Get predictions
            predictions = self.get_predictions_formatted(
                api_name=api_names[0] if api_names and len(api_names) == 1 else None,
                metric_type=metric_types[0] if metric_types and len(metric_types) == 1 else None,
                start_time=start_time,
                end_time=end_time,
                limit=limit_per_section
            )
            
            # Filter by requested API names and metric types
            if api_names or metric_types:
                filtered_predictions = {}
                for m_type, apis in predictions.items():
                    if not metric_types or m_type in metric_types:
                        filtered_predictions[m_type] = {}
                        for api, data in apis.items():
                            if not api_names or api in api_names:
                                filtered_predictions[m_type][api] = data
                predictions = filtered_predictions
            
            result['predictions'] = predictions
            
            # Get anomalies
            anomalies = self.get_anomalies_formatted(
                api_name=api_names[0] if api_names and len(api_names) == 1 else None,
                metric_type=metric_types[0] if metric_types and len(metric_types) == 1 else None,
                start_time=start_time,
                end_time=end_time,
                limit=limit_per_section
            )
            
            # Filter by requested API names and metric types
            if api_names or metric_types:
                filtered_anomalies = {}
                for m_type, apis in anomalies.items():
                    if not metric_types or m_type in metric_types:
                        filtered_anomalies[m_type] = {}
                        for api, data in apis.items():
                            if not api_names or api in api_names:
                                filtered_anomalies[m_type][api] = data
                anomalies = filtered_anomalies
            
            result['anomalies'] = anomalies
            
            # Get recommendations
            recommendations = self.get_recommendations_formatted(
                api_name=api_names[0] if api_names and len(api_names) == 1 else None,
                limit=limit_per_section
            )
            
            # Filter by requested API names
            if api_names:
                filtered_recommendations = {}
                for api, data in recommendations.items():
                    if api in api_names:
                        filtered_recommendations[api] = data
                recommendations = filtered_recommendations
            
            result['recommendations'] = recommendations
            
            # Get comparative metrics
            comparative_data = self.get_comparative_metrics_formatted(
                metric_type=metric_types[0] if metric_types and len(metric_types) == 1 else None,
                start_time=start_time,
                end_time=end_time,
                limit=limit_per_section
            )
            
            # Filter by requested metric types
            if metric_types:
                filtered_comparative = {}
                for m_type, data in comparative_data.items():
                    if m_type in metric_types:
                        filtered_comparative[m_type] = data
                comparative_data = filtered_comparative
            
            result['comparative_data'] = comparative_data
            
            return result
            
        except Exception as e:
            logger.error(f"Error exporting all data for UI: {e}")
            return {
                'historical_data': {},
                'predictions': {},
                'anomalies': {},
                'recommendations': {},
                'comparative_data': {}
            }
    
    # Database management methods
    
    def get_available_api_names(self) -> List[str]:
        """
        Get all available API names in the database.
        
        Returns:
            List of API names
        """
        try:
            # Query unique API names from historical metrics
            result = self.conn.execute(
                "SELECT DISTINCT api_name FROM historical_metrics ORDER BY api_name"
            ).fetchall()
            
            return [row['api_name'] for row in result]
            
        except Exception as e:
            logger.error(f"Error getting available API names: {e}")
            return []
    
    def get_available_metric_types(self) -> List[str]:
        """
        Get all available metric types in the database.
        
        Returns:
            List of metric types
        """
        try:
            # Query unique metric types from historical metrics
            result = self.conn.execute(
                "SELECT DISTINCT metric_type FROM historical_metrics ORDER BY metric_type"
            ).fetchall()
            
            return [row['metric_type'] for row in result]
            
        except Exception as e:
            logger.error(f"Error getting available metric types: {e}")
            return []
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the database.
        
        Returns:
            Dictionary with database statistics
        """
        try:
            stats = {}
            
            # Count historical metrics
            stats['historical_metrics_count'] = self.conn.execute(
                "SELECT COUNT(*) as count FROM historical_metrics"
            ).fetchone()['count']
            
            # Count predictions
            stats['predictions_count'] = self.conn.execute(
                "SELECT COUNT(*) as count FROM predictions"
            ).fetchone()['count']
            
            # Count anomalies
            stats['anomalies_count'] = self.conn.execute(
                "SELECT COUNT(*) as count FROM anomalies"
            ).fetchone()['count']
            
            # Count recommendations
            stats['recommendations_count'] = self.conn.execute(
                "SELECT COUNT(*) as count FROM recommendations"
            ).fetchone()['count']
            
            # Count comparative metrics
            stats['comparative_metrics_count'] = self.conn.execute(
                "SELECT COUNT(*) as count FROM comparative_metrics"
            ).fetchone()['count']
            
            # Count unique APIs
            stats['unique_apis_count'] = self.conn.execute(
                "SELECT COUNT(DISTINCT api_name) as count FROM historical_metrics"
            ).fetchone()['count']
            
            # Count unique metric types
            stats['unique_metric_types_count'] = self.conn.execute(
                "SELECT COUNT(DISTINCT metric_type) as count FROM historical_metrics"
            ).fetchone()['count']
            
            # Calculate date range
            date_range = self.conn.execute(
                """
                SELECT 
                    MIN(timestamp) as min_date,
                    MAX(timestamp) as max_date
                FROM historical_metrics
                """
            ).fetchone()
            
            stats['min_date'] = date_range['min_date']
            stats['max_date'] = date_range['max_date']
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}
    
    def close(self) -> None:
        """Close the database connection."""
        try:
            self.conn.close()
            logger.info("Closed database connection")
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DuckDB API Metrics Repository")
    parser.add_argument("--db-path", default="api_metrics.duckdb", help="Path to DuckDB database")
    parser.add_argument("--generate-sample", action="store_true", help="Generate sample data")
    parser.add_argument("--export", action="store_true", help="Export all data for UI")
    args = parser.parse_args()
    
    repo = DuckDBAPIMetricsRepository(args.db_path)
    
    if args.generate_sample:
        print("Generating sample data...")
        
        # Sample APIs
        apis = ["OpenAI", "Anthropic", "Cohere", "Groq", "Mistral"]
        
        # Sample metrics
        metrics = ["latency", "cost", "throughput", "success_rate", "tokens_per_second"]
        
        # Generate timestamps for the past 30 days
        now = datetime.datetime.now()
        timestamps = [now - datetime.timedelta(days=30) + datetime.timedelta(hours=i) for i in range(24*30)]
        
        # Generate historical metrics
        for api in apis:
            for metric in metrics:
                batch = []
                for i, ts in enumerate(timestamps):
                    # Generate a value that depends on API and metric
                    if metric == "latency":
                        value = 200 + i * 0.1  # Increasing trend
                        if api == "Groq":
                            value *= 0.5  # Groq is faster
                        elif api == "Anthropic":
                            value *= 1.2  # Anthropic is slower
                    elif metric == "cost":
                        value = 0.02 + i * 0.0001  # Slightly increasing cost
                        if api == "OpenAI":
                            value *= 1.5  # OpenAI is more expensive
                        elif api == "Cohere":
                            value *= 0.8  # Cohere is cheaper
                    elif metric == "throughput":
                        value = 100 - i * 0.05  # Decreasing trend
                    elif metric == "success_rate":
                        value = 98 - i * 0.01  # Slight decrease in success rate
                    else:  # tokens_per_second
                        value = 1000 + i * 0.2  # Increasing trend
                    
                    # Add some noise
                    value += (np.random.random() - 0.5) * value * 0.1
                    
                    # Ensure positive values
                    value = max(0, value)
                    
                    # Add to batch
                    batch.append({
                        'api_name': api,
                        'metric_type': metric,
                        'timestamp': ts.isoformat(),
                        'value': value
                    })
                
                # Save batch
                repo.save_historical_metrics_batch(batch)
        
        # Generate predictions for the next 14 days
        future_timestamps = [now + datetime.timedelta(hours=i) for i in range(24*14)]
        
        for api in apis:
            for metric in metrics:
                batch = []
                
                # Get the last historical value
                last_value = repo.get_historical_metrics(
                    api_name=api,
                    metric_type=metric,
                    limit=1
                )[0]['value']
                
                for i, ts in enumerate(future_timestamps):
                    # Predict value based on the last historical value and trend
                    if metric == "latency":
                        value = last_value + i * 0.1
                    elif metric == "cost":
                        value = last_value + i * 0.0001
                    elif metric == "throughput":
                        value = last_value - i * 0.05
                    elif metric == "success_rate":
                        value = last_value - i * 0.01
                    else:  # tokens_per_second
                        value = last_value + i * 0.2
                    
                    # Add some noise
                    noise = (np.random.random() - 0.5) * value * 0.05
                    value += noise
                    
                    # Calculate bounds
                    lower_bound = max(0, value - value * 0.2)
                    upper_bound = value + value * 0.2
                    
                    # Add to batch
                    batch.append({
                        'api_name': api,
                        'metric_type': metric,
                        'timestamp': ts.isoformat(),
                        'value': value,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound,
                        'confidence': 0.9,
                        'model_type': 'linear'
                    })
                
                # Save batch
                repo.save_predictions_batch(batch)
        
        # Generate anomalies
        for api in apis:
            for metric in metrics:
                # Get historical data for this API and metric
                historical_data = repo.get_historical_metrics(
                    api_name=api,
                    metric_type=metric
                )
                
                # Select 5 random points for anomalies
                if len(historical_data) > 5:
                    anomaly_indices = np.random.choice(len(historical_data), size=5, replace=False)
                    
                    batch = []
                    for idx in anomaly_indices:
                        data_point = historical_data[idx]
                        
                        # Anomaly types and severities
                        anomaly_types = ["spike", "trend_break", "oscillation", "seasonal"]
                        severities = ["low", "medium", "high", "critical"]
                        
                        # Random type and severity
                        anomaly_type = np.random.choice(anomaly_types)
                        severity = np.random.choice(severities)
                        
                        # Description based on type
                        description = f"Detected {anomaly_type} in {metric} for {api}"
                        
                        # Add to batch
                        batch.append({
                            'api_name': api,
                            'metric_type': metric,
                            'timestamp': data_point['timestamp'],
                            'value': data_point['value'],
                            'type': anomaly_type,
                            'confidence': 0.7 + np.random.random() * 0.3,
                            'severity': severity,
                            'description': description
                        })
                    
                    # Save batch
                    repo.save_anomalies_batch(batch)
        
        # Generate recommendations
        for api in apis:
            batch = []
            
            # Common recommendations
            recommendations = [
                {
                    'title': f"Optimize Batch Size for {api}",
                    'description': f"Increase batch size for {api} API calls to reduce overall latency and cost.",
                    'impact': 0.15 + np.random.random() * 0.1,
                    'effort': "Low",
                    'implementation_time': "Days",
                    'roi_period': "Weeks",
                    'status': "New"
                },
                {
                    'title': f"Implement Caching for {api}",
                    'description': f"Cache frequently repeated {api} API calls to reduce costs.",
                    'impact': 0.2 + np.random.random() * 0.15,
                    'effort': "Medium",
                    'implementation_time': "Weeks",
                    'roi_period': "Months",
                    'status': "New"
                }
            ]
            
            # API-specific recommendations
            if api == "OpenAI":
                batch.append({
                    'api_name': api,
                    'title': "Use GPT-3.5 Instead of GPT-4 Where Possible",
                    'description': "Switch to GPT-3.5 for tasks that don't require GPT-4's capabilities to reduce costs.",
                    'impact': 0.4,
                    'effort': "Low",
                    'implementation_time': "Days",
                    'roi_period': "Weeks",
                    'status': "New"
                })
            elif api == "Anthropic":
                batch.append({
                    'api_name': api,
                    'title': "Optimize Prompt Design for Claude",
                    'description': "Redesign prompts to reduce token usage with Claude API.",
                    'impact': 0.25,
                    'effort': "Medium",
                    'implementation_time': "Weeks",
                    'roi_period': "Months",
                    'status': "New"
                })
            elif api == "Groq":
                batch.append({
                    'api_name': api,
                    'title': "Increase Groq Usage for Latency-Critical Tasks",
                    'description': "Shift more latency-sensitive workloads to Groq for better performance.",
                    'impact': 0.35,
                    'effort': "Medium",
                    'implementation_time': "Weeks",
                    'roi_period': "Weeks",
                    'status': "New"
                })
            
            # Add common recommendations to batch
            for rec in recommendations:
                rec['api_name'] = api
                batch.append(rec)
            
            # Save batch
            repo.save_recommendations_batch(batch)
        
        # Generate comparative metrics
        for metric in metrics:
            batch = []
            
            # Use daily data points for comparative metrics
            for i in range(0, len(timestamps), 24):
                ts = timestamps[i]
                values = {}
                
                for api in apis:
                    # Get historical data for this timestamp
                    data = repo.get_historical_metrics(
                        api_name=api,
                        metric_type=metric,
                        start_time=ts,
                        end_time=ts + datetime.timedelta(hours=1)
                    )
                    
                    if data:
                        values[api] = data[0]['value']
                
                if values:
                    batch.append({
                        'metric_type': metric,
                        'timestamp': ts.isoformat(),
                        'values': values
                    })
            
            # Save batch
            repo.save_comparative_metrics_batch(batch)
        
        print("Sample data generation complete!")
    
    if args.export:
        print("Exporting all data for UI...")
        data = repo.export_all_data_for_ui()
        
        # Print some stats
        apis = set()
        metrics = set()
        
        for m_type, api_data in data['historical_data'].items():
            metrics.add(m_type)
            for api in api_data:
                apis.add(api)
        
        print(f"Exported data for {len(apis)} APIs and {len(metrics)} metric types")
        print(f"Historical metrics: {sum(len(data['historical_data'].get(m, {}).get(a, [])) for m in metrics for a in apis)}")
        print(f"Predictions: {sum(len(data['predictions'].get(m, {}).get(a, [])) for m in metrics for a in apis)}")
        print(f"Anomalies: {sum(len(data['anomalies'].get(m, {}).get(a, [])) for m in metrics for a in apis)}")
        print(f"Recommendations: {sum(len(data['recommendations'].get(a, [])) for a in apis)}")
        print(f"Comparative metrics: {sum(len(data['comparative_data'].get(m, [])) for m in metrics)}")
    
    # Get database stats
    stats = repo.get_database_stats()
    print("\nDatabase Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Close the connection
    repo.close()