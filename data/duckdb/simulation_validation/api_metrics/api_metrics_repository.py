"""
API Metrics Repository for DuckDB integration in the IPFS Accelerate Framework.

This module provides a repository for storing and retrieving API performance metrics,
including time series data, predictions, anomalies, and recommendations.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import time
import duckdb
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

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
        Initialize the DuckDB API Metrics Repository.
        
        Args:
            db_path: Path to the DuckDB database file
            create_if_missing: Whether to create the database if it doesn't exist
            config: Additional configuration options
        """
        self.db_path = db_path
        self.config = config or {}
        self.conn = None
        
        # Connect to the database
        self._connect(create_if_missing)
        
        # Initialize tables if needed
        if create_if_missing:
            self._initialize_tables()
    
    def _connect(self, create_if_missing: bool = True) -> None:
        """
        Connect to the DuckDB database.
        
        Args:
            create_if_missing: Whether to create the database if it doesn't exist
        """
        try:
            # Check if the database file exists
            db_exists = os.path.exists(self.db_path)
            
            if not db_exists and not create_if_missing:
                raise FileNotFoundError(f"Database file {self.db_path} not found")
            
            # Connect to the database
            self.conn = duckdb.connect(self.db_path)
            
            logger.info(f"Connected to DuckDB database at {self.db_path}")
        except Exception as e:
            logger.error(f"Error connecting to database: {str(e)}")
            raise
    
    def _initialize_tables(self) -> None:
        """
        Initialize the database tables if they don't exist.
        """
        try:
            # Create historical metrics table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS api_metrics (
                    id INTEGER PRIMARY KEY,
                    timestamp TIMESTAMP,
                    endpoint VARCHAR,
                    model VARCHAR,
                    response_time DOUBLE,
                    status_code INTEGER,
                    tokens INTEGER,
                    success BOOLEAN,
                    error_message VARCHAR,
                    metadata VARCHAR
                )
            """)
            
            # Create predictions table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS api_predictions (
                    id INTEGER PRIMARY KEY,
                    timestamp TIMESTAMP,
                    endpoint VARCHAR,
                    model VARCHAR,
                    predicted_value DOUBLE,
                    prediction_type VARCHAR,
                    confidence DOUBLE,
                    horizon INTEGER,
                    features VARCHAR
                )
            """)
            
            # Create anomalies table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS api_anomalies (
                    id INTEGER PRIMARY KEY,
                    timestamp TIMESTAMP,
                    endpoint VARCHAR,
                    model VARCHAR,
                    metric_type VARCHAR,
                    severity DOUBLE,
                    description VARCHAR,
                    threshold DOUBLE,
                    actual_value DOUBLE,
                    metadata VARCHAR
                )
            """)
            
            # Create recommendations table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS api_recommendations (
                    id INTEGER PRIMARY KEY,
                    timestamp TIMESTAMP,
                    endpoint VARCHAR,
                    model VARCHAR,
                    recommendation_type VARCHAR,
                    priority INTEGER,
                    description VARCHAR,
                    impact_score DOUBLE,
                    implementation_cost INTEGER,
                    metadata VARCHAR
                )
            """)
            
            # Create comparative analysis table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS api_comparative_analysis (
                    id INTEGER PRIMARY KEY,
                    timestamp TIMESTAMP,
                    baseline_endpoint VARCHAR,
                    comparison_endpoint VARCHAR,
                    baseline_model VARCHAR,
                    comparison_model VARCHAR,
                    metric VARCHAR,
                    baseline_value DOUBLE,
                    comparison_value DOUBLE,
                    difference DOUBLE,
                    percent_change DOUBLE,
                    significance DOUBLE,
                    metadata VARCHAR
                )
            """)
            
            # Create indices for common queries
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_api_metrics_timestamp ON api_metrics(timestamp)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_api_metrics_endpoint ON api_metrics(endpoint)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_api_metrics_model ON api_metrics(model)")
            
            logger.info("Database tables initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database tables: {str(e)}")
            raise
    
    def close(self) -> None:
        """
        Close the database connection.
        """
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def store_metric(self, metric: Dict[str, Any]) -> int:
        """
        Store a single API performance metric.
        
        Args:
            metric: Dictionary containing the metric data
            
        Returns:
            The ID of the stored metric
        """
        try:
            # Ensure timestamp is a datetime object
            if isinstance(metric.get('timestamp'), str):
                metric['timestamp'] = datetime.fromisoformat(metric['timestamp'].replace('Z', '+00:00'))
            elif not metric.get('timestamp'):
                metric['timestamp'] = datetime.now()
            
            # Convert metadata to JSON string if it's a dictionary
            if isinstance(metric.get('metadata'), dict):
                metric['metadata'] = json.dumps(metric['metadata'])
            
            # Insert the metric
            query = """
                INSERT INTO api_metrics (
                    timestamp, endpoint, model, response_time,
                    status_code, tokens, success, error_message, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                RETURNING id
            """
            
            result = self.conn.execute(
                query,
                (
                    metric.get('timestamp'),
                    metric.get('endpoint'),
                    metric.get('model'),
                    metric.get('response_time'),
                    metric.get('status_code'),
                    metric.get('tokens'),
                    metric.get('success', True),
                    metric.get('error_message'),
                    metric.get('metadata')
                )
            ).fetchone()
            
            return result[0] if result else -1
        except Exception as e:
            logger.error(f"Error storing metric: {str(e)}")
            raise
    
    def store_metrics_batch(self, metrics: List[Dict[str, Any]]) -> List[int]:
        """
        Store multiple API performance metrics in batch.
        
        Args:
            metrics: List of dictionaries containing metric data
            
        Returns:
            List of IDs for the stored metrics
        """
        if not metrics:
            return []
        
        try:
            ids = []
            for metric in metrics:
                metric_id = self.store_metric(metric)
                ids.append(metric_id)
            return ids
        except Exception as e:
            logger.error(f"Error storing metrics batch: {str(e)}")
            raise
    
    def get_metrics(self, 
                    start_time: Optional[datetime] = None,
                    end_time: Optional[datetime] = None,
                    endpoint: Optional[str] = None,
                    model: Optional[str] = None,
                    limit: int = 1000,
                    offset: int = 0) -> List[Dict[str, Any]]:
        """
        Retrieve API performance metrics based on filters.
        
        Args:
            start_time: Optional start time filter
            end_time: Optional end time filter
            endpoint: Optional endpoint filter
            model: Optional model filter
            limit: Maximum number of records to return
            offset: Number of records to skip
            
        Returns:
            List of metrics matching the filters
        """
        try:
            conditions = []
            params = []
            
            if start_time:
                conditions.append("timestamp >= ?")
                params.append(start_time)
            
            if end_time:
                conditions.append("timestamp <= ?")
                params.append(end_time)
            
            if endpoint:
                conditions.append("endpoint = ?")
                params.append(endpoint)
            
            if model:
                conditions.append("model = ?")
                params.append(model)
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            query = f"""
                SELECT id, timestamp, endpoint, model, response_time,
                       status_code, tokens, success, error_message, metadata
                FROM api_metrics
                WHERE {where_clause}
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
            """
            
            params.extend([limit, offset])
            
            result = self.conn.execute(query, params).fetchall()
            
            # Convert to list of dictionaries
            metrics = []
            for row in result:
                metric = {
                    'id': row[0],
                    'timestamp': row[1],
                    'endpoint': row[2],
                    'model': row[3],
                    'response_time': row[4],
                    'status_code': row[5],
                    'tokens': row[6],
                    'success': row[7],
                    'error_message': row[8],
                    'metadata': row[9]
                }
                
                # Parse metadata if it's a JSON string
                if isinstance(metric['metadata'], str) and metric['metadata']:
                    try:
                        metric['metadata'] = json.loads(metric['metadata'])
                    except:
                        pass
                
                metrics.append(metric)
            
            return metrics
        except Exception as e:
            logger.error(f"Error retrieving metrics: {str(e)}")
            raise
    
    def get_metrics_aggregated(self,
                             start_time: Optional[datetime] = None,
                             end_time: Optional[datetime] = None,
                             endpoint: Optional[str] = None,
                             model: Optional[str] = None,
                             group_by: str = 'hour',
                             metric_type: str = 'response_time') -> List[Dict[str, Any]]:
        """
        Retrieve aggregated API performance metrics.
        
        Args:
            start_time: Optional start time filter
            end_time: Optional end time filter
            endpoint: Optional endpoint filter
            model: Optional model filter
            group_by: Time interval for grouping (hour, day, week, month)
            metric_type: Type of metric to aggregate (response_time, tokens, etc.)
            
        Returns:
            List of aggregated metrics
        """
        try:
            conditions = []
            params = []
            
            if start_time:
                conditions.append("timestamp >= ?")
                params.append(start_time)
            
            if end_time:
                conditions.append("timestamp <= ?")
                params.append(end_time)
            
            if endpoint:
                conditions.append("endpoint = ?")
                params.append(endpoint)
            
            if model:
                conditions.append("model = ?")
                params.append(model)
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            # Determine the date_trunc parameter based on group_by
            interval = {
                'minute': 'minute', 
                'hour': 'hour',
                'day': 'day',
                'week': 'week',
                'month': 'month'
            }.get(group_by.lower(), 'hour')
            
            # Adjust metric type
            if metric_type not in ['response_time', 'tokens', 'status_code']:
                metric_type = 'response_time'
            
            query = f"""
                SELECT 
                    date_trunc('{interval}', timestamp) as time_bucket,
                    AVG({metric_type}) as avg_value,
                    MIN({metric_type}) as min_value,
                    MAX({metric_type}) as max_value,
                    COUNT(*) as count,
                    SUM(CASE WHEN success = true THEN 1 ELSE 0 END) as success_count,
                    endpoint,
                    model
                FROM api_metrics
                WHERE {where_clause}
                GROUP BY time_bucket, endpoint, model
                ORDER BY time_bucket ASC
            """
            
            result = self.conn.execute(query, params).fetchall()
            
            # Convert to list of dictionaries
            aggregated = []
            for row in result:
                item = {
                    'timestamp': row[0],
                    'avg_value': row[1],
                    'min_value': row[2],
                    'max_value': row[3],
                    'count': row[4],
                    'success_rate': row[5] / row[4] if row[4] > 0 else 0,
                    'endpoint': row[6],
                    'model': row[7],
                    'metric_type': metric_type
                }
                aggregated.append(item)
            
            return aggregated
        except Exception as e:
            logger.error(f"Error retrieving aggregated metrics: {str(e)}")
            raise
    
    def store_prediction(self, prediction: Dict[str, Any]) -> int:
        """
        Store a predictive analytics result.
        
        Args:
            prediction: Dictionary containing the prediction data
            
        Returns:
            The ID of the stored prediction
        """
        try:
            # Ensure timestamp is a datetime object
            if isinstance(prediction.get('timestamp'), str):
                prediction['timestamp'] = datetime.fromisoformat(prediction['timestamp'].replace('Z', '+00:00'))
            elif not prediction.get('timestamp'):
                prediction['timestamp'] = datetime.now()
            
            # Convert features to JSON string if it's a dictionary
            if isinstance(prediction.get('features'), dict):
                prediction['features'] = json.dumps(prediction['features'])
            
            # Insert the prediction
            query = """
                INSERT INTO api_predictions (
                    timestamp, endpoint, model, predicted_value,
                    prediction_type, confidence, horizon, features
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                RETURNING id
            """
            
            result = self.conn.execute(
                query,
                (
                    prediction.get('timestamp'),
                    prediction.get('endpoint'),
                    prediction.get('model'),
                    prediction.get('predicted_value'),
                    prediction.get('prediction_type'),
                    prediction.get('confidence'),
                    prediction.get('horizon'),
                    prediction.get('features')
                )
            ).fetchone()
            
            return result[0] if result else -1
        except Exception as e:
            logger.error(f"Error storing prediction: {str(e)}")
            raise
    
    def get_predictions(self,
                      start_time: Optional[datetime] = None,
                      end_time: Optional[datetime] = None,
                      endpoint: Optional[str] = None,
                      model: Optional[str] = None,
                      prediction_type: Optional[str] = None,
                      limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Retrieve predictive analytics results.
        
        Args:
            start_time: Optional start time filter
            end_time: Optional end time filter
            endpoint: Optional endpoint filter
            model: Optional model filter
            prediction_type: Optional prediction type filter
            limit: Maximum number of records to return
            
        Returns:
            List of predictions matching the filters
        """
        try:
            conditions = []
            params = []
            
            if start_time:
                conditions.append("timestamp >= ?")
                params.append(start_time)
            
            if end_time:
                conditions.append("timestamp <= ?")
                params.append(end_time)
            
            if endpoint:
                conditions.append("endpoint = ?")
                params.append(endpoint)
            
            if model:
                conditions.append("model = ?")
                params.append(model)
            
            if prediction_type:
                conditions.append("prediction_type = ?")
                params.append(prediction_type)
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            query = f"""
                SELECT id, timestamp, endpoint, model, predicted_value,
                       prediction_type, confidence, horizon, features
                FROM api_predictions
                WHERE {where_clause}
                ORDER BY timestamp DESC
                LIMIT ?
            """
            
            params.append(limit)
            
            result = self.conn.execute(query, params).fetchall()
            
            # Convert to list of dictionaries
            predictions = []
            for row in result:
                prediction = {
                    'id': row[0],
                    'timestamp': row[1],
                    'endpoint': row[2],
                    'model': row[3],
                    'predicted_value': row[4],
                    'prediction_type': row[5],
                    'confidence': row[6],
                    'horizon': row[7],
                    'features': row[8]
                }
                
                # Parse features if it's a JSON string
                if isinstance(prediction['features'], str) and prediction['features']:
                    try:
                        prediction['features'] = json.loads(prediction['features'])
                    except:
                        pass
                
                predictions.append(prediction)
            
            return predictions
        except Exception as e:
            logger.error(f"Error retrieving predictions: {str(e)}")
            raise
    
    def store_anomaly(self, anomaly: Dict[str, Any]) -> int:
        """
        Store an anomaly detection result.
        
        Args:
            anomaly: Dictionary containing the anomaly data
            
        Returns:
            The ID of the stored anomaly
        """
        try:
            # Ensure timestamp is a datetime object
            if isinstance(anomaly.get('timestamp'), str):
                anomaly['timestamp'] = datetime.fromisoformat(anomaly['timestamp'].replace('Z', '+00:00'))
            elif not anomaly.get('timestamp'):
                anomaly['timestamp'] = datetime.now()
            
            # Convert metadata to JSON string if it's a dictionary
            if isinstance(anomaly.get('metadata'), dict):
                anomaly['metadata'] = json.dumps(anomaly['metadata'])
            
            # Insert the anomaly
            query = """
                INSERT INTO api_anomalies (
                    timestamp, endpoint, model, metric_type,
                    severity, description, threshold, actual_value, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                RETURNING id
            """
            
            result = self.conn.execute(
                query,
                (
                    anomaly.get('timestamp'),
                    anomaly.get('endpoint'),
                    anomaly.get('model'),
                    anomaly.get('metric_type'),
                    anomaly.get('severity'),
                    anomaly.get('description'),
                    anomaly.get('threshold'),
                    anomaly.get('actual_value'),
                    anomaly.get('metadata')
                )
            ).fetchone()
            
            return result[0] if result else -1
        except Exception as e:
            logger.error(f"Error storing anomaly: {str(e)}")
            raise
    
    def get_anomalies(self,
                    start_time: Optional[datetime] = None,
                    end_time: Optional[datetime] = None,
                    endpoint: Optional[str] = None,
                    model: Optional[str] = None,
                    metric_type: Optional[str] = None,
                    min_severity: float = 0.0,
                    limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Retrieve anomaly detection results.
        
        Args:
            start_time: Optional start time filter
            end_time: Optional end time filter
            endpoint: Optional endpoint filter
            model: Optional model filter
            metric_type: Optional metric type filter
            min_severity: Minimum severity threshold
            limit: Maximum number of records to return
            
        Returns:
            List of anomalies matching the filters
        """
        try:
            conditions = []
            params = []
            
            if start_time:
                conditions.append("timestamp >= ?")
                params.append(start_time)
            
            if end_time:
                conditions.append("timestamp <= ?")
                params.append(end_time)
            
            if endpoint:
                conditions.append("endpoint = ?")
                params.append(endpoint)
            
            if model:
                conditions.append("model = ?")
                params.append(model)
            
            if metric_type:
                conditions.append("metric_type = ?")
                params.append(metric_type)
            
            conditions.append("severity >= ?")
            params.append(min_severity)
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            query = f"""
                SELECT id, timestamp, endpoint, model, metric_type,
                       severity, description, threshold, actual_value, metadata
                FROM api_anomalies
                WHERE {where_clause}
                ORDER BY severity DESC, timestamp DESC
                LIMIT ?
            """
            
            params.append(limit)
            
            result = self.conn.execute(query, params).fetchall()
            
            # Convert to list of dictionaries
            anomalies = []
            for row in result:
                anomaly = {
                    'id': row[0],
                    'timestamp': row[1],
                    'endpoint': row[2],
                    'model': row[3],
                    'metric_type': row[4],
                    'severity': row[5],
                    'description': row[6],
                    'threshold': row[7],
                    'actual_value': row[8],
                    'metadata': row[9]
                }
                
                # Parse metadata if it's a JSON string
                if isinstance(anomaly['metadata'], str) and anomaly['metadata']:
                    try:
                        anomaly['metadata'] = json.loads(anomaly['metadata'])
                    except:
                        pass
                
                anomalies.append(anomaly)
            
            return anomalies
        except Exception as e:
            logger.error(f"Error retrieving anomalies: {str(e)}")
            raise
    
    def store_recommendation(self, recommendation: Dict[str, Any]) -> int:
        """
        Store a recommendation.
        
        Args:
            recommendation: Dictionary containing the recommendation data
            
        Returns:
            The ID of the stored recommendation
        """
        try:
            # Ensure timestamp is a datetime object
            if isinstance(recommendation.get('timestamp'), str):
                recommendation['timestamp'] = datetime.fromisoformat(recommendation['timestamp'].replace('Z', '+00:00'))
            elif not recommendation.get('timestamp'):
                recommendation['timestamp'] = datetime.now()
            
            # Convert metadata to JSON string if it's a dictionary
            if isinstance(recommendation.get('metadata'), dict):
                recommendation['metadata'] = json.dumps(recommendation['metadata'])
            
            # Insert the recommendation
            query = """
                INSERT INTO api_recommendations (
                    timestamp, endpoint, model, recommendation_type,
                    priority, description, impact_score, implementation_cost, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                RETURNING id
            """
            
            result = self.conn.execute(
                query,
                (
                    recommendation.get('timestamp'),
                    recommendation.get('endpoint'),
                    recommendation.get('model'),
                    recommendation.get('recommendation_type'),
                    recommendation.get('priority'),
                    recommendation.get('description'),
                    recommendation.get('impact_score'),
                    recommendation.get('implementation_cost'),
                    recommendation.get('metadata')
                )
            ).fetchone()
            
            return result[0] if result else -1
        except Exception as e:
            logger.error(f"Error storing recommendation: {str(e)}")
            raise
    
    def get_recommendations(self,
                          endpoint: Optional[str] = None,
                          model: Optional[str] = None,
                          recommendation_type: Optional[str] = None,
                          min_priority: int = 0,
                          limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve recommendations.
        
        Args:
            endpoint: Optional endpoint filter
            model: Optional model filter
            recommendation_type: Optional recommendation type filter
            min_priority: Minimum priority threshold (0-5, with 5 being highest)
            limit: Maximum number of records to return
            
        Returns:
            List of recommendations matching the filters
        """
        try:
            conditions = []
            params = []
            
            if endpoint:
                conditions.append("endpoint = ?")
                params.append(endpoint)
            
            if model:
                conditions.append("model = ?")
                params.append(model)
            
            if recommendation_type:
                conditions.append("recommendation_type = ?")
                params.append(recommendation_type)
            
            conditions.append("priority >= ?")
            params.append(min_priority)
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            query = f"""
                SELECT id, timestamp, endpoint, model, recommendation_type,
                       priority, description, impact_score, implementation_cost, metadata
                FROM api_recommendations
                WHERE {where_clause}
                ORDER BY priority DESC, impact_score DESC
                LIMIT ?
            """
            
            params.append(limit)
            
            result = self.conn.execute(query, params).fetchall()
            
            # Convert to list of dictionaries
            recommendations = []
            for row in result:
                recommendation = {
                    'id': row[0],
                    'timestamp': row[1],
                    'endpoint': row[2],
                    'model': row[3],
                    'recommendation_type': row[4],
                    'priority': row[5],
                    'description': row[6],
                    'impact_score': row[7],
                    'implementation_cost': row[8],
                    'metadata': row[9]
                }
                
                # Parse metadata if it's a JSON string
                if isinstance(recommendation['metadata'], str) and recommendation['metadata']:
                    try:
                        recommendation['metadata'] = json.loads(recommendation['metadata'])
                    except:
                        pass
                
                recommendations.append(recommendation)
            
            return recommendations
        except Exception as e:
            logger.error(f"Error retrieving recommendations: {str(e)}")
            raise
    
    def generate_sample_data(self, num_records: int = 1000, days_back: int = 30) -> None:
        """
        Generate sample API metrics data for testing.
        
        Args:
            num_records: Number of sample records to generate
            days_back: Number of days back to generate data for
        """
        try:
            # Sample endpoints and models
            endpoints = [
                '/v1/completions',
                '/v1/chat/completions',
                '/v1/embeddings',
                '/v1/images/generations',
                '/v1/audio/transcriptions'
            ]
            
            models = [
                'gpt-4',
                'gpt-3.5-turbo',
                'claude-3-opus',
                'claude-3-sonnet',
                'text-embedding-3-large',
                'dall-e-3',
                'whisper-large-v3'
            ]
            
            # Generate random timestamps over the past days_back days
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days_back)
            time_range = (end_time - start_time).total_seconds()
            
            metrics = []
            for _ in range(num_records):
                # Random timestamp
                random_seconds = np.random.rand() * time_range
                timestamp = start_time + timedelta(seconds=random_seconds)
                
                # Random endpoint and model
                endpoint = np.random.choice(endpoints)
                model = np.random.choice(models)
                
                # Generate appropriate response time based on model
                base_response_time = {
                    'gpt-4': 2.0,
                    'gpt-3.5-turbo': 0.8,
                    'claude-3-opus': 1.8,
                    'claude-3-sonnet': 1.0,
                    'text-embedding-3-large': 0.2,
                    'dall-e-3': 4.0,
                    'whisper-large-v3': 1.5
                }.get(model, 1.0)
                
                # Add some randomness to response time
                response_time = max(0.05, np.random.normal(base_response_time, base_response_time / 3))
                
                # 98% success rate
                success = np.random.rand() > 0.02
                status_code = 200 if success else np.random.choice([429, 500, 502, 503])
                
                # Generate random token count based on endpoint
                tokens = 0
                if 'completions' in endpoint:
                    tokens = int(np.random.normal(500, 200))
                elif 'embeddings' in endpoint:
                    tokens = int(np.random.normal(100, 30))
                
                # Create metric record
                metric = {
                    'timestamp': timestamp,
                    'endpoint': endpoint,
                    'model': model,
                    'response_time': response_time,
                    'status_code': status_code,
                    'tokens': max(0, tokens),
                    'success': success,
                    'error_message': None if success else f"Error: {status_code}",
                    'metadata': json.dumps({
                        'user_id': f"user_{np.random.randint(1, 100)}",
                        'region': np.random.choice(['us-east', 'us-west', 'eu-west', 'ap-southeast']),
                        'client_version': f"v{np.random.randint(1, 5)}.{np.random.randint(0, 10)}"
                    })
                }
                
                metrics.append(metric)
            
            # Store metrics in batches
            batch_size = 100
            for i in range(0, len(metrics), batch_size):
                batch = metrics[i:i+batch_size]
                self.store_metrics_batch(batch)
            
            # Generate some predictions
            for endpoint in endpoints:
                for model in models[:3]:  # Only generate for the first few models
                    for horizon in [24, 48, 168]:  # 1 day, 2 days, 1 week
                        prediction = {
                            'timestamp': datetime.now(),
                            'endpoint': endpoint,
                            'model': model,
                            'predicted_value': np.random.normal(1.5, 0.5),
                            'prediction_type': 'response_time',
                            'confidence': np.random.uniform(0.6, 0.95),
                            'horizon': horizon,
                            'features': json.dumps({
                                'historical_mean': np.random.normal(1.2, 0.3),
                                'trend': np.random.choice(['increasing', 'decreasing', 'stable']),
                                'seasonality_strength': np.random.uniform(0.1, 0.5)
                            })
                        }
                        self.store_prediction(prediction)
            
            # Generate some anomalies
            for _ in range(10):
                endpoint = np.random.choice(endpoints)
                model = np.random.choice(models)
                anomaly = {
                    'timestamp': start_time + timedelta(seconds=np.random.rand() * time_range),
                    'endpoint': endpoint,
                    'model': model,
                    'metric_type': np.random.choice(['response_time', 'error_rate', 'throughput']),
                    'severity': np.random.uniform(0.5, 1.0),
                    'description': f"Unusual {np.random.choice(['spike', 'drop', 'pattern'])} detected",
                    'threshold': np.random.uniform(1.5, 3.0),
                    'actual_value': np.random.uniform(3.0, 6.0),
                    'metadata': json.dumps({
                        'detection_algorithm': np.random.choice(['z-score', 'IQR', 'ARIMA', 'isolation_forest']),
                        'affected_users': np.random.randint(1, 50),
                        'duration_minutes': np.random.randint(5, 60)
                    })
                }
                self.store_anomaly(anomaly)
            
            # Generate some recommendations
            recommendation_types = [
                'caching',
                'rate_limiting',
                'model_switching',
                'batch_processing',
                'reliability_improvement'
            ]
            
            for recommendation_type in recommendation_types:
                for priority in range(1, 6):  # 1-5 priority levels
                    if np.random.rand() > 0.5:  # 50% chance to generate for each type/priority combo
                        endpoint = np.random.choice(endpoints)
                        model = np.random.choice(models)
                        recommendation = {
                            'timestamp': datetime.now(),
                            'endpoint': endpoint,
                            'model': model,
                            'recommendation_type': recommendation_type,
                            'priority': priority,
                            'description': f"Implement {recommendation_type} for improved performance",
                            'impact_score': np.random.uniform(0.1, 1.0),
                            'implementation_cost': np.random.randint(1, 5),
                            'metadata': json.dumps({
                                'estimated_savings': f"${np.random.randint(100, 5000)}",
                                'estimated_effort_days': np.random.randint(1, 20),
                                'affected_services': np.random.randint(1, 5)
                            })
                        }
                        self.store_recommendation(recommendation)
            
            logger.info(f"Generated {num_records} sample API metrics records")
        except Exception as e:
            logger.error(f"Error generating sample data: {str(e)}")
            raise
