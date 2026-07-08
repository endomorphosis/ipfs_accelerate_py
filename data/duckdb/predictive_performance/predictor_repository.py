"""
Predictor Repository for DuckDB integration in the IPFS Accelerate Framework.

This module provides a repository for storing and retrieving predictive performance data,
including model predictions, actual measurements, performance metrics, and hardware-model mappings.
"""

import os
import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import duckdb
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class DuckDBPredictorRepository:
    """
    DuckDB-based repository for predictive performance data.
    
    This class provides methods for storing and retrieving performance predictions,
    actual performance measurements, model-hardware mappings, and training data
    for the predictive performance modeling system.
    """
    
    def __init__(
        self, 
        db_path: str = "predictor.duckdb",
        create_if_missing: bool = True,
        config: Dict[str, Any] = None
    ):
        """
        Initialize the DuckDB Predictor Repository.
        
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
            # Create performance prediction table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_predictions (
                    id INTEGER PRIMARY KEY,
                    timestamp TIMESTAMP,
                    model_name VARCHAR,
                    model_family VARCHAR,
                    hardware_platform VARCHAR,
                    batch_size INTEGER,
                    sequence_length INTEGER,
                    precision VARCHAR,
                    mode VARCHAR,
                    throughput DOUBLE,
                    latency DOUBLE,
                    memory_usage DOUBLE,
                    confidence_score DOUBLE,
                    prediction_source VARCHAR,
                    prediction_id VARCHAR,
                    metadata VARCHAR
                )
            """)
            
            # Create actual performance measurements table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_measurements (
                    id INTEGER PRIMARY KEY,
                    timestamp TIMESTAMP,
                    model_name VARCHAR,
                    model_family VARCHAR,
                    hardware_platform VARCHAR,
                    batch_size INTEGER,
                    sequence_length INTEGER,
                    precision VARCHAR,
                    mode VARCHAR,
                    throughput DOUBLE,
                    latency DOUBLE,
                    memory_usage DOUBLE,
                    measurement_source VARCHAR,
                    measurement_id VARCHAR,
                    metadata VARCHAR
                )
            """)
            
            # Create hardware-model mapping table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS hardware_model_mappings (
                    id INTEGER PRIMARY KEY,
                    timestamp TIMESTAMP,
                    model_name VARCHAR,
                    model_family VARCHAR,
                    hardware_platform VARCHAR,
                    compatibility_score DOUBLE,
                    recommendation_rank INTEGER,
                    is_primary_recommendation BOOLEAN,
                    reason VARCHAR,
                    mapping_id VARCHAR,
                    metadata VARCHAR
                )
            """)
            
            # Create prediction models table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS prediction_models (
                    id INTEGER PRIMARY KEY,
                    timestamp TIMESTAMP,
                    model_type VARCHAR,
                    target_metric VARCHAR,
                    hardware_platform VARCHAR,
                    model_family VARCHAR,
                    serialized_model BLOB,
                    features_list VARCHAR,
                    training_score DOUBLE,
                    validation_score DOUBLE,
                    test_score DOUBLE,
                    model_id VARCHAR,
                    metadata VARCHAR
                )
            """)
            
            # Create prediction errors table (for tracking prediction accuracy)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS prediction_errors (
                    id INTEGER PRIMARY KEY,
                    timestamp TIMESTAMP,
                    prediction_id VARCHAR,
                    measurement_id VARCHAR,
                    model_name VARCHAR,
                    hardware_platform VARCHAR,
                    metric VARCHAR,
                    predicted_value DOUBLE,
                    actual_value DOUBLE,
                    absolute_error DOUBLE,
                    relative_error DOUBLE,
                    metadata VARCHAR
                )
            """)
            
            # Create recommendation history table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS recommendation_history (
                    id INTEGER PRIMARY KEY,
                    timestamp TIMESTAMP,
                    user_id VARCHAR,
                    model_name VARCHAR,
                    model_family VARCHAR,
                    batch_size INTEGER,
                    sequence_length INTEGER,
                    precision VARCHAR,
                    mode VARCHAR,
                    primary_recommendation VARCHAR,
                    fallback_options VARCHAR,
                    compatible_hardware VARCHAR,
                    reason VARCHAR,
                    recommendation_id VARCHAR,
                    was_accepted BOOLEAN,
                    user_feedback VARCHAR,
                    metadata VARCHAR
                )
            """)
            
            # Create prediction feature importance table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS feature_importance (
                    id INTEGER PRIMARY KEY,
                    timestamp TIMESTAMP,
                    model_id VARCHAR,
                    feature_name VARCHAR,
                    importance_score DOUBLE,
                    rank INTEGER,
                    method VARCHAR,
                    metadata VARCHAR
                )
            """)
            
            # Create indices for common queries
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_pred_model_hw ON performance_predictions(model_name, hardware_platform)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_meas_model_hw ON performance_measurements(model_name, hardware_platform)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_mapping_model ON hardware_model_mappings(model_name)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_pred_model_id ON prediction_models(model_id)")
            
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
    
    def store_prediction(self, prediction: Dict[str, Any]) -> int:
        """
        Store a performance prediction.
        
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
            
            # Convert metadata to JSON string if it's a dictionary
            if isinstance(prediction.get('metadata'), dict):
                prediction['metadata'] = json.dumps(prediction['metadata'])
            
            # Generate a prediction ID if not provided
            if not prediction.get('prediction_id'):
                prediction['prediction_id'] = f"pred-{int(time.time())}-{hash(prediction.get('model_name', ''))}"
            
            # Insert the prediction
            query = """
                INSERT INTO performance_predictions (
                    timestamp, model_name, model_family, hardware_platform,
                    batch_size, sequence_length, precision, mode,
                    throughput, latency, memory_usage, confidence_score,
                    prediction_source, prediction_id, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                RETURNING id
            """
            
            result = self.conn.execute(
                query,
                (
                    prediction.get('timestamp'),
                    prediction.get('model_name'),
                    prediction.get('model_family'),
                    prediction.get('hardware_platform'),
                    prediction.get('batch_size'),
                    prediction.get('sequence_length'),
                    prediction.get('precision'),
                    prediction.get('mode'),
                    prediction.get('throughput'),
                    prediction.get('latency'),
                    prediction.get('memory_usage'),
                    prediction.get('confidence_score'),
                    prediction.get('prediction_source'),
                    prediction.get('prediction_id'),
                    prediction.get('metadata')
                )
            ).fetchone()
            
            return result[0] if result else -1
        except Exception as e:
            logger.error(f"Error storing prediction: {str(e)}")
            raise
    
    def store_predictions_batch(self, predictions: List[Dict[str, Any]]) -> List[int]:
        """
        Store multiple performance predictions in batch.
        
        Args:
            predictions: List of dictionaries containing prediction data
            
        Returns:
            List of IDs for the stored predictions
        """
        if not predictions:
            return []
        
        try:
            ids = []
            for prediction in predictions:
                pred_id = self.store_prediction(prediction)
                ids.append(pred_id)
            return ids
        except Exception as e:
            logger.error(f"Error storing predictions batch: {str(e)}")
            raise
    
    def get_predictions(self, 
                      model_name: Optional[str] = None,
                      model_family: Optional[str] = None,
                      hardware_platform: Optional[str] = None,
                      batch_size: Optional[int] = None,
                      prediction_id: Optional[str] = None,
                      limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve performance predictions based on filters.
        
        Args:
            model_name: Optional model name filter
            model_family: Optional model family filter
            hardware_platform: Optional hardware platform filter
            batch_size: Optional batch size filter
            prediction_id: Optional prediction ID filter
            limit: Maximum number of records to return
            
        Returns:
            List of predictions matching the filters
        """
        try:
            conditions = []
            params = []
            
            if model_name:
                conditions.append("model_name = ?")
                params.append(model_name)
            
            if model_family:
                conditions.append("model_family = ?")
                params.append(model_family)
            
            if hardware_platform:
                conditions.append("hardware_platform = ?")
                params.append(hardware_platform)
            
            if batch_size is not None:
                conditions.append("batch_size = ?")
                params.append(batch_size)
            
            if prediction_id:
                conditions.append("prediction_id = ?")
                params.append(prediction_id)
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            query = f"""
                SELECT id, timestamp, model_name, model_family, hardware_platform,
                       batch_size, sequence_length, precision, mode,
                       throughput, latency, memory_usage, confidence_score,
                       prediction_source, prediction_id, metadata
                FROM performance_predictions
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
                    'model_name': row[2],
                    'model_family': row[3],
                    'hardware_platform': row[4],
                    'batch_size': row[5],
                    'sequence_length': row[6],
                    'precision': row[7],
                    'mode': row[8],
                    'throughput': row[9],
                    'latency': row[10],
                    'memory_usage': row[11],
                    'confidence_score': row[12],
                    'prediction_source': row[13],
                    'prediction_id': row[14],
                    'metadata': row[15]
                }
                
                # Parse metadata if it's a JSON string
                if isinstance(prediction['metadata'], str) and prediction['metadata']:
                    try:
                        prediction['metadata'] = json.loads(prediction['metadata'])
                    except:
                        pass
                
                predictions.append(prediction)
            
            return predictions
        except Exception as e:
            logger.error(f"Error retrieving predictions: {str(e)}")
            raise
    
    def store_measurement(self, measurement: Dict[str, Any]) -> int:
        """
        Store an actual performance measurement.
        
        Args:
            measurement: Dictionary containing the measurement data
            
        Returns:
            The ID of the stored measurement
        """
        try:
            # Ensure timestamp is a datetime object
            if isinstance(measurement.get('timestamp'), str):
                measurement['timestamp'] = datetime.fromisoformat(measurement['timestamp'].replace('Z', '+00:00'))
            elif not measurement.get('timestamp'):
                measurement['timestamp'] = datetime.now()
            
            # Convert metadata to JSON string if it's a dictionary
            if isinstance(measurement.get('metadata'), dict):
                measurement['metadata'] = json.dumps(measurement['metadata'])
            
            # Generate a measurement ID if not provided
            if not measurement.get('measurement_id'):
                measurement['measurement_id'] = f"meas-{int(time.time())}-{hash(measurement.get('model_name', ''))}"
            
            # Insert the measurement
            query = """
                INSERT INTO performance_measurements (
                    timestamp, model_name, model_family, hardware_platform,
                    batch_size, sequence_length, precision, mode,
                    throughput, latency, memory_usage, measurement_source,
                    measurement_id, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                RETURNING id
            """
            
            result = self.conn.execute(
                query,
                (
                    measurement.get('timestamp'),
                    measurement.get('model_name'),
                    measurement.get('model_family'),
                    measurement.get('hardware_platform'),
                    measurement.get('batch_size'),
                    measurement.get('sequence_length'),
                    measurement.get('precision'),
                    measurement.get('mode'),
                    measurement.get('throughput'),
                    measurement.get('latency'),
                    measurement.get('memory_usage'),
                    measurement.get('measurement_source'),
                    measurement.get('measurement_id'),
                    measurement.get('metadata')
                )
            ).fetchone()
            
            return result[0] if result else -1
        except Exception as e:
            logger.error(f"Error storing measurement: {str(e)}")
            raise
    
    def get_measurements(self, 
                       model_name: Optional[str] = None,
                       model_family: Optional[str] = None,
                       hardware_platform: Optional[str] = None,
                       batch_size: Optional[int] = None,
                       measurement_id: Optional[str] = None,
                       start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None,
                       limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve performance measurements based on filters.
        
        Args:
            model_name: Optional model name filter
            model_family: Optional model family filter
            hardware_platform: Optional hardware platform filter
            batch_size: Optional batch size filter
            measurement_id: Optional measurement ID filter
            start_time: Optional start time filter
            end_time: Optional end time filter
            limit: Maximum number of records to return
            
        Returns:
            List of measurements matching the filters
        """
        try:
            conditions = []
            params = []
            
            if model_name:
                conditions.append("model_name = ?")
                params.append(model_name)
            
            if model_family:
                conditions.append("model_family = ?")
                params.append(model_family)
            
            if hardware_platform:
                conditions.append("hardware_platform = ?")
                params.append(hardware_platform)
            
            if batch_size is not None:
                conditions.append("batch_size = ?")
                params.append(batch_size)
            
            if measurement_id:
                conditions.append("measurement_id = ?")
                params.append(measurement_id)
            
            if start_time:
                conditions.append("timestamp >= ?")
                params.append(start_time)
            
            if end_time:
                conditions.append("timestamp <= ?")
                params.append(end_time)
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            query = f"""
                SELECT id, timestamp, model_name, model_family, hardware_platform,
                       batch_size, sequence_length, precision, mode,
                       throughput, latency, memory_usage, measurement_source,
                       measurement_id, metadata
                FROM performance_measurements
                WHERE {where_clause}
                ORDER BY timestamp DESC
                LIMIT ?
            """
            
            params.append(limit)
            
            result = self.conn.execute(query, params).fetchall()
            
            # Convert to list of dictionaries
            measurements = []
            for row in result:
                measurement = {
                    'id': row[0],
                    'timestamp': row[1],
                    'model_name': row[2],
                    'model_family': row[3],
                    'hardware_platform': row[4],
                    'batch_size': row[5],
                    'sequence_length': row[6],
                    'precision': row[7],
                    'mode': row[8],
                    'throughput': row[9],
                    'latency': row[10],
                    'memory_usage': row[11],
                    'measurement_source': row[12],
                    'measurement_id': row[13],
                    'metadata': row[14]
                }
                
                # Parse metadata if it's a JSON string
                if isinstance(measurement['metadata'], str) and measurement['metadata']:
                    try:
                        measurement['metadata'] = json.loads(measurement['metadata'])
                    except:
                        pass
                
                measurements.append(measurement)
            
            return measurements
        except Exception as e:
            logger.error(f"Error retrieving measurements: {str(e)}")
            raise
    
    def store_hardware_model_mapping(self, mapping: Dict[str, Any]) -> int:
        """
        Store a hardware-model mapping.
        
        Args:
            mapping: Dictionary containing the mapping data
            
        Returns:
            The ID of the stored mapping
        """
        try:
            # Ensure timestamp is a datetime object
            if isinstance(mapping.get('timestamp'), str):
                mapping['timestamp'] = datetime.fromisoformat(mapping['timestamp'].replace('Z', '+00:00'))
            elif not mapping.get('timestamp'):
                mapping['timestamp'] = datetime.now()
            
            # Convert metadata to JSON string if it's a dictionary
            if isinstance(mapping.get('metadata'), dict):
                mapping['metadata'] = json.dumps(mapping['metadata'])
            
            # Generate a mapping ID if not provided
            if not mapping.get('mapping_id'):
                mapping['mapping_id'] = f"map-{int(time.time())}-{hash(mapping.get('model_name', ''))}-{hash(mapping.get('hardware_platform', ''))}"
            
            # Insert the mapping
            query = """
                INSERT INTO hardware_model_mappings (
                    timestamp, model_name, model_family, hardware_platform,
                    compatibility_score, recommendation_rank, is_primary_recommendation,
                    reason, mapping_id, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                RETURNING id
            """
            
            result = self.conn.execute(
                query,
                (
                    mapping.get('timestamp'),
                    mapping.get('model_name'),
                    mapping.get('model_family'),
                    mapping.get('hardware_platform'),
                    mapping.get('compatibility_score'),
                    mapping.get('recommendation_rank'),
                    mapping.get('is_primary_recommendation'),
                    mapping.get('reason'),
                    mapping.get('mapping_id'),
                    mapping.get('metadata')
                )
            ).fetchone()
            
            return result[0] if result else -1
        except Exception as e:
            logger.error(f"Error storing hardware-model mapping: {str(e)}")
            raise
    
    def get_hardware_model_mappings(self, 
                                model_name: Optional[str] = None,
                                model_family: Optional[str] = None,
                                hardware_platform: Optional[str] = None,
                                is_primary: Optional[bool] = None,
                                limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve hardware-model mappings based on filters.
        
        Args:
            model_name: Optional model name filter
            model_family: Optional model family filter
            hardware_platform: Optional hardware platform filter
            is_primary: Optional filter for primary recommendations
            limit: Maximum number of records to return
            
        Returns:
            List of mappings matching the filters
        """
        try:
            conditions = []
            params = []
            
            if model_name:
                conditions.append("model_name = ?")
                params.append(model_name)
            
            if model_family:
                conditions.append("model_family = ?")
                params.append(model_family)
            
            if hardware_platform:
                conditions.append("hardware_platform = ?")
                params.append(hardware_platform)
            
            if is_primary is not None:
                conditions.append("is_primary_recommendation = ?")
                params.append(is_primary)
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            query = f"""
                SELECT id, timestamp, model_name, model_family, hardware_platform,
                       compatibility_score, recommendation_rank, is_primary_recommendation,
                       reason, mapping_id, metadata
                FROM hardware_model_mappings
                WHERE {where_clause}
                ORDER BY 
                    model_name,
                    recommendation_rank IS NULL, 
                    recommendation_rank,
                    compatibility_score DESC
                LIMIT ?
            """
            
            params.append(limit)
            
            result = self.conn.execute(query, params).fetchall()
            
            # Convert to list of dictionaries
            mappings = []
            for row in result:
                mapping = {
                    'id': row[0],
                    'timestamp': row[1],
                    'model_name': row[2],
                    'model_family': row[3],
                    'hardware_platform': row[4],
                    'compatibility_score': row[5],
                    'recommendation_rank': row[6],
                    'is_primary_recommendation': row[7],
                    'reason': row[8],
                    'mapping_id': row[9],
                    'metadata': row[10]
                }
                
                # Parse metadata if it's a JSON string
                if isinstance(mapping['metadata'], str) and mapping['metadata']:
                    try:
                        mapping['metadata'] = json.loads(mapping['metadata'])
                    except:
                        pass
                
                mappings.append(mapping)
            
            return mappings
        except Exception as e:
            logger.error(f"Error retrieving hardware-model mappings: {str(e)}")
            raise
    
    def store_prediction_model(self, prediction_model: Dict[str, Any]) -> int:
        """
        Store a prediction model.
        
        Args:
            prediction_model: Dictionary containing the prediction model data
            
        Returns:
            The ID of the stored prediction model
        """
        try:
            # Ensure timestamp is a datetime object
            if isinstance(prediction_model.get('timestamp'), str):
                prediction_model['timestamp'] = datetime.fromisoformat(prediction_model['timestamp'].replace('Z', '+00:00'))
            elif not prediction_model.get('timestamp'):
                prediction_model['timestamp'] = datetime.now()
            
            # Convert metadata to JSON string if it's a dictionary
            if isinstance(prediction_model.get('metadata'), dict):
                prediction_model['metadata'] = json.dumps(prediction_model['metadata'])
            
            # Convert features_list to JSON string if it's a list
            if isinstance(prediction_model.get('features_list'), list):
                prediction_model['features_list'] = json.dumps(prediction_model['features_list'])
            
            # Generate a model ID if not provided
            if not prediction_model.get('model_id'):
                prediction_model['model_id'] = f"model-{int(time.time())}-{hash(prediction_model.get('model_type', ''))}-{hash(prediction_model.get('target_metric', ''))}"
            
            # Insert the prediction model
            query = """
                INSERT INTO prediction_models (
                    timestamp, model_type, target_metric, hardware_platform,
                    model_family, serialized_model, features_list,
                    training_score, validation_score, test_score,
                    model_id, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                RETURNING id
            """
            
            result = self.conn.execute(
                query,
                (
                    prediction_model.get('timestamp'),
                    prediction_model.get('model_type'),
                    prediction_model.get('target_metric'),
                    prediction_model.get('hardware_platform'),
                    prediction_model.get('model_family'),
                    prediction_model.get('serialized_model'),
                    prediction_model.get('features_list'),
                    prediction_model.get('training_score'),
                    prediction_model.get('validation_score'),
                    prediction_model.get('test_score'),
                    prediction_model.get('model_id'),
                    prediction_model.get('metadata')
                )
            ).fetchone()
            
            return result[0] if result else -1
        except Exception as e:
            logger.error(f"Error storing prediction model: {str(e)}")
            raise
    
    def get_prediction_models(self, 
                           model_type: Optional[str] = None,
                           target_metric: Optional[str] = None,
                           hardware_platform: Optional[str] = None,
                           model_family: Optional[str] = None,
                           model_id: Optional[str] = None,
                           limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve prediction models based on filters.
        
        Args:
            model_type: Optional model type filter
            target_metric: Optional target metric filter
            hardware_platform: Optional hardware platform filter
            model_family: Optional model family filter
            model_id: Optional model ID filter
            limit: Maximum number of records to return
            
        Returns:
            List of prediction models matching the filters
        """
        try:
            conditions = []
            params = []
            
            if model_type:
                conditions.append("model_type = ?")
                params.append(model_type)
            
            if target_metric:
                conditions.append("target_metric = ?")
                params.append(target_metric)
            
            if hardware_platform:
                conditions.append("hardware_platform = ?")
                params.append(hardware_platform)
            
            if model_family:
                conditions.append("model_family = ?")
                params.append(model_family)
            
            if model_id:
                conditions.append("model_id = ?")
                params.append(model_id)
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            query = f"""
                SELECT id, timestamp, model_type, target_metric, hardware_platform,
                       model_family, serialized_model, features_list,
                       training_score, validation_score, test_score,
                       model_id, metadata
                FROM prediction_models
                WHERE {where_clause}
                ORDER BY timestamp DESC
                LIMIT ?
            """
            
            params.append(limit)
            
            result = self.conn.execute(query, params).fetchall()
            
            # Convert to list of dictionaries
            models = []
            for row in result:
                model = {
                    'id': row[0],
                    'timestamp': row[1],
                    'model_type': row[2],
                    'target_metric': row[3],
                    'hardware_platform': row[4],
                    'model_family': row[5],
                    'serialized_model': row[6],
                    'features_list': row[7],
                    'training_score': row[8],
                    'validation_score': row[9],
                    'test_score': row[10],
                    'model_id': row[11],
                    'metadata': row[12]
                }
                
                # Parse metadata and features_list if they're JSON strings
                for field in ['metadata', 'features_list']:
                    if isinstance(model[field], str) and model[field]:
                        try:
                            model[field] = json.loads(model[field])
                        except:
                            pass
                
                models.append(model)
            
            return models
        except Exception as e:
            logger.error(f"Error retrieving prediction models: {str(e)}")
            raise
    
    def store_prediction_error(self, error: Dict[str, Any]) -> int:
        """
        Store a prediction error record.
        
        Args:
            error: Dictionary containing the prediction error data
            
        Returns:
            The ID of the stored error record
        """
        try:
            # Ensure timestamp is a datetime object
            if isinstance(error.get('timestamp'), str):
                error['timestamp'] = datetime.fromisoformat(error['timestamp'].replace('Z', '+00:00'))
            elif not error.get('timestamp'):
                error['timestamp'] = datetime.now()
            
            # Convert metadata to JSON string if it's a dictionary
            if isinstance(error.get('metadata'), dict):
                error['metadata'] = json.dumps(error['metadata'])
            
            # Calculate relative error if not provided
            if 'relative_error' not in error and 'actual_value' in error and 'predicted_value' in error:
                actual = error.get('actual_value', 0)
                if actual != 0:
                    error['relative_error'] = abs(error.get('predicted_value', 0) - actual) / abs(actual)
                else:
                    error['relative_error'] = None
            
            # Insert the error record
            query = """
                INSERT INTO prediction_errors (
                    timestamp, prediction_id, measurement_id, model_name,
                    hardware_platform, metric, predicted_value, actual_value,
                    absolute_error, relative_error, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                RETURNING id
            """
            
            result = self.conn.execute(
                query,
                (
                    error.get('timestamp'),
                    error.get('prediction_id'),
                    error.get('measurement_id'),
                    error.get('model_name'),
                    error.get('hardware_platform'),
                    error.get('metric'),
                    error.get('predicted_value'),
                    error.get('actual_value'),
                    error.get('absolute_error'),
                    error.get('relative_error'),
                    error.get('metadata')
                )
            ).fetchone()
            
            return result[0] if result else -1
        except Exception as e:
            logger.error(f"Error storing prediction error: {str(e)}")
            raise
    
    def get_prediction_errors(self, 
                           model_name: Optional[str] = None,
                           hardware_platform: Optional[str] = None,
                           metric: Optional[str] = None,
                           prediction_id: Optional[str] = None,
                           measurement_id: Optional[str] = None,
                           limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve prediction errors based on filters.
        
        Args:
            model_name: Optional model name filter
            hardware_platform: Optional hardware platform filter
            metric: Optional metric filter
            prediction_id: Optional prediction ID filter
            measurement_id: Optional measurement ID filter
            limit: Maximum number of records to return
            
        Returns:
            List of prediction errors matching the filters
        """
        try:
            conditions = []
            params = []
            
            if model_name:
                conditions.append("model_name = ?")
                params.append(model_name)
            
            if hardware_platform:
                conditions.append("hardware_platform = ?")
                params.append(hardware_platform)
            
            if metric:
                conditions.append("metric = ?")
                params.append(metric)
            
            if prediction_id:
                conditions.append("prediction_id = ?")
                params.append(prediction_id)
            
            if measurement_id:
                conditions.append("measurement_id = ?")
                params.append(measurement_id)
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            query = f"""
                SELECT id, timestamp, prediction_id, measurement_id, model_name,
                       hardware_platform, metric, predicted_value, actual_value,
                       absolute_error, relative_error, metadata
                FROM prediction_errors
                WHERE {where_clause}
                ORDER BY timestamp DESC
                LIMIT ?
            """
            
            params.append(limit)
            
            result = self.conn.execute(query, params).fetchall()
            
            # Convert to list of dictionaries
            errors = []
            for row in result:
                error = {
                    'id': row[0],
                    'timestamp': row[1],
                    'prediction_id': row[2],
                    'measurement_id': row[3],
                    'model_name': row[4],
                    'hardware_platform': row[5],
                    'metric': row[6],
                    'predicted_value': row[7],
                    'actual_value': row[8],
                    'absolute_error': row[9],
                    'relative_error': row[10],
                    'metadata': row[11]
                }
                
                # Parse metadata if it's a JSON string
                if isinstance(error['metadata'], str) and error['metadata']:
                    try:
                        error['metadata'] = json.loads(error['metadata'])
                    except:
                        pass
                
                errors.append(error)
            
            return errors
        except Exception as e:
            logger.error(f"Error retrieving prediction errors: {str(e)}")
            raise
    
    def store_recommendation(self, recommendation: Dict[str, Any]) -> int:
        """
        Store a hardware recommendation.
        
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
            
            # Convert list fields to JSON strings
            for field in ['fallback_options', 'compatible_hardware']:
                if isinstance(recommendation.get(field), list):
                    recommendation[field] = json.dumps(recommendation[field])
            
            # Generate a recommendation ID if not provided
            if not recommendation.get('recommendation_id'):
                recommendation['recommendation_id'] = f"rec-{int(time.time())}-{hash(recommendation.get('model_name', ''))}"
            
            # Insert the recommendation
            query = """
                INSERT INTO recommendation_history (
                    timestamp, user_id, model_name, model_family,
                    batch_size, sequence_length, precision, mode,
                    primary_recommendation, fallback_options, compatible_hardware,
                    reason, recommendation_id, was_accepted, user_feedback,
                    metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                RETURNING id
            """
            
            result = self.conn.execute(
                query,
                (
                    recommendation.get('timestamp'),
                    recommendation.get('user_id'),
                    recommendation.get('model_name'),
                    recommendation.get('model_family'),
                    recommendation.get('batch_size'),
                    recommendation.get('sequence_length'),
                    recommendation.get('precision'),
                    recommendation.get('mode'),
                    recommendation.get('primary_recommendation'),
                    recommendation.get('fallback_options'),
                    recommendation.get('compatible_hardware'),
                    recommendation.get('reason'),
                    recommendation.get('recommendation_id'),
                    recommendation.get('was_accepted'),
                    recommendation.get('user_feedback'),
                    recommendation.get('metadata')
                )
            ).fetchone()
            
            return result[0] if result else -1
        except Exception as e:
            logger.error(f"Error storing recommendation: {str(e)}")
            raise
    
    def get_recommendations(self, 
                         model_name: Optional[str] = None,
                         model_family: Optional[str] = None,
                         user_id: Optional[str] = None,
                         primary_recommendation: Optional[str] = None,
                         was_accepted: Optional[bool] = None,
                         recommendation_id: Optional[str] = None,
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None,
                         limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve hardware recommendations based on filters.
        
        Args:
            model_name: Optional model name filter
            model_family: Optional model family filter
            user_id: Optional user ID filter
            primary_recommendation: Optional primary recommendation filter
            was_accepted: Optional filter for accepted recommendations
            recommendation_id: Optional recommendation ID filter
            start_time: Optional start time filter
            end_time: Optional end time filter
            limit: Maximum number of records to return
            
        Returns:
            List of recommendations matching the filters
        """
        try:
            conditions = []
            params = []
            
            if model_name:
                conditions.append("model_name = ?")
                params.append(model_name)
            
            if model_family:
                conditions.append("model_family = ?")
                params.append(model_family)
            
            if user_id:
                conditions.append("user_id = ?")
                params.append(user_id)
            
            if primary_recommendation:
                conditions.append("primary_recommendation = ?")
                params.append(primary_recommendation)
            
            if was_accepted is not None:
                conditions.append("was_accepted = ?")
                params.append(was_accepted)
            
            if recommendation_id:
                conditions.append("recommendation_id = ?")
                params.append(recommendation_id)
            
            if start_time:
                conditions.append("timestamp >= ?")
                params.append(start_time)
            
            if end_time:
                conditions.append("timestamp <= ?")
                params.append(end_time)
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            query = f"""
                SELECT id, timestamp, user_id, model_name, model_family,
                       batch_size, sequence_length, precision, mode,
                       primary_recommendation, fallback_options, compatible_hardware,
                       reason, recommendation_id, was_accepted, user_feedback,
                       metadata
                FROM recommendation_history
                WHERE {where_clause}
                ORDER BY timestamp DESC
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
                    'user_id': row[2],
                    'model_name': row[3],
                    'model_family': row[4],
                    'batch_size': row[5],
                    'sequence_length': row[6],
                    'precision': row[7],
                    'mode': row[8],
                    'primary_recommendation': row[9],
                    'fallback_options': row[10],
                    'compatible_hardware': row[11],
                    'reason': row[12],
                    'recommendation_id': row[13],
                    'was_accepted': row[14],
                    'user_feedback': row[15],
                    'metadata': row[16]
                }
                
                # Parse JSON fields
                for field in ['fallback_options', 'compatible_hardware', 'metadata']:
                    if isinstance(recommendation[field], str) and recommendation[field]:
                        try:
                            recommendation[field] = json.loads(recommendation[field])
                        except:
                            pass
                
                recommendations.append(recommendation)
            
            return recommendations
        except Exception as e:
            logger.error(f"Error retrieving recommendations: {str(e)}")
            raise
    
    def store_feature_importance(self, importance: Dict[str, Any]) -> int:
        """
        Store a feature importance record.
        
        Args:
            importance: Dictionary containing the feature importance data
            
        Returns:
            The ID of the stored feature importance record
        """
        try:
            # Ensure timestamp is a datetime object
            if isinstance(importance.get('timestamp'), str):
                importance['timestamp'] = datetime.fromisoformat(importance['timestamp'].replace('Z', '+00:00'))
            elif not importance.get('timestamp'):
                importance['timestamp'] = datetime.now()
            
            # Convert metadata to JSON string if it's a dictionary
            if isinstance(importance.get('metadata'), dict):
                importance['metadata'] = json.dumps(importance['metadata'])
            
            # Insert the feature importance record
            query = """
                INSERT INTO feature_importance (
                    timestamp, model_id, feature_name, importance_score,
                    rank, method, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                RETURNING id
            """
            
            result = self.conn.execute(
                query,
                (
                    importance.get('timestamp'),
                    importance.get('model_id'),
                    importance.get('feature_name'),
                    importance.get('importance_score'),
                    importance.get('rank'),
                    importance.get('method'),
                    importance.get('metadata')
                )
            ).fetchone()
            
            return result[0] if result else -1
        except Exception as e:
            logger.error(f"Error storing feature importance: {str(e)}")
            raise
    
    def get_feature_importance(self, 
                             model_id: Optional[str] = None,
                             feature_name: Optional[str] = None,
                             method: Optional[str] = None,
                             limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve feature importance records based on filters.
        
        Args:
            model_id: Optional model ID filter
            feature_name: Optional feature name filter
            method: Optional method filter
            limit: Maximum number of records to return
            
        Returns:
            List of feature importance records matching the filters
        """
        try:
            conditions = []
            params = []
            
            if model_id:
                conditions.append("model_id = ?")
                params.append(model_id)
            
            if feature_name:
                conditions.append("feature_name = ?")
                params.append(feature_name)
            
            if method:
                conditions.append("method = ?")
                params.append(method)
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            query = f"""
                SELECT id, timestamp, model_id, feature_name, importance_score,
                       rank, method, metadata
                FROM feature_importance
                WHERE {where_clause}
                ORDER BY model_id, rank IS NULL, rank, importance_score DESC
                LIMIT ?
            """
            
            params.append(limit)
            
            result = self.conn.execute(query, params).fetchall()
            
            # Convert to list of dictionaries
            importance_records = []
            for row in result:
                importance = {
                    'id': row[0],
                    'timestamp': row[1],
                    'model_id': row[2],
                    'feature_name': row[3],
                    'importance_score': row[4],
                    'rank': row[5],
                    'method': row[6],
                    'metadata': row[7]
                }
                
                # Parse metadata if it's a JSON string
                if isinstance(importance['metadata'], str) and importance['metadata']:
                    try:
                        importance['metadata'] = json.loads(importance['metadata'])
                    except:
                        pass
                
                importance_records.append(importance)
            
            return importance_records
        except Exception as e:
            logger.error(f"Error retrieving feature importance records: {str(e)}")
            raise
    
    def get_prediction_accuracy_stats(self, 
                                  model_name: Optional[str] = None,
                                  hardware_platform: Optional[str] = None,
                                  metric: Optional[str] = None,
                                  start_time: Optional[datetime] = None,
                                  end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Calculate prediction accuracy statistics.
        
        Args:
            model_name: Optional model name filter
            hardware_platform: Optional hardware platform filter
            metric: Optional metric filter
            start_time: Optional start time filter
            end_time: Optional end time filter
            
        Returns:
            Dictionary with prediction accuracy statistics
        """
        try:
            conditions = []
            params = []
            
            if model_name:
                conditions.append("model_name = ?")
                params.append(model_name)
            
            if hardware_platform:
                conditions.append("hardware_platform = ?")
                params.append(hardware_platform)
            
            if metric:
                conditions.append("metric = ?")
                params.append(metric)
            
            if start_time:
                conditions.append("timestamp >= ?")
                params.append(start_time)
            
            if end_time:
                conditions.append("timestamp <= ?")
                params.append(end_time)
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            # Query for basic statistics
            query = f"""
                SELECT
                    metric,
                    COUNT(*) as count,
                    AVG(absolute_error) as mean_absolute_error,
                    AVG(relative_error) as mean_relative_error,
                    STDDEV(absolute_error) as std_absolute_error,
                    MIN(absolute_error) as min_absolute_error,
                    MAX(absolute_error) as max_absolute_error,
                    AVG(predicted_value) as mean_predicted,
                    AVG(actual_value) as mean_actual
                FROM prediction_errors
                WHERE {where_clause}
                GROUP BY metric
            """
            
            result = self.conn.execute(query, params).fetchall()
            
            # Organize results by metric
            stats = {}
            for row in result:
                metric = row[0]
                stats[metric] = {
                    'count': row[1],
                    'mean_absolute_error': row[2],
                    'mean_relative_error': row[3],
                    'std_absolute_error': row[4],
                    'min_absolute_error': row[5],
                    'max_absolute_error': row[6],
                    'mean_predicted': row[7],
                    'mean_actual': row[8]
                }
                
                # Add bias (average prediction error with sign)
                query_bias = f"""
                    SELECT AVG(predicted_value - actual_value)
                    FROM prediction_errors
                    WHERE {where_clause} AND metric = ?
                """
                
                bias_params = params.copy()
                bias_params.append(metric)
                
                bias_result = self.conn.execute(query_bias, bias_params).fetchone()
                stats[metric]['bias'] = bias_result[0] if bias_result else 0
                
                # Calculate R² (coefficient of determination)
                query_r2 = f"""
                    WITH data AS (
                        SELECT 
                            actual_value, 
                            predicted_value,
                            AVG(actual_value) OVER () as avg_actual
                        FROM prediction_errors
                        WHERE {where_clause} AND metric = ?
                    )
                    SELECT 
                        1 - SUM(POWER(actual_value - predicted_value, 2)) / 
                            NULLIF(SUM(POWER(actual_value - avg_actual, 2)), 0)
                    FROM data
                """
                
                r2_params = params.copy()
                r2_params.append(metric)
                
                r2_result = self.conn.execute(query_r2, r2_params).fetchone()
                stats[metric]['r_squared'] = r2_result[0] if r2_result else None
            
            # Add overall statistics
            if stats:
                all_metrics = list(stats.keys())
                overall = {
                    'count': sum(stats[m]['count'] for m in all_metrics),
                    'metrics': all_metrics,
                    'overall_mean_relative_error': sum(stats[m]['mean_relative_error'] * stats[m]['count'] for m in all_metrics) / 
                                                sum(stats[m]['count'] for m in all_metrics) if sum(stats[m]['count'] for m in all_metrics) > 0 else 0
                }
                stats['overall'] = overall
            
            return stats
        except Exception as e:
            logger.error(f"Error calculating prediction accuracy statistics: {str(e)}")
            raise
    
    def generate_sample_data(self, num_models: int = 5) -> None:
        """
        Generate sample data for testing.
        
        Args:
            num_models: Number of sample models to generate
        """
        try:
            # Sample model names and families
            model_configs = [
                {"name": "bert-base-uncased", "family": "embedding"},
                {"name": "gpt2", "family": "text_generation"},
                {"name": "t5-small", "family": "text_generation"},
                {"name": "vit-base-patch16-224", "family": "vision"},
                {"name": "whisper-small", "family": "audio"},
                {"name": "clip-vit-base-patch32", "family": "multimodal"},
                {"name": "llama-7b", "family": "text_generation"},
                {"name": "roberta-base", "family": "embedding"},
                {"name": "stable-diffusion-v1-5", "family": "diffusion"},
                {"name": "resnet-50", "family": "vision"}
            ]
            
            # Sample hardware platforms
            hardware_platforms = [
                "cpu",
                "cuda",
                "rocm",
                "mps",
                "openvino",
                "webgpu",
                "webnn"
            ]
            
            # Sample batch sizes
            batch_sizes = [1, 4, 8, 16, 32]
            
            # Sample precision options
            precision_options = ["fp32", "fp16", "int8"]
            
            # Sample prediction sources
            prediction_sources = [
                "hardware_model_predictor",
                "ml_regression_model",
                "ensemble_predictor",
                "heuristic_model",
                "historical_data"
            ]
            
            # Sample models
            selected_models = np.random.choice(model_configs, size=min(num_models, len(model_configs)), replace=False)
            
            for model_config in selected_models:
                model_name = model_config["name"]
                model_family = model_config["family"]
                
                # Generate hardware mappings
                for i, hw in enumerate(np.random.choice(hardware_platforms, size=min(4, len(hardware_platforms)), replace=False)):
                    compatibility_score = np.random.uniform(0.3, 0.95)
                    is_primary = (i == 0)  # First hardware is primary
                    
                    mapping = {
                        'model_name': model_name,
                        'model_family': model_family,
                        'hardware_platform': hw,
                        'compatibility_score': compatibility_score,
                        'recommendation_rank': i + 1 if is_primary else i + 2,
                        'is_primary_recommendation': is_primary,
                        'reason': f"Based on {model_family} model characteristics and {hw} capabilities",
                        'metadata': {
                            'score_components': {
                                'compute_compatibility': np.random.uniform(0.5, 1.0),
                                'memory_compatibility': np.random.uniform(0.5, 1.0),
                                'precision_support': np.random.uniform(0.5, 1.0)
                            }
                        }
                    }
                    
                    self.store_hardware_model_mapping(mapping)
                
                # Generate predictions and measurements
                for hw in np.random.choice(hardware_platforms, size=min(3, len(hardware_platforms)), replace=False):
                    for batch_size in np.random.choice(batch_sizes, size=2, replace=False):
                        for precision in np.random.choice(precision_options, size=1):
                            # Base performance values
                            base_throughput = 1000 / batch_size if batch_size > 0 else 1000
                            base_latency = 10 * batch_size
                            base_memory = 100 * batch_size
                            
                            # Hardware factors
                            if hw == "cuda":
                                hw_factor = 2.0
                            elif hw == "rocm":
                                hw_factor = 1.8
                            elif hw == "mps":
                                hw_factor = 1.5
                            elif hw == "webgpu":
                                hw_factor = 1.3
                            elif hw == "webnn":
                                hw_factor = 1.2
                            elif hw == "openvino":
                                hw_factor = 1.4
                            else:
                                hw_factor = 1.0
                            
                            # Precision factors
                            if precision == "fp16":
                                precision_factor = 1.5
                                memory_factor = 0.5
                            elif precision == "int8":
                                precision_factor = 2.0
                                memory_factor = 0.25
                            else:
                                precision_factor = 1.0
                                memory_factor = 1.0
                            
                            # Generate prediction
                            throughput_pred = base_throughput * hw_factor * precision_factor * (1 + np.random.normal(0, 0.1))
                            latency_pred = base_latency / (hw_factor * precision_factor) * (1 + np.random.normal(0, 0.1))
                            memory_pred = base_memory * memory_factor * (1 + np.random.normal(0, 0.05))
                            
                            prediction = {
                                'model_name': model_name,
                                'model_family': model_family,
                                'hardware_platform': hw,
                                'batch_size': batch_size,
                                'sequence_length': 128,
                                'precision': precision,
                                'mode': 'inference',
                                'throughput': throughput_pred,
                                'latency': latency_pred,
                                'memory_usage': memory_pred,
                                'confidence_score': np.random.uniform(0.7, 0.95),
                                'prediction_source': np.random.choice(prediction_sources),
                                'metadata': {
                                    'prediction_time_ms': np.random.randint(5, 50),
                                    'features_used': ['model_size', 'hardware_platform', 'batch_size', 'precision']
                                }
                            }
                            
                            prediction_id = self.store_prediction(prediction)
                            
                            # Generate actual measurement with some deviation from prediction
                            throughput_actual = throughput_pred * np.random.uniform(0.85, 1.15)
                            latency_actual = latency_pred * np.random.uniform(0.85, 1.15)
                            memory_actual = memory_pred * np.random.uniform(0.9, 1.1)
                            
                            measurement = {
                                'model_name': model_name,
                                'model_family': model_family,
                                'hardware_platform': hw,
                                'batch_size': batch_size,
                                'sequence_length': 128,
                                'precision': precision,
                                'mode': 'inference',
                                'throughput': throughput_actual,
                                'latency': latency_actual,
                                'memory_usage': memory_actual,
                                'measurement_source': 'benchmark',
                                'metadata': {
                                    'device_temperature': np.random.uniform(40, 80),
                                    'runtime_environment': 'Python 3.10',
                                    'benchmark_repetitions': 10
                                }
                            }
                            
                            measurement_id = self.store_measurement(measurement)
                            
                            # Generate prediction errors
                            for metric_name, pred_val, actual_val in [
                                ('throughput', throughput_pred, throughput_actual),
                                ('latency', latency_pred, latency_actual),
                                ('memory_usage', memory_pred, memory_actual)
                            ]:
                                absolute_error = abs(pred_val - actual_val)
                                relative_error = abs(pred_val - actual_val) / abs(actual_val) if abs(actual_val) > 1e-10 else 0
                                
                                error = {
                                    'prediction_id': str(prediction_id),
                                    'measurement_id': str(measurement_id),
                                    'model_name': model_name,
                                    'hardware_platform': hw,
                                    'metric': metric_name,
                                    'predicted_value': pred_val,
                                    'actual_value': actual_val,
                                    'absolute_error': absolute_error,
                                    'relative_error': relative_error,
                                    'metadata': {
                                        'batch_size': batch_size,
                                        'precision': precision
                                    }
                                }
                                
                                self.store_prediction_error(error)
                
                # Generate prediction models
                for target_metric in ['throughput', 'latency', 'memory_usage']:
                    # Randomly select some hardware platforms to create models for
                    for hw in np.random.choice(hardware_platforms, size=2, replace=False):
                        model_type = np.random.choice([
                            'RandomForestRegressor',
                            'GradientBoostingRegressor',
                            'VotingRegressor',
                            'StackingRegressor'
                        ])
                        
                        # Create a dummy serialized model (would be a real serialized model in practice)
                        serialized_model = b'dummy_serialized_model_data'
                        
                        # Sample feature list
                        features = [
                            'model_size',
                            'batch_size',
                            'sequence_length',
                            'precision_numeric',
                            'hardware_compute_score',
                            'hardware_memory_score',
                            'parallelism_score',
                            'compute_memory_ratio'
                        ]
                        
                        # Generate training, validation, and test scores
                        base_score = np.random.uniform(0.7, 0.9)
                        
                        model = {
                            'model_type': model_type,
                            'target_metric': target_metric,
                            'hardware_platform': hw,
                            'model_family': model_family,
                            'serialized_model': serialized_model,
                            'features_list': features,
                            'training_score': base_score,
                            'validation_score': base_score * np.random.uniform(0.9, 1.0),
                            'test_score': base_score * np.random.uniform(0.85, 0.98),
                            'metadata': {
                                'training_time_seconds': np.random.uniform(10, 300),
                                'hyperparameters': {
                                    'n_estimators': np.random.randint(50, 200),
                                    'max_depth': np.random.randint(3, 10),
                                    'learning_rate': np.random.uniform(0.01, 0.2)
                                }
                            }
                        }
                        
                        model_id = self.store_prediction_model(model)
                        
                        # Generate feature importance records
                        for i, feature in enumerate(np.random.choice(features, size=len(features), replace=False)):
                            importance_score = np.random.uniform(0.01, 0.3)
                            
                            importance = {
                                'model_id': str(model_id),
                                'feature_name': feature,
                                'importance_score': importance_score,
                                'rank': i + 1,
                                'method': np.random.choice(['permutation', 'gini', 'shap']),
                                'metadata': {
                                    'confidence_interval': [
                                        importance_score - np.random.uniform(0.01, 0.05),
                                        importance_score + np.random.uniform(0.01, 0.05)
                                    ]
                                }
                            }
                            
                            self.store_feature_importance(importance)
                
                # Generate some recommendations
                for _ in range(3):
                    # Select random configuration
                    batch_size = np.random.choice(batch_sizes)
                    precision = np.random.choice(precision_options)
                    
                    # Get compatible hardware (random subset for demo)
                    compatible_hw = list(np.random.choice(hardware_platforms, size=min(4, len(hardware_platforms)), replace=False))
                    primary_hw = compatible_hw[0]
                    fallback_hw = compatible_hw[1:] if len(compatible_hw) > 1 else []
                    
                    recommendation = {
                        'user_id': f"user-{np.random.randint(1, 10)}",
                        'model_name': model_name,
                        'model_family': model_family,
                        'batch_size': batch_size,
                        'sequence_length': 128,
                        'precision': precision,
                        'mode': 'inference',
                        'primary_recommendation': primary_hw,
                        'fallback_options': fallback_hw,
                        'compatible_hardware': compatible_hw,
                        'reason': f"Based on {model_family} model characteristics and optimal performance",
                        'was_accepted': np.random.choice([True, False], p=[0.8, 0.2]),
                        'user_feedback': np.random.choice([None, "Works great!", "Too slow", "Memory issues"], p=[0.7, 0.1, 0.1, 0.1]),
                        'metadata': {
                            'request_source': np.random.choice(['web', 'api', 'cli']),
                            'timing_ms': np.random.randint(10, 100)
                        }
                    }
                    
                    self.store_recommendation(recommendation)
            
            logger.info(f"Generated sample data for {num_models} models")
        except Exception as e:
            logger.error(f"Error generating sample data: {str(e)}")
            raise