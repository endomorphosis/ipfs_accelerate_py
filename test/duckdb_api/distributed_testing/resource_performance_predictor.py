#!/usr/bin/env python3
"""
Distributed Testing Framework - Resource Performance Predictor

This module provides ML-based prediction for resource requirements based on historical task execution data.
It enables optimized resource allocation by predicting CPU, memory, and GPU requirements for tasks.

Core responsibilities:
- Collection of task execution metrics
- Model training for resource prediction
- Resource requirement prediction for new tasks
- Performance analysis and scaling factor calculation

Usage:
    # Import and initialize
    from data.duckdb.distributed_testing.resource_performance_predictor import ResourcePerformancePredictor
    
    # Create predictor
    predictor = ResourcePerformancePredictor()
    
    # Record task execution data
    predictor.record_task_execution(
        task_id="task-1",
        execution_data={
            "model_type": "text_embedding",
            "model_name": "bert-base-uncased",
            "batch_size": 32,
            "cpu_cores_used": 2,
            "memory_mb_used": 3800,
            "gpu_memory_mb_used": 1950,
            "execution_time_ms": 145,
            "success": True
        }
    )
    
    # Predict resource requirements for a new task
    prediction = predictor.predict_resource_requirements({
        "model_type": "text_embedding",
        "model_name": "bert-base-uncased",
        "batch_size": 64
    })
    
    # Get scaling factor for resources
    scaling = predictor.get_resource_scaling_factor({
        "model_type": "text_embedding",
        "batch_size": 32,
        "batch_size_target": 64
    })
"""

import os
import sys
import json
import time
import uuid
import math
import logging
import threading
import traceback
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
import sqlite3

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("resource_performance_predictor")

# Add parent directory to path to import modules from parent
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Try to import optional ML dependencies
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    logger.warning("Pandas not available. Using simplified prediction methods.")
    PANDAS_AVAILABLE = False

try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    ML_AVAILABLE = True
except ImportError:
    logger.warning("Scikit-learn not available. Using simplified prediction methods.")
    ML_AVAILABLE = False

# Default values for resource requirements (fallback when no data available)
DEFAULT_CPU_CORES = {
    "text_embedding": 2,
    "text_generation": 4,
    "vision": 2,
    "audio": 2,
    "multimodal": 4,
    "default": 2
}

DEFAULT_MEMORY_MB = {
    "text_embedding": 4096,
    "text_generation": 8192,
    "vision": 4096,
    "audio": 4096,
    "multimodal": 8192,
    "default": 4096
}

DEFAULT_GPU_MEMORY_MB = {
    "text_embedding": 2048,
    "text_generation": 4096,
    "vision": 2048,
    "audio": 2048,
    "multimodal": 4096,
    "default": 2048
}

# Scaling factors for batch size (used when not enough data available)
DEFAULT_BATCH_SCALING = {
    "cpu_cores": 0.3,      # CPU scales sub-linearly with batch size
    "memory_mb": 0.8,      # Memory scales almost linearly with batch size
    "gpu_memory_mb": 0.9,  # GPU memory scales almost linearly with batch size
}

# Database schema version
SCHEMA_VERSION = 1


class ResourcePerformancePredictor:
    """
    Resource Performance Predictor for the distributed testing framework.
    
    Predicts resource requirements based on historical task execution data
    and model characteristics.
    """
    
    def __init__(self, database_path: Optional[str] = None):
        """
        Initialize the Resource Performance Predictor.
        
        Args:
            database_path: Path to SQLite database file for persistent storage
        """
        # Initialize database connection
        self.database_path = database_path or ":memory:"
        self.conn = self._initialize_database()
        
        # Initialize ML models if available
        self.models = {}
        self.scalers = {}
        self.ml_initialized = False
        
        # Data stats
        self.data_count = self._get_data_count()
        self.last_training_time = 0
        self.training_interval = 3600  # Re-train models every hour (if new data available)
        
        # Training status
        self.training_lock = threading.Lock()
        self.training_enabled = True
        
        # Start background training thread
        self.running = True
        self.training_thread = threading.Thread(target=self._training_loop)
        self.training_thread.daemon = True
        self.training_thread.start()
        
        logger.info(f"Resource Performance Predictor initialized with {self.data_count} historical records")
    
    def record_task_execution(self, task_id: str, execution_data: Dict[str, Any]) -> bool:
        """
        Record task execution data for future predictions.
        
        Args:
            task_id: Unique identifier for the task
            execution_data: Dictionary of execution metrics
                {
                    "model_type": str, 
                    "model_name": str,
                    "batch_size": int,
                    "cpu_cores_used": int,
                    "memory_mb_used": int,
                    "gpu_memory_mb_used": int,
                    "execution_time_ms": int,
                    "success": bool,
                    ...
                }
        
        Returns:
            bool: Success status
        """
        try:
            # Extract required fields
            model_type = execution_data.get("model_type", "unknown")
            model_name = execution_data.get("model_name", "unknown")
            batch_size = execution_data.get("batch_size", 1)
            
            # Resource usage
            cpu_cores = execution_data.get("cpu_cores_used", 0)
            memory_mb = execution_data.get("memory_mb_used", 0)
            gpu_memory_mb = execution_data.get("gpu_memory_mb_used", 0)
            
            # Performance metrics
            execution_time_ms = execution_data.get("execution_time_ms", 0)
            success = 1 if execution_data.get("success", True) else 0
            
            # Additional fields (convert to JSON for storage)
            additional_data = json.dumps({k: v for k, v in execution_data.items() 
                                         if k not in ["model_type", "model_name", "batch_size", 
                                                     "cpu_cores_used", "memory_mb_used", "gpu_memory_mb_used",
                                                     "execution_time_ms", "success"]})
            
            # Insert into database
            with self.conn:
                self.conn.execute(
                    "INSERT INTO execution_data (task_id, timestamp, model_type, model_name, batch_size, "
                    "cpu_cores, memory_mb, gpu_memory_mb, execution_time_ms, success, additional_data) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (task_id, time.time(), model_type, model_name, batch_size, 
                     cpu_cores, memory_mb, gpu_memory_mb, execution_time_ms, success, additional_data)
                )
            
            # Update data count
            self.data_count += 1
            
            logger.debug(f"Recorded execution data for task {task_id} ({model_type}: {model_name}, batch_size: {batch_size})")
            return True
        
        except Exception as e:
            logger.error(f"Error recording task execution data: {e}")
            logger.debug(traceback.format_exc())
            return False
    
    def predict_resource_requirements(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict resource requirements for a task based on historical data.
        
        Args:
            task_data: Dictionary of task characteristics
                {
                    "model_type": str,
                    "model_name": str,
                    "batch_size": int,
                    ...
                }
        
        Returns:
            dict: Predicted resource requirements
                {
                    "cpu_cores": int,
                    "memory_mb": int,
                    "gpu_memory_mb": int,
                    "confidence": float,
                    "prediction_method": str,
                    "execution_time_ms": int (optional)
                }
        """
        try:
            # Extract task characteristics
            model_type = task_data.get("model_type", "unknown")
            model_name = task_data.get("model_name", "unknown")
            batch_size = task_data.get("batch_size", 1)
            
            # Choose prediction method based on available data and dependencies
            if ML_AVAILABLE and PANDAS_AVAILABLE and self.data_count >= 10 and self.ml_initialized:
                # Use ML-based prediction
                prediction = self._predict_with_ml(task_data)
                prediction["prediction_method"] = "ml"
            elif self.data_count >= 5:
                # Use simple statistics-based prediction
                prediction = self._predict_with_stats(task_data)
                prediction["prediction_method"] = "stats"
            else:
                # Use defaults with scaling for batch size
                prediction = self._predict_with_defaults(task_data)
                prediction["prediction_method"] = "defaults"
            
            # Make sure all required fields are present
            for field in ["cpu_cores", "memory_mb", "gpu_memory_mb"]:
                if field not in prediction:
                    # Use defaults if field is missing
                    if field == "cpu_cores":
                        prediction[field] = DEFAULT_CPU_CORES.get(model_type, DEFAULT_CPU_CORES["default"])
                    elif field == "memory_mb":
                        prediction[field] = DEFAULT_MEMORY_MB.get(model_type, DEFAULT_MEMORY_MB["default"])
                    elif field == "gpu_memory_mb":
                        prediction[field] = DEFAULT_GPU_MEMORY_MB.get(model_type, DEFAULT_GPU_MEMORY_MB["default"])
            
            # Ensure values are reasonable (apply bounds)
            prediction["cpu_cores"] = max(1, min(32, prediction["cpu_cores"]))
            prediction["memory_mb"] = max(512, min(65536, prediction["memory_mb"]))
            prediction["gpu_memory_mb"] = max(0, min(32768, prediction["gpu_memory_mb"]))
            
            # Add prediction metadata
            prediction["timestamp"] = time.time()
            prediction["model_type"] = model_type
            prediction["model_name"] = model_name
            prediction["batch_size"] = batch_size
            
            logger.debug(f"Predicted resources for {model_type}:{model_name} (batch={batch_size}): "
                         f"CPU={prediction['cpu_cores']}, Memory={prediction['memory_mb']}MB, "
                         f"GPU={prediction['gpu_memory_mb']}MB")
            
            return prediction
        
        except Exception as e:
            logger.error(f"Error predicting resource requirements: {e}")
            logger.debug(traceback.format_exc())
            
            # Fallback to defaults
            model_type = task_data.get("model_type", "default")
            return {
                "cpu_cores": DEFAULT_CPU_CORES.get(model_type, DEFAULT_CPU_CORES["default"]),
                "memory_mb": DEFAULT_MEMORY_MB.get(model_type, DEFAULT_MEMORY_MB["default"]),
                "gpu_memory_mb": DEFAULT_GPU_MEMORY_MB.get(model_type, DEFAULT_GPU_MEMORY_MB["default"]),
                "confidence": 0.5,
                "prediction_method": "fallback",
                "model_type": model_type,
                "model_name": task_data.get("model_name", "unknown"),
                "batch_size": task_data.get("batch_size", 1),
                "timestamp": time.time()
            }
    
    def get_resource_scaling_factor(self, scaling_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate resource scaling factors based on historical data.
        
        Args:
            scaling_data: Dictionary of scaling parameters
                {
                    "model_type": str,
                    "batch_size": int,
                    "batch_size_target": int,
                    "model_name": str (optional)
                }
        
        Returns:
            dict: Scaling factors for different resource types
                {
                    "cpu_cores": float,
                    "memory_mb": float,
                    "gpu_memory_mb": float,
                    "confidence": float
                }
        """
        try:
            # Extract scaling parameters
            model_type = scaling_data.get("model_type", "unknown")
            batch_size_source = scaling_data.get("batch_size", 1)
            batch_size_target = scaling_data.get("batch_size_target", batch_size_source * 2)
            model_name = scaling_data.get("model_name")
            
            # If source and target are the same, return 1.0 scaling factors
            if batch_size_source == batch_size_target:
                return {
                    "cpu_cores": 1.0,
                    "memory_mb": 1.0,
                    "gpu_memory_mb": 1.0,
                    "confidence": 1.0
                }
            
            # Determine batch size ratio
            batch_ratio = batch_size_target / max(1, batch_size_source)
            
            # Query database for scaling data if we have enough records
            if self.data_count >= 10:
                scaling_factors = self._get_scaling_factors_from_data(model_type, model_name, batch_size_source, batch_size_target)
                if scaling_factors:
                    return scaling_factors
            
            # Fallback to theoretical scaling factors with batch size ratio
            # Different resources scale differently with batch size
            result = {
                "cpu_cores": max(1.0, batch_ratio ** DEFAULT_BATCH_SCALING["cpu_cores"]),
                "memory_mb": max(1.0, batch_ratio ** DEFAULT_BATCH_SCALING["memory_mb"]),
                "gpu_memory_mb": max(1.0, batch_ratio ** DEFAULT_BATCH_SCALING["gpu_memory_mb"]),
                "confidence": 0.7
            }
            
            logger.debug(f"Calculated scaling factors for {model_type} from batch {batch_size_source} "
                         f"to {batch_size_target}: {result}")
            
            return result
        
        except Exception as e:
            logger.error(f"Error calculating resource scaling factors: {e}")
            logger.debug(traceback.format_exc())
            
            # Return default scaling based on batch ratio
            batch_ratio = scaling_data.get("batch_size_target", 2) / max(1, scaling_data.get("batch_size", 1))
            return {
                "cpu_cores": max(1.0, batch_ratio ** DEFAULT_BATCH_SCALING["cpu_cores"]),
                "memory_mb": max(1.0, batch_ratio ** DEFAULT_BATCH_SCALING["memory_mb"]),
                "gpu_memory_mb": max(1.0, batch_ratio ** DEFAULT_BATCH_SCALING["gpu_memory_mb"]),
                "confidence": 0.5
            }
    
    def train_models(self) -> bool:
        """
        Train or update ML models with latest execution data.
        
        Returns:
            bool: Success status
        """
        if not ML_AVAILABLE or not PANDAS_AVAILABLE:
            logger.warning("ML libraries not available. Cannot train models.")
            return False
        
        if self.data_count < 10:
            logger.warning("Not enough data to train models. Need at least 10 data points.")
            return False
        
        with self.training_lock:
            try:
                # Get data from database
                query = "SELECT * FROM execution_data WHERE success = 1"
                data = pd.read_sql_query(query, self.conn)
                
                if len(data) < 10:
                    logger.warning(f"Not enough successful executions to train models ({len(data)} available)")
                    return False
                
                # Process data
                # Convert model_type to categorical
                data['model_type_cat'] = data['model_type'].astype('category').cat.codes
                
                # Target variables
                targets = ['cpu_cores', 'memory_mb', 'gpu_memory_mb']
                
                # Features (add more as needed)
                features = ['model_type_cat', 'batch_size']
                
                # Train models for each target
                for target in targets:
                    # Skip if not enough non-zero values
                    if (data[target] > 0).sum() < 5:
                        logger.warning(f"Not enough non-zero data for {target}. Skipping model training.")
                        continue
                    
                    # Prepare data
                    X = data[features]
                    y = data[target]
                    
                    # Create model pipeline
                    model = Pipeline([
                        ('scaler', StandardScaler()),
                        ('regressor', GradientBoostingRegressor(n_estimators=100, max_depth=3))
                    ])
                    
                    # Train model
                    model.fit(X, y)
                    
                    # Store model
                    self.models[target] = model
                
                # Update training metadata
                self.ml_initialized = True
                self.last_training_time = time.time()
                
                logger.info(f"ML models trained successfully with {len(data)} data points")
                return True
            
            except Exception as e:
                logger.error(f"Error training ML models: {e}")
                logger.debug(traceback.format_exc())
                return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistical information about the predictor.
        
        Returns:
            dict: Statistical information
        """
        try:
            stats = {
                "data_count": self.data_count,
                "ml_initialized": self.ml_initialized,
                "last_training_time": self.last_training_time,
                "database_path": self.database_path,
                "model_counts": {}
            }
            
            # Get model type counts
            with self.conn:
                cursor = self.conn.execute(
                    "SELECT model_type, COUNT(*) FROM execution_data GROUP BY model_type"
                )
                for model_type, count in cursor.fetchall():
                    stats["model_counts"][model_type] = count
            
            # Get prediction accuracy if enough data
            if self.data_count >= 20:
                with self.conn:
                    cursor = self.conn.execute(
                        "SELECT AVG(ABS(cpu_cores - predicted_cpu_cores) / cpu_cores), "
                        "AVG(ABS(memory_mb - predicted_memory_mb) / memory_mb), "
                        "AVG(ABS(gpu_memory_mb - predicted_gpu_memory_mb) / gpu_memory_mb) "
                        "FROM prediction_validation WHERE cpu_cores > 0 AND memory_mb > 0"
                    )
                    cpu_error, mem_error, gpu_error = cursor.fetchone()
                    
                    stats["accuracy"] = {
                        "cpu_cores": 1.0 - (cpu_error if cpu_error is not None else 0),
                        "memory_mb": 1.0 - (mem_error if mem_error is not None else 0),
                        "gpu_memory_mb": 1.0 - (gpu_error if gpu_error is not None else 0)
                    }
            
            return stats
        
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            logger.debug(traceback.format_exc())
            return {"data_count": self.data_count, "error": str(e)}
    
    def cleanup(self) -> None:
        """
        Cleanup resources and stop background thread.
        """
        self.running = False
        
        # Wait for thread to terminate
        if self.training_thread.is_alive():
            self.training_thread.join(timeout=5.0)
        
        # Close database connection
        if self.conn:
            self.conn.close()
        
        logger.info("Resource Performance Predictor cleaned up")
    
    # Internal helper methods
    def _initialize_database(self) -> sqlite3.Connection:
        """Initialize the SQLite database."""
        conn = sqlite3.connect(self.database_path, check_same_thread=False)
        
        # Create tables if they don't exist
        with conn:
            # Table for execution data
            conn.execute("""
                CREATE TABLE IF NOT EXISTS execution_data (
                    id INTEGER PRIMARY KEY,
                    task_id TEXT,
                    timestamp REAL,
                    model_type TEXT,
                    model_name TEXT,
                    batch_size INTEGER,
                    cpu_cores REAL,
                    memory_mb REAL,
                    gpu_memory_mb REAL,
                    execution_time_ms REAL,
                    success INTEGER,
                    additional_data TEXT
                )
            """)
            
            # Table for prediction validation
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prediction_validation (
                    id INTEGER PRIMARY KEY,
                    task_id TEXT,
                    timestamp REAL,
                    model_type TEXT,
                    model_name TEXT,
                    batch_size INTEGER,
                    cpu_cores REAL,
                    memory_mb REAL,
                    gpu_memory_mb REAL,
                    predicted_cpu_cores REAL,
                    predicted_memory_mb REAL,
                    predicted_gpu_memory_mb REAL,
                    prediction_method TEXT
                )
            """)
            
            # Table for metadata
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            
            # Check if schema version is stored
            cursor = conn.execute("SELECT value FROM metadata WHERE key = 'schema_version'")
            row = cursor.fetchone()
            
            if row is None:
                # First time initialization
                conn.execute(
                    "INSERT INTO metadata (key, value) VALUES (?, ?)",
                    ("schema_version", str(SCHEMA_VERSION))
                )
                conn.execute(
                    "INSERT INTO metadata (key, value) VALUES (?, ?)",
                    ("creation_time", str(time.time()))
                )
            elif int(row[0]) < SCHEMA_VERSION:
                # Schema migration needed
                self._migrate_schema(conn, int(row[0]))
                conn.execute(
                    "UPDATE metadata SET value = ? WHERE key = 'schema_version'",
                    (str(SCHEMA_VERSION),)
                )
        
        return conn
    
    def _migrate_schema(self, conn: sqlite3.Connection, old_version: int) -> None:
        """Migrate database schema from old version to current version."""
        # Example migration; add actual migrations as needed
        if old_version < 1:
            # Migration to version 1
            with conn:
                conn.execute("ALTER TABLE execution_data ADD COLUMN additional_data TEXT")
        
        # Add more migrations as needed
    
    def _get_data_count(self) -> int:
        """Get count of execution data records."""
        with self.conn:
            cursor = self.conn.execute("SELECT COUNT(*) FROM execution_data")
            return cursor.fetchone()[0]
    
    def _predict_with_ml(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict using ML models."""
        # Extract features
        model_type = task_data.get("model_type", "unknown")
        model_name = task_data.get("model_name", "unknown")
        batch_size = task_data.get("batch_size", 1)
        
        # Convert model_type to categorical code
        # First, get all model types from database
        with self.conn:
            cursor = self.conn.execute("SELECT DISTINCT model_type FROM execution_data")
            model_types = [row[0] for row in cursor.fetchall()]
        
        # Map model_type to categorical code
        if model_type in model_types:
            model_type_code = model_types.index(model_type)
        else:
            # Unknown model type, use default prediction
            return self._predict_with_defaults(task_data)
        
        # Prepare features
        features = np.array([[model_type_code, batch_size]])
        
        # Make predictions
        predictions = {}
        confidence = 0.8  # Base confidence
        
        for target in ['cpu_cores', 'memory_mb', 'gpu_memory_mb']:
            if target in self.models:
                try:
                    # Predict
                    predicted_value = self.models[target].predict(features)[0]
                    
                    # Apply bounds (resource predictions should never be negative)
                    predicted_value = max(0, predicted_value)
                    
                    # Round to appropriate precision
                    if target == 'cpu_cores':
                        predicted_value = max(1, round(predicted_value))
                    else:
                        predicted_value = max(1, int(predicted_value))
                    
                    predictions[target] = predicted_value
                except Exception as e:
                    logger.error(f"Error using ML model for {target}: {e}")
                    logger.debug(traceback.format_exc())
                    confidence -= 0.1  # Reduce confidence on error
                    
                    # Fallback to stats prediction for this target
                    stats_prediction = self._predict_with_stats(task_data)
                    predictions[target] = stats_prediction.get(target, DEFAULT_CPU_CORES.get(model_type, DEFAULT_CPU_CORES["default"]) if target == 'cpu_cores' else 
                                                              DEFAULT_MEMORY_MB.get(model_type, DEFAULT_MEMORY_MB["default"]) if target == 'memory_mb' else 
                                                              DEFAULT_GPU_MEMORY_MB.get(model_type, DEFAULT_GPU_MEMORY_MB["default"]))
            else:
                # Model not available, use stats prediction for this target
                stats_prediction = self._predict_with_stats(task_data)
                predictions[target] = stats_prediction.get(target, DEFAULT_CPU_CORES.get(model_type, DEFAULT_CPU_CORES["default"]) if target == 'cpu_cores' else 
                                                          DEFAULT_MEMORY_MB.get(model_type, DEFAULT_MEMORY_MB["default"]) if target == 'memory_mb' else 
                                                          DEFAULT_GPU_MEMORY_MB.get(model_type, DEFAULT_GPU_MEMORY_MB["default"]))
                confidence -= 0.1  # Reduce confidence
        
        # Add confidence
        predictions["confidence"] = max(0.5, confidence)
        
        return predictions
    
    def _predict_with_stats(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict using statistical methods."""
        # Extract task characteristics
        model_type = task_data.get("model_type", "unknown")
        model_name = task_data.get("model_name", "unknown")
        batch_size = task_data.get("batch_size", 1)
        
        # Query database for similar tasks
        try:
            # First, try exact match on model name and batch size
            with self.conn:
                cursor = self.conn.execute(
                    "SELECT AVG(cpu_cores), AVG(memory_mb), AVG(gpu_memory_mb), AVG(execution_time_ms), COUNT(*) "
                    "FROM execution_data WHERE model_name = ? AND batch_size = ? AND success = 1",
                    (model_name, batch_size)
                )
                row = cursor.fetchone()
                
                exact_match = (row[4] >= 3)  # Need at least 3 samples for reliable stats
                
                if exact_match:
                    return {
                        "cpu_cores": max(1, int(round(row[0]))),
                        "memory_mb": max(1, int(row[1])),
                        "gpu_memory_mb": max(0, int(row[2])),
                        "execution_time_ms": max(1, int(row[3])),
                        "confidence": 0.9,
                        "match_type": "exact"
                    }
                
                # Next, try match on model type and batch size
                cursor = self.conn.execute(
                    "SELECT AVG(cpu_cores), AVG(memory_mb), AVG(gpu_memory_mb), AVG(execution_time_ms), COUNT(*) "
                    "FROM execution_data WHERE model_type = ? AND batch_size = ? AND success = 1",
                    (model_type, batch_size)
                )
                row = cursor.fetchone()
                
                type_batch_match = (row[4] >= 3)
                
                if type_batch_match:
                    return {
                        "cpu_cores": max(1, int(round(row[0]))),
                        "memory_mb": max(1, int(row[1])),
                        "gpu_memory_mb": max(0, int(row[2])),
                        "execution_time_ms": max(1, int(row[3])),
                        "confidence": 0.8,
                        "match_type": "type_batch"
                    }
                
                # Next, try match on model name (any batch size) and scale
                cursor = self.conn.execute(
                    "SELECT AVG(cpu_cores), AVG(memory_mb), AVG(gpu_memory_mb), AVG(execution_time_ms), AVG(batch_size), COUNT(*) "
                    "FROM execution_data WHERE model_name = ? AND success = 1",
                    (model_name,)
                )
                row = cursor.fetchone()
                
                model_match = (row[5] >= 3)
                
                if model_match:
                    # Get average batch size in the results
                    avg_batch_size = row[4]
                    
                    # Calculate scaling factors based on batch size difference
                    scaling_data = {
                        "model_type": model_type,
                        "batch_size": avg_batch_size,
                        "batch_size_target": batch_size,
                        "model_name": model_name
                    }
                    scaling = self.get_resource_scaling_factor(scaling_data)
                    
                    return {
                        "cpu_cores": max(1, int(round(row[0] * scaling["cpu_cores"]))),
                        "memory_mb": max(1, int(row[1] * scaling["memory_mb"])),
                        "gpu_memory_mb": max(0, int(row[2] * scaling["gpu_memory_mb"])),
                        "execution_time_ms": max(1, int(row[3] * scaling["cpu_cores"])),  # Execution time scales with CPU
                        "confidence": 0.7,
                        "match_type": "model_scaled"
                    }
                
                # Finally, try match on model type (any batch size) and scale
                cursor = self.conn.execute(
                    "SELECT AVG(cpu_cores), AVG(memory_mb), AVG(gpu_memory_mb), AVG(execution_time_ms), AVG(batch_size), COUNT(*) "
                    "FROM execution_data WHERE model_type = ? AND success = 1",
                    (model_type,)
                )
                row = cursor.fetchone()
                
                type_match = (row[5] >= 3)
                
                if type_match:
                    # Get average batch size in the results
                    avg_batch_size = row[4]
                    
                    # Calculate scaling factors based on batch size difference
                    scaling_data = {
                        "model_type": model_type,
                        "batch_size": avg_batch_size,
                        "batch_size_target": batch_size
                    }
                    scaling = self.get_resource_scaling_factor(scaling_data)
                    
                    return {
                        "cpu_cores": max(1, int(round(row[0] * scaling["cpu_cores"]))),
                        "memory_mb": max(1, int(row[1] * scaling["memory_mb"])),
                        "gpu_memory_mb": max(0, int(row[2] * scaling["gpu_memory_mb"])),
                        "execution_time_ms": max(1, int(row[3] * scaling["cpu_cores"])),  # Execution time scales with CPU
                        "confidence": 0.6,
                        "match_type": "type_scaled"
                    }
                
                # No good matches, use defaults
                return self._predict_with_defaults(task_data)
        
        except Exception as e:
            logger.error(f"Error in statistical prediction: {e}")
            logger.debug(traceback.format_exc())
            return self._predict_with_defaults(task_data)
    
    def _predict_with_defaults(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict using default values."""
        model_type = task_data.get("model_type", "default")
        batch_size = task_data.get("batch_size", 1)
        
        # Get base values
        cpu_cores = DEFAULT_CPU_CORES.get(model_type, DEFAULT_CPU_CORES["default"])
        memory_mb = DEFAULT_MEMORY_MB.get(model_type, DEFAULT_MEMORY_MB["default"])
        gpu_memory_mb = DEFAULT_GPU_MEMORY_MB.get(model_type, DEFAULT_GPU_MEMORY_MB["default"])
        
        # Apply batch size scaling (simple power law)
        batch_factor = max(1, batch_size) / 32  # Normalized to batch size 32
        
        # Different resources scale differently with batch size
        cpu_scale = batch_factor ** DEFAULT_BATCH_SCALING["cpu_cores"]
        memory_scale = batch_factor ** DEFAULT_BATCH_SCALING["memory_mb"]
        gpu_scale = batch_factor ** DEFAULT_BATCH_SCALING["gpu_memory_mb"]
        
        return {
            "cpu_cores": max(1, int(round(cpu_cores * cpu_scale))),
            "memory_mb": max(1, int(memory_mb * memory_scale)),
            "gpu_memory_mb": max(0, int(gpu_memory_mb * gpu_scale)),
            "confidence": 0.5,
            "match_type": "default"
        }
    
    def _get_scaling_factors_from_data(self, model_type: str, model_name: Optional[str], 
                                      batch_size_source: int, batch_size_target: int) -> Optional[Dict[str, float]]:
        """Calculate scaling factors from actual data."""
        try:
            # Determine batch size ranges for source and target
            source_min = max(1, int(batch_size_source * 0.8))
            source_max = int(batch_size_source * 1.2)
            target_min = max(1, int(batch_size_target * 0.8))
            target_max = int(batch_size_target * 1.2)
            
            # Build the query
            query = """
                SELECT 
                    AVG(CASE WHEN b.batch_size BETWEEN ? AND ? THEN b.cpu_cores / a.cpu_cores ELSE NULL END) as cpu_scale,
                    AVG(CASE WHEN b.batch_size BETWEEN ? AND ? THEN b.memory_mb / a.memory_mb ELSE NULL END) as memory_scale,
                    AVG(CASE WHEN b.batch_size BETWEEN ? AND ? THEN b.gpu_memory_mb / a.gpu_memory_mb ELSE NULL END) as gpu_scale,
                    COUNT(*) as pair_count
                FROM 
                    execution_data a
                JOIN 
                    execution_data b ON a.model_type = b.model_type
                WHERE 
                    a.model_type = ? AND
                    a.batch_size BETWEEN ? AND ? AND
                    b.batch_size BETWEEN ? AND ? AND
                    a.success = 1 AND 
                    b.success = 1
            """
            
            params = [
                target_min, target_max,
                target_min, target_max,
                target_min, target_max,
                model_type,
                source_min, source_max,
                target_min, target_max
            ]
            
            # Add model name filter if provided
            if model_name:
                query = query.replace("a.model_type = b.model_type", 
                                     "a.model_type = b.model_type AND a.model_name = b.model_name")
                query = query.replace("a.model_type = ?", 
                                     "a.model_type = ? AND a.model_name = ?")
                params.insert(7, model_name)  # Insert after model_type in the params list
            
            # Execute query
            with self.conn:
                cursor = self.conn.execute(query, params)
                row = cursor.fetchone()
                
                # Check if we have enough data pairs
                if row and row[3] >= 3 and row[0] is not None:
                    # We have scaling factors from real data
                    return {
                        "cpu_cores": max(1.0, row[0]),
                        "memory_mb": max(1.0, row[1] if row[1] is not None else batch_size_target / batch_size_source),
                        "gpu_memory_mb": max(1.0, row[2] if row[2] is not None else batch_size_target / batch_size_source),
                        "confidence": min(0.9, 0.6 + (row[3] / 20))  # Higher confidence with more data pairs
                    }
            
            return None
        
        except Exception as e:
            logger.error(f"Error calculating scaling factors from data: {e}")
            logger.debug(traceback.format_exc())
            return None
    
    def _training_loop(self) -> None:
        """Background thread for periodic model training."""
        while self.running:
            try:
                # Check if training is needed
                current_time = time.time()
                current_count = self._get_data_count()
                
                if (current_time - self.last_training_time >= self.training_interval or
                    current_count >= self.data_count + 10):  # At least 10 new data points
                    
                    # Train models if enabled
                    if self.training_enabled and ML_AVAILABLE and PANDAS_AVAILABLE:
                        logger.info("Starting model training in background thread")
                        self.train_models()
                        
                        # Update data count
                        self.data_count = current_count
            
            except Exception as e:
                logger.error(f"Error in training loop: {e}")
                logger.debug(traceback.format_exc())
            
            # Sleep for a while
            time.sleep(60)  # Check every minute


# Main function for testing
if __name__ == "__main__":
    """Run standalone test of the Resource Performance Predictor."""
    predictor = ResourcePerformancePredictor()
    
    # Record some test data
    for i in range(10):
        predictor.record_task_execution(
            task_id=f"task-{i}",
            execution_data={
                "model_type": "text_embedding",
                "model_name": "bert-base-uncased",
                "batch_size": 32,
                "cpu_cores_used": 2,
                "memory_mb_used": 4096,
                "gpu_memory_mb_used": 2048,
                "execution_time_ms": 150,
                "success": True
            }
        )
    
    # Predict resource requirements
    prediction = predictor.predict_resource_requirements({
        "model_type": "text_embedding",
        "model_name": "bert-base-uncased",
        "batch_size": 64
    })
    
    print(f"Prediction: {json.dumps(prediction, indent=2)}")
    
    # Get scaling factor
    scaling = predictor.get_resource_scaling_factor({
        "model_type": "text_embedding",
        "batch_size": 32,
        "batch_size_target": 64
    })
    
    print(f"Scaling factors: {json.dumps(scaling, indent=2)}")
    
    # Train models (if ML available)
    if ML_AVAILABLE and PANDAS_AVAILABLE:
        predictor.train_models()
    
    # Cleanup
    predictor.cleanup()