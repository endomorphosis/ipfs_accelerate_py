"""
Calibration Repository for DuckDB integration in the IPFS Accelerate Framework.

This module provides a repository for storing and retrieving calibration data,
including parameters, results, and historical calibration records.
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

class DuckDBCalibrationRepository:
    """
    DuckDB-based repository for simulation calibration data.
    
    This class provides methods for storing and retrieving calibration parameters,
    results, cross-validation data, uncertainty metrics, and historical calibration records.
    """
    
    def __init__(
        self, 
        db_path: str = "calibration.duckdb",
        create_if_missing: bool = True,
        config: Dict[str, Any] = None
    ):
        """
        Initialize the DuckDB Calibration Repository.
        
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
            # Create calibration parameters table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS calibration_parameters (
                    id INTEGER PRIMARY KEY,
                    timestamp TIMESTAMP,
                    name VARCHAR,
                    value DOUBLE,
                    uncertainty DOUBLE,
                    description VARCHAR,
                    source VARCHAR,
                    calibration_id VARCHAR,
                    metadata VARCHAR
                )
            """)
            
            # Create calibration results table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS calibration_results (
                    id INTEGER PRIMARY KEY,
                    timestamp TIMESTAMP,
                    calibration_id VARCHAR,
                    error_before DOUBLE,
                    error_after DOUBLE,
                    error_reduction_percent DOUBLE,
                    calibrator_type VARCHAR,
                    iterations INTEGER,
                    converged BOOLEAN,
                    runtime_seconds DOUBLE,
                    dataset_id VARCHAR,
                    hardware_id VARCHAR,
                    simulation_id VARCHAR,
                    metadata VARCHAR
                )
            """)
            
            # Create cross-validation table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS cross_validation_results (
                    id INTEGER PRIMARY KEY,
                    timestamp TIMESTAMP,
                    validation_id VARCHAR,
                    calibration_id VARCHAR,
                    fold INTEGER,
                    train_error DOUBLE,
                    validation_error DOUBLE,
                    generalization_gap DOUBLE,
                    dataset_id VARCHAR,
                    calibrator_type VARCHAR,
                    parameters VARCHAR,
                    metadata VARCHAR
                )
            """)
            
            # Create parameter sensitivity table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS parameter_sensitivity (
                    id INTEGER PRIMARY KEY,
                    timestamp TIMESTAMP,
                    parameter_name VARCHAR,
                    sensitivity DOUBLE,
                    relative_sensitivity DOUBLE,
                    non_linearity DOUBLE,
                    analysis_id VARCHAR,
                    calibration_id VARCHAR,
                    importance_rank INTEGER,
                    threshold_value DOUBLE,
                    metadata VARCHAR
                )
            """)
            
            # Create uncertainty quantification table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS uncertainty_quantification (
                    id INTEGER PRIMARY KEY,
                    timestamp TIMESTAMP,
                    parameter_name VARCHAR,
                    mean_value DOUBLE,
                    std_value DOUBLE,
                    cv_value DOUBLE,
                    ci_lower DOUBLE,
                    ci_upper DOUBLE,
                    uncertainty_level VARCHAR,
                    analysis_id VARCHAR,
                    calibration_id VARCHAR,
                    confidence_level DOUBLE,
                    sample_size INTEGER,
                    metadata VARCHAR
                )
            """)
            
            # Create calibration history table (ties everything together)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS calibration_history (
                    id INTEGER PRIMARY KEY,
                    timestamp TIMESTAMP,
                    calibration_id VARCHAR,
                    user_id VARCHAR,
                    calibrator_type VARCHAR,
                    dataset_size INTEGER,
                    hardware_platforms VARCHAR,
                    simulation_config VARCHAR,
                    best_parameters VARCHAR,
                    final_error DOUBLE,
                    improvement_percent DOUBLE,
                    description VARCHAR,
                    tags VARCHAR,
                    status VARCHAR,
                    metadata VARCHAR
                )
            """)
            
            # Create calibration drift tracking table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS calibration_drift (
                    id INTEGER PRIMARY KEY,
                    timestamp TIMESTAMP,
                    calibration_id VARCHAR,
                    drift_value DOUBLE,
                    drift_type VARCHAR,
                    threshold_value DOUBLE,
                    requires_recalibration BOOLEAN,
                    affected_parameters VARCHAR,
                    description VARCHAR,
                    metadata VARCHAR
                )
            """)
            
            # Create indices for common queries
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_params_calibration_id ON calibration_parameters(calibration_id)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_results_calibration_id ON calibration_results(calibration_id)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_history_timestamp ON calibration_history(timestamp)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_cv_validation_id ON cross_validation_results(validation_id)")
            
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
    
    def store_parameter(self, parameter: Dict[str, Any]) -> int:
        """
        Store a single calibration parameter.
        
        Args:
            parameter: Dictionary containing the parameter data
            
        Returns:
            The ID of the stored parameter
        """
        try:
            # Ensure timestamp is a datetime object
            if isinstance(parameter.get('timestamp'), str):
                parameter['timestamp'] = datetime.fromisoformat(parameter['timestamp'].replace('Z', '+00:00'))
            elif not parameter.get('timestamp'):
                parameter['timestamp'] = datetime.now()
            
            # Convert metadata to JSON string if it's a dictionary
            if isinstance(parameter.get('metadata'), dict):
                parameter['metadata'] = json.dumps(parameter['metadata'])
            
            # Insert the parameter
            query = """
                INSERT INTO calibration_parameters (
                    timestamp, name, value, uncertainty, description,
                    source, calibration_id, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                RETURNING id
            """
            
            result = self.conn.execute(
                query,
                (
                    parameter.get('timestamp'),
                    parameter.get('name'),
                    parameter.get('value'),
                    parameter.get('uncertainty'),
                    parameter.get('description'),
                    parameter.get('source'),
                    parameter.get('calibration_id'),
                    parameter.get('metadata')
                )
            ).fetchone()
            
            return result[0] if result else -1
        except Exception as e:
            logger.error(f"Error storing parameter: {str(e)}")
            raise
    
    def store_parameters_batch(self, parameters: List[Dict[str, Any]]) -> List[int]:
        """
        Store multiple calibration parameters in batch.
        
        Args:
            parameters: List of dictionaries containing parameter data
            
        Returns:
            List of IDs for the stored parameters
        """
        if not parameters:
            return []
        
        try:
            ids = []
            for parameter in parameters:
                param_id = self.store_parameter(parameter)
                ids.append(param_id)
            return ids
        except Exception as e:
            logger.error(f"Error storing parameters batch: {str(e)}")
            raise
    
    def get_parameters(self, 
                     calibration_id: Optional[str] = None,
                     parameter_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve calibration parameters based on filters.
        
        Args:
            calibration_id: Optional calibration ID filter
            parameter_names: Optional list of parameter names to retrieve
            
        Returns:
            List of parameters matching the filters
        """
        try:
            conditions = []
            params = []
            
            if calibration_id:
                conditions.append("calibration_id = ?")
                params.append(calibration_id)
            
            if parameter_names:
                placeholders = ', '.join(['?' for _ in parameter_names])
                conditions.append(f"name IN ({placeholders})")
                params.extend(parameter_names)
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            query = f"""
                SELECT id, timestamp, name, value, uncertainty,
                       description, source, calibration_id, metadata
                FROM calibration_parameters
                WHERE {where_clause}
                ORDER BY name
            """
            
            result = self.conn.execute(query, params).fetchall()
            
            # Convert to list of dictionaries
            parameters = []
            for row in result:
                parameter = {
                    'id': row[0],
                    'timestamp': row[1],
                    'name': row[2],
                    'value': row[3],
                    'uncertainty': row[4],
                    'description': row[5],
                    'source': row[6],
                    'calibration_id': row[7],
                    'metadata': row[8]
                }
                
                # Parse metadata if it's a JSON string
                if isinstance(parameter['metadata'], str) and parameter['metadata']:
                    try:
                        parameter['metadata'] = json.loads(parameter['metadata'])
                    except:
                        pass
                
                parameters.append(parameter)
            
            return parameters
        except Exception as e:
            logger.error(f"Error retrieving parameters: {str(e)}")
            raise
    
    def store_calibration_result(self, result: Dict[str, Any]) -> int:
        """
        Store a calibration result.
        
        Args:
            result: Dictionary containing the calibration result data
            
        Returns:
            The ID of the stored result
        """
        try:
            # Ensure timestamp is a datetime object
            if isinstance(result.get('timestamp'), str):
                result['timestamp'] = datetime.fromisoformat(result['timestamp'].replace('Z', '+00:00'))
            elif not result.get('timestamp'):
                result['timestamp'] = datetime.now()
            
            # Convert metadata to JSON string if it's a dictionary
            if isinstance(result.get('metadata'), dict):
                result['metadata'] = json.dumps(result['metadata'])
            
            # Calculate error reduction percentage if not provided
            if 'error_reduction_percent' not in result and 'error_before' in result and 'error_after' in result:
                error_before = result.get('error_before', 0)
                error_after = result.get('error_after', 0)
                if error_before > 0:
                    result['error_reduction_percent'] = ((error_before - error_after) / error_before) * 100
            
            # Insert the result
            query = """
                INSERT INTO calibration_results (
                    timestamp, calibration_id, error_before, error_after,
                    error_reduction_percent, calibrator_type, iterations,
                    converged, runtime_seconds, dataset_id, hardware_id,
                    simulation_id, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                RETURNING id
            """
            
            result_data = self.conn.execute(
                query,
                (
                    result.get('timestamp'),
                    result.get('calibration_id'),
                    result.get('error_before'),
                    result.get('error_after'),
                    result.get('error_reduction_percent'),
                    result.get('calibrator_type'),
                    result.get('iterations'),
                    result.get('converged'),
                    result.get('runtime_seconds'),
                    result.get('dataset_id'),
                    result.get('hardware_id'),
                    result.get('simulation_id'),
                    result.get('metadata')
                )
            ).fetchone()
            
            return result_data[0] if result_data else -1
        except Exception as e:
            logger.error(f"Error storing calibration result: {str(e)}")
            raise
    
    def get_calibration_results(self, 
                              calibration_id: Optional[str] = None,
                              start_time: Optional[datetime] = None,
                              end_time: Optional[datetime] = None,
                              hardware_id: Optional[str] = None,
                              limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve calibration results based on filters.
        
        Args:
            calibration_id: Optional calibration ID filter
            start_time: Optional start time filter
            end_time: Optional end time filter
            hardware_id: Optional hardware ID filter
            limit: Maximum number of records to return
            
        Returns:
            List of calibration results matching the filters
        """
        try:
            conditions = []
            params = []
            
            if calibration_id:
                conditions.append("calibration_id = ?")
                params.append(calibration_id)
            
            if start_time:
                conditions.append("timestamp >= ?")
                params.append(start_time)
            
            if end_time:
                conditions.append("timestamp <= ?")
                params.append(end_time)
            
            if hardware_id:
                conditions.append("hardware_id = ?")
                params.append(hardware_id)
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            query = f"""
                SELECT id, timestamp, calibration_id, error_before, error_after,
                       error_reduction_percent, calibrator_type, iterations,
                       converged, runtime_seconds, dataset_id, hardware_id,
                       simulation_id, metadata
                FROM calibration_results
                WHERE {where_clause}
                ORDER BY timestamp DESC
                LIMIT ?
            """
            
            params.append(limit)
            
            result = self.conn.execute(query, params).fetchall()
            
            # Convert to list of dictionaries
            calibration_results = []
            for row in result:
                cal_result = {
                    'id': row[0],
                    'timestamp': row[1],
                    'calibration_id': row[2],
                    'error_before': row[3],
                    'error_after': row[4],
                    'error_reduction_percent': row[5],
                    'calibrator_type': row[6],
                    'iterations': row[7],
                    'converged': row[8],
                    'runtime_seconds': row[9],
                    'dataset_id': row[10],
                    'hardware_id': row[11],
                    'simulation_id': row[12],
                    'metadata': row[13]
                }
                
                # Parse metadata if it's a JSON string
                if isinstance(cal_result['metadata'], str) and cal_result['metadata']:
                    try:
                        cal_result['metadata'] = json.loads(cal_result['metadata'])
                    except:
                        pass
                
                calibration_results.append(cal_result)
            
            return calibration_results
        except Exception as e:
            logger.error(f"Error retrieving calibration results: {str(e)}")
            raise
    
    def store_cross_validation_result(self, validation_result: Dict[str, Any]) -> int:
        """
        Store a cross-validation result.
        
        Args:
            validation_result: Dictionary containing the cross-validation result data
            
        Returns:
            The ID of the stored validation result
        """
        try:
            # Ensure timestamp is a datetime object
            if isinstance(validation_result.get('timestamp'), str):
                validation_result['timestamp'] = datetime.fromisoformat(validation_result['timestamp'].replace('Z', '+00:00'))
            elif not validation_result.get('timestamp'):
                validation_result['timestamp'] = datetime.now()
            
            # Convert parameters and metadata to JSON string if they're dictionaries
            if isinstance(validation_result.get('parameters'), dict):
                validation_result['parameters'] = json.dumps(validation_result['parameters'])
            
            if isinstance(validation_result.get('metadata'), dict):
                validation_result['metadata'] = json.dumps(validation_result['metadata'])
            
            # Insert the validation result
            query = """
                INSERT INTO cross_validation_results (
                    timestamp, validation_id, calibration_id, fold,
                    train_error, validation_error, generalization_gap,
                    dataset_id, calibrator_type, parameters, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                RETURNING id
            """
            
            result = self.conn.execute(
                query,
                (
                    validation_result.get('timestamp'),
                    validation_result.get('validation_id'),
                    validation_result.get('calibration_id'),
                    validation_result.get('fold'),
                    validation_result.get('train_error'),
                    validation_result.get('validation_error'),
                    validation_result.get('generalization_gap'),
                    validation_result.get('dataset_id'),
                    validation_result.get('calibrator_type'),
                    validation_result.get('parameters'),
                    validation_result.get('metadata')
                )
            ).fetchone()
            
            return result[0] if result else -1
        except Exception as e:
            logger.error(f"Error storing cross-validation result: {str(e)}")
            raise
    
    def get_cross_validation_results(self, 
                                   validation_id: Optional[str] = None,
                                   calibration_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve cross-validation results based on filters.
        
        Args:
            validation_id: Optional validation ID filter
            calibration_id: Optional calibration ID filter
            
        Returns:
            List of cross-validation results matching the filters
        """
        try:
            conditions = []
            params = []
            
            if validation_id:
                conditions.append("validation_id = ?")
                params.append(validation_id)
            
            if calibration_id:
                conditions.append("calibration_id = ?")
                params.append(calibration_id)
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            query = f"""
                SELECT id, timestamp, validation_id, calibration_id, fold,
                       train_error, validation_error, generalization_gap,
                       dataset_id, calibrator_type, parameters, metadata
                FROM cross_validation_results
                WHERE {where_clause}
                ORDER BY fold
            """
            
            result = self.conn.execute(query, params).fetchall()
            
            # Convert to list of dictionaries
            validation_results = []
            for row in result:
                val_result = {
                    'id': row[0],
                    'timestamp': row[1],
                    'validation_id': row[2],
                    'calibration_id': row[3],
                    'fold': row[4],
                    'train_error': row[5],
                    'validation_error': row[6],
                    'generalization_gap': row[7],
                    'dataset_id': row[8],
                    'calibrator_type': row[9],
                    'parameters': row[10],
                    'metadata': row[11]
                }
                
                # Parse parameters and metadata if they're JSON strings
                for field in ['parameters', 'metadata']:
                    if isinstance(val_result[field], str) and val_result[field]:
                        try:
                            val_result[field] = json.loads(val_result[field])
                        except:
                            pass
                
                validation_results.append(val_result)
            
            return validation_results
        except Exception as e:
            logger.error(f"Error retrieving cross-validation results: {str(e)}")
            raise
    
    def store_parameter_sensitivity(self, sensitivity: Dict[str, Any]) -> int:
        """
        Store a parameter sensitivity result.
        
        Args:
            sensitivity: Dictionary containing the parameter sensitivity data
            
        Returns:
            The ID of the stored sensitivity result
        """
        try:
            # Ensure timestamp is a datetime object
            if isinstance(sensitivity.get('timestamp'), str):
                sensitivity['timestamp'] = datetime.fromisoformat(sensitivity['timestamp'].replace('Z', '+00:00'))
            elif not sensitivity.get('timestamp'):
                sensitivity['timestamp'] = datetime.now()
            
            # Convert metadata to JSON string if it's a dictionary
            if isinstance(sensitivity.get('metadata'), dict):
                sensitivity['metadata'] = json.dumps(sensitivity['metadata'])
            
            # Insert the sensitivity result
            query = """
                INSERT INTO parameter_sensitivity (
                    timestamp, parameter_name, sensitivity, relative_sensitivity,
                    non_linearity, analysis_id, calibration_id, importance_rank,
                    threshold_value, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                RETURNING id
            """
            
            result = self.conn.execute(
                query,
                (
                    sensitivity.get('timestamp'),
                    sensitivity.get('parameter_name'),
                    sensitivity.get('sensitivity'),
                    sensitivity.get('relative_sensitivity'),
                    sensitivity.get('non_linearity'),
                    sensitivity.get('analysis_id'),
                    sensitivity.get('calibration_id'),
                    sensitivity.get('importance_rank'),
                    sensitivity.get('threshold_value'),
                    sensitivity.get('metadata')
                )
            ).fetchone()
            
            return result[0] if result else -1
        except Exception as e:
            logger.error(f"Error storing parameter sensitivity: {str(e)}")
            raise
    
    def get_parameter_sensitivities(self, 
                                  analysis_id: Optional[str] = None,
                                  calibration_id: Optional[str] = None,
                                  parameter_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve parameter sensitivities based on filters.
        
        Args:
            analysis_id: Optional analysis ID filter
            calibration_id: Optional calibration ID filter
            parameter_names: Optional list of parameter names to filter
            
        Returns:
            List of parameter sensitivities matching the filters
        """
        try:
            conditions = []
            params = []
            
            if analysis_id:
                conditions.append("analysis_id = ?")
                params.append(analysis_id)
            
            if calibration_id:
                conditions.append("calibration_id = ?")
                params.append(calibration_id)
            
            if parameter_names:
                placeholders = ', '.join(['?' for _ in parameter_names])
                conditions.append(f"parameter_name IN ({placeholders})")
                params.extend(parameter_names)
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            query = f"""
                SELECT id, timestamp, parameter_name, sensitivity, relative_sensitivity,
                       non_linearity, analysis_id, calibration_id, importance_rank,
                       threshold_value, metadata
                FROM parameter_sensitivity
                WHERE {where_clause}
                ORDER BY importance_rank IS NULL, importance_rank, relative_sensitivity DESC
            """
            
            result = self.conn.execute(query, params).fetchall()
            
            # Convert to list of dictionaries
            sensitivities = []
            for row in result:
                sensitivity = {
                    'id': row[0],
                    'timestamp': row[1],
                    'parameter_name': row[2],
                    'sensitivity': row[3],
                    'relative_sensitivity': row[4],
                    'non_linearity': row[5],
                    'analysis_id': row[6],
                    'calibration_id': row[7],
                    'importance_rank': row[8],
                    'threshold_value': row[9],
                    'metadata': row[10]
                }
                
                # Parse metadata if it's a JSON string
                if isinstance(sensitivity['metadata'], str) and sensitivity['metadata']:
                    try:
                        sensitivity['metadata'] = json.loads(sensitivity['metadata'])
                    except:
                        pass
                
                sensitivities.append(sensitivity)
            
            return sensitivities
        except Exception as e:
            logger.error(f"Error retrieving parameter sensitivities: {str(e)}")
            raise
    
    def store_uncertainty_quantification(self, uncertainty: Dict[str, Any]) -> int:
        """
        Store an uncertainty quantification result.
        
        Args:
            uncertainty: Dictionary containing the uncertainty quantification data
            
        Returns:
            The ID of the stored uncertainty result
        """
        try:
            # Ensure timestamp is a datetime object
            if isinstance(uncertainty.get('timestamp'), str):
                uncertainty['timestamp'] = datetime.fromisoformat(uncertainty['timestamp'].replace('Z', '+00:00'))
            elif not uncertainty.get('timestamp'):
                uncertainty['timestamp'] = datetime.now()
            
            # Convert metadata to JSON string if it's a dictionary
            if isinstance(uncertainty.get('metadata'), dict):
                uncertainty['metadata'] = json.dumps(uncertainty['metadata'])
            
            # Insert the uncertainty result
            query = """
                INSERT INTO uncertainty_quantification (
                    timestamp, parameter_name, mean_value, std_value,
                    cv_value, ci_lower, ci_upper, uncertainty_level,
                    analysis_id, calibration_id, confidence_level,
                    sample_size, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                RETURNING id
            """
            
            result = self.conn.execute(
                query,
                (
                    uncertainty.get('timestamp'),
                    uncertainty.get('parameter_name'),
                    uncertainty.get('mean_value'),
                    uncertainty.get('std_value'),
                    uncertainty.get('cv_value'),
                    uncertainty.get('ci_lower'),
                    uncertainty.get('ci_upper'),
                    uncertainty.get('uncertainty_level'),
                    uncertainty.get('analysis_id'),
                    uncertainty.get('calibration_id'),
                    uncertainty.get('confidence_level'),
                    uncertainty.get('sample_size'),
                    uncertainty.get('metadata')
                )
            ).fetchone()
            
            return result[0] if result else -1
        except Exception as e:
            logger.error(f"Error storing uncertainty quantification: {str(e)}")
            raise
    
    def get_uncertainty_quantifications(self, 
                                      analysis_id: Optional[str] = None,
                                      calibration_id: Optional[str] = None,
                                      parameter_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve uncertainty quantifications based on filters.
        
        Args:
            analysis_id: Optional analysis ID filter
            calibration_id: Optional calibration ID filter
            parameter_names: Optional list of parameter names to filter
            
        Returns:
            List of uncertainty quantifications matching the filters
        """
        try:
            conditions = []
            params = []
            
            if analysis_id:
                conditions.append("analysis_id = ?")
                params.append(analysis_id)
            
            if calibration_id:
                conditions.append("calibration_id = ?")
                params.append(calibration_id)
            
            if parameter_names:
                placeholders = ', '.join(['?' for _ in parameter_names])
                conditions.append(f"parameter_name IN ({placeholders})")
                params.extend(parameter_names)
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            query = f"""
                SELECT id, timestamp, parameter_name, mean_value, std_value,
                       cv_value, ci_lower, ci_upper, uncertainty_level,
                       analysis_id, calibration_id, confidence_level,
                       sample_size, metadata
                FROM uncertainty_quantification
                WHERE {where_clause}
                ORDER BY 
                    CASE uncertainty_level 
                        WHEN 'high' THEN 1 
                        WHEN 'medium' THEN 2 
                        WHEN 'low' THEN 3 
                        ELSE 4 
                    END,
                    cv_value DESC
            """
            
            result = self.conn.execute(query, params).fetchall()
            
            # Convert to list of dictionaries
            uncertainties = []
            for row in result:
                uncertainty = {
                    'id': row[0],
                    'timestamp': row[1],
                    'parameter_name': row[2],
                    'mean_value': row[3],
                    'std_value': row[4],
                    'cv_value': row[5],
                    'ci_lower': row[6],
                    'ci_upper': row[7],
                    'uncertainty_level': row[8],
                    'analysis_id': row[9],
                    'calibration_id': row[10],
                    'confidence_level': row[11],
                    'sample_size': row[12],
                    'metadata': row[13]
                }
                
                # Parse metadata if it's a JSON string
                if isinstance(uncertainty['metadata'], str) and uncertainty['metadata']:
                    try:
                        uncertainty['metadata'] = json.loads(uncertainty['metadata'])
                    except:
                        pass
                
                uncertainties.append(uncertainty)
            
            return uncertainties
        except Exception as e:
            logger.error(f"Error retrieving uncertainty quantifications: {str(e)}")
            raise
    
    def store_calibration_history(self, history_entry: Dict[str, Any]) -> int:
        """
        Store a calibration history entry.
        
        Args:
            history_entry: Dictionary containing the calibration history data
            
        Returns:
            The ID of the stored history entry
        """
        try:
            # Ensure timestamp is a datetime object
            if isinstance(history_entry.get('timestamp'), str):
                history_entry['timestamp'] = datetime.fromisoformat(history_entry['timestamp'].replace('Z', '+00:00'))
            elif not history_entry.get('timestamp'):
                history_entry['timestamp'] = datetime.now()
            
            # Convert JSON fields
            for field in ['best_parameters', 'hardware_platforms', 'simulation_config', 'tags', 'metadata']:
                if isinstance(history_entry.get(field), (dict, list)):
                    history_entry[field] = json.dumps(history_entry[field])
            
            # Insert the history entry
            query = """
                INSERT INTO calibration_history (
                    timestamp, calibration_id, user_id, calibrator_type,
                    dataset_size, hardware_platforms, simulation_config,
                    best_parameters, final_error, improvement_percent,
                    description, tags, status, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                RETURNING id
            """
            
            result = self.conn.execute(
                query,
                (
                    history_entry.get('timestamp'),
                    history_entry.get('calibration_id'),
                    history_entry.get('user_id'),
                    history_entry.get('calibrator_type'),
                    history_entry.get('dataset_size'),
                    history_entry.get('hardware_platforms'),
                    history_entry.get('simulation_config'),
                    history_entry.get('best_parameters'),
                    history_entry.get('final_error'),
                    history_entry.get('improvement_percent'),
                    history_entry.get('description'),
                    history_entry.get('tags'),
                    history_entry.get('status'),
                    history_entry.get('metadata')
                )
            ).fetchone()
            
            return result[0] if result else -1
        except Exception as e:
            logger.error(f"Error storing calibration history: {str(e)}")
            raise
    
    def get_calibration_history(self, 
                              start_time: Optional[datetime] = None,
                              end_time: Optional[datetime] = None,
                              calibration_id: Optional[str] = None,
                              status: Optional[str] = None,
                              tag: Optional[str] = None,
                              limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve calibration history entries based on filters.
        
        Args:
            start_time: Optional start time filter
            end_time: Optional end time filter
            calibration_id: Optional calibration ID filter
            status: Optional status filter
            tag: Optional tag filter
            limit: Maximum number of records to return
            
        Returns:
            List of calibration history entries matching the filters
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
            
            if calibration_id:
                conditions.append("calibration_id = ?")
                params.append(calibration_id)
            
            if status:
                conditions.append("status = ?")
                params.append(status)
            
            if tag:
                conditions.append("tags LIKE ?")
                params.append(f"%{tag}%")
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            query = f"""
                SELECT id, timestamp, calibration_id, user_id, calibrator_type,
                       dataset_size, hardware_platforms, simulation_config,
                       best_parameters, final_error, improvement_percent,
                       description, tags, status, metadata
                FROM calibration_history
                WHERE {where_clause}
                ORDER BY timestamp DESC
                LIMIT ?
            """
            
            params.append(limit)
            
            result = self.conn.execute(query, params).fetchall()
            
            # Convert to list of dictionaries
            history_entries = []
            for row in result:
                entry = {
                    'id': row[0],
                    'timestamp': row[1],
                    'calibration_id': row[2],
                    'user_id': row[3],
                    'calibrator_type': row[4],
                    'dataset_size': row[5],
                    'hardware_platforms': row[6],
                    'simulation_config': row[7],
                    'best_parameters': row[8],
                    'final_error': row[9],
                    'improvement_percent': row[10],
                    'description': row[11],
                    'tags': row[12],
                    'status': row[13],
                    'metadata': row[14]
                }
                
                # Parse JSON fields
                for field in ['best_parameters', 'hardware_platforms', 'simulation_config', 'tags', 'metadata']:
                    if isinstance(entry[field], str) and entry[field]:
                        try:
                            entry[field] = json.loads(entry[field])
                        except:
                            pass
                
                history_entries.append(entry)
            
            return history_entries
        except Exception as e:
            logger.error(f"Error retrieving calibration history: {str(e)}")
            raise
    
    def store_calibration_drift(self, drift: Dict[str, Any]) -> int:
        """
        Store a calibration drift measurement.
        
        Args:
            drift: Dictionary containing the calibration drift data
            
        Returns:
            The ID of the stored drift record
        """
        try:
            # Ensure timestamp is a datetime object
            if isinstance(drift.get('timestamp'), str):
                drift['timestamp'] = datetime.fromisoformat(drift['timestamp'].replace('Z', '+00:00'))
            elif not drift.get('timestamp'):
                drift['timestamp'] = datetime.now()
            
            # Convert affected_parameters and metadata to JSON string if they're dictionaries or lists
            for field in ['affected_parameters', 'metadata']:
                if isinstance(drift.get(field), (dict, list)):
                    drift[field] = json.dumps(drift[field])
            
            # Insert the drift record
            query = """
                INSERT INTO calibration_drift (
                    timestamp, calibration_id, drift_value, drift_type,
                    threshold_value, requires_recalibration, affected_parameters,
                    description, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                RETURNING id
            """
            
            result = self.conn.execute(
                query,
                (
                    drift.get('timestamp'),
                    drift.get('calibration_id'),
                    drift.get('drift_value'),
                    drift.get('drift_type'),
                    drift.get('threshold_value'),
                    drift.get('requires_recalibration'),
                    drift.get('affected_parameters'),
                    drift.get('description'),
                    drift.get('metadata')
                )
            ).fetchone()
            
            return result[0] if result else -1
        except Exception as e:
            logger.error(f"Error storing calibration drift: {str(e)}")
            raise
    
    def get_calibration_drift(self, 
                            calibration_id: Optional[str] = None,
                            start_time: Optional[datetime] = None,
                            end_time: Optional[datetime] = None,
                            requires_recalibration: Optional[bool] = None,
                            limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve calibration drift records based on filters.
        
        Args:
            calibration_id: Optional calibration ID filter
            start_time: Optional start time filter
            end_time: Optional end time filter
            requires_recalibration: Optional filter for records that require recalibration
            limit: Maximum number of records to return
            
        Returns:
            List of calibration drift records matching the filters
        """
        try:
            conditions = []
            params = []
            
            if calibration_id:
                conditions.append("calibration_id = ?")
                params.append(calibration_id)
            
            if start_time:
                conditions.append("timestamp >= ?")
                params.append(start_time)
            
            if end_time:
                conditions.append("timestamp <= ?")
                params.append(end_time)
            
            if requires_recalibration is not None:
                conditions.append("requires_recalibration = ?")
                params.append(requires_recalibration)
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            query = f"""
                SELECT id, timestamp, calibration_id, drift_value, drift_type,
                       threshold_value, requires_recalibration, affected_parameters,
                       description, metadata
                FROM calibration_drift
                WHERE {where_clause}
                ORDER BY timestamp DESC
                LIMIT ?
            """
            
            params.append(limit)
            
            result = self.conn.execute(query, params).fetchall()
            
            # Convert to list of dictionaries
            drift_records = []
            for row in result:
                drift = {
                    'id': row[0],
                    'timestamp': row[1],
                    'calibration_id': row[2],
                    'drift_value': row[3],
                    'drift_type': row[4],
                    'threshold_value': row[5],
                    'requires_recalibration': row[6],
                    'affected_parameters': row[7],
                    'description': row[8],
                    'metadata': row[9]
                }
                
                # Parse affected_parameters and metadata if they're JSON strings
                for field in ['affected_parameters', 'metadata']:
                    if isinstance(drift[field], str) and drift[field]:
                        try:
                            drift[field] = json.loads(drift[field])
                        except:
                            pass
                
                drift_records.append(drift)
            
            return drift_records
        except Exception as e:
            logger.error(f"Error retrieving calibration drift: {str(e)}")
            raise
    
    def generate_sample_data(self, num_calibrations: int = 5) -> None:
        """
        Generate sample calibration data for testing.
        
        Args:
            num_calibrations: Number of sample calibrations to generate
        """
        try:
            # Sample calibrator types
            calibrator_types = [
                'basic',
                'multi_parameter',
                'bayesian',
                'neural_network',
                'ensemble'
            ]
            
            # Sample parameter names
            parameter_names = [
                'global_scale',
                'global_offset',
                'response_time_scale',
                'response_time_offset',
                'memory_usage_scale',
                'memory_usage_offset',
                'throughput_scale',
                'throughput_offset',
                'error_rate_scale',
                'latency_multiplier'
            ]
            
            # Sample hardware platforms
            hardware_platforms = [
                'cuda',
                'cpu',
                'webgpu',
                'webnn',
                'openvino'
            ]
            
            # Generate calibrations
            for i in range(num_calibrations):
                # Generate a unique calibration ID
                calibration_id = f"cal-{int(time.time())}-{i}"
                
                # Select a random calibrator type
                calibrator_type = np.random.choice(calibrator_types)
                
                # Generate parameters for this calibration
                parameters = []
                for param_name in np.random.choice(parameter_names, size=5, replace=False):
                    value = np.random.normal(1.0, 0.2)
                    uncertainty = np.random.uniform(0.01, 0.1)
                    
                    parameter = {
                        'name': param_name,
                        'value': value,
                        'uncertainty': uncertainty,
                        'description': f"Calibrated {param_name}",
                        'source': 'sample_data',
                        'calibration_id': calibration_id,
                        'metadata': {
                            'original_value': 1.0,
                            'adjustment': value - 1.0
                        }
                    }
                    parameters.append(parameter)
                
                # Store parameters
                self.store_parameters_batch(parameters)
                
                # Generate a calibration result
                error_before = np.random.uniform(0.1, 0.5)
                error_after = error_before * np.random.uniform(0.3, 0.8)  # 20-70% improvement
                
                result = {
                    'calibration_id': calibration_id,
                    'error_before': error_before,
                    'error_after': error_after,
                    'calibrator_type': calibrator_type,
                    'iterations': np.random.randint(10, 100),
                    'converged': np.random.rand() > 0.1,  # 90% converge
                    'runtime_seconds': np.random.uniform(5, 60),
                    'dataset_id': f"dataset-{np.random.randint(1, 10)}",
                    'hardware_id': np.random.choice(hardware_platforms),
                    'simulation_id': f"sim-{np.random.randint(1, 20)}",
                    'metadata': {
                        'temperature': np.random.uniform(40, 80),
                        'batch_size': np.random.randint(1, 32)
                    }
                }
                
                self.store_calibration_result(result)
                
                # Generate cross-validation results
                validation_id = f"val-{int(time.time())}-{i}"
                for fold in range(5):
                    train_error = np.random.uniform(0.05, 0.2)
                    validation_error = train_error * np.random.uniform(0.8, 1.5)
                    
                    cv_result = {
                        'validation_id': validation_id,
                        'calibration_id': calibration_id,
                        'fold': fold + 1,
                        'train_error': train_error,
                        'validation_error': validation_error,
                        'generalization_gap': validation_error - train_error,
                        'dataset_id': f"dataset-{np.random.randint(1, 10)}",
                        'calibrator_type': calibrator_type,
                        'parameters': {param['name']: param['value'] for param in parameters},
                        'metadata': {
                            'data_split': f"80/20 fold {fold+1}"
                        }
                    }
                    
                    self.store_cross_validation_result(cv_result)
                
                # Generate parameter sensitivities
                analysis_id = f"sensitivity-{int(time.time())}-{i}"
                for idx, param in enumerate(parameters):
                    sensitivity = np.random.uniform(0.01, 0.5)
                    
                    sensitivity_record = {
                        'parameter_name': param['name'],
                        'sensitivity': sensitivity,
                        'relative_sensitivity': sensitivity / np.random.uniform(0.1, 1.0),
                        'non_linearity': np.random.uniform(-0.2, 0.2),
                        'analysis_id': analysis_id,
                        'calibration_id': calibration_id,
                        'importance_rank': idx + 1,
                        'threshold_value': 0.05,
                        'metadata': {
                            'analysis_method': 'perturbation'
                        }
                    }
                    
                    self.store_parameter_sensitivity(sensitivity_record)
                
                # Generate uncertainty quantifications
                analysis_id = f"uncertainty-{int(time.time())}-{i}"
                for param in parameters:
                    mean_value = param['value']
                    std_value = param['uncertainty']
                    cv_value = std_value / abs(mean_value) if abs(mean_value) > 1e-10 else 0.0
                    
                    # Determine uncertainty level
                    if cv_value < 0.1:
                        uncertainty_level = 'low'
                    elif cv_value < 0.3:
                        uncertainty_level = 'medium'
                    else:
                        uncertainty_level = 'high'
                    
                    uncertainty_record = {
                        'parameter_name': param['name'],
                        'mean_value': mean_value,
                        'std_value': std_value,
                        'cv_value': cv_value,
                        'ci_lower': mean_value - 1.96 * std_value,
                        'ci_upper': mean_value + 1.96 * std_value,
                        'uncertainty_level': uncertainty_level,
                        'analysis_id': analysis_id,
                        'calibration_id': calibration_id,
                        'confidence_level': 0.95,
                        'sample_size': 100,
                        'metadata': {
                            'distribution': 'normal'
                        }
                    }
                    
                    self.store_uncertainty_quantification(uncertainty_record)
                
                # Generate calibration history
                history_entry = {
                    'calibration_id': calibration_id,
                    'user_id': f"user-{np.random.randint(1, 10)}",
                    'calibrator_type': calibrator_type,
                    'dataset_size': np.random.randint(100, 10000),
                    'hardware_platforms': hardware_platforms,
                    'simulation_config': {
                        'version': f"1.{np.random.randint(0, 10)}",
                        'mode': np.random.choice(['accurate', 'fast', 'balanced']),
                        'features': np.random.randint(10, 100)
                    },
                    'best_parameters': {param['name']: param['value'] for param in parameters},
                    'final_error': error_after,
                    'improvement_percent': ((error_before - error_after) / error_before) * 100,
                    'description': f"Sample calibration {i+1}",
                    'tags': ["sample", calibrator_type, "testing"],
                    'status': np.random.choice(['completed', 'failed', 'in_progress'], p=[0.8, 0.1, 0.1]),
                    'metadata': {
                        'generated': True,
                        'date': datetime.now().isoformat()
                    }
                }
                
                self.store_calibration_history(history_entry)
                
                # Generate drift records
                for j in range(np.random.randint(1, 4)):
                    drift_days = np.random.randint(1, 30)
                    drift_value = np.random.uniform(0.01, 0.2)
                    threshold = 0.1
                    
                    drift_record = {
                        'timestamp': datetime.now() - timedelta(days=drift_days),
                        'calibration_id': calibration_id,
                        'drift_value': drift_value,
                        'drift_type': np.random.choice(['parameter', 'data', 'environment']),
                        'threshold_value': threshold,
                        'requires_recalibration': drift_value > threshold,
                        'affected_parameters': [p['name'] for p in np.random.choice(parameters, size=2, replace=False)],
                        'description': f"Detected drift in hardware performance",
                        'metadata': {
                            'detection_method': 'statistical_test',
                            'confidence': np.random.uniform(0.85, 0.99)
                        }
                    }
                    
                    self.store_calibration_drift(drift_record)
            
            logger.info(f"Generated sample data for {num_calibrations} calibrations")
        except Exception as e:
            logger.error(f"Error generating sample data: {str(e)}")
            raise