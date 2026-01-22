#!/usr/bin/env python3
"""
Database Integration Module for the Simulation Accuracy and Validation Framework.

This module provides comprehensive integration between the Simulation Accuracy and Validation
Framework and the DuckDB database backend, enabling efficient storage, retrieval, and
analysis of simulation validation data.
"""

import os
import sys
import json
import logging
import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("simulation_validation_db_integration")

# Import the database API
try:
    from duckdb_api.core.benchmark_db_api import BenchmarkDBAPI
except ImportError:
    logger.error("Failed to import BenchmarkDBAPI. Make sure duckdb_api is properly installed.")
    sys.exit(1)

# Import the core schema module
from duckdb_api.simulation_validation.core.schema import (
    SIMULATION_VALIDATION_SCHEMA,
    SimulationValidationSchema
)

# Import base classes
from duckdb_api.simulation_validation.core.base import (
    SimulationResult,
    HardwareResult,
    ValidationResult
)


class SimulationValidationDBIntegration:
    """
    Database integration class for the Simulation Accuracy and Validation Framework.
    
    This class provides methods for storing, retrieving, and analyzing simulation validation
    data using the DuckDB database backend.
    """
    
    def __init__(self, db_path: str = "./benchmark_db.duckdb", debug: bool = False):
        """
        Initialize the database integration module.
        
        Args:
            db_path: Path to the DuckDB database
            debug: Enable debug logging
        """
        self.db_path = db_path
        
        # Set up logging
        if debug:
            logger.setLevel(logging.DEBUG)
        
        # Initialize database connection
        try:
            self.db_api = BenchmarkDBAPI(db_path=db_path, debug=debug)
            logger.info(f"Connected to database at {db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            self.db_api = None
        
        # Initialize schema
        self._initialize_schema()
    
    def _initialize_schema(self) -> None:
        """Initialize the database schema for simulation validation data."""
        if not self.db_api:
            logger.error("Cannot initialize schema: No database connection")
            return
        
        try:
            # Create the tables using SimulationValidationSchema
            SimulationValidationSchema.create_tables(self.db_api._get_connection())
            logger.info("Database schema initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database schema: {e}")
    
    def store_simulation_result(self, result: SimulationResult) -> str:
        """
        Store a simulation result in the database.
        
        Args:
            result: SimulationResult object
            
        Returns:
            ID of the stored result
        """
        if not self.db_api:
            logger.error("Cannot store simulation result: No database connection")
            return None
        
        try:
            # Convert SimulationResult to database record
            record = SimulationValidationSchema.simulation_result_to_db_dict(result)
            
            # Store in database
            conn = self.db_api._get_connection()
            conn.execute(
                """
                INSERT INTO simulation_results 
                VALUES (:id, :model_id, :hardware_id, :batch_size, :precision, 
                        :timestamp, :simulation_version, :additional_metadata,
                        :throughput_items_per_second, :average_latency_ms, 
                        :memory_peak_mb, :power_consumption_w, 
                        :initialization_time_ms, :warmup_time_ms, 
                        CURRENT_TIMESTAMP)
                """,
                record
            )
            conn.commit()
            
            logger.info(f"Stored simulation result with ID: {record['id']}")
            return record['id']
        except Exception as e:
            logger.error(f"Failed to store simulation result: {e}")
            if self.db_api:
                self.db_api._get_connection().rollback()
            return None
    
    def store_hardware_result(self, result: HardwareResult) -> str:
        """
        Store a hardware result in the database.
        
        Args:
            result: HardwareResult object
            
        Returns:
            ID of the stored result
        """
        if not self.db_api:
            logger.error("Cannot store hardware result: No database connection")
            return None
        
        try:
            # Convert HardwareResult to database record
            record = SimulationValidationSchema.hardware_result_to_db_dict(result)
            
            # Store in database
            conn = self.db_api._get_connection()
            conn.execute(
                """
                INSERT INTO hardware_results 
                VALUES (:id, :model_id, :hardware_id, :batch_size, :precision, 
                        :timestamp, :hardware_details, :test_environment, :additional_metadata,
                        :throughput_items_per_second, :average_latency_ms, 
                        :memory_peak_mb, :power_consumption_w, 
                        :initialization_time_ms, :warmup_time_ms, 
                        CURRENT_TIMESTAMP)
                """,
                record
            )
            conn.commit()
            
            logger.info(f"Stored hardware result with ID: {record['id']}")
            return record['id']
        except Exception as e:
            logger.error(f"Failed to store hardware result: {e}")
            if self.db_api:
                self.db_api._get_connection().rollback()
            return None
    
    def store_validation_result(self, result: ValidationResult) -> str:
        """
        Store a validation result in the database.
        
        Args:
            result: ValidationResult object
            
        Returns:
            ID of the stored result
        """
        if not self.db_api:
            logger.error("Cannot store validation result: No database connection")
            return None
        
        try:
            # First, store the simulation and hardware results
            sim_id = self.store_simulation_result(result.simulation_result)
            hw_id = self.store_hardware_result(result.hardware_result)
            
            if not sim_id or not hw_id:
                logger.error("Failed to store simulation or hardware results")
                return None
            
            # Convert ValidationResult to database record
            record = SimulationValidationSchema.validation_result_to_db_dict(
                result, sim_id, hw_id
            )
            
            # Store in database
            conn = self.db_api._get_connection()
            conn.execute(
                """
                INSERT INTO validation_results 
                VALUES (:id, :simulation_result_id, :hardware_result_id, 
                        :validation_timestamp, :validation_version, 
                        :metrics_comparison, :additional_metrics,
                        :overall_accuracy_score, :throughput_mape, 
                        :latency_mape, :memory_mape, :power_mape, 
                        CURRENT_TIMESTAMP)
                """,
                record
            )
            conn.commit()
            
            logger.info(f"Stored validation result with ID: {record['id']}")
            return record['id']
        except Exception as e:
            logger.error(f"Failed to store validation result: {e}")
            if self.db_api:
                self.db_api._get_connection().rollback()
            return None
    
    def store_calibration_history(
        self,
        hardware_type: str,
        model_type: str,
        previous_parameters: Dict[str, Any],
        updated_parameters: Dict[str, Any],
        validation_results_before: Optional[List[Dict[str, Any]]] = None,
        validation_results_after: Optional[List[Dict[str, Any]]] = None,
        improvement_metrics: Optional[Dict[str, Any]] = None,
        calibration_version: str = "v1"
    ) -> str:
        """
        Store calibration history in the database.
        
        Args:
            hardware_type: Type of hardware
            model_type: Type of model
            previous_parameters: Parameters before calibration
            updated_parameters: Parameters after calibration
            validation_results_before: Validation results before calibration
            validation_results_after: Validation results after calibration
            improvement_metrics: Metrics quantifying the calibration improvement
            calibration_version: Version of the calibration methodology
            
        Returns:
            ID of the stored calibration history record
        """
        if not self.db_api:
            logger.error("Cannot store calibration history: No database connection")
            return None
        
        try:
            # Convert to database record
            record = SimulationValidationSchema.calibration_to_db_dict(
                hardware_type=hardware_type,
                model_type=model_type,
                previous_parameters=previous_parameters,
                updated_parameters=updated_parameters,
                validation_results_before=validation_results_before,
                validation_results_after=validation_results_after,
                improvement_metrics=improvement_metrics,
                calibration_version=calibration_version
            )
            
            # Store in database
            conn = self.db_api._get_connection()
            conn.execute(
                """
                INSERT INTO calibration_history 
                VALUES (:id, :timestamp, :hardware_type, :model_type, 
                        :previous_parameters, :updated_parameters, 
                        :validation_results_before, :validation_results_after, 
                        :improvement_metrics, :calibration_version, 
                        CURRENT_TIMESTAMP)
                """,
                record
            )
            conn.commit()
            
            logger.info(f"Stored calibration history with ID: {record['id']}")
            return record['id']
        except Exception as e:
            logger.error(f"Failed to store calibration history: {e}")
            if self.db_api:
                self.db_api._get_connection().rollback()
            return None
    
    def store_drift_detection(
        self,
        hardware_type: str,
        model_type: str,
        drift_metrics: Dict[str, Any],
        is_significant: bool,
        historical_window_start: str,
        historical_window_end: str,
        new_window_start: str,
        new_window_end: str,
        thresholds_used: Dict[str, float]
    ) -> str:
        """
        Store drift detection results in the database.
        
        Args:
            hardware_type: Type of hardware
            model_type: Type of model
            drift_metrics: Metrics quantifying drift
            is_significant: Whether the drift is statistically significant
            historical_window_start: Start of historical window
            historical_window_end: End of historical window
            new_window_start: Start of new window
            new_window_end: End of new window
            thresholds_used: Thresholds used for drift detection
            
        Returns:
            ID of the stored drift detection record
        """
        if not self.db_api:
            logger.error("Cannot store drift detection: No database connection")
            return None
        
        try:
            # Convert to database record
            record = SimulationValidationSchema.drift_detection_to_db_dict(
                hardware_type=hardware_type,
                model_type=model_type,
                drift_metrics=drift_metrics,
                is_significant=is_significant,
                historical_window_start=historical_window_start,
                historical_window_end=historical_window_end,
                new_window_start=new_window_start,
                new_window_end=new_window_end,
                thresholds_used=thresholds_used
            )
            
            # Store in database
            conn = self.db_api._get_connection()
            conn.execute(
                """
                INSERT INTO drift_detection 
                VALUES (:id, :timestamp, :hardware_type, :model_type, 
                        :drift_metrics, :is_significant, 
                        :historical_window_start, :historical_window_end, 
                        :new_window_start, :new_window_end, 
                        :thresholds_used, CURRENT_TIMESTAMP)
                """,
                record
            )
            conn.commit()
            
            logger.info(f"Stored drift detection with ID: {record['id']}")
            return record['id']
        except Exception as e:
            logger.error(f"Failed to store drift detection: {e}")
            if self.db_api:
                self.db_api._get_connection().rollback()
            return None
    
    def store_simulation_parameters(
        self,
        hardware_type: str,
        model_type: str,
        parameters: Dict[str, Any],
        version: str,
        is_current: bool = True
    ) -> str:
        """
        Store simulation parameters in the database.
        
        Args:
            hardware_type: Type of hardware
            model_type: Type of model
            parameters: Simulation parameters
            version: Version of the parameters
            is_current: Whether these are the current parameters
            
        Returns:
            ID of the stored parameters record
        """
        if not self.db_api:
            logger.error("Cannot store simulation parameters: No database connection")
            return None
        
        try:
            # Generate ID
            params_id = SimulationValidationSchema.generate_id("params")
            
            # If setting as current, mark all others as not current
            conn = self.db_api._get_connection()
            if is_current:
                conn.execute(
                    """
                    UPDATE simulation_parameters
                    SET is_current = FALSE
                    WHERE hardware_type = ? AND model_type = ?
                    """,
                    [hardware_type, model_type]
                )
            
            # Store parameters
            conn.execute(
                """
                INSERT INTO simulation_parameters 
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                [params_id, hardware_type, model_type, json.dumps(parameters), version, is_current]
            )
            conn.commit()
            
            logger.info(f"Stored simulation parameters with ID: {params_id}")
            return params_id
        except Exception as e:
            logger.error(f"Failed to store simulation parameters: {e}")
            if self.db_api:
                self.db_api._get_connection().rollback()
            return None
    
    def get_validation_results(
        self,
        hardware_id: Optional[str] = None,
        model_id: Optional[str] = None,
        batch_size: Optional[int] = None,
        precision: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Retrieve validation results from the database.
        
        Args:
            hardware_id: Filter by hardware ID
            model_id: Filter by model ID
            batch_size: Filter by batch size
            precision: Filter by precision
            start_date: Filter by validation date (start)
            end_date: Filter by validation date (end)
            limit: Maximum number of results to return
            
        Returns:
            List of validation result records
        """
        if not self.db_api:
            logger.error("Cannot get validation results: No database connection")
            return []
        
        try:
            # Build query conditions
            conditions = []
            params = {}
            
            if hardware_id:
                conditions.append("hr.hardware_id = :hardware_id")
                params["hardware_id"] = hardware_id
            
            if model_id:
                conditions.append("hr.model_id = :model_id")
                params["model_id"] = model_id
            
            if batch_size:
                conditions.append("hr.batch_size = :batch_size")
                params["batch_size"] = batch_size
            
            if precision:
                conditions.append("hr.precision = :precision")
                params["precision"] = precision
            
            if start_date:
                conditions.append("vr.validation_timestamp >= :start_date")
                params["start_date"] = start_date
            
            if end_date:
                conditions.append("vr.validation_timestamp <= :end_date")
                params["end_date"] = end_date
            
            # Build query
            query = f"""
                SELECT 
                    vr.id as validation_id,
                    sr.id as simulation_id,
                    hr.id as hardware_id,
                    vr.validation_timestamp,
                    vr.validation_version,
                    vr.metrics_comparison,
                    vr.additional_metrics,
                    vr.overall_accuracy_score,
                    vr.throughput_mape,
                    vr.latency_mape,
                    vr.memory_mape,
                    vr.power_mape,
                    sr.model_id as model_id,
                    sr.hardware_id as hardware_type,
                    sr.batch_size,
                    sr.precision,
                    sr.simulation_version,
                    hr.hardware_details,
                    hr.test_environment
                FROM validation_results vr
                JOIN simulation_results sr ON vr.simulation_result_id = sr.id
                JOIN hardware_results hr ON vr.hardware_result_id = hr.id
                {" WHERE " + " AND ".join(conditions) if conditions else ""}
                ORDER BY vr.validation_timestamp DESC
                LIMIT :limit
            """
            params["limit"] = limit
            
            # Execute query
            conn = self.db_api._get_connection()
            result = conn.execute(query, params)
            rows = result.fetchall()
            
            # Convert to list of dictionaries
            validation_results = []
            for row in rows:
                record = {}
                for idx, column in enumerate(result.description):
                    record[column[0]] = row[idx]
                validation_results.append(record)
            
            logger.info(f"Retrieved {len(validation_results)} validation results")
            return validation_results
        except Exception as e:
            logger.error(f"Failed to get validation results: {e}")
            return []
    
    def get_calibration_history(
        self,
        hardware_type: Optional[str] = None,
        model_type: Optional[str] = None,
        calibration_version: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Retrieve calibration history from the database.
        
        Args:
            hardware_type: Filter by hardware type
            model_type: Filter by model type
            calibration_version: Filter by calibration version
            start_date: Filter by calibration date (start)
            end_date: Filter by calibration date (end)
            limit: Maximum number of results to return
            
        Returns:
            List of calibration history records
        """
        if not self.db_api:
            logger.error("Cannot get calibration history: No database connection")
            return []
        
        try:
            # Build query conditions
            conditions = []
            params = {}
            
            if hardware_type:
                conditions.append("hardware_type = :hardware_type")
                params["hardware_type"] = hardware_type
            
            if model_type:
                conditions.append("model_type = :model_type")
                params["model_type"] = model_type
            
            if calibration_version:
                conditions.append("calibration_version = :calibration_version")
                params["calibration_version"] = calibration_version
            
            if start_date:
                conditions.append("timestamp >= :start_date")
                params["start_date"] = start_date
            
            if end_date:
                conditions.append("timestamp <= :end_date")
                params["end_date"] = end_date
            
            # Build query
            query = f"""
                SELECT * FROM calibration_history
                {" WHERE " + " AND ".join(conditions) if conditions else ""}
                ORDER BY timestamp DESC
                LIMIT :limit
            """
            params["limit"] = limit
            
            # Execute query
            conn = self.db_api._get_connection()
            result = conn.execute(query, params)
            rows = result.fetchall()
            
            # Convert to list of dictionaries
            calibration_history = []
            for row in rows:
                record = {}
                for idx, column in enumerate(result.description):
                    record[column[0]] = row[idx]
                calibration_history.append(record)
            
            logger.info(f"Retrieved {len(calibration_history)} calibration history records")
            return calibration_history
        except Exception as e:
            logger.error(f"Failed to get calibration history: {e}")
            return []
    
    def get_drift_detection_results(
        self,
        hardware_type: Optional[str] = None,
        model_type: Optional[str] = None,
        is_significant: Optional[bool] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Retrieve drift detection results from the database.
        
        Args:
            hardware_type: Filter by hardware type
            model_type: Filter by model type
            is_significant: Filter by significance
            start_date: Filter by detection date (start)
            end_date: Filter by detection date (end)
            limit: Maximum number of results to return
            
        Returns:
            List of drift detection records
        """
        if not self.db_api:
            logger.error("Cannot get drift detection results: No database connection")
            return []
        
        try:
            # Build query conditions
            conditions = []
            params = {}
            
            if hardware_type:
                conditions.append("hardware_type = :hardware_type")
                params["hardware_type"] = hardware_type
            
            if model_type:
                conditions.append("model_type = :model_type")
                params["model_type"] = model_type
            
            if is_significant is not None:
                conditions.append("is_significant = :is_significant")
                params["is_significant"] = is_significant
            
            if start_date:
                conditions.append("timestamp >= :start_date")
                params["start_date"] = start_date
            
            if end_date:
                conditions.append("timestamp <= :end_date")
                params["end_date"] = end_date
            
            # Build query
            query = f"""
                SELECT * FROM drift_detection
                {" WHERE " + " AND ".join(conditions) if conditions else ""}
                ORDER BY timestamp DESC
                LIMIT :limit
            """
            params["limit"] = limit
            
            # Execute query
            conn = self.db_api._get_connection()
            result = conn.execute(query, params)
            rows = result.fetchall()
            
            # Convert to list of dictionaries
            drift_results = []
            for row in rows:
                record = {}
                for idx, column in enumerate(result.description):
                    record[column[0]] = row[idx]
                drift_results.append(record)
            
            logger.info(f"Retrieved {len(drift_results)} drift detection records")
            return drift_results
        except Exception as e:
            logger.error(f"Failed to get drift detection results: {e}")
            return []
    
    def get_current_simulation_parameters(
        self,
        hardware_type: str,
        model_type: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve current simulation parameters from the database.
        
        Args:
            hardware_type: Hardware type
            model_type: Model type
            
        Returns:
            Current simulation parameters or None if not found
        """
        if not self.db_api:
            logger.error("Cannot get simulation parameters: No database connection")
            return None
        
        try:
            # Query for current parameters
            conn = self.db_api._get_connection()
            result = conn.execute(
                """
                SELECT * FROM simulation_parameters
                WHERE hardware_type = ? AND model_type = ? AND is_current = TRUE
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                [hardware_type, model_type]
            )
            row = result.fetchone()
            
            if not row:
                logger.warning(f"No current simulation parameters found for {hardware_type}/{model_type}")
                return None
            
            # Extract parameters JSON
            params_json = row[3]  # Assuming parameters is the 4th column (index 3)
            parameters = json.loads(params_json)
            
            logger.info(f"Retrieved current simulation parameters for {hardware_type}/{model_type}")
            return parameters
        except Exception as e:
            logger.error(f"Failed to get simulation parameters: {e}")
            return None
    
    def get_model_hardware_accuracy(self) -> List[Dict[str, Any]]:
        """
        Get accuracy metrics for model-hardware combinations.
        
        Returns:
            List of accuracy metrics by model and hardware
        """
        if not self.db_api:
            logger.error("Cannot get model-hardware accuracy: No database connection")
            return []
        
        try:
            # Query for aggregated accuracy metrics
            query = """
                SELECT 
                    sr.model_id,
                    sr.hardware_id as hardware_type,
                    COUNT(*) as num_validations,
                    AVG(vr.overall_accuracy_score) as avg_accuracy,
                    AVG(vr.throughput_mape) as avg_throughput_mape,
                    AVG(vr.latency_mape) as avg_latency_mape,
                    AVG(vr.memory_mape) as avg_memory_mape,
                    AVG(vr.power_mape) as avg_power_mape,
                    MAX(vr.validation_timestamp) as last_validation
                FROM validation_results vr
                JOIN simulation_results sr ON vr.simulation_result_id = sr.id
                GROUP BY sr.model_id, sr.hardware_id
                ORDER BY avg_accuracy ASC
            """
            
            # Execute query
            conn = self.db_api._get_connection()
            result = conn.execute(query)
            rows = result.fetchall()
            
            # Convert to list of dictionaries
            accuracy_metrics = []
            for row in rows:
                record = {}
                for idx, column in enumerate(result.description):
                    record[column[0]] = row[idx]
                accuracy_metrics.append(record)
            
            logger.info(f"Retrieved accuracy metrics for {len(accuracy_metrics)} model-hardware combinations")
            return accuracy_metrics
        except Exception as e:
            logger.error(f"Failed to get model-hardware accuracy: {e}")
            return []
    
    def get_simulation_vs_hardware_values(
        self,
        model_id: Optional[str] = None,
        hardware_id: Optional[str] = None,
        metric: str = "throughput_items_per_second",
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get pairs of simulation and hardware values for comparison.
        
        Args:
            model_id: Filter by model ID
            hardware_id: Filter by hardware ID
            metric: Metric to compare
            limit: Maximum number of results to return
            
        Returns:
            List of simulation vs. hardware value pairs
        """
        if not self.db_api:
            logger.error("Cannot get simulation vs. hardware values: No database connection")
            return []
        
        try:
            # Build query conditions
            conditions = []
            params = {}
            
            if model_id:
                conditions.append("sr.model_id = :model_id")
                params["model_id"] = model_id
            
            if hardware_id:
                conditions.append("sr.hardware_id = :hardware_id")
                params["hardware_id"] = hardware_id
            
            # Build query
            query = f"""
                SELECT 
                    sr.model_id,
                    sr.hardware_id as hardware_type,
                    sr.batch_size,
                    sr.precision,
                    vr.validation_timestamp,
                    sr.{metric} as simulation_value,
                    hr.{metric} as hardware_value,
                    (ABS(sr.{metric} - hr.{metric}) / hr.{metric} * 100) as mape
                FROM validation_results vr
                JOIN simulation_results sr ON vr.simulation_result_id = sr.id
                JOIN hardware_results hr ON vr.hardware_result_id = hr.id
                WHERE sr.{metric} IS NOT NULL AND hr.{metric} IS NOT NULL
                {" AND " + " AND ".join(conditions) if conditions else ""}
                ORDER BY vr.validation_timestamp DESC
                LIMIT :limit
            """
            params["limit"] = limit
            
            # Execute query
            conn = self.db_api._get_connection()
            result = conn.execute(query, params)
            rows = result.fetchall()
            
            # Convert to list of dictionaries
            comparison_data = []
            for row in rows:
                record = {}
                for idx, column in enumerate(result.description):
                    record[column[0]] = row[idx]
                comparison_data.append(record)
            
            logger.info(f"Retrieved {len(comparison_data)} simulation vs. hardware value pairs")
            return comparison_data
        except Exception as e:
            logger.error(f"Failed to get simulation vs. hardware values: {e}")
            return []
    
    def get_validation_metrics_over_time(
        self,
        hardware_type: Optional[str] = None,
        model_id: Optional[str] = None,
        metric: str = "throughput_mape",
        time_bucket: str = "day",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get validation metrics aggregated over time.
        
        Args:
            hardware_type: Filter by hardware type
            model_id: Filter by model ID
            metric: Metric to aggregate
            time_bucket: Time bucket for aggregation (day, week, month)
            start_date: Filter by start date
            end_date: Filter by end date
            limit: Maximum number of results to return
            
        Returns:
            List of time-aggregated metrics
        """
        if not self.db_api:
            logger.error("Cannot get validation metrics over time: No database connection")
            return []
        
        try:
            # Map time bucket to SQL function
            time_func = "DATE_TRUNC('day', validation_timestamp)"
            if time_bucket == "week":
                time_func = "DATE_TRUNC('week', validation_timestamp)"
            elif time_bucket == "month":
                time_func = "DATE_TRUNC('month', validation_timestamp)"
            
            # Build query conditions
            conditions = []
            params = {}
            
            if hardware_type:
                conditions.append("sr.hardware_id = :hardware_type")
                params["hardware_type"] = hardware_type
            
            if model_id:
                conditions.append("sr.model_id = :model_id")
                params["model_id"] = model_id
            
            if start_date:
                conditions.append("vr.validation_timestamp >= :start_date")
                params["start_date"] = start_date
            
            if end_date:
                conditions.append("vr.validation_timestamp <= :end_date")
                params["end_date"] = end_date
            
            # Build query
            query = f"""
                SELECT 
                    {time_func} as time_period,
                    COUNT(*) as num_validations,
                    AVG(vr.{metric}) as avg_metric,
                    MIN(vr.{metric}) as min_metric,
                    MAX(vr.{metric}) as max_metric,
                    STDDEV(vr.{metric}) as stddev_metric
                FROM validation_results vr
                JOIN simulation_results sr ON vr.simulation_result_id = sr.id
                {" WHERE " + " AND ".join(conditions) if conditions else ""}
                GROUP BY time_period
                ORDER BY time_period DESC
                LIMIT :limit
            """
            params["limit"] = limit
            
            # Execute query
            conn = self.db_api._get_connection()
            result = conn.execute(query, params)
            rows = result.fetchall()
            
            # Convert to list of dictionaries
            time_series = []
            for row in rows:
                record = {}
                for idx, column in enumerate(result.description):
                    record[column[0]] = row[idx]
                time_series.append(record)
            
            logger.info(f"Retrieved metrics over time for {len(time_series)} time periods")
            return time_series
        except Exception as e:
            logger.error(f"Failed to get validation metrics over time: {e}")
            return []
    
    def detect_accuracy_drift(
        self,
        hardware_type: Optional[str] = None,
        model_id: Optional[str] = None,
        metric: str = "throughput_mape",
        window_size: int = 10,
        threshold: float = 1.5  # Standard deviations from historical mean
    ) -> List[Dict[str, Any]]:
        """
        Detect significant drift in validation accuracy.
        
        Args:
            hardware_type: Filter by hardware type
            model_id: Filter by model ID
            metric: Metric to analyze
            window_size: Number of recent samples to compare
            threshold: Threshold for significance (standard deviations)
            
        Returns:
            List of detected drift events
        """
        if not self.db_api:
            logger.error("Cannot detect accuracy drift: No database connection")
            return []
        
        try:
            # Build query conditions
            conditions = []
            params = {}
            
            if hardware_type:
                conditions.append("sr.hardware_id = :hardware_type")
                params["hardware_type"] = hardware_type
            
            if model_id:
                conditions.append("sr.model_id = :model_id")
                params["model_id"] = model_id
            
            # Execute query to get validation results
            query = f"""
                SELECT 
                    vr.id,
                    sr.model_id,
                    sr.hardware_id as hardware_type,
                    vr.validation_timestamp,
                    vr.{metric} as metric_value
                FROM validation_results vr
                JOIN simulation_results sr ON vr.simulation_result_id = sr.id
                {" WHERE " + " AND ".join(conditions) if conditions else ""}
                AND vr.{metric} IS NOT NULL
                ORDER BY vr.validation_timestamp ASC
            """
            
            conn = self.db_api._get_connection()
            result = conn.execute(query, params)
            rows = result.fetchall()
            
            # Convert to list of dictionaries
            validations = []
            for row in rows:
                record = {}
                for idx, column in enumerate(result.description):
                    record[column[0]] = row[idx]
                validations.append(record)
            
            # If insufficient data, return empty list
            if len(validations) < window_size * 2:
                logger.warning(f"Insufficient data for drift detection: {len(validations)} samples")
                return []
            
            # Group by model and hardware type
            grouped = {}
            for v in validations:
                key = (v["model_id"], v["hardware_type"])
                if key not in grouped:
                    grouped[key] = []
                grouped[key].append(v)
            
            # Detect drift for each group
            drift_events = []
            import numpy as np
            
            for (model, hardware), group in grouped.items():
                # Sort by timestamp
                group.sort(key=lambda x: x["validation_timestamp"])
                
                # Need sufficient samples
                if len(group) < window_size * 2:
                    continue
                
                # Analyze in sliding windows
                for i in range(len(group) - window_size * 2 + 1):
                    historical = group[i:i+window_size]
                    recent = group[i+window_size:i+window_size*2]
                    
                    historical_values = [float(h["metric_value"]) for h in historical]
                    recent_values = [float(r["metric_value"]) for r in recent]
                    
                    historical_mean = np.mean(historical_values)
                    historical_std = np.std(historical_values)
                    
                    recent_mean = np.mean(recent_values)
                    
                    if historical_std == 0:
                        historical_std = 0.001  # Avoid division by zero
                    
                    # Calculate z-score
                    z_score = abs(recent_mean - historical_mean) / historical_std
                    
                    # Check if drift is significant
                    is_significant = z_score > threshold
                    
                    if is_significant:
                        drift_event = {
                            "model_id": model,
                            "hardware_type": hardware,
                            "metric": metric,
                            "historical_window_start": historical[0]["validation_timestamp"],
                            "historical_window_end": historical[-1]["validation_timestamp"],
                            "recent_window_start": recent[0]["validation_timestamp"],
                            "recent_window_end": recent[-1]["validation_timestamp"],
                            "historical_mean": historical_mean,
                            "recent_mean": recent_mean,
                            "z_score": z_score,
                            "threshold": threshold,
                            "is_significant": is_significant
                        }
                        drift_events.append(drift_event)
            
            logger.info(f"Detected {len(drift_events)} drift events")
            return drift_events
        except Exception as e:
            logger.error(f"Failed to detect accuracy drift: {e}")
            return []
    
    def analyze_calibration_effectiveness(
        self,
        hardware_type: Optional[str] = None,
        model_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze the effectiveness of calibration over time.
        
        Args:
            hardware_type: Filter by hardware type
            model_type: Filter by model type
            start_date: Filter by start date
            end_date: Filter by end date
            
        Returns:
            Dictionary with calibration effectiveness metrics
        """
        if not self.db_api:
            logger.error("Cannot analyze calibration effectiveness: No database connection")
            return {}
        
        try:
            # Build query conditions
            conditions = []
            params = {}
            
            if hardware_type:
                conditions.append("hardware_type = :hardware_type")
                params["hardware_type"] = hardware_type
            
            if model_type:
                conditions.append("model_type = :model_type")
                params["model_type"] = model_type
            
            if start_date:
                conditions.append("timestamp >= :start_date")
                params["start_date"] = start_date
            
            if end_date:
                conditions.append("timestamp <= :end_date")
                params["end_date"] = end_date
            
            # Get calibration history
            query = f"""
                SELECT 
                    id,
                    hardware_type,
                    model_type,
                    timestamp,
                    improvement_metrics
                FROM calibration_history
                {" WHERE " + " AND ".join(conditions) if conditions else ""}
                ORDER BY timestamp ASC
            """
            
            conn = self.db_api._get_connection()
            result = conn.execute(query, params)
            rows = result.fetchall()
            
            # Process results
            calibration_events = []
            for row in rows:
                # Extract improvement metrics
                improvement_metrics = json.loads(row[4]) if row[4] else {}
                
                event = {
                    "id": row[0],
                    "hardware_type": row[1],
                    "model_type": row[2],
                    "timestamp": row[3],
                    "improvement_metrics": improvement_metrics
                }
                calibration_events.append(event)
            
            # If no events, return empty result
            if not calibration_events:
                return {
                    "status": "no_data",
                    "message": "No calibration events found for the specified criteria"
                }
            
            # Analyze improvements over time
            import numpy as np
            
            # Group by hardware and model type
            grouped = {}
            for event in calibration_events:
                key = (event["hardware_type"], event["model_type"])
                if key not in grouped:
                    grouped[key] = []
                grouped[key].append(event)
            
            # Calculate statistics for each group
            group_stats = {}
            overall_stats = {
                "total_events": len(calibration_events),
                "mean_relative_improvement": [],
                "mean_mape_before": [],
                "mean_mape_after": []
            }
            
            for (hardware, model), events in grouped.items():
                improvements = []
                mapes_before = []
                mapes_after = []
                
                for event in events:
                    metrics = event["improvement_metrics"]
                    if "overall" in metrics:
                        overall = metrics["overall"]
                        if "relative_improvement_pct" in overall:
                            improvements.append(float(overall["relative_improvement_pct"]))
                        
                        if "before_mape" in overall:
                            mapes_before.append(float(overall["before_mape"]))
                        
                        if "after_mape" in overall:
                            mapes_after.append(float(overall["after_mape"]))
                
                # Calculate statistics if we have data
                if improvements:
                    stats = {
                        "count": len(improvements),
                        "mean_relative_improvement": np.mean(improvements),
                        "median_relative_improvement": np.median(improvements),
                        "max_relative_improvement": np.max(improvements),
                        "min_relative_improvement": np.min(improvements),
                        "std_relative_improvement": np.std(improvements)
                    }
                    
                    if mapes_before:
                        stats["mean_mape_before"] = np.mean(mapes_before)
                    
                    if mapes_after:
                        stats["mean_mape_after"] = np.mean(mapes_after)
                    
                    group_stats[f"{hardware}-{model}"] = stats
                    
                    # Add to overall stats
                    overall_stats["mean_relative_improvement"].extend(improvements)
                    overall_stats["mean_mape_before"].extend(mapes_before)
                    overall_stats["mean_mape_after"].extend(mapes_after)
            
            # Calculate overall statistics
            if overall_stats["mean_relative_improvement"]:
                overall_stats["mean_relative_improvement"] = np.mean(overall_stats["mean_relative_improvement"])
            else:
                overall_stats["mean_relative_improvement"] = None
            
            if overall_stats["mean_mape_before"]:
                overall_stats["mean_mape_before"] = np.mean(overall_stats["mean_mape_before"])
            else:
                overall_stats["mean_mape_before"] = None
            
            if overall_stats["mean_mape_after"]:
                overall_stats["mean_mape_after"] = np.mean(overall_stats["mean_mape_after"])
            else:
                overall_stats["mean_mape_after"] = None
            
            # Prepare result
            analysis = {
                "status": "success",
                "overall": overall_stats,
                "by_hardware_model": group_stats,
                "calibration_events": len(calibration_events),
                "hardware_model_combinations": len(grouped)
            }
            
            logger.info(f"Analyzed effectiveness of {len(calibration_events)} calibration events")
            return analysis
        except Exception as e:
            logger.error(f"Failed to analyze calibration effectiveness: {e}")
            return {"status": "error", "message": str(e)}
    
    def export_data_for_visualization(
        self,
        query_type: str,
        hardware_type: Optional[str] = None,
        model_id: Optional[str] = None,
        metric: str = "throughput_items_per_second",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 1000
    ) -> Dict[str, Any]:
        """
        Export data from the database in a format suitable for visualization.
        
        Args:
            query_type: Type of data to export (e.g., "sim_vs_hw", "mape_by_hardware", "drift")
            hardware_type: Filter by hardware type
            model_id: Filter by model ID
            metric: Metric to analyze
            start_date: Filter by start date
            end_date: Filter by end date
            limit: Maximum number of results to return
            
        Returns:
            Dictionary with data for visualization
        """
        if not self.db_api:
            logger.error("Cannot export data for visualization: No database connection")
            return {"status": "error", "message": "No database connection"}
        
        try:
            result = {
                "status": "success",
                "query_type": query_type,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            if query_type == "sim_vs_hw":
                # Get simulation vs. hardware values
                data = self.get_simulation_vs_hardware_values(
                    model_id=model_id,
                    hardware_id=hardware_type,
                    metric=metric,
                    limit=limit
                )
                result["data"] = data
                result["metric"] = metric
                
            elif query_type == "mape_by_hardware":
                # Get MAPE by hardware
                query = f"""
                    SELECT 
                        sr.hardware_id as hardware_type,
                        sr.model_id,
                        AVG(vr.{metric.replace('items_per_second', 'mape')}) as avg_mape,
                        COUNT(*) as num_validations
                    FROM validation_results vr
                    JOIN simulation_results sr ON vr.simulation_result_id = sr.id
                    WHERE vr.{metric.replace('items_per_second', 'mape')} IS NOT NULL
                """
                
                conditions = []
                params = {}
                
                if hardware_type:
                    conditions.append("sr.hardware_id = :hardware_type")
                    params["hardware_type"] = hardware_type
                
                if model_id:
                    conditions.append("sr.model_id = :model_id")
                    params["model_id"] = model_id
                
                if start_date:
                    conditions.append("vr.validation_timestamp >= :start_date")
                    params["start_date"] = start_date
                
                if end_date:
                    conditions.append("vr.validation_timestamp <= :end_date")
                    params["end_date"] = end_date
                
                if conditions:
                    query += " AND " + " AND ".join(conditions)
                
                query += """
                    GROUP BY sr.hardware_id, sr.model_id
                    ORDER BY avg_mape ASC
                    LIMIT :limit
                """
                params["limit"] = limit
                
                conn = self.db_api._get_connection()
                db_result = conn.execute(query, params)
                rows = db_result.fetchall()
                
                data = []
                for row in rows:
                    record = {}
                    for idx, column in enumerate(db_result.description):
                        record[column[0]] = row[idx]
                    data.append(record)
                
                result["data"] = data
                result["metric"] = metric.replace('items_per_second', 'mape')
                
            elif query_type == "drift":
                # Get drift detection results
                data = self.get_drift_detection_results(
                    hardware_type=hardware_type,
                    model_type=model_id,
                    is_significant=True,
                    start_date=start_date,
                    end_date=end_date,
                    limit=limit
                )
                result["data"] = data
                
            elif query_type == "metrics_over_time":
                # Get validation metrics over time
                data = self.get_validation_metrics_over_time(
                    hardware_type=hardware_type,
                    model_id=model_id,
                    metric=metric.replace('items_per_second', 'mape'),
                    start_date=start_date,
                    end_date=end_date,
                    limit=limit
                )
                result["data"] = data
                result["metric"] = metric.replace('items_per_second', 'mape')
                
            else:
                result["status"] = "error"
                result["message"] = f"Unknown query type: {query_type}"
            
            return result
        except Exception as e:
            logger.error(f"Failed to export data for visualization: {e}")
            return {"status": "error", "message": str(e)}
    
    def close(self):
        """Close the database connection."""
        # There's no explicit close method in the BenchmarkDBAPI
        # The DuckDB connections are created on demand and closed after use
        logger.info("Database connection resources released")


def get_db_integration_instance(db_path: str = "./benchmark_db.duckdb", debug: bool = False) -> SimulationValidationDBIntegration:
    """
    Get an instance of the SimulationValidationDBIntegration.
    
    Args:
        db_path: Path to the DuckDB database
        debug: Enable debug logging
        
    Returns:
        SimulationValidationDBIntegration instance
    """
    return SimulationValidationDBIntegration(db_path=db_path, debug=debug)