#!/usr/bin/env python3
"""
Database schema for the Simulation Accuracy and Validation Framework.

This module defines the database schema for storing simulation results, hardware
measurements, validation results, and calibration data.
"""

import datetime
from typing import Dict, Any, List, Optional, Union


# Table schemas for DuckDB
SIMULATION_RESULTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS simulation_results (
    id VARCHAR PRIMARY KEY,
    model_id VARCHAR NOT NULL,
    hardware_id VARCHAR NOT NULL,
    batch_size INTEGER NOT NULL,
    precision VARCHAR NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    simulation_version VARCHAR NOT NULL,
    additional_metadata JSON,
    throughput_items_per_second DOUBLE,
    average_latency_ms DOUBLE,
    memory_peak_mb DOUBLE,
    power_consumption_w DOUBLE,
    initialization_time_ms DOUBLE,
    warmup_time_ms DOUBLE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

HARDWARE_RESULTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS hardware_results (
    id VARCHAR PRIMARY KEY,
    model_id VARCHAR NOT NULL,
    hardware_id VARCHAR NOT NULL,
    batch_size INTEGER NOT NULL,
    precision VARCHAR NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    hardware_details JSON,
    test_environment JSON,
    additional_metadata JSON,
    throughput_items_per_second DOUBLE,
    average_latency_ms DOUBLE,
    memory_peak_mb DOUBLE,
    power_consumption_w DOUBLE,
    initialization_time_ms DOUBLE,
    warmup_time_ms DOUBLE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

VALIDATION_RESULTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS validation_results (
    id VARCHAR PRIMARY KEY,
    simulation_result_id VARCHAR NOT NULL,
    hardware_result_id VARCHAR NOT NULL,
    validation_timestamp TIMESTAMP NOT NULL,
    validation_version VARCHAR NOT NULL,
    metrics_comparison JSON NOT NULL,
    additional_metrics JSON,
    overall_accuracy_score DOUBLE,
    throughput_mape DOUBLE,
    latency_mape DOUBLE,
    memory_mape DOUBLE,
    power_mape DOUBLE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (simulation_result_id) REFERENCES simulation_results (id),
    FOREIGN KEY (hardware_result_id) REFERENCES hardware_results (id)
);
"""

CALIBRATION_HISTORY_SCHEMA = """
CREATE TABLE IF NOT EXISTS calibration_history (
    id VARCHAR PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    hardware_type VARCHAR NOT NULL,
    model_type VARCHAR NOT NULL,
    previous_parameters JSON NOT NULL,
    updated_parameters JSON NOT NULL,
    validation_results_before JSON,
    validation_results_after JSON,
    improvement_metrics JSON,
    calibration_version VARCHAR NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

DRIFT_DETECTION_SCHEMA = """
CREATE TABLE IF NOT EXISTS drift_detection (
    id VARCHAR PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    hardware_type VARCHAR NOT NULL,
    model_type VARCHAR NOT NULL,
    drift_metrics JSON NOT NULL,
    is_significant BOOLEAN NOT NULL,
    historical_window_start TIMESTAMP NOT NULL,
    historical_window_end TIMESTAMP NOT NULL,
    new_window_start TIMESTAMP NOT NULL,
    new_window_end TIMESTAMP NOT NULL,
    thresholds_used JSON NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

SIMULATION_PARAMETERS_SCHEMA = """
CREATE TABLE IF NOT EXISTS simulation_parameters (
    id VARCHAR PRIMARY KEY,
    hardware_type VARCHAR NOT NULL,
    model_type VARCHAR NOT NULL,
    parameters JSON NOT NULL,
    version VARCHAR NOT NULL,
    is_current BOOLEAN NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# Combined schema
SIMULATION_VALIDATION_SCHEMA = [
    SIMULATION_RESULTS_SCHEMA,
    HARDWARE_RESULTS_SCHEMA,
    VALIDATION_RESULTS_SCHEMA,
    CALIBRATION_HISTORY_SCHEMA,
    DRIFT_DETECTION_SCHEMA,
    SIMULATION_PARAMETERS_SCHEMA
]


class SimulationValidationSchema:
    """Class for managing the simulation validation database schema."""
    
    @staticmethod
    def create_tables(db_conn) -> None:
        """
        Create the necessary tables in the database.
        
        Args:
            db_conn: Database connection
        """
        try:
            for schema in SIMULATION_VALIDATION_SCHEMA:
                db_conn.execute(schema)
            
            db_conn.commit()
            print("Successfully created simulation validation tables")
            
        except Exception as e:
            print(f"Error creating simulation validation tables: {e}")
            db_conn.rollback()
    
    @staticmethod
    def generate_id(prefix: str) -> str:
        """
        Generate a unique ID for database records.
        
        Args:
            prefix: Prefix for the ID (e.g., "sim", "hw", "val")
            
        Returns:
            A unique ID string
        """
        import uuid
        import time
        
        timestamp = int(time.time())
        random_part = uuid.uuid4().hex[:8]
        return f"{prefix}_{timestamp}_{random_part}"
    
    @staticmethod
    def simulation_result_to_db_dict(sim_result, generate_id: bool = True) -> Dict[str, Any]:
        """
        Convert a SimulationResult to a database record dictionary.
        
        Args:
            sim_result: SimulationResult object
            generate_id: Whether to generate a new ID
            
        Returns:
            Dictionary for database insertion
        """
        record = {
            "model_id": sim_result.model_id,
            "hardware_id": sim_result.hardware_id,
            "batch_size": sim_result.batch_size,
            "precision": sim_result.precision,
            "timestamp": sim_result.timestamp,
            "simulation_version": sim_result.simulation_version,
            "additional_metadata": sim_result.additional_metadata
        }
        
        # Extract specific metrics
        metrics = sim_result.metrics
        record["throughput_items_per_second"] = metrics.get("throughput_items_per_second")
        record["average_latency_ms"] = metrics.get("average_latency_ms")
        record["memory_peak_mb"] = metrics.get("memory_peak_mb")
        record["power_consumption_w"] = metrics.get("power_consumption_w")
        record["initialization_time_ms"] = metrics.get("initialization_time_ms")
        record["warmup_time_ms"] = metrics.get("warmup_time_ms")
        
        if generate_id:
            record["id"] = SimulationValidationSchema.generate_id("sim")
        
        return record
    
    @staticmethod
    def hardware_result_to_db_dict(hw_result, generate_id: bool = True) -> Dict[str, Any]:
        """
        Convert a HardwareResult to a database record dictionary.
        
        Args:
            hw_result: HardwareResult object
            generate_id: Whether to generate a new ID
            
        Returns:
            Dictionary for database insertion
        """
        record = {
            "model_id": hw_result.model_id,
            "hardware_id": hw_result.hardware_id,
            "batch_size": hw_result.batch_size,
            "precision": hw_result.precision,
            "timestamp": hw_result.timestamp,
            "hardware_details": hw_result.hardware_details,
            "test_environment": hw_result.test_environment,
            "additional_metadata": hw_result.additional_metadata
        }
        
        # Extract specific metrics
        metrics = hw_result.metrics
        record["throughput_items_per_second"] = metrics.get("throughput_items_per_second")
        record["average_latency_ms"] = metrics.get("average_latency_ms")
        record["memory_peak_mb"] = metrics.get("memory_peak_mb")
        record["power_consumption_w"] = metrics.get("power_consumption_w")
        record["initialization_time_ms"] = metrics.get("initialization_time_ms")
        record["warmup_time_ms"] = metrics.get("warmup_time_ms")
        
        if generate_id:
            record["id"] = SimulationValidationSchema.generate_id("hw")
        
        return record
    
    @staticmethod
    def validation_result_to_db_dict(val_result, sim_id: str, hw_id: str, generate_id: bool = True) -> Dict[str, Any]:
        """
        Convert a ValidationResult to a database record dictionary.
        
        Args:
            val_result: ValidationResult object
            sim_id: ID of the simulation result in the database
            hw_id: ID of the hardware result in the database
            generate_id: Whether to generate a new ID
            
        Returns:
            Dictionary for database insertion
        """
        record = {
            "simulation_result_id": sim_id,
            "hardware_result_id": hw_id,
            "validation_timestamp": val_result.validation_timestamp,
            "validation_version": val_result.validation_version,
            "metrics_comparison": val_result.metrics_comparison,
            "additional_metrics": val_result.additional_metrics
        }
        
        # Extract specific metrics if available
        throughput_metrics = val_result.metrics_comparison.get("throughput_items_per_second", {})
        latency_metrics = val_result.metrics_comparison.get("average_latency_ms", {})
        memory_metrics = val_result.metrics_comparison.get("memory_peak_mb", {})
        power_metrics = val_result.metrics_comparison.get("power_consumption_w", {})
        
        record["throughput_mape"] = throughput_metrics.get("mape")
        record["latency_mape"] = latency_metrics.get("mape")
        record["memory_mape"] = memory_metrics.get("mape")
        record["power_mape"] = power_metrics.get("mape")
        
        # Calculate overall accuracy score as average of available MAPEs
        mapes = [v for v in [
            record["throughput_mape"], 
            record["latency_mape"],
            record["memory_mape"],
            record["power_mape"]
        ] if v is not None]
        
        if mapes:
            record["overall_accuracy_score"] = sum(mapes) / len(mapes)
        
        if generate_id:
            record["id"] = SimulationValidationSchema.generate_id("val")
        
        return record
    
    @staticmethod
    def calibration_to_db_dict(
        hardware_type: str,
        model_type: str,
        previous_parameters: Dict[str, Any],
        updated_parameters: Dict[str, Any],
        validation_results_before: Optional[List[Dict[str, Any]]] = None,
        validation_results_after: Optional[List[Dict[str, Any]]] = None,
        improvement_metrics: Optional[Dict[str, Any]] = None,
        calibration_version: str = "v1",
        generate_id: bool = True
    ) -> Dict[str, Any]:
        """
        Create a database record dictionary for calibration history.
        
        Args:
            hardware_type: Type of hardware
            model_type: Type of model
            previous_parameters: Parameters before calibration
            updated_parameters: Parameters after calibration
            validation_results_before: Validation results before calibration
            validation_results_after: Validation results after calibration
            improvement_metrics: Metrics quantifying the calibration improvement
            calibration_version: Version of the calibration methodology
            generate_id: Whether to generate a new ID
            
        Returns:
            Dictionary for database insertion
        """
        record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "hardware_type": hardware_type,
            "model_type": model_type,
            "previous_parameters": previous_parameters,
            "updated_parameters": updated_parameters,
            "validation_results_before": validation_results_before,
            "validation_results_after": validation_results_after,
            "improvement_metrics": improvement_metrics,
            "calibration_version": calibration_version
        }
        
        if generate_id:
            record["id"] = SimulationValidationSchema.generate_id("cal")
        
        return record
    
    @staticmethod
    def drift_detection_to_db_dict(
        hardware_type: str,
        model_type: str,
        drift_metrics: Dict[str, Any],
        is_significant: bool,
        historical_window_start: str,
        historical_window_end: str,
        new_window_start: str,
        new_window_end: str,
        thresholds_used: Dict[str, float],
        generate_id: bool = True
    ) -> Dict[str, Any]:
        """
        Create a database record dictionary for drift detection.
        
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
            generate_id: Whether to generate a new ID
            
        Returns:
            Dictionary for database insertion
        """
        record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "hardware_type": hardware_type,
            "model_type": model_type,
            "drift_metrics": drift_metrics,
            "is_significant": is_significant,
            "historical_window_start": historical_window_start,
            "historical_window_end": historical_window_end,
            "new_window_start": new_window_start,
            "new_window_end": new_window_end,
            "thresholds_used": thresholds_used
        }
        
        if generate_id:
            record["id"] = SimulationValidationSchema.generate_id("drift")
        
        return record