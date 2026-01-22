#!/usr/bin/env python3
"""
Distributed Testing Framework - Database API Extensions for Result Aggregation

This module extends the BenchmarkDBAPI class with methods specifically designed
to support the ResultAggregatorService. It provides specialized query methods
for retrieving and filtering data at different aggregation levels.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("aggregation_db_extensions")

# Add parent directory to path to import modules from parent
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


def extend_benchmark_db_api(cls):
    """
    Decorator function to extend the BenchmarkDBAPI class with methods
    for supporting the ResultAggregatorService.
    
    Args:
        cls: The BenchmarkDBAPI class to extend
        
    Returns:
        The extended BenchmarkDBAPI class
    """
    
    def get_performance_results(self, 
                              aggregation_level: str, 
                              filter_params: Dict[str, Any] = None,
                              time_range: Optional[Tuple[datetime, datetime]] = None) -> List[Dict[str, Any]]:
        """Get performance results, filtered and formatted for aggregation.
        
        Args:
            aggregation_level: Level of aggregation (test_run, model, hardware, etc.)
            filter_params: Parameters to filter results by
            time_range: Optional time range to filter by (start, end)
            
        Returns:
            List of performance result dictionaries
        """
        filter_params = filter_params or {}
        
        # Build the SQL query
        query = """
            SELECT 
                pr.result_id,
                pr.run_id,
                pr.model_id,
                m.model_name,
                m.model_family,
                m.modality,
                pr.hardware_id,
                h.device_name,
                h.hardware_type,
                pr.test_case,
                pr.batch_size,
                pr.precision,
                pr.total_time_seconds,
                pr.average_latency_ms,
                pr.throughput_items_per_second,
                pr.memory_peak_mb,
                pr.iterations,
                pr.warmup_iterations,
                pr.is_simulated,
                tr.started_at AS timestamp,
                pr.metrics
            FROM 
                performance_results pr
            JOIN 
                models m ON pr.model_id = m.model_id
            JOIN 
                hardware_platforms h ON pr.hardware_id = h.hardware_id
            JOIN 
                test_runs tr ON pr.run_id = tr.run_id
            WHERE 1=1
        """
        
        params = {}
        
        # Add filters
        if "model_id" in filter_params:
            query += " AND pr.model_id = ?";
            params["p1"] = filter_params["model_id"]
            
        if "model_name" in filter_params:
            query += " AND m.model_name = ?";
            params["p2"] = filter_params["model_name"]
            
        if "model_family" in filter_params:
            query += " AND m.model_family = ?";
            params["p3"] = filter_params["model_family"]
            
        if "hardware_id" in filter_params:
            query += " AND pr.hardware_id = ?";
            params["p4"] = filter_params["hardware_id"]
            
        if "hardware_type" in filter_params:
            query += " AND h.hardware_type = ?";
            params["p5"] = filter_params["hardware_type"]
            
        if "run_id" in filter_params:
            query += " AND pr.run_id = ?";
            params["p6"] = filter_params["run_id"]
            
        if "precision" in filter_params:
            query += " AND pr.precision = ?";
            params["p7"] = filter_params["precision"]
            
        if "batch_size" in filter_params:
            query += " AND pr.batch_size = ?";
            params["p8"] = filter_params["batch_size"]
            
        if "is_simulated" in filter_params:
            query += " AND pr.is_simulated = ?";
            params["p9"] = filter_params["is_simulated"]
            
        # Add time range filter if specified
        if time_range:
            start_time, end_time = time_range
            query += " AND tr.started_at >= ? AND tr.started_at <= ?"
            params["start_time"] = start_time
            params["end_time"] = end_time
            
        # Order by timestamp
        query += " ORDER BY tr.started_at DESC"
        
        # Execute the query
        try:
            cursor = self.conn.cursor()
            result = cursor.execute(query, params).fetchall()
            
            # Convert to list of dictionaries
            columns = [col[0] for col in cursor.description]
            results = []
            
            for row in result:
                # Convert row to dictionary
                result_dict = dict(zip(columns, row))
                
                # Parse JSON fields if needed
                if "metrics" in result_dict and result_dict["metrics"]:
                    try:
                        result_dict["metrics"] = self.db_json_to_dict(result_dict["metrics"])
                    except Exception as e:
                        logger.warning(f"Error parsing metrics JSON: {e}")
                        
                results.append(result_dict)
                
            logger.debug(f"Retrieved {len(results)} performance results")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving performance results: {e}")
            return []
    
    def get_compatibility_results(self, 
                                aggregation_level: str, 
                                filter_params: Dict[str, Any] = None,
                                time_range: Optional[Tuple[datetime, datetime]] = None) -> List[Dict[str, Any]]:
        """Get hardware compatibility results, filtered and formatted for aggregation.
        
        Args:
            aggregation_level: Level of aggregation (test_run, model, hardware, etc.)
            filter_params: Parameters to filter results by
            time_range: Optional time range to filter by (start, end)
            
        Returns:
            List of compatibility result dictionaries
        """
        filter_params = filter_params or {}
        
        # Build the SQL query
        query = """
            SELECT 
                hc.compatibility_id,
                hc.run_id,
                hc.model_id,
                m.model_name,
                m.model_family,
                m.modality,
                hc.hardware_id,
                h.device_name,
                h.hardware_type,
                hc.is_compatible,
                hc.detection_success,
                hc.initialization_success,
                hc.error_message,
                hc.error_type,
                hc.suggested_fix,
                hc.workaround_available,
                hc.compatibility_score,
                hc.is_simulated,
                tr.started_at AS timestamp,
                hc.metadata
            FROM 
                hardware_compatibility hc
            JOIN 
                models m ON hc.model_id = m.model_id
            JOIN 
                hardware_platforms h ON hc.hardware_id = h.hardware_id
            JOIN 
                test_runs tr ON hc.run_id = tr.run_id
            WHERE 1=1
        """
        
        params = {}
        
        # Add filters
        if "model_id" in filter_params:
            query += " AND hc.model_id = ?";
            params["p1"] = filter_params["model_id"]
            
        if "model_name" in filter_params:
            query += " AND m.model_name = ?";
            params["p2"] = filter_params["model_name"]
            
        if "model_family" in filter_params:
            query += " AND m.model_family = ?";
            params["p3"] = filter_params["model_family"]
            
        if "hardware_id" in filter_params:
            query += " AND hc.hardware_id = ?";
            params["p4"] = filter_params["hardware_id"]
            
        if "hardware_type" in filter_params:
            query += " AND h.hardware_type = ?";
            params["p5"] = filter_params["hardware_type"]
            
        if "run_id" in filter_params:
            query += " AND hc.run_id = ?";
            params["p6"] = filter_params["run_id"]
            
        if "is_compatible" in filter_params:
            query += " AND hc.is_compatible = ?";
            params["p7"] = filter_params["is_compatible"]
            
        if "is_simulated" in filter_params:
            query += " AND hc.is_simulated = ?";
            params["p8"] = filter_params["is_simulated"]
            
        # Add time range filter if specified
        if time_range:
            start_time, end_time = time_range
            query += " AND tr.started_at >= ? AND tr.started_at <= ?"
            params["start_time"] = start_time
            params["end_time"] = end_time
            
        # Order by timestamp
        query += " ORDER BY tr.started_at DESC"
        
        # Execute the query
        try:
            cursor = self.conn.cursor()
            result = cursor.execute(query, params).fetchall()
            
            # Convert to list of dictionaries
            columns = [col[0] for col in cursor.description]
            results = []
            
            for row in result:
                # Convert row to dictionary
                result_dict = dict(zip(columns, row))
                
                # Parse JSON fields if needed
                if "metadata" in result_dict and result_dict["metadata"]:
                    try:
                        result_dict["metadata"] = self.db_json_to_dict(result_dict["metadata"])
                    except Exception as e:
                        logger.warning(f"Error parsing metadata JSON: {e}")
                        
                results.append(result_dict)
                
            logger.debug(f"Retrieved {len(results)} compatibility results")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving compatibility results: {e}")
            return []
    
    def get_integration_test_results(self, 
                                   aggregation_level: str, 
                                   filter_params: Dict[str, Any] = None,
                                   time_range: Optional[Tuple[datetime, datetime]] = None) -> List[Dict[str, Any]]:
        """Get integration test results, filtered and formatted for aggregation.
        
        Args:
            aggregation_level: Level of aggregation (test_run, model, hardware, etc.)
            filter_params: Parameters to filter results by
            time_range: Optional time range to filter by (start, end)
            
        Returns:
            List of integration test result dictionaries
        """
        filter_params = filter_params or {}
        
        # Build the SQL query
        query = """
            SELECT 
                itr.test_result_id,
                itr.run_id,
                itr.test_module,
                itr.test_class,
                itr.test_name,
                itr.status,
                itr.execution_time_seconds,
                itr.hardware_id,
                itr.model_id,
                itr.error_message,
                itr.error_traceback,
                tr.started_at AS timestamp,
                itr.metadata
            FROM 
                integration_test_results itr
            JOIN 
                test_runs tr ON itr.run_id = tr.run_id
            LEFT JOIN
                models m ON itr.model_id = m.model_id
            LEFT JOIN
                hardware_platforms h ON itr.hardware_id = h.hardware_id
            WHERE 1=1
        """
        
        params = {}
        
        # Add filters
        if "test_module" in filter_params:
            query += " AND itr.test_module = ?";
            params["p1"] = filter_params["test_module"]
            
        if "test_class" in filter_params:
            query += " AND itr.test_class = ?";
            params["p2"] = filter_params["test_class"]
            
        if "test_name" in filter_params:
            query += " AND itr.test_name = ?";
            params["p3"] = filter_params["test_name"]
            
        if "status" in filter_params:
            query += " AND itr.status = ?";
            params["p4"] = filter_params["status"]
            
        if "model_id" in filter_params:
            query += " AND itr.model_id = ?";
            params["p5"] = filter_params["model_id"]
            
        if "hardware_id" in filter_params:
            query += " AND itr.hardware_id = ?";
            params["p6"] = filter_params["hardware_id"]
            
        if "run_id" in filter_params:
            query += " AND itr.run_id = ?";
            params["p7"] = filter_params["run_id"]
            
        # Add time range filter if specified
        if time_range:
            start_time, end_time = time_range
            query += " AND tr.started_at >= ? AND tr.started_at <= ?"
            params["start_time"] = start_time
            params["end_time"] = end_time
            
        # Order by timestamp
        query += " ORDER BY tr.started_at DESC"
        
        # Execute the query
        try:
            cursor = self.conn.cursor()
            result = cursor.execute(query, params).fetchall()
            
            # Convert to list of dictionaries
            columns = [col[0] for col in cursor.description]
            results = []
            
            for row in result:
                # Convert row to dictionary
                result_dict = dict(zip(columns, row))
                
                # Parse JSON fields if needed
                if "metadata" in result_dict and result_dict["metadata"]:
                    try:
                        result_dict["metadata"] = self.db_json_to_dict(result_dict["metadata"])
                    except Exception as e:
                        logger.warning(f"Error parsing metadata JSON: {e}")
                        
                results.append(result_dict)
                
            logger.debug(f"Retrieved {len(results)} integration test results")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving integration test results: {e}")
            return []
    
    def get_web_platform_results(self, 
                               aggregation_level: str, 
                               filter_params: Dict[str, Any] = None,
                               time_range: Optional[Tuple[datetime, datetime]] = None) -> List[Dict[str, Any]]:
        """Get web platform test results, filtered and formatted for aggregation.
        
        Args:
            aggregation_level: Level of aggregation (test_run, model, hardware, etc.)
            filter_params: Parameters to filter results by
            time_range: Optional time range to filter by (start, end)
            
        Returns:
            List of web platform result dictionaries
        """
        filter_params = filter_params or {}
        
        # Build the SQL query
        query = """
            SELECT 
                wpr.result_id,
                wpr.run_id,
                wpr.model_id,
                m.model_name,
                m.model_family,
                m.modality,
                wpr.hardware_id,
                wpr.platform,
                wpr.browser,
                wpr.browser_version,
                wpr.test_file,
                wpr.success,
                wpr.load_time_ms,
                wpr.initialization_time_ms,
                wpr.inference_time_ms,
                wpr.total_time_ms,
                wpr.shader_compilation_time_ms,
                wpr.memory_usage_mb,
                wpr.error_message,
                tr.started_at AS timestamp,
                wpr.metrics
            FROM 
                web_platform_results wpr
            JOIN 
                models m ON wpr.model_id = m.model_id
            JOIN 
                test_runs tr ON wpr.run_id = tr.run_id
            WHERE 1=1
        """
        
        params = {}
        
        # Add filters
        if "model_id" in filter_params:
            query += " AND wpr.model_id = ?";
            params["p1"] = filter_params["model_id"]
            
        if "model_name" in filter_params:
            query += " AND m.model_name = ?";
            params["p2"] = filter_params["model_name"]
            
        if "model_family" in filter_params:
            query += " AND m.model_family = ?";
            params["p3"] = filter_params["model_family"]
            
        if "platform" in filter_params:
            query += " AND wpr.platform = ?";
            params["p4"] = filter_params["platform"]
            
        if "browser" in filter_params:
            query += " AND wpr.browser = ?";
            params["p5"] = filter_params["browser"]
            
        if "success" in filter_params:
            query += " AND wpr.success = ?";
            params["p6"] = filter_params["success"]
            
        if "run_id" in filter_params:
            query += " AND wpr.run_id = ?";
            params["p7"] = filter_params["run_id"]
            
        # Add time range filter if specified
        if time_range:
            start_time, end_time = time_range
            query += " AND tr.started_at >= ? AND tr.started_at <= ?"
            params["start_time"] = start_time
            params["end_time"] = end_time
            
        # Order by timestamp
        query += " ORDER BY tr.started_at DESC"
        
        # Execute the query
        try:
            cursor = self.conn.cursor()
            result = cursor.execute(query, params).fetchall()
            
            # Convert to list of dictionaries
            columns = [col[0] for col in cursor.description]
            results = []
            
            for row in result:
                # Convert row to dictionary
                result_dict = dict(zip(columns, row))
                
                # Parse JSON fields if needed
                if "metrics" in result_dict and result_dict["metrics"]:
                    try:
                        result_dict["metrics"] = self.db_json_to_dict(result_dict["metrics"])
                    except Exception as e:
                        logger.warning(f"Error parsing metrics JSON: {e}")
                        
                results.append(result_dict)
                
            logger.debug(f"Retrieved {len(results)} web platform results")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving web platform results: {e}")
            return []
    
    def get_hardware_info(self, hardware_id: str) -> Dict[str, Any]:
        """Get information about a specific hardware platform.
        
        Args:
            hardware_id: ID of the hardware platform
            
        Returns:
            Dictionary with hardware information, or empty dict if not found
        """
        query = """
            SELECT 
                hardware_id,
                hardware_type,
                device_name,
                platform,
                platform_version,
                driver_version,
                memory_gb,
                compute_units,
                metadata
            FROM 
                hardware_platforms
            WHERE 
                hardware_id = ?
        """
        
        try:
            cursor = self.conn.cursor()
            result = cursor.execute(query, {"hardware_id": hardware_id}).fetchone()
            
            if not result:
                logger.warning(f"Hardware ID {hardware_id} not found")
                return {}
                
            # Convert to dictionary
            columns = [col[0] for col in cursor.description]
            hardware_info = dict(zip(columns, result))
            
            # Parse JSON fields if needed
            if "metadata" in hardware_info and hardware_info["metadata"]:
                try:
                    hardware_info["metadata"] = self.db_json_to_dict(hardware_info["metadata"])
                except Exception as e:
                    logger.warning(f"Error parsing metadata JSON: {e}")
                    
            return hardware_info
            
        except Exception as e:
            logger.error(f"Error retrieving hardware info: {e}")
            return {}
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get information about a specific model.
        
        Args:
            model_id: ID of the model
            
        Returns:
            Dictionary with model information, or empty dict if not found
        """
        query = """
            SELECT 
                model_id,
                model_name,
                model_family,
                modality,
                source,
                version,
                parameters_million,
                metadata
            FROM 
                models
            WHERE 
                model_id = ?
        """
        
        try:
            cursor = self.conn.cursor()
            result = cursor.execute(query, {"model_id": model_id}).fetchone()
            
            if not result:
                logger.warning(f"Model ID {model_id} not found")
                return {}
                
            # Convert to dictionary
            columns = [col[0] for col in cursor.description]
            model_info = dict(zip(columns, result))
            
            # Parse JSON fields if needed
            if "metadata" in model_info and model_info["metadata"]:
                try:
                    model_info["metadata"] = self.db_json_to_dict(model_info["metadata"])
                except Exception as e:
                    logger.warning(f"Error parsing metadata JSON: {e}")
                    
            return model_info
            
        except Exception as e:
            logger.error(f"Error retrieving model info: {e}")
            return {}
    
    def add_performance_anomaly(self,
                              worker_id: Optional[str],
                              entity_type: str,
                              entity_id: Optional[str] = None,
                              metric: str = "",
                              timestamp: Optional[datetime] = None,
                              value: float = 0.0,
                              baseline_mean: float = 0.0,
                              baseline_stdev: float = 0.0,
                              z_score: float = 0.0,
                              is_high: bool = True) -> bool:
        """Add a performance anomaly record to the database.
        
        Args:
            worker_id: ID of the worker that detected the anomaly (or None for task type)
            entity_type: Type of entity (worker or task_type)
            entity_id: ID of the entity (required for task_type)
            metric: Name of the metric that showed anomalous behavior
            timestamp: Timestamp when the anomaly occurred
            value: The anomalous value
            baseline_mean: Mean of the baseline for this metric
            baseline_stdev: Standard deviation of the baseline
            z_score: Z-score of the anomalous value
            is_high: Whether the anomaly is a high value (True) or low value (False)
            
        Returns:
            True if successful, False otherwise
        """
        # Use current timestamp if not provided
        if timestamp is None:
            timestamp = datetime.now()
            
        # For task_type anomalies, entity_id is required
        if entity_type == "task_type" and entity_id is None:
            logger.error("Entity ID is required for task_type anomalies")
            return False
            
        # Prepare entity ID for worker anomalies
        if entity_type == "worker" and worker_id is not None:
            entity_id = worker_id
            
        # Create the anomaly record
        try:
            cursor = self.conn.cursor()
            
            query = """
                INSERT INTO performance_anomalies (
                    entity_type,
                    entity_id,
                    metric,
                    detected_at,
                    anomaly_timestamp,
                    value,
                    baseline_mean,
                    baseline_stdev,
                    z_score,
                    is_high
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            cursor.execute(query, {
                "entity_type": entity_type,
                "entity_id": entity_id,
                "metric": metric,
                "detected_at": datetime.now(),
                "anomaly_timestamp": timestamp,
                "value": value,
                "baseline_mean": baseline_mean,
                "baseline_stdev": baseline_stdev,
                "z_score": z_score,
                "is_high": is_high
            })
            
            self.conn.commit()
            logger.debug(f"Added performance anomaly record for {entity_type} {entity_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding performance anomaly: {e}")
            return False
    
    def add_performance_trend(self,
                            worker_id: Optional[str],
                            entity_type: str,
                            entity_id: Optional[str] = None,
                            metric: str = "",
                            slope: float = 0.0,
                            p_value: float = 0.0,
                            r_squared: float = 0.0,
                            is_significant: bool = False,
                            direction: str = "increasing",
                            forecast_values: List[float] = None) -> bool:
        """Add a performance trend record to the database.
        
        Args:
            worker_id: ID of the worker that detected the trend (or None for task type)
            entity_type: Type of entity (worker or task_type)
            entity_id: ID of the entity (required for task_type)
            metric: Name of the metric that showed a trend
            slope: Slope of the trend line
            p_value: P-value of the trend (statistical significance)
            r_squared: R-squared value of the trend (goodness of fit)
            is_significant: Whether the trend is statistically significant
            direction: Direction of the trend (increasing or decreasing)
            forecast_values: List of forecasted values
            
        Returns:
            True if successful, False otherwise
        """
        # For task_type trends, entity_id is required
        if entity_type == "task_type" and entity_id is None:
            logger.error("Entity ID is required for task_type trends")
            return False
            
        # Prepare entity ID for worker trends
        if entity_type == "worker" and worker_id is not None:
            entity_id = worker_id
            
        # Convert forecast values to JSON string
        if forecast_values is None:
            forecast_values = []
            
        forecast_json = self.dict_to_db_json(forecast_values)
        
        # Create the trend record
        try:
            cursor = self.conn.cursor()
            
            query = """
                INSERT INTO performance_trends (
                    entity_type,
                    entity_id,
                    metric,
                    detected_at,
                    slope,
                    p_value,
                    r_squared,
                    is_significant,
                    direction,
                    forecast_values
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            cursor.execute(query, {
                "entity_type": entity_type,
                "entity_id": entity_id,
                "metric": metric,
                "detected_at": datetime.now(),
                "slope": slope,
                "p_value": p_value,
                "r_squared": r_squared,
                "is_significant": is_significant,
                "direction": direction,
                "forecast_values": forecast_json
            })
            
            self.conn.commit()
            logger.debug(f"Added performance trend record for {entity_type} {entity_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding performance trend: {e}")
            return False
    
    # Add the methods to the class
    cls.get_performance_results = get_performance_results
    cls.get_compatibility_results = get_compatibility_results
    cls.get_integration_test_results = get_integration_test_results
    cls.get_web_platform_results = get_web_platform_results
    cls.get_hardware_info = get_hardware_info
    cls.get_model_info = get_model_info
    cls.add_performance_anomaly = add_performance_anomaly
    cls.add_performance_trend = add_performance_trend
    
    return cls