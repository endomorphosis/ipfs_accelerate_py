#!/usr/bin/env python3
"""
Multi-Model Empirical Validation for Predictive Performance System.

This module provides empirical validation capabilities for the Multi-Model Execution
predictions by comparing them with actual measurements from the Web Resource Pool.
It collects validation metrics, calculates error rates, and enables continuous
refinement of prediction models based on real-world data.

Key features:
1. Validation metrics collection and analysis
2. Error rate calculation and trend analysis
3. Prediction model refinement based on empirical data
4. Validation dataset management for continuous improvement
5. Visualization of prediction accuracy over time
"""

import os
import sys
import time
import json
import logging
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("predictive_performance.multi_model_empirical_validation")

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


class MultiModelEmpiricalValidator:
    """
    Empirical validator for Multi-Model Execution predictions.
    
    This class handles the collection, analysis, and management of validation metrics
    for multi-model execution predictions compared to actual measurements.
    It enables continuous refinement of prediction models based on empirical data.
    """
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        validation_history_size: int = 100,
        error_threshold: float = 0.15,
        refinement_interval: int = 10,
        enable_trend_analysis: bool = True,
        enable_visualization: bool = True,
        verbose: bool = False
    ):
        """
        Initialize the empirical validator.
        
        Args:
            db_path: Path to database for storing validation metrics
            validation_history_size: Maximum number of validation records to keep in memory
            error_threshold: Threshold for acceptable prediction error (15% by default)
            refinement_interval: Number of validations between model refinements
            enable_trend_analysis: Whether to analyze error trends over time
            enable_visualization: Whether to enable visualization of validation metrics
            verbose: Whether to enable verbose logging
        """
        self.db_path = db_path
        self.validation_history_size = validation_history_size
        self.error_threshold = error_threshold
        self.refinement_interval = refinement_interval
        self.enable_trend_analysis = enable_trend_analysis
        self.enable_visualization = enable_visualization
        
        # Set logging level
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        # Initialize validation metrics storage
        self.validation_metrics = {
            "records": [],
            "execution_count": 0,
            "validation_count": 0,
            "last_validation_time": 0,
            "refinement_count": 0,
            "error_rates": {
                "throughput": [],
                "latency": [],
                "memory": []
            },
            "error_trends": {
                "throughput": [],
                "latency": [],
                "memory": []
            },
            "hardware_specific": {},
            "strategy_specific": {}
        }
        
        # Initialize database connection
        self.db_conn = None
        if self.db_path:
            try:
                import duckdb
                self.db_conn = duckdb.connect(self.db_path)
                self._initialize_database_tables()
                logger.info(f"Database connection established to {self.db_path}")
            except ImportError:
                logger.warning("duckdb not available, will operate without database storage")
            except Exception as e:
                logger.error(f"Error connecting to database: {e}")
                traceback.print_exc()
        
        logger.info(f"MultiModelEmpiricalValidator initialized with "
                   f"history_size={validation_history_size}, "
                   f"error_threshold={error_threshold}, "
                   f"refinement_interval={refinement_interval}")
    
    def _initialize_database_tables(self):
        """Initialize database tables for storing validation metrics."""
        if not self.db_conn:
            return
        
        try:
            # Create validation metrics table
            self.db_conn.execute("""
            CREATE TABLE IF NOT EXISTS multi_model_validation_metrics (
                id INTEGER PRIMARY KEY,
                timestamp TIMESTAMP,
                model_count INTEGER,
                hardware_platform VARCHAR,
                execution_strategy VARCHAR,
                predicted_throughput DOUBLE,
                actual_throughput DOUBLE,
                predicted_latency DOUBLE,
                actual_latency DOUBLE,
                predicted_memory DOUBLE,
                actual_memory DOUBLE,
                throughput_error_rate DOUBLE,
                latency_error_rate DOUBLE,
                memory_error_rate DOUBLE,
                model_configs VARCHAR,
                optimization_goal VARCHAR
            )
            """)
            
            # Create model refinement metrics table
            self.db_conn.execute("""
            CREATE TABLE IF NOT EXISTS multi_model_refinement_metrics (
                id INTEGER PRIMARY KEY,
                timestamp TIMESTAMP,
                refinement_count INTEGER,
                validation_count INTEGER,
                pre_refinement_throughput_error DOUBLE,
                post_refinement_throughput_error DOUBLE,
                pre_refinement_latency_error DOUBLE,
                post_refinement_latency_error DOUBLE,
                pre_refinement_memory_error DOUBLE,
                post_refinement_memory_error DOUBLE,
                improvement_percent DOUBLE,
                refinement_method VARCHAR
            )
            """)
            
            # Create error trend table
            self.db_conn.execute("""
            CREATE TABLE IF NOT EXISTS multi_model_error_trends (
                id INTEGER PRIMARY KEY,
                timestamp TIMESTAMP,
                metric VARCHAR,
                error_rate_avg_10 DOUBLE,
                error_rate_avg_20 DOUBLE,
                error_rate_avg_50 DOUBLE,
                trend_direction VARCHAR,
                trend_strength DOUBLE
            )
            """)
            
            logger.info("Database tables initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database tables: {e}")
            traceback.print_exc()
    
    def validate_prediction(
        self,
        prediction: Dict[str, Any],
        actual_measurement: Dict[str, Any],
        model_configs: List[Dict[str, Any]],
        hardware_platform: str,
        execution_strategy: str,
        optimization_goal: str = "latency"
    ) -> Dict[str, Any]:
        """
        Validate a prediction against actual measurement.
        
        Args:
            prediction: Dictionary with predicted performance metrics
            actual_measurement: Dictionary with actual measured performance metrics
            model_configs: List of model configurations used for prediction
            hardware_platform: Hardware platform used for execution
            execution_strategy: Execution strategy used (parallel, sequential, batched)
            optimization_goal: Optimization goal (latency, throughput, memory)
            
        Returns:
            Dictionary with validation metrics and error rates
        """
        # Increment execution count
        self.validation_metrics["execution_count"] += 1
        
        # Extract metrics from prediction and actual measurement
        try:
            # Extract predicted metrics
            predicted_metrics = prediction.get("total_metrics", {})
            predicted_throughput = predicted_metrics.get("combined_throughput", 0)
            predicted_latency = predicted_metrics.get("combined_latency", 0)
            predicted_memory = predicted_metrics.get("combined_memory", 0)
            
            # Extract actual metrics
            actual_throughput = actual_measurement.get("actual_throughput", 0)
            actual_latency = actual_measurement.get("actual_latency", 0)
            actual_memory = actual_measurement.get("actual_memory", 0)
            
            # Ensure we have valid values
            if predicted_throughput <= 0 or actual_throughput <= 0:
                logger.warning("Invalid throughput values detected")
                predicted_throughput = max(0.001, predicted_throughput)
                actual_throughput = max(0.001, actual_throughput)
            
            if predicted_latency <= 0 or actual_latency <= 0:
                logger.warning("Invalid latency values detected")
                predicted_latency = max(0.001, predicted_latency)
                actual_latency = max(0.001, actual_latency)
            
            if predicted_memory <= 0 or actual_memory <= 0:
                logger.warning("Invalid memory values detected")
                predicted_memory = max(0.001, predicted_memory)
                actual_memory = max(0.001, actual_memory)
            
            # Calculate error rates
            throughput_error = abs(predicted_throughput - actual_throughput) / predicted_throughput
            latency_error = abs(predicted_latency - actual_latency) / predicted_latency
            memory_error = abs(predicted_memory - actual_memory) / predicted_memory
            
        except Exception as e:
            logger.error(f"Error calculating validation metrics: {e}")
            throughput_error = 0.0
            latency_error = 0.0
            memory_error = 0.0
            traceback.print_exc()
        
        # Create validation record
        validation_record = {
            "timestamp": time.time(),
            "model_count": len(model_configs),
            "hardware_platform": hardware_platform,
            "execution_strategy": execution_strategy,
            "predicted_throughput": predicted_throughput,
            "actual_throughput": actual_throughput,
            "predicted_latency": predicted_latency,
            "actual_latency": actual_latency,
            "predicted_memory": predicted_memory,
            "actual_memory": actual_memory,
            "throughput_error": throughput_error,
            "latency_error": latency_error,
            "memory_error": memory_error,
            "optimization_goal": optimization_goal,
            "model_types": [config.get("model_type", "") for config in model_configs]
        }
        
        # Store validation record
        self._store_validation_record(validation_record)
        
        # Calculate validation metrics
        validation_metrics = self._calculate_validation_metrics(validation_record)
        
        return validation_metrics
    
    def _store_validation_record(self, validation_record: Dict[str, Any]):
        """
        Store a validation record in memory and database.
        
        Args:
            validation_record: Dictionary with validation metrics
        """
        # Store in memory
        self.validation_metrics["records"].append(validation_record)
        self.validation_metrics["validation_count"] += 1
        self.validation_metrics["last_validation_time"] = validation_record["timestamp"]
        
        # Limit history size
        if len(self.validation_metrics["records"]) > self.validation_history_size:
            self.validation_metrics["records"] = self.validation_metrics["records"][-self.validation_history_size:]
        
        # Update error rates
        self.validation_metrics["error_rates"]["throughput"].append(validation_record["throughput_error"])
        self.validation_metrics["error_rates"]["latency"].append(validation_record["latency_error"])
        self.validation_metrics["error_rates"]["memory"].append(validation_record["memory_error"])
        
        # Update hardware-specific metrics
        hardware_platform = validation_record["hardware_platform"]
        if hardware_platform not in self.validation_metrics["hardware_specific"]:
            self.validation_metrics["hardware_specific"][hardware_platform] = {
                "count": 0,
                "throughput_errors": [],
                "latency_errors": [],
                "memory_errors": []
            }
        
        hw_metrics = self.validation_metrics["hardware_specific"][hardware_platform]
        hw_metrics["count"] += 1
        hw_metrics["throughput_errors"].append(validation_record["throughput_error"])
        hw_metrics["latency_errors"].append(validation_record["latency_error"])
        hw_metrics["memory_errors"].append(validation_record["memory_error"])
        
        # Update strategy-specific metrics
        execution_strategy = validation_record["execution_strategy"]
        if execution_strategy not in self.validation_metrics["strategy_specific"]:
            self.validation_metrics["strategy_specific"][execution_strategy] = {
                "count": 0,
                "throughput_errors": [],
                "latency_errors": [],
                "memory_errors": []
            }
        
        strategy_metrics = self.validation_metrics["strategy_specific"][execution_strategy]
        strategy_metrics["count"] += 1
        strategy_metrics["throughput_errors"].append(validation_record["throughput_error"])
        strategy_metrics["latency_errors"].append(validation_record["latency_error"])
        strategy_metrics["memory_errors"].append(validation_record["memory_error"])
        
        # Store in database if available
        if self.db_conn:
            try:
                self.db_conn.execute(
                    """
                    INSERT INTO multi_model_validation_metrics 
                    (timestamp, model_count, hardware_platform, execution_strategy, 
                     predicted_throughput, actual_throughput, predicted_latency, actual_latency,
                     predicted_memory, actual_memory, throughput_error_rate, latency_error_rate,
                     memory_error_rate, model_configs, optimization_goal)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        validation_record["timestamp"],
                        validation_record["model_count"],
                        validation_record["hardware_platform"],
                        validation_record["execution_strategy"],
                        validation_record["predicted_throughput"],
                        validation_record["actual_throughput"],
                        validation_record["predicted_latency"],
                        validation_record["actual_latency"],
                        validation_record["predicted_memory"],
                        validation_record["actual_memory"],
                        validation_record["throughput_error"],
                        validation_record["latency_error"],
                        validation_record["memory_error"],
                        json.dumps(validation_record["model_types"]),
                        validation_record["optimization_goal"]
                    )
                )
            except Exception as e:
                logger.error(f"Error storing validation record in database: {e}")
    
    def _calculate_validation_metrics(self, validation_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate validation metrics based on the latest validation record.
        
        Args:
            validation_record: Latest validation record
            
        Returns:
            Dictionary with validation metrics
        """
        # Calculate average error rates
        avg_throughput_error = np.mean(self.validation_metrics["error_rates"]["throughput"][-10:]) if self.validation_metrics["error_rates"]["throughput"] else 0
        avg_latency_error = np.mean(self.validation_metrics["error_rates"]["latency"][-10:]) if self.validation_metrics["error_rates"]["latency"] else 0
        avg_memory_error = np.mean(self.validation_metrics["error_rates"]["memory"][-10:]) if self.validation_metrics["error_rates"]["memory"] else 0
        
        # Calculate error trends if enabled and have enough data
        trend_metrics = {}
        if self.enable_trend_analysis and len(self.validation_metrics["error_rates"]["throughput"]) >= 20:
            trend_metrics = self._calculate_error_trends()
        
        # Calculate hardware-specific metrics
        hardware_platform = validation_record["hardware_platform"]
        hw_metrics = {}
        if hardware_platform in self.validation_metrics["hardware_specific"]:
            hw_data = self.validation_metrics["hardware_specific"][hardware_platform]
            hw_metrics = {
                "avg_throughput_error": np.mean(hw_data["throughput_errors"][-10:]) if hw_data["throughput_errors"] else 0,
                "avg_latency_error": np.mean(hw_data["latency_errors"][-10:]) if hw_data["latency_errors"] else 0,
                "avg_memory_error": np.mean(hw_data["memory_errors"][-10:]) if hw_data["memory_errors"] else 0,
                "count": hw_data["count"]
            }
        
        # Calculate strategy-specific metrics
        execution_strategy = validation_record["execution_strategy"]
        strategy_metrics = {}
        if execution_strategy in self.validation_metrics["strategy_specific"]:
            strategy_data = self.validation_metrics["strategy_specific"][execution_strategy]
            strategy_metrics = {
                "avg_throughput_error": np.mean(strategy_data["throughput_errors"][-10:]) if strategy_data["throughput_errors"] else 0,
                "avg_latency_error": np.mean(strategy_data["latency_errors"][-10:]) if strategy_data["latency_errors"] else 0,
                "avg_memory_error": np.mean(strategy_data["memory_errors"][-10:]) if strategy_data["memory_errors"] else 0,
                "count": strategy_data["count"]
            }
        
        # Determine if model refinement is needed
        needs_refinement = self._check_if_refinement_needed()
        
        # Create validation metrics
        metrics = {
            "validation_count": self.validation_metrics["validation_count"],
            "current_errors": {
                "throughput": validation_record["throughput_error"],
                "latency": validation_record["latency_error"],
                "memory": validation_record["memory_error"]
            },
            "average_errors": {
                "throughput": avg_throughput_error,
                "latency": avg_latency_error,
                "memory": avg_memory_error
            },
            "hardware_metrics": hw_metrics,
            "strategy_metrics": strategy_metrics,
            "needs_refinement": needs_refinement,
            "timestamp": validation_record["timestamp"]
        }
        
        # Add trend metrics if available
        if trend_metrics:
            metrics["error_trends"] = trend_metrics
        
        return metrics
    
    def _calculate_error_trends(self) -> Dict[str, Any]:
        """
        Calculate error trends over time.
        
        Returns:
            Dictionary with error trend metrics
        """
        trend_metrics = {}
        
        for metric_name in ["throughput", "latency", "memory"]:
            error_values = self.validation_metrics["error_rates"][metric_name]
            
            if len(error_values) < 20:
                continue
            
            # Calculate moving averages for different window sizes
            avg_10 = np.mean(error_values[-10:])
            avg_20 = np.mean(error_values[-20:])
            
            # Calculate longer-term average if available
            avg_50 = np.mean(error_values[-50:]) if len(error_values) >= 50 else avg_20
            
            # Determine trend direction and strength
            # Negative trend (improving) if recent average is lower than older average
            trend_direction = "improving" if avg_10 < avg_20 else "worsening"
            
            # Calculate trend strength as percentage change
            if avg_20 > 0:
                trend_strength = abs(avg_10 - avg_20) / avg_20
            else:
                trend_strength = 0.0
            
            # Store trend metrics
            trend_metrics[metric_name] = {
                "avg_10": avg_10,
                "avg_20": avg_20,
                "avg_50": avg_50,
                "direction": trend_direction,
                "strength": trend_strength
            }
            
            # Store in validation metrics history
            self.validation_metrics["error_trends"][metric_name].append({
                "timestamp": time.time(),
                "avg_10": avg_10,
                "avg_20": avg_20,
                "avg_50": avg_50,
                "direction": trend_direction,
                "strength": trend_strength
            })
            
            # Store in database if available
            if self.db_conn:
                try:
                    self.db_conn.execute(
                        """
                        INSERT INTO multi_model_error_trends
                        (timestamp, metric, error_rate_avg_10, error_rate_avg_20, error_rate_avg_50, 
                         trend_direction, trend_strength)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            time.time(),
                            metric_name,
                            avg_10,
                            avg_20,
                            avg_50,
                            trend_direction,
                            trend_strength
                        )
                    )
                except Exception as e:
                    logger.error(f"Error storing error trend in database: {e}")
        
        return trend_metrics
    
    def _check_if_refinement_needed(self) -> bool:
        """
        Check if model refinement is needed based on validation metrics.
        
        Returns:
            True if refinement is needed, False otherwise
        """
        # Check if we have enough validation records
        if self.validation_metrics["validation_count"] < self.refinement_interval:
            return False
        
        # Check if it's time for refinement based on interval
        if self.validation_metrics["validation_count"] % self.refinement_interval != 0:
            return False
        
        # Check if error rates exceed threshold
        recent_throughput_errors = self.validation_metrics["error_rates"]["throughput"][-self.refinement_interval:]
        recent_latency_errors = self.validation_metrics["error_rates"]["latency"][-self.refinement_interval:]
        recent_memory_errors = self.validation_metrics["error_rates"]["memory"][-self.refinement_interval:]
        
        avg_throughput_error = np.mean(recent_throughput_errors) if recent_throughput_errors else 0
        avg_latency_error = np.mean(recent_latency_errors) if recent_latency_errors else 0
        avg_memory_error = np.mean(recent_memory_errors) if recent_memory_errors else 0
        
        # If any error rate exceeds threshold, refinement is needed
        if (avg_throughput_error > self.error_threshold or
            avg_latency_error > self.error_threshold or
            avg_memory_error > self.error_threshold):
            return True
        
        # Check for worsening trends if trend analysis is enabled
        if self.enable_trend_analysis and len(self.validation_metrics["error_trends"]["throughput"]) > 0:
            for metric_name in ["throughput", "latency", "memory"]:
                trends = self.validation_metrics["error_trends"][metric_name]
                if not trends:
                    continue
                
                latest_trend = trends[-1]
                if latest_trend["direction"] == "worsening" and latest_trend["strength"] > 0.1:
                    # If a significant worsening trend is detected, refinement is needed
                    return True
        
        return False
    
    def record_model_refinement(
        self,
        pre_refinement_errors: Dict[str, float],
        post_refinement_errors: Dict[str, float],
        refinement_method: str
    ) -> Dict[str, Any]:
        """
        Record metrics for a model refinement.
        
        Args:
            pre_refinement_errors: Error rates before refinement
            post_refinement_errors: Error rates after refinement
            refinement_method: Method used for refinement (incremental, window, weighted)
            
        Returns:
            Dictionary with refinement metrics
        """
        self.validation_metrics["refinement_count"] += 1
        
        # Calculate improvement percentages
        throughput_improvement = (pre_refinement_errors.get("throughput", 0) - post_refinement_errors.get("throughput", 0)) / pre_refinement_errors.get("throughput", 1) * 100
        latency_improvement = (pre_refinement_errors.get("latency", 0) - post_refinement_errors.get("latency", 0)) / pre_refinement_errors.get("latency", 1) * 100
        memory_improvement = (pre_refinement_errors.get("memory", 0) - post_refinement_errors.get("memory", 0)) / pre_refinement_errors.get("memory", 1) * 100
        
        # Calculate overall improvement
        overall_improvement = (throughput_improvement + latency_improvement + memory_improvement) / 3
        
        # Create refinement record
        refinement_record = {
            "timestamp": time.time(),
            "refinement_count": self.validation_metrics["refinement_count"],
            "validation_count": self.validation_metrics["validation_count"],
            "pre_refinement_errors": pre_refinement_errors,
            "post_refinement_errors": post_refinement_errors,
            "throughput_improvement": throughput_improvement,
            "latency_improvement": latency_improvement,
            "memory_improvement": memory_improvement,
            "overall_improvement": overall_improvement,
            "refinement_method": refinement_method
        }
        
        # Store in database if available
        if self.db_conn:
            try:
                self.db_conn.execute(
                    """
                    INSERT INTO multi_model_refinement_metrics
                    (timestamp, refinement_count, validation_count,
                     pre_refinement_throughput_error, post_refinement_throughput_error,
                     pre_refinement_latency_error, post_refinement_latency_error,
                     pre_refinement_memory_error, post_refinement_memory_error,
                     improvement_percent, refinement_method)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        refinement_record["timestamp"],
                        refinement_record["refinement_count"],
                        refinement_record["validation_count"],
                        pre_refinement_errors.get("throughput", 0),
                        post_refinement_errors.get("throughput", 0),
                        pre_refinement_errors.get("latency", 0),
                        post_refinement_errors.get("latency", 0),
                        pre_refinement_errors.get("memory", 0),
                        post_refinement_errors.get("memory", 0),
                        overall_improvement,
                        refinement_method
                    )
                )
            except Exception as e:
                logger.error(f"Error storing refinement record in database: {e}")
        
        return refinement_record
    
    def get_validation_metrics(self, include_history: bool = False) -> Dict[str, Any]:
        """
        Get comprehensive validation metrics.
        
        Args:
            include_history: Whether to include full validation history
            
        Returns:
            Dictionary with validation metrics
        """
        # Calculate overall metrics
        metrics = {
            "validation_count": self.validation_metrics["validation_count"],
            "execution_count": self.validation_metrics["execution_count"],
            "refinement_count": self.validation_metrics["refinement_count"],
            "last_validation_time": self.validation_metrics["last_validation_time"]
        }
        
        # Calculate average error rates
        error_rates = {}
        for metric_name in ["throughput", "latency", "memory"]:
            values = self.validation_metrics["error_rates"][metric_name]
            if not values:
                continue
                
            # Calculate average error rate
            avg_error = np.mean(values)
            error_rates[f"avg_{metric_name}_error"] = avg_error
            
            # Calculate recent error rate (last 10 validations)
            recent_values = values[-10:] if len(values) >= 10 else values
            recent_error = np.mean(recent_values)
            error_rates[f"recent_{metric_name}_error"] = recent_error
            
            # Calculate error trend
            if len(values) >= 20:
                older_values = values[-20:-10]
                older_avg = np.mean(older_values)
                trend = recent_error - older_avg
                error_rates[f"{metric_name}_error_trend"] = trend
                error_rates[f"{metric_name}_error_trend_direction"] = "improving" if trend < 0 else "worsening"
        
        metrics["error_rates"] = error_rates
        
        # Add hardware-specific metrics
        hardware_metrics = {}
        for platform, data in self.validation_metrics["hardware_specific"].items():
            hardware_metrics[platform] = {
                "count": data["count"],
                "avg_throughput_error": np.mean(data["throughput_errors"][-10:]) if data["throughput_errors"] else 0,
                "avg_latency_error": np.mean(data["latency_errors"][-10:]) if data["latency_errors"] else 0,
                "avg_memory_error": np.mean(data["memory_errors"][-10:]) if data["memory_errors"] else 0
            }
        
        metrics["hardware_metrics"] = hardware_metrics
        
        # Add strategy-specific metrics
        strategy_metrics = {}
        for strategy, data in self.validation_metrics["strategy_specific"].items():
            strategy_metrics[strategy] = {
                "count": data["count"],
                "avg_throughput_error": np.mean(data["throughput_errors"][-10:]) if data["throughput_errors"] else 0,
                "avg_latency_error": np.mean(data["latency_errors"][-10:]) if data["latency_errors"] else 0,
                "avg_memory_error": np.mean(data["memory_errors"][-10:]) if data["memory_errors"] else 0
            }
        
        metrics["strategy_metrics"] = strategy_metrics
        
        # Add validation history if requested
        if include_history:
            metrics["history"] = self.validation_metrics["records"]
        
        # Add database statistics if available
        if self.db_conn:
            try:
                # Get validation count from database
                db_validation_count = self.db_conn.execute(
                    "SELECT COUNT(*) FROM multi_model_validation_metrics"
                ).fetchone()[0]
                
                # Get average error rates from database
                db_error_rates = self.db_conn.execute(
                    """
                    SELECT 
                        AVG(throughput_error_rate) as avg_throughput_error,
                        AVG(latency_error_rate) as avg_latency_error,
                        AVG(memory_error_rate) as avg_memory_error
                    FROM multi_model_validation_metrics
                    """
                ).fetchone()
                
                # Get refinement count from database
                db_refinement_count = self.db_conn.execute(
                    "SELECT COUNT(*) FROM multi_model_refinement_metrics"
                ).fetchone()[0]
                
                # Get average improvement from refinements
                db_improvement = self.db_conn.execute(
                    """
                    SELECT 
                        AVG(improvement_percent) as avg_improvement
                    FROM multi_model_refinement_metrics
                    """
                ).fetchone()[0]
                
                metrics["database"] = {
                    "validation_count": db_validation_count,
                    "refinement_count": db_refinement_count,
                    "avg_throughput_error": db_error_rates[0],
                    "avg_latency_error": db_error_rates[1],
                    "avg_memory_error": db_error_rates[2],
                    "avg_refinement_improvement": db_improvement
                }
            except Exception as e:
                logger.error(f"Error getting database statistics: {e}")
        
        return metrics
    
    def get_refinement_recommendations(self) -> Dict[str, Any]:
        """
        Get recommendations for model refinement based on validation metrics.
        
        Returns:
            Dictionary with refinement recommendations
        """
        # Check if we have enough validation data
        if self.validation_metrics["validation_count"] < 10:
            return {
                "refinement_needed": False,
                "reason": "Insufficient validation data",
                "recommendation": "Collect more validation data before refinement"
            }
        
        # Calculate average error rates
        avg_throughput_error = np.mean(self.validation_metrics["error_rates"]["throughput"][-10:]) if self.validation_metrics["error_rates"]["throughput"] else 0
        avg_latency_error = np.mean(self.validation_metrics["error_rates"]["latency"][-10:]) if self.validation_metrics["error_rates"]["latency"] else 0
        avg_memory_error = np.mean(self.validation_metrics["error_rates"]["memory"][-10:]) if self.validation_metrics["error_rates"]["memory"] else 0
        
        # Determine if refinement is needed
        refinement_needed = False
        reason = ""
        if avg_throughput_error > self.error_threshold:
            refinement_needed = True
            reason += f"Throughput error ({avg_throughput_error:.2%}) exceeds threshold ({self.error_threshold:.2%}). "
        
        if avg_latency_error > self.error_threshold:
            refinement_needed = True
            reason += f"Latency error ({avg_latency_error:.2%}) exceeds threshold ({self.error_threshold:.2%}). "
        
        if avg_memory_error > self.error_threshold:
            refinement_needed = True
            reason += f"Memory error ({avg_memory_error:.2%}) exceeds threshold ({self.error_threshold:.2%}). "
        
        # Check for worsening trends
        if self.enable_trend_analysis:
            for metric_name in ["throughput", "latency", "memory"]:
                if len(self.validation_metrics["error_rates"][metric_name]) < 20:
                    continue
                
                recent_avg = np.mean(self.validation_metrics["error_rates"][metric_name][-10:])
                older_avg = np.mean(self.validation_metrics["error_rates"][metric_name][-20:-10])
                
                if recent_avg > older_avg and (recent_avg - older_avg) / older_avg > 0.1:
                    refinement_needed = True
                    reason += f"{metric_name.capitalize()} error is trending worse (increased by {((recent_avg - older_avg) / older_avg):.2%}). "
        
        # Determine recommended refinement method
        recommended_method = "incremental"
        if refinement_needed:
            # Check error patterns to suggest appropriate method
            if avg_throughput_error > 0.3 or avg_latency_error > 0.3 or avg_memory_error > 0.3:
                # High error rates might require more significant update
                recommended_method = "window"
            elif self.enable_trend_analysis:
                # Check if errors are consistently worsening
                consistent_worsening = True
                for metric_name in ["throughput", "latency", "memory"]:
                    if len(self.validation_metrics["error_rates"][metric_name]) < 20:
                        consistent_worsening = False
                        break
                    
                    recent_avg = np.mean(self.validation_metrics["error_rates"][metric_name][-10:])
                    older_avg = np.mean(self.validation_metrics["error_rates"][metric_name][-20:-10])
                    
                    if recent_avg <= older_avg:
                        consistent_worsening = False
                        break
                
                if consistent_worsening:
                    # Consistent worsening might require weighted update
                    recommended_method = "weighted"
        
        # Create recommendation
        recommendation = {
            "refinement_needed": refinement_needed,
            "reason": reason.strip() if reason else "Error rates within acceptable range",
            "recommended_method": recommended_method if refinement_needed else None,
            "error_rates": {
                "throughput": avg_throughput_error,
                "latency": avg_latency_error,
                "memory": avg_memory_error
            },
            "threshold": self.error_threshold
        }
        
        # Add hardware-specific recommendations
        hardware_recommendations = {}
        for platform, metrics in self.validation_metrics["hardware_specific"].items():
            if len(metrics["throughput_errors"]) < 5:
                continue
                
            avg_throughput = np.mean(metrics["throughput_errors"][-5:])
            avg_latency = np.mean(metrics["latency_errors"][-5:])
            avg_memory = np.mean(metrics["memory_errors"][-5:])
            
            needs_refinement = (avg_throughput > self.error_threshold or
                               avg_latency > self.error_threshold or
                               avg_memory > self.error_threshold)
            
            hardware_recommendations[platform] = {
                "refinement_needed": needs_refinement,
                "error_rates": {
                    "throughput": avg_throughput,
                    "latency": avg_latency,
                    "memory": avg_memory
                }
            }
        
        recommendation["hardware_recommendations"] = hardware_recommendations
        
        # Add strategy-specific recommendations
        strategy_recommendations = {}
        for strategy, metrics in self.validation_metrics["strategy_specific"].items():
            if len(metrics["throughput_errors"]) < 5:
                continue
                
            avg_throughput = np.mean(metrics["throughput_errors"][-5:])
            avg_latency = np.mean(metrics["latency_errors"][-5:])
            avg_memory = np.mean(metrics["memory_errors"][-5:])
            
            needs_refinement = (avg_throughput > self.error_threshold or
                               avg_latency > self.error_threshold or
                               avg_memory > self.error_threshold)
            
            strategy_recommendations[strategy] = {
                "refinement_needed": needs_refinement,
                "error_rates": {
                    "throughput": avg_throughput,
                    "latency": avg_latency,
                    "memory": avg_memory
                }
            }
        
        recommendation["strategy_recommendations"] = strategy_recommendations
        
        return recommendation
    
    def generate_validation_dataset(self) -> Dict[str, Any]:
        """
        Generate a validation dataset for model refinement.
        
        Returns:
            Dictionary with validation dataset
        """
        # Check if we have enough validation records
        if len(self.validation_metrics["records"]) < 10:
            logger.warning("Insufficient validation records for dataset generation")
            return {
                "success": False,
                "reason": "Insufficient validation records",
                "min_required": 10,
                "available": len(self.validation_metrics["records"])
            }
        
        try:
            # Create dataset from validation records
            records = []
            for record in self.validation_metrics["records"]:
                dataset_record = {
                    "model_count": record["model_count"],
                    "hardware_platform": record["hardware_platform"],
                    "execution_strategy": record["execution_strategy"],
                    "model_types": record.get("model_types", []),
                    "actual_throughput": record["actual_throughput"],
                    "actual_latency": record["actual_latency"],
                    "actual_memory": record["actual_memory"]
                }
                records.append(dataset_record)
            
            # Create Pandas DataFrame if available
            try:
                import pandas as pd
                df = pd.DataFrame(records)
                return {
                    "success": True,
                    "records": records,
                    "dataframe": df,
                    "record_count": len(records)
                }
            except ImportError:
                # Return just the records if pandas is not available
                return {
                    "success": True,
                    "records": records,
                    "record_count": len(records)
                }
                
        except Exception as e:
            logger.error(f"Error generating validation dataset: {e}")
            traceback.print_exc()
            return {
                "success": False,
                "reason": f"Error generating dataset: {str(e)}"
            }
    
    def visualize_validation_metrics(self, metric_type: str = "error_rates") -> Dict[str, Any]:
        """
        Visualize validation metrics.
        
        Args:
            metric_type: Type of metrics to visualize (error_rates, trends, hardware, strategy)
            
        Returns:
            Dictionary with visualization data
        """
        if not self.enable_visualization:
            return {
                "success": False,
                "reason": "Visualization is disabled"
            }
        
        try:
            # Attempt to use matplotlib if available
            import matplotlib.pyplot as plt
            
            # Create a figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if metric_type == "error_rates":
                # Plot error rates over time
                for metric_name in ["throughput", "latency", "memory"]:
                    values = self.validation_metrics["error_rates"][metric_name]
                    if not values:
                        continue
                    
                    ax.plot(range(len(values)), values, label=f"{metric_name.capitalize()} Error")
                
                ax.set_xlabel("Validation Count")
                ax.set_ylabel("Error Rate")
                ax.set_title("Prediction Error Rates Over Time")
                ax.legend()
                ax.grid(True)
                
                # Add threshold line
                ax.axhline(y=self.error_threshold, color='r', linestyle='--', label=f"Error Threshold ({self.error_threshold:.2f})")
                
            elif metric_type == "trends":
                # Plot error trends over time
                for metric_name in ["throughput", "latency", "memory"]:
                    trends = self.validation_metrics["error_trends"][metric_name]
                    if not trends:
                        continue
                    
                    # Extract trend data
                    timestamps = [t["timestamp"] for t in trends]
                    avg_10 = [t["avg_10"] for t in trends]
                    avg_20 = [t["avg_20"] for t in trends]
                    
                    # Plot trend lines
                    ax.plot(range(len(trends)), avg_10, label=f"{metric_name.capitalize()} (10-point avg)")
                    ax.plot(range(len(trends)), avg_20, linestyle='--', label=f"{metric_name.capitalize()} (20-point avg)")
                
                ax.set_xlabel("Trend Analysis Count")
                ax.set_ylabel("Error Rate")
                ax.set_title("Error Rate Trends Over Time")
                ax.legend()
                ax.grid(True)
                
            elif metric_type == "hardware":
                # Plot hardware-specific error rates
                platforms = list(self.validation_metrics["hardware_specific"].keys())
                metrics = ["throughput", "latency", "memory"]
                
                # If too many platforms, limit to top 5 by count
                if len(platforms) > 5:
                    platforms_by_count = sorted(platforms, 
                                               key=lambda p: self.validation_metrics["hardware_specific"][p]["count"],
                                               reverse=True)
                    platforms = platforms_by_count[:5]
                
                # Set up bar positions
                x = np.arange(len(platforms))
                width = 0.25
                
                # Plot bars for each metric
                for i, metric in enumerate(metrics):
                    error_values = [np.mean(self.validation_metrics["hardware_specific"][p][f"{metric}_errors"]) 
                                    for p in platforms]
                    ax.bar(x + i*width, error_values, width, label=f"{metric.capitalize()} Error")
                
                ax.set_xlabel("Hardware Platform")
                ax.set_ylabel("Average Error Rate")
                ax.set_title("Error Rates by Hardware Platform")
                ax.set_xticks(x + width)
                ax.set_xticklabels(platforms)
                ax.legend()
                
            elif metric_type == "strategy":
                # Plot strategy-specific error rates
                strategies = list(self.validation_metrics["strategy_specific"].keys())
                metrics = ["throughput", "latency", "memory"]
                
                # Set up bar positions
                x = np.arange(len(strategies))
                width = 0.25
                
                # Plot bars for each metric
                for i, metric in enumerate(metrics):
                    error_values = [np.mean(self.validation_metrics["strategy_specific"][s][f"{metric}_errors"]) 
                                    for s in strategies]
                    ax.bar(x + i*width, error_values, width, label=f"{metric.capitalize()} Error")
                
                ax.set_xlabel("Execution Strategy")
                ax.set_ylabel("Average Error Rate")
                ax.set_title("Error Rates by Execution Strategy")
                ax.set_xticks(x + width)
                ax.set_xticklabels(strategies)
                ax.legend()
                
            else:
                return {
                    "success": False,
                    "reason": f"Unknown metric type: {metric_type}"
                }
            
            # Save figure to buffer
            import io
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            
            # Return figure data
            return {
                "success": True,
                "figure": fig,
                "image_data": buf,
                "metric_type": metric_type
            }
            
        except ImportError as e:
            return {
                "success": False,
                "reason": f"Visualization requires matplotlib: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Error visualizing validation metrics: {e}")
            traceback.print_exc()
            return {
                "success": False,
                "reason": f"Error generating visualization: {str(e)}"
            }
    
    def close(self) -> bool:
        """
        Close the validator and release resources.
        
        Returns:
            Success status
        """
        success = True
        
        # Close database connection
        if self.db_conn:
            try:
                logger.info("Closing database connection")
                self.db_conn.close()
                logger.info("Database connection closed")
            except Exception as e:
                logger.error(f"Error closing database connection: {e}")
                success = False
        
        logger.info(f"MultiModelEmpiricalValidator closed (success={'yes' if success else 'no'})")
        return success


# Example usage
if __name__ == "__main__":
    # Configure detailed logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    logger.info("Starting MultiModelEmpiricalValidator example")
    
    # Create validator
    validator = MultiModelEmpiricalValidator(
        validation_history_size=100,
        error_threshold=0.15,
        refinement_interval=10,
        enable_trend_analysis=True,
        enable_visualization=True,
        verbose=True
    )
    
    # Generate some simulated predictions and measurements
    for i in range(20):
        # Simulated prediction
        prediction = {
            "total_metrics": {
                "combined_throughput": 100.0 * (1 + 0.1 * np.random.randn()),
                "combined_latency": 50.0 * (1 + 0.1 * np.random.randn()),
                "combined_memory": 2000.0 * (1 + 0.1 * np.random.randn())
            }
        }
        
        # Simulated actual measurement
        actual = {
            "actual_throughput": 100.0 * (1 + 0.15 * np.random.randn()),
            "actual_latency": 50.0 * (1 + 0.15 * np.random.randn()),
            "actual_memory": 2000.0 * (1 + 0.15 * np.random.randn())
        }
        
        # Simulated model configs
        model_configs = [
            {"model_name": "bert-base-uncased", "model_type": "text_embedding", "batch_size": 4},
            {"model_name": "vit-base-patch16-224", "model_type": "vision", "batch_size": 1}
        ]
        
        # Validate prediction
        validation_metrics = validator.validate_prediction(
            prediction=prediction,
            actual_measurement=actual,
            model_configs=model_configs,
            hardware_platform="webgpu",
            execution_strategy="parallel"
        )
        
        logger.info(f"Validation #{i+1}: "
                   f"Throughput error: {validation_metrics['current_errors']['throughput']:.2%}, "
                   f"Latency error: {validation_metrics['current_errors']['latency']:.2%}, "
                   f"Memory error: {validation_metrics['current_errors']['memory']:.2%}")
    
    # Get validation metrics
    metrics = validator.get_validation_metrics()
    logger.info(f"Validation count: {metrics['validation_count']}")
    logger.info(f"Average throughput error: {metrics['error_rates'].get('avg_throughput_error', 0):.2%}")
    logger.info(f"Average latency error: {metrics['error_rates'].get('avg_latency_error', 0):.2%}")
    logger.info(f"Average memory error: {metrics['error_rates'].get('avg_memory_error', 0):.2%}")
    
    # Get refinement recommendations
    recommendations = validator.get_refinement_recommendations()
    logger.info(f"Refinement needed: {recommendations['refinement_needed']}")
    if recommendations['refinement_needed']:
        logger.info(f"Reason: {recommendations['reason']}")
        logger.info(f"Recommended method: {recommendations['recommended_method']}")
    
    # Generate validation dataset
    dataset = validator.generate_validation_dataset()
    if dataset["success"]:
        logger.info(f"Generated validation dataset with {dataset['record_count']} records")
    
    # Visualize validation metrics if matplotlib is available
    try:
        import matplotlib
        visualization = validator.visualize_validation_metrics()
        if visualization["success"]:
            logger.info("Visualization generated successfully")
    except ImportError:
        logger.info("Visualization skipped (matplotlib not available)")
    
    # Close validator
    validator.close()
    logger.info("MultiModelEmpiricalValidator example completed")