#!/usr/bin/env python3
"""
Simulation Accuracy Validation Framework

This module provides the core implementation of the simulation accuracy validation framework,
which measures and tracks the accuracy of simulated benchmarks compared to real hardware results.
"""

import os
import logging
import datetime
import json
from typing import Dict, List, Optional, Union, Any, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("simulation_validation")

# Default metrics for validation
DEFAULT_METRICS = [
    "throughput_items_per_second",
    "average_latency_ms", 
    "memory_peak_mb",
    "initialization_time_ms"
]

class SimulationValidationError(Exception):
    """Exception raised for errors in the simulation validation framework."""
    pass


class SimulationValidator:
    """
    Core class for validating simulation accuracy against real hardware results.
    """
    
    def __init__(
        self,
        db_api=None,
        metrics: List[str] = None,
        output_dir: str = "./simulation_validation_results",
        confidence_threshold: float = 0.9,
        metrics_threshold: Dict[str, float] = None
    ):
        """
        Initialize the simulation validator.
        
        Args:
            db_api: Database API for accessing benchmark results
            metrics: List of metrics to validate (defaults to DEFAULT_METRICS)
            output_dir: Directory to store validation results
            confidence_threshold: Threshold for overall simulation confidence
            metrics_threshold: Dictionary of thresholds for specific metrics
        """
        self.db_api = db_api
        self.metrics = metrics or DEFAULT_METRICS
        self.output_dir = output_dir
        self.confidence_threshold = confidence_threshold
        self.metrics_threshold = metrics_threshold or {
            "throughput_items_per_second": 0.85,  # 85% accuracy
            "average_latency_ms": 0.85,           # 85% accuracy
            "memory_peak_mb": 0.9,                # 90% accuracy
            "initialization_time_ms": 0.8,        # 80% accuracy
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Dictionary to store validation results
        self.validation_results = {}
        
        logger.info(f"Initialized SimulationValidator with metrics: {self.metrics}")
    
    def load_benchmark_data(
        self,
        hardware_types: List[str] = None,
        model_types: List[str] = None,
        time_range: int = 90,  # Days
        include_simulated: bool = True,
        include_real: bool = True
    ) -> pd.DataFrame:
        """
        Load benchmark data for validation.
        
        Args:
            hardware_types: List of hardware types to include
            model_types: List of model types to include
            time_range: Time range in days to include
            include_simulated: Whether to include simulated results
            include_real: Whether to include real hardware results
            
        Returns:
            DataFrame with benchmark data
        """
        if not self.db_api:
            raise SimulationValidationError("No database API provided. Unable to load benchmark data.")
        
        # Create filter conditions
        conditions = []
        
        # Time range filter
        if time_range:
            cutoff_date = datetime.datetime.now() - datetime.timedelta(days=time_range)
            conditions.append(f"timestamp >= '{cutoff_date.isoformat()}'")
        
        # Hardware types filter
        if hardware_types:
            hardware_list = ", ".join([f"'{hw}'" for hw in hardware_types])
            conditions.append(f"hardware_type IN ({hardware_list})")
        
        # Model types filter
        if model_types:
            model_list = ", ".join([f"'{model}'" for model in model_types])
            conditions.append(f"model_type IN ({model_list})")
        
        # Simulation status filters
        simulation_conditions = []
        if include_simulated:
            simulation_conditions.append("is_simulation = TRUE")
        if include_real:
            simulation_conditions.append("is_simulation = FALSE")
        
        if simulation_conditions:
            conditions.append(f"({' OR '.join(simulation_conditions)})")
        
        # Build WHERE clause
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        # Define required columns
        columns = [
            "benchmark_id", "hardware_type", "model_type", "model_name", 
            "batch_size", "precision", "is_simulation", "timestamp"
        ] + self.metrics
        
        # Query database
        query = f"""
        SELECT {', '.join(columns)}
        FROM benchmark_results
        WHERE {where_clause}
        ORDER BY timestamp DESC
        """
        
        logger.info(f"Executing query: {query}")
        
        try:
            result = self.db_api.execute_query(query)
            logger.info(f"Loaded {len(result)} benchmark results for validation")
            return result
        except Exception as e:
            logger.error(f"Error loading benchmark data: {e}")
            raise SimulationValidationError(f"Failed to load benchmark data: {e}")
    
    def prepare_validation_pairs(
        self,
        df: pd.DataFrame,
        match_criteria: List[str] = None
    ) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Prepare pairs of simulated and real results for comparison.
        
        Args:
            df: DataFrame with benchmark data
            match_criteria: Criteria for matching simulated and real results
                (defaults to ["hardware_type", "model_type", "model_name", "batch_size", "precision"])
                
        Returns:
            Dictionary mapping criteria strings to tuples of (simulated_df, real_df)
        """
        if match_criteria is None:
            match_criteria = ["hardware_type", "model_type", "model_name", "batch_size", "precision"]
        
        # Split into simulated and real dataframes
        sim_df = df[df["is_simulation"] == True]
        real_df = df[df["is_simulation"] == False]
        
        if sim_df.empty:
            logger.warning("No simulated results found in the data")
            return {}
        
        if real_df.empty:
            logger.warning("No real hardware results found in the data")
            return {}
        
        # Group by match criteria
        sim_groups = sim_df.groupby(match_criteria)
        real_groups = real_df.groupby(match_criteria)
        
        # Find common groups
        sim_keys = set(sim_groups.groups.keys())
        real_keys = set(real_groups.groups.keys())
        common_keys = sim_keys.intersection(real_keys)
        
        logger.info(f"Found {len(common_keys)} matching configurations for validation")
        
        # Create pairs for each common key
        validation_pairs = {}
        for key in common_keys:
            key_str = "_".join(str(k) for k in key)
            validation_pairs[key_str] = (
                sim_groups.get_group(key),
                real_groups.get_group(key)
            )
        
        return validation_pairs
    
    def calculate_simulation_accuracy(
        self,
        sim_df: pd.DataFrame,
        real_df: pd.DataFrame,
        metric: str
    ) -> Dict[str, float]:
        """
        Calculate accuracy metrics for a single performance metric.
        
        Args:
            sim_df: DataFrame with simulated results
            real_df: DataFrame with real hardware results
            metric: Performance metric to validate
            
        Returns:
            Dictionary with accuracy metrics
        """
        # Get relevant values
        sim_values = sim_df[metric].values
        real_values = real_df[metric].values
        
        # Handle empty dataframes
        if len(sim_values) == 0 or len(real_values) == 0:
            logger.warning(f"Empty data for metric {metric}")
            return {
                "mean_error": float('nan'),
                "median_error": float('nan'),
                "max_error": float('nan'),
                "min_error": float('nan'),
                "rmse": float('nan'),
                "correlation": float('nan'),
                "accuracy_score": 0.0
            }
        
        # Use average values for comparison
        sim_avg = np.mean(sim_values)
        real_avg = np.mean(real_values)
        
        # Calculate basic error metrics
        abs_error = abs(sim_avg - real_avg)
        rel_error = abs_error / real_avg if real_avg != 0 else float('inf')
        
        # Calculate percentage error for each value
        if len(sim_values) == len(real_values):
            percentage_errors = np.abs(sim_values - real_values) / np.abs(real_values)
            percentage_errors = percentage_errors[~np.isnan(percentage_errors)]  # Remove NaN values
            
            mean_error = np.mean(percentage_errors) if len(percentage_errors) > 0 else float('nan')
            median_error = np.median(percentage_errors) if len(percentage_errors) > 0 else float('nan')
            max_error = np.max(percentage_errors) if len(percentage_errors) > 0 else float('nan')
            min_error = np.min(percentage_errors) if len(percentage_errors) > 0 else float('nan')
        else:
            # If lengths don't match, just use aggregated values
            mean_error = rel_error
            median_error = rel_error
            max_error = rel_error
            min_error = rel_error
        
        # Calculate RMSE
        if len(sim_values) == len(real_values):
            rmse = np.sqrt(np.mean((sim_values - real_values) ** 2))
        else:
            rmse = abs_error
        
        # Calculate correlation if possible
        if len(sim_values) > 1 and len(real_values) > 1 and len(sim_values) == len(real_values):
            correlation, _ = stats.pearsonr(sim_values, real_values)
        else:
            correlation = float('nan')
        
        # Calculate overall accuracy score (1.0 is perfect, 0.0 is worst)
        # We use 1 - min(rel_error, 1.0) to ensure score is between 0 and 1
        accuracy_score = max(0.0, 1.0 - min(rel_error, 1.0))
        
        return {
            "mean_error": float(mean_error),
            "median_error": float(median_error),
            "max_error": float(max_error),
            "min_error": float(min_error),
            "rmse": float(rmse),
            "correlation": float(correlation),
            "accuracy_score": float(accuracy_score)
        }
    
    def validate_configuration(
        self,
        sim_df: pd.DataFrame,
        real_df: pd.DataFrame,
        config_key: str
    ) -> Dict[str, Any]:
        """
        Validate simulation accuracy for a specific configuration.
        
        Args:
            sim_df: DataFrame with simulated results
            real_df: DataFrame with real hardware results
            config_key: Configuration key
            
        Returns:
            Dictionary with validation results
        """
        # Extract configuration components
        config_components = config_key.split("_")
        
        # Get median values for each metric
        validation_results = {
            "configuration": {
                "hardware_type": config_components[0] if len(config_components) > 0 else "unknown",
                "model_type": config_components[1] if len(config_components) > 1 else "unknown",
                "model_name": config_components[2] if len(config_components) > 2 else "unknown",
                "batch_size": config_components[3] if len(config_components) > 3 else "unknown",
                "precision": config_components[4] if len(config_components) > 4 else "unknown",
            },
            "sim_count": len(sim_df),
            "real_count": len(real_df),
            "metrics": {},
            "overall_accuracy": 0.0,
            "validation_timestamp": datetime.datetime.now().isoformat()
        }
        
        # Calculate accuracy for each metric
        metric_scores = []
        for metric in self.metrics:
            if metric in sim_df.columns and metric in real_df.columns:
                metric_results = self.calculate_simulation_accuracy(sim_df, real_df, metric)
                validation_results["metrics"][metric] = metric_results
                metric_scores.append(metric_results["accuracy_score"])
            else:
                logger.warning(f"Metric {metric} not found in both dataframes")
        
        # Calculate overall accuracy as weighted average
        if metric_scores:
            overall_accuracy = np.mean(metric_scores)
            validation_results["overall_accuracy"] = float(overall_accuracy)
        
        # Determine if simulation passes validation
        threshold = self.confidence_threshold
        validation_results["passes_validation"] = validation_results["overall_accuracy"] >= threshold
        
        return validation_results
    
    def validate_all_configurations(
        self,
        validation_pairs: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Validate all configuration pairs.
        
        Args:
            validation_pairs: Dictionary mapping config keys to (sim_df, real_df) tuples
            
        Returns:
            Dictionary mapping config keys to validation results
        """
        all_results = {}
        
        for config_key, (sim_df, real_df) in validation_pairs.items():
            logger.info(f"Validating configuration: {config_key}")
            
            # Validate this configuration
            config_results = self.validate_configuration(sim_df, real_df, config_key)
            
            # Store results
            all_results[config_key] = config_results
        
        # Store all validation results
        self.validation_results = all_results
        
        return all_results
    
    def calculate_overall_simulation_quality(self) -> Dict[str, Any]:
        """
        Calculate overall simulation quality metrics across all configurations.
        
        Returns:
            Dictionary with overall simulation quality metrics
        """
        if not self.validation_results:
            logger.warning("No validation results available. Run validate_configurations first.")
            return {
                "overall_accuracy": 0.0,
                "configurations_validated": 0,
                "configurations_passed": 0,
                "pass_rate": 0.0,
                "metric_accuracies": {}
            }
        
        # Count configurations
        total_configs = len(self.validation_results)
        passed_configs = sum(1 for results in self.validation_results.values() 
                             if results.get("passes_validation", False))
        
        # Calculate pass rate
        pass_rate = passed_configs / total_configs if total_configs > 0 else 0.0
        
        # Calculate average accuracy for each metric
        metric_accuracies = {}
        for metric in self.metrics:
            accuracies = []
            for config_results in self.validation_results.values():
                if metric in config_results.get("metrics", {}):
                    accuracy = config_results["metrics"][metric]["accuracy_score"]
                    if not np.isnan(accuracy):
                        accuracies.append(accuracy)
            
            if accuracies:
                metric_accuracies[metric] = float(np.mean(accuracies))
            else:
                metric_accuracies[metric] = 0.0
        
        # Calculate overall average accuracy
        overall_accuracy = np.mean([results["overall_accuracy"] 
                                  for results in self.validation_results.values()])
        
        return {
            "overall_accuracy": float(overall_accuracy),
            "configurations_validated": total_configs,
            "configurations_passed": passed_configs,
            "pass_rate": float(pass_rate),
            "metric_accuracies": metric_accuracies,
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    def save_validation_results(
        self, 
        output_file: Optional[str] = None
    ) -> str:
        """
        Save validation results to file.
        
        Args:
            output_file: Path to save results (defaults to output_dir/simulation_validation_YYYYMMDD.json)
            
        Returns:
            Path to saved file
        """
        if not self.validation_results:
            logger.warning("No validation results to save")
            return ""
        
        # Generate default output filename if not provided
        if output_file is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.output_dir, f"simulation_validation_{timestamp}.json")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Prepare data to save
        save_data = {
            "validation_results": self.validation_results,
            "overall_quality": self.calculate_overall_simulation_quality(),
            "validation_timestamp": datetime.datetime.now().isoformat(),
            "metrics_validated": self.metrics,
            "confidence_threshold": self.confidence_threshold
        }
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        logger.info(f"Saved validation results to {output_file}")
        return output_file
    
    def load_validation_results(
        self,
        input_file: str
    ) -> Dict[str, Any]:
        """
        Load validation results from file.
        
        Args:
            input_file: Path to load results from
            
        Returns:
            Loaded validation results
        """
        if not os.path.exists(input_file):
            logger.error(f"Input file not found: {input_file}")
            return {}
        
        try:
            with open(input_file, 'r') as f:
                results = json.load(f)
            
            # Update instance variables
            self.validation_results = results.get("validation_results", {})
            self.metrics = results.get("metrics_validated", self.metrics)
            self.confidence_threshold = results.get("confidence_threshold", self.confidence_threshold)
            
            logger.info(f"Loaded validation results from {input_file}")
            return results
        except Exception as e:
            logger.error(f"Error loading validation results: {e}")
            return {}
    
    def run_validation(
        self,
        hardware_types: List[str] = None,
        model_types: List[str] = None,
        time_range: int = 90,
        match_criteria: List[str] = None,
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run a complete validation workflow.
        
        Args:
            hardware_types: List of hardware types to include
            model_types: List of model types to include
            time_range: Time range in days to include
            match_criteria: Criteria for matching simulated and real results
            output_file: Path to save results
            
        Returns:
            Dictionary with overall simulation quality metrics
        """
        # Load benchmark data
        logger.info("Loading benchmark data...")
        df = self.load_benchmark_data(
            hardware_types=hardware_types,
            model_types=model_types,
            time_range=time_range
        )
        
        # Prepare validation pairs
        logger.info("Preparing validation pairs...")
        validation_pairs = self.prepare_validation_pairs(df, match_criteria)
        
        if not validation_pairs:
            logger.warning("No matching configurations found for validation")
            return {"error": "No matching configurations found for validation"}
        
        # Validate all configurations
        logger.info(f"Validating {len(validation_pairs)} configurations...")
        self.validate_all_configurations(validation_pairs)
        
        # Calculate overall quality
        logger.info("Calculating overall simulation quality...")
        quality_metrics = self.calculate_overall_simulation_quality()
        
        # Save results
        if output_file:
            logger.info(f"Saving validation results to {output_file}...")
            self.save_validation_results(output_file)
        
        return quality_metrics