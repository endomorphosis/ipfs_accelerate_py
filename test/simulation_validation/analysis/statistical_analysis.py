#!/usr/bin/env python3
"""
Statistical Analysis for Simulation Validation

This module provides statistical analysis tools for simulation validation results.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from scipy import stats
import json

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("simulation_validation.analysis")


class SimulationAnalysisError(Exception):
    """Exception raised for errors in simulation analysis."""
    pass


class StatisticalAnalyzer:
    """Statistical analyzer for simulation validation results."""
    
    def __init__(self, validation_results: Optional[Dict[str, Any]] = None):
        """
        Initialize the statistical analyzer.
        
        Args:
            validation_results: Validation results from SimulationValidator
        """
        self.validation_results = validation_results or {}
        self.analysis_results = {}
        
        logger.info("Initialized StatisticalAnalyzer")
    
    def load_validation_results(self, file_path: str) -> Dict[str, Any]:
        """
        Load validation results from a file.
        
        Args:
            file_path: Path to the validation results file
            
        Returns:
            Loaded validation results
        """
        if not os.path.exists(file_path):
            raise SimulationAnalysisError(f"File not found: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            self.validation_results = data.get("validation_results", {})
            logger.info(f"Loaded validation results from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading validation results: {e}")
            raise SimulationAnalysisError(f"Failed to load validation results: {e}")
    
    def convert_to_dataframe(self) -> pd.DataFrame:
        """
        Convert validation results to a DataFrame for analysis.
        
        Returns:
            DataFrame with validation results
        """
        if not self.validation_results:
            raise SimulationAnalysisError("No validation results available for analysis")
        
        records = []
        
        for config_key, config_results in self.validation_results.items():
            # Extract configuration
            config = config_results.get("configuration", {})
            hardware_type = config.get("hardware_type", "unknown")
            model_type = config.get("model_type", "unknown")
            model_name = config.get("model_name", "unknown")
            batch_size = config.get("batch_size", "unknown")
            precision = config.get("precision", "unknown")
            
            # Get overall metrics
            overall_accuracy = config_results.get("overall_accuracy", 0.0)
            passes_validation = config_results.get("passes_validation", False)
            sim_count = config_results.get("sim_count", 0)
            real_count = config_results.get("real_count", 0)
            
            # Create base record
            base_record = {
                "config_key": config_key,
                "hardware_type": hardware_type,
                "model_type": model_type,
                "model_name": model_name,
                "batch_size": batch_size,
                "precision": precision,
                "overall_accuracy": overall_accuracy,
                "passes_validation": passes_validation,
                "sim_count": sim_count,
                "real_count": real_count
            }
            
            # Extract metrics
            metrics = config_results.get("metrics", {})
            
            # Create a record for each metric
            if metrics:
                for metric_name, metric_results in metrics.items():
                    record = base_record.copy()
                    record["metric_name"] = metric_name
                    record["accuracy_score"] = metric_results.get("accuracy_score", 0.0)
                    record["mean_error"] = metric_results.get("mean_error", float('nan'))
                    record["median_error"] = metric_results.get("median_error", float('nan'))
                    record["max_error"] = metric_results.get("max_error", float('nan'))
                    record["min_error"] = metric_results.get("min_error", float('nan'))
                    record["rmse"] = metric_results.get("rmse", float('nan'))
                    record["correlation"] = metric_results.get("correlation", float('nan'))
                    
                    records.append(record)
            else:
                # Add a record without metric details
                record = base_record.copy()
                record["metric_name"] = "overall"
                record["accuracy_score"] = overall_accuracy
                record["mean_error"] = float('nan')
                record["median_error"] = float('nan')
                record["max_error"] = float('nan')
                record["min_error"] = float('nan')
                record["rmse"] = float('nan')
                record["correlation"] = float('nan')
                
                records.append(record)
        
        df = pd.DataFrame(records)
        logger.info(f"Created DataFrame with {len(df)} records from validation results")
        return df
    
    def analyze_by_hardware_type(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Dict[str, Any]]:
        """
        Analyze validation results by hardware type.
        
        Args:
            df: DataFrame with validation results (if None, will be created)
            
        Returns:
            Analysis results by hardware type
        """
        if df is None:
            df = self.convert_to_dataframe()
        
        if df.empty:
            logger.warning("Empty DataFrame for analysis")
            return {}
        
        # Group by hardware type
        grouped = df.groupby("hardware_type")
        
        analysis = {}
        
        for hardware_type, group in grouped:
            # Calculate overall statistics
            overall_accuracy = group["overall_accuracy"].mean()
            pass_rate = group["passes_validation"].mean()
            config_count = len(group["config_key"].unique())
            
            # Calculate metric-specific statistics
            metric_stats = {}
            for metric, metric_group in group.groupby("metric_name"):
                if metric == "overall":
                    continue
                    
                metric_stats[metric] = {
                    "accuracy_score": metric_group["accuracy_score"].mean(),
                    "mean_error": metric_group["mean_error"].mean(),
                    "median_error": metric_group["median_error"].mean(),
                    "max_error": metric_group["max_error"].mean(),
                    "min_error": metric_group["min_error"].mean(),
                    "rmse": metric_group["rmse"].mean(),
                    "correlation": metric_group["correlation"].mean(),
                    "config_count": len(metric_group)
                }
            
            # Store results
            analysis[hardware_type] = {
                "overall_accuracy": float(overall_accuracy),
                "pass_rate": float(pass_rate),
                "config_count": int(config_count),
                "metrics": metric_stats
            }
        
        # Store analysis results
        self.analysis_results["by_hardware_type"] = analysis
        logger.info(f"Analyzed results for {len(analysis)} hardware types")
        
        return analysis
    
    def analyze_by_model_type(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Dict[str, Any]]:
        """
        Analyze validation results by model type.
        
        Args:
            df: DataFrame with validation results (if None, will be created)
            
        Returns:
            Analysis results by model type
        """
        if df is None:
            df = self.convert_to_dataframe()
        
        if df.empty:
            logger.warning("Empty DataFrame for analysis")
            return {}
        
        # Group by model type
        grouped = df.groupby("model_type")
        
        analysis = {}
        
        for model_type, group in grouped:
            # Calculate overall statistics
            overall_accuracy = group["overall_accuracy"].mean()
            pass_rate = group["passes_validation"].mean()
            config_count = len(group["config_key"].unique())
            
            # Calculate metric-specific statistics
            metric_stats = {}
            for metric, metric_group in group.groupby("metric_name"):
                if metric == "overall":
                    continue
                    
                metric_stats[metric] = {
                    "accuracy_score": metric_group["accuracy_score"].mean(),
                    "mean_error": metric_group["mean_error"].mean(),
                    "median_error": metric_group["median_error"].mean(),
                    "max_error": metric_group["max_error"].mean(),
                    "min_error": metric_group["min_error"].mean(),
                    "rmse": metric_group["rmse"].mean(),
                    "correlation": metric_group["correlation"].mean(),
                    "config_count": len(metric_group)
                }
            
            # Store results
            analysis[model_type] = {
                "overall_accuracy": float(overall_accuracy),
                "pass_rate": float(pass_rate),
                "config_count": int(config_count),
                "metrics": metric_stats
            }
        
        # Store analysis results
        self.analysis_results["by_model_type"] = analysis
        logger.info(f"Analyzed results for {len(analysis)} model types")
        
        return analysis
    
    def analyze_by_metric(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Dict[str, float]]:
        """
        Analyze validation results by metric.
        
        Args:
            df: DataFrame with validation results (if None, will be created)
            
        Returns:
            Analysis results by metric
        """
        if df is None:
            df = self.convert_to_dataframe()
        
        if df.empty:
            logger.warning("Empty DataFrame for analysis")
            return {}
        
        # Filter out "overall" metric
        df_metrics = df[df["metric_name"] != "overall"]
        
        # Group by metric
        grouped = df_metrics.groupby("metric_name")
        
        analysis = {}
        
        for metric, group in grouped:
            # Calculate statistics
            accuracy_score = group["accuracy_score"].mean()
            mean_error = group["mean_error"].mean()
            median_error = group["median_error"].mean()
            max_error = group["max_error"].max()
            min_error = group["min_error"].min()
            rmse = group["rmse"].mean()
            correlation = group["correlation"].mean()
            config_count = len(group)
            
            # Store results
            analysis[metric] = {
                "accuracy_score": float(accuracy_score),
                "mean_error": float(mean_error),
                "median_error": float(median_error),
                "max_error": float(max_error),
                "min_error": float(min_error),
                "rmse": float(rmse),
                "correlation": float(correlation),
                "config_count": int(config_count)
            }
        
        # Store analysis results
        self.analysis_results["by_metric"] = analysis
        logger.info(f"Analyzed results for {len(analysis)} metrics")
        
        return analysis
    
    def analyze_worst_configurations(
        self, 
        df: Optional[pd.DataFrame] = None,
        n: int = 10
    ) -> Dict[str, Dict[str, Any]]:
        """
        Identify worst-performing configurations.
        
        Args:
            df: DataFrame with validation results (if None, will be created)
            n: Number of worst configurations to return
            
        Returns:
            Worst-performing configurations
        """
        if df is None:
            df = self.convert_to_dataframe()
        
        if df.empty:
            logger.warning("Empty DataFrame for analysis")
            return {}
        
        # Get unique configurations
        config_df = df.drop_duplicates(subset=["config_key"])
        
        # Sort by overall accuracy (ascending)
        worst_configs = config_df.sort_values("overall_accuracy").head(n)
        
        # Convert to dictionary
        results = {}
        for _, row in worst_configs.iterrows():
            config_key = row["config_key"]
            results[config_key] = {
                "hardware_type": row["hardware_type"],
                "model_type": row["model_type"],
                "model_name": row["model_name"],
                "batch_size": row["batch_size"],
                "precision": row["precision"],
                "overall_accuracy": float(row["overall_accuracy"]),
                "passes_validation": bool(row["passes_validation"])
            }
        
        # Store analysis results
        self.analysis_results["worst_configurations"] = results
        logger.info(f"Identified {len(results)} worst-performing configurations")
        
        return results
    
    def analyze_calibration_candidates(
        self,
        df: Optional[pd.DataFrame] = None,
        threshold: float = 0.7,
        min_accuracy: float = 0.5
    ) -> Dict[str, Dict[str, Any]]:
        """
        Identify configurations that are candidates for calibration.
        
        Args:
            df: DataFrame with validation results (if None, will be created)
            threshold: Threshold for identifying calibration candidates
            min_accuracy: Minimum accuracy for calibration candidates
            
        Returns:
            Calibration candidates
        """
        if df is None:
            df = self.convert_to_dataframe()
        
        if df.empty:
            logger.warning("Empty DataFrame for analysis")
            return {}
        
        # Get unique configurations
        config_df = df.drop_duplicates(subset=["config_key"])
        
        # Filter to get calibration candidates:
        # - Failed validation but accuracy >= min_accuracy
        # - Has some predictive value but needs improvement
        candidates = config_df[
            (config_df["overall_accuracy"] < threshold) & 
            (config_df["overall_accuracy"] >= min_accuracy)
        ]
        
        # Sort by accuracy (highest potential first)
        candidates = candidates.sort_values("overall_accuracy", ascending=False)
        
        # Convert to dictionary
        results = {}
        for _, row in candidates.iterrows():
            config_key = row["config_key"]
            results[config_key] = {
                "hardware_type": row["hardware_type"],
                "model_type": row["model_type"],
                "model_name": row["model_name"],
                "batch_size": row["batch_size"],
                "precision": row["precision"],
                "overall_accuracy": float(row["overall_accuracy"]),
                "calibration_potential": float(threshold - row["overall_accuracy"])
            }
        
        # Store analysis results
        self.analysis_results["calibration_candidates"] = results
        logger.info(f"Identified {len(results)} calibration candidates")
        
        return results
    
    def analyze_drift_over_time(
        self,
        time_window: int = 90,
        db_api = None
    ) -> Dict[str, Any]:
        """
        Analyze simulation drift over time.
        
        Args:
            time_window: Time window in days
            db_api: Database API for accessing historical validation results
            
        Returns:
            Drift analysis results
        """
        if db_api is None:
            logger.warning("No database API provided for drift analysis")
            return {
                "error": "No database API provided for drift analysis",
                "drift_detected": False,
                "drift_metrics": {}
            }
        
        try:
            # Query validation results over time
            query = f"""
            SELECT 
                validation_id, 
                validation_timestamp, 
                overall_accuracy,
                pass_rate,
                configuration_count
            FROM 
                simulation_validation_history
            WHERE 
                validation_timestamp >= DATEADD('day', -{time_window}, CURRENT_TIMESTAMP())
            ORDER BY 
                validation_timestamp
            """
            
            results = db_api.execute_query(query)
            
            if results.empty:
                logger.warning("No historical validation results found")
                return {
                    "error": "No historical validation results found",
                    "drift_detected": False,
                    "drift_metrics": {}
                }
            
            # Calculate trend
            x = np.arange(len(results))
            y = results["overall_accuracy"].values
            
            # Perform linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # Calculate drift metrics
            drift_metrics = {
                "slope": float(slope),
                "intercept": float(intercept),
                "r_squared": float(r_value**2),
                "p_value": float(p_value),
                "std_err": float(std_err),
                "trend_direction": "increasing" if slope > 0 else "decreasing",
                "statistically_significant": p_value < 0.05
            }
            
            # Determine if drift is detected
            drift_detected = p_value < 0.05 and abs(slope) > 0.01
            
            # Calculate additional metrics
            earliest_accuracy = results["overall_accuracy"].iloc[0]
            latest_accuracy = results["overall_accuracy"].iloc[-1]
            accuracy_change = latest_accuracy - earliest_accuracy
            
            drift_results = {
                "drift_detected": drift_detected,
                "drift_metrics": drift_metrics,
                "earliest_accuracy": float(earliest_accuracy),
                "latest_accuracy": float(latest_accuracy),
                "accuracy_change": float(accuracy_change),
                "time_window_days": time_window,
                "num_validations": len(results)
            }
            
            # Store analysis results
            self.analysis_results["drift_over_time"] = drift_results
            logger.info(f"Analyzed drift over time (drift detected: {drift_detected})")
            
            return drift_results
            
        except Exception as e:
            logger.error(f"Error analyzing drift over time: {e}")
            return {
                "error": f"Error analyzing drift over time: {e}",
                "drift_detected": False,
                "drift_metrics": {}
            }
    
    def run_comprehensive_analysis(
        self,
        df: Optional[pd.DataFrame] = None,
        include_drift: bool = False,
        db_api = None,
        time_window: int = 90
    ) -> Dict[str, Any]:
        """
        Run a comprehensive analysis of validation results.
        
        Args:
            df: DataFrame with validation results (if None, will be created)
            include_drift: Whether to include drift analysis
            db_api: Database API for accessing historical validation results
            time_window: Time window in days for drift analysis
            
        Returns:
            Comprehensive analysis results
        """
        if df is None:
            df = self.convert_to_dataframe()
        
        if df.empty:
            logger.warning("Empty DataFrame for analysis")
            return {"error": "No data available for analysis"}
        
        # Run all analyses
        logger.info("Running comprehensive analysis...")
        
        hardware_analysis = self.analyze_by_hardware_type(df)
        model_analysis = self.analyze_by_model_type(df)
        metric_analysis = self.analyze_by_metric(df)
        worst_configs = self.analyze_worst_configurations(df)
        calibration_candidates = self.analyze_calibration_candidates(df)
        
        # Include drift analysis if requested
        drift_analysis = None
        if include_drift and db_api is not None:
            logger.info("Running drift analysis...")
            drift_analysis = self.analyze_drift_over_time(time_window, db_api)
        
        # Gather all results
        comprehensive_results = {
            "by_hardware_type": hardware_analysis,
            "by_model_type": model_analysis,
            "by_metric": metric_analysis,
            "worst_configurations": worst_configs,
            "calibration_candidates": calibration_candidates
        }
        
        if drift_analysis:
            comprehensive_results["drift_over_time"] = drift_analysis
        
        # Store results
        self.analysis_results = comprehensive_results
        
        logger.info("Comprehensive analysis complete")
        return comprehensive_results
    
    def save_analysis_results(self, output_file: str) -> str:
        """
        Save analysis results to file.
        
        Args:
            output_file: Path to save results
            
        Returns:
            Path to saved file
        """
        if not self.analysis_results:
            logger.warning("No analysis results to save")
            return ""
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(self.analysis_results, f, indent=2)
        
        logger.info(f"Saved analysis results to {output_file}")
        return output_file
    
    def load_analysis_results(self, input_file: str) -> Dict[str, Any]:
        """
        Load analysis results from file.
        
        Args:
            input_file: Path to load results from
            
        Returns:
            Loaded analysis results
        """
        if not os.path.exists(input_file):
            logger.error(f"Input file not found: {input_file}")
            return {}
        
        try:
            with open(input_file, 'r') as f:
                results = json.load(f)
            
            # Update instance variable
            self.analysis_results = results
            
            logger.info(f"Loaded analysis results from {input_file}")
            return results
        except Exception as e:
            logger.error(f"Error loading analysis results: {e}")
            return {}