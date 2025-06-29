#!/usr/bin/env python3
"""
Predictive Modeling for the Simulation Accuracy and Validation Framework.

This module provides predictive modeling capabilities for simulation accuracy,
allowing for forecasting future accuracy metrics based on historical validation
results. The module includes:
- Time series forecasting for accuracy metrics
- Feature-based regression models for accuracy prediction
- Classification models for binary accuracy assessments
- Ensemble prediction techniques for robust forecasting
- Model performance evaluation and comparison
"""

import logging
import numpy as np
import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("analysis.predictive_modeling")

# Import base class
from duckdb_api.simulation_validation.analysis.base import AnalysisMethod
from duckdb_api.simulation_validation.core.base import (
    SimulationResult,
    HardwareResult,
    ValidationResult
)

class PredictiveModeling(AnalysisMethod):
    """
    Predictive modeling for simulation accuracy metrics.
    
    This class extends the basic AnalysisMethod to provide advanced predictive
    modeling techniques for forecasting simulation accuracy:
    - Time series forecasting for accuracy metrics
    - Feature-based regression models for accuracy prediction
    - Classification models for binary accuracy assessments
    - Ensemble prediction techniques for robust forecasting
    - Model performance evaluation and comparison
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the predictive modeling method.
        
        Args:
            config: Configuration options for the analysis method
        """
        super().__init__(config)
        
        # Default configuration
        default_config = {
            # Common metrics to predict
            "metrics_to_predict": [
                "throughput_items_per_second",
                "average_latency_ms",
                "memory_peak_mb",
                "power_consumption_w"
            ],
            
            # Target metrics for prediction
            "target_metrics": [
                "mape",  # Mean Absolute Percentage Error
                "overall_accuracy_score"  # Overall accuracy score
            ],
            
            # Time series forecasting configuration
            "time_series_forecasting": {
                "enabled": True,
                "methods": ["arima", "exponential_smoothing", "prophet"],
                "forecast_horizon": 10,  # Number of periods to forecast
                "interval_width": 0.95,  # Prediction interval width (95%)
                "min_history_periods": 10,  # Minimum periods required for forecasting
                "seasonal_period": 7  # Seasonal period (e.g., 7 for weekly seasonality)
            },
            
            # Regression model configuration
            "regression_model": {
                "enabled": True,
                "methods": ["random_forest", "gradient_boosting", "linear"],
                "validation_split": 0.2,  # Validation split ratio
                "random_state": 42,  # Random state for reproducibility
                "feature_selection": True,  # Whether to perform feature selection
                "min_samples": 15  # Minimum samples required for regression modeling
            },
            
            # Classification model configuration
            "classification_model": {
                "enabled": True,
                "methods": ["random_forest", "gradient_boosting", "logistic"],
                "threshold": 0.1,  # Accuracy threshold for binary classification
                "validation_split": 0.2,  # Validation split ratio
                "random_state": 42,  # Random state for reproducibility
                "min_samples": 20  # Minimum samples required for classification modeling
            },
            
            # Ensemble prediction configuration
            "ensemble_prediction": {
                "enabled": True,
                "methods": ["voting", "stacking"],
                "weights": {  # Weights for different methods (must sum to 1.0)
                    "time_series": 0.4,
                    "regression": 0.4,
                    "classification": 0.2
                }
            },
            
            # Model evaluation configuration
            "model_evaluation": {
                "enabled": True,
                "metrics": ["rmse", "mae", "mape", "r2"],
                "cross_validation_folds": 5,  # Number of folds for cross-validation
                "comparison_plot": True  # Whether to generate comparison plots
            }
        }
        
        # Apply default config values if not specified
        for key, value in default_config.items():
            if key not in self.config:
                self.config[key] = value
            elif isinstance(value, dict) and isinstance(self.config[key], dict):
                # Merge nested dictionaries
                for nested_key, nested_value in value.items():
                    if nested_key not in self.config[key]:
                        self.config[key][nested_key] = nested_value
    
    def analyze(
        self, 
        validation_results: List[ValidationResult]
    ) -> Dict[str, Any]:
        """
        Perform predictive modeling on validation results.
        
        Args:
            validation_results: List of validation results to analyze
            
        Returns:
            Dictionary containing predictive modeling results and insights
        """
        # Check requirements
        meets_req, error_msg = self.check_requirements(validation_results)
        if not meets_req:
            logger.warning(f"Requirements not met for predictive modeling: {error_msg}")
            return {"status": "error", "message": error_msg}
        
        # Initialize results dictionary
        analysis_results = {
            "status": "success",
            "timestamp": datetime.datetime.now().isoformat(),
            "num_validation_results": len(validation_results),
            "metrics_predicted": self.config["metrics_to_predict"],
            "target_metrics": self.config["target_metrics"],
            "methods": {},
            "forecasts": {},
            "model_evaluation": {},
            "insights": {
                "key_findings": [],
                "recommendations": []
            }
        }
        
        # Extract data for prediction
        prediction_data = self._extract_prediction_data(validation_results)
        
        if not prediction_data:
            return {
                "status": "error",
                "message": "Failed to extract data for predictive modeling"
            }
        
        # Split data into features (X) and targets (y)
        X, y, timestamps, metadata = self._prepare_model_data(prediction_data)
        
        if X is None or y is None:
            return {
                "status": "error",
                "message": "Insufficient data for predictive modeling"
            }
        
        # Perform time series forecasting if enabled
        if self.config["time_series_forecasting"]["enabled"]:
            try:
                # Check if we have enough data points
                min_periods = self.config["time_series_forecasting"]["min_history_periods"]
                if len(timestamps) >= min_periods:
                    ts_forecasts = self._forecast_time_series(
                        y, timestamps, prediction_data)
                    analysis_results["methods"]["time_series_forecasting"] = ts_forecasts
                    
                    # Add forecasts to results
                    for metric, forecast in ts_forecasts.get("forecasts", {}).items():
                        if metric not in analysis_results["forecasts"]:
                            analysis_results["forecasts"][metric] = {}
                        
                        analysis_results["forecasts"][metric]["time_series"] = forecast
                else:
                    analysis_results["methods"]["time_series_forecasting"] = {
                        "status": "skipped",
                        "message": f"Insufficient data points for time series forecasting. "
                                f"Required: {min_periods}, Provided: {len(timestamps)}"
                    }
            except Exception as e:
                logger.error(f"Error in time series forecasting: {e}")
                analysis_results["methods"]["time_series_forecasting"] = {
                    "status": "error",
                    "message": str(e)
                }
        
        # Perform regression modeling if enabled
        if self.config["regression_model"]["enabled"]:
            try:
                # Check if we have enough data points
                min_samples = self.config["regression_model"]["min_samples"]
                if X.shape[0] >= min_samples:
                    regression_results = self._build_regression_models(X, y, metadata)
                    analysis_results["methods"]["regression_model"] = regression_results
                    
                    # Add forecasts to results
                    for metric, prediction in regression_results.get("predictions", {}).items():
                        if metric not in analysis_results["forecasts"]:
                            analysis_results["forecasts"][metric] = {}
                        
                        analysis_results["forecasts"][metric]["regression"] = prediction
                else:
                    analysis_results["methods"]["regression_model"] = {
                        "status": "skipped",
                        "message": f"Insufficient data points for regression modeling. "
                                f"Required: {min_samples}, Provided: {X.shape[0]}"
                    }
            except Exception as e:
                logger.error(f"Error in regression modeling: {e}")
                analysis_results["methods"]["regression_model"] = {
                    "status": "error",
                    "message": str(e)
                }
        
        # Perform classification modeling if enabled
        if self.config["classification_model"]["enabled"]:
            try:
                # Check if we have enough data points
                min_samples = self.config["classification_model"]["min_samples"]
                if X.shape[0] >= min_samples:
                    classification_results = self._build_classification_models(X, y, metadata)
                    analysis_results["methods"]["classification_model"] = classification_results
                    
                    # Add predictions to results
                    for metric, prediction in classification_results.get("predictions", {}).items():
                        if metric not in analysis_results["forecasts"]:
                            analysis_results["forecasts"][metric] = {}
                        
                        analysis_results["forecasts"][metric]["classification"] = prediction
                else:
                    analysis_results["methods"]["classification_model"] = {
                        "status": "skipped",
                        "message": f"Insufficient data points for classification modeling. "
                                f"Required: {min_samples}, Provided: {X.shape[0]}"
                    }
            except Exception as e:
                logger.error(f"Error in classification modeling: {e}")
                analysis_results["methods"]["classification_model"] = {
                    "status": "error",
                    "message": str(e)
                }
        
        # Perform ensemble prediction if enabled
        if self.config["ensemble_prediction"]["enabled"]:
            try:
                ensemble_results = self._build_ensemble_prediction(analysis_results["forecasts"])
                analysis_results["methods"]["ensemble_prediction"] = ensemble_results
                
                # Add ensemble forecasts to results
                for metric, forecast in ensemble_results.get("forecasts", {}).items():
                    if metric not in analysis_results["forecasts"]:
                        analysis_results["forecasts"][metric] = {}
                    
                    analysis_results["forecasts"][metric]["ensemble"] = forecast
            except Exception as e:
                logger.error(f"Error in ensemble prediction: {e}")
                analysis_results["methods"]["ensemble_prediction"] = {
                    "status": "error",
                    "message": str(e)
                }
        
        # Evaluate models if enabled
        if self.config["model_evaluation"]["enabled"]:
            try:
                evaluation_results = self._evaluate_models(
                    analysis_results["methods"], X, y, timestamps)
                analysis_results["model_evaluation"] = evaluation_results
            except Exception as e:
                logger.error(f"Error in model evaluation: {e}")
                analysis_results["model_evaluation"] = {
                    "status": "error",
                    "message": str(e)
                }
        
        # Generate key findings
        analysis_results["insights"]["key_findings"] = self._generate_key_findings(
            analysis_results["forecasts"],
            analysis_results["model_evaluation"],
            prediction_data
        )
        
        # Generate recommendations
        analysis_results["insights"]["recommendations"] = self._generate_recommendations(
            analysis_results["forecasts"],
            analysis_results["model_evaluation"],
            analysis_results["insights"]["key_findings"]
        )
        
        return analysis_results
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get information about the capabilities of the predictive modeling.
        
        Returns:
            Dictionary describing the capabilities
        """
        return {
            "name": "Predictive Modeling",
            "description": "Builds predictive models for forecasting simulation accuracy",
            "methods": [
                {
                    "name": "Time Series Forecasting",
                    "description": "Forecasts accuracy metrics using time series analysis",
                    "enabled": self.config["time_series_forecasting"]["enabled"],
                    "techniques": self.config["time_series_forecasting"]["methods"]
                },
                {
                    "name": "Regression Modeling",
                    "description": "Predicts accuracy metrics using feature-based regression",
                    "enabled": self.config["regression_model"]["enabled"],
                    "techniques": self.config["regression_model"]["methods"]
                },
                {
                    "name": "Classification Modeling",
                    "description": "Predicts whether accuracy will meet thresholds",
                    "enabled": self.config["classification_model"]["enabled"],
                    "techniques": self.config["classification_model"]["methods"]
                },
                {
                    "name": "Ensemble Prediction",
                    "description": "Combines multiple prediction methods for robust forecasts",
                    "enabled": self.config["ensemble_prediction"]["enabled"],
                    "techniques": self.config["ensemble_prediction"]["methods"]
                },
                {
                    "name": "Model Evaluation",
                    "description": "Evaluates and compares prediction model performance",
                    "enabled": self.config["model_evaluation"]["enabled"],
                    "metrics": self.config["model_evaluation"]["metrics"]
                }
            ],
            "output_format": {
                "forecasts": "Accuracy metric forecasts for each prediction method",
                "model_evaluation": "Performance metrics for predictive models",
                "insights": "Key findings and recommendations based on forecasts"
            }
        }
    
    def get_requirements(self) -> Dict[str, Any]:
        """
        Get information about the requirements of this analysis method.
        
        Returns:
            Dictionary describing the requirements
        """
        # Define minimum requirements
        requirements = {
            "min_validation_results": 10,
            "required_metrics": self.config["metrics_to_predict"],
            "optimal_validation_results": 30,
            "time_series_requirements": {
                "min_samples": self.config["time_series_forecasting"]["min_history_periods"],
                "time_series_required": True
            },
            "regression_requirements": {
                "min_samples": self.config["regression_model"]["min_samples"]
            },
            "classification_requirements": {
                "min_samples": self.config["classification_model"]["min_samples"]
            }
        }
        
        return requirements
    
    def _extract_prediction_data(
        self,
        validation_results: List[ValidationResult]
    ) -> Dict[str, Any]:
        """
        Extract data for predictive modeling from validation results.
        
        Args:
            validation_results: List of validation results
            
        Returns:
            Dictionary with structured data for prediction
        """
        if not validation_results:
            return {}
        
        # Define metrics to predict
        metrics_to_predict = self.config["metrics_to_predict"]
        target_metrics = self.config["target_metrics"]
        
        # Initialize data structure
        prediction_data = {
            "timestamps": [],
            "features": [],
            "targets": [],
            "feature_names": [],
            "target_names": [],
            "metadata": []
        }
        
        # Extract data from validation results
        for result in validation_results:
            # Skip if no validation timestamp
            if not hasattr(result, "validation_timestamp") or not result.validation_timestamp:
                continue
                
            # Process timestamp
            try:
                if isinstance(result.validation_timestamp, str):
                    timestamp = datetime.datetime.fromisoformat(result.validation_timestamp)
                else:
                    timestamp = result.validation_timestamp
                
                prediction_data["timestamps"].append(timestamp)
            except Exception as e:
                logger.warning(f"Error parsing timestamp: {e}")
                continue
            
            # Extract features
            features = []
            feature_names = []
            
            # Hardware and simulation features
            hw_features = {
                "batch_size": result.hardware_result.batch_size,
                "hardware_id": result.hardware_result.hardware_id,
                "model_id": result.hardware_result.model_id,
                "precision": result.hardware_result.precision
            }
            
            # Process hardware features
            for name, value in hw_features.items():
                if name not in feature_names:
                    feature_names.append(name)
                
                # Convert non-numeric values to category codes
                if name in ["hardware_id", "model_id", "precision"]:
                    # Simplified encoding for demonstration
                    if not hasattr(self, "_category_encodings"):
                        self._category_encodings = {}
                    
                    if name not in self._category_encodings:
                        self._category_encodings[name] = {}
                    
                    if value not in self._category_encodings[name]:
                        self._category_encodings[name][value] = len(self._category_encodings[name])
                    
                    features.append(self._category_encodings[name][value])
                else:
                    features.append(value)
            
            # Add performance metrics as features
            for metric in metrics_to_predict:
                if (metric in result.simulation_result.metrics and 
                    metric in result.hardware_result.metrics):
                    
                    sim_val = result.simulation_result.metrics[metric]
                    hw_val = result.hardware_result.metrics[metric]
                    
                    if sim_val is not None and hw_val is not None:
                        if f"{metric}_sim" not in feature_names:
                            feature_names.append(f"{metric}_sim")
                        if f"{metric}_hw" not in feature_names:
                            feature_names.append(f"{metric}_hw")
                        
                        features.append(sim_val)
                        features.append(hw_val)
            
            # Extract targets (accuracy metrics)
            targets = []
            target_names = []
            
            for metric in target_metrics:
                # Check for direct metrics in metrics_comparison
                for performance_metric, comparison in result.metrics_comparison.items():
                    target_key = f"{performance_metric}_{metric}"
                    
                    if metric in comparison:
                        if target_key not in target_names:
                            target_names.append(target_key)
                        
                        targets.append(comparison[metric])
                
                # Check for metrics in additional_metrics
                if hasattr(result, "additional_metrics") and result.additional_metrics:
                    if metric in result.additional_metrics:
                        if metric not in target_names:
                            target_names.append(metric)
                        
                        targets.append(result.additional_metrics[metric])
            
            # Extract metadata for context
            metadata = {
                "hardware_id": result.hardware_result.hardware_id,
                "model_id": result.hardware_result.model_id,
                "batch_size": result.hardware_result.batch_size,
                "precision": result.hardware_result.precision,
                "timestamp": timestamp
            }
            
            # Add data to prediction_data
            prediction_data["features"].append(features)
            prediction_data["targets"].append(targets)
            prediction_data["metadata"].append(metadata)
        
        # Set feature and target names
        prediction_data["feature_names"] = feature_names
        prediction_data["target_names"] = target_names
        
        return prediction_data
    
    def _prepare_model_data(
        self,
        prediction_data: Dict[str, Any]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[datetime.datetime], List[Dict[str, Any]]]:
        """
        Prepare data for model training and prediction.
        
        Args:
            prediction_data: Dictionary with extracted prediction data
            
        Returns:
            Tuple containing:
                - Feature matrix X (or None if insufficient data)
                - Target matrix y (or None if insufficient data)
                - List of timestamps
                - List of metadata dictionaries
        """
        # Check if we have features and targets
        if (not prediction_data or 
            "features" not in prediction_data or 
            "targets" not in prediction_data or
            not prediction_data["features"] or
            not prediction_data["targets"]):
            return None, None, [], []
        
        # Get data
        features = prediction_data["features"]
        targets = prediction_data["targets"]
        timestamps = prediction_data["timestamps"]
        metadata = prediction_data["metadata"]
        
        # Check if all features have the same length
        feature_lengths = [len(f) for f in features]
        if min(feature_lengths) != max(feature_lengths):
            logger.warning(f"Inconsistent feature lengths: min={min(feature_lengths)}, max={max(feature_lengths)}")
            return None, None, timestamps, metadata
        
        # Check if all targets have the same length
        target_lengths = [len(t) for t in targets]
        if min(target_lengths) != max(target_lengths):
            logger.warning(f"Inconsistent target lengths: min={min(target_lengths)}, max={max(target_lengths)}")
            return None, None, timestamps, metadata
        
        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(targets)
        
        # Replace NaN values
        X = np.nan_to_num(X, nan=0.0)
        y = np.nan_to_num(y, nan=0.0)
        
        return X, y, timestamps, metadata
    
    def _forecast_time_series(
        self,
        y: np.ndarray,
        timestamps: List[datetime.datetime],
        prediction_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform time series forecasting for accuracy metrics.
        
        Args:
            y: Target matrix
            timestamps: List of timestamps
            prediction_data: Dictionary with prediction data
            
        Returns:
            Dictionary with time series forecasting results
        """
        # Initialize results
        results = {
            "methods_used": [],
            "forecasts": {}
        }
        
        # Get time series methods to use
        methods = self.config["time_series_forecasting"]["methods"]
        forecast_horizon = self.config["time_series_forecasting"]["forecast_horizon"]
        interval_width = self.config["time_series_forecasting"]["interval_width"]
        seasonal_period = self.config["time_series_forecasting"]["seasonal_period"]
        
        # Get target names
        target_names = prediction_data["target_names"]
        
        # Create a time-indexed DataFrame for forecasting
        try:
            import pandas as pd
            
            # Create DataFrame
            df = pd.DataFrame()
            df['timestamp'] = timestamps
            df = df.set_index('timestamp')
            
            # Add target variables
            for i, name in enumerate(target_names):
                df[name] = y[:, i]
            
            # Sort by timestamp
            df = df.sort_index()
            
        except ImportError:
            # If pandas is not available, use simple list-based approach
            logger.warning("pandas not available, using simplified forecasting")
            
            # Sort data by timestamp
            sorted_indices = np.argsort(timestamps)
            sorted_y = y[sorted_indices]
            sorted_timestamps = [timestamps[i] for i in sorted_indices]
            
            # Create a simulated DataFrame
            df = {"index": sorted_timestamps}
            for i, name in enumerate(target_names):
                df[name] = sorted_y[:, i]
        
        # Apply ARIMA forecasting if enabled
        if "arima" in methods:
            try:
                arima_forecasts = self._forecast_arima(df, target_names, forecast_horizon)
                
                # Add to results
                results["methods_used"].append("arima")
                
                # Add forecasts for each target
                for target, forecast in arima_forecasts.items():
                    if target not in results["forecasts"]:
                        results["forecasts"][target] = {}
                    
                    results["forecasts"][target]["arima"] = forecast
                    
            except Exception as e:
                logger.warning(f"Error in ARIMA forecasting: {e}")
        
        # Apply exponential smoothing if enabled
        if "exponential_smoothing" in methods:
            try:
                ets_forecasts = self._forecast_exponential_smoothing(
                    df, target_names, forecast_horizon, seasonal_period)
                
                # Add to results
                results["methods_used"].append("exponential_smoothing")
                
                # Add forecasts for each target
                for target, forecast in ets_forecasts.items():
                    if target not in results["forecasts"]:
                        results["forecasts"][target] = {}
                    
                    results["forecasts"][target]["exponential_smoothing"] = forecast
                    
            except Exception as e:
                logger.warning(f"Error in exponential smoothing forecasting: {e}")
        
        # Apply Prophet forecasting if enabled
        if "prophet" in methods:
            try:
                prophet_forecasts = self._forecast_prophet(
                    df, target_names, forecast_horizon, interval_width)
                
                # Add to results
                results["methods_used"].append("prophet")
                
                # Add forecasts for each target
                for target, forecast in prophet_forecasts.items():
                    if target not in results["forecasts"]:
                        results["forecasts"][target] = {}
                    
                    results["forecasts"][target]["prophet"] = forecast
                    
            except Exception as e:
                logger.warning(f"Error in Prophet forecasting: {e}")
        
        return results
    
    def _forecast_arima(
        self,
        df: Union[Dict[str, List], Any],
        target_names: List[str],
        forecast_horizon: int
    ) -> Dict[str, Dict[str, Any]]:
        """
        Perform ARIMA forecasting for each target variable.
        
        Args:
            df: DataFrame with time series data
            target_names: List of target variable names
            forecast_horizon: Number of periods to forecast
            
        Returns:
            Dictionary mapping target names to forecast results
        """
        # Initialize results
        forecasts = {}
        
        # Try to import statsmodels for ARIMA
        try:
            import statsmodels.api as sm
            from statsmodels.tsa.arima.model import ARIMA
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            import pandas as pd
        except ImportError:
            logger.warning("statsmodels not available for ARIMA forecasting")
            return forecasts
        
        # Check if df is a dictionary (simple approach) or DataFrame
        is_dataframe = not isinstance(df, dict)
        
        # Process each target variable
        for target in target_names:
            try:
                # Extract time series
                if is_dataframe:
                    series = df[target]
                else:
                    series = df[target]
                
                # Auto-select ARIMA order
                try:
                    # Try auto_arima from pmdarima if available
                    from pmdarima import auto_arima
                    auto_model = auto_arima(
                        series,
                        start_p=0, start_q=0,
                        max_p=5, max_q=5,
                        seasonal=False,
                        stepwise=True,
                        suppress_warnings=True,
                        error_action='ignore'
                    )
                    arima_order = auto_model.order
                except ImportError:
                    # Fall back to simple default order
                    arima_order = (1, 1, 1)
                
                # Fit ARIMA model
                model = ARIMA(series, order=arima_order)
                model_fit = model.fit()
                
                # Forecast
                forecast = model_fit.forecast(steps=forecast_horizon)
                
                # Calculate prediction intervals
                pred_intervals = model_fit.get_forecast(steps=forecast_horizon).conf_int()
                lower = pred_intervals.iloc[:, 0].tolist() if is_dataframe else pred_intervals[:, 0].tolist()
                upper = pred_intervals.iloc[:, 1].tolist() if is_dataframe else pred_intervals[:, 1].tolist()
                
                # Convert forecast to list
                forecast_values = forecast.tolist() if is_dataframe else forecast.tolist()
                
                # Get forecast dates
                if is_dataframe:
                    last_date = df.index[-1]
                    forecast_dates = []
                    for i in range(1, forecast_horizon + 1):
                        if isinstance(last_date, pd.Timestamp):
                            # For daily data
                            next_date = last_date + pd.Timedelta(days=i)
                        else:
                            # For non-timestamp indices
                            next_date = last_date + i
                        forecast_dates.append(next_date)
                else:
                    # Simple approach for non-DataFrame case
                    forecast_dates = [i + 1 for i in range(forecast_horizon)]
                
                # Store forecast result
                forecasts[target] = {
                    "forecast": forecast_values,
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "forecast_horizon": forecast_horizon,
                    "model_params": {
                        "order": arima_order
                    }
                }
                
                # Add dates if available
                if is_dataframe:
                    forecasts[target]["forecast_dates"] = [
                        d.isoformat() if hasattr(d, 'isoformat') else str(d)
                        for d in forecast_dates
                    ]
                
            except Exception as e:
                logger.warning(f"Error forecasting {target} with ARIMA: {e}")
        
        return forecasts
    
    def _forecast_exponential_smoothing(
        self,
        df: Union[Dict[str, List], Any],
        target_names: List[str],
        forecast_horizon: int,
        seasonal_period: int
    ) -> Dict[str, Dict[str, Any]]:
        """
        Perform exponential smoothing forecasting for each target variable.
        
        Args:
            df: DataFrame with time series data
            target_names: List of target variable names
            forecast_horizon: Number of periods to forecast
            seasonal_period: Seasonal period for model
            
        Returns:
            Dictionary mapping target names to forecast results
        """
        # Initialize results
        forecasts = {}
        
        # Try to import statsmodels for exponential smoothing
        try:
            import statsmodels.api as sm
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            import pandas as pd
        except ImportError:
            logger.warning("statsmodels not available for exponential smoothing")
            return forecasts
        
        # Check if df is a dictionary (simple approach) or DataFrame
        is_dataframe = not isinstance(df, dict)
        
        # Process each target variable
        for target in target_names:
            try:
                # Extract time series
                if is_dataframe:
                    series = df[target]
                else:
                    series = df[target]
                
                # Check if we have enough data for seasonality
                use_seasonal = len(series) >= 2 * seasonal_period
                
                # Fit exponential smoothing model
                if use_seasonal:
                    model = ExponentialSmoothing(
                        series,
                        trend='add',
                        seasonal='add',
                        seasonal_periods=seasonal_period
                    )
                else:
                    model = ExponentialSmoothing(
                        series,
                        trend='add',
                        seasonal=None
                    )
                
                model_fit = model.fit()
                
                # Forecast
                forecast = model_fit.forecast(forecast_horizon)
                
                # Convert forecast to list
                forecast_values = forecast.tolist() if is_dataframe else forecast.tolist()
                
                # Get forecast dates
                if is_dataframe:
                    last_date = df.index[-1]
                    forecast_dates = []
                    for i in range(1, forecast_horizon + 1):
                        if isinstance(last_date, pd.Timestamp):
                            # For daily data
                            next_date = last_date + pd.Timedelta(days=i)
                        else:
                            # For non-timestamp indices
                            next_date = last_date + i
                        forecast_dates.append(next_date)
                else:
                    # Simple approach for non-DataFrame case
                    forecast_dates = [i + 1 for i in range(forecast_horizon)]
                
                # Store forecast result
                forecasts[target] = {
                    "forecast": forecast_values,
                    "forecast_horizon": forecast_horizon,
                    "model_params": {
                        "trend": "additive",
                        "seasonal": "additive" if use_seasonal else None,
                        "seasonal_period": seasonal_period if use_seasonal else None
                    }
                }
                
                # Add dates if available
                if is_dataframe:
                    forecasts[target]["forecast_dates"] = [
                        d.isoformat() if hasattr(d, 'isoformat') else str(d)
                        for d in forecast_dates
                    ]
                
                # Calculate prediction intervals (estimate based on forecast std)
                try:
                    # Estimate residual standard deviation
                    residuals = model_fit.resid
                    residual_std = np.std(residuals)
                    
                    # Calculate prediction intervals
                    z_value = 1.96  # 95% confidence interval
                    margin = z_value * residual_std
                    
                    lower = [max(0, f - margin) for f in forecast_values]
                    upper = [f + margin for f in forecast_values]
                    
                    forecasts[target]["lower_bound"] = lower
                    forecasts[target]["upper_bound"] = upper
                except:
                    # Skip intervals if calculation fails
                    pass
                
            except Exception as e:
                logger.warning(f"Error forecasting {target} with exponential smoothing: {e}")
        
        return forecasts
    
    def _forecast_prophet(
        self,
        df: Union[Dict[str, List], Any],
        target_names: List[str],
        forecast_horizon: int,
        interval_width: float
    ) -> Dict[str, Dict[str, Any]]:
        """
        Perform Prophet forecasting for each target variable.
        
        Args:
            df: DataFrame with time series data
            target_names: List of target variable names
            forecast_horizon: Number of periods to forecast
            interval_width: Prediction interval width (e.g., 0.95 for 95%)
            
        Returns:
            Dictionary mapping target names to forecast results
        """
        # Initialize results
        forecasts = {}
        
        # Try to import Prophet
        try:
            from prophet import Prophet
            import pandas as pd
        except ImportError:
            logger.warning("Prophet not available for forecasting")
            return forecasts
        
        # Check if df is a dictionary (simple approach) or DataFrame
        is_dataframe = not isinstance(df, dict)
        
        # Prophet requires a specific data format
        # Process each target variable
        for target in target_names:
            try:
                # Extract time series
                if is_dataframe:
                    # Create Prophet DataFrame
                    prophet_df = pd.DataFrame()
                    prophet_df['ds'] = df.index
                    prophet_df['y'] = df[target].values
                else:
                    # Simple approach for non-DataFrame case
                    prophet_df = pd.DataFrame()
                    prophet_df['ds'] = df["index"]
                    prophet_df['y'] = df[target]
                
                # Fit Prophet model
                model = Prophet(interval_width=interval_width)
                model.fit(prophet_df)
                
                # Create future DataFrame
                future = model.make_future_dataframe(periods=forecast_horizon)
                
                # Forecast
                forecast = model.predict(future)
                
                # Extract forecasted values (last forecast_horizon rows)
                forecast_values = forecast['yhat'].tail(forecast_horizon).tolist()
                lower_bound = forecast['yhat_lower'].tail(forecast_horizon).tolist()
                upper_bound = forecast['yhat_upper'].tail(forecast_horizon).tolist()
                forecast_dates = forecast['ds'].tail(forecast_horizon).tolist()
                
                # Store forecast result
                forecasts[target] = {
                    "forecast": forecast_values,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "forecast_horizon": forecast_horizon,
                    "model_params": {
                        "interval_width": interval_width
                    }
                }
                
                # Add dates
                forecasts[target]["forecast_dates"] = [
                    d.isoformat() if hasattr(d, 'isoformat') else str(d)
                    for d in forecast_dates
                ]
                
            except Exception as e:
                logger.warning(f"Error forecasting {target} with Prophet: {e}")
        
        return forecasts
    
    def _build_regression_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metadata: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Build regression models for predicting accuracy metrics.
        
        Args:
            X: Feature matrix
            y: Target matrix
            metadata: List of metadata dictionaries
            
        Returns:
            Dictionary with regression model results
        """
        # Initialize results
        results = {
            "methods_used": [],
            "models": {},
            "feature_importance": {},
            "predictions": {}
        }
        
        # Get regression methods to use
        methods = self.config["regression_model"]["methods"]
        validation_split = self.config["regression_model"]["validation_split"]
        random_state = self.config["regression_model"]["random_state"]
        
        # Try to import scikit-learn
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            logger.warning("scikit-learn not available for regression modeling")
            return {
                "status": "error",
                "message": "scikit-learn not available for regression modeling"
            }
        
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=random_state)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Build models for each target variable
        for j in range(y.shape[1]):
            # Get target name
            target_name = f"target_{j}"  # Fallback if actual name not available
            
            # Train models for this target
            target_models = {}
            
            # Random Forest Regressor
            if "random_forest" in methods:
                try:
                    rf_model = RandomForestRegressor(
                        n_estimators=100, random_state=random_state)
                    rf_model.fit(X_train_scaled, y_train[:, j])
                    
                    # Evaluate on validation set
                    y_pred = rf_model.predict(X_val_scaled)
                    
                    # Calculate metrics
                    mse = mean_squared_error(y_val[:, j], y_pred)
                    mae = mean_absolute_error(y_val[:, j], y_pred)
                    r2 = r2_score(y_val[:, j], y_pred)
                    
                    # Store model and metrics
                    target_models["random_forest"] = {
                        "model": rf_model,
                        "metrics": {
                            "mse": float(mse),
                            "mae": float(mae),
                            "r2": float(r2),
                            "rmse": float(np.sqrt(mse))
                        }
                    }
                    
                    # Add feature importance
                    feature_importance = rf_model.feature_importances_
                    
                    if target_name not in results["feature_importance"]:
                        results["feature_importance"][target_name] = {}
                    
                    results["feature_importance"][target_name]["random_forest"] = feature_importance.tolist()
                    
                    # Add to methods used
                    if "random_forest" not in results["methods_used"]:
                        results["methods_used"].append("random_forest")
                    
                except Exception as e:
                    logger.warning(f"Error building Random Forest model for {target_name}: {e}")
            
            # Gradient Boosting Regressor
            if "gradient_boosting" in methods:
                try:
                    gb_model = GradientBoostingRegressor(
                        n_estimators=100, random_state=random_state)
                    gb_model.fit(X_train_scaled, y_train[:, j])
                    
                    # Evaluate on validation set
                    y_pred = gb_model.predict(X_val_scaled)
                    
                    # Calculate metrics
                    mse = mean_squared_error(y_val[:, j], y_pred)
                    mae = mean_absolute_error(y_val[:, j], y_pred)
                    r2 = r2_score(y_val[:, j], y_pred)
                    
                    # Store model and metrics
                    target_models["gradient_boosting"] = {
                        "model": gb_model,
                        "metrics": {
                            "mse": float(mse),
                            "mae": float(mae),
                            "r2": float(r2),
                            "rmse": float(np.sqrt(mse))
                        }
                    }
                    
                    # Add feature importance
                    feature_importance = gb_model.feature_importances_
                    
                    if target_name not in results["feature_importance"]:
                        results["feature_importance"][target_name] = {}
                    
                    results["feature_importance"][target_name]["gradient_boosting"] = feature_importance.tolist()
                    
                    # Add to methods used
                    if "gradient_boosting" not in results["methods_used"]:
                        results["methods_used"].append("gradient_boosting")
                    
                except Exception as e:
                    logger.warning(f"Error building Gradient Boosting model for {target_name}: {e}")
            
            # Linear Regression
            if "linear" in methods:
                try:
                    lr_model = LinearRegression()
                    lr_model.fit(X_train_scaled, y_train[:, j])
                    
                    # Evaluate on validation set
                    y_pred = lr_model.predict(X_val_scaled)
                    
                    # Calculate metrics
                    mse = mean_squared_error(y_val[:, j], y_pred)
                    mae = mean_absolute_error(y_val[:, j], y_pred)
                    r2 = r2_score(y_val[:, j], y_pred)
                    
                    # Store model and metrics
                    target_models["linear"] = {
                        "model": lr_model,
                        "metrics": {
                            "mse": float(mse),
                            "mae": float(mae),
                            "r2": float(r2),
                            "rmse": float(np.sqrt(mse))
                        }
                    }
                    
                    # Add coefficients
                    coefficients = lr_model.coef_
                    
                    if target_name not in results["feature_importance"]:
                        results["feature_importance"][target_name] = {}
                    
                    results["feature_importance"][target_name]["linear"] = coefficients.tolist()
                    
                    # Add to methods used
                    if "linear" not in results["methods_used"]:
                        results["methods_used"].append("linear")
                    
                except Exception as e:
                    logger.warning(f"Error building Linear Regression model for {target_name}: {e}")
            
            # Store models for this target
            results["models"][target_name] = target_models
            
            # Select best model for predictions
            best_model = None
            best_method = None
            best_r2 = -float('inf')
            
            for method, model_info in target_models.items():
                r2 = model_info["metrics"]["r2"]
                if r2 > best_r2:
                    best_r2 = r2
                    best_model = model_info["model"]
                    best_method = method
            
            # Make predictions if we have a best model
            if best_model is not None:
                # Use the best model to predict on the entire dataset
                X_scaled = scaler.transform(X)
                y_pred = best_model.predict(X_scaled)
                
                # Store predictions
                results["predictions"][target_name] = {
                    "method": best_method,
                    "values": y_pred.tolist(),
                    "r2": float(best_r2)
                }
        
        return results
    
    def _build_classification_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metadata: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Build classification models for predicting accuracy thresholds.
        
        Args:
            X: Feature matrix
            y: Target matrix
            metadata: List of metadata dictionaries
            
        Returns:
            Dictionary with classification model results
        """
        # Initialize results
        results = {
            "methods_used": [],
            "models": {},
            "feature_importance": {},
            "predictions": {}
        }
        
        # Get classification methods to use
        methods = self.config["classification_model"]["methods"]
        threshold = self.config["classification_model"]["threshold"]
        validation_split = self.config["classification_model"]["validation_split"]
        random_state = self.config["classification_model"]["random_state"]
        
        # Try to import scikit-learn
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            logger.warning("scikit-learn not available for classification modeling")
            return {
                "status": "error",
                "message": "scikit-learn not available for classification modeling"
            }
        
        # Convert regression targets to binary classification
        # For accuracy metrics, lower values are better (e.g., MAPE)
        y_binary = np.zeros_like(y, dtype=int)
        for j in range(y.shape[1]):
            y_binary[:, j] = (y[:, j] <= threshold).astype(int)
        
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_binary, test_size=validation_split, random_state=random_state)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Build models for each target variable
        for j in range(y.shape[1]):
            # Get target name
            target_name = f"target_{j}"  # Fallback if actual name not available
            
            # Train models for this target
            target_models = {}
            
            # Random Forest Classifier
            if "random_forest" in methods:
                try:
                    rf_model = RandomForestClassifier(
                        n_estimators=100, random_state=random_state)
                    rf_model.fit(X_train_scaled, y_train[:, j])
                    
                    # Evaluate on validation set
                    y_pred = rf_model.predict(X_val_scaled)
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_val[:, j], y_pred)
                    precision = precision_score(y_val[:, j], y_pred, zero_division=0)
                    recall = recall_score(y_val[:, j], y_pred, zero_division=0)
                    f1 = f1_score(y_val[:, j], y_pred, zero_division=0)
                    
                    # Store model and metrics
                    target_models["random_forest"] = {
                        "model": rf_model,
                        "metrics": {
                            "accuracy": float(accuracy),
                            "precision": float(precision),
                            "recall": float(recall),
                            "f1": float(f1)
                        }
                    }
                    
                    # Add feature importance
                    feature_importance = rf_model.feature_importances_
                    
                    if target_name not in results["feature_importance"]:
                        results["feature_importance"][target_name] = {}
                    
                    results["feature_importance"][target_name]["random_forest"] = feature_importance.tolist()
                    
                    # Add to methods used
                    if "random_forest" not in results["methods_used"]:
                        results["methods_used"].append("random_forest")
                    
                except Exception as e:
                    logger.warning(f"Error building Random Forest classifier for {target_name}: {e}")
            
            # Gradient Boosting Classifier
            if "gradient_boosting" in methods:
                try:
                    gb_model = GradientBoostingClassifier(
                        n_estimators=100, random_state=random_state)
                    gb_model.fit(X_train_scaled, y_train[:, j])
                    
                    # Evaluate on validation set
                    y_pred = gb_model.predict(X_val_scaled)
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_val[:, j], y_pred)
                    precision = precision_score(y_val[:, j], y_pred, zero_division=0)
                    recall = recall_score(y_val[:, j], y_pred, zero_division=0)
                    f1 = f1_score(y_val[:, j], y_pred, zero_division=0)
                    
                    # Store model and metrics
                    target_models["gradient_boosting"] = {
                        "model": gb_model,
                        "metrics": {
                            "accuracy": float(accuracy),
                            "precision": float(precision),
                            "recall": float(recall),
                            "f1": float(f1)
                        }
                    }
                    
                    # Add feature importance
                    feature_importance = gb_model.feature_importances_
                    
                    if target_name not in results["feature_importance"]:
                        results["feature_importance"][target_name] = {}
                    
                    results["feature_importance"][target_name]["gradient_boosting"] = feature_importance.tolist()
                    
                    # Add to methods used
                    if "gradient_boosting" not in results["methods_used"]:
                        results["methods_used"].append("gradient_boosting")
                    
                except Exception as e:
                    logger.warning(f"Error building Gradient Boosting classifier for {target_name}: {e}")
            
            # Logistic Regression
            if "logistic" in methods:
                try:
                    lr_model = LogisticRegression(random_state=random_state)
                    lr_model.fit(X_train_scaled, y_train[:, j])
                    
                    # Evaluate on validation set
                    y_pred = lr_model.predict(X_val_scaled)
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_val[:, j], y_pred)
                    precision = precision_score(y_val[:, j], y_pred, zero_division=0)
                    recall = recall_score(y_val[:, j], y_pred, zero_division=0)
                    f1 = f1_score(y_val[:, j], y_pred, zero_division=0)
                    
                    # Store model and metrics
                    target_models["logistic"] = {
                        "model": lr_model,
                        "metrics": {
                            "accuracy": float(accuracy),
                            "precision": float(precision),
                            "recall": float(recall),
                            "f1": float(f1)
                        }
                    }
                    
                    # Add coefficients
                    coefficients = lr_model.coef_[0]
                    
                    if target_name not in results["feature_importance"]:
                        results["feature_importance"][target_name] = {}
                    
                    results["feature_importance"][target_name]["logistic"] = coefficients.tolist()
                    
                    # Add to methods used
                    if "logistic" not in results["methods_used"]:
                        results["methods_used"].append("logistic")
                    
                except Exception as e:
                    logger.warning(f"Error building Logistic Regression for {target_name}: {e}")
            
            # Store models for this target
            results["models"][target_name] = target_models
            
            # Select best model for predictions
            best_model = None
            best_method = None
            best_f1 = -float('inf')
            
            for method, model_info in target_models.items():
                f1 = model_info["metrics"]["f1"]
                if f1 > best_f1:
                    best_f1 = f1
                    best_model = model_info["model"]
                    best_method = method
            
            # Make predictions if we have a best model
            if best_model is not None:
                # Use the best model to predict on the entire dataset
                X_scaled = scaler.transform(X)
                y_pred_proba = best_model.predict_proba(X_scaled)[:, 1]  # Probability of class 1
                
                # Store predictions
                results["predictions"][target_name] = {
                    "method": best_method,
                    "values": y_pred_proba.tolist(),
                    "threshold": threshold,
                    "f1": float(best_f1)
                }
        
        return results
    
    def _build_ensemble_prediction(
        self,
        forecasts: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Build ensemble predictions by combining multiple prediction methods.
        
        Args:
            forecasts: Dictionary with forecasts from different methods
            
        Returns:
            Dictionary with ensemble prediction results
        """
        # Initialize results
        results = {
            "methods_used": [],
            "forecasts": {}
        }
        
        # Get ensemble methods to use
        methods = self.config["ensemble_prediction"]["methods"]
        weights = self.config["ensemble_prediction"]["weights"]
        
        # Process each target variable
        for target, target_forecasts in forecasts.items():
            # Skip if no forecasts available
            if not target_forecasts:
                continue
            
            # Initialize ensemble forecasts
            ensemble_forecast = {}
            
            # Apply voting ensemble if enabled
            if "voting" in methods:
                try:
                    # Get available methods
                    available_methods = []
                    method_weights = {}
                    
                    # Check time series forecasting
                    if "time_series" in target_forecasts:
                        ts_methods = [m for m in target_forecasts["time_series"] if m != "forecast_dates"]
                        for method in ts_methods:
                            available_methods.append(("time_series", method))
                            method_weights[("time_series", method)] = weights.get("time_series", 0.4) / len(ts_methods)
                    
                    # Check regression
                    if "regression" in target_forecasts:
                        available_methods.append(("regression", "regression"))
                        method_weights[("regression", "regression")] = weights.get("regression", 0.4)
                    
                    # Check classification
                    if "classification" in target_forecasts:
                        available_methods.append(("classification", "classification"))
                        method_weights[("classification", "classification")] = weights.get("classification", 0.2)
                    
                    # Skip if no methods available
                    if not available_methods:
                        continue
                    
                    # Normalize weights
                    total_weight = sum(method_weights.values())
                    if total_weight > 0:
                        for key in method_weights:
                            method_weights[key] = method_weights[key] / total_weight
                    
                    # Create weighted average ensemble
                    ensemble_values = None
                    weighted_sum = None
                    
                    for method_type, method_name in available_methods:
                        # Get forecast values
                        if method_type == "time_series":
                            forecast_values = target_forecasts["time_series"].get(method_name, {}).get("forecast", [])
                        else:
                            forecast_values = target_forecasts[method_type].get("values", [])
                        
                        # Skip if no forecast values
                        if not forecast_values:
                            continue
                        
                        # Get weight
                        weight = method_weights.get((method_type, method_name), 0.0)
                        
                        # Add to weighted sum
                        if weighted_sum is None:
                            weighted_sum = np.array(forecast_values) * weight
                            ensemble_values = np.array(forecast_values)
                        else:
                            # Ensure same length
                            min_length = min(len(weighted_sum), len(forecast_values))
                            weighted_sum = weighted_sum[:min_length] + np.array(forecast_values)[:min_length] * weight
                            ensemble_values = ensemble_values[:min_length]
                    
                    # Check if we have ensemble values
                    if ensemble_values is not None and weighted_sum is not None:
                        # Store ensemble forecast
                        ensemble_forecast = {
                            "forecast": weighted_sum.tolist(),
                            "methods": available_methods,
                            "weights": {str(k): v for k, v in method_weights.items()}
                        }
                        
                        # Add forecast dates if available
                        if "time_series" in target_forecasts and "forecast_dates" in target_forecasts["time_series"]:
                            ensemble_forecast["forecast_dates"] = target_forecasts["time_series"]["forecast_dates"]
                        
                        # Add to methods used
                        if "voting" not in results["methods_used"]:
                            results["methods_used"].append("voting")
                
                except Exception as e:
                    logger.warning(f"Error building voting ensemble for {target}: {e}")
            
            # Apply stacking ensemble if enabled
            # (For a complete implementation, stacking would involve training a meta-model)
            
            # Store ensemble forecast
            if ensemble_forecast:
                results["forecasts"][target] = ensemble_forecast
        
        return results
    
    def _evaluate_models(
        self,
        method_results: Dict[str, Dict[str, Any]],
        X: np.ndarray,
        y: np.ndarray,
        timestamps: List[datetime.datetime]
    ) -> Dict[str, Any]:
        """
        Evaluate and compare predictive models.
        
        Args:
            method_results: Results from different prediction methods
            X: Feature matrix
            y: Target matrix
            timestamps: List of timestamps
            
        Returns:
            Dictionary with model evaluation results
        """
        # Initialize results
        evaluation_results = {
            "best_models": {},
            "comparison": {}
        }
        
        # Get evaluation metrics
        eval_metrics = self.config["model_evaluation"]["metrics"]
        
        # Check time series forecasting results
        if "time_series_forecasting" in method_results:
            ts_results = method_results["time_series_forecasting"]
            
            # Extract target metrics
            for target, forecasts in ts_results.get("forecasts", {}).items():
                evaluation_results["comparison"][target] = {}
                
                # Compare different time series methods
                for method, forecast in forecasts.items():
                    if method != "forecast_dates":
                        forecast_values = forecast.get("forecast", [])
                        
                        evaluation_results["comparison"][target][method] = {
                            "forecast_values": forecast_values[:5]  # Include first 5 forecast values
                        }
        
        # Check regression model results
        if "regression_model" in method_results:
            reg_results = method_results["regression_model"]
            
            # Extract target metrics
            for target, models in reg_results.get("models", {}).items():
                if target not in evaluation_results["comparison"]:
                    evaluation_results["comparison"][target] = {}
                
                # Compare different regression methods
                for method, model_info in models.items():
                    metrics = model_info.get("metrics", {})
                    
                    evaluation_results["comparison"][target][f"regression_{method}"] = {
                        "metrics": metrics
                    }
        
        # Check classification model results
        if "classification_model" in method_results:
            cls_results = method_results["classification_model"]
            
            # Extract target metrics
            for target, models in cls_results.get("models", {}).items():
                if target not in evaluation_results["comparison"]:
                    evaluation_results["comparison"][target] = {}
                
                # Compare different classification methods
                for method, model_info in models.items():
                    metrics = model_info.get("metrics", {})
                    
                    evaluation_results["comparison"][target][f"classification_{method}"] = {
                        "metrics": metrics
                    }
        
        # Determine best models for each target
        for target, comparison in evaluation_results["comparison"].items():
            # Initialize best model scoring
            best_score = -float('inf')
            best_model = None
            
            # Score each model
            for model_name, model_info in comparison.items():
                # Score based on available metrics
                score = 0.0
                
                # Regression metrics
                if "metrics" in model_info:
                    metrics = model_info["metrics"]
                    
                    if "r2" in metrics:
                        score += metrics["r2"] * 2.0  # Higher weight for R2
                    
                    if "rmse" in metrics:
                        # Lower RMSE is better, so use negative value
                        score -= metrics["rmse"] * 0.5
                
                # Classification metrics
                if "metrics" in model_info:
                    metrics = model_info["metrics"]
                    
                    if "f1" in metrics:
                        score += metrics["f1"] * 1.5  # Higher weight for F1
                    
                    if "accuracy" in metrics:
                        score += metrics["accuracy"]
                
                # Update best model if score is better
                if score > best_score:
                    best_score = score
                    best_model = model_name
            
            # Store best model
            if best_model is not None:
                evaluation_results["best_models"][target] = {
                    "model": best_model,
                    "score": float(best_score)
                }
        
        return evaluation_results
    
    def _generate_key_findings(
        self,
        forecasts: Dict[str, Dict[str, Any]],
        model_evaluation: Dict[str, Any],
        prediction_data: Dict[str, Any]
    ) -> List[str]:
        """
        Generate key findings based on predictive modeling results.
        
        Args:
            forecasts: Dictionary with forecast results
            model_evaluation: Dictionary with model evaluation results
            prediction_data: Dictionary with prediction data
            
        Returns:
            List of key findings
        """
        findings = []
        
        # Skip if no forecasts
        if not forecasts:
            findings.append("Insufficient data for predictive modeling")
            return findings
        
        # Extract target names
        target_names = prediction_data.get("target_names", [])
        
        # Analyze forecasts for each target
        for target, methods in forecasts.items():
            # Get target name (use index or name if available)
            try:
                target_index = int(target.split("_")[1])
                target_name = target_names[target_index] if target_index < len(target_names) else target
            except:
                target_name = target
            
            # Get ensemble forecast if available
            if "ensemble" in methods:
                ensemble = methods["ensemble"]
                forecast_values = ensemble.get("forecast", [])
                
                if forecast_values:
                    # Analyze trend direction
                    if len(forecast_values) >= 2:
                        first_value = forecast_values[0]
                        last_value = forecast_values[-1]
                        
                        # Calculate percent change
                        if first_value != 0:
                            percent_change = (last_value - first_value) / abs(first_value) * 100
                            direction = "improving" if percent_change < 0 else "degrading"
                            
                            # Add finding
                            findings.append(
                                f"Forecast shows {direction} trend for {target_name} "
                                f"({percent_change:.1f}% change over forecast period)"
                            )
            
            # Get time series forecast if available
            elif "time_series" in methods:
                # Find best time series method
                best_method = None
                forecast_values = None
                
                for method, forecast in methods["time_series"].items():
                    if method != "forecast_dates":
                        # Use first available method
                        if forecast_values is None:
                            best_method = method
                            forecast_values = forecast.get("forecast", [])
                
                if forecast_values:
                    # Analyze trend direction
                    if len(forecast_values) >= 2:
                        first_value = forecast_values[0]
                        last_value = forecast_values[-1]
                        
                        # Calculate percent change
                        if first_value != 0:
                            percent_change = (last_value - first_value) / abs(first_value) * 100
                            direction = "improving" if percent_change < 0 else "degrading"
                            
                            # Add finding
                            findings.append(
                                f"{best_method.capitalize()} forecast shows {direction} trend for {target_name} "
                                f"({percent_change:.1f}% change over forecast period)"
                            )
            
            # Get regression prediction if available
            elif "regression" in methods:
                # Regression predictions are for the existing data points
                # We can't easily extract trend information without additional context
                pass
        
        # Add findings about best models
        if "best_models" in model_evaluation:
            for target, model_info in model_evaluation["best_models"].items():
                # Get target name (use index or name if available)
                try:
                    target_index = int(target.split("_")[1])
                    target_name = target_names[target_index] if target_index < len(target_names) else target
                except:
                    target_name = target
                
                best_model = model_info.get("model", "unknown")
                
                findings.append(
                    f"Best predictive model for {target_name} is {best_model}"
                )
        
        # Add findings about feature importance
        feature_importance = {}
        
        # Check regression model feature importance
        for method_name, method_results in {
            "regression_model": "regression", 
            "classification_model": "classification"
        }.items():
            if method_name in feature_importance:
                for target, importance_dict in feature_importance[method_name].get("feature_importance", {}).items():
                    # Find method with highest score
                    best_method = None
                    best_importance = None
                    
                    for method, importance in importance_dict.items():
                        if best_importance is None:
                            best_method = method
                            best_importance = importance
                    
                    if best_importance is not None:
                        # Get feature names
                        feature_names = prediction_data.get("feature_names", [])
                        
                        # Get top features
                        if len(feature_names) == len(best_importance):
                            # Create feature importance pairs
                            importance_pairs = list(zip(feature_names, best_importance))
                            
                            # Sort by absolute importance
                            importance_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
                            
                            # Get top 3 features
                            top_features = importance_pairs[:3]
                            
                            # Add finding
                            features_str = ", ".join([f"{name} ({importance:.2f})" 
                                                     for name, importance in top_features])
                            
                            findings.append(
                                f"Top features for predicting {target} using {method_results}: {features_str}"
                            )
        
        # Ensure at least one finding
        if not findings:
            findings.append("Predictive models built successfully, but no clear trends identified")
        
        return findings
    
    def _generate_recommendations(
        self,
        forecasts: Dict[str, Dict[str, Any]],
        model_evaluation: Dict[str, Any],
        key_findings: List[str]
    ) -> List[str]:
        """
        Generate recommendations based on predictive modeling results.
        
        Args:
            forecasts: Dictionary with forecast results
            model_evaluation: Dictionary with model evaluation results
            key_findings: List of key findings
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Skip if no forecasts
        if not forecasts:
            recommendations.append("Collect more validation data to enable accurate forecasting")
            return recommendations
        
        # Track metrics with degrading trends
        degrading_metrics = []
        
        # Analyze forecasts for potential issues
        for target, methods in forecasts.items():
            # Get ensemble forecast if available
            if "ensemble" in methods:
                ensemble = methods["ensemble"]
                forecast_values = ensemble.get("forecast", [])
                
                if forecast_values and len(forecast_values) >= 2:
                    first_value = forecast_values[0]
                    last_value = forecast_values[-1]
                    
                    # Calculate percent change
                    if first_value != 0:
                        percent_change = (last_value - first_value) / abs(first_value) * 100
                        
                        # Check for degrading trend
                        if percent_change > 10:  # More than 10% degradation
                            degrading_metrics.append((target, percent_change))
            
            # Get time series forecast if no ensemble
            elif "time_series" in methods:
                # Find best time series method
                forecast_values = None
                
                for method, forecast in methods["time_series"].items():
                    if method != "forecast_dates":
                        # Use first available method
                        if forecast_values is None:
                            forecast_values = forecast.get("forecast", [])
                
                if forecast_values and len(forecast_values) >= 2:
                    first_value = forecast_values[0]
                    last_value = forecast_values[-1]
                    
                    # Calculate percent change
                    if first_value != 0:
                        percent_change = (last_value - first_value) / abs(first_value) * 100
                        
                        # Check for degrading trend
                        if percent_change > 10:  # More than 10% degradation
                            degrading_metrics.append((target, percent_change))
        
        # Add recommendations based on degrading metrics
        if degrading_metrics:
            # Sort by percent change (highest first)
            degrading_metrics.sort(key=lambda x: x[1], reverse=True)
            
            # Add recommendation for most degrading metric
            worst_metric, worst_change = degrading_metrics[0]
            recommendations.append(
                f"Prioritize improvement of {worst_metric} as it shows the strongest "
                f"degrading trend ({worst_change:.1f}% forecasted degradation)"
            )
            
            # Add general recommendation if multiple metrics are degrading
            if len(degrading_metrics) > 1:
                metrics_list = ", ".join([m[0] for m in degrading_metrics])
                recommendations.append(
                    f"Establish monitoring alerts for these metrics with degrading trends: {metrics_list}"
                )
        
        # Add recommendations based on model evaluation
        if "comparison" in model_evaluation:
            low_r2_metrics = []
            
            for target, model_comparison in model_evaluation["comparison"].items():
                for model_name, model_info in model_comparison.items():
                    if "metrics" in model_info and "r2" in model_info["metrics"]:
                        r2 = model_info["metrics"]["r2"]
                        
                        # Check for low R2 (poor prediction quality)
                        if r2 < 0.5:
                            low_r2_metrics.append((target, r2))
                            break  # Only consider one model per target
            
            if low_r2_metrics:
                # Sort by R2 (lowest first)
                low_r2_metrics.sort(key=lambda x: x[1])
                
                # Add recommendation for metric with lowest R2
                worst_metric, worst_r2 = low_r2_metrics[0]
                recommendations.append(
                    f"Collect more diverse validation data for {worst_metric} "
                    f"as current predictive models have low accuracy (R² = {worst_r2:.2f})"
                )
        
        # Add recommendation about best model if available
        if "best_models" in model_evaluation and model_evaluation["best_models"]:
            best_model_types = {}
            
            for target, model_info in model_evaluation["best_models"].items():
                model_name = model_info.get("model", "")
                
                if "regression" in model_name:
                    model_type = "regression"
                elif "classification" in model_name:
                    model_type = "classification"
                else:
                    model_type = "time_series"
                
                if model_type not in best_model_types:
                    best_model_types[model_type] = 0
                
                best_model_types[model_type] += 1
            
            # Find most common best model type
            if best_model_types:
                best_type = max(best_model_types.items(), key=lambda x: x[1])[0]
                
                if best_type == "regression":
                    recommendations.append(
                        "Focus on feature-based models for future accuracy predictions, "
                        "as they perform best with the current data"
                    )
                elif best_type == "classification":
                    recommendations.append(
                        "Consider using classification models for threshold-based "
                        "accuracy predictions in monitoring systems"
                    )
                else:  # time_series
                    recommendations.append(
                        "Use time series forecasting for future accuracy predictions, "
                        "as they perform best with the current data"
                    )
        
        # Add general recommendations if specific ones are limited
        if len(recommendations) < 2:
            recommendations.append(
                "Set up regular retraining of predictive models as more "
                "validation data becomes available"
            )
            
            recommendations.append(
                "Integrate predictive models into monitoring systems to "
                "proactively detect accuracy degradation"
            )
        
        return recommendations