"""
Cross-validation system for calibration parameter tuning.

This module provides functionality for cross-validating calibration parameters
to ensure they generalize well to unseen simulation scenarios.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional, Union, Callable
import json
from datetime import datetime
import os
import uuid
from sklearn.model_selection import KFold, ShuffleSplit
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

# Import calibrators
from .basic_calibrator import BasicCalibrator
from .advanced_calibrator import AdvancedCalibrator, MultiParameterCalibrator

# Setup logger
logger = logging.getLogger(__name__)

class CalibrationCrossValidator:
    """
    Performs cross-validation for calibration parameter tuning.
    
    This class validates calibration parameters by splitting simulation and hardware
    results into training and validation sets, and evaluating performance on the
    validation set to ensure parameters generalize well.
    """
    
    def __init__(
        self, 
        n_splits: int = 5,
        test_size: float = 0.2,
        random_state: int = 42,
        result_file: Optional[str] = None,
        calibrator_type: str = "basic"
    ):
        """
        Initialize the CalibrationCrossValidator.
        
        Args:
            n_splits: Number of cross-validation splits
            test_size: Proportion of data to use for validation
            random_state: Random seed for reproducibility
            result_file: Optional path to file for storing validation results
            calibrator_type: Type of calibrator to use ("basic", "multi_parameter", etc.)
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.random_state = random_state
        self.result_file = result_file
        self.calibrator_type = calibrator_type
        self.results = []
        
        # Load previous results if available
        if self.result_file and os.path.exists(self.result_file):
            try:
                with open(self.result_file, 'r') as f:
                    self.results = json.load(f)
                logger.info(f"Loaded {len(self.results)} cross-validation results from file")
            except Exception as e:
                logger.error(f"Error loading cross-validation results: {str(e)}")
                self.results = []
    
    def cross_validate(
        self, 
        simulation_results: List[Dict[str, Any]], 
        hardware_results: List[Dict[str, Any]],
        initial_parameters: Dict[str, Any],
        parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        group_by: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform cross-validation for calibration parameter tuning.
        
        Args:
            simulation_results: List of simulation result dictionaries
            hardware_results: List of hardware result dictionaries
            initial_parameters: Initial parameter dictionary
            parameter_bounds: Optional dictionary mapping parameter names to (min, max) tuples
            group_by: Optional field to group results by (e.g., "hardware_id", "model_id")
            
        Returns:
            Dictionary with cross-validation results
        """
        logger.info(f"Starting cross-validation with {self.n_splits} splits")
        
        if not simulation_results or not hardware_results:
            logger.warning("Empty simulation or hardware results provided")
            return {
                "error": "Empty simulation or hardware results provided",
                "status": "failed",
                "timestamp": datetime.now().isoformat()
            }
        
        # Ensure simulation and hardware results match up
        if len(simulation_results) != len(hardware_results):
            logger.warning(f"Length mismatch: simulation_results={len(simulation_results)}, "
                         f"hardware_results={len(hardware_results)}")
            return {
                "error": "Length mismatch between simulation and hardware results",
                "status": "failed",
                "timestamp": datetime.now().isoformat()
            }
        
        # If grouping is requested, validate and prepare data
        if group_by:
            return self._cross_validate_grouped(
                simulation_results, hardware_results, initial_parameters, 
                parameter_bounds, group_by
            )
        
        # Create instances of the calibrator
        calibrator = self._create_calibrator()
        
        # Setup cross-validation splitter
        cv = ShuffleSplit(
            n_splits=self.n_splits, 
            test_size=self.test_size, 
            random_state=self.random_state
        )
        
        # Store fold results
        fold_results = []
        all_train_errors = []
        all_val_errors = []
        all_parameters = []
        
        # Create indices for cross-validation
        indices = np.arange(len(simulation_results))
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(indices)):
            logger.info(f"Processing fold {fold+1}/{self.n_splits}")
            
            # Extract training and validation data
            train_sim = [simulation_results[i] for i in train_idx]
            train_hw = [hardware_results[i] for i in train_idx]
            val_sim = [simulation_results[i] for i in val_idx]
            val_hw = [hardware_results[i] for i in val_idx]
            
            # Calibrate on training data
            calibrated_params = calibrator.calibrate(
                train_sim, train_hw, initial_parameters, parameter_bounds
            )
            
            # Calculate errors on training and validation sets
            train_error = self._calculate_error(train_sim, train_hw, calibrated_params)
            val_error = self._calculate_error(val_sim, val_hw, calibrated_params)
            
            # Store results
            fold_result = {
                "fold": fold + 1,
                "train_size": len(train_idx),
                "val_size": len(val_idx),
                "train_error": float(train_error),
                "val_error": float(val_error),
                "calibrated_parameters": calibrated_params
            }
            fold_results.append(fold_result)
            
            all_train_errors.append(train_error)
            all_val_errors.append(val_error)
            all_parameters.append(calibrated_params)
            
            logger.info(f"Fold {fold+1} train error: {train_error:.6f}, val error: {val_error:.6f}")
        
        # Calculate average errors
        avg_train_error = np.mean(all_train_errors)
        avg_val_error = np.mean(all_val_errors)
        train_error_std = np.std(all_train_errors)
        val_error_std = np.std(all_val_errors)
        
        # Calculate generalization gap
        generalization_gap = avg_val_error - avg_train_error
        generalization_gap_percentage = (generalization_gap / avg_train_error * 100 
                                       if avg_train_error > 0 else 0.0)
        
        # Determine if the calibration generalizes well
        generalizes_well = generalization_gap_percentage < 20.0  # Less than 20% performance drop
        
        # Calculate parameter stability
        param_stability = self._calculate_parameter_stability(all_parameters)
        
        # Create validation result summary
        result = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "calibrator_type": self.calibrator_type,
            "n_splits": self.n_splits,
            "test_size": self.test_size,
            "random_state": self.random_state,
            "num_samples": len(simulation_results),
            "initial_parameters": initial_parameters,
            "fold_results": fold_results,
            "avg_train_error": float(avg_train_error),
            "avg_val_error": float(avg_val_error),
            "train_error_std": float(train_error_std),
            "val_error_std": float(val_error_std),
            "generalization_gap": float(generalization_gap),
            "generalization_gap_percentage": float(generalization_gap_percentage),
            "generalizes_well": generalizes_well,
            "parameter_stability": param_stability,
            "status": "success"
        }
        
        # Calculate recommended parameters (average of all folds)
        recommended_params = self._calculate_recommended_parameters(all_parameters, param_stability)
        result["recommended_parameters"] = recommended_params
        
        # Calculate error with recommended parameters
        recommended_error = self._calculate_error(
            simulation_results, hardware_results, recommended_params
        )
        initial_error = self._calculate_error(
            simulation_results, hardware_results, initial_parameters
        )
        
        result["recommended_error"] = float(recommended_error)
        result["initial_error"] = float(initial_error)
        result["improvement"] = float((initial_error - recommended_error) / initial_error * 100 
                                    if initial_error > 0 else 0.0)
        
        logger.info(f"Cross-validation completed successfully")
        logger.info(f"Average train error: {avg_train_error:.6f}, val error: {avg_val_error:.6f}")
        logger.info(f"Generalization gap: {generalization_gap_percentage:.2f}%")
        logger.info(f"Initial error: {initial_error:.6f}, recommended error: {recommended_error:.6f}, "
                    f"improvement: {result['improvement']:.2f}%")
        
        # Store result
        self.results.append(result)
        self._save_results()
        
        return result
    
    def _cross_validate_grouped(
        self, 
        simulation_results: List[Dict[str, Any]], 
        hardware_results: List[Dict[str, Any]],
        initial_parameters: Dict[str, Any],
        parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        group_by: str = None
    ) -> Dict[str, Any]:
        """
        Perform cross-validation with grouping.
        
        Args:
            simulation_results: List of simulation result dictionaries
            hardware_results: List of hardware result dictionaries
            initial_parameters: Initial parameter dictionary
            parameter_bounds: Optional dictionary mapping parameter names to (min, max) tuples
            group_by: Field to group results by (e.g., "hardware_id", "model_id")
            
        Returns:
            Dictionary with cross-validation results
        """
        logger.info(f"Starting grouped cross-validation by {group_by}")
        
        # Validate group_by field exists in both simulation and hardware results
        if not all(group_by in r for r in simulation_results):
            logger.warning(f"Group field '{group_by}' not found in all simulation results")
            return {
                "error": f"Group field '{group_by}' not found in all simulation results",
                "status": "failed",
                "timestamp": datetime.now().isoformat()
            }
        
        if not all(group_by in r for r in hardware_results):
            logger.warning(f"Group field '{group_by}' not found in all hardware results")
            return {
                "error": f"Group field '{group_by}' not found in all hardware results",
                "status": "failed",
                "timestamp": datetime.now().isoformat()
            }
        
        # Group the results
        groups = {}
        for i in range(len(simulation_results)):
            sim_result = simulation_results[i]
            hw_result = hardware_results[i]
            
            group_value = sim_result[group_by]
            if group_value != hw_result[group_by]:
                logger.warning(f"Group mismatch at index {i}: "
                             f"simulation {group_value} != hardware {hw_result[group_by]}")
                continue
                
            if group_value not in groups:
                groups[group_value] = {"sim": [], "hw": []}
                
            groups[group_value]["sim"].append(sim_result)
            groups[group_value]["hw"].append(hw_result)
        
        # Validate we have enough groups
        if len(groups) < 2:
            logger.warning(f"Not enough groups for cross-validation: {len(groups)} groups")
            return {
                "error": f"Not enough groups for cross-validation: {len(groups)} groups",
                "status": "failed",
                "timestamp": datetime.now().isoformat()
            }
        
        # Setup leave-one-group-out cross-validation
        group_keys = list(groups.keys())
        n_splits = min(self.n_splits, len(group_keys))
        
        # Store fold results
        fold_results = []
        all_train_errors = []
        all_val_errors = []
        all_parameters = []
        
        # Create calibrator
        calibrator = self._create_calibrator()
        
        for fold in range(n_splits):
            logger.info(f"Processing fold {fold+1}/{n_splits}")
            
            # Select validation group
            val_group_idx = fold % len(group_keys)
            val_group = group_keys[val_group_idx]
            
            # Extract training and validation data
            val_sim = groups[val_group]["sim"]
            val_hw = groups[val_group]["hw"]
            
            train_sim = []
            train_hw = []
            for g, data in groups.items():
                if g != val_group:
                    train_sim.extend(data["sim"])
                    train_hw.extend(data["hw"])
            
            # Calibrate on training data
            calibrated_params = calibrator.calibrate(
                train_sim, train_hw, initial_parameters, parameter_bounds
            )
            
            # Calculate errors on training and validation sets
            train_error = self._calculate_error(train_sim, train_hw, calibrated_params)
            val_error = self._calculate_error(val_sim, val_hw, calibrated_params)
            
            # Store results
            fold_result = {
                "fold": fold + 1,
                "train_groups": [g for g in group_keys if g != val_group],
                "val_group": val_group,
                "train_size": len(train_sim),
                "val_size": len(val_sim),
                "train_error": float(train_error),
                "val_error": float(val_error),
                "calibrated_parameters": calibrated_params
            }
            fold_results.append(fold_result)
            
            all_train_errors.append(train_error)
            all_val_errors.append(val_error)
            all_parameters.append(calibrated_params)
            
            logger.info(f"Fold {fold+1} train error: {train_error:.6f}, val error: {val_error:.6f}")
        
        # Calculate average errors
        avg_train_error = np.mean(all_train_errors)
        avg_val_error = np.mean(all_val_errors)
        train_error_std = np.std(all_train_errors)
        val_error_std = np.std(all_val_errors)
        
        # Calculate generalization gap
        generalization_gap = avg_val_error - avg_train_error
        generalization_gap_percentage = (generalization_gap / avg_train_error * 100 
                                       if avg_train_error > 0 else 0.0)
        
        # Determine if the calibration generalizes well
        generalizes_well = generalization_gap_percentage < 20.0  # Less than 20% performance drop
        
        # Calculate parameter stability
        param_stability = self._calculate_parameter_stability(all_parameters)
        
        # Create validation result summary
        result = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "calibrator_type": self.calibrator_type,
            "group_by": group_by,
            "n_splits": n_splits,
            "num_groups": len(group_keys),
            "group_values": group_keys,
            "initial_parameters": initial_parameters,
            "fold_results": fold_results,
            "avg_train_error": float(avg_train_error),
            "avg_val_error": float(avg_val_error),
            "train_error_std": float(train_error_std),
            "val_error_std": float(val_error_std),
            "generalization_gap": float(generalization_gap),
            "generalization_gap_percentage": float(generalization_gap_percentage),
            "generalizes_well": generalizes_well,
            "parameter_stability": param_stability,
            "status": "success"
        }
        
        # Calculate recommended parameters (average of all folds)
        recommended_params = self._calculate_recommended_parameters(all_parameters, param_stability)
        result["recommended_parameters"] = recommended_params
        
        # Calculate error with recommended parameters
        recommended_error = self._calculate_error(
            simulation_results, hardware_results, recommended_params
        )
        initial_error = self._calculate_error(
            simulation_results, hardware_results, initial_parameters
        )
        
        result["recommended_error"] = float(recommended_error)
        result["initial_error"] = float(initial_error)
        result["improvement"] = float((initial_error - recommended_error) / initial_error * 100 
                                    if initial_error > 0 else 0.0)
        
        logger.info(f"Grouped cross-validation completed successfully")
        logger.info(f"Average train error: {avg_train_error:.6f}, val error: {avg_val_error:.6f}")
        logger.info(f"Generalization gap: {generalization_gap_percentage:.2f}%")
        logger.info(f"Initial error: {initial_error:.6f}, recommended error: {recommended_error:.6f}, "
                    f"improvement: {result['improvement']:.2f}%")
        
        # Store result
        self.results.append(result)
        self._save_results()
        
        return result
    
    def _create_calibrator(self) -> Union[BasicCalibrator, AdvancedCalibrator]:
        """
        Create an instance of the calibrator based on calibrator_type.
        
        Returns:
            Calibrator instance
        """
        if self.calibrator_type == "basic":
            return BasicCalibrator()
        elif self.calibrator_type == "multi_parameter":
            return MultiParameterCalibrator()
        else:
            logger.warning(f"Unknown calibrator type: {self.calibrator_type}, "
                         f"falling back to BasicCalibrator")
            return BasicCalibrator()
    
    def _calculate_error(
        self, 
        simulation_results: List[Dict[str, Any]], 
        hardware_results: List[Dict[str, Any]],
        parameters: Dict[str, Any]
    ) -> float:
        """
        Calculate error between adjusted simulation results and hardware results.
        
        Args:
            simulation_results: List of simulation result dictionaries
            hardware_results: List of hardware result dictionaries
            parameters: Parameter dictionary
            
        Returns:
            Error value
        """
        # Extract metrics from results
        sim_metrics = self._extract_metrics(simulation_results)
        hw_metrics = self._extract_metrics(hardware_results)
        
        # Calculate error for each metric
        total_error = 0.0
        count = 0
        
        for metric in set(sim_metrics.keys()).intersection(hw_metrics.keys()):
            # Apply parameters to adjust simulation values
            sim_values = sim_metrics[metric]
            hw_values = hw_metrics[metric]
            
            # Apply parameters to adjust simulation values
            adjusted_sim_values = self._apply_parameters_to_metric(sim_values, parameters, metric)
            
            # Calculate Mean Squared Error
            mse = np.mean((np.array(adjusted_sim_values) - np.array(hw_values)) ** 2)
            total_error += mse
            count += 1
        
        # Return average error across metrics
        return total_error / count if count > 0 else 0.0
    
    def _extract_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, List[float]]:
        """
        Extract metrics from result dictionaries.
        
        Args:
            results: List of result dictionaries
            
        Returns:
            Dictionary mapping metric names to lists of values
        """
        metrics = {}
        
        for result in results:
            for key, value in result.items():
                if isinstance(value, (int, float)) and not key.startswith("_"):
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append(float(value))
        
        return metrics
    
    def _apply_parameters_to_metric(
        self, 
        sim_values: List[float], 
        parameters: Dict[str, Any], 
        metric_name: str
    ) -> List[float]:
        """
        Apply parameters to adjust simulation values for a metric.
        
        Args:
            sim_values: List of simulation values for the metric
            parameters: Current parameter dictionary
            metric_name: Name of the metric
            
        Returns:
            List of adjusted simulation values
        """
        adjusted_values = np.array(sim_values).copy()
        
        # Apply metric-specific parameters
        scale_param = f"{metric_name}_scale"
        offset_param = f"{metric_name}_offset"
        
        if scale_param in parameters:
            adjusted_values *= parameters[scale_param]
        
        if offset_param in parameters:
            adjusted_values += parameters[offset_param]
        
        # Apply global parameters
        if "global_scale" in parameters:
            adjusted_values *= parameters["global_scale"]
        
        if "global_offset" in parameters:
            adjusted_values += parameters["global_offset"]
        
        return adjusted_values.tolist()
    
    def _calculate_parameter_stability(self, parameter_sets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate stability metrics for calibrated parameters across folds.
        
        Args:
            parameter_sets: List of parameter dictionaries from each fold
            
        Returns:
            Dictionary with stability metrics
        """
        # Extract all parameter names
        param_names = set()
        for params in parameter_sets:
            param_names.update(params.keys())
        
        # Calculate statistics for each parameter
        stability = {}
        
        for param in param_names:
            # Extract values for this parameter from all folds where it exists
            values = [params[param] for params in parameter_sets if param in params]
            
            if not values:
                continue
                
            # Calculate statistics
            mean_value = np.mean(values)
            std_value = np.std(values)
            min_value = np.min(values)
            max_value = np.max(values)
            
            # Calculate coefficient of variation (relative stability)
            if abs(mean_value) < 1e-10:
                cv = float('inf')  # Avoid division by zero
            else:
                cv = std_value / abs(mean_value)
            
            # Determine stability level
            if cv < 0.1:
                stability_level = "high"
            elif cv < 0.3:
                stability_level = "medium"
            else:
                stability_level = "low"
            
            stability[param] = {
                "mean": float(mean_value),
                "std": float(std_value),
                "min": float(min_value),
                "max": float(max_value),
                "cv": float(cv),
                "stability": stability_level
            }
        
        return stability
    
    def _calculate_recommended_parameters(
        self, 
        parameter_sets: List[Dict[str, Any]],
        stability: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate recommended parameters based on cross-validation results.
        
        Args:
            parameter_sets: List of parameter dictionaries from each fold
            stability: Parameter stability metrics
            
        Returns:
            Recommended parameters dictionary
        """
        # Extract all parameter names
        param_names = set()
        for params in parameter_sets:
            param_names.update(params.keys())
        
        # Calculate recommended value for each parameter
        recommended = {}
        
        for param in param_names:
            # Extract values for this parameter from all folds where it exists
            values = [params[param] for params in parameter_sets if param in params]
            
            if not values:
                continue
            
            # Get stability info
            param_stability = stability.get(param, {}).get("stability", "low")
            
            # For high stability parameters, use mean
            if param_stability == "high":
                recommended[param] = float(np.mean(values))
            # For medium stability, use mean but round to fewer digits
            elif param_stability == "medium":
                mean_value = np.mean(values)
                # Round to 2 significant digits
                if abs(mean_value) < 1e-10:
                    recommended[param] = 0.0
                else:
                    power = np.floor(np.log10(abs(mean_value)))
                    recommended[param] = float(round(mean_value, int(1 - power)))
            # For low stability, be conservative (close to original)
            else:
                # Use the median value for robustness
                recommended[param] = float(np.median(values))
        
        return recommended
    
    def _save_results(self) -> None:
        """Save cross-validation results to file."""
        if not self.result_file:
            return
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(self.result_file)), exist_ok=True)
            
            with open(self.result_file, 'w') as f:
                json.dump(self.results, f, indent=2)
                
            logger.info(f"Saved {len(self.results)} cross-validation results to file")
        except Exception as e:
            logger.error(f"Error saving cross-validation results: {str(e)}")
    
    def get_results(self) -> List[Dict[str, Any]]:
        """
        Get the cross-validation results.
        
        Returns:
            List of result dictionaries
        """
        return self.results
    
    def analyze_generalization(self) -> Dict[str, Any]:
        """
        Analyze generalization performance across validation results.
        
        Returns:
            Dictionary with analysis results
        """
        if not self.results:
            return {"error": "No cross-validation results available"}
        
        try:
            # Extract data for analysis
            timestamps = []
            train_errors = []
            val_errors = []
            gaps = []
            improvements = []
            calibrator_types = []
            
            for result in self.results:
                if result.get("status") != "success":
                    continue
                    
                timestamps.append(result["timestamp"])
                train_errors.append(result.get("avg_train_error", 0.0))
                val_errors.append(result.get("avg_val_error", 0.0))
                gaps.append(result.get("generalization_gap_percentage", 0.0))
                improvements.append(result.get("improvement", 0.0))
                calibrator_types.append(result.get("calibrator_type", "unknown"))
            
            if not timestamps:
                return {"error": "No successful cross-validation results available"}
            
            # Calculate averages
            avg_train_error = np.mean(train_errors)
            avg_val_error = np.mean(val_errors)
            avg_gap = np.mean(gaps)
            avg_improvement = np.mean(improvements)
            
            # Calculate by calibrator type
            calibrator_stats = {}
            unique_types = set(calibrator_types)
            
            for cal_type in unique_types:
                indices = [i for i, t in enumerate(calibrator_types) if t == cal_type]
                
                type_train_errors = [train_errors[i] for i in indices]
                type_val_errors = [val_errors[i] for i in indices]
                type_gaps = [gaps[i] for i in indices]
                type_improvements = [improvements[i] for i in indices]
                
                calibrator_stats[cal_type] = {
                    "count": len(indices),
                    "avg_train_error": float(np.mean(type_train_errors)),
                    "avg_val_error": float(np.mean(type_val_errors)),
                    "avg_gap": float(np.mean(type_gaps)),
                    "avg_improvement": float(np.mean(type_improvements))
                }
            
            # Generate recommendations
            recommendations = []
            
            if avg_gap > 25.0:
                recommendations.append("High generalization gap detected. Consider using more data or simpler calibration models.")
            
            if avg_improvement < 5.0:
                recommendations.append("Low improvement detected. Consider adjusting parameter bounds or trying different calibrator types.")
            
            # Get best calibrator type
            if len(unique_types) > 1:
                best_type = min(unique_types, key=lambda t: calibrator_stats[t]["avg_gap"])
                recommendations.append(f"Based on generalization performance, consider using '{best_type}' calibrator type.")
            
            return {
                "analysis_timestamp": datetime.now().isoformat(),
                "num_results": len(timestamps),
                "avg_train_error": float(avg_train_error),
                "avg_val_error": float(avg_val_error),
                "avg_generalization_gap": float(avg_gap),
                "avg_improvement": float(avg_improvement),
                "calibrator_stats": calibrator_stats,
                "recommendations": recommendations,
                "data": {
                    "timestamps": timestamps,
                    "train_errors": train_errors,
                    "val_errors": val_errors,
                    "generalization_gaps": gaps,
                    "improvements": improvements,
                    "calibrator_types": calibrator_types
                }
            }
        except Exception as e:
            logger.error(f"Error analyzing generalization: {str(e)}")
            return {"error": f"Error analyzing generalization: {str(e)}"}
    
    def visualize_results(
        self, 
        output_file: Optional[str] = None,
        result_id: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8)
    ) -> Union[str, None]:
        """
        Visualize cross-validation results.
        
        Args:
            output_file: Optional path to save visualization file
            result_id: Optional ID of specific result to visualize
            figsize: Figure size (width, height) in inches
            
        Returns:
            Path to saved visualization file or None if visualizing failed
        """
        try:
            # Setup seaborn if available
            if SEABORN_AVAILABLE:
                sns.set(style="whitegrid")
            
            # Select results to visualize
            if result_id:
                results_to_viz = [r for r in self.results if r.get("id") == result_id]
                if not results_to_viz:
                    logger.warning(f"No result found with ID {result_id}")
                    return None
                single_result = True
            else:
                results_to_viz = self.results
                single_result = False
            
            # Check if we have results to visualize
            if not results_to_viz:
                logger.warning("No results available for visualization")
                return None
            
            if single_result:
                return self._visualize_single_result(results_to_viz[0], output_file, figsize)
            else:
                return self._visualize_multiple_results(results_to_viz, output_file, figsize)
                
        except Exception as e:
            logger.error(f"Error visualizing results: {str(e)}")
            return None
    
    def _visualize_single_result(
        self, 
        result: Dict[str, Any],
        output_file: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8)
    ) -> Union[str, None]:
        """
        Visualize a single cross-validation result.
        
        Args:
            result: Result dictionary to visualize
            output_file: Optional path to save visualization file
            figsize: Figure size (width, height) in inches
            
        Returns:
            Path to saved visualization file or None if visualizing failed
        """
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f"Cross-Validation Results - {result.get('calibrator_type', 'unknown')}", fontsize=16)
        
        # Plot 1: Errors by fold
        fold_numbers = [r.get("fold", i+1) for i, r in enumerate(result.get("fold_results", []))]
        train_errors = [r.get("train_error", 0.0) for r in result.get("fold_results", [])]
        val_errors = [r.get("val_error", 0.0) for r in result.get("fold_results", [])]
        
        axs[0, 0].plot(fold_numbers, train_errors, 'o-', label='Training Error')
        axs[0, 0].plot(fold_numbers, val_errors, 'o-', label='Validation Error')
        axs[0, 0].set_xlabel('Fold')
        axs[0, 0].set_ylabel('Error')
        axs[0, 0].set_title('Errors by Fold')
        axs[0, 0].legend()
        
        # Plot 2: Parameter stability
        param_names = []
        stability_values = []
        
        for param, data in result.get("parameter_stability", {}).items():
            param_names.append(param)
            stability_values.append(data.get("cv", 0.0))
        
        if param_names:
            y_pos = np.arange(len(param_names))
            axs[0, 1].barh(y_pos, stability_values)
            axs[0, 1].set_yticks(y_pos)
            axs[0, 1].set_yticklabels(param_names)
            axs[0, 1].invert_yaxis()  # Labels read top-to-bottom
            axs[0, 1].set_xlabel('Coefficient of Variation')
            axs[0, 1].set_title('Parameter Stability')
        else:
            axs[0, 1].text(0.5, 0.5, 'No parameter stability data available', 
                           horizontalalignment='center', verticalalignment='center')
        
        # Plot 3: Before vs After Error
        labels = ['Initial', 'Recommended']
        values = [result.get("initial_error", 0.0), result.get("recommended_error", 0.0)]
        
        axs[1, 0].bar(labels, values)
        axs[1, 0].set_ylabel('Error')
        axs[1, 0].set_title('Error Comparison')
        
        # Add improvement percentage
        if "improvement" in result:
            improvement = result["improvement"]
            axs[1, 0].text(1, values[1] * 1.1, f"{improvement:.2f}% improvement", 
                           horizontalalignment='center')
        
        # Plot 4: Parameter Changes
        if "initial_parameters" in result and "recommended_parameters" in result:
            initial_params = result["initial_parameters"]
            recommended_params = result["recommended_parameters"]
            
            params_to_plot = set(initial_params.keys()).intersection(recommended_params.keys())
            param_names = []
            change_values = []
            
            for param in params_to_plot:
                if isinstance(initial_params[param], (int, float)) and isinstance(recommended_params[param], (int, float)):
                    param_names.append(param)
                    if abs(initial_params[param]) < 1e-10:
                        change_values.append(0.0)
                    else:
                        change_pct = (recommended_params[param] - initial_params[param]) / abs(initial_params[param]) * 100
                        change_values.append(change_pct)
            
            if param_names:
                y_pos = np.arange(len(param_names))
                axs[1, 1].barh(y_pos, change_values)
                axs[1, 1].set_yticks(y_pos)
                axs[1, 1].set_yticklabels(param_names)
                axs[1, 1].invert_yaxis()  # Labels read top-to-bottom
                axs[1, 1].set_xlabel('Percentage Change')
                axs[1, 1].set_title('Parameter Changes')
                
                # Add a vertical line at 0
                axs[1, 1].axvline(x=0, color='k', linestyle='-', alpha=0.3)
            else:
                axs[1, 1].text(0.5, 0.5, 'No parameter change data available', 
                               horizontalalignment='center', verticalalignment='center')
        else:
            axs[1, 1].text(0.5, 0.5, 'No parameter change data available', 
                           horizontalalignment='center', verticalalignment='center')
        
        # Add metadata
        timestamp = result.get("timestamp", "Unknown")
        n_splits = result.get("n_splits", 0)
        fig.text(0.5, 0.01, f"Date: {timestamp} | Splits: {n_splits}", ha='center')
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save figure if output file is provided
        if output_file:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            
            plt.savefig(output_file)
            logger.info(f"Saved visualization to {output_file}")
            plt.close(fig)
            return output_file
        else:
            plt.show()
            return None
    
    def _visualize_multiple_results(
        self, 
        results: List[Dict[str, Any]],
        output_file: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8)
    ) -> Union[str, None]:
        """
        Visualize multiple cross-validation results.
        
        Args:
            results: List of result dictionaries to visualize
            output_file: Optional path to save visualization file
            figsize: Figure size (width, height) in inches
            
        Returns:
            Path to saved visualization file or None if visualizing failed
        """
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f"Cross-Validation Analysis ({len(results)} results)", fontsize=16)
        
        # Extract data
        timestamps = []
        train_errors = []
        val_errors = []
        gaps = []
        improvements = []
        calibrator_types = []
        
        for result in results:
            if result.get("status") != "success":
                continue
                
            timestamps.append(result["timestamp"])
            train_errors.append(result.get("avg_train_error", 0.0))
            val_errors.append(result.get("avg_val_error", 0.0))
            gaps.append(result.get("generalization_gap_percentage", 0.0))
            improvements.append(result.get("improvement", 0.0))
            calibrator_types.append(result.get("calibrator_type", "unknown"))
        
        # Convert timestamps to datetime objects for sorting
        datetime_timestamps = [datetime.fromisoformat(ts) for ts in timestamps]
        
        # Sort all data by timestamp
        sorted_indices = np.argsort(datetime_timestamps)
        sorted_timestamps = [timestamps[i] for i in sorted_indices]
        sorted_train_errors = [train_errors[i] for i in sorted_indices]
        sorted_val_errors = [val_errors[i] for i in sorted_indices]
        sorted_gaps = [gaps[i] for i in sorted_indices]
        sorted_improvements = [improvements[i] for i in sorted_indices]
        sorted_calibrator_types = [calibrator_types[i] for i in sorted_indices]
        
        # Plot 1: Train and Validation Errors over time
        x_range = np.arange(len(sorted_timestamps))
        
        axs[0, 0].plot(x_range, sorted_train_errors, 'o-', label='Training Error')
        axs[0, 0].plot(x_range, sorted_val_errors, 'o-', label='Validation Error')
        axs[0, 0].set_xlabel('Time')
        axs[0, 0].set_ylabel('Error')
        axs[0, 0].set_title('Errors over Time')
        axs[0, 0].set_xticks(x_range)
        axs[0, 0].set_xticklabels([])  # Hide detailed timestamps
        axs[0, 0].legend()
        
        # Plot 2: Generalization Gap over time
        axs[0, 1].plot(x_range, sorted_gaps, 'o-')
        axs[0, 1].set_xlabel('Time')
        axs[0, 1].set_ylabel('Generalization Gap (%)')
        axs[0, 1].set_title('Generalization Gap over Time')
        axs[0, 1].set_xticks(x_range)
        axs[0, 1].set_xticklabels([])  # Hide detailed timestamps
        
        # Add a horizontal line at 20% (generalization threshold)
        axs[0, 1].axhline(y=20.0, color='r', linestyle='--', alpha=0.5)
        
        # Plot 3: Improvement over time
        axs[1, 0].plot(x_range, sorted_improvements, 'o-')
        axs[1, 0].set_xlabel('Time')
        axs[1, 0].set_ylabel('Improvement (%)')
        axs[1, 0].set_title('Improvement over Time')
        axs[1, 0].set_xticks(x_range)
        axs[1, 0].set_xticklabels([])  # Hide detailed timestamps
        
        # Plot 4: Calibrator Type Performance
        unique_types = list(set(calibrator_types))
        type_gaps = []
        type_improvements = []
        
        for cal_type in unique_types:
            indices = [i for i, t in enumerate(calibrator_types) if t == cal_type]
            
            type_train_errors = [train_errors[i] for i in indices]
            type_val_errors = [val_errors[i] for i in indices]
            type_gaps.append(np.mean([gaps[i] for i in indices]))
            type_improvements.append(np.mean([improvements[i] for i in indices]))
        
        if unique_types:
            x = np.arange(len(unique_types))
            width = 0.35
            
            axs[1, 1].bar(x - width/2, type_gaps, width, label='Avg. Gap')
            axs[1, 1].bar(x + width/2, type_improvements, width, label='Avg. Improvement')
            axs[1, 1].set_xlabel('Calibrator Type')
            axs[1, 1].set_ylabel('Percentage')
            axs[1, 1].set_title('Calibrator Type Performance')
            axs[1, 1].set_xticks(x)
            axs[1, 1].set_xticklabels(unique_types)
            axs[1, 1].legend()
        else:
            axs[1, 1].text(0.5, 0.5, 'No calibrator type data available', 
                           horizontalalignment='center', verticalalignment='center')
        
        # Add metadata
        fig.text(0.5, 0.01, f"Number of Results: {len(results)} | Date Range: {sorted_timestamps[0][:10]} to {sorted_timestamps[-1][:10]}", ha='center')
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save figure if output file is provided
        if output_file:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            
            plt.savefig(output_file)
            logger.info(f"Saved visualization to {output_file}")
            plt.close(fig)
            return output_file
        else:
            plt.show()
            return None
    
    def clear_results(self) -> None:
        """Clear the cross-validation results."""
        self.results = []
        if self.result_file and os.path.exists(self.result_file):
            try:
                os.remove(self.result_file)
                logger.info(f"Removed result file: {self.result_file}")
            except Exception as e:
                logger.error(f"Error removing result file: {str(e)}")