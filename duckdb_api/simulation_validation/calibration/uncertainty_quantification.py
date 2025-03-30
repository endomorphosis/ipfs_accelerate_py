"""
Uncertainty quantification for calibration parameters.

This module provides functionality for quantifying uncertainty in calibration parameters
and propagating that uncertainty to simulation results.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional, Union, Callable
import json
from datetime import datetime
import os
import uuid
from scipy import stats

# Setup logger
logger = logging.getLogger(__name__)

class UncertaintyQuantifier:
    """
    Quantifies uncertainty in calibration parameters and predictions.
    
    This class analyzes the uncertainty in calibrated parameters and propagates
    that uncertainty to simulation results, providing confidence intervals and
    reliability metrics.
    """
    
    def __init__(
        self, 
        confidence_level: float = 0.95,
        n_samples: int = 1000,
        result_file: Optional[str] = None
    ):
        """
        Initialize the UncertaintyQuantifier.
        
        Args:
            confidence_level: Confidence level for intervals (0-1)
            n_samples: Number of Monte Carlo samples for uncertainty propagation
            result_file: Optional path to file for storing uncertainty results
        """
        self.confidence_level = confidence_level
        self.n_samples = n_samples
        self.result_file = result_file
        self.results = []
        
        # Load previous results if available
        if self.result_file and os.path.exists(self.result_file):
            try:
                with open(self.result_file, 'r') as f:
                    self.results = json.load(f)
                logger.info(f"Loaded {len(self.results)} uncertainty results from file")
            except Exception as e:
                logger.error(f"Error loading uncertainty results: {str(e)}")
                self.results = []
    
    def quantify_parameter_uncertainty(
        self, 
        parameter_sets: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Quantify uncertainty in calibration parameters.
        
        Args:
            parameter_sets: List of parameter dictionaries (e.g., from cross-validation)
            
        Returns:
            Dictionary with uncertainty metrics and confidence intervals
        """
        logger.info("Quantifying parameter uncertainty")
        
        if not parameter_sets:
            logger.warning("No parameter sets provided")
            return {
                "error": "No parameter sets provided",
                "status": "failed",
                "timestamp": datetime.now().isoformat()
            }
        
        # Extract all parameter names
        param_names = set()
        for params in parameter_sets:
            param_names.update(params.keys())
        
        # Calculate uncertainty metrics for each parameter
        uncertainty = {}
        
        for param in param_names:
            # Extract values for this parameter from all sets where it exists
            values = [params[param] for params in parameter_sets if param in params 
                    and isinstance(params[param], (int, float))]
            
            if not values or len(values) < 2:
                continue
                
            # Calculate statistics
            mean_value = np.mean(values)
            std_value = np.std(values)
            
            # Calculate confidence interval
            if len(values) >= 30:
                # Use normal distribution
                z = stats.norm.ppf((1 + self.confidence_level) / 2)
                margin = z * (std_value / np.sqrt(len(values)))
                ci_lower = mean_value - margin
                ci_upper = mean_value + margin
            else:
                # Use t-distribution
                t = stats.t.ppf((1 + self.confidence_level) / 2, len(values) - 1)
                margin = t * (std_value / np.sqrt(len(values)))
                ci_lower = mean_value - margin
                ci_upper = mean_value + margin
            
            # Calculate coefficient of variation
            if abs(mean_value) < 1e-10:
                cv = float('inf')  # Avoid division by zero
            else:
                cv = std_value / abs(mean_value)
            
            # Determine uncertainty level
            if cv < 0.1:
                uncertainty_level = "low"
            elif cv < 0.3:
                uncertainty_level = "medium"
            else:
                uncertainty_level = "high"
            
            uncertainty[param] = {
                "mean": float(mean_value),
                "std": float(std_value),
                "cv": float(cv),
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper),
                "uncertainty_level": uncertainty_level,
                "sample_size": len(values)
            }
        
        # Create uncertainty result
        result = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "confidence_level": self.confidence_level,
            "parameter_uncertainty": uncertainty,
            "status": "success"
        }
        
        # Store result
        self.results.append(result)
        self._save_results()
        
        logger.info(f"Parameter uncertainty quantification completed for {len(uncertainty)} parameters")
        return result
    
    def propagate_uncertainty(
        self, 
        parameter_uncertainty: Dict[str, Dict[str, Any]],
        simulation_results: List[Dict[str, Any]],
        error_function: Callable[[Dict[str, Any]], float]
    ) -> Dict[str, Any]:
        """
        Propagate parameter uncertainty to error function.
        
        Args:
            parameter_uncertainty: Parameter uncertainty metrics dictionary
            simulation_results: List of simulation result dictionaries
            error_function: Function that takes parameters and returns error value
            
        Returns:
            Dictionary with propagated uncertainty metrics
        """
        logger.info("Propagating parameter uncertainty")
        
        # Generate parameter samples based on uncertainty
        parameter_samples = self._generate_parameter_samples(parameter_uncertainty)
        
        if not parameter_samples:
            logger.warning("No valid parameter samples generated")
            return {
                "error": "No valid parameter samples generated",
                "status": "failed",
                "timestamp": datetime.now().isoformat()
            }
        
        # Evaluate error function for each parameter sample
        errors = []
        
        for params in parameter_samples:
            try:
                error = error_function(params)
                errors.append(error)
            except Exception as e:
                logger.warning(f"Error evaluating parameters: {str(e)}")
                continue
        
        if not errors:
            logger.warning("No valid error evaluations")
            return {
                "error": "No valid error evaluations",
                "status": "failed",
                "timestamp": datetime.now().isoformat()
            }
        
        # Calculate error statistics
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        
        # Calculate error confidence interval
        sorted_errors = np.sort(errors)
        lower_idx = int(len(errors) * (1 - self.confidence_level) / 2)
        upper_idx = int(len(errors) * (1 + self.confidence_level) / 2)
        
        ci_lower = sorted_errors[lower_idx]
        ci_upper = sorted_errors[upper_idx]
        
        # Calculate relative uncertainty (coefficient of variation)
        if abs(mean_error) < 1e-10:
            relative_uncertainty = float('inf')  # Avoid division by zero
        else:
            relative_uncertainty = std_error / abs(mean_error)
        
        # Determine uncertainty level
        if relative_uncertainty < 0.1:
            uncertainty_level = "low"
        elif relative_uncertainty < 0.3:
            uncertainty_level = "medium"
        else:
            uncertainty_level = "high"
        
        # Create propagation result
        result = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "confidence_level": self.confidence_level,
            "n_samples": self.n_samples,
            "error_statistics": {
                "mean": float(mean_error),
                "std": float(std_error),
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper),
                "relative_uncertainty": float(relative_uncertainty),
                "uncertainty_level": uncertainty_level
            },
            "status": "success"
        }
        
        # Store result
        self.results.append(result)
        self._save_results()
        
        logger.info(f"Uncertainty propagation completed with {len(errors)} valid evaluations")
        logger.info(f"Mean error: {mean_error:.6f}, std: {std_error:.6f}")
        logger.info(f"Confidence interval: [{ci_lower:.6f}, {ci_upper:.6f}]")
        
        return result
    
    def _generate_parameter_samples(
        self, 
        parameter_uncertainty: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate parameter samples based on uncertainty metrics.
        
        Args:
            parameter_uncertainty: Parameter uncertainty metrics dictionary
            
        Returns:
            List of parameter dictionaries sampled from uncertainty distributions
        """
        # Create list of parameter names
        param_names = list(parameter_uncertainty.keys())
        
        if not param_names:
            logger.warning("No parameters with uncertainty metrics")
            return []
        
        # Generate samples for each parameter
        param_samples = {}
        
        for param in param_names:
            metrics = parameter_uncertainty[param]
            
            mean = metrics["mean"]
            std = metrics["std"]
            
            # Check if standard deviation is valid
            if std <= 0:
                logger.warning(f"Invalid standard deviation for parameter {param}: {std}")
                continue
            
            # Generate samples from normal distribution
            samples = np.random.normal(mean, std, size=self.n_samples)
            param_samples[param] = samples
        
        # Combine parameter samples into parameter dictionaries
        parameter_dicts = []
        
        for i in range(self.n_samples):
            params = {}
            for param in param_names:
                if param in param_samples:
                    params[param] = float(param_samples[param][i])
            
            parameter_dicts.append(params)
        
        return parameter_dicts
    
    def estimate_reliability(
        self, 
        parameter_uncertainty: Dict[str, Dict[str, Any]],
        simulation_results: List[Dict[str, Any]],
        error_threshold: float,
        error_function: Callable[[Dict[str, Any]], float]
    ) -> Dict[str, Any]:
        """
        Estimate reliability of calibration parameters based on uncertainty.
        
        Args:
            parameter_uncertainty: Parameter uncertainty metrics dictionary
            simulation_results: List of simulation result dictionaries
            error_threshold: Threshold for acceptable error
            error_function: Function that takes parameters and returns error value
            
        Returns:
            Dictionary with reliability metrics
        """
        logger.info(f"Estimating reliability with error threshold {error_threshold}")
        
        # Generate parameter samples based on uncertainty
        parameter_samples = self._generate_parameter_samples(parameter_uncertainty)
        
        if not parameter_samples:
            logger.warning("No valid parameter samples generated")
            return {
                "error": "No valid parameter samples generated",
                "status": "failed",
                "timestamp": datetime.now().isoformat()
            }
        
        # Evaluate error function for each parameter sample
        errors = []
        
        for params in parameter_samples:
            try:
                error = error_function(params)
                errors.append(error)
            except Exception as e:
                logger.warning(f"Error evaluating parameters: {str(e)}")
                continue
        
        if not errors:
            logger.warning("No valid error evaluations")
            return {
                "error": "No valid error evaluations",
                "status": "failed",
                "timestamp": datetime.now().isoformat()
            }
        
        # Calculate reliability metrics
        reliability = np.mean([e <= error_threshold for e in errors])
        
        # Calculate error statistics
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        min_error = np.min(errors)
        max_error = np.max(errors)
        
        # Calculate confidence interval for reliability
        z = stats.norm.ppf((1 + self.confidence_level) / 2)
        margin = z * np.sqrt((reliability * (1 - reliability)) / len(errors))
        reliability_ci_lower = max(0.0, reliability - margin)
        reliability_ci_upper = min(1.0, reliability + margin)
        
        # Determine reliability level
        if reliability > 0.95:
            reliability_level = "high"
        elif reliability > 0.8:
            reliability_level = "medium"
        else:
            reliability_level = "low"
        
        # Create reliability result
        result = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "confidence_level": self.confidence_level,
            "n_samples": self.n_samples,
            "error_threshold": float(error_threshold),
            "reliability": float(reliability),
            "reliability_ci_lower": float(reliability_ci_lower),
            "reliability_ci_upper": float(reliability_ci_upper),
            "reliability_level": reliability_level,
            "error_statistics": {
                "mean": float(mean_error),
                "std": float(std_error),
                "min": float(min_error),
                "max": float(max_error)
            },
            "status": "success"
        }
        
        # Store result
        self.results.append(result)
        self._save_results()
        
        logger.info(f"Reliability estimation completed with {len(errors)} valid evaluations")
        logger.info(f"Reliability: {reliability:.4f} ({reliability*100:.1f}%)")
        logger.info(f"Reliability CI: [{reliability_ci_lower:.4f}, {reliability_ci_upper:.4f}]")
        
        return result
    
    def sensitivity_analysis(
        self, 
        parameter_uncertainty: Dict[str, Dict[str, Any]],
        simulation_results: List[Dict[str, Any]],
        error_function: Callable[[Dict[str, Any]], float],
        perturbation_factor: float = 0.1
    ) -> Dict[str, Any]:
        """
        Perform sensitivity analysis to identify critical parameters.
        
        Args:
            parameter_uncertainty: Parameter uncertainty metrics dictionary
            simulation_results: List of simulation result dictionaries
            error_function: Function that takes parameters and returns error value
            perturbation_factor: Factor for parameter perturbation (0-1)
            
        Returns:
            Dictionary with sensitivity analysis results
        """
        logger.info(f"Performing sensitivity analysis with perturbation factor {perturbation_factor}")
        
        # Create baseline parameters (mean values)
        baseline_params = {param: metrics["mean"] for param, metrics in parameter_uncertainty.items()}
        
        # Calculate baseline error
        try:
            baseline_error = error_function(baseline_params)
        except Exception as e:
            logger.warning(f"Error evaluating baseline parameters: {str(e)}")
            return {
                "error": f"Error evaluating baseline parameters: {str(e)}",
                "status": "failed",
                "timestamp": datetime.now().isoformat()
            }
        
        # Calculate sensitivity for each parameter
        sensitivity = {}
        
        for param, metrics in parameter_uncertainty.items():
            mean = metrics["mean"]
            std = metrics["std"]
            
            # Perturb parameter up and down
            perturbation = std * perturbation_factor if std > 0 else abs(mean * perturbation_factor)
            
            # Create parameter sets with this parameter perturbed
            params_up = baseline_params.copy()
            params_up[param] = mean + perturbation
            
            params_down = baseline_params.copy()
            params_down[param] = mean - perturbation
            
            # Calculate errors with perturbed parameters
            try:
                error_up = error_function(params_up)
                error_down = error_function(params_down)
                
                # Calculate sensitivity (average absolute change in error)
                sensitivity_value = (abs(error_up - baseline_error) + abs(error_down - baseline_error)) / 2
                
                # Calculate relative sensitivity (normalized by baseline error)
                if abs(baseline_error) < 1e-10:
                    relative_sensitivity = float('inf')  # Avoid division by zero
                else:
                    relative_sensitivity = sensitivity_value / abs(baseline_error)
                
                # Calculate non-linearity (assymetry of sensitivity)
                non_linearity = abs(error_up - baseline_error) - abs(error_down - baseline_error)
                if abs(baseline_error) < 1e-10:
                    relative_non_linearity = float('inf')  # Avoid division by zero
                else:
                    relative_non_linearity = non_linearity / abs(baseline_error)
                
                # Determine sensitivity level
                if relative_sensitivity < 0.01:
                    sensitivity_level = "very_low"
                elif relative_sensitivity < 0.05:
                    sensitivity_level = "low"
                elif relative_sensitivity < 0.1:
                    sensitivity_level = "medium"
                elif relative_sensitivity < 0.2:
                    sensitivity_level = "high"
                else:
                    sensitivity_level = "very_high"
                
                sensitivity[param] = {
                    "sensitivity": float(sensitivity_value),
                    "relative_sensitivity": float(relative_sensitivity),
                    "non_linearity": float(non_linearity),
                    "relative_non_linearity": float(relative_non_linearity),
                    "mean": float(mean),
                    "perturbation": float(perturbation),
                    "error_up": float(error_up),
                    "error_down": float(error_down),
                    "sensitivity_level": sensitivity_level
                }
                
            except Exception as e:
                logger.warning(f"Error evaluating perturbed parameters for {param}: {str(e)}")
                continue
        
        # Sort parameters by sensitivity
        sorted_params = sorted(sensitivity.keys(), key=lambda p: sensitivity[p]["relative_sensitivity"], reverse=True)
        
        # Identify critical parameters (high sensitivity and uncertainty)
        critical_params = []
        
        for param in sorted_params:
            if (sensitivity[param]["sensitivity_level"] in ["high", "very_high"] and
                parameter_uncertainty[param]["uncertainty_level"] in ["medium", "high"]):
                critical_params.append(param)
        
        # Create sensitivity analysis result
        result = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "perturbation_factor": perturbation_factor,
            "baseline_error": float(baseline_error),
            "sensitivity": sensitivity,
            "sorted_parameters": sorted_params,
            "critical_parameters": critical_params,
            "status": "success"
        }
        
        # Store result
        self.results.append(result)
        self._save_results()
        
        logger.info(f"Sensitivity analysis completed for {len(sensitivity)} parameters")
        logger.info(f"Identified {len(critical_params)} critical parameters: {critical_params}")
        
        return result
    
    def _save_results(self) -> None:
        """Save uncertainty results to file."""
        if not self.result_file:
            return
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(self.result_file)), exist_ok=True)
            
            with open(self.result_file, 'w') as f:
                json.dump(self.results, f, indent=2)
                
            logger.info(f"Saved {len(self.results)} uncertainty results to file")
        except Exception as e:
            logger.error(f"Error saving uncertainty results: {str(e)}")
    
    def get_results(self) -> List[Dict[str, Any]]:
        """
        Get the uncertainty quantification results.
        
        Returns:
            List of result dictionaries
        """
        return self.results
    
    def generate_report(
        self, 
        result_id: Optional[str] = None,
        format: str = "text"
    ) -> str:
        """
        Generate a report from uncertainty results.
        
        Args:
            result_id: Optional ID of specific result to report on (latest if None)
            format: Report format ("text", "markdown", or "json")
            
        Returns:
            Report string in the specified format
        """
        # Select result to report on
        if result_id:
            results_to_report = [r for r in self.results if r.get("id") == result_id]
            if not results_to_report:
                logger.warning(f"No result found with ID {result_id}")
                return f"No result found with ID {result_id}"
        else:
            # Use the latest result
            if not self.results:
                logger.warning("No results available for reporting")
                return "No results available for reporting"
            results_to_report = [self.results[-1]]
        
        result = results_to_report[0]
        
        # Generate report in the specified format
        if format == "json":
            return json.dumps(result, indent=2)
        elif format == "markdown":
            return self._generate_markdown_report(result)
        else:
            return self._generate_text_report(result)
    
    def _generate_text_report(self, result: Dict[str, Any]) -> str:
        """
        Generate a text report from an uncertainty result.
        
        Args:
            result: Uncertainty result dictionary
            
        Returns:
            Text report string
        """
        lines = []
        lines.append("Uncertainty Quantification Report")
        lines.append("=" * 40)
        lines.append(f"Generated: {datetime.now().isoformat()}")
        lines.append(f"Result ID: {result.get('id', 'N/A')}")
        lines.append(f"Timestamp: {result.get('timestamp', 'N/A')}")
        lines.append(f"Status: {result.get('status', 'N/A')}")
        lines.append(f"Confidence Level: {result.get('confidence_level', 'N/A')}")
        
        if "parameter_uncertainty" in result:
            lines.append("\nParameter Uncertainty:")
            lines.append("-" * 40)
            
            param_uncertainty = result["parameter_uncertainty"]
            
            # Sort parameters by uncertainty level
            sorted_params = sorted(
                param_uncertainty.keys(),
                key=lambda p: {"low": 0, "medium": 1, "high": 2}.get(
                    param_uncertainty[p].get("uncertainty_level", "low"), 0),
                reverse=True
            )
            
            for param in sorted_params:
                metrics = param_uncertainty[param]
                lines.append(f"{param}:")
                lines.append(f"  Mean: {metrics.get('mean', 'N/A'):.6f}")
                lines.append(f"  Std: {metrics.get('std', 'N/A'):.6f}")
                lines.append(f"  CV: {metrics.get('cv', 'N/A'):.4f}")
                lines.append(f"  CI: [{metrics.get('ci_lower', 'N/A'):.6f}, {metrics.get('ci_upper', 'N/A'):.6f}]")
                lines.append(f"  Uncertainty: {metrics.get('uncertainty_level', 'N/A')}")
                lines.append("")
        
        if "error_statistics" in result:
            lines.append("\nError Statistics:")
            lines.append("-" * 40)
            
            error_stats = result["error_statistics"]
            lines.append(f"Mean: {error_stats.get('mean', 'N/A'):.6f}")
            lines.append(f"Std: {error_stats.get('std', 'N/A'):.6f}")
            
            if "ci_lower" in error_stats and "ci_upper" in error_stats:
                lines.append(f"CI: [{error_stats.get('ci_lower', 'N/A'):.6f}, {error_stats.get('ci_upper', 'N/A'):.6f}]")
            
            if "min" in error_stats and "max" in error_stats:
                lines.append(f"Min: {error_stats.get('min', 'N/A'):.6f}")
                lines.append(f"Max: {error_stats.get('max', 'N/A'):.6f}")
            
            if "relative_uncertainty" in error_stats:
                lines.append(f"Relative Uncertainty: {error_stats.get('relative_uncertainty', 'N/A'):.4f}")
            
            if "uncertainty_level" in error_stats:
                lines.append(f"Uncertainty Level: {error_stats.get('uncertainty_level', 'N/A')}")
            
            lines.append("")
        
        if "reliability" in result:
            lines.append("\nReliability Analysis:")
            lines.append("-" * 40)
            lines.append(f"Error Threshold: {result.get('error_threshold', 'N/A'):.6f}")
            lines.append(f"Reliability: {result.get('reliability', 'N/A'):.4f} ({result.get('reliability', 0.0) * 100:.1f}%)")
            lines.append(f"Reliability CI: [{result.get('reliability_ci_lower', 'N/A'):.4f}, {result.get('reliability_ci_upper', 'N/A'):.4f}]")
            lines.append(f"Reliability Level: {result.get('reliability_level', 'N/A')}")
            lines.append("")
        
        if "sensitivity" in result:
            lines.append("\nSensitivity Analysis:")
            lines.append("-" * 40)
            lines.append(f"Perturbation Factor: {result.get('perturbation_factor', 'N/A')}")
            lines.append(f"Baseline Error: {result.get('baseline_error', 'N/A'):.6f}")
            
            sensitivity = result["sensitivity"]
            sorted_params = result.get("sorted_parameters", list(sensitivity.keys()))
            
            for param in sorted_params:
                metrics = sensitivity[param]
                lines.append(f"{param}:")
                lines.append(f"  Sensitivity: {metrics.get('sensitivity', 'N/A'):.6f}")
                lines.append(f"  Relative Sensitivity: {metrics.get('relative_sensitivity', 'N/A'):.4f}")
                lines.append(f"  Level: {metrics.get('sensitivity_level', 'N/A')}")
                lines.append("")
            
            if "critical_parameters" in result and result["critical_parameters"]:
                lines.append("\nCritical Parameters:")
                for param in result["critical_parameters"]:
                    lines.append(f"  - {param}")
                lines.append("")
        
        return "\n".join(lines)
    
    def _generate_markdown_report(self, result: Dict[str, Any]) -> str:
        """
        Generate a markdown report from an uncertainty result.
        
        Args:
            result: Uncertainty result dictionary
            
        Returns:
            Markdown report string
        """
        lines = []
        lines.append("# Uncertainty Quantification Report")
        lines.append("")
        lines.append(f"- **Generated:** {datetime.now().isoformat()}")
        lines.append(f"- **Result ID:** {result.get('id', 'N/A')}")
        lines.append(f"- **Timestamp:** {result.get('timestamp', 'N/A')}")
        lines.append(f"- **Status:** {result.get('status', 'N/A')}")
        lines.append(f"- **Confidence Level:** {result.get('confidence_level', 'N/A')}")
        
        if "parameter_uncertainty" in result:
            lines.append("")
            lines.append("## Parameter Uncertainty")
            
            param_uncertainty = result["parameter_uncertainty"]
            
            # Sort parameters by uncertainty level
            sorted_params = sorted(
                param_uncertainty.keys(),
                key=lambda p: {"low": 0, "medium": 1, "high": 2}.get(
                    param_uncertainty[p].get("uncertainty_level", "low"), 0),
                reverse=True
            )
            
            # Create a table
            lines.append("")
            lines.append("| Parameter | Mean | Std | CV | CI Lower | CI Upper | Uncertainty |")
            lines.append("|-----------|------|-----|----|---------|---------| ------------|")
            
            for param in sorted_params:
                metrics = param_uncertainty[param]
                lines.append(f"| {param} | "
                           f"{metrics.get('mean', 'N/A'):.6f} | "
                           f"{metrics.get('std', 'N/A'):.6f} | "
                           f"{metrics.get('cv', 'N/A'):.4f} | "
                           f"{metrics.get('ci_lower', 'N/A'):.6f} | "
                           f"{metrics.get('ci_upper', 'N/A'):.6f} | "
                           f"{metrics.get('uncertainty_level', 'N/A')} |")
        
        if "error_statistics" in result:
            lines.append("")
            lines.append("## Error Statistics")
            lines.append("")
            
            error_stats = result["error_statistics"]
            lines.append(f"- **Mean:** {error_stats.get('mean', 'N/A'):.6f}")
            lines.append(f"- **Std:** {error_stats.get('std', 'N/A'):.6f}")
            
            if "ci_lower" in error_stats and "ci_upper" in error_stats:
                lines.append(f"- **CI:** [{error_stats.get('ci_lower', 'N/A'):.6f}, {error_stats.get('ci_upper', 'N/A'):.6f}]")
            
            if "min" in error_stats and "max" in error_stats:
                lines.append(f"- **Min:** {error_stats.get('min', 'N/A'):.6f}")
                lines.append(f"- **Max:** {error_stats.get('max', 'N/A'):.6f}")
            
            if "relative_uncertainty" in error_stats:
                lines.append(f"- **Relative Uncertainty:** {error_stats.get('relative_uncertainty', 'N/A'):.4f}")
            
            if "uncertainty_level" in error_stats:
                lines.append(f"- **Uncertainty Level:** {error_stats.get('uncertainty_level', 'N/A')}")
        
        if "reliability" in result:
            lines.append("")
            lines.append("## Reliability Analysis")
            lines.append("")
            lines.append(f"- **Error Threshold:** {result.get('error_threshold', 'N/A'):.6f}")
            lines.append(f"- **Reliability:** {result.get('reliability', 'N/A'):.4f} ({result.get('reliability', 0.0) * 100:.1f}%)")
            lines.append(f"- **Reliability CI:** [{result.get('reliability_ci_lower', 'N/A'):.4f}, {result.get('reliability_ci_upper', 'N/A'):.4f}]")
            lines.append(f"- **Reliability Level:** {result.get('reliability_level', 'N/A')}")
        
        if "sensitivity" in result:
            lines.append("")
            lines.append("## Sensitivity Analysis")
            lines.append("")
            lines.append(f"- **Perturbation Factor:** {result.get('perturbation_factor', 'N/A')}")
            lines.append(f"- **Baseline Error:** {result.get('baseline_error', 'N/A'):.6f}")
            
            sensitivity = result["sensitivity"]
            sorted_params = result.get("sorted_parameters", list(sensitivity.keys()))
            
            # Create a table
            lines.append("")
            lines.append("| Parameter | Sensitivity | Relative Sensitivity | Level |")
            lines.append("|-----------|------------|----------------------|-------|")
            
            for param in sorted_params:
                metrics = sensitivity[param]
                lines.append(f"| {param} | "
                           f"{metrics.get('sensitivity', 'N/A'):.6f} | "
                           f"{metrics.get('relative_sensitivity', 'N/A'):.4f} | "
                           f"{metrics.get('sensitivity_level', 'N/A')} |")
            
            if "critical_parameters" in result and result["critical_parameters"]:
                lines.append("")
                lines.append("### Critical Parameters")
                lines.append("")
                for param in result["critical_parameters"]:
                    lines.append(f"- {param}")
        
        return "\n".join(lines)
    
    def clear_results(self) -> None:
        """Clear the uncertainty quantification results."""
        self.results = []
        if self.result_file and os.path.exists(self.result_file):
            try:
                os.remove(self.result_file)
                logger.info(f"Removed result file: {self.result_file}")
            except Exception as e:
                logger.error(f"Error removing result file: {str(e)}")