#!/usr/bin/env python3
"""
Main script for running the calibration system.

This script provides a command-line interface for running various components
of the calibration system for the Simulation Accuracy and Validation Framework.
"""

import argparse
import logging
import sys
import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional, Union, Callable

# Import calibration components
from .basic_calibrator import BasicCalibrator
from .advanced_calibrator import (
    AdvancedCalibrator, 
    MultiParameterCalibrator, 
    BayesianOptimizationCalibrator,
    NeuralNetworkCalibrator,
    EnsembleCalibrator
)
from .parameter_discovery import ParameterDiscovery, AdaptiveCalibrationScheduler
from .cross_validation import CalibrationCrossValidator
from .uncertainty_quantification import UncertaintyQuantifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('calibration.log')
    ]
)

logger = logging.getLogger(__name__)

def load_data(sim_file: str, hw_file: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load simulation and hardware result data from files.
    
    Args:
        sim_file: Path to simulation result file (JSON)
        hw_file: Path to hardware result file (JSON)
        
    Returns:
        Tuple of (simulation_results, hardware_results)
    """
    logger.info(f"Loading simulation results from {sim_file}")
    logger.info(f"Loading hardware results from {hw_file}")
    
    try:
        with open(sim_file, 'r') as f:
            simulation_results = json.load(f)
        
        with open(hw_file, 'r') as f:
            hardware_results = json.load(f)
        
        logger.info(f"Loaded {len(simulation_results)} simulation results and {len(hardware_results)} hardware results")
        return simulation_results, hardware_results
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def load_parameters(param_file: str) -> Dict[str, Any]:
    """
    Load calibration parameters from file.
    
    Args:
        param_file: Path to parameter file (JSON)
        
    Returns:
        Dictionary of parameters
    """
    logger.info(f"Loading parameters from {param_file}")
    
    try:
        with open(param_file, 'r') as f:
            parameters = json.load(f)
        
        logger.info(f"Loaded {len(parameters)} parameters")
        return parameters
    except Exception as e:
        logger.error(f"Error loading parameters: {str(e)}")
        raise

def save_parameters(parameters: Dict[str, Any], output_file: str) -> None:
    """
    Save calibration parameters to file.
    
    Args:
        parameters: Dictionary of parameters
        output_file: Path to output file (JSON)
    """
    logger.info(f"Saving parameters to {output_file}")
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(parameters, f, indent=2)
        
        logger.info(f"Parameters saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving parameters: {str(e)}")
        raise

def create_calibrator(
    calibrator_type: str,
    learning_rate: float = 0.1,
    max_iterations: int = 100,
    history_file: Optional[str] = None
) -> Union[BasicCalibrator, AdvancedCalibrator]:
    """
    Create a calibrator instance based on type.
    
    Args:
        calibrator_type: Type of calibrator to create
        learning_rate: Learning rate for calibration
        max_iterations: Maximum iterations for calibration
        history_file: Optional file to store calibration history
        
    Returns:
        Calibrator instance
    """
    logger.info(f"Creating {calibrator_type} calibrator")
    
    if calibrator_type == "basic":
        return BasicCalibrator(learning_rate=learning_rate, max_iterations=max_iterations)
    elif calibrator_type == "multi_parameter":
        return MultiParameterCalibrator(
            learning_rate=learning_rate, 
            max_iterations=max_iterations,
            history_file=history_file
        )
    elif calibrator_type == "bayesian":
        return BayesianOptimizationCalibrator(
            learning_rate=learning_rate, 
            max_iterations=max_iterations,
            history_file=history_file
        )
    elif calibrator_type == "neural_network":
        return NeuralNetworkCalibrator(
            learning_rate=learning_rate, 
            max_iterations=max_iterations,
            history_file=history_file
        )
    elif calibrator_type == "ensemble":
        return EnsembleCalibrator(
            learning_rate=learning_rate, 
            max_iterations=max_iterations,
            history_file=history_file
        )
    else:
        logger.warning(f"Unknown calibrator type: {calibrator_type}, falling back to BasicCalibrator")
        return BasicCalibrator(learning_rate=learning_rate, max_iterations=max_iterations)

def run_calibration(
    simulation_results: List[Dict[str, Any]],
    hardware_results: List[Dict[str, Any]],
    initial_parameters: Dict[str, Any],
    calibrator_type: str = "basic",
    learning_rate: float = 0.1,
    max_iterations: int = 100,
    history_file: Optional[str] = None,
    parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None
) -> Dict[str, Any]:
    """
    Run calibration on the provided data.
    
    Args:
        simulation_results: List of simulation result dictionaries
        hardware_results: List of hardware result dictionaries
        initial_parameters: Initial calibration parameters
        calibrator_type: Type of calibrator to use
        learning_rate: Learning rate for calibration
        max_iterations: Maximum iterations for calibration
        history_file: Optional file to store calibration history
        parameter_bounds: Optional bounds for parameters
        
    Returns:
        Dictionary with calibration results
    """
    logger.info(f"Running calibration with {calibrator_type} calibrator")
    
    # Create calibrator
    calibrator = create_calibrator(
        calibrator_type, learning_rate, max_iterations, history_file
    )
    
    # Run calibration
    calibrated_parameters = calibrator.calibrate(
        simulation_results, hardware_results, initial_parameters, parameter_bounds
    )
    
    # Evaluate calibration effectiveness
    evaluation = calibrator.evaluate_calibration(
        simulation_results, simulation_results, hardware_results
    )
    
    # Create result object
    result = {
        "calibrator_type": calibrator_type,
        "timestamp": datetime.now().isoformat(),
        "initial_parameters": initial_parameters,
        "calibrated_parameters": calibrated_parameters,
        "evaluation": evaluation
    }
    
    logger.info(f"Calibration completed: {evaluation['overall']}")
    return result

def run_parameter_discovery(
    simulation_results: List[Dict[str, Any]],
    hardware_results: List[Dict[str, Any]],
    initial_parameters: Dict[str, Any],
    sensitivity_threshold: float = 0.01,
    discovery_iterations: int = 100,
    exploration_range: float = 0.5,
    result_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run parameter discovery to identify sensitive parameters.
    
    Args:
        simulation_results: List of simulation result dictionaries
        hardware_results: List of hardware result dictionaries
        initial_parameters: Initial calibration parameters
        sensitivity_threshold: Threshold for parameter sensitivity
        discovery_iterations: Number of iterations for parameter discovery
        exploration_range: Range for parameter exploration
        result_file: Optional file to store discovery results
        
    Returns:
        Dictionary with discovery results
    """
    logger.info(f"Running parameter discovery with {discovery_iterations} iterations")
    
    # Create parameter discovery instance
    discovery = ParameterDiscovery(
        sensitivity_threshold=sensitivity_threshold,
        discovery_iterations=discovery_iterations,
        exploration_range=exploration_range,
        result_file=result_file
    )
    
    # Create error function for discovery
    def error_function(params):
        # Create a temporary BasicCalibrator to calculate error
        basic_calibrator = BasicCalibrator()
        return basic_calibrator._calculate_error(simulation_results, hardware_results, params)
    
    # Run discovery
    result = discovery.discover_parameters(
        error_function=error_function,
        initial_parameters=initial_parameters
    )
    
    logger.info(f"Parameter discovery completed with {len(result['sensitive_parameters'])} sensitive parameters")
    return result

def run_cross_validation(
    simulation_results: List[Dict[str, Any]],
    hardware_results: List[Dict[str, Any]],
    initial_parameters: Dict[str, Any],
    n_splits: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
    calibrator_type: str = "basic",
    result_file: Optional[str] = None,
    group_by: Optional[str] = None,
    visualization_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run cross-validation for calibration parameters.
    
    Args:
        simulation_results: List of simulation result dictionaries
        hardware_results: List of hardware result dictionaries
        initial_parameters: Initial calibration parameters
        n_splits: Number of cross-validation splits
        test_size: Proportion of data to use for validation
        random_state: Random seed for reproducibility
        calibrator_type: Type of calibrator to use
        result_file: Optional file to store validation results
        group_by: Optional field to group results by
        visualization_file: Optional file to save visualization
        
    Returns:
        Dictionary with cross-validation results
    """
    logger.info(f"Running cross-validation with {n_splits} splits")
    
    # Create cross-validator
    cross_validator = CalibrationCrossValidator(
        n_splits=n_splits,
        test_size=test_size,
        random_state=random_state,
        result_file=result_file,
        calibrator_type=calibrator_type
    )
    
    # Run cross-validation
    result = cross_validator.cross_validate(
        simulation_results=simulation_results,
        hardware_results=hardware_results,
        initial_parameters=initial_parameters,
        group_by=group_by
    )
    
    # Create visualization if requested
    if visualization_file and result["status"] == "success":
        cross_validator.visualize_results(
            output_file=visualization_file,
            result_id=result["id"]
        )
    
    logger.info(f"Cross-validation completed with status: {result['status']}")
    return result

def run_uncertainty_quantification(
    simulation_results: List[Dict[str, Any]],
    hardware_results: List[Dict[str, Any]],
    parameter_sets: List[Dict[str, Any]],
    confidence_level: float = 0.95,
    n_samples: int = 1000,
    result_file: Optional[str] = None,
    error_threshold: Optional[float] = None,
    perturbation_factor: float = 0.1,
    report_file: Optional[str] = None,
    report_format: str = "markdown"
) -> Dict[str, Any]:
    """
    Run uncertainty quantification for calibration parameters.
    
    Args:
        simulation_results: List of simulation result dictionaries
        hardware_results: List of hardware result dictionaries
        parameter_sets: List of parameter dictionaries
        confidence_level: Confidence level for intervals
        n_samples: Number of Monte Carlo samples
        result_file: Optional file to store uncertainty results
        error_threshold: Optional threshold for reliability estimation
        perturbation_factor: Factor for parameter perturbation
        report_file: Optional file to save report
        report_format: Format for report
        
    Returns:
        Dictionary with uncertainty results
    """
    logger.info(f"Running uncertainty quantification with {confidence_level} confidence level")
    
    # Create uncertainty quantifier
    uncertainty_quantifier = UncertaintyQuantifier(
        confidence_level=confidence_level,
        n_samples=n_samples,
        result_file=result_file
    )
    
    # Quantify parameter uncertainty
    parameter_uncertainty = uncertainty_quantifier.quantify_parameter_uncertainty(
        parameter_sets=parameter_sets
    )
    
    # Create error function for uncertainty propagation
    def error_function(params):
        # Create a temporary BasicCalibrator to calculate error
        basic_calibrator = BasicCalibrator()
        return basic_calibrator._calculate_error(simulation_results, hardware_results, params)
    
    # Propagate uncertainty
    propagation_result = uncertainty_quantifier.propagate_uncertainty(
        parameter_uncertainty=parameter_uncertainty["parameter_uncertainty"],
        simulation_results=simulation_results,
        error_function=error_function
    )
    
    # Run reliability estimation if threshold is provided
    if error_threshold is not None:
        reliability_result = uncertainty_quantifier.estimate_reliability(
            parameter_uncertainty=parameter_uncertainty["parameter_uncertainty"],
            simulation_results=simulation_results,
            error_threshold=error_threshold,
            error_function=error_function
        )
    else:
        reliability_result = None
    
    # Run sensitivity analysis
    sensitivity_result = uncertainty_quantifier.sensitivity_analysis(
        parameter_uncertainty=parameter_uncertainty["parameter_uncertainty"],
        simulation_results=simulation_results,
        error_function=error_function,
        perturbation_factor=perturbation_factor
    )
    
    # Generate report if requested
    if report_file:
        report = uncertainty_quantifier.generate_report(
            result_id=sensitivity_result["id"],
            format=report_format
        )
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(report_file)), exist_ok=True)
            
            with open(report_file, 'w') as f:
                f.write(report)
            
            logger.info(f"Uncertainty report saved to {report_file}")
        except Exception as e:
            logger.error(f"Error saving uncertainty report: {str(e)}")
    
    # Create result summary
    result = {
        "parameter_uncertainty": parameter_uncertainty,
        "propagation_result": propagation_result,
        "reliability_result": reliability_result,
        "sensitivity_result": sensitivity_result
    }
    
    logger.info(f"Uncertainty quantification completed")
    return result

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Calibration System Command Line Interface")
    
    # Common arguments
    parser.add_argument("--sim-file", type=str, required=True,
                        help="Path to simulation result file (JSON)")
    parser.add_argument("--hw-file", type=str, required=True,
                        help="Path to hardware result file (JSON)")
    parser.add_argument("--param-file", type=str, required=True,
                        help="Path to parameter file (JSON)")
    parser.add_argument("--output-file", type=str, default="calibrated_parameters.json",
                        help="Path to output file for calibrated parameters (JSON)")
    
    # Command selection
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Calibration command
    cal_parser = subparsers.add_parser("calibrate", help="Run calibration")
    cal_parser.add_argument("--calibrator", type=str, default="basic",
                          choices=["basic", "multi_parameter", "bayesian", "neural_network", "ensemble"],
                          help="Type of calibrator to use")
    cal_parser.add_argument("--learning-rate", type=float, default=0.1,
                          help="Learning rate for calibration")
    cal_parser.add_argument("--max-iterations", type=int, default=100,
                          help="Maximum iterations for calibration")
    cal_parser.add_argument("--history-file", type=str, default=None,
                          help="File to store calibration history")
    cal_parser.add_argument("--bounds-file", type=str, default=None,
                          help="File containing parameter bounds (JSON)")
    
    # Parameter discovery command
    disc_parser = subparsers.add_parser("discover", help="Run parameter discovery")
    disc_parser.add_argument("--sensitivity-threshold", type=float, default=0.01,
                           help="Threshold for parameter sensitivity")
    disc_parser.add_argument("--discovery-iterations", type=int, default=100,
                           help="Number of iterations for parameter discovery")
    disc_parser.add_argument("--exploration-range", type=float, default=0.5,
                           help="Range for parameter exploration")
    disc_parser.add_argument("--result-file", type=str, default=None,
                           help="File to store discovery results")
    
    # Cross-validation command
    cv_parser = subparsers.add_parser("cross-validate", help="Run cross-validation")
    cv_parser.add_argument("--n-splits", type=int, default=5,
                         help="Number of cross-validation splits")
    cv_parser.add_argument("--test-size", type=float, default=0.2,
                         help="Proportion of data to use for validation")
    cv_parser.add_argument("--random-state", type=int, default=42,
                         help="Random seed for reproducibility")
    cv_parser.add_argument("--calibrator", type=str, default="basic",
                         choices=["basic", "multi_parameter", "bayesian", "neural_network", "ensemble"],
                         help="Type of calibrator to use")
    cv_parser.add_argument("--result-file", type=str, default=None,
                         help="File to store validation results")
    cv_parser.add_argument("--group-by", type=str, default=None,
                         help="Field to group results by")
    cv_parser.add_argument("--visualization-file", type=str, default=None,
                         help="File to save visualization")
    
    # Uncertainty quantification command
    uq_parser = subparsers.add_parser("uncertainty", help="Run uncertainty quantification")
    uq_parser.add_argument("--parameter-sets-file", type=str, required=True,
                         help="File containing parameter sets for uncertainty quantification (JSON)")
    uq_parser.add_argument("--confidence-level", type=float, default=0.95,
                         help="Confidence level for intervals")
    uq_parser.add_argument("--n-samples", type=int, default=1000,
                         help="Number of Monte Carlo samples")
    uq_parser.add_argument("--result-file", type=str, default=None,
                         help="File to store uncertainty results")
    uq_parser.add_argument("--error-threshold", type=float, default=None,
                         help="Threshold for reliability estimation")
    uq_parser.add_argument("--perturbation-factor", type=float, default=0.1,
                         help="Factor for parameter perturbation")
    uq_parser.add_argument("--report-file", type=str, default=None,
                         help="File to save report")
    uq_parser.add_argument("--report-format", type=str, default="markdown",
                         choices=["text", "markdown", "json"],
                         help="Format for report")
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        # Load data
        simulation_results, hardware_results = load_data(args.sim_file, args.hw_file)
        
        # Load parameters
        initial_parameters = load_parameters(args.param_file)
        
        # Execute command
        if args.command == "calibrate":
            # Load parameter bounds if provided
            parameter_bounds = None
            if args.bounds_file:
                try:
                    with open(args.bounds_file, 'r') as f:
                        parameter_bounds = json.load(f)
                except Exception as e:
                    logger.error(f"Error loading parameter bounds: {str(e)}")
                    parameter_bounds = None
            
            # Run calibration
            result = run_calibration(
                simulation_results=simulation_results,
                hardware_results=hardware_results,
                initial_parameters=initial_parameters,
                calibrator_type=args.calibrator,
                learning_rate=args.learning_rate,
                max_iterations=args.max_iterations,
                history_file=args.history_file,
                parameter_bounds=parameter_bounds
            )
            
            # Save calibrated parameters
            save_parameters(result["calibrated_parameters"], args.output_file)
            
        elif args.command == "discover":
            # Run parameter discovery
            result = run_parameter_discovery(
                simulation_results=simulation_results,
                hardware_results=hardware_results,
                initial_parameters=initial_parameters,
                sensitivity_threshold=args.sensitivity_threshold,
                discovery_iterations=args.discovery_iterations,
                exploration_range=args.exploration_range,
                result_file=args.result_file
            )
            
            # Save optimal parameters
            if "optimal_parameters" in result:
                save_parameters(result["optimal_parameters"], args.output_file)
            
        elif args.command == "cross-validate":
            # Run cross-validation
            result = run_cross_validation(
                simulation_results=simulation_results,
                hardware_results=hardware_results,
                initial_parameters=initial_parameters,
                n_splits=args.n_splits,
                test_size=args.test_size,
                random_state=args.random_state,
                calibrator_type=args.calibrator,
                result_file=args.result_file,
                group_by=args.group_by,
                visualization_file=args.visualization_file
            )
            
            # Save recommended parameters
            if "status" in result and result["status"] == "success" and "recommended_parameters" in result:
                save_parameters(result["recommended_parameters"], args.output_file)
            
        elif args.command == "uncertainty":
            # Load parameter sets for uncertainty quantification
            try:
                with open(args.parameter_sets_file, 'r') as f:
                    parameter_sets = json.load(f)
            except Exception as e:
                logger.error(f"Error loading parameter sets: {str(e)}")
                return 1
            
            # Run uncertainty quantification
            result = run_uncertainty_quantification(
                simulation_results=simulation_results,
                hardware_results=hardware_results,
                parameter_sets=parameter_sets,
                confidence_level=args.confidence_level,
                n_samples=args.n_samples,
                result_file=args.result_file,
                error_threshold=args.error_threshold,
                perturbation_factor=args.perturbation_factor,
                report_file=args.report_file,
                report_format=args.report_format
            )
            
            # For uncertainty quantification, we don't save parameters to the output file
            
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1
        
        logger.info(f"Command {args.command} completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error executing command: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())