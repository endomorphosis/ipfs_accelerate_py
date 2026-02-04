#!/usr/bin/env python3
"""
Command-line tool for running simulation calibration with DuckDB integration.

This script provides a command-line interface for running various components
of the calibration system with integrated DuckDB storage for the Simulation 
Accuracy and Validation Framework.
"""

import argparse
import logging
import sys
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import uuid

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import calibration components
from data.duckdb.simulation_validation.calibration.basic_calibrator import BasicCalibrator
from data.duckdb.simulation_validation.calibration.advanced_calibrator import (
    AdvancedCalibrator, 
    MultiParameterCalibrator, 
    BayesianOptimizationCalibrator,
    NeuralNetworkCalibrator,
    EnsembleCalibrator
)
from data.duckdb.simulation_validation.calibration.parameter_discovery import (
    ParameterDiscovery, 
    AdaptiveCalibrationScheduler
)
from data.duckdb.simulation_validation.calibration.cross_validation import CalibrationCrossValidator
from data.duckdb.simulation_validation.calibration.uncertainty_quantification import UncertaintyQuantifier
from data.duckdb.simulation_validation.calibration.calibration_repository import DuckDBCalibrationRepository
from data.duckdb.simulation_validation.calibration.repository_adapter import (
    CalibratorDuckDBAdapter,
    CrossValidatorDuckDBAdapter, 
    ParameterDiscoveryDuckDBAdapter,
    UncertaintyQuantifierDuckDBAdapter,
    SchedulerDuckDBAdapter
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('calibration_duckdb.log')
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
    repository: Optional[DuckDBCalibrationRepository] = None,
    calibration_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> CalibratorDuckDBAdapter:
    """
    Create a calibrator instance with DuckDB integration based on type.
    
    Args:
        calibrator_type: Type of calibrator to create
        learning_rate: Learning rate for calibration
        max_iterations: Maximum iterations for calibration
        repository: Optional DuckDB repository for storing results
        calibration_id: Optional calibration ID
        metadata: Optional metadata for this calibration
        
    Returns:
        Calibrator adapter instance
    """
    logger.info(f"Creating {calibrator_type} calibrator")
    
    # Create base calibrator
    if calibrator_type == "basic":
        calibrator = BasicCalibrator(learning_rate=learning_rate, max_iterations=max_iterations)
    elif calibrator_type == "multi_parameter":
        calibrator = MultiParameterCalibrator(
            learning_rate=learning_rate, 
            max_iterations=max_iterations
        )
    elif calibrator_type == "bayesian":
        calibrator = BayesianOptimizationCalibrator(
            learning_rate=learning_rate, 
            max_iterations=max_iterations
        )
    elif calibrator_type == "neural_network":
        calibrator = NeuralNetworkCalibrator(
            learning_rate=learning_rate, 
            max_iterations=max_iterations
        )
    elif calibrator_type == "ensemble":
        calibrator = EnsembleCalibrator(
            learning_rate=learning_rate, 
            max_iterations=max_iterations
        )
    else:
        logger.warning(f"Unknown calibrator type: {calibrator_type}, falling back to BasicCalibrator")
        calibrator = BasicCalibrator(learning_rate=learning_rate, max_iterations=max_iterations)
    
    # If no repository provided, return the base calibrator
    if repository is None:
        return calibrator
    
    # Create adapter with repository integration
    adapter = CalibratorDuckDBAdapter(
        calibrator=calibrator,
        repository=repository,
        calibration_id=calibration_id or f"cal-{uuid.uuid4()}",
        metadata=metadata or {}
    )
    
    return adapter

def run_calibration(
    simulation_results: List[Dict[str, Any]],
    hardware_results: List[Dict[str, Any]],
    initial_parameters: Dict[str, Any],
    calibrator_type: str = "basic",
    learning_rate: float = 0.1,
    max_iterations: int = 100,
    parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    repository: Optional[DuckDBCalibrationRepository] = None,
    calibration_id: Optional[str] = None,
    hardware_id: Optional[str] = None,
    dataset_id: Optional[str] = None,
    simulation_id: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Run calibration with DuckDB integration.
    
    Args:
        simulation_results: List of simulation result dictionaries
        hardware_results: List of hardware result dictionaries
        initial_parameters: Initial parameter dictionary
        calibrator_type: Type of calibrator to use
        learning_rate: Learning rate for calibration
        max_iterations: Maximum iterations for calibration
        parameter_bounds: Optional dictionary mapping parameter names to (min, max) tuples
        repository: Optional DuckDB repository for storing results
        calibration_id: Optional calibration ID
        hardware_id: Optional hardware identifier
        dataset_id: Optional dataset identifier
        simulation_id: Optional simulation configuration identifier
        description: Optional description of this calibration
        tags: Optional list of tags for this calibration
        
    Returns:
        Dictionary with calibration results
    """
    logger.info(f"Running calibration with {calibrator_type} calibrator")
    
    # Create calibrator with repository integration if provided
    metadata = {
        "description": description or f"Calibration run with {calibrator_type}",
        "tags": tags or [calibrator_type],
        "hardware_id": hardware_id,
        "dataset_id": dataset_id,
        "simulation_id": simulation_id,
        "timestamp": datetime.now().isoformat()
    }
    
    calibrator = create_calibrator(
        calibrator_type, learning_rate, max_iterations, 
        repository, calibration_id, metadata
    )
    
    # Run calibration
    if isinstance(calibrator, CalibratorDuckDBAdapter):
        # Use the adapter to run calibration and store results
        result = calibrator.calibrate(
            simulation_results, hardware_results, initial_parameters, parameter_bounds,
            dataset_id, hardware_id, simulation_id
        )
    else:
        # Just run calibration without storage
        start_time = time.time()
        calibrated_parameters = calibrator.calibrate(
            simulation_results, hardware_results, initial_parameters, parameter_bounds
        )
        runtime = time.time() - start_time
        
        # Calculate error before and after
        error_before = calibrator._calculate_error(simulation_results, hardware_results, initial_parameters)
        error_after = calibrator._calculate_error(simulation_results, hardware_results, calibrated_parameters)
        
        # Create result object
        result = {
            "calibrator_type": calibrator_type,
            "parameters": calibrated_parameters,
            "error_before": error_before,
            "error_after": error_after,
            "improvement_percent": ((error_before - error_after) / error_before * 100) if error_before > 0 else 0.0,
            "runtime_seconds": runtime
        }
    
    logger.info(f"Calibration completed: {result}")
    return result

def run_cross_validation(
    simulation_results: List[Dict[str, Any]],
    hardware_results: List[Dict[str, Any]],
    initial_parameters: Dict[str, Any],
    n_splits: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
    calibrator_type: str = "basic",
    repository: Optional[DuckDBCalibrationRepository] = None,
    validation_id: Optional[str] = None,
    calibration_id: Optional[str] = None,
    dataset_id: Optional[str] = None,
    group_by: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Run cross-validation with DuckDB integration.
    
    Args:
        simulation_results: List of simulation result dictionaries
        hardware_results: List of hardware result dictionaries
        initial_parameters: Initial parameter dictionary
        n_splits: Number of cross-validation splits
        test_size: Proportion of data to use for validation
        random_state: Random seed for reproducibility
        calibrator_type: Type of calibrator to use
        repository: Optional DuckDB repository for storing results
        validation_id: Optional validation ID
        calibration_id: Optional calibration ID to associate with
        dataset_id: Optional dataset identifier
        group_by: Optional field to group results by
        description: Optional description of this validation
        tags: Optional list of tags for this validation
        
    Returns:
        Dictionary with cross-validation results
    """
    logger.info(f"Running cross-validation with {n_splits} splits and {calibrator_type} calibrator")
    
    # Create cross-validator
    cross_validator = CalibrationCrossValidator(
        n_splits=n_splits,
        test_size=test_size,
        random_state=random_state,
        calibrator_type=calibrator_type
    )
    
    # Run cross-validation with repository integration if provided
    if repository is not None:
        metadata = {
            "description": description or f"Cross-validation with {calibrator_type}",
            "tags": tags or [calibrator_type, "cross-validation"],
            "dataset_id": dataset_id,
            "timestamp": datetime.now().isoformat()
        }
        
        # Create adapter
        adapter = CrossValidatorDuckDBAdapter(
            cross_validator=cross_validator,
            repository=repository,
            validation_id=validation_id,
            metadata=metadata
        )
        
        # Run cross-validation with adapter
        result = adapter.cross_validate(
            simulation_results, hardware_results, initial_parameters,
            None, group_by, calibration_id, dataset_id
        )
    else:
        # Run cross-validation without repository integration
        result = cross_validator.cross_validate(
            simulation_results, hardware_results, initial_parameters,
            None, group_by
        )
    
    logger.info(f"Cross-validation completed with status: {result.get('status')}")
    return result

def run_parameter_discovery(
    simulation_results: List[Dict[str, Any]],
    hardware_results: List[Dict[str, Any]],
    initial_parameters: Dict[str, Any],
    sensitivity_threshold: float = 0.01,
    discovery_iterations: int = 100,
    exploration_range: float = 0.5,
    repository: Optional[DuckDBCalibrationRepository] = None,
    analysis_id: Optional[str] = None,
    calibration_id: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Run parameter discovery with DuckDB integration.
    
    Args:
        simulation_results: List of simulation result dictionaries
        hardware_results: List of hardware result dictionaries
        initial_parameters: Initial parameter dictionary
        sensitivity_threshold: Threshold for parameter sensitivity
        discovery_iterations: Number of iterations for parameter discovery
        exploration_range: Range for parameter exploration
        repository: Optional DuckDB repository for storing results
        analysis_id: Optional analysis ID
        calibration_id: Optional calibration ID to associate with
        description: Optional description of this analysis
        tags: Optional list of tags for this analysis
        
    Returns:
        Dictionary with parameter discovery results
    """
    logger.info(f"Running parameter discovery with {discovery_iterations} iterations")
    
    # Create parameter discovery instance
    discovery = ParameterDiscovery(
        sensitivity_threshold=sensitivity_threshold,
        discovery_iterations=discovery_iterations,
        exploration_range=exploration_range
    )
    
    # Create error function for discovery
    def error_function(params):
        # Create a temporary BasicCalibrator to calculate error
        basic_calibrator = BasicCalibrator()
        return basic_calibrator._calculate_error(simulation_results, hardware_results, params)
    
    # Run discovery with repository integration if provided
    if repository is not None:
        metadata = {
            "description": description or "Parameter discovery analysis",
            "tags": tags or ["parameter-discovery"],
            "timestamp": datetime.now().isoformat()
        }
        
        # Create adapter
        adapter = ParameterDiscoveryDuckDBAdapter(
            parameter_discovery=discovery,
            repository=repository,
            analysis_id=analysis_id,
            metadata=metadata
        )
        
        # Run discovery with adapter
        result = adapter.discover_parameters(
            error_function, initial_parameters, None, calibration_id
        )
    else:
        # Run discovery without repository integration
        result = discovery.discover_parameters(
            error_function=error_function,
            initial_parameters=initial_parameters
        )
    
    logger.info(f"Parameter discovery completed with {len(result.get('sensitive_parameters', []))} sensitive parameters")
    return result

def run_uncertainty_quantification(
    parameter_sets: List[Dict[str, Any]],
    simulation_results: List[Dict[str, Any]],
    hardware_results: List[Dict[str, Any]],
    confidence_level: float = 0.95,
    n_samples: int = 1000,
    perturbation_factor: float = 0.1,
    repository: Optional[DuckDBCalibrationRepository] = None,
    analysis_id: Optional[str] = None,
    calibration_id: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Run uncertainty quantification with DuckDB integration.
    
    Args:
        parameter_sets: List of parameter dictionaries
        simulation_results: List of simulation result dictionaries
        hardware_results: List of hardware result dictionaries
        confidence_level: Confidence level for intervals
        n_samples: Number of Monte Carlo samples
        perturbation_factor: Factor for parameter perturbation
        repository: Optional DuckDB repository for storing results
        analysis_id: Optional analysis ID
        calibration_id: Optional calibration ID to associate with
        description: Optional description of this analysis
        tags: Optional list of tags for this analysis
        
    Returns:
        Dictionary with uncertainty results
    """
    logger.info(f"Running uncertainty quantification with {confidence_level} confidence level")
    
    # Create uncertainty quantifier
    uncertainty_quantifier = UncertaintyQuantifier(
        confidence_level=confidence_level,
        n_samples=n_samples
    )
    
    # Create error function for uncertainty propagation
    def error_function(params):
        # Create a temporary BasicCalibrator to calculate error
        basic_calibrator = BasicCalibrator()
        return basic_calibrator._calculate_error(simulation_results, hardware_results, params)
    
    # Run uncertainty quantification with repository integration if provided
    if repository is not None:
        metadata = {
            "description": description or "Uncertainty quantification analysis",
            "tags": tags or ["uncertainty-quantification"],
            "timestamp": datetime.now().isoformat()
        }
        
        # Create adapter
        adapter = UncertaintyQuantifierDuckDBAdapter(
            uncertainty_quantifier=uncertainty_quantifier,
            repository=repository,
            analysis_id=analysis_id,
            metadata=metadata
        )
        
        # Run uncertainty quantification with adapter
        param_uncertainty = adapter.quantify_parameter_uncertainty(
            parameter_sets, calibration_id
        )
        
        # Run sensitivity analysis with adapter
        sensitivity_result = adapter.sensitivity_analysis(
            param_uncertainty.get("parameter_uncertainty", {}),
            simulation_results,
            error_function,
            perturbation_factor,
            calibration_id
        )
        
        # Combine results
        result = {
            "parameter_uncertainty": param_uncertainty,
            "sensitivity_analysis": sensitivity_result,
            "analysis_id": adapter.analysis_id
        }
    else:
        # Run uncertainty quantification without repository integration
        param_uncertainty = uncertainty_quantifier.quantify_parameter_uncertainty(
            parameter_sets
        )
        
        # Run sensitivity analysis
        sensitivity_result = uncertainty_quantifier.sensitivity_analysis(
            param_uncertainty.get("parameter_uncertainty", {}),
            simulation_results,
            error_function,
            perturbation_factor
        )
        
        # Combine results
        result = {
            "parameter_uncertainty": param_uncertainty,
            "sensitivity_analysis": sensitivity_result
        }
    
    logger.info(f"Uncertainty quantification completed")
    return result

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Calibration System with DuckDB Integration")
    
    # Common arguments
    parser.add_argument("--sim-file", type=str, required=True,
                        help="Path to simulation result file (JSON)")
    parser.add_argument("--hw-file", type=str, required=True,
                        help="Path to hardware result file (JSON)")
    parser.add_argument("--param-file", type=str, required=True,
                        help="Path to parameter file (JSON)")
    parser.add_argument("--output-file", type=str, default="calibrated_parameters.json",
                        help="Path to output file for calibrated parameters (JSON)")
    parser.add_argument("--db-path", type=str, default="calibration.duckdb",
                        help="Path to DuckDB database file")
    parser.add_argument("--hardware-id", type=str, default=None,
                        help="Hardware identifier (e.g., 'cuda', 'webgpu')")
    parser.add_argument("--dataset-id", type=str, default=None,
                        help="Dataset identifier")
    parser.add_argument("--simulation-id", type=str, default=None,
                        help="Simulation configuration identifier")
    parser.add_argument("--description", type=str, default=None,
                        help="Description of this calibration/analysis")
    parser.add_argument("--tags", type=str, default=None,
                        help="Comma-separated list of tags")
    
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
    cal_parser.add_argument("--calibration-id", type=str, default=None,
                          help="Calibration ID (generated if not provided)")
    cal_parser.add_argument("--bounds-file", type=str, default=None,
                          help="File containing parameter bounds (JSON)")
    cal_parser.add_argument("--no-db", action="store_true",
                          help="Do not use DuckDB for storage")
    
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
    cv_parser.add_argument("--validation-id", type=str, default=None,
                         help="Validation ID (generated if not provided)")
    cv_parser.add_argument("--calibration-id", type=str, default=None,
                         help="Calibration ID to associate with")
    cv_parser.add_argument("--group-by", type=str, default=None,
                         help="Field to group results by")
    cv_parser.add_argument("--no-db", action="store_true",
                         help="Do not use DuckDB for storage")
    
    # Parameter discovery command
    disc_parser = subparsers.add_parser("discover", help="Run parameter discovery")
    disc_parser.add_argument("--sensitivity-threshold", type=float, default=0.01,
                           help="Threshold for parameter sensitivity")
    disc_parser.add_argument("--discovery-iterations", type=int, default=100,
                           help="Number of iterations for parameter discovery")
    disc_parser.add_argument("--exploration-range", type=float, default=0.5,
                           help="Range for parameter exploration")
    disc_parser.add_argument("--analysis-id", type=str, default=None,
                           help="Analysis ID (generated if not provided)")
    disc_parser.add_argument("--calibration-id", type=str, default=None,
                           help="Calibration ID to associate with")
    disc_parser.add_argument("--no-db", action="store_true",
                           help="Do not use DuckDB for storage")
    
    # Uncertainty quantification command
    uq_parser = subparsers.add_parser("uncertainty", help="Run uncertainty quantification")
    uq_parser.add_argument("--parameter-sets-file", type=str, required=True,
                         help="File containing parameter sets for uncertainty quantification (JSON)")
    uq_parser.add_argument("--confidence-level", type=float, default=0.95,
                         help="Confidence level for intervals")
    uq_parser.add_argument("--n-samples", type=int, default=1000,
                         help="Number of Monte Carlo samples")
    uq_parser.add_argument("--perturbation-factor", type=float, default=0.1,
                         help="Factor for parameter perturbation")
    uq_parser.add_argument("--analysis-id", type=str, default=None,
                         help="Analysis ID (generated if not provided)")
    uq_parser.add_argument("--calibration-id", type=str, default=None,
                         help="Calibration ID to associate with")
    uq_parser.add_argument("--no-db", action="store_true",
                         help="Do not use DuckDB for storage")
    
    # Sample data generation command
    sample_parser = subparsers.add_parser("generate-sample", help="Generate sample data in DuckDB")
    sample_parser.add_argument("--num-calibrations", type=int, default=5,
                             help="Number of sample calibrations to generate")
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        if args.command == "generate-sample":
            # Create repository
            repository = DuckDBCalibrationRepository(db_path=args.db_path)
            
            # Generate sample data
            repository.generate_sample_data(num_calibrations=args.num_calibrations)
            
            logger.info(f"Sample data generation completed")
            return 0
        
        # Load data
        simulation_results, hardware_results = load_data(args.sim_file, args.hw_file)
        
        # Load parameters
        initial_parameters = load_parameters(args.param_file)
        
        # Process tags if provided
        tags = args.tags.split(",") if args.tags else None
        
        # Create repository if needed
        repository = None
        if not getattr(args, "no_db", False):
            repository = DuckDBCalibrationRepository(db_path=args.db_path)
        
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
                parameter_bounds=parameter_bounds,
                repository=repository,
                calibration_id=args.calibration_id,
                hardware_id=args.hardware_id,
                dataset_id=args.dataset_id,
                simulation_id=args.simulation_id,
                description=args.description,
                tags=tags
            )
            
            # Save calibrated parameters
            if "parameters" in result:
                save_parameters(result["parameters"], args.output_file)
            else:
                save_parameters(result.get("calibrated_parameters", {}), args.output_file)
            
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
                repository=repository,
                validation_id=args.validation_id,
                calibration_id=args.calibration_id,
                dataset_id=args.dataset_id,
                group_by=args.group_by,
                description=args.description,
                tags=tags
            )
            
            # Save recommended parameters if available
            if "recommended_parameters" in result:
                save_parameters(result["recommended_parameters"], args.output_file)
            
        elif args.command == "discover":
            # Run parameter discovery
            result = run_parameter_discovery(
                simulation_results=simulation_results,
                hardware_results=hardware_results,
                initial_parameters=initial_parameters,
                sensitivity_threshold=args.sensitivity_threshold,
                discovery_iterations=args.discovery_iterations,
                exploration_range=args.exploration_range,
                repository=repository,
                analysis_id=args.analysis_id,
                calibration_id=args.calibration_id,
                description=args.description,
                tags=tags
            )
            
            # Save optimal parameters if available
            if "optimal_parameters" in result:
                save_parameters(result["optimal_parameters"], args.output_file)
            
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
                parameter_sets=parameter_sets,
                simulation_results=simulation_results,
                hardware_results=hardware_results,
                confidence_level=args.confidence_level,
                n_samples=args.n_samples,
                perturbation_factor=args.perturbation_factor,
                repository=repository,
                analysis_id=args.analysis_id,
                calibration_id=args.calibration_id,
                description=args.description,
                tags=tags
            )
            
            # For uncertainty quantification, we don't save parameters to the output file
            
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1
        
        # Clean up repository connection
        if repository is not None:
            repository.close()
        
        logger.info(f"Command {args.command} completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error executing command: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())