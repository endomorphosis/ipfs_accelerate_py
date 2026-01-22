#!/usr/bin/env python3
"""
Simulation Accuracy and Validation Framework - Main Integration Module

This module serves as the primary entry point and integration layer for the Simulation
Accuracy and Validation Framework. It orchestrates the various components of the framework
to provide a comprehensive system for validating, calibrating, and monitoring hardware
simulation accuracy.
"""

import os
import sys
import logging
import datetime
import json
import uuid
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("simulation_validation_framework")

# Import base classes
from duckdb_api.simulation_validation.core.base import (
    SimulationResult, 
    HardwareResult, 
    ValidationResult,
    SimulationValidator,
    SimulationCalibrator,
    DriftDetector,
    ValidationReporter,
    SimulationAccuracyFramework
)

# Import core methodology and components
from duckdb_api.simulation_validation.methodology import ValidationMethodology
from duckdb_api.simulation_validation.comparison.comparison_pipeline import ComparisonPipeline
from duckdb_api.simulation_validation.statistical.statistical_validator import StatisticalValidator

# Import enhanced statistical validator
try:
    from duckdb_api.simulation_validation.statistical.enhanced_statistical_validator import EnhancedStatisticalValidator
    enhanced_validator_available = True
except ImportError:
    logger.warning("EnhancedStatisticalValidator not available, using basic statistical validator")
    enhanced_validator_available = False

# Optional imports - these may not exist yet but will be implemented later
try:
    from duckdb_api.simulation_validation.calibration.basic_calibrator import BasicSimulationCalibrator
    calibrator_available = True
except ImportError:
    logger.warning("BasicSimulationCalibrator not available, calibration functions will be limited")
    calibrator_available = False

try:
    from duckdb_api.simulation_validation.calibration.advanced_calibrator import AdvancedSimulationCalibrator
    advanced_calibrator_available = True
except ImportError:
    logger.warning("AdvancedSimulationCalibrator not available, advanced calibration functions will be limited")
    advanced_calibrator_available = False

try:
    from duckdb_api.simulation_validation.calibration.parameter_discovery import AutomaticParameterDiscovery
    parameter_discovery_available = True
except ImportError:
    logger.warning("AutomaticParameterDiscovery not available, parameter discovery functions will be limited")
    parameter_discovery_available = False

try:
    from duckdb_api.simulation_validation.drift_detection.basic_detector import BasicDriftDetector
    drift_detector_available = True
except ImportError:
    logger.warning("BasicDriftDetector not available, drift detection functions will be limited")
    drift_detector_available = False

try:
    from duckdb_api.simulation_validation.visualization.validation_reporter import ValidationReporterImpl
    reporter_available = True
except ImportError:
    logger.warning("ValidationReporterImpl not available, reporting functions will be limited")
    reporter_available = False

try:
    from duckdb_api.simulation_validation.visualization.validation_visualizer import ValidationVisualizer
    visualizer_available = True
except ImportError:
    logger.warning("ValidationVisualizer not available, visualization functions will be limited")
    visualizer_available = False

class SimulationValidationFramework:
    """
    Main integration class for the Simulation Accuracy and Validation Framework.
    
    This class brings together all components of the framework to provide a
    comprehensive system for validating simulation accuracy, calibrating simulation
    parameters, detecting drift in simulation accuracy, and generating reports.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Simulation Validation Framework with configuration.
        
        Args:
            config_path: Path to configuration file (JSON format)
        """
        # Load configuration
        self.config = self._load_configuration(config_path)
        
        # Initialize core components
        self.methodology = ValidationMethodology(self.config.get("methodology", {}))
        self.comparison_pipeline = ComparisonPipeline(self.config.get("comparison_pipeline", {}))
        
        # Use enhanced statistical validator if available and enabled
        use_enhanced_validator = self.config.get("use_enhanced_validator", True)
        if enhanced_validator_available and use_enhanced_validator:
            self.statistical_validator = EnhancedStatisticalValidator(self.config.get("statistical_validator", {}))
            logger.info("Using enhanced statistical validator with advanced metrics")
        else:
            self.statistical_validator = StatisticalValidator(self.config.get("statistical_validator", {}))
            logger.info("Using basic statistical validator")
        
        # Initialize optional components if available
        self.calibrator = None
        self.parameter_discovery = None
        
        # Initialize calibrator based on available implementations and configuration
        if self.config.get("enable_calibration", True):
            use_advanced_calibrator = self.config.get("use_advanced_calibrator", True)
            if advanced_calibrator_available and use_advanced_calibrator:
                logger.info("Using advanced simulation calibrator with enhanced techniques")
                self.calibrator = AdvancedSimulationCalibrator(self.config.get("calibrator", {}))
            elif calibrator_available:
                logger.info("Using basic simulation calibrator")
                self.calibrator = BasicSimulationCalibrator(self.config.get("calibrator", {}))
        
        # Initialize parameter discovery if available
        if parameter_discovery_available and self.config.get("enable_parameter_discovery", True):
            logger.info("Initializing automatic parameter discovery")
            self.parameter_discovery = AutomaticParameterDiscovery(
                self.config.get("parameter_discovery", {})
            )
        
        self.drift_detector = None
        if drift_detector_available and self.config.get("enable_drift_detection", True):
            self.drift_detector = BasicDriftDetector(self.config.get("drift_detector", {}))
        
        self.reporter = None
        if reporter_available and self.config.get("enable_reporting", True):
            self.reporter = ValidationReporterImpl(self.config.get("reporter", {}))
        
        # Initialize visualizer if available
        self.visualizer = None
        if visualizer_available and self.config.get("enable_visualization", True):
            self.visualizer = ValidationVisualizer(self.config.get("visualizer", {}))
        
        # Initialize database connection if specified
        self.db_api = None
        if "database" in self.config and self.config["database"].get("enabled", False):
            self._initialize_database()
        
        # Create the core framework class that brings everything together
        self.framework = SimulationAccuracyFramework(
            validator=self.statistical_validator,
            calibrator=self.calibrator,
            drift_detector=self.drift_detector,
            reporter=self.reporter,
            visualizer=self.visualizer,
            db_api=self.db_api,
            parameter_discovery=self.parameter_discovery
        )
        
        logger.info("Simulation Validation Framework initialized")
    
    def validate(
        self,
        simulation_results: List[SimulationResult],
        hardware_results: List[HardwareResult],
        protocol: str = "standard"
    ) -> List[ValidationResult]:
        """
        Validate simulation results against hardware results.
        
        Args:
            simulation_results: List of simulation results to validate
            hardware_results: List of hardware results to compare against
            protocol: Validation protocol to use (standard, comprehensive, or minimal)
            
        Returns:
            List of validation results
        """
        logger.info(f"Starting validation using {protocol} protocol")
        
        # Apply validation protocol
        protocol_config = self.methodology.get_validation_protocol(protocol)
        metrics_to_validate = protocol_config["metrics"]
        statistical_metrics = protocol_config["statistical_metrics"]
        
        # Update validator configuration for this run
        validator_config = self.statistical_validator.config.copy()
        validator_config["metrics_to_validate"] = metrics_to_validate
        validator_config["error_metrics"] = statistical_metrics
        self.statistical_validator.config = validator_config
        
        # Run comparison pipeline
        validation_results = self.comparison_pipeline.run_pipeline(
            simulation_results, hardware_results)
        
        if not validation_results:
            logger.warning("No validation results generated")
            return []
        
        logger.info(f"Generated {len(validation_results)} validation results")
        
        # Store results in database if available
        if self.db_api:
            self._store_validation_results(validation_results)
        
        return validation_results
    
    def calibrate(
        self,
        validation_results: List[ValidationResult],
        simulation_parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calibrate simulation parameters based on validation results.
        
        Args:
            validation_results: List of validation results to use for calibration
            simulation_parameters: Current simulation parameters
            
        Returns:
            Updated simulation parameters
        """
        if not self.calibrator:
            logger.warning("Calibrator not available, cannot perform calibration")
            return simulation_parameters
        
        logger.info("Starting calibration process")
        
        # Run calibration
        updated_parameters = self.calibrator.calibrate(
            validation_results, simulation_parameters)
        
        # Calculate improvement
        if validation_results:
            # Apply calibration to last simulation result
            last_sim_result = validation_results[-1].simulation_result
            calibrated_result = self.calibrator.apply_calibration(
                last_sim_result, updated_parameters)
            
            # Revalidate with calibrated result
            recalibrated_validation = self.statistical_validator.validate(
                calibrated_result, validation_results[-1].hardware_result)
            
            # Evaluate improvement
            improvement = self.calibrator.evaluate_calibration(
                [validation_results[-1]], [recalibrated_validation])
            
            logger.info(f"Calibration completed with {improvement['overall']['relative_improvement']:.2f}% overall improvement")
            
            # Store calibration in database if available
            if self.db_api:
                self._store_calibration_results(
                    validation_results, 
                    simulation_parameters, 
                    updated_parameters, 
                    improvement
                )
        else:
            logger.warning("No validation results provided for calibration")
        
        return updated_parameters
    
    def detect_drift(
        self,
        historical_validation_results: List[ValidationResult],
        new_validation_results: List[ValidationResult]
    ) -> Dict[str, Any]:
        """
        Detect drift in simulation accuracy.
        
        Args:
            historical_validation_results: Historical validation results
            new_validation_results: New validation results
            
        Returns:
            Drift detection results
        """
        if not self.drift_detector:
            logger.warning("Drift detector not available, cannot perform drift detection")
            return {"status": "error", "message": "Drift detector not available"}
        
        logger.info("Starting drift detection")
        
        # Run drift detection
        drift_results = self.drift_detector.detect_drift(
            historical_validation_results, new_validation_results)
        
        # Store drift detection results in database if available
        if self.db_api and drift_results.get("status") == "success":
            self._store_drift_detection_results(drift_results)
        
        return drift_results
    
    def generate_report(
        self,
        validation_results: List[ValidationResult],
        format: str = "html",
        include_visualizations: bool = True,
        output_path: Optional[str] = None,
        hardware_id: Optional[str] = None,
        model_id: Optional[str] = None
    ) -> str:
        """
        Generate a validation report.
        
        Args:
            validation_results: List of validation results to include in the report
            format: Output format (html, markdown, etc.)
            include_visualizations: Whether to include visualizations
            output_path: Path to save the report to (if None, returns the report content)
            hardware_id: Optional filter for hardware ID
            model_id: Optional filter for model ID
            
        Returns:
            Report content (if output_path is None) or path to saved report
        """
        if not validation_results:
            return "No validation results to report"
        
        # Generate report content
        if self.reporter:
            logger.info(f"Generating report in {format} format")
            
            if output_path:
                return self.reporter.export_report(
                    validation_results,
                    output_path,
                    format=format,
                    include_visualizations=include_visualizations
                )
            else:
                return self.reporter.generate_report(
                    validation_results,
                    format=format,
                    include_visualizations=include_visualizations
                )
        else:
            # Fall back to methodology's report generator
            logger.info(f"Reporter not available, using methodology's report generator")
            
            # Filter results if needed
            filtered_results = validation_results
            if hardware_id or model_id:
                filtered_results = []
                for result in validation_results:
                    if ((hardware_id is None or result.hardware_result.hardware_id == hardware_id) and
                        (model_id is None or result.hardware_result.model_id == model_id)):
                        filtered_results.append(result)
            
            report = self.methodology.generate_validation_report(
                filtered_results,
                hardware_id=hardware_id,
                model_id=model_id,
                report_format=format
            )
            
            if output_path:
                try:
                    with open(output_path, 'w') as f:
                        f.write(report)
                    return output_path
                except Exception as e:
                    logger.error(f"Error saving report to {output_path}: {e}")
                    return report
            else:
                return report
    
    def summarize_validation(self, validation_results: List[ValidationResult]) -> Dict[str, Any]:
        """
        Generate a summary of validation results.
        
        Args:
            validation_results: List of validation results
            
        Returns:
            Dictionary with summary statistics
        """
        return self.statistical_validator.summarize_validation(validation_results)
    
    def calculate_confidence(
        self,
        validation_results: List[ValidationResult],
        hardware_id: str,
        model_id: str
    ) -> Dict[str, Any]:
        """
        Calculate confidence score for simulation accuracy.
        
        Args:
            validation_results: List of validation results
            hardware_id: Hardware identifier
            model_id: Model identifier
            
        Returns:
            Dictionary with confidence metrics
        """
        return self.statistical_validator.calculate_confidence_score(
            validation_results, hardware_id, model_id)
    
    def check_calibration_needed(
        self,
        validation_results: List[ValidationResult],
        hardware_id: str,
        model_id: str
    ) -> Dict[str, Any]:
        """
        Check if calibration is needed based on validation results.
        
        Args:
            validation_results: List of validation results
            hardware_id: Hardware identifier
            model_id: Model identifier
            
        Returns:
            Dictionary with calibration recommendation
        """
        return self.methodology.check_calibration_needed(
            validation_results, hardware_id, model_id)
    
    def check_drift_detection_needed(
        self,
        validation_results: List[ValidationResult],
        hardware_id: str,
        model_id: str
    ) -> Dict[str, Any]:
        """
        Check if drift detection is needed based on validation results.
        
        Args:
            validation_results: List of validation results
            hardware_id: Hardware identifier
            model_id: Model identifier
            
        Returns:
            Dictionary with drift detection recommendation
        """
        return self.methodology.check_drift_detection_needed(
            validation_results, hardware_id, model_id)
    
    def discover_parameters(
        self,
        validation_results: List[ValidationResult]
    ) -> Dict[str, Any]:
        """
        Discover important parameters and their sensitivity.
        
        This method uses the AutomaticParameterDiscovery to analyze validation results
        and identify parameters that have significant impact on simulation accuracy.
        It also analyzes parameter sensitivity to different conditions (batch size,
        precision, etc.) and provides recommendations for calibration.
        
        Args:
            validation_results: List of validation results for analysis
            
        Returns:
            Dictionary of discovered parameters and their importance
        """
        if not self.parameter_discovery:
            logger.warning("Parameter discovery not available, cannot discover parameters")
            return {
                "status": "error",
                "message": "Parameter discovery not available"
            }
        
        if not validation_results:
            logger.warning("No validation results provided for parameter discovery")
            return {
                "status": "error",
                "message": "No validation results provided"
            }
        
        logger.info("Starting parameter discovery")
        
        # Run parameter discovery
        parameter_recommendations = self.parameter_discovery.discover_parameters(validation_results)
        
        # Store results in database if available
        if self.db_api and parameter_recommendations:
            self._store_parameter_discovery_results(parameter_recommendations)
        
        return parameter_recommendations
    
    def analyze_parameter_sensitivity(
        self,
        validation_results: List[ValidationResult],
        parameter_name: str
    ) -> Dict[str, Any]:
        """
        Analyze sensitivity of a specific parameter in detail.
        
        This method performs a detailed sensitivity analysis of a specific parameter,
        examining how it affects simulation accuracy under different conditions such
        as batch size and precision.
        
        Args:
            validation_results: List of validation results
            parameter_name: Name of the parameter to analyze
            
        Returns:
            Dictionary of sensitivity analysis results
        """
        if not self.parameter_discovery:
            logger.warning("Parameter discovery not available, cannot analyze parameter sensitivity")
            return {
                "status": "error",
                "message": "Parameter discovery not available"
            }
        
        if not validation_results:
            logger.warning("No validation results provided for parameter sensitivity analysis")
            return {
                "status": "error",
                "message": "No validation results provided"
            }
        
        logger.info(f"Analyzing sensitivity of parameter: {parameter_name}")
        
        # Run parameter sensitivity analysis
        sensitivity_results = self.parameter_discovery.analyze_parameter_sensitivity(
            validation_results, parameter_name
        )
        
        return sensitivity_results
    
    def generate_parameter_insight_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive report with insights about parameters.
        
        This method creates a detailed report containing parameter importance,
        sensitivity, and relationships, along with optimization recommendations
        and key findings.
        
        Returns:
            Dictionary with detailed insights and recommendations
        """
        if not self.parameter_discovery:
            logger.warning("Parameter discovery not available, cannot generate parameter insight report")
            return {
                "status": "error",
                "message": "Parameter discovery not available"
            }
        
        logger.info("Generating parameter insight report")
        
        # Generate parameter insight report
        insight_report = self.parameter_discovery.generate_insight_report()
        
        return insight_report
            
    def visualize_mape_comparison(
        self,
        validation_results: List[ValidationResult],
        metric_name: str = "all",
        hardware_ids: Optional[List[str]] = None,
        model_ids: Optional[List[str]] = None,
        sort_by: str = "hardware",
        interactive: Optional[bool] = None,
        output_path: Optional[str] = None,
        title: Optional[str] = None
    ) -> Union[str, None]:
        """
        Create a chart comparing MAPE values across different hardware/models.
        
        Args:
            validation_results: List of validation results
            metric_name: Name of the metric to visualize, or "all" for average
            hardware_ids: List of hardware IDs to include (if None, include all)
            model_ids: List of model IDs to include (if None, include all)
            sort_by: Sort results by "hardware", "model", or "value"
            interactive: Whether to create an interactive plot
            output_path: Path to save the chart
            title: Custom title for the chart
            
        Returns:
            Path to the saved chart if output_path is provided, otherwise HTML content
        """
        if not self.visualizer:
            logger.warning("Visualizer not available, cannot create MAPE comparison chart")
            return None
            
        logger.info(f"Creating MAPE comparison visualization for metric '{metric_name}'")
        return self.visualizer.create_mape_comparison_chart(
            validation_results,
            metric_name=metric_name,
            hardware_ids=hardware_ids,
            model_ids=model_ids,
            sort_by=sort_by,
            interactive=interactive,
            output_path=output_path,
            title=title
        )
    
    def visualize_metric_comparison(
        self,
        validation_results: List[ValidationResult],
        hardware_id: str,
        model_id: str,
        metrics: Optional[List[str]] = None,
        show_absolute_values: bool = False,
        interactive: Optional[bool] = None,
        output_path: Optional[str] = None,
        title: Optional[str] = None
    ) -> Union[str, None]:
        """
        Create a chart comparing simulation vs hardware values for specific metrics.
        
        Args:
            validation_results: List of validation results
            hardware_id: Hardware ID to visualize
            model_id: Model ID to visualize
            metrics: List of metrics to include (if None, include common metrics)
            show_absolute_values: Whether to show absolute values or normalized
            interactive: Whether to create an interactive plot
            output_path: Path to save the chart
            title: Custom title for the chart
            
        Returns:
            Path to the saved chart if output_path is provided, otherwise HTML content
        """
        if not self.visualizer:
            logger.warning("Visualizer not available, cannot create metric comparison chart")
            return None
            
        logger.info(f"Creating metric comparison visualization for {model_id} on {hardware_id}")
        return self.visualizer.create_metric_comparison_chart(
            validation_results,
            hardware_id=hardware_id,
            model_id=model_id,
            metrics=metrics,
            show_absolute_values=show_absolute_values,
            interactive=interactive,
            output_path=output_path,
            title=title
        )
    
    def visualize_hardware_comparison_heatmap(
        self,
        validation_results: List[ValidationResult],
        metric_name: str = "all",
        model_ids: Optional[List[str]] = None,
        interactive: Optional[bool] = None,
        output_path: Optional[str] = None,
        title: Optional[str] = None
    ) -> Union[str, None]:
        """
        Create a heatmap comparing simulation accuracy across hardware types.
        
        Args:
            validation_results: List of validation results
            metric_name: Name of the metric to visualize, or "all" for average
            model_ids: List of model IDs to include (if None, include all)
            interactive: Whether to create an interactive plot
            output_path: Path to save the chart
            title: Custom title for the chart
            
        Returns:
            Path to the saved chart if output_path is provided, otherwise HTML content
        """
        if not self.visualizer:
            logger.warning("Visualizer not available, cannot create hardware comparison heatmap")
            return None
            
        logger.info(f"Creating hardware comparison heatmap for metric '{metric_name}'")
        return self.visualizer.create_hardware_comparison_heatmap(
            validation_results,
            metric_name=metric_name,
            model_ids=model_ids,
            interactive=interactive,
            output_path=output_path,
            title=title
        )
    
    def visualize_drift_detection(
        self,
        drift_results: Dict[str, Any],
        interactive: Optional[bool] = None,
        output_path: Optional[str] = None,
        title: Optional[str] = None
    ) -> Union[str, None]:
        """
        Create a visualization of drift detection results.
        
        Args:
            drift_results: Results from the drift detector
            interactive: Whether to create an interactive plot
            output_path: Path to save the visualization
            title: Custom title for the visualization
            
        Returns:
            Path to the saved visualization if output_path is provided, otherwise HTML content
        """
        if not self.visualizer:
            logger.warning("Visualizer not available, cannot create drift detection visualization")
            return None
            
        logger.info("Creating drift detection visualization")
        return self.visualizer.create_drift_detection_visualization(
            drift_results,
            interactive=interactive,
            output_path=output_path,
            title=title
        )
    
    def visualize_calibration_improvement(
        self,
        before_calibration: List[ValidationResult],
        after_calibration: List[ValidationResult],
        interactive: Optional[bool] = None,
        output_path: Optional[str] = None,
        title: Optional[str] = None
    ) -> Union[str, None]:
        """
        Create a chart showing the improvement from calibration.
        
        Args:
            before_calibration: Validation results before calibration
            after_calibration: Validation results after calibration
            interactive: Whether to create an interactive plot
            output_path: Path to save the chart
            title: Custom title for the chart
            
        Returns:
            Path to the saved chart if output_path is provided, otherwise HTML content
        """
        if not self.visualizer:
            logger.warning("Visualizer not available, cannot create calibration improvement chart")
            return None
            
        logger.info("Creating calibration improvement visualization")
        return self.visualizer.create_calibration_improvement_chart(
            before_calibration,
            after_calibration,
            interactive=interactive,
            output_path=output_path,
            title=title
        )
    
    def create_comprehensive_dashboard(
        self,
        validation_results: List[ValidationResult],
        hardware_id: Optional[str] = None,
        model_id: Optional[str] = None,
        output_path: Optional[str] = None,
        title: Optional[str] = None,
        include_sections: Optional[List[str]] = None
    ) -> Union[str, None]:
        """
        Create a comprehensive dashboard with multiple visualizations.
        
        Args:
            validation_results: List of validation results
            hardware_id: Optional hardware ID filter
            model_id: Optional model ID filter
            output_path: Path to save the dashboard
            title: Custom title for the dashboard
            include_sections: List of sections to include (defaults to all)
            
        Returns:
            Path to the saved dashboard if output_path is provided, otherwise HTML content
        """
        if not self.visualizer:
            logger.warning("Visualizer not available, cannot create comprehensive dashboard")
            return None
            
        logger.info("Creating comprehensive validation dashboard")
        return self.visualizer.create_comprehensive_dashboard(
            validation_results,
            hardware_id=hardware_id,
            model_id=model_id,
            output_path=output_path,
            title=title,
            include_sections=include_sections
        )
    
    def create_validation_plan(
        self,
        hardware_id: str,
        model_id: str,
        protocol: str = "standard",
        existing_validation_results: Optional[List[ValidationResult]] = None
    ) -> Dict[str, Any]:
        """
        Create a validation plan for a hardware-model combination.
        
        Args:
            hardware_id: Hardware identifier
            model_id: Model identifier
            protocol: Validation protocol to use
            existing_validation_results: Optional list of existing validation results
            
        Returns:
            Dictionary with validation plan details
        """
        return self.methodology.create_validation_plan(
            hardware_id, model_id, protocol, existing_validation_results)
    
    def load_validation_results(
        self,
        hardware_id: Optional[str] = None,
        model_id: Optional[str] = None,
        batch_size: Optional[int] = None,
        precision: Optional[str] = None,
        limit: int = 100
    ) -> List[ValidationResult]:
        """
        Load validation results from database.
        
        Args:
            hardware_id: Optional filter for hardware ID
            model_id: Optional filter for model ID
            batch_size: Optional filter for batch size
            precision: Optional filter for precision
            limit: Maximum number of results to return
            
        Returns:
            List of validation results
        """
        if not self.db_api:
            logger.warning("Database not available, cannot load validation results")
            return []
        
        try:
            # Build query conditions
            conditions = []
            params = {}
            
            if hardware_id:
                conditions.append("hardware_result.hardware_id = :hardware_id")
                params["hardware_id"] = hardware_id
            
            if model_id:
                conditions.append("hardware_result.model_id = :model_id")
                params["model_id"] = model_id
            
            if batch_size:
                conditions.append("hardware_result.batch_size = :batch_size")
                params["batch_size"] = batch_size
            
            if precision:
                conditions.append("hardware_result.precision = :precision")
                params["precision"] = precision
            
            # Build query
            query = f"""
                SELECT 
                    validation_results.id as id,
                    simulation_results.id as simulation_result_id,
                    hardware_results.id as hardware_result_id,
                    validation_results.validation_timestamp,
                    validation_results.validation_version,
                    validation_results.metrics_comparison,
                    validation_results.additional_metrics
                FROM 
                    validation_results
                JOIN 
                    simulation_results ON validation_results.simulation_result_id = simulation_results.id
                JOIN 
                    hardware_results ON validation_results.hardware_result_id = hardware_results.id
                {" WHERE " + " AND ".join(conditions) if conditions else ""}
                ORDER BY 
                    validation_results.validation_timestamp DESC
                LIMIT :limit
            """
            params["limit"] = limit
            
            result = self.db_api.execute(query, params)
            rows = result.fetchall()
            
            validation_results = []
            for row in rows:
                # Load simulation result
                sim_result = self._load_simulation_result(row["simulation_result_id"])
                
                # Load hardware result
                hw_result = self._load_hardware_result(row["hardware_result_id"])
                
                if sim_result and hw_result:
                    # Create validation result
                    validation_result = ValidationResult(
                        simulation_result=sim_result,
                        hardware_result=hw_result,
                        metrics_comparison=row["metrics_comparison"],
                        validation_timestamp=row["validation_timestamp"],
                        validation_version=row["validation_version"],
                        additional_metrics=row["additional_metrics"]
                    )
                    
                    validation_results.append(validation_result)
            
            logger.info(f"Loaded {len(validation_results)} validation results from database")
            return validation_results
            
        except Exception as e:
            logger.error(f"Error loading validation results from database: {e}")
            return []
    
    def _load_configuration(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration from file or use defaults.
        
        Args:
            config_path: Path to configuration file (JSON format)
            
        Returns:
            Configuration dictionary
        """
        config = {}
        
        # Default configuration
        default_config = {
            "enable_calibration": True,
            "enable_drift_detection": True,
            "enable_reporting": True,
            "enable_parameter_discovery": True,
            "use_enhanced_validator": True,
            "use_advanced_calibrator": True,
            "database": {
                "enabled": False
            },
            "methodology": {},
            "comparison_pipeline": {},
            "statistical_validator": {},
            "calibrator": {},
            "drift_detector": {},
            "reporter": {},
            "parameter_discovery": {
                "metrics_to_analyze": [
                    "throughput_items_per_second", 
                    "average_latency_ms", 
                    "memory_peak_mb", 
                    "power_consumption_w"
                ],
                "min_samples_for_analysis": 5,
                "sensitivity_threshold": 0.05,
                "importance_calculation_method": "permutation",
                "cross_parameter_analysis": True,
                "batch_size_analysis": True,
                "precision_analysis": True,
                "model_size_analysis": True
            }
        }
        
        # Load configuration from file if provided
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration from {config_path}: {e}")
                logger.info("Using default configuration")
        
        # Apply default values for missing keys
        for key, value in default_config.items():
            if key not in config:
                config[key] = value
            elif isinstance(value, dict) and isinstance(config[key], dict):
                # Merge nested dictionaries
                for nested_key, nested_value in value.items():
                    if nested_key not in config[key]:
                        config[key][nested_key] = nested_value
        
        return config
    
    def _initialize_database(self) -> None:
        """Initialize database connection and create tables if needed."""
        try:
            db_config = self.config["database"]
            
            # Import database API
            from duckdb_api.core.db_api import BenchmarkDBAPI
            
            # Create database connection
            self.db_api = BenchmarkDBAPI(
                db_path=db_config.get("db_path", "benchmark_db.duckdb"),
                create_if_missing=True
            )
            
            # Create tables if they don't exist
            from duckdb_api.simulation_validation.core.schema import SimulationValidationSchema
            SimulationValidationSchema.create_tables(self.db_api.conn)
            
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            self.db_api = None
    
    def _store_validation_results(self, validation_results: List[ValidationResult]) -> None:
        """
        Store validation results in database.
        
        Args:
            validation_results: List of validation results to store
        """
        if not self.db_api:
            return
        
        try:
            from duckdb_api.simulation_validation.core.schema import SimulationValidationSchema as schema
            
            for val_result in validation_results:
                # Store simulation result
                sim_record = schema.simulation_result_to_db_dict(val_result.simulation_result)
                sim_id = sim_record["id"]
                
                self.db_api.execute(
                    """
                    INSERT INTO simulation_results 
                    VALUES (:id, :model_id, :hardware_id, :batch_size, :precision, 
                            :timestamp, :simulation_version, :additional_metadata,
                            :throughput_items_per_second, :average_latency_ms, 
                            :memory_peak_mb, :power_consumption_w, 
                            :initialization_time_ms, :warmup_time_ms, 
                            CURRENT_TIMESTAMP)
                    """,
                    sim_record
                )
                
                # Store hardware result
                hw_record = schema.hardware_result_to_db_dict(val_result.hardware_result)
                hw_id = hw_record["id"]
                
                self.db_api.execute(
                    """
                    INSERT INTO hardware_results 
                    VALUES (:id, :model_id, :hardware_id, :batch_size, :precision, 
                            :timestamp, :hardware_details, :test_environment, :additional_metadata,
                            :throughput_items_per_second, :average_latency_ms, 
                            :memory_peak_mb, :power_consumption_w, 
                            :initialization_time_ms, :warmup_time_ms, 
                            CURRENT_TIMESTAMP)
                    """,
                    hw_record
                )
                
                # Store validation result
                val_record = schema.validation_result_to_db_dict(val_result, sim_id, hw_id)
                
                self.db_api.execute(
                    """
                    INSERT INTO validation_results 
                    VALUES (:id, :simulation_result_id, :hardware_result_id, 
                            :validation_timestamp, :validation_version, 
                            :metrics_comparison, :additional_metrics,
                            :overall_accuracy_score, :throughput_mape, 
                            :latency_mape, :memory_mape, :power_mape, 
                            CURRENT_TIMESTAMP)
                    """,
                    val_record
                )
            
            # Commit transaction
            self.db_api.commit()
            logger.info(f"Stored {len(validation_results)} validation results in database")
            
        except Exception as e:
            logger.error(f"Error storing validation results in database: {e}")
            self.db_api.rollback()
    
    def _store_calibration_results(
        self,
        validation_results: List[ValidationResult],
        previous_parameters: Dict[str, Any],
        updated_parameters: Dict[str, Any],
        improvement_metrics: Dict[str, Any]
    ) -> None:
        """
        Store calibration results in database.
        
        Args:
            validation_results: Validation results used for calibration
            previous_parameters: Parameters before calibration
            updated_parameters: Parameters after calibration
            improvement_metrics: Metrics quantifying the calibration improvement
        """
        if not self.db_api:
            return
        
        try:
            # Determine hardware and model types
            if validation_results:
                hardware_id = validation_results[0].hardware_result.hardware_id
                model_id = validation_results[0].hardware_result.model_id
            else:
                hardware_id = "unknown"
                model_id = "unknown"
            
            from duckdb_api.simulation_validation.core.schema import SimulationValidationSchema as schema
            
            # Prepare calibration record
            cal_record = schema.calibration_to_db_dict(
                hardware_type=hardware_id,
                model_type=model_id,
                previous_parameters=previous_parameters,
                updated_parameters=updated_parameters,
                improvement_metrics=improvement_metrics,
                calibration_version=updated_parameters.get("calibration_version", "v1.0")
            )
            
            # Store calibration record
            self.db_api.execute(
                """
                INSERT INTO calibration_history 
                VALUES (:id, :timestamp, :hardware_type, :model_type, 
                        :previous_parameters, :updated_parameters, 
                        :validation_results_before, :validation_results_after, 
                        :improvement_metrics, :calibration_version, 
                        CURRENT_TIMESTAMP)
                """,
                cal_record
            )
            
            # Commit transaction
            self.db_api.commit()
            logger.info(f"Stored calibration results for {hardware_id}/{model_id} in database")
            
        except Exception as e:
            logger.error(f"Error storing calibration results in database: {e}")
            self.db_api.rollback()
    
    def _store_drift_detection_results(self, drift_results: Dict[str, Any]) -> None:
        """
        Store drift detection results in database.
        
        Args:
            drift_results: Drift detection results
        """
        if not self.db_api:
            return
        
        try:
            from duckdb_api.simulation_validation.core.schema import SimulationValidationSchema as schema
            
            # Extract hardware and model types
            hardware_type = drift_results.get("hardware_type", "unknown")
            model_type = drift_results.get("model_type", "unknown")
            
            # Prepare drift record
            drift_record = schema.drift_detection_to_db_dict(
                hardware_type=hardware_type,
                model_type=model_type,
                drift_metrics=drift_results.get("drift_metrics", {}),
                is_significant=drift_results.get("is_significant", False),
                historical_window_start=drift_results.get("historical_window_start", ""),
                historical_window_end=drift_results.get("historical_window_end", ""),
                new_window_start=drift_results.get("new_window_start", ""),
                new_window_end=drift_results.get("new_window_end", ""),
                thresholds_used=drift_results.get("thresholds_used", {})
            )
            
            # Store drift record
            self.db_api.execute(
                """
                INSERT INTO drift_detection 
                VALUES (:id, :timestamp, :hardware_type, :model_type, 
                        :drift_metrics, :is_significant, 
                        :historical_window_start, :historical_window_end, 
                        :new_window_start, :new_window_end, 
                        :thresholds_used, CURRENT_TIMESTAMP)
                """,
                drift_record
            )
            
            # Commit transaction
            self.db_api.commit()
            logger.info(f"Stored drift detection results for {hardware_type}/{model_type} in database")
            
        except Exception as e:
            logger.error(f"Error storing drift detection results in database: {e}")
            self.db_api.rollback()
    
    def _load_simulation_result(self, simulation_id: str) -> Optional[SimulationResult]:
        """
        Load a simulation result from the database.
        
        Args:
            simulation_id: ID of the simulation result
            
        Returns:
            SimulationResult object or None if not found
        """
        try:
            result = self.db_api.execute(
                """
                SELECT * FROM simulation_results WHERE id = :id
                """,
                {"id": simulation_id}
            )
            
            row = result.fetchone()
            if not row:
                return None
            
            # Extract specific metrics
            metrics = {}
            for metric in ["throughput_items_per_second", "average_latency_ms", 
                          "memory_peak_mb", "power_consumption_w",
                          "initialization_time_ms", "warmup_time_ms"]:
                if row[metric] is not None:
                    metrics[metric] = row[metric]
            
            # Create simulation result
            sim_result = SimulationResult(
                model_id=row["model_id"],
                hardware_id=row["hardware_id"],
                metrics=metrics,
                batch_size=row["batch_size"],
                precision=row["precision"],
                timestamp=row["timestamp"],
                simulation_version=row["simulation_version"],
                additional_metadata=row["additional_metadata"]
            )
            
            return sim_result
            
        except Exception as e:
            logger.error(f"Error loading simulation result {simulation_id} from database: {e}")
            return None
    
    def _load_hardware_result(self, hardware_id: str) -> Optional[HardwareResult]:
        """
        Load a hardware result from the database.
        
        Args:
            hardware_id: ID of the hardware result
            
        Returns:
            HardwareResult object or None if not found
        """
        try:
            result = self.db_api.execute(
                """
                SELECT * FROM hardware_results WHERE id = :id
                """,
                {"id": hardware_id}
            )
            
            row = result.fetchone()
            if not row:
                return None
            
            # Extract specific metrics
            metrics = {}
            for metric in ["throughput_items_per_second", "average_latency_ms", 
                          "memory_peak_mb", "power_consumption_w",
                          "initialization_time_ms", "warmup_time_ms"]:
                if row[metric] is not None:
                    metrics[metric] = row[metric]
            
            # Create hardware result
            hw_result = HardwareResult(
                model_id=row["model_id"],
                hardware_id=row["hardware_id"],
                metrics=metrics,
                batch_size=row["batch_size"],
                precision=row["precision"],
                timestamp=row["timestamp"],
                hardware_details=row["hardware_details"],
                test_environment=row["test_environment"],
                additional_metadata=row["additional_metadata"]
            )
            
            return hw_result
            
        except Exception as e:
            logger.error(f"Error loading hardware result {hardware_id} from database: {e}")
            return None
    
    def _store_parameter_discovery_results(self, parameter_recommendations: Dict[str, Any]) -> None:
        """
        Store parameter discovery results in database.
        
        Args:
            parameter_recommendations: Results from parameter discovery
        """
        if not self.db_api:
            return
        
        try:
            # Check if parameter_discovery table exists, create it if not
            self.db_api.execute("""
                CREATE TABLE IF NOT EXISTS parameter_discovery (
                    id VARCHAR PRIMARY KEY,
                    timestamp VARCHAR,
                    parameters_by_metric JSON,
                    overall_priority_list JSON,
                    sensitivity_insights JSON,
                    optimization_recommendations JSON,
                    key_findings JSON,
                    created_at TIMESTAMP
                )
            """)
            
            # Generate a unique ID for this discovery session
            discovery_id = str(uuid.uuid4())
            timestamp = datetime.datetime.now().isoformat()
            
            # Extract relevant sections from the parameter recommendations
            parameters_by_metric = parameter_recommendations.get("parameters_by_metric", {})
            overall_priority_list = parameter_recommendations.get("overall_priority_list", [])
            sensitivity_insights = parameter_recommendations.get("sensitivity_insights", {})
            optimization_recommendations = parameter_recommendations.get("optimization_recommendations", {})
            
            # Extract key findings if available
            key_findings = []
            if "insights" in parameter_recommendations and "key_findings" in parameter_recommendations["insights"]:
                key_findings = parameter_recommendations["insights"]["key_findings"]
            
            # Convert to JSON strings
            parameters_by_metric_json = json.dumps(parameters_by_metric)
            overall_priority_list_json = json.dumps(overall_priority_list)
            sensitivity_insights_json = json.dumps(sensitivity_insights)
            optimization_recommendations_json = json.dumps(optimization_recommendations)
            key_findings_json = json.dumps(key_findings)
            
            # Store in database
            self.db_api.execute("""
                INSERT INTO parameter_discovery
                VALUES (:id, :timestamp, :parameters_by_metric, :overall_priority_list,
                        :sensitivity_insights, :optimization_recommendations, :key_findings,
                        CURRENT_TIMESTAMP)
            """, {
                "id": discovery_id,
                "timestamp": timestamp,
                "parameters_by_metric": parameters_by_metric_json,
                "overall_priority_list": overall_priority_list_json,
                "sensitivity_insights": sensitivity_insights_json,
                "optimization_recommendations": optimization_recommendations_json,
                "key_findings": key_findings_json
            })
            
            # Commit transaction
            self.db_api.commit()
            logger.info(f"Stored parameter discovery results in database with ID: {discovery_id}")
            
        except Exception as e:
            logger.error(f"Error storing parameter discovery results in database: {e}")
            self.db_api.rollback()


def get_framework_instance(config_path: Optional[str] = None) -> SimulationValidationFramework:
    """
    Get an instance of the SimulationValidationFramework.
    
    Args:
        config_path: Path to configuration file (JSON format)
        
    Returns:
        SimulationValidationFramework instance
    """
    # For testing purposes, disable database integration
    test_config = {"database": {"enabled": False}}
    
    if config_path:
        # Load configuration from file
        try:
            import json
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                # Ensure database is disabled for testing
                if "database" in loaded_config:
                    loaded_config["database"]["enabled"] = False
                test_config.update(loaded_config)
        except Exception as e:
            logger.warning(f"Error loading config from {config_path}: {e}")
    
    return SimulationValidationFramework(test_config)