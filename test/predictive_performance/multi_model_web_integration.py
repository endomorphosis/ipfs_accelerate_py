#!/usr/bin/env python3
"""
Multi-Model Web Integration for Predictive Performance System.

This module integrates the Multi-Model Execution Predictor, WebNN/WebGPU Resource Pool,
and Empirical Validation systems into a unified framework for browser-based model execution
with prediction-guided optimization and continuous refinement.

Key features:
1. Comprehensive integration between prediction, execution, and validation components
2. Browser-specific execution strategies with adaptive optimization
3. Multi-model execution across heterogeneous web backends (WebGPU, WebNN, CPU)
4. Empirical validation for continuous refinement of performance models
5. Tensor sharing for memory-efficient model execution across browser environments
"""

import os
import sys
import time
import json
import logging
import traceback
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("predictive_performance.multi_model_web_integration")

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import multi-model execution predictor
try:
    from ipfs_accelerate_py.predictive_performance.multi_model_execution import MultiModelPredictor
except ImportError as e:
    logger.error(f"Error importing MultiModelPredictor: {e}")
    logger.error("Make sure multi_model_execution.py is available in the predictive_performance directory")
    MultiModelPredictor = None

# Import empirical validation
try:
    from ipfs_accelerate_py.predictive_performance.multi_model_empirical_validation import MultiModelEmpiricalValidator
    VALIDATOR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Error importing MultiModelEmpiricalValidator: {e}")
    logger.warning("Continuing without empirical validation capabilities")
    VALIDATOR_AVAILABLE = False

# Import resource pool integration
try:
    from ipfs_accelerate_py.predictive_performance.multi_model_resource_pool_integration import MultiModelResourcePoolIntegration
    RESOURCE_POOL_INTEGRATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Error importing MultiModelResourcePoolIntegration: {e}")
    logger.warning("Continuing without Resource Pool integration (will use simulation mode)")
    RESOURCE_POOL_INTEGRATION_AVAILABLE = False

# Import web resource pool adapter
try:
    from ipfs_accelerate_py.predictive_performance.web_resource_pool_adapter import WebResourcePoolAdapter
    WEB_ADAPTER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Error importing WebResourcePoolAdapter: {e}")
    logger.warning("Continuing without Web Resource Pool Adapter (will use default adapter)")
    WEB_ADAPTER_AVAILABLE = False


class MultiModelWebIntegration:
    """
    Integration framework for Browser-based Multi-Model Execution with Prediction and Validation.
    
    This class unifies the Multi-Model Execution Predictor, WebNN/WebGPU Resource Pool Adapter,
    and Empirical Validation systems into a comprehensive framework for executing multiple models
    in browser environments with prediction-guided optimization and continuous refinement.
    """
    
    def __init__(
        self,
        predictor: Optional[Any] = None,
        validator: Optional[Any] = None,
        resource_pool_integration: Optional[Any] = None,
        web_adapter: Optional[Any] = None,
        max_connections: int = 4,
        browser_preferences: Optional[Dict[str, str]] = None,
        enable_validation: bool = True,
        enable_tensor_sharing: bool = True,
        enable_strategy_optimization: bool = True,
        db_path: Optional[str] = None,
        validation_interval: int = 10,
        refinement_interval: int = 50,
        browser_capability_detection: bool = True,
        verbose: bool = False
    ):
        """
        Initialize the Multi-Model Web Integration framework.
        
        Args:
            predictor: Existing MultiModelPredictor instance (will create new if None)
            validator: Existing MultiModelEmpiricalValidator instance (will create new if None)
            resource_pool_integration: Existing MultiModelResourcePoolIntegration instance (will create new if None)
            web_adapter: Existing WebResourcePoolAdapter instance (will create new if None)
            max_connections: Maximum browser connections for resource pool
            browser_preferences: Browser preferences by model type
            enable_validation: Whether to enable empirical validation
            enable_tensor_sharing: Whether to enable tensor sharing between models
            enable_strategy_optimization: Whether to optimize execution strategies for browsers
            db_path: Path to database for storing results
            validation_interval: Interval for empirical validation in executions
            refinement_interval: Interval for model refinement in validations
            browser_capability_detection: Whether to detect browser capabilities
            verbose: Whether to enable verbose logging
        """
        self.max_connections = max_connections
        self.browser_preferences = browser_preferences or {}
        self.enable_validation = enable_validation
        self.enable_tensor_sharing = enable_tensor_sharing
        self.enable_strategy_optimization = enable_strategy_optimization
        self.db_path = db_path
        self.validation_interval = validation_interval
        self.refinement_interval = refinement_interval
        self.browser_capability_detection = browser_capability_detection
        
        # Set logging level
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        # Initialize predictor (create new if not provided)
        if predictor is not None:
            self.predictor = predictor
        elif MultiModelPredictor is not None:
            self.predictor = MultiModelPredictor(
                resource_pool_integration=True,
                verbose=verbose
            )
        else:
            self.predictor = None
            logger.error("Unable to initialize MultiModelPredictor")
        
        # Initialize validator (create new if not provided)
        if validator is not None:
            self.validator = validator
        elif VALIDATOR_AVAILABLE and enable_validation:
            self.validator = MultiModelEmpiricalValidator(
                db_path=db_path,
                validation_history_size=100,
                error_threshold=0.15,
                refinement_interval=refinement_interval,
                enable_trend_analysis=True,
                enable_visualization=True,
                verbose=verbose
            )
        else:
            self.validator = None
            if enable_validation:
                logger.warning("MultiModelEmpiricalValidator not available, validation will be disabled")
        
        # Initialize web adapter (create new if not provided)
        if web_adapter is not None:
            self.web_adapter = web_adapter
        elif WEB_ADAPTER_AVAILABLE:
            self.web_adapter = WebResourcePoolAdapter(
                max_connections=max_connections,
                browser_preferences=browser_preferences,
                enable_tensor_sharing=enable_tensor_sharing,
                enable_strategy_optimization=enable_strategy_optimization,
                browser_capability_detection=browser_capability_detection,
                db_path=db_path,
                verbose=verbose
            )
        else:
            self.web_adapter = None
            logger.warning("WebResourcePoolAdapter not available, some features will be limited")
        
        # Initialize resource pool integration (create new if not provided)
        if resource_pool_integration is not None:
            self.resource_pool_integration = resource_pool_integration
        elif RESOURCE_POOL_INTEGRATION_AVAILABLE:
            # Use web adapter as resource pool if available
            resource_pool = self.web_adapter if self.web_adapter else None
            
            self.resource_pool_integration = MultiModelResourcePoolIntegration(
                predictor=self.predictor,
                resource_pool=resource_pool,
                validator=self.validator,
                max_connections=max_connections,
                browser_preferences=browser_preferences,
                enable_empirical_validation=enable_validation,
                validation_interval=validation_interval,
                prediction_refinement=True,
                db_path=db_path,
                error_threshold=0.15,
                enable_adaptive_optimization=enable_strategy_optimization,
                enable_trend_analysis=True,
                verbose=verbose
            )
        else:
            self.resource_pool_integration = None
            logger.error("MultiModelResourcePoolIntegration not available")
        
        # Statistics and metrics
        self.execution_stats = {
            "total_executions": 0,
            "browser_executions": {},
            "strategy_executions": {},
            "validation_metrics": {
                "validation_count": 0,
                "refinement_count": 0,
                "average_errors": {}
            },
            "browser_capabilities": {}
        }
        
        # Initialization status
        self.initialized = False
        logger.info(f"MultiModelWebIntegration created "
                   f"(predictor={'available' if self.predictor else 'unavailable'}, "
                   f"validator={'available' if self.validator else 'unavailable'}, "
                   f"web_adapter={'available' if self.web_adapter else 'unavailable'}, "
                   f"resource_pool_integration={'available' if self.resource_pool_integration else 'unavailable'})")
    
    def initialize(self) -> bool:
        """
        Initialize the integration framework with all components.
        
        Returns:
            bool: Success status
        """
        if self.initialized:
            logger.warning("MultiModelWebIntegration already initialized")
            return True
        
        success = True
        
        # Initialize web adapter if available
        if self.web_adapter:
            logger.info("Initializing web adapter")
            adapter_success = self.web_adapter.initialize()
            if not adapter_success:
                logger.error("Failed to initialize web adapter")
                success = False
            else:
                logger.info("Web adapter initialized successfully")
                
                # Get browser capabilities
                if self.browser_capability_detection:
                    capabilities = self.web_adapter.get_browser_capabilities()
                    self.execution_stats["browser_capabilities"] = capabilities
                    logger.info(f"Detected {len(capabilities)} browsers with capabilities")
        else:
            logger.warning("No web adapter available, some features will be limited")
        
        # Initialize resource pool integration
        if self.resource_pool_integration:
            logger.info("Initializing resource pool integration")
            integration_success = self.resource_pool_integration.initialize()
            if not integration_success:
                logger.error("Failed to initialize resource pool integration")
                success = False
            else:
                logger.info("Resource pool integration initialized successfully")
        else:
            logger.error("No resource pool integration available, execution will fail")
            success = False
        
        self.initialized = success
        logger.info(f"MultiModelWebIntegration initialization {'successful' if success else 'failed'}")
        return success
    
    def execute_models(
        self,
        model_configs: List[Dict[str, Any]],
        hardware_platform: Optional[str] = "webgpu",
        execution_strategy: Optional[str] = None,
        optimization_goal: str = "latency",
        browser: Optional[str] = None,
        validate_predictions: bool = True,
        return_detailed_metrics: bool = False
    ) -> Dict[str, Any]:
        """
        Execute multiple models with browser-based hardware acceleration.
        
        Args:
            model_configs: List of model configurations to execute
            hardware_platform: Hardware platform for execution (webgpu, webnn, cpu)
            execution_strategy: Strategy for execution (None for automatic recommendation)
            optimization_goal: Metric to optimize ("latency", "throughput", or "memory")
            browser: Browser to use for execution (None for automatic selection)
            validate_predictions: Whether to validate predictions against actual measurements
            return_detailed_metrics: Whether to return detailed performance metrics
            
        Returns:
            Dictionary with execution results and measurements
        """
        if not self.initialized:
            logger.error("MultiModelWebIntegration not initialized")
            return {"success": False, "error": "Not initialized"}
        
        # Ensure resource pool integration is available
        if not self.resource_pool_integration:
            logger.error("Resource pool integration not available")
            return {"success": False, "error": "Resource pool integration not available"}
        
        # Start timing
        start_time = time.time()
        
        # If browser is specified, use browser-specific strategy
        if browser and self.web_adapter:
            logger.info(f"Using specified browser: {browser}")
            
            # Get optimal strategy for browser if not specified
            if execution_strategy is None and self.enable_strategy_optimization:
                execution_strategy = self.web_adapter.get_optimal_strategy(
                    model_configs=model_configs,
                    browser=browser,
                    optimization_goal=optimization_goal
                )
                logger.info(f"Selected optimal strategy for {browser}: {execution_strategy}")
            
            # Execute models with web adapter
            result = self.web_adapter.execute_models(
                model_configs=model_configs,
                execution_strategy=execution_strategy or "auto",
                optimization_goal=optimization_goal,
                browser=browser,
                return_metrics=return_detailed_metrics
            )
            
            # Update execution statistics
            self._update_execution_stats(result, browser, execution_strategy or result.get("execution_strategy", "auto"))
            
            # Add overall timing information
            result["total_time_ms"] = (time.time() - start_time) * 1000
            
            return result
        
        # Otherwise use the resource pool integration
        else:
            logger.info(f"Using resource pool integration with {hardware_platform} hardware")
            
            # Execute with strategy
            result = self.resource_pool_integration.execute_with_strategy(
                model_configs=model_configs,
                hardware_platform=hardware_platform,
                execution_strategy=execution_strategy,
                optimization_goal=optimization_goal,
                return_measurements=return_detailed_metrics,
                validate_predictions=validate_predictions and self.enable_validation
            )
            
            # Update execution statistics
            actual_strategy = result.get("execution_strategy", execution_strategy or "auto")
            self._update_execution_stats(result, "resource_pool", actual_strategy)
            
            return result
    
    def compare_strategies(
        self,
        model_configs: List[Dict[str, Any]],
        hardware_platform: Optional[str] = "webgpu",
        browser: Optional[str] = None,
        optimization_goal: str = "latency"
    ) -> Dict[str, Any]:
        """
        Compare different execution strategies for a set of models.
        
        Args:
            model_configs: List of model configurations to execute
            hardware_platform: Hardware platform for execution (ignored if browser is specified)
            browser: Browser to use for execution (None for hardware_platform-based execution)
            optimization_goal: Metric to optimize ("latency", "throughput", or "memory")
            
        Returns:
            Dictionary with comparison results
        """
        if not self.initialized:
            logger.error("MultiModelWebIntegration not initialized")
            return {"success": False, "error": "Not initialized"}
        
        # If browser is specified, use web adapter comparison
        if browser and self.web_adapter:
            logger.info(f"Comparing strategies for {len(model_configs)} models on {browser}")
            
            # Compare strategies with web adapter
            comparison = self.web_adapter.compare_strategies(
                model_configs=model_configs,
                browser=browser,
                optimization_goal=optimization_goal
            )
            
            return comparison
        
        # Otherwise use the resource pool integration
        elif self.resource_pool_integration:
            logger.info(f"Comparing strategies for {len(model_configs)} models on {hardware_platform}")
            
            # Compare strategies with resource pool integration
            comparison = self.resource_pool_integration.compare_strategies(
                model_configs=model_configs,
                hardware_platform=hardware_platform,
                optimization_goal=optimization_goal
            )
            
            return comparison
        
        else:
            logger.error("Neither web adapter nor resource pool integration available")
            return {"success": False, "error": "No execution backend available"}
    
    def get_optimal_browser(self, model_type: str) -> Optional[str]:
        """
        Get the optimal browser for a specific model type.
        
        Args:
            model_type: Type of model (text_embedding, vision, audio, etc.)
            
        Returns:
            Browser name or None if web adapter is not available
        """
        if not self.web_adapter:
            logger.warning("Web adapter not available, cannot determine optimal browser")
            return None
        
        browser = self.web_adapter.get_optimal_browser(model_type)
        return browser
    
    def get_optimal_strategy(
        self,
        model_configs: List[Dict[str, Any]],
        browser: Optional[str] = None,
        hardware_platform: str = "webgpu",
        optimization_goal: str = "latency"
    ) -> str:
        """
        Get the optimal execution strategy for a set of models.
        
        Args:
            model_configs: List of model configurations
            browser: Browser to use (prioritized over hardware_platform)
            hardware_platform: Hardware platform if browser not specified
            optimization_goal: Metric to optimize
            
        Returns:
            Optimal execution strategy
        """
        # If browser is specified, use web adapter strategy selection
        if browser and self.web_adapter:
            return self.web_adapter.get_optimal_strategy(
                model_configs=model_configs,
                browser=browser,
                optimization_goal=optimization_goal
            )
        
        # Otherwise use predictor directly
        elif self.predictor:
            recommendation = self.predictor.recommend_execution_strategy(
                model_configs=model_configs,
                hardware_platform=hardware_platform,
                optimization_goal=optimization_goal
            )
            return recommendation["recommended_strategy"]
        
        # Default strategy based on model count
        else:
            count = len(model_configs)
            if count <= 3:
                return "parallel"
            elif count >= 8:
                return "sequential"
            else:
                return "batched"
    
    def get_browser_capabilities(self) -> Dict[str, Dict[str, Any]]:
        """
        Get detected browser capabilities.
        
        Returns:
            Dictionary with browser capabilities
        """
        if not self.web_adapter:
            logger.warning("Web adapter not available, cannot get browser capabilities")
            return {}
        
        return self.web_adapter.get_browser_capabilities()
    
    def get_validation_metrics(self, include_history: bool = False) -> Dict[str, Any]:
        """
        Get validation metrics and error statistics.
        
        Args:
            include_history: Whether to include full validation history
            
        Returns:
            Dictionary with validation metrics and error statistics
        """
        if not self.validator and not self.resource_pool_integration:
            logger.warning("Neither validator nor resource pool integration available")
            return self.execution_stats["validation_metrics"]
        
        # Use resource pool integration's metrics if available
        if self.resource_pool_integration:
            metrics = self.resource_pool_integration.get_validation_metrics(include_history=include_history)
            
            # Update execution stats with validation metrics
            if "validation_count" in metrics:
                self.execution_stats["validation_metrics"]["validation_count"] = metrics["validation_count"]
            
            if "error_rates" in metrics:
                self.execution_stats["validation_metrics"]["average_errors"] = metrics["error_rates"]
            
            return metrics
        
        # Use validator directly if resource pool integration not available
        elif self.validator:
            return self.validator.get_validation_metrics(include_history=include_history)
        
        # Return default metrics if neither is available
        else:
            return self.execution_stats["validation_metrics"]
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """
        Get execution statistics.
        
        Returns:
            Dictionary with execution statistics
        """
        # Add validation metrics to execution stats
        if self.resource_pool_integration:
            metrics = self.resource_pool_integration.get_validation_metrics(include_history=False)
            
            # Update execution stats with validation metrics
            if "validation_count" in metrics:
                self.execution_stats["validation_metrics"]["validation_count"] = metrics["validation_count"]
            
            if "error_rates" in metrics:
                self.execution_stats["validation_metrics"]["average_errors"] = metrics["error_rates"]
        
        # Add web adapter stats if available
        if self.web_adapter:
            web_stats = self.web_adapter.get_execution_statistics()
            self.execution_stats["web_adapter_stats"] = web_stats
        
        return self.execution_stats
    
    def _update_execution_stats(self, result: Dict[str, Any], backend: str, strategy: str):
        """
        Update execution statistics based on execution result.
        
        Args:
            result: Execution result dictionary
            backend: Backend used for execution (browser name or "resource_pool")
            strategy: Execution strategy used
        """
        # Update total executions
        self.execution_stats["total_executions"] += 1
        
        # Update backend executions
        self.execution_stats["browser_executions"][backend] = self.execution_stats["browser_executions"].get(backend, 0) + 1
        
        # Update strategy executions
        self.execution_stats["strategy_executions"][strategy] = self.execution_stats["strategy_executions"].get(strategy, 0) + 1
        
        # Update validation metrics if available
        if self.resource_pool_integration:
            metrics = self.resource_pool_integration.get_validation_metrics(include_history=False)
            
            # Update execution stats with validation metrics
            if "validation_count" in metrics:
                self.execution_stats["validation_metrics"]["validation_count"] = metrics["validation_count"]
            
            if "error_rates" in metrics:
                self.execution_stats["validation_metrics"]["average_errors"] = metrics["error_rates"]
    
    def visualize_performance(self, metric_type: str = "error_rates") -> Dict[str, Any]:
        """
        Visualize performance metrics.
        
        Args:
            metric_type: Type of metrics to visualize (error_rates, browser_comparison, strategy_comparison)
            
        Returns:
            Dictionary with visualization data
        """
        if self.validator and hasattr(self.validator, "visualize_validation_metrics"):
            return self.validator.visualize_validation_metrics(metric_type=metric_type)
        else:
            logger.warning("Validator not available or doesn't support visualization")
            return {"success": False, "reason": "Visualization not available"}
    
    def close(self) -> bool:
        """
        Close the integration and release resources.
        
        Returns:
            Success status
        """
        success = True
        
        # Close resource pool integration
        if self.resource_pool_integration:
            try:
                logger.info("Closing resource pool integration")
                integration_success = self.resource_pool_integration.close()
                if not integration_success:
                    logger.error("Error closing resource pool integration")
                    success = False
            except Exception as e:
                logger.error(f"Exception closing resource pool integration: {e}")
                traceback.print_exc()
                success = False
        
        # Close web adapter
        if self.web_adapter:
            try:
                logger.info("Closing web adapter")
                adapter_success = self.web_adapter.close()
                if not adapter_success:
                    logger.error("Error closing web adapter")
                    success = False
            except Exception as e:
                logger.error(f"Exception closing web adapter: {e}")
                traceback.print_exc()
                success = False
        
        # Close validator
        if self.validator and hasattr(self.validator, "close"):
            try:
                logger.info("Closing validator")
                validator_success = self.validator.close()
                if not validator_success:
                    logger.error("Error closing validator")
                    success = False
            except Exception as e:
                logger.error(f"Exception closing validator: {e}")
                traceback.print_exc()
                success = False
        
        logger.info(f"MultiModelWebIntegration closed (success={'yes' if success else 'no'})")
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
    
    logger.info("Starting MultiModelWebIntegration example")
    
    # Create the integration
    integration = MultiModelWebIntegration(
        max_connections=2,
        enable_validation=True,
        enable_tensor_sharing=True,
        enable_strategy_optimization=True,
        browser_capability_detection=True,
        verbose=True
    )
    
    # Initialize
    success = integration.initialize()
    if not success:
        logger.error("Failed to initialize integration")
        sys.exit(1)
    
    try:
        # Get browser capabilities
        capabilities = integration.get_browser_capabilities()
        logger.info(f"Detected {len(capabilities)} browsers with capabilities")
        
        for browser, caps in capabilities.items():
            logger.info(f"{browser}: WebGPU={caps.get('webgpu', False)}, WebNN={caps.get('webnn', False)}")
        
        # Define model configurations for testing
        model_configs = [
            {"model_name": "bert-base-uncased", "model_type": "text_embedding", "batch_size": 4},
            {"model_name": "vit-base-patch16-224", "model_type": "vision", "batch_size": 1}
        ]
        
        # Get optimal browser for text embedding
        optimal_browser = integration.get_optimal_browser("text_embedding")
        logger.info(f"Optimal browser for text embedding: {optimal_browser}")
        
        # Get optimal strategy
        optimal_strategy = integration.get_optimal_strategy(
            model_configs=model_configs,
            browser=optimal_browser,
            optimization_goal="throughput"
        )
        logger.info(f"Optimal strategy: {optimal_strategy}")
        
        # Execute models with automatic strategy selection
        logger.info("Executing models with automatic strategy selection")
        result = integration.execute_models(
            model_configs=model_configs,
            optimization_goal="throughput",
            browser=optimal_browser,
            validate_predictions=True
        )
        
        logger.info(f"Execution complete with strategy: {result.get('execution_strategy', 'unknown')}")
        logger.info(f"Throughput: {result.get('throughput', 0):.2f} items/sec")
        logger.info(f"Latency: {result.get('latency', 0):.2f} ms")
        logger.info(f"Memory usage: {result.get('memory_usage', 0):.2f} MB")
        
        # Compare execution strategies
        logger.info("Comparing execution strategies")
        comparison = integration.compare_strategies(
            model_configs=model_configs,
            browser=optimal_browser,
            optimization_goal="throughput"
        )
        
        logger.info(f"Best strategy: {comparison.get('best_strategy', 'unknown')}")
        logger.info(f"Recommended strategy: {comparison.get('recommended_strategy', 'unknown')}")
        logger.info(f"Recommendation accuracy: {comparison.get('recommendation_accuracy', False)}")
        
        # Get validation metrics
        metrics = integration.get_validation_metrics()
        logger.info(f"Validation count: {metrics.get('validation_count', 0)}")
        
        # Get execution statistics
        stats = integration.get_execution_statistics()
        logger.info(f"Total executions: {stats['total_executions']}")
        logger.info(f"Browser executions: {stats['browser_executions']}")
        logger.info(f"Strategy executions: {stats['strategy_executions']}")
        
    finally:
        # Close the integration
        integration.close()
        logger.info("MultiModelWebIntegration example completed")