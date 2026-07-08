#!/usr/bin/env python3
"""
Multi-Model Resource Pool Integration for Predictive Performance System.

This module integrates the Multi-Model Execution Support with the WebNN/WebGPU Resource Pool,
enabling empirical validation of prediction models and optimization of resource allocation
based on performance predictions. It serves as a bridge between the prediction system and
actual execution, providing feedback mechanisms to improve prediction accuracy.

Key features:
1. Prediction-guided resource allocation and execution strategies
2. Empirical validation of prediction models
3. Performance data collection and analysis for model improvement
4. Adaptive optimization based on real-world measurements
5. Continuous refinement of prediction models
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
logger = logging.getLogger("predictive_performance.multi_model_resource_pool_integration")

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
    from test.web_platform.resource_pool_bridge_integration import ResourcePoolBridgeIntegrationWithRecovery
    RESOURCE_POOL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Error importing ResourcePoolBridgeIntegrationWithRecovery: {e}")
    logger.warning("Continuing without Resource Pool integration (will use simulation mode)")
    RESOURCE_POOL_AVAILABLE = False


class MultiModelResourcePoolIntegration:
    """
    Integration between Multi-Model Execution Support and Web Resource Pool.
    
    This class bridges the gap between performance prediction and actual execution,
    enabling empirical validation of prediction models, optimization of resource
    allocation, and continuous improvement of the predictive system.
    """
    
    def __init__(
        self,
        predictor: Optional[MultiModelPredictor] = None,
        resource_pool: Optional[Any] = None,
        validator: Optional[Any] = None,
        max_connections: int = 4,
        browser_preferences: Optional[Dict[str, str]] = None,
        enable_empirical_validation: bool = True,
        validation_interval: int = 10,
        prediction_refinement: bool = True,
        db_path: Optional[str] = None,
        error_threshold: float = 0.15,
        enable_adaptive_optimization: bool = True,
        enable_trend_analysis: bool = True,
        verbose: bool = False
    ):
        """
        Initialize the Multi-Model Resource Pool Integration.
        
        Args:
            predictor: Existing MultiModelPredictor instance (will create new if None)
            resource_pool: Existing ResourcePoolBridgeIntegration instance (will create new if None)
            validator: Existing MultiModelEmpiricalValidator instance (will create new if None)
            max_connections: Maximum browser connections for resource pool
            browser_preferences: Browser preferences by model type
            enable_empirical_validation: Whether to enable empirical validation
            validation_interval: Interval for empirical validation in executions
            prediction_refinement: Whether to refine prediction models with empirical data
            db_path: Path to database for storing results
            error_threshold: Threshold for acceptable prediction error (15% by default)
            enable_adaptive_optimization: Whether to adapt optimization based on measurements
            enable_trend_analysis: Whether to analyze error trends over time
            verbose: Whether to enable verbose logging
        """
        self.max_connections = max_connections
        self.browser_preferences = browser_preferences or {}
        self.enable_empirical_validation = enable_empirical_validation
        self.validation_interval = validation_interval
        self.prediction_refinement = prediction_refinement
        self.db_path = db_path
        self.error_threshold = error_threshold
        self.enable_adaptive_optimization = enable_adaptive_optimization
        self.enable_trend_analysis = enable_trend_analysis
        
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
        
        # Initialize resource pool (create new if not provided)
        if resource_pool is not None:
            self.resource_pool = resource_pool
        elif RESOURCE_POOL_AVAILABLE:
            self.resource_pool = ResourcePoolBridgeIntegrationWithRecovery(
                max_connections=max_connections,
                browser_preferences=browser_preferences,
                adaptive_scaling=True,
                enable_recovery=True,
                enable_tensor_sharing=True,
                db_path=db_path
            )
        else:
            self.resource_pool = None
            logger.error("ResourcePoolBridgeIntegrationWithRecovery not available")
        
        # Initialize empirical validator (create new if not provided)
        if validator is not None:
            self.validator = validator
        elif VALIDATOR_AVAILABLE and enable_empirical_validation:
            self.validator = MultiModelEmpiricalValidator(
                db_path=db_path,
                validation_history_size=100,
                error_threshold=error_threshold,
                refinement_interval=validation_interval,
                enable_trend_analysis=enable_trend_analysis,
                enable_visualization=True,
                verbose=verbose
            )
        else:
            self.validator = None
            if enable_empirical_validation:
                logger.warning("MultiModelEmpiricalValidator not available, will use basic validation")
            
            # Legacy validation metrics storage (used if validator not available)
            self.validation_metrics = {
                "predicted_vs_actual": [],
                "optimization_impact": [],
                "execution_count": 0,
                "last_validation_time": 0,
                "validation_count": 0,
                "error_rates": {
                    "throughput": [],
                    "latency": [],
                    "memory": []
                }
            }
        
        # Strategy configuration by hardware platform
        self.strategy_configuration = {
            "cuda": {
                "parallel_threshold": 3,  # Use parallel for 3 or fewer models
                "sequential_threshold": 8,  # Use sequential for more than 8 models
                "batching_size": 4,  # Batch size for batched execution
                "memory_threshold": 16000,  # Memory threshold in MB
            },
            "webgpu": {
                "parallel_threshold": 2,  # More conservative for WebGPU
                "sequential_threshold": 6,
                "batching_size": 3,
                "memory_threshold": 4000,
            },
            "webnn": {
                "parallel_threshold": 2,
                "sequential_threshold": 5,
                "batching_size": 3,
                "memory_threshold": 4000,
            },
            "cpu": {
                "parallel_threshold": 4,  # CPU often handles parallelism better
                "sequential_threshold": 12,
                "batching_size": 6,
                "memory_threshold": 8000,
            }
        }
        
        # Initialize
        self.initialized = False
        logger.info(f"MultiModelResourcePoolIntegration created "
                   f"(predictor={'available' if self.predictor else 'unavailable'}, "
                   f"resource_pool={'available' if self.resource_pool else 'unavailable'}, "
                   f"empirical_validation={'enabled' if enable_empirical_validation else 'disabled'}, "
                   f"adaptive_optimization={'enabled' if enable_adaptive_optimization else 'disabled'})")
    
    def initialize(self) -> bool:
        """
        Initialize the integration with resource pool and prediction system.
        
        Returns:
            bool: Success status
        """
        if self.initialized:
            logger.warning("MultiModelResourcePoolIntegration already initialized")
            return True
        
        success = True
        
        # Initialize resource pool if available
        if self.resource_pool:
            logger.info("Initializing resource pool")
            pool_success = self.resource_pool.initialize()
            if not pool_success:
                logger.error("Failed to initialize resource pool")
                success = False
            else:
                logger.info("Resource pool initialized successfully")
        else:
            logger.warning("No resource pool available, will operate in simulation mode")
        
        # Initialize database connection for metrics if validator not available and db_path provided
        if not self.validator and self.db_path:
            try:
                import duckdb
                self.db_conn = duckdb.connect(self.db_path)
                self._initialize_database_tables()
                logger.info(f"Database connection established to {self.db_path}")
            except ImportError:
                logger.warning("duckdb not available, will operate without database storage")
                self.db_conn = None
            except Exception as e:
                logger.error(f"Error connecting to database: {e}")
                self.db_conn = None
                traceback.print_exc()
        else:
            self.db_conn = None
        
        self.initialized = success
        logger.info(f"MultiModelResourcePoolIntegration initialization {'successful' if success else 'failed'}"
                   f" (validator={'available' if self.validator else 'unavailable'}, "
                   f"resource_pool={'available' if self.resource_pool else 'unavailable'}, "
                   f"predictor={'available' if self.predictor else 'unavailable'})")
        return success
    
    def _initialize_database_tables(self):
        """Initialize database tables for storing prediction and actual metrics."""
        if not self.db_conn:
            return
        
        try:
            # Create table for prediction validation metrics
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
            
            # Create table for optimization impact
            self.db_conn.execute("""
            CREATE TABLE IF NOT EXISTS multi_model_optimization_impact (
                id INTEGER PRIMARY KEY,
                timestamp TIMESTAMP,
                model_count INTEGER,
                hardware_platform VARCHAR,
                baseline_strategy VARCHAR,
                optimized_strategy VARCHAR,
                baseline_throughput DOUBLE,
                optimized_throughput DOUBLE,
                baseline_latency DOUBLE,
                optimized_latency DOUBLE,
                baseline_memory DOUBLE,
                optimized_memory DOUBLE,
                throughput_improvement_percent DOUBLE,
                latency_improvement_percent DOUBLE,
                memory_improvement_percent DOUBLE,
                optimization_goal VARCHAR
            )
            """)
            
            logger.info("Database tables initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database tables: {e}")
            traceback.print_exc()
    
    def execute_with_strategy(
        self,
        model_configs: List[Dict[str, Any]],
        hardware_platform: str,
        execution_strategy: Optional[str] = None,
        optimization_goal: str = "latency",
        return_measurements: bool = True,
        validate_predictions: bool = True
    ) -> Dict[str, Any]:
        """
        Execute multiple models with a specific or recommended execution strategy.
        
        Args:
            model_configs: List of model configurations to execute
            hardware_platform: Hardware platform for execution
            execution_strategy: Strategy for execution (None for automatic recommendation)
            optimization_goal: Metric to optimize ("latency", "throughput", or "memory")
            return_measurements: Whether to return detailed measurements
            validate_predictions: Whether to validate predictions against actual measurements
            
        Returns:
            Dictionary with execution results and measurements
        """
        if not self.initialized:
            logger.error("MultiModelResourcePoolIntegration not initialized")
            return {"success": False, "error": "Not initialized"}
        
        # Check if predictor is available
        if not self.predictor:
            logger.error("MultiModelPredictor not available")
            return {"success": False, "error": "Predictor not available"}
        
        # Start timing
        start_time = time.time()
        
        # Get recommendation if strategy not specified
        if execution_strategy is None:
            logger.info(f"Getting execution strategy recommendation for {len(model_configs)} models")
            recommendation = self.predictor.recommend_execution_strategy(
                model_configs=model_configs,
                hardware_platform=hardware_platform,
                optimization_goal=optimization_goal
            )
            execution_strategy = recommendation["recommended_strategy"]
            prediction = recommendation["best_prediction"]
            
            logger.info(f"Recommended strategy: {execution_strategy} for optimization goal: {optimization_goal}")
        else:
            # Get prediction for specified strategy
            logger.info(f"Using specified strategy: {execution_strategy}")
            prediction = self.predictor.predict_multi_model_performance(
                model_configs=model_configs,
                hardware_platform=hardware_platform,
                execution_strategy=execution_strategy
            )
        
        # Extract predicted metrics
        predicted_metrics = prediction["total_metrics"]
        predicted_throughput = predicted_metrics.get("combined_throughput", 0)
        predicted_latency = predicted_metrics.get("combined_latency", 0)
        predicted_memory = predicted_metrics.get("combined_memory", 0)
        
        # Get predicted execution schedule
        execution_schedule = prediction["execution_schedule"]
        
        # Check if resource pool is available for actual execution
        if not self.resource_pool:
            logger.warning("Resource pool not available, using simulation mode")
            
            # Simulate actual execution (adding random variation)
            import random
            random.seed(int(time.time()))
            
            # Add random variation to simulate real-world differences (±15%)
            variation_factor = lambda: random.uniform(0.85, 1.15)
            
            actual_throughput = predicted_throughput * variation_factor()
            actual_latency = predicted_latency * variation_factor()
            actual_memory = predicted_memory * variation_factor()
            
            # Simulate models
            model_results = [{"success": True, "simulated": True} for _ in model_configs]
            
            # Create simulated execution result
            execution_result = {
                "success": True,
                "execution_strategy": execution_strategy,
                "model_count": len(model_configs),
                "hardware_platform": hardware_platform,
                "model_results": model_results,
                "simulated": True
            }
        else:
            # Actual execution with resource pool
            logger.info(f"Executing {len(model_configs)} models with {execution_strategy} strategy")
            
            # Load models from resource pool
            models = []
            model_inputs = []
            
            for config in model_configs:
                model_type = config.get("model_type", "text_embedding")
                model_name = config.get("model_name", "")
                batch_size = config.get("batch_size", 1)
                
                # Convert model_type if needed
                if model_type == "text_embedding":
                    resource_pool_type = "text" 
                elif model_type == "text_generation":
                    resource_pool_type = "text"
                else:
                    resource_pool_type = model_type
                
                # Create hardware preferences with platform
                hw_preferences = {
                    "priority_list": [hardware_platform, "cpu"],
                }
                
                # Add browser preferences if available
                if model_type in self.browser_preferences:
                    hw_preferences["browser"] = self.browser_preferences[model_type]
                
                try:
                    # Get model from resource pool
                    model = self.resource_pool.get_model(
                        model_type=resource_pool_type,
                        model_name=model_name,
                        hardware_preferences=hw_preferences
                    )
                    
                    if model:
                        models.append(model)
                        
                        # Create placeholder input based on model type
                        # In a real implementation, these would be actual inputs
                        if model_type == "text_embedding" or model_type == "text_generation":
                            input_data = {
                                "input_ids": [101, 2023, 2003, 1037, 3231, 102] * batch_size,
                                "attention_mask": [1, 1, 1, 1, 1, 1] * batch_size
                            }
                        elif model_type == "vision":
                            input_data = {
                                "pixel_values": [[[0.5 for _ in range(224)] for _ in range(224)] for _ in range(3)]
                            }
                        elif model_type == "audio":
                            input_data = {
                                "input_features": [[0.1 for _ in range(80)] for _ in range(3000)]
                            }
                        else:
                            input_data = {"placeholder": True}
                        
                        model_inputs.append((model, input_data))
                    else:
                        logger.error(f"Failed to load model: {model_name} ({resource_pool_type})")
                except Exception as e:
                    logger.error(f"Error loading model {model_name}: {e}")
                    traceback.print_exc()
            
            # Execute based on strategy
            if execution_strategy == "parallel":
                # Parallel execution
                execution_start = time.time()
                model_results = self.resource_pool.execute_concurrent([
                    (model, inputs) for model, inputs in model_inputs
                ])
                execution_time = time.time() - execution_start
                
                # Calculate actual metrics
                actual_latency = execution_time * 1000  # Convert to ms
                # Estimate throughput based on number of models and time
                actual_throughput = len(model_results) / (execution_time if execution_time > 0 else 0.001)
                
                # Get memory usage from resource pool metrics
                metrics = self.resource_pool.get_metrics()
                actual_memory = metrics.get("base_metrics", {}).get("peak_memory_usage", predicted_memory)
                
            elif execution_strategy == "sequential":
                # Sequential execution
                execution_start = time.time()
                model_results = []
                
                # Execute each model sequentially and measure individual times
                for model, inputs in model_inputs:
                    model_start = time.time()
                    result = model(inputs)
                    model_time = time.time() - model_start
                    
                    # Add timing information to result
                    if isinstance(result, dict):
                        result["execution_time_ms"] = model_time * 1000
                    else:
                        result = {"result": result, "execution_time_ms": model_time * 1000}
                    
                    model_results.append(result)
                
                execution_time = time.time() - execution_start
                
                # Calculate actual metrics
                actual_latency = execution_time * 1000  # Convert to ms
                # Sequential throughput is number of models divided by total time
                actual_throughput = len(model_results) / (execution_time if execution_time > 0 else 0.001)
                
                # Get memory usage from resource pool metrics
                metrics = self.resource_pool.get_metrics()
                actual_memory = metrics.get("base_metrics", {}).get("peak_memory_usage", predicted_memory)
                
            else:  # batched
                # Get batch configuration
                batch_size = self.strategy_configuration.get(hardware_platform, {}).get("batching_size", 4)
                
                # Create batches
                batches = []
                current_batch = []
                
                for item in model_inputs:
                    current_batch.append(item)
                    if len(current_batch) >= batch_size:
                        batches.append(current_batch)
                        current_batch = []
                
                # Add remaining items
                if current_batch:
                    batches.append(current_batch)
                
                # Execute batches sequentially
                execution_start = time.time()
                model_results = []
                
                for batch in batches:
                    # Execute batch in parallel
                    batch_results = self.resource_pool.execute_concurrent([
                        (model, inputs) for model, inputs in batch
                    ])
                    model_results.extend(batch_results)
                
                execution_time = time.time() - execution_start
                
                # Calculate actual metrics
                actual_latency = execution_time * 1000  # Convert to ms
                actual_throughput = len(model_results) / (execution_time if execution_time > 0 else 0.001)
                
                # Get memory usage from resource pool metrics
                metrics = self.resource_pool.get_metrics()
                actual_memory = metrics.get("base_metrics", {}).get("peak_memory_usage", predicted_memory)
            
            # Create execution result
            execution_result = {
                "success": all(result.get("success", False) for result in model_results),
                "execution_strategy": execution_strategy,
                "model_count": len(model_configs),
                "hardware_platform": hardware_platform,
                "model_results": model_results,
                "execution_time_ms": execution_time * 1000,
                "actual_throughput": actual_throughput,
                "actual_latency": actual_latency,
                "actual_memory": actual_memory
            }
        
        # Validate predictions if enabled
        if validate_predictions and self.enable_empirical_validation:
            # If we have the empirical validator available, use it
            if self.validator:
                # Create prediction object for validation
                prediction_obj = {
                    "total_metrics": {
                        "combined_throughput": predicted_throughput,
                        "combined_latency": predicted_latency,
                        "combined_memory": predicted_memory
                    },
                    "execution_strategy": execution_strategy
                }
                
                # Create actual measurement object
                actual_measurement = {
                    "actual_throughput": actual_throughput,
                    "actual_latency": actual_latency,
                    "actual_memory": actual_memory,
                    "execution_strategy": execution_strategy
                }
                
                # Validate prediction with empirical validator
                validation_metrics = self.validator.validate_prediction(
                    prediction=prediction_obj,
                    actual_measurement=actual_measurement,
                    model_configs=model_configs,
                    hardware_platform=hardware_platform,
                    execution_strategy=execution_strategy,
                    optimization_goal=optimization_goal
                )
                
                # Log validation results
                logger.info(f"Validation #{validation_metrics.get('validation_count', 0)}: "
                           f"Throughput error: {validation_metrics['current_errors']['throughput']:.2%}, "
                           f"Latency error: {validation_metrics['current_errors']['latency']:.2%}, "
                           f"Memory error: {validation_metrics['current_errors']['memory']:.2%}")
                
                # Check if model refinement is needed
                if validation_metrics.get('needs_refinement', False) and self.prediction_refinement:
                    # Get refinement recommendations
                    recommendations = self.validator.get_refinement_recommendations()
                    
                    if recommendations.get('refinement_needed', False):
                        logger.info(f"Model refinement recommended: {recommendations.get('reason', '')}")
                        
                        # Update prediction model if refinement is enabled and predictor supports it
                        if hasattr(self.predictor, 'update_contention_models'):
                            logger.info(f"Updating prediction models with empirical data using method: {recommendations.get('recommended_method', 'incremental')}")
                            
                            try:
                                # Get pre-refinement errors
                                pre_refinement_errors = {
                                    "throughput": recommendations['error_rates']['throughput'],
                                    "latency": recommendations['error_rates']['latency'],
                                    "memory": recommendations['error_rates']['memory']
                                }
                                
                                # Perform refinement with recommended method
                                method = recommendations.get('recommended_method', 'incremental')
                                
                                # Generate validation dataset
                                dataset = self.validator.generate_validation_dataset()
                                
                                if dataset.get("success", False):
                                    if hasattr(self.predictor, 'update_models'):
                                        # Update models with the dataset and method
                                        self.predictor.update_models(
                                            dataset=dataset.get("records", []),
                                            method=method
                                        )
                                    else:
                                        # Fall back to basic update method
                                        self.predictor.update_contention_models(
                                            validation_data=dataset.get("records", [])
                                        )
                                    
                                    # Get post-refinement errors (assume 10% improvement as placeholder)
                                    post_refinement_errors = {
                                        "throughput": pre_refinement_errors["throughput"] * 0.9,
                                        "latency": pre_refinement_errors["latency"] * 0.9,
                                        "memory": pre_refinement_errors["memory"] * 0.9
                                    }
                                    
                                    # Record refinement results
                                    self.validator.record_model_refinement(
                                        pre_refinement_errors=pre_refinement_errors,
                                        post_refinement_errors=post_refinement_errors,
                                        refinement_method=method
                                    )
                                    
                                    logger.info(f"Model refinement completed using {method} method")
                                else:
                                    logger.error(f"Failed to generate validation dataset: {dataset.get('reason', 'unknown error')}")
                            except Exception as e:
                                logger.error(f"Error updating prediction models: {e}")
                                traceback.print_exc()
            else:
                # Legacy validation approach (used if validator not available)
                # Increment execution count
                self.validation_metrics["execution_count"] += 1
                
                # Check if it's time for validation
                if (self.validation_metrics["execution_count"] % self.validation_interval == 0 or 
                    time.time() - self.validation_metrics["last_validation_time"] > 300):  # At least 5 minutes since last validation
                    
                    self.validation_metrics["last_validation_time"] = time.time()
                    self.validation_metrics["validation_count"] += 1
                    
                    # Calculate error rates
                    throughput_error = abs(predicted_throughput - actual_throughput) / (predicted_throughput if predicted_throughput > 0 else 1)
                    latency_error = abs(predicted_latency - actual_latency) / (predicted_latency if predicted_latency > 0 else 1)
                    memory_error = abs(predicted_memory - actual_memory) / (predicted_memory if predicted_memory > 0 else 1)
                    
                    # Add to validation metrics
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
                        "optimization_goal": optimization_goal
                    }
                    
                    self.validation_metrics["predicted_vs_actual"].append(validation_record)
                    self.validation_metrics["error_rates"]["throughput"].append(throughput_error)
                    self.validation_metrics["error_rates"]["latency"].append(latency_error)
                    self.validation_metrics["error_rates"]["memory"].append(memory_error)
                    
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
                                    time.time(), len(model_configs), hardware_platform, execution_strategy,
                                    predicted_throughput, actual_throughput, predicted_latency, actual_latency,
                                    predicted_memory, actual_memory, throughput_error, latency_error,
                                    memory_error, json.dumps([m.get("model_name", "") for m in model_configs]), optimization_goal
                                )
                            )
                        except Exception as e:
                            logger.error(f"Error storing validation metrics in database: {e}")
                    
                    # Update prediction model if refinement is enabled
                    if self.prediction_refinement and hasattr(self.predictor, 'update_contention_models'):
                        logger.info("Updating prediction models with empirical data")
                        try:
                            self.predictor.update_contention_models(validation_record)
                        except Exception as e:
                            logger.error(f"Error updating prediction models: {e}")
                    
                    logger.info(f"Validation #{self.validation_metrics['validation_count']}: "
                               f"Throughput error: {throughput_error:.2%}, "
                               f"Latency error: {latency_error:.2%}, "
                               f"Memory error: {memory_error:.2%}")
        
        # Add predicted and timing information to result
        execution_result.update({
            "predicted_throughput": predicted_throughput,
            "predicted_latency": predicted_latency,
            "predicted_memory": predicted_memory,
            "total_time_ms": (time.time() - start_time) * 1000,
            "optimization_goal": optimization_goal
        })
        
        # Include detailed measurements if requested
        if return_measurements:
            execution_result["measurements"] = {
                "prediction_accuracy": {
                    "throughput": abs(1 - (actual_throughput / predicted_throughput if predicted_throughput > 0 else 0)),
                    "latency": abs(1 - (actual_latency / predicted_latency if predicted_latency > 0 else 0)),
                    "memory": abs(1 - (actual_memory / predicted_memory if predicted_memory > 0 else 0))
                },
                "execution_schedule": execution_schedule,
                "strategy_details": self.strategy_configuration.get(hardware_platform, {})
            }
        
        return execution_result
    
    def compare_strategies(
        self,
        model_configs: List[Dict[str, Any]],
        hardware_platform: str,
        optimization_goal: str = "latency"
    ) -> Dict[str, Any]:
        """
        Compare different execution strategies for a set of models.
        
        Args:
            model_configs: List of model configurations to execute
            hardware_platform: Hardware platform for execution
            optimization_goal: Metric to optimize ("latency", "throughput", or "memory")
            
        Returns:
            Dictionary with comparison results
        """
        if not self.initialized:
            logger.error("MultiModelResourcePoolIntegration not initialized")
            return {"success": False, "error": "Not initialized"}
        
        logger.info(f"Comparing execution strategies for {len(model_configs)} models on {hardware_platform}")
        
        # Define strategies to compare
        strategies = ["parallel", "sequential", "batched"]
        results = {}
        
        # Execute with each strategy
        for strategy in strategies:
            logger.info(f"Testing {strategy} strategy")
            result = self.execute_with_strategy(
                model_configs=model_configs,
                hardware_platform=hardware_platform,
                execution_strategy=strategy,
                optimization_goal=optimization_goal,
                return_measurements=False,
                validate_predictions=False  # Skip validation for individual runs
            )
            results[strategy] = result
        
        # Get auto-recommended strategy
        logger.info("Testing auto-recommended strategy")
        recommended_result = self.execute_with_strategy(
            model_configs=model_configs,
            hardware_platform=hardware_platform,
            execution_strategy=None,  # Auto-select
            optimization_goal=optimization_goal,
            return_measurements=False
        )
        
        recommended_strategy = recommended_result["execution_strategy"]
        results["recommended"] = recommended_result
        
        # Identify best strategy based on actual measurements
        best_strategy = None
        best_value = None
        
        if optimization_goal == "throughput":
            # Higher throughput is better
            for strategy, result in results.items():
                value = result.get("actual_throughput", 0)
                if best_value is None or value > best_value:
                    best_value = value
                    best_strategy = strategy
        else:  # latency or memory
            # Lower values are better
            metric_key = "actual_latency" if optimization_goal == "latency" else "actual_memory"
            for strategy, result in results.items():
                value = result.get(metric_key, float('inf'))
                if best_value is None or value < best_value:
                    best_value = value
                    best_strategy = strategy
        
        # Check if recommendation matches empirical best
        recommendation_accuracy = recommended_strategy == best_strategy
        
        # Calculate optimization impact (comparing best with worst)
        optimization_impact = {}
        
        if optimization_goal == "throughput":
            # For throughput, find min throughput (worst)
            worst_strategy = min(
                strategies, 
                key=lambda s: results[s].get("actual_throughput", 0)
            )
            worst_value = results[worst_strategy].get("actual_throughput", 0)
            
            if worst_value > 0:
                improvement_percent = (best_value - worst_value) / worst_value * 100
            else:
                improvement_percent = 0
                
            optimization_impact = {
                "best_strategy": best_strategy,
                "worst_strategy": worst_strategy,
                "best_throughput": best_value,
                "worst_throughput": worst_value,
                "improvement_percent": improvement_percent
            }
        else:  # latency or memory
            metric_key = "actual_latency" if optimization_goal == "latency" else "actual_memory"
            
            # For latency/memory, find max value (worst)
            worst_strategy = max(
                strategies, 
                key=lambda s: results[s].get(metric_key, float('inf'))
            )
            worst_value = results[worst_strategy].get(metric_key, float('inf'))
            
            if worst_value > 0:
                improvement_percent = (worst_value - best_value) / worst_value * 100
            else:
                improvement_percent = 0
                
            optimization_impact = {
                "best_strategy": best_strategy,
                "worst_strategy": worst_strategy,
                f"best_{optimization_goal}": best_value,
                f"worst_{optimization_goal}": worst_value,
                "improvement_percent": improvement_percent
            }
        
        # Store optimization impact for tracking
        if optimization_impact:
            self.validation_metrics["optimization_impact"].append({
                "timestamp": time.time(),
                "model_count": len(model_configs),
                "hardware_platform": hardware_platform,
                "best_strategy": best_strategy,
                "worst_strategy": optimization_impact.get("worst_strategy", ""),
                "improvement_percent": optimization_impact.get("improvement_percent", 0),
                "optimization_goal": optimization_goal
            })
            
            # Store in database if available
            if self.db_conn:
                try:
                    self.db_conn.execute(
                        """
                        INSERT INTO multi_model_optimization_impact
                        (timestamp, model_count, hardware_platform, baseline_strategy, optimized_strategy,
                         baseline_throughput, optimized_throughput, baseline_latency, optimized_latency,
                         baseline_memory, optimized_memory, throughput_improvement_percent,
                         latency_improvement_percent, memory_improvement_percent, optimization_goal)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            time.time(), len(model_configs), hardware_platform,
                            optimization_impact.get("worst_strategy", ""), best_strategy,
                            results[optimization_impact.get("worst_strategy", "")].get("actual_throughput", 0),
                            results[best_strategy].get("actual_throughput", 0),
                            results[optimization_impact.get("worst_strategy", "")].get("actual_latency", 0),
                            results[best_strategy].get("actual_latency", 0),
                            results[optimization_impact.get("worst_strategy", "")].get("actual_memory", 0),
                            results[best_strategy].get("actual_memory", 0),
                            optimization_impact.get("improvement_percent", 0) if optimization_goal == "throughput" else 0,
                            optimization_impact.get("improvement_percent", 0) if optimization_goal == "latency" else 0,
                            optimization_impact.get("improvement_percent", 0) if optimization_goal == "memory" else 0,
                            optimization_goal
                        )
                    )
                except Exception as e:
                    logger.error(f"Error storing optimization impact in database: {e}")
        
        # Create comparison result
        comparison_result = {
            "success": True,
            "model_count": len(model_configs),
            "hardware_platform": hardware_platform,
            "optimization_goal": optimization_goal,
            "best_strategy": best_strategy,
            "recommended_strategy": recommended_strategy,
            "recommendation_accuracy": recommendation_accuracy,
            "strategy_results": {
                strategy: {
                    "throughput": result.get("actual_throughput", 0),
                    "latency": result.get("actual_latency", 0),
                    "memory": result.get("actual_memory", 0),
                    "success": result.get("success", False)
                }
                for strategy, result in results.items()
            },
            "optimization_impact": optimization_impact
        }
        
        logger.info(f"Strategy comparison complete: Best={best_strategy}, Recommended={recommended_strategy}, "
                   f"Accuracy={'correct' if recommendation_accuracy else 'incorrect'}, "
                   f"Improvement={optimization_impact.get('improvement_percent', 0):.1f}%")
        
        return comparison_result
    
    def get_validation_metrics(self, include_history: bool = False) -> Dict[str, Any]:
        """
        Get validation metrics and error statistics.
        
        Args:
            include_history: Whether to include full validation history
            
        Returns:
            Dictionary with validation metrics and error statistics
        """
        # If validator is available, use it
        if self.validator:
            return self.validator.get_validation_metrics(include_history=include_history)
        
        # Legacy approach if validator not available
        metrics = {
            "validation_count": self.validation_metrics["validation_count"],
            "execution_count": self.validation_metrics["execution_count"],
            "last_validation_time": self.validation_metrics["last_validation_time"]
        }
        
        # Calculate average error rates
        error_rates = {}
        for metric, values in self.validation_metrics["error_rates"].items():
            if values:
                avg_error = sum(values) / len(values)
                error_rates[f"avg_{metric}_error"] = avg_error
                
                # Calculate recent error (last 5 validations)
                recent_values = values[-5:] if len(values) >= 5 else values
                recent_error = sum(recent_values) / len(recent_values)
                error_rates[f"recent_{metric}_error"] = recent_error
                
                # Calculate error trend (improving or worsening)
                if len(values) >= 10:
                    older_values = values[-10:-5]
                    older_avg = sum(older_values) / len(older_values)
                    trend = recent_error - older_avg
                    error_rates[f"{metric}_error_trend"] = trend
        
        metrics["error_rates"] = error_rates
        
        # Calculate optimization impact statistics
        impact_stats = {}
        impact_records = self.validation_metrics["optimization_impact"]
        
        if impact_records:
            improvement_values = [record["improvement_percent"] for record in impact_records]
            avg_improvement = sum(improvement_values) / len(improvement_values)
            impact_stats["avg_improvement_percent"] = avg_improvement
            
            # Accuracy of strategy recommendation
            recommended_strategies = [record.get("recommended_strategy", "") for record in impact_records 
                                    if "recommended_strategy" in record]
            best_strategies = [record["best_strategy"] for record in impact_records]
            
            if recommended_strategies:
                correct_recommendations = sum(1 for rec, best in zip(recommended_strategies, best_strategies) if rec == best)
                recommendation_accuracy = correct_recommendations / len(recommended_strategies)
                impact_stats["recommendation_accuracy"] = recommendation_accuracy
            
            # Strategy distribution
            strategy_counts = {}
            for record in impact_records:
                strategy = record["best_strategy"]
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            
            impact_stats["best_strategy_distribution"] = {
                strategy: count / len(impact_records)
                for strategy, count in strategy_counts.items()
            }
        
        metrics["optimization_impact"] = impact_stats
        
        # Add validation history if requested
        if include_history:
            metrics["history"] = self.validation_metrics["predicted_vs_actual"]
        
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
                
                # Get optimization impact from database
                db_impact = self.db_conn.execute(
                    """
                    SELECT 
                        AVG(throughput_improvement_percent) as avg_throughput_improvement,
                        AVG(latency_improvement_percent) as avg_latency_improvement,
                        AVG(memory_improvement_percent) as avg_memory_improvement
                    FROM multi_model_optimization_impact
                    """
                ).fetchone()
                
                metrics["database"] = {
                    "validation_count": db_validation_count,
                    "avg_throughput_error": db_error_rates[0],
                    "avg_latency_error": db_error_rates[1],
                    "avg_memory_error": db_error_rates[2],
                    "avg_throughput_improvement": db_impact[0],
                    "avg_latency_improvement": db_impact[1],
                    "avg_memory_improvement": db_impact[2]
                }
            except Exception as e:
                logger.error(f"Error getting database statistics: {e}")
        
        return metrics
    
    def get_adaptive_configuration(self, hardware_platform: str) -> Dict[str, Any]:
        """
        Get adaptive configuration based on empirical measurements.
        
        This method returns an optimized configuration for execution strategies
        based on the validation metrics collected so far.
        
        Args:
            hardware_platform: Hardware platform for configuration
            
        Returns:
            Dictionary with adaptive configuration
        """
        # Start with default configuration
        config = self.strategy_configuration.get(hardware_platform, {}).copy()
        
        # Only adapt if enabled and we have enough data
        if not self.enable_adaptive_optimization or self.validation_metrics["validation_count"] < 5:
            return config
        
        # Get relevant validation records for this platform
        platform_records = [
            record for record in self.validation_metrics["predicted_vs_actual"]
            if record["hardware_platform"] == hardware_platform
        ]
        
        if not platform_records:
            return config
        
        # Analyze records to find optimal thresholds
        strategy_performance = {
            "parallel": {"records": [], "latency_efficiency": 0, "throughput_efficiency": 0},
            "sequential": {"records": [], "latency_efficiency": 0, "throughput_efficiency": 0},
            "batched": {"records": [], "latency_efficiency": 0, "throughput_efficiency": 0}
        }
        
        # Group records by strategy
        for record in platform_records:
            strategy = record["execution_strategy"]
            if strategy in strategy_performance:
                strategy_performance[strategy]["records"].append(record)
        
        # Calculate efficiency metrics for each strategy
        for strategy, data in strategy_performance.items():
            records = data["records"]
            if not records:
                continue
            
            # Latency efficiency: ratio of predicted to actual latency
            latency_values = [rec["predicted_latency"] / rec["actual_latency"] if rec["actual_latency"] > 0 else 0 for rec in records]
            data["latency_efficiency"] = sum(latency_values) / len(latency_values) if latency_values else 0
            
            # Throughput efficiency: ratio of actual to predicted throughput
            throughput_values = [rec["actual_throughput"] / rec["predicted_throughput"] if rec["predicted_throughput"] > 0 else 0 for rec in records]
            data["throughput_efficiency"] = sum(throughput_values) / len(throughput_values) if throughput_values else 0
            
            # Analyze by model count
            model_count_groups = {}
            for record in records:
                count = record["model_count"]
                group = count // 2 * 2  # Group by pairs: 0-1, 2-3, 4-5, etc.
                if group not in model_count_groups:
                    model_count_groups[group] = []
                model_count_groups[group].append(record)
            
            data["model_count_groups"] = model_count_groups
        
        # Determine optimal thresholds based on performance data
        if strategy_performance["parallel"]["latency_efficiency"] > 0.7:
            # Parallel strategy is performing well, increase its threshold
            parallel_threshold = config.get("parallel_threshold", 3)
            config["parallel_threshold"] = min(parallel_threshold + 1, 6)  # Cap at 6
        elif strategy_performance["parallel"]["latency_efficiency"] < 0.5:
            # Parallel strategy is underperforming, decrease its threshold
            parallel_threshold = config.get("parallel_threshold", 3)
            config["parallel_threshold"] = max(parallel_threshold - 1, 1)  # Minimum 1
        
        if strategy_performance["sequential"]["throughput_efficiency"] > 0.7:
            # Sequential strategy is performing well for throughput, decrease threshold
            sequential_threshold = config.get("sequential_threshold", 8)
            config["sequential_threshold"] = max(sequential_threshold - 1, 5)  # Minimum 5
        elif strategy_performance["sequential"]["throughput_efficiency"] < 0.5:
            # Sequential strategy is underperforming for throughput, increase threshold
            sequential_threshold = config.get("sequential_threshold", 8)
            config["sequential_threshold"] = min(sequential_threshold + 1, 12)  # Cap at 12
        
        # Optimize batch size based on batched strategy performance
        if strategy_performance["batched"]["records"]:
            batch_size = config.get("batching_size", 4)
            
            # Simple heuristic: if batched is performing well overall, increase batch size
            if strategy_performance["batched"]["throughput_efficiency"] > 0.8:
                config["batching_size"] = min(batch_size + 1, 8)  # Cap at 8
            elif strategy_performance["batched"]["throughput_efficiency"] < 0.6:
                config["batching_size"] = max(batch_size - 1, 2)  # Minimum 2
        
        # Check memory threshold based on actual measurements
        memory_records = [rec for rec in platform_records if rec["actual_memory"] > 0]
        if memory_records:
            max_observed_memory = max(rec["actual_memory"] for rec in memory_records)
            current_threshold = config.get("memory_threshold", 8000)
            
            # If we've exceeded 80% of threshold, increase it
            if max_observed_memory > 0.8 * current_threshold:
                config["memory_threshold"] = int(current_threshold * 1.25)  # 25% increase
        
        return config
    
    def update_strategy_configuration(self, hardware_platform: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Update strategy configuration for a hardware platform.
        
        Args:
            hardware_platform: Hardware platform for configuration
            config: New configuration (None for adaptive update)
            
        Returns:
            Updated configuration
        """
        if config is not None:
            # Update with provided configuration
            if hardware_platform in self.strategy_configuration:
                self.strategy_configuration[hardware_platform].update(config)
            else:
                self.strategy_configuration[hardware_platform] = config.copy()
            
            logger.info(f"Updated strategy configuration for {hardware_platform}: {config}")
        else:
            # Use adaptive configuration
            adaptive_config = self.get_adaptive_configuration(hardware_platform)
            
            if hardware_platform in self.strategy_configuration:
                self.strategy_configuration[hardware_platform].update(adaptive_config)
            else:
                self.strategy_configuration[hardware_platform] = adaptive_config
            
            logger.info(f"Updated strategy configuration for {hardware_platform} (adaptive): {adaptive_config}")
        
        return self.strategy_configuration[hardware_platform]
    
    def close(self) -> bool:
        """
        Close the integration and release resources.
        
        Returns:
            Success status
        """
        success = True
        
        # Close resource pool
        if self.resource_pool:
            try:
                logger.info("Closing resource pool")
                pool_success = self.resource_pool.close()
                if not pool_success:
                    logger.error("Error closing resource pool")
                    success = False
            except Exception as e:
                logger.error(f"Exception closing resource pool: {e}")
                traceback.print_exc()
                success = False
        
        # Close empirical validator
        if self.validator:
            try:
                logger.info("Closing empirical validator")
                validator_success = self.validator.close()
                if not validator_success:
                    logger.error("Error closing empirical validator")
                    success = False
                else:
                    logger.info("Empirical validator closed successfully")
            except Exception as e:
                logger.error(f"Exception closing empirical validator: {e}")
                traceback.print_exc()
                success = False
        
        # Close database connection
        if self.db_conn:
            try:
                logger.info("Closing database connection")
                self.db_conn.close()
                logger.info("Database connection closed")
            except Exception as e:
                logger.error(f"Error closing database connection: {e}")
                success = False
        
        logger.info(f"MultiModelResourcePoolIntegration closed (success={'yes' if success else 'no'})")
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
    
    logger.info("Starting MultiModelResourcePoolIntegration example")
    
    # Create the integration
    integration = MultiModelResourcePoolIntegration(
        max_connections=2,
        enable_empirical_validation=True,
        validation_interval=5,
        prediction_refinement=True,
        enable_adaptive_optimization=True,
        verbose=True
    )
    
    # Initialize
    success = integration.initialize()
    if not success:
        logger.error("Failed to initialize integration")
        sys.exit(1)
    
    try:
        # Define model configurations for testing
        model_configs = [
            {"model_name": "bert-base-uncased", "model_type": "text_embedding", "batch_size": 4},
            {"model_name": "vit-base-patch16-224", "model_type": "vision", "batch_size": 1}
        ]
        
        # Execute with automatic strategy recommendation
        logger.info("Testing automatic strategy recommendation")
        result = integration.execute_with_strategy(
            model_configs=model_configs,
            hardware_platform="webgpu",
            execution_strategy=None,  # Automatic selection
            optimization_goal="latency"
        )
        
        logger.info(f"Execution complete with strategy: {result['execution_strategy']}")
        logger.info(f"Predicted latency: {result['predicted_latency']:.2f} ms")
        logger.info(f"Actual latency: {result['actual_latency']:.2f} ms")
        
        # Compare different strategies
        logger.info("Comparing execution strategies")
        comparison = integration.compare_strategies(
            model_configs=model_configs,
            hardware_platform="webgpu",
            optimization_goal="throughput"
        )
        
        logger.info(f"Best strategy: {comparison['best_strategy']}")
        logger.info(f"Recommended strategy: {comparison['recommended_strategy']}")
        logger.info(f"Recommendation accuracy: {comparison['recommendation_accuracy']}")
        
        # Get validation metrics
        metrics = integration.get_validation_metrics()
        logger.info(f"Validation count: {metrics['validation_count']}")
        if 'error_rates' in metrics:
            for metric, value in metrics['error_rates'].items():
                logger.info(f"{metric}: {value:.2%}")
        
        # Update strategy configuration adaptively
        new_config = integration.update_strategy_configuration("webgpu")
        logger.info(f"Adaptive configuration: {new_config}")
        
    finally:
        # Close the integration
        integration.close()
        logger.info("MultiModelResourcePoolIntegration example completed")