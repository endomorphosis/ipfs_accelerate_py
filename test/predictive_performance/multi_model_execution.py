#!/usr/bin/env python3
"""
Multi-Model Execution Support for the Predictive Performance System.

This module provides functionality to predict performance metrics for scenarios
where multiple models are executed concurrently on the same hardware. It accounts
for resource contention, parallel execution benefits, and memory sharing
opportunities between models.

Key features:
1. Resource contention modeling for CPU, GPU, and memory
2. Cross-model tensor sharing efficiency prediction
3. Parallel execution scheduling simulation
4. Memory optimization modeling
5. Power usage prediction for multi-model workloads
6. Integration with Web Resource Pool for browser-based execution
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("predictive_performance.multi_model_execution")

# Suppress non-critical warnings
warnings.filterwarnings('ignore', category=UserWarning)

class MultiModelPredictor:
    """
    Predicts performance metrics for concurrent execution of multiple models.
    
    This class provides functionality to estimate throughput, latency, memory usage,
    and power consumption when multiple AI models are executed concurrently on the
    same hardware platform, accounting for resource contention and sharing.
    """
    
    def __init__(
        self,
        single_model_predictor=None,
        contention_model_path: Optional[str] = None,
        cross_model_sharing_config: Optional[str] = None,
        resource_pool_integration: bool = True,
        verbose: bool = False
    ):
        """
        Initialize the multi-model predictor.
        
        Args:
            single_model_predictor: Existing single-model performance predictor instance
            contention_model_path: Path to trained resource contention models
            cross_model_sharing_config: Path to cross-model tensor sharing configuration
            resource_pool_integration: Whether to integrate with Web Resource Pool
            verbose: Whether to enable verbose logging
        """
        self.single_model_predictor = single_model_predictor
        self.contention_model_path = contention_model_path
        self.cross_model_sharing_config = cross_model_sharing_config
        self.resource_pool_integration = resource_pool_integration
        
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        # Initialize contention models
        self.cpu_contention_model = None
        self.gpu_contention_model = None
        self.memory_contention_model = None
        
        # Initialize sharing optimization models
        self.tensor_sharing_model = None
        
        # Load models if paths provided
        if self.contention_model_path:
            self._load_contention_models()
        
        # Load cross-model sharing configuration
        self.sharing_config = {}
        if self.cross_model_sharing_config:
            self._load_sharing_config()
        else:
            # Default configuration based on model families
            self._initialize_default_sharing_config()
        
        logger.info("Multi-Model Execution Predictor initialized")
    
    def _load_contention_models(self):
        """Load trained resource contention models."""
        logger.debug(f"Loading contention models from {self.contention_model_path}")
        
        # Placeholder for actual model loading
        # In a complete implementation, this would load scikit-learn or other ML models
        
        logger.info("Resource contention models loaded")
    
    def _load_sharing_config(self):
        """Load cross-model tensor sharing configuration."""
        logger.debug(f"Loading sharing configuration from {self.cross_model_sharing_config}")
        
        try:
            import json
            with open(self.cross_model_sharing_config, 'r') as f:
                self.sharing_config = json.load(f)
            
            logger.info("Cross-model sharing configuration loaded")
        except Exception as e:
            logger.error(f"Failed to load sharing configuration: {e}")
            # Fall back to default configuration
            self._initialize_default_sharing_config()
    
    def _initialize_default_sharing_config(self):
        """Initialize default cross-model sharing configuration."""
        logger.debug("Initializing default sharing configuration")
        
        # Define sharing compatibility for different model types
        self.sharing_config = {
            "text_embedding": {
                "compatible_types": ["text_embedding", "text_generation"],
                "sharing_efficiency": 0.4,  # 40% of embeddings can be shared
                "memory_reduction": 0.25    # 25% memory reduction from sharing
            },
            "text_generation": {
                "compatible_types": ["text_embedding", "text_generation"],
                "sharing_efficiency": 0.3,
                "memory_reduction": 0.2
            },
            "vision": {
                "compatible_types": ["vision", "multimodal"],
                "sharing_efficiency": 0.35,
                "memory_reduction": 0.3
            },
            "audio": {
                "compatible_types": ["audio", "multimodal"],
                "sharing_efficiency": 0.25,
                "memory_reduction": 0.15
            },
            "multimodal": {
                "compatible_types": ["vision", "text_embedding", "audio", "multimodal"],
                "sharing_efficiency": 0.2,
                "memory_reduction": 0.1
            }
        }
        
        logger.info("Default sharing configuration initialized")
    
    def predict_multi_model_performance(
        self,
        model_configs: List[Dict[str, Any]],
        hardware_platform: str,
        execution_strategy: str = "parallel",
        resource_constraints: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Predict performance metrics for concurrent execution of multiple models.
        
        Args:
            model_configs: List of model configurations to execute concurrently
            hardware_platform: Hardware platform for execution
            execution_strategy: Strategy for execution ("parallel", "sequential", or "batched")
            resource_constraints: Optional resource constraints (memory limit, etc.)
            
        Returns:
            Dictionary with predicted performance metrics
        """
        logger.info(f"Predicting performance for {len(model_configs)} models on {hardware_platform}")
        logger.debug(f"Execution strategy: {execution_strategy}")
        
        # Get single-model predictions first
        single_model_predictions = []
        
        if self.single_model_predictor:
            for config in model_configs:
                # Get prediction for individual model
                prediction = self.single_model_predictor.predict(
                    model_name=config.get("model_name", ""),
                    model_type=config.get("model_type", ""),
                    hardware_platform=hardware_platform,
                    batch_size=config.get("batch_size", 1)
                )
                single_model_predictions.append(prediction)
        else:
            # Simulate predictions if no predictor available
            logger.warning("No single-model predictor available, using simulation")
            for config in model_configs:
                # Create simulated prediction
                prediction = self._simulate_single_model_prediction(config, hardware_platform)
                single_model_predictions.append(prediction)
        
        # Calculate resource contention
        contention_factors = self._calculate_resource_contention(
            single_model_predictions,
            hardware_platform,
            execution_strategy
        )
        
        # Calculate sharing benefits
        sharing_benefits = self._calculate_sharing_benefits(
            model_configs,
            single_model_predictions
        )
        
        # Calculate total metrics with contention and sharing
        total_metrics = self._calculate_multi_model_metrics(
            single_model_predictions,
            contention_factors,
            sharing_benefits,
            execution_strategy
        )
        
        # Add execution scheduling information
        scheduling_info = self._generate_execution_schedule(
            model_configs,
            single_model_predictions,
            contention_factors,
            execution_strategy
        )
        
        # Combine all results
        result = {
            "total_metrics": total_metrics,
            "individual_predictions": single_model_predictions,
            "contention_factors": contention_factors,
            "sharing_benefits": sharing_benefits,
            "execution_schedule": scheduling_info,
            "execution_strategy": execution_strategy,
            "model_count": len(model_configs),
            "hardware_platform": hardware_platform
        }
        
        return result
    
    def _simulate_single_model_prediction(
        self,
        model_config: Dict[str, Any],
        hardware_platform: str
    ) -> Dict[str, Any]:
        """
        Simulate prediction for a single model when no predictor is available.
        
        Args:
            model_config: Model configuration
            hardware_platform: Hardware platform
            
        Returns:
            Simulated prediction
        """
        model_type = model_config.get("model_type", "text_embedding")
        batch_size = model_config.get("batch_size", 1)
        
        # Base metrics by model type
        base_metrics = {
            "text_embedding": {"throughput": 100, "latency": 10, "memory": 1000},
            "text_generation": {"throughput": 20, "latency": 100, "memory": 4000},
            "vision": {"throughput": 50, "latency": 30, "memory": 2000},
            "audio": {"throughput": 10, "latency": 200, "memory": 3000},
            "multimodal": {"throughput": 5, "latency": 300, "memory": 6000}
        }
        
        # Hardware factors
        hw_factors = {
            "cpu": {"throughput": 1.0, "latency": 1.0, "memory": 1.0},
            "cuda": {"throughput": 8.0, "latency": 0.2, "memory": 1.2},
            "rocm": {"throughput": 7.0, "latency": 0.25, "memory": 1.2},
            "openvino": {"throughput": 3.0, "latency": 0.5, "memory": 0.9},
            "webgpu": {"throughput": 2.5, "latency": 0.6, "memory": 1.0}
        }
        
        # Get base metrics for model type
        metrics = base_metrics.get(model_type, base_metrics["text_embedding"])
        
        # Apply hardware factors
        factors = hw_factors.get(hardware_platform, hw_factors["cpu"])
        
        # Calculate metrics with batch size effects
        throughput = metrics["throughput"] * factors["throughput"] * (batch_size ** 0.7)
        latency = metrics["latency"] * factors["latency"] * (1 + 0.1 * batch_size)
        memory = metrics["memory"] * factors["memory"] * (1 + 0.2 * (batch_size - 1))
        
        # Add some randomness
        import random
        random.seed(hash(f"{model_type}_{hardware_platform}_{batch_size}"))
        throughput *= random.uniform(0.9, 1.1)
        latency *= random.uniform(0.9, 1.1)
        memory *= random.uniform(0.9, 1.1)
        
        return {
            "model_config": model_config,
            "hardware_platform": hardware_platform,
            "throughput": throughput,
            "latency": latency,
            "memory": memory,
            "simulated": True
        }
    
    def _calculate_resource_contention(
        self,
        single_model_predictions: List[Dict[str, Any]],
        hardware_platform: str,
        execution_strategy: str
    ) -> Dict[str, float]:
        """
        Calculate resource contention factors when running multiple models.
        
        Args:
            single_model_predictions: List of individual model predictions
            hardware_platform: Hardware platform
            execution_strategy: Execution strategy
            
        Returns:
            Dictionary with contention factors for different resources
        """
        logger.debug("Calculating resource contention factors")
        
        # Extract total resource usage
        total_memory = sum(pred["memory"] for pred in single_model_predictions)
        
        # Calculate CPU contention based on model count
        model_count = len(single_model_predictions)
        
        # Different contention models for different hardware platforms
        if hardware_platform in ["cuda", "rocm"]:
            # GPU contention factors
            compute_contention = 1.0 + 0.15 * (model_count - 1)  # 15% penalty per additional model
            memory_bandwidth_contention = 1.0 + 0.25 * (model_count - 1)  # 25% penalty per additional model
            
            if execution_strategy == "parallel":
                # Parallel execution has higher contention
                compute_contention *= 1.2
                memory_bandwidth_contention *= 1.3
            elif execution_strategy == "batched":
                # Batched execution has moderate contention
                compute_contention *= 1.1
                memory_bandwidth_contention *= 1.15
            
        elif hardware_platform in ["webgpu", "webnn"]:
            # WebGPU/WebNN contention factors
            compute_contention = 1.0 + 0.2 * (model_count - 1)  # 20% penalty per additional model
            memory_bandwidth_contention = 1.0 + 0.3 * (model_count - 1)  # 30% penalty per additional model
            
            if execution_strategy == "parallel":
                compute_contention *= 1.25
                memory_bandwidth_contention *= 1.35
            elif execution_strategy == "batched":
                compute_contention *= 1.15
                memory_bandwidth_contention *= 1.2
            
        else:
            # CPU contention factors
            compute_contention = 1.0 + 0.1 * (model_count - 1)  # 10% penalty per additional model
            memory_bandwidth_contention = 1.0 + 0.15 * (model_count - 1)  # 15% penalty per additional model
            
            if execution_strategy == "parallel":
                compute_contention *= 1.15
                memory_bandwidth_contention *= 1.25
            elif execution_strategy == "batched":
                compute_contention *= 1.05
                memory_bandwidth_contention *= 1.1
        
        # Memory contention occurs when total memory exceeds threshold
        # We assume different thresholds for different platforms
        memory_thresholds = {
            "cpu": 32000,  # 32 GB
            "cuda": 24000,  # 24 GB
            "rocm": 16000,  # 16 GB
            "openvino": 8000,  # 8 GB
            "webgpu": 4000,  # 4 GB
            "webnn": 4000,  # 4 GB
        }
        
        threshold = memory_thresholds.get(hardware_platform, 8000)
        memory_contention = 1.0
        
        if total_memory > threshold:
            # Calculate memory contention based on overflow
            overflow_ratio = total_memory / threshold
            memory_contention = overflow_ratio ** 1.5  # Non-linear penalty for memory overflow
        
        return {
            "compute_contention": compute_contention,
            "memory_bandwidth_contention": memory_bandwidth_contention,
            "memory_contention": memory_contention,
            "model_count": model_count,
            "total_memory": total_memory
        }
    
    def _calculate_sharing_benefits(
        self,
        model_configs: List[Dict[str, Any]],
        single_model_predictions: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate benefits from cross-model tensor sharing.
        
        Args:
            model_configs: List of model configurations
            single_model_predictions: List of individual model predictions
            
        Returns:
            Dictionary with sharing benefit factors
        """
        logger.debug("Calculating cross-model sharing benefits")
        
        # Group models by type
        model_types = {}
        for config in model_configs:
            model_type = config.get("model_type", "")
            if model_type in model_types:
                model_types[model_type].append(config)
            else:
                model_types[model_type] = [config]
        
        # Calculate sharing benefits for each type
        memory_savings = 0.0
        compute_savings = 0.0
        
        # Track compatible pairs for sharing
        compatible_pairs = 0
        
        # Check all model pairs for compatibility
        for i, config1 in enumerate(model_configs):
            type1 = config1.get("model_type", "")
            
            # Skip if type not in sharing config
            if type1 not in self.sharing_config:
                continue
                
            sharing_info = self.sharing_config[type1]
            compatible_types = sharing_info.get("compatible_types", [])
            
            for j in range(i+1, len(model_configs)):
                config2 = model_configs[j]
                type2 = config2.get("model_type", "")
                
                # Check if types are compatible for sharing
                if type2 in compatible_types:
                    compatible_pairs += 1
                    
                    # Get sharing metrics
                    sharing_efficiency = sharing_info.get("sharing_efficiency", 0.0)
                    memory_reduction = sharing_info.get("memory_reduction", 0.0)
                    
                    # Accumulate savings
                    memory_savings += memory_reduction
                    compute_savings += sharing_efficiency * 0.5  # Compute savings are typically half of sharing efficiency
        
        # Calculate final benefit factors
        total_models = len(model_configs)
        
        if total_models <= 1 or compatible_pairs == 0:
            # No sharing possible with 0 or 1 models or no compatible pairs
            memory_benefit = 1.0
            compute_benefit = 1.0
        else:
            # Scale benefits based on model count and compatible pairs
            # The formula provides diminishing returns as more models are added
            max_pairs = (total_models * (total_models - 1)) / 2
            pair_ratio = compatible_pairs / max_pairs
            
            # Memory benefit: Reduce memory requirements
            memory_benefit = 1.0 - (memory_savings * pair_ratio / total_models)
            memory_benefit = max(0.7, memory_benefit)  # Cap at 30% reduction
            
            # Compute benefit: Reduce computation through shared operations
            compute_benefit = 1.0 - (compute_savings * pair_ratio / total_models)
            compute_benefit = max(0.8, compute_benefit)  # Cap at 20% reduction
        
        return {
            "memory_benefit": memory_benefit,
            "compute_benefit": compute_benefit,
            "compatible_pairs": compatible_pairs,
            "total_models": total_models
        }
    
    def _calculate_multi_model_metrics(
        self,
        single_model_predictions: List[Dict[str, Any]],
        contention_factors: Dict[str, float],
        sharing_benefits: Dict[str, float],
        execution_strategy: str
    ) -> Dict[str, float]:
        """
        Calculate total performance metrics for multi-model execution.
        
        Args:
            single_model_predictions: List of individual model predictions
            contention_factors: Resource contention factors
            sharing_benefits: Cross-model sharing benefit factors
            execution_strategy: Execution strategy
            
        Returns:
            Dictionary with combined performance metrics
        """
        logger.debug("Calculating multi-model execution metrics")
        
        # Get contention factors
        compute_contention = contention_factors["compute_contention"]
        memory_bandwidth_contention = contention_factors["memory_bandwidth_contention"]
        memory_contention = contention_factors["memory_contention"]
        
        # Get sharing benefits
        memory_benefit = sharing_benefits["memory_benefit"]
        compute_benefit = sharing_benefits["compute_benefit"]
        
        # Calculate combined metrics based on execution strategy
        if execution_strategy == "sequential":
            # Sequential execution: Sum latencies, take max memory, no throughput improvement
            total_latency = sum(pred["latency"] for pred in single_model_predictions)
            total_memory = max(pred["memory"] for pred in single_model_predictions)
            total_memory *= memory_benefit  # Apply sharing benefit
            
            # For sequential, throughput is determined by total latency
            total_throughput = sum(pred["throughput"] for pred in single_model_predictions) / len(single_model_predictions)
            
            # Apply contention only to memory bandwidth (affects latency)
            total_latency *= memory_bandwidth_contention * compute_benefit
            
        elif execution_strategy == "parallel":
            # Parallel execution: Take max latency, sum memory, potential throughput improvement
            total_latency = max(pred["latency"] for pred in single_model_predictions)
            total_memory = sum(pred["memory"] for pred in single_model_predictions)
            
            # Apply sharing benefit to memory
            total_memory *= memory_benefit
            
            # Apply contention to latency
            total_latency *= compute_contention * memory_bandwidth_contention
            
            # For parallel, throughput is sum with contention applied
            raw_throughput = sum(pred["throughput"] for pred in single_model_predictions)
            total_throughput = raw_throughput / (compute_contention * compute_benefit)
            
        else:  # batched
            # Batched execution: Between sequential and parallel
            # Use weighted average of sequential and parallel metrics
            
            # Calculate sequential metrics
            seq_latency = sum(pred["latency"] for pred in single_model_predictions)
            seq_memory = max(pred["memory"] for pred in single_model_predictions)
            seq_throughput = sum(pred["throughput"] for pred in single_model_predictions) / len(single_model_predictions)
            
            # Calculate parallel metrics
            par_latency = max(pred["latency"] for pred in single_model_predictions)
            par_memory = sum(pred["memory"] for pred in single_model_predictions)
            raw_throughput = sum(pred["throughput"] for pred in single_model_predictions)
            par_throughput = raw_throughput / compute_contention
            
            # Weight between sequential and parallel (60% parallel, 40% sequential)
            weight_parallel = 0.6
            weight_sequential = 0.4
            
            total_latency = (par_latency * weight_parallel) + (seq_latency * weight_sequential)
            total_memory = (par_memory * weight_parallel) + (seq_memory * weight_sequential)
            total_throughput = (par_throughput * weight_parallel) + (seq_throughput * weight_sequential)
            
            # Apply sharing benefits
            total_memory *= memory_benefit
            total_throughput /= compute_benefit
            
            # Apply contention
            total_latency *= (compute_contention * 0.7) + (memory_bandwidth_contention * 0.3)
        
        # Apply memory contention to all strategies if it exceeds threshold
        if memory_contention > 1.0:
            # Memory contention affects both latency and throughput
            total_latency *= memory_contention
            total_throughput /= memory_contention
        
        # Round to reasonable precision
        total_latency = round(total_latency, 2)
        total_memory = round(total_memory, 2)
        total_throughput = round(total_throughput, 2)
        
        return {
            "combined_throughput": total_throughput,
            "combined_latency": total_latency,
            "combined_memory": total_memory,
            "execution_strategy": execution_strategy,
            "model_count": len(single_model_predictions)
        }
    
    def _generate_execution_schedule(
        self,
        model_configs: List[Dict[str, Any]],
        single_model_predictions: List[Dict[str, Any]],
        contention_factors: Dict[str, float],
        execution_strategy: str
    ) -> Dict[str, Any]:
        """
        Generate an execution schedule for multiple models.
        
        Args:
            model_configs: List of model configurations
            single_model_predictions: List of individual model predictions
            contention_factors: Resource contention factors
            execution_strategy: Execution strategy
            
        Returns:
            Dictionary with execution scheduling information
        """
        logger.debug("Generating execution schedule")
        
        # Create schedule based on strategy
        if execution_strategy == "sequential":
            # For sequential, create a simple ordering based on model size
            # Smaller models first to minimize memory fluctuations
            order = []
            for i, pred in enumerate(single_model_predictions):
                order.append((i, pred["memory"]))
            
            # Sort by memory (ascending)
            order.sort(key=lambda x: x[1])
            
            # Create timeline based on latencies
            timeline = []
            current_time = 0
            
            for idx, _ in order:
                pred = single_model_predictions[idx]
                config = model_configs[idx]
                
                start_time = current_time
                # Apply contention factor to latency
                adjusted_latency = pred["latency"] * contention_factors["memory_bandwidth_contention"]
                end_time = start_time + adjusted_latency
                
                timeline.append({
                    "model": config.get("model_name", f"model_{idx}"),
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": adjusted_latency
                })
                
                current_time = end_time
            
            total_execution_time = current_time
            
            return {
                "execution_order": [model_configs[idx]["model_name"] for idx, _ in order],
                "timeline": timeline,
                "total_execution_time": total_execution_time,
                "strategy": "sequential"
            }
            
        elif execution_strategy == "parallel":
            # For parallel, all models start at the same time
            # but finish at different times based on their latency
            timeline = []
            max_end_time = 0
            
            for i, pred in enumerate(single_model_predictions):
                config = model_configs[i]
                
                start_time = 0
                # Apply contention factors to latency
                adjusted_latency = pred["latency"] * contention_factors["compute_contention"] * contention_factors["memory_bandwidth_contention"]
                end_time = start_time + adjusted_latency
                
                timeline.append({
                    "model": config.get("model_name", f"model_{i}"),
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": adjusted_latency
                })
                
                max_end_time = max(max_end_time, end_time)
            
            return {
                "execution_order": "parallel",
                "timeline": timeline,
                "total_execution_time": max_end_time,
                "strategy": "parallel"
            }
            
        else:  # batched
            # For batched, group models into batches based on memory usage
            # We'll use a simple bin packing algorithm
            
            # First, calculate memory threshold (this would be hardware-specific)
            memory_threshold = contention_factors.get("total_memory", 0) * 0.5  # 50% of total
            
            # Create items to pack with index and memory
            items = [(i, pred["memory"]) for i, pred in enumerate(single_model_predictions)]
            
            # Sort by memory (descending) to improve packing
            items.sort(key=lambda x: x[1], reverse=True)
            
            # Create batches using first-fit decreasing
            batches = []
            for idx, memory in items:
                # Try to add to existing batch
                added = False
                for batch in batches:
                    batch_memory = sum(single_model_predictions[i]["memory"] for i in batch)
                    if batch_memory + memory <= memory_threshold:
                        batch.append(idx)
                        added = True
                        break
                
                # If not added to any existing batch, create new batch
                if not added:
                    batches.append([idx])
            
            # Create timeline based on batches
            timeline = []
            current_time = 0
            
            for batch_idx, batch in enumerate(batches):
                # For each batch, execute models in parallel
                batch_timeline = []
                max_latency = 0
                
                for idx in batch:
                    pred = single_model_predictions[idx]
                    config = model_configs[idx]
                    
                    start_time = current_time
                    # Apply contention factors to latency, batch has lower contention than full parallel
                    adjusted_latency = pred["latency"] * (contention_factors["compute_contention"] * 0.8)
                    end_time = start_time + adjusted_latency
                    
                    batch_timeline.append({
                        "model": config.get("model_name", f"model_{idx}"),
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration": adjusted_latency,
                        "batch": batch_idx
                    })
                    
                    max_latency = max(max_latency, adjusted_latency)
                
                # Update current time based on max latency in batch
                current_time += max_latency
                timeline.extend(batch_timeline)
            
            # Convert batch indices to model names for clarity
            batch_order = [[model_configs[idx]["model_name"] for idx in batch] for batch in batches]
            
            return {
                "batch_execution_order": batch_order,
                "batches": len(batches),
                "timeline": timeline,
                "total_execution_time": current_time,
                "strategy": "batched"
            }
    
    def recommend_execution_strategy(
        self, 
        model_configs: List[Dict[str, Any]],
        hardware_platform: str,
        optimization_goal: str = "latency"
    ) -> Dict[str, Any]:
        """
        Recommend the best execution strategy for a set of models.
        
        Args:
            model_configs: List of model configurations to execute
            hardware_platform: Hardware platform for execution
            optimization_goal: Metric to optimize ("latency", "throughput", or "memory")
            
        Returns:
            Dictionary with recommended strategy and predicted metrics
        """
        logger.info(f"Recommending execution strategy for {len(model_configs)} models")
        logger.debug(f"Optimization goal: {optimization_goal}")
        
        # Try all execution strategies
        strategies = ["parallel", "sequential", "batched"]
        predictions = {}
        
        for strategy in strategies:
            prediction = self.predict_multi_model_performance(
                model_configs,
                hardware_platform,
                execution_strategy=strategy
            )
            predictions[strategy] = prediction
        
        # Determine best strategy based on optimization goal
        if optimization_goal == "latency":
            # Find strategy with lowest combined latency
            latencies = {
                strategy: pred["total_metrics"]["combined_latency"]
                for strategy, pred in predictions.items()
            }
            best_strategy = min(latencies, key=latencies.get)
            
        elif optimization_goal == "throughput":
            # Find strategy with highest combined throughput
            throughputs = {
                strategy: pred["total_metrics"]["combined_throughput"]
                for strategy, pred in predictions.items()
            }
            best_strategy = max(throughputs, key=throughputs.get)
            
        else:  # memory
            # Find strategy with lowest combined memory
            memories = {
                strategy: pred["total_metrics"]["combined_memory"]
                for strategy, pred in predictions.items()
            }
            best_strategy = min(memories, key=memories.get)
        
        # Prepare result with all predictions and recommendation
        result = {
            "recommended_strategy": best_strategy,
            "optimization_goal": optimization_goal,
            "all_predictions": {
                strategy: pred["total_metrics"]
                for strategy, pred in predictions.items()
            },
            "best_prediction": predictions[best_strategy],
            "model_count": len(model_configs),
            "hardware_platform": hardware_platform
        }
        
        return result

# Example usage
if __name__ == "__main__":
    # Initialize the multi-model predictor
    predictor = MultiModelPredictor(verbose=True)
    
    # Define some example model configurations
    model_configs = [
        {"model_name": "bert-base-uncased", "model_type": "text_embedding", "batch_size": 4},
        {"model_name": "vit-base-patch16-224", "model_type": "vision", "batch_size": 1},
        {"model_name": "t5-small", "model_type": "text_generation", "batch_size": 2}
    ]
    
    # Predict performance for concurrent execution
    prediction = predictor.predict_multi_model_performance(
        model_configs,
        hardware_platform="cuda",
        execution_strategy="parallel"
    )
    
    # Print results
    print("\nMulti-Model Execution Prediction:")
    print(f"Total Throughput: {prediction['total_metrics']['combined_throughput']:.2f} items/sec")
    print(f"Total Latency: {prediction['total_metrics']['combined_latency']:.2f} ms")
    print(f"Total Memory: {prediction['total_metrics']['combined_memory']:.2f} MB")
    
    # Recommend best execution strategy
    recommendation = predictor.recommend_execution_strategy(
        model_configs,
        hardware_platform="cuda",
        optimization_goal="throughput"
    )
    
    print("\nExecution Strategy Recommendation:")
    print(f"Recommended Strategy: {recommendation['recommended_strategy']}")
    print(f"Optimization Goal: {recommendation['optimization_goal']}")
    
    for strategy, metrics in recommendation['all_predictions'].items():
        print(f"\n{strategy.capitalize()} Strategy:")
        print(f"  Throughput: {metrics['combined_throughput']:.2f} items/sec")
        print(f"  Latency: {metrics['combined_latency']:.2f} ms")
        print(f"  Memory: {metrics['combined_memory']:.2f} MB")