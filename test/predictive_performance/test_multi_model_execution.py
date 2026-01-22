#!/usr/bin/env python3
"""
Test script for the Multi-Model Execution Support module.

This script tests the core functionality of the multi-model execution predictor,
including resource contention modeling, tensor sharing benefits, and execution
strategy recommendation.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore")

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

# Import the module to test
from predictive_performance.multi_model_execution import MultiModelPredictor


class TestMultiModelExecution(unittest.TestCase):
    """Test cases for the Multi-Model Execution Support module."""
    
    def setUp(self):
        """Set up before each test."""
        # Create a MultiModelPredictor instance
        self.predictor = MultiModelPredictor(verbose=True)
        
        # Define test model configurations
        self.model_configs = [
            {"model_name": "bert-base-uncased", "model_type": "text_embedding", "batch_size": 4},
            {"model_name": "vit-base-patch16-224", "model_type": "vision", "batch_size": 1}
        ]
    
    def test_initialization(self):
        """Test that the predictor initializes correctly."""
        self.assertIsNotNone(self.predictor)
        self.assertIsNotNone(self.predictor.sharing_config)
        
        # Check sharing config
        self.assertIn("text_embedding", self.predictor.sharing_config)
        self.assertIn("vision", self.predictor.sharing_config)
        
        # Check sharing compatibility
        text_sharing = self.predictor.sharing_config["text_embedding"]
        self.assertIn("compatible_types", text_sharing)
        self.assertIn("text_generation", text_sharing["compatible_types"])
    
    def test_single_model_prediction(self):
        """Test prediction for a single model."""
        # Create a model config
        model_config = {"model_name": "bert-base-uncased", "model_type": "text_embedding", "batch_size": 4}
        
        # Get prediction
        prediction = self.predictor._simulate_single_model_prediction(model_config, "cuda")
        
        # Check prediction has expected keys
        self.assertIn("throughput", prediction)
        self.assertIn("latency", prediction)
        self.assertIn("memory", prediction)
        
        # Check values are reasonable
        self.assertGreater(prediction["throughput"], 0)
        self.assertGreater(prediction["latency"], 0)
        self.assertGreater(prediction["memory"], 0)
    
    def test_resource_contention(self):
        """Test resource contention calculation."""
        # Create simulated single model predictions
        single_preds = [
            self.predictor._simulate_single_model_prediction(
                {"model_name": "bert-base-uncased", "model_type": "text_embedding", "batch_size": 4},
                "cuda"
            ),
            self.predictor._simulate_single_model_prediction(
                {"model_name": "vit-base-patch16-224", "model_type": "vision", "batch_size": 1},
                "cuda"
            )
        ]
        
        # Calculate contention
        contention = self.predictor._calculate_resource_contention(
            single_preds,
            "cuda",
            "parallel"
        )
        
        # Check contention has expected keys
        self.assertIn("compute_contention", contention)
        self.assertIn("memory_bandwidth_contention", contention)
        self.assertIn("memory_contention", contention)
        
        # Check contention is reasonable (higher than 1.0 for compute and memory bandwidth)
        self.assertGreater(contention["compute_contention"], 1.0)
        self.assertGreater(contention["memory_bandwidth_contention"], 1.0)
    
    def test_sharing_benefits(self):
        """Test tensor sharing benefits calculation."""
        # Calculate sharing benefits
        benefits = self.predictor._calculate_sharing_benefits(
            self.model_configs,
            [
                self.predictor._simulate_single_model_prediction(
                    self.model_configs[0], "cuda"
                ),
                self.predictor._simulate_single_model_prediction(
                    self.model_configs[1], "cuda"
                )
            ]
        )
        
        # Check benefits has expected keys
        self.assertIn("memory_benefit", benefits)
        self.assertIn("compute_benefit", benefits)
        self.assertIn("compatible_pairs", benefits)
        
        # Check benefits are reasonable (should be <= 1.0)
        self.assertLessEqual(benefits["memory_benefit"], 1.0)
        self.assertLessEqual(benefits["compute_benefit"], 1.0)
    
    def test_execution_schedule(self):
        """Test execution schedule generation."""
        # Get single model predictions
        single_preds = [
            self.predictor._simulate_single_model_prediction(
                self.model_configs[0], "cuda"
            ),
            self.predictor._simulate_single_model_prediction(
                self.model_configs[1], "cuda"
            )
        ]
        
        # Calculate contention
        contention = self.predictor._calculate_resource_contention(
            single_preds,
            "cuda",
            "parallel"
        )
        
        # Generate execution schedule for parallel execution
        schedule = self.predictor._generate_execution_schedule(
            self.model_configs,
            single_preds,
            contention,
            "parallel"
        )
        
        # Check schedule has expected keys
        self.assertIn("total_execution_time", schedule)
        self.assertIn("timeline", schedule)
        
        # Check timeline has events for each model
        self.assertEqual(len(schedule["timeline"]), len(self.model_configs))
        
        # For parallel execution, all start times should be 0
        for event in schedule["timeline"]:
            self.assertEqual(event["start_time"], 0)
    
    def test_multi_model_metrics(self):
        """Test multi-model metrics calculation."""
        # Get single model predictions
        single_preds = [
            self.predictor._simulate_single_model_prediction(
                self.model_configs[0], "cuda"
            ),
            self.predictor._simulate_single_model_prediction(
                self.model_configs[1], "cuda"
            )
        ]
        
        # Calculate contention
        contention = self.predictor._calculate_resource_contention(
            single_preds,
            "cuda",
            "parallel"
        )
        
        # Calculate sharing benefits
        benefits = self.predictor._calculate_sharing_benefits(
            self.model_configs,
            single_preds
        )
        
        # Calculate metrics
        metrics = self.predictor._calculate_multi_model_metrics(
            single_preds,
            contention,
            benefits,
            "parallel"
        )
        
        # Check metrics has expected keys
        self.assertIn("combined_throughput", metrics)
        self.assertIn("combined_latency", metrics)
        self.assertIn("combined_memory", metrics)
        
        # Check values are reasonable
        self.assertGreater(metrics["combined_throughput"], 0)
        self.assertGreater(metrics["combined_latency"], 0)
        self.assertGreater(metrics["combined_memory"], 0)
    
    def test_predict_multi_model_performance(self):
        """Test full multi-model performance prediction."""
        # Predict performance
        prediction = self.predictor.predict_multi_model_performance(
            self.model_configs,
            hardware_platform="cuda",
            execution_strategy="parallel"
        )
        
        # Check prediction has expected keys
        self.assertIn("total_metrics", prediction)
        self.assertIn("individual_predictions", prediction)
        self.assertIn("contention_factors", prediction)
        self.assertIn("sharing_benefits", prediction)
        self.assertIn("execution_schedule", prediction)
        
        # Check total metrics
        total_metrics = prediction["total_metrics"]
        self.assertIn("combined_throughput", total_metrics)
        self.assertIn("combined_latency", total_metrics)
        self.assertIn("combined_memory", total_metrics)
        
        # Check individual predictions
        self.assertEqual(len(prediction["individual_predictions"]), len(self.model_configs))
    
    def test_recommend_execution_strategy(self):
        """Test execution strategy recommendation."""
        # Get recommendation
        recommendation = self.predictor.recommend_execution_strategy(
            self.model_configs,
            hardware_platform="cuda",
            optimization_goal="throughput"
        )
        
        # Check recommendation has expected keys
        self.assertIn("recommended_strategy", recommendation)
        self.assertIn("optimization_goal", recommendation)
        self.assertIn("all_predictions", recommendation)
        
        # Check that a valid strategy was recommended
        self.assertIn(recommendation["recommended_strategy"], ["parallel", "sequential", "batched"])
        
        # Check that all strategies were evaluated
        self.assertEqual(len(recommendation["all_predictions"]), 3)
        
        # Check optimization goal
        self.assertEqual(recommendation["optimization_goal"], "throughput")
    
    def test_different_strategies(self):
        """Test prediction with different execution strategies."""
        # Test all strategies
        strategies = ["parallel", "sequential", "batched"]
        
        for strategy in strategies:
            # Predict performance
            prediction = self.predictor.predict_multi_model_performance(
                self.model_configs,
                hardware_platform="cuda",
                execution_strategy=strategy
            )
            
            # Check prediction is valid
            self.assertIn("total_metrics", prediction)
            self.assertIn("execution_schedule", prediction)
            
            # Check execution strategy is correct
            self.assertEqual(prediction["execution_strategy"], strategy)
            
            # Check schedule strategy matches
            self.assertEqual(prediction["execution_schedule"]["strategy"], strategy)
    
    def test_different_hardware(self):
        """Test prediction with different hardware platforms."""
        # Test multiple hardware platforms
        platforms = ["cpu", "cuda", "openvino", "webgpu"]
        
        for platform in platforms:
            # Predict performance
            prediction = self.predictor.predict_multi_model_performance(
                self.model_configs,
                hardware_platform=platform,
                execution_strategy="parallel"
            )
            
            # Check prediction is valid
            self.assertIn("total_metrics", prediction)
            self.assertIn("contention_factors", prediction)
            
            # Check hardware platform is correct
            self.assertEqual(prediction["hardware_platform"], platform)


if __name__ == "__main__":
    unittest.main()