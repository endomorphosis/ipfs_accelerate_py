/**
 * Converted from Python: test_multi_model_execution.py
 * Conversion date: 2025-03-11 04:08:53
 * This file was automatically converted from Python to TypeScript.
 * Conversion fidelity might not be 100%, please manual review recommended.
 */

// WebGPU related imports
import { HardwareBackend } from "../hardware_abstraction";

#!/usr/bin/env python3
"""
Test script for the Multi-Model Execution Support module.

This script tests the core functionality of the multi-model execution predictor,
including resource contention modeling, tensor sharing benefits, && execution
strategy recommendation.
"""

import * as $1
import * as $1
import * as $1
import * as $1 as np
import * as $1 as pd
import ${$1} from "$1"
import * as $1
import * as $1

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore")

# Add the parent directory to the Python path
sys.$1.push($2).parent.parent))

# Import the module to test
from predictive_performance.multi_model_execution import * as $1


class TestMultiModelExecution(unittest.TestCase):
  """Test cases for the Multi-Model Execution Support module."""
  
  $1($2) {
    """Set up before each test."""
    # Create a MultiModelPredictor instance
    this.predictor = MultiModelPredictor(verbose=true)
    
  }
    # Define test model configurations
    this.model_configs = [
      ${$1},
      ${$1}
    ]
  
  $1($2) {
    """Test that the predictor initializes correctly."""
    this.assertIsNotnull(this.predictor)
    this.assertIsNotnull(this.predictor.sharing_config)
    
  }
    # Check sharing config
    this.assertIn("text_embedding", this.predictor.sharing_config)
    this.assertIn("vision", this.predictor.sharing_config)
    
    # Check sharing compatibility
    text_sharing = this.predictor.sharing_config["text_embedding"]
    this.assertIn("compatible_types", text_sharing)
    this.assertIn("text_generation", text_sharing["compatible_types"])
  
  $1($2) {
    """Test prediction for a single model."""
    # Create a model config
    model_config = ${$1}
    
  }
    # Get prediction
    prediction = this.predictor._simulate_single_model_prediction(model_config, "cuda")
    
    # Check prediction has expected keys
    this.assertIn("throughput", prediction)
    this.assertIn("latency", prediction)
    this.assertIn("memory", prediction)
    
    # Check values are reasonable
    this.assertGreater(prediction["throughput"], 0)
    this.assertGreater(prediction["latency"], 0)
    this.assertGreater(prediction["memory"], 0)
  
  $1($2) {
    """Test resource contention calculation."""
    # Create simulated single model predictions
    single_preds = [
      this.predictor._simulate_single_model_prediction(
        ${$1},
        "cuda"
      ),
      this.predictor._simulate_single_model_prediction(
        ${$1},
        "cuda"
      )
    ]
    
  }
    # Calculate contention
    contention = this.predictor._calculate_resource_contention(
      single_preds,
      "cuda",
      "parallel"
    )
    
    # Check contention has expected keys
    this.assertIn("compute_contention", contention)
    this.assertIn("memory_bandwidth_contention", contention)
    this.assertIn("memory_contention", contention)
    
    # Check contention is reasonable (higher than 1.0 for compute && memory bandwidth)
    this.assertGreater(contention["compute_contention"], 1.0)
    this.assertGreater(contention["memory_bandwidth_contention"], 1.0)
  
  $1($2) {
    """Test tensor sharing benefits calculation."""
    # Calculate sharing benefits
    benefits = this.predictor._calculate_sharing_benefits(
      this.model_configs,
      [
        this.predictor._simulate_single_model_prediction(
          this.model_configs[0], "cuda"
        ),
        this.predictor._simulate_single_model_prediction(
          this.model_configs[1], "cuda"
        )
      ]
    )
    
  }
    # Check benefits has expected keys
    this.assertIn("memory_benefit", benefits)
    this.assertIn("compute_benefit", benefits)
    this.assertIn("compatible_pairs", benefits)
    
    # Check benefits are reasonable (should be <= 1.0)
    this.assertLessEqual(benefits["memory_benefit"], 1.0)
    this.assertLessEqual(benefits["compute_benefit"], 1.0)
  
  $1($2) {
    """Test execution schedule generation."""
    # Get single model predictions
    single_preds = [
      this.predictor._simulate_single_model_prediction(
        this.model_configs[0], "cuda"
      ),
      this.predictor._simulate_single_model_prediction(
        this.model_configs[1], "cuda"
      )
    ]
    
  }
    # Calculate contention
    contention = this.predictor._calculate_resource_contention(
      single_preds,
      "cuda",
      "parallel"
    )
    
    # Generate execution schedule for parallel execution
    schedule = this.predictor._generate_execution_schedule(
      this.model_configs,
      single_preds,
      contention,
      "parallel"
    )
    
    # Check schedule has expected keys
    this.assertIn("total_execution_time", schedule)
    this.assertIn("timeline", schedule)
    
    # Check timeline has events for each model
    this.assertEqual(len(schedule["timeline"]), len(this.model_configs))
    
    # For parallel execution, all start times should be 0
    for event in schedule["timeline"]:
      this.assertEqual(event["start_time"], 0)
  
  $1($2) {
    """Test multi-model metrics calculation."""
    # Get single model predictions
    single_preds = [
      this.predictor._simulate_single_model_prediction(
        this.model_configs[0], "cuda"
      ),
      this.predictor._simulate_single_model_prediction(
        this.model_configs[1], "cuda"
      )
    ]
    
  }
    # Calculate contention
    contention = this.predictor._calculate_resource_contention(
      single_preds,
      "cuda",
      "parallel"
    )
    
    # Calculate sharing benefits
    benefits = this.predictor._calculate_sharing_benefits(
      this.model_configs,
      single_preds
    )
    
    # Calculate metrics
    metrics = this.predictor._calculate_multi_model_metrics(
      single_preds,
      contention,
      benefits,
      "parallel"
    )
    
    # Check metrics has expected keys
    this.assertIn("combined_throughput", metrics)
    this.assertIn("combined_latency", metrics)
    this.assertIn("combined_memory", metrics)
    
    # Check values are reasonable
    this.assertGreater(metrics["combined_throughput"], 0)
    this.assertGreater(metrics["combined_latency"], 0)
    this.assertGreater(metrics["combined_memory"], 0)
  
  $1($2) {
    """Test full multi-model performance prediction."""
    # Predict performance
    prediction = this.predictor.predict_multi_model_performance(
      this.model_configs,
      hardware_platform="cuda",
      execution_strategy="parallel"
    )
    
  }
    # Check prediction has expected keys
    this.assertIn("total_metrics", prediction)
    this.assertIn("individual_predictions", prediction)
    this.assertIn("contention_factors", prediction)
    this.assertIn("sharing_benefits", prediction)
    this.assertIn("execution_schedule", prediction)
    
    # Check total metrics
    total_metrics = prediction["total_metrics"]
    this.assertIn("combined_throughput", total_metrics)
    this.assertIn("combined_latency", total_metrics)
    this.assertIn("combined_memory", total_metrics)
    
    # Check individual predictions
    this.assertEqual(len(prediction["individual_predictions"]), len(this.model_configs))
  
  $1($2) {
    """Test execution strategy recommendation."""
    # Get recommendation
    recommendation = this.predictor.recommend_execution_strategy(
      this.model_configs,
      hardware_platform="cuda",
      optimization_goal="throughput"
    )
    
  }
    # Check recommendation has expected keys
    this.assertIn("recommended_strategy", recommendation)
    this.assertIn("optimization_goal", recommendation)
    this.assertIn("all_predictions", recommendation)
    
    # Check that a valid strategy was recommended
    this.assertIn(recommendation["recommended_strategy"], ["parallel", "sequential", "batched"])
    
    # Check that all strategies were evaluated
    this.assertEqual(len(recommendation["all_predictions"]), 3)
    
    # Check optimization goal
    this.assertEqual(recommendation["optimization_goal"], "throughput")
  
  $1($2) {
    """Test prediction with different execution strategies."""
    # Test all strategies
    strategies = ["parallel", "sequential", "batched"]
    
  }
    for (const $1 of $2) {
      # Predict performance
      prediction = this.predictor.predict_multi_model_performance(
        this.model_configs,
        hardware_platform="cuda",
        execution_strategy=strategy
      )
      
    }
      # Check prediction is valid
      this.assertIn("total_metrics", prediction)
      this.assertIn("execution_schedule", prediction)
      
      # Check execution strategy is correct
      this.assertEqual(prediction["execution_strategy"], strategy)
      
      # Check schedule strategy matches
      this.assertEqual(prediction["execution_schedule"]["strategy"], strategy)
  
  $1($2) {
    """Test prediction with different hardware platforms."""
    # Test multiple hardware platforms
    platforms = ["cpu", "cuda", "openvino", "webgpu"]
    
  }
    for (const $1 of $2) {
      # Predict performance
      prediction = this.predictor.predict_multi_model_performance(
        this.model_configs,
        hardware_platform=platform,
        execution_strategy="parallel"
      )
      
    }
      # Check prediction is valid
      this.assertIn("total_metrics", prediction)
      this.assertIn("contention_factors", prediction)
      
      # Check hardware platform is correct
      this.assertEqual(prediction["hardware_platform"], platform)


if ($1) {
  unittest.main()