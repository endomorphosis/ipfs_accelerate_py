#!/usr/bin/env python3
"""
Test Batch Generator Testing Script.

This script demonstrates and validates the Test Batch Generator functionality
of the Active Learning System, which is used to create optimized batches of
test configurations for benchmarking.
"""

import os
import sys
import json
import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from pprint import pprint

# Add parent directory to the Python path to allow importing the module
sys.path.append(str(Path(__file__).parent.parent))

# Import the active learning module directly from the file
import importlib.util
spec = importlib.util.spec_from_file_location(
    "active_learning", 
    str(Path(__file__).parent / "active_learning.py")
)
active_learning_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(active_learning_module)
ActiveLearningSystem = active_learning_module.ActiveLearningSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_batch_generator")

def setup_test_data():
    """Create test data for batch generation."""
    # Create an instance of the active learning system
    active_learning = ActiveLearningSystem()
    
    # Generate test configurations (using the system's built-in function)
    configs = active_learning.recommend_configurations(budget=50)
    
    # Add some metadata for better visualization
    for i, config in enumerate(configs):
        config["id"] = i
        
    return active_learning, configs

def test_basic_batch_generation():
    """Test basic batch generation without special constraints."""
    logger.info("Testing basic batch generation")
    active_learning, configs = setup_test_data()
    
    # Generate a batch with default settings
    batch = active_learning.suggest_test_batch(
        configurations=configs,
        batch_size=10,
        ensure_diversity=True
    )
    
    logger.info(f"Generated batch with {len(batch)} configurations")
    print("\nBasic Batch Generation Results:")
    print(f"Original configurations: {len(configs)}")
    print(f"Batch size: {len(batch)}")
    print(f"Selected configurations (first 3):")
    print(batch[['model_type', 'hardware', 'batch_size', 'expected_information_gain', 'selection_order']].head(3))
    
    # Validate that the batch has the right size
    assert len(batch) <= 10, f"Batch size should be ≤ 10, got {len(batch)}"
    
    # Validate that selection_order column was added
    assert 'selection_order' in batch.columns, "Batch should have selection_order column"
    
    return batch

def test_hardware_constrained_batch():
    """Test batch generation with hardware constraints."""
    logger.info("Testing hardware-constrained batch generation")
    active_learning, configs = setup_test_data()
    
    # Define hardware constraints
    hardware_constraints = {
        'cpu': 2,
        'cuda': 3,
        'openvino': 1,
        'webgpu': 1
    }
    
    # Generate a batch with hardware constraints
    batch = active_learning.suggest_test_batch(
        configurations=configs,
        batch_size=10,
        ensure_diversity=True,
        hardware_constraints=hardware_constraints
    )
    
    logger.info(f"Generated hardware-constrained batch with {len(batch)} configurations")
    print("\nHardware-Constrained Batch Generation Results:")
    print(f"Original configurations: {len(configs)}")
    print(f"Batch size: {len(batch)}")
    
    # Check hardware counts
    hw_counts = batch['hardware'].value_counts().to_dict()
    print(f"Hardware counts: {hw_counts}")
    
    # Validate hardware constraints
    for hw, limit in hardware_constraints.items():
        count = hw_counts.get(hw, 0)
        assert count <= limit, f"Hardware {hw} exceeded constraint: {count} > {limit}"
    
    return batch

def test_hardware_availability():
    """Test batch generation with hardware availability factors."""
    logger.info("Testing hardware availability weighting")
    active_learning, configs = setup_test_data()
    
    # Define hardware availability (probabilities of 0-1)
    hardware_availability = {
        'cpu': 1.0,    # Fully available
        'cuda': 0.8,   # 80% available
        'openvino': 0.5, # 50% available
        'webgpu': 0.2  # 20% available
    }
    
    # Generate a batch with hardware availability weighting
    batch = active_learning.suggest_test_batch(
        configurations=configs,
        batch_size=10,
        ensure_diversity=True,
        hardware_availability=hardware_availability
    )
    
    logger.info(f"Generated availability-weighted batch with {len(batch)} configurations")
    print("\nHardware Availability Batch Generation Results:")
    print(f"Original configurations: {len(configs)}")
    print(f"Batch size: {len(batch)}")
    
    # Check hardware counts
    hw_counts = batch['hardware'].value_counts().to_dict()
    print(f"Hardware counts: {hw_counts}")
    
    # No strict validation here, but we can observe the distribution trends
    
    return batch

def test_diversity_weighting():
    """Test batch generation with different diversity weights."""
    logger.info("Testing diversity weighting impact")
    active_learning, configs = setup_test_data()
    
    results = {}
    
    # Test different diversity weights
    for weight in [0.1, 0.5, 0.9]:
        batch = active_learning.suggest_test_batch(
            configurations=configs,
            batch_size=10,
            ensure_diversity=True,
            diversity_weight=weight
        )
        
        results[weight] = batch
        
        logger.info(f"Generated batch with diversity weight {weight}")
    
    print("\nDiversity Weighting Results:")
    print("Hardware distribution with different diversity weights:")
    
    for weight, batch in results.items():
        hw_counts = batch['hardware'].value_counts().to_dict()
        model_counts = batch['model_type'].value_counts().to_dict()
        print(f"\nDiversity weight {weight}:")
        print(f"Hardware distribution: {hw_counts}")
        print(f"Model type distribution: {model_counts}")
    
    # The higher the diversity weight, the more evenly distributed the configs should be
    
    return results

def test_combined_constraints():
    """Test batch generation with both hardware constraints and availability."""
    logger.info("Testing combined constraints")
    active_learning, configs = setup_test_data()
    
    # Define constraints
    hardware_constraints = {
        'cpu': 3,
        'cuda': 3,
        'openvino': 2,
        'webgpu': 1
    }
    
    hardware_availability = {
        'cpu': 1.0,
        'cuda': 0.7,
        'openvino': 0.5,
        'webgpu': 0.3
    }
    
    # Generate batch with combined constraints
    batch = active_learning.suggest_test_batch(
        configurations=configs,
        batch_size=10,
        ensure_diversity=True,
        hardware_constraints=hardware_constraints,
        hardware_availability=hardware_availability,
        diversity_weight=0.6
    )
    
    logger.info(f"Generated batch with combined constraints: {len(batch)} configurations")
    print("\nCombined Constraints Batch Generation Results:")
    print(f"Original configurations: {len(configs)}")
    print(f"Batch size: {len(batch)}")
    print(f"Hardware counts: {batch['hardware'].value_counts().to_dict()}")
    print(f"Model type counts: {batch['model_type'].value_counts().to_dict()}")
    
    # Validate hardware constraints
    hw_counts = batch['hardware'].value_counts().to_dict()
    for hw, limit in hardware_constraints.items():
        count = hw_counts.get(hw, 0)
        assert count <= limit, f"Hardware {hw} exceeded constraint: {count} > {limit}"
    
    return batch

def test_integration_with_hardware_recommender():
    """
    Test integration between batch generation and hardware recommender.
    
    This is a simulated test since we don't have actual hardware recommender available.
    """
    logger.info("Testing batch generation integration with hardware recommender")
    active_learning, _ = setup_test_data()
    
    # Create mock hardware recommender
    class MockHardwareRecommender:
        def recommend_hardware(self, **kwargs):
            model_type = kwargs.get("model_type", "")
            
            # Simple logic to simulate hardware recommendations
            if model_type == "text_embedding":
                recommended = "cuda"
            elif model_type == "vision":
                recommended = "webgpu"
            elif model_type == "audio":
                recommended = "openvino"
            else:
                recommended = "cpu"
                
            return {
                "recommended_hardware": recommended,
                "throughput_score": 0.8,
                "estimated_improvement": 0.3,
                "alternatives": ["cpu"]
            }
    
    # Get integrated recommendations
    hw_recommender = MockHardwareRecommender()
    integrated_results = active_learning.integrate_with_hardware_recommender(
        hardware_recommender=hw_recommender,
        test_budget=20,
        optimize_for="throughput"
    )
    
    # Create a batch from the integrated recommendations
    batch = active_learning.suggest_test_batch(
        configurations=integrated_results["recommendations"],
        batch_size=10,
        ensure_diversity=True
    )
    
    logger.info(f"Generated integrated batch with {len(batch)} configurations")
    print("\nIntegrated Batch Generation Results:")
    print(f"Hardware recommendations: {len(integrated_results['recommendations'])}")
    print(f"Batch size: {len(batch)}")
    print(f"Hardware counts: {batch['hardware'].value_counts().to_dict()}")
    print(f"Model type counts: {batch['model_type'].value_counts().to_dict()}")
    
    return batch

def run_all_tests():
    """Run all test cases."""
    test_basic_batch_generation()
    test_hardware_constrained_batch()
    test_hardware_availability()
    test_diversity_weighting()
    test_combined_constraints()
    test_integration_with_hardware_recommender()
    
    logger.info("All tests completed successfully!")

def main():
    """Main function to run tests based on command line arguments."""
    parser = argparse.ArgumentParser(description="Test the Test Batch Generator functionality")
    parser.add_argument("--test", choices=["basic", "hardware", "availability", 
                                          "diversity", "combined", "integration", "all"],
                        default="all", help="Test to run")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Batch size for test generation")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.test == "basic":
        test_basic_batch_generation()
    elif args.test == "hardware":
        test_hardware_constrained_batch()
    elif args.test == "availability":
        test_hardware_availability()
    elif args.test == "diversity":
        test_diversity_weighting()
    elif args.test == "combined":
        test_combined_constraints()
    elif args.test == "integration":
        test_integration_with_hardware_recommender()
    elif args.test == "all":
        run_all_tests()

if __name__ == "__main__":
    main()