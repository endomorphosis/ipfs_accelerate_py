#!/usr/bin/env python3
"""
Test script for the integration between Active Learning and Hardware Recommender systems.

This script validates the integration between the ActiveLearningSystem and HardwareRecommender
components of the Predictive Performance System, ensuring they work together correctly.

Usage:
    python test_integration.py
"""

import os
import sys
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_integration")

# Imports
try:
    from active_learning import ActiveLearningSystem
    from hardware_recommender import HardwareRecommender
    from predict import PerformancePredictor
except ImportError as e:
    logger.error(f"Failed to import required components: {e}")
    logger.info("Make sure you're running this script from the predictive_performance directory")
    sys.exit(1)

def test_active_learning_initialization():
    """Test the initialization of the ActiveLearningSystem."""
    logger.info("Testing ActiveLearningSystem initialization...")
    
    try:
        # Initialize with synthetic data
        active_learner = ActiveLearningSystem()
        logger.info("✅ ActiveLearningSystem initialized successfully")
        return active_learner
    except Exception as e:
        logger.error(f"❌ Failed to initialize ActiveLearningSystem: {e}")
        return None

def test_hardware_recommender_initialization():
    """Test the initialization of the HardwareRecommender."""
    logger.info("Testing HardwareRecommender initialization...")
    
    try:
        # Initialize predictor first
        predictor = PerformancePredictor()
        
        # Initialize hardware recommender
        hw_recommender = HardwareRecommender(
            predictor=predictor,
            available_hardware=["cpu", "cuda", "rocm", "mps", "openvino", "qnn", "webnn", "webgpu"],
            confidence_threshold=0.7
        )
        logger.info("✅ HardwareRecommender initialized successfully")
        return hw_recommender
    except Exception as e:
        logger.error(f"❌ Failed to initialize HardwareRecommender: {e}")
        return None

def test_simple_recommendations(active_learner):
    """Test getting basic recommendations from the ActiveLearningSystem."""
    logger.info("Testing basic recommendations...")
    
    try:
        # Get recommendations
        recommendations = active_learner.recommend_configurations(budget=5)
        
        if not recommendations or len(recommendations) == 0:
            logger.error("❌ No recommendations returned")
            return False
        
        logger.info(f"✅ Got {len(recommendations)} recommendations")
        
        # Print the first recommendation
        if len(recommendations) > 0:
            first_rec = recommendations[0]
            logger.info(f"First recommendation: {first_rec['model_name']} on {first_rec['hardware']} with batch size {first_rec['batch_size']}")
            logger.info(f"Expected information gain: {first_rec.get('expected_information_gain', 'N/A')}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Failed to get recommendations: {e}")
        return False

def test_integration(active_learner, hw_recommender):
    """Test the integration between ActiveLearningSystem and HardwareRecommender."""
    logger.info("Testing integration between ActiveLearningSystem and HardwareRecommender...")
    
    try:
        # Get integrated recommendations
        integrated_results = active_learner.integrate_with_hardware_recommender(
            hardware_recommender=hw_recommender,
            test_budget=5,
            optimize_for="throughput"
        )
        
        if not integrated_results or "recommendations" not in integrated_results:
            logger.error("❌ No integrated recommendations returned")
            return False
        
        recommendations = integrated_results["recommendations"]
        logger.info(f"✅ Got {len(recommendations)} integrated recommendations")
        
        # Check for required fields
        expected_fields = ["model_name", "hardware", "batch_size", "recommended_hardware", "combined_score"]
        for field in expected_fields:
            if field not in recommendations[0]:
                logger.error(f"❌ Missing required field in recommendations: {field}")
                return False
        
        # Print some information about the results
        logger.info(f"Total candidates considered: {integrated_results.get('total_candidates', 'N/A')}")
        logger.info(f"Enhanced candidates: {integrated_results.get('enhanced_candidates', 'N/A')}")
        logger.info(f"Final recommendations: {integrated_results.get('final_recommendations', 'N/A')}")
        
        # Print details of the first recommendation
        if len(recommendations) > 0:
            first_rec = recommendations[0]
            logger.info("First integrated recommendation:")
            logger.info(f"  - Model: {first_rec['model_name']}")
            logger.info(f"  - Current Hardware: {first_rec['hardware']}")
            logger.info(f"  - Recommended Hardware: {first_rec.get('recommended_hardware', 'N/A')}")
            logger.info(f"  - Hardware Match: {first_rec.get('hardware_match', 'N/A')}")
            logger.info(f"  - Combined Score: {first_rec.get('combined_score', 'N/A')}")
        
        # Save the results to a file for inspection
        output_dir = Path("test_output")
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / "integrated_test_results.json", "w") as f:
            json.dump(integrated_results, f, indent=2, default=str)
        
        logger.info(f"Saved test results to {output_dir / 'integrated_test_results.json'}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Failed to test integration: {e}")
        return False

def run_all_tests():
    """Run all tests."""
    logger.info("Starting tests...")
    
    # Track test results
    results = {
        "active_learning_init": False,
        "hardware_recommender_init": False,
        "simple_recommendations": False,
        "integration": False
    }
    
    # Test ActiveLearningSystem initialization
    active_learner = test_active_learning_initialization()
    results["active_learning_init"] = active_learner is not None
    
    # Test HardwareRecommender initialization
    hw_recommender = test_hardware_recommender_initialization()
    results["hardware_recommender_init"] = hw_recommender is not None
    
    # Skip further tests if initialization failed
    if not active_learner or not hw_recommender:
        logger.error("❌ Component initialization failed, skipping further tests")
        print_summary(results)
        return
    
    # Test simple recommendations
    results["simple_recommendations"] = test_simple_recommendations(active_learner)
    
    # Test integration
    results["integration"] = test_integration(active_learner, hw_recommender)
    
    # Print summary
    print_summary(results)

def print_summary(results):
    """Print a summary of the test results."""
    logger.info("\n=== Test Summary ===")
    for test, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test}: {status}")
    
    # Overall status
    all_passed = all(results.values())
    logger.info(f"\nOverall Status: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")

if __name__ == "__main__":
    run_all_tests()