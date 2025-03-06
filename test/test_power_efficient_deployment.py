#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for Power-Efficient Model Deployment Pipeline

This script implements tests for the power-efficient model deployment pipeline.
It verifies that the pipeline correctly prepares, loads, and runs inference
on models with appropriate power optimizations.

Date: April 2025
"""

import os
import sys
import json
import time
import unittest
import tempfile
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent))

# Import power efficient deployment components
try:
    from power_efficient_deployment import (
        PowerEfficientDeployment,
        PowerProfile,
        DeploymentTarget
    )
    HAS_POWER_DEPLOYMENT = True
except ImportError as e:
    logger.error(f"Failed to import power_efficient_deployment module: {e}")
    HAS_POWER_DEPLOYMENT = False

# Try importing thermal monitoring components
try:
    from mobile_thermal_monitoring import MobileThermalMonitor
    HAS_THERMAL_MONITORING = True
except ImportError:
    logger.warning("Warning: mobile_thermal_monitoring could not be imported. Thermal tests will be skipped.")
    HAS_THERMAL_MONITORING = False

# Try importing Qualcomm quantization support
try:
    from qualcomm_quantization_support import QualcommQuantization
    HAS_QUALCOMM_QUANTIZATION = True
except ImportError:
    logger.warning("Warning: qualcomm_quantization_support could not be imported. Quantization tests will be skipped.")
    HAS_QUALCOMM_QUANTIZATION = False

class TestPowerEfficientDeployment(unittest.TestCase):
    """Test cases for power-efficient model deployment."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        if not HAS_POWER_DEPLOYMENT:
            raise unittest.SkipTest("power_efficient_deployment module not available")
        
        # Create temporary directory for test models
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.test_model_dir = cls.temp_dir.name
        
        # Create a mock model file
        cls.test_model_path = os.path.join(cls.test_model_dir, "test_model.onnx")
        with open(cls.test_model_path, "w") as f:
            f.write("Mock ONNX model file for testing")
        
        # Create a database file
        cls.db_path = os.path.join(cls.test_model_dir, "test_db.duckdb")
        
        # Set global mock mode for testing
        os.environ["QUALCOMM_MOCK"] = "1"
        
        logger.info("Test environment set up")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        # Clean up temporary directory
        cls.temp_dir.cleanup()
        
        # Clear environment variables
        if "QUALCOMM_MOCK" in os.environ:
            del os.environ["QUALCOMM_MOCK"]
        
        logger.info("Test environment cleaned up")
    
    def setUp(self):
        """Set up each test."""
        # Create deployment instance with different profiles for testing
        self.deployments = {
            "balanced": PowerEfficientDeployment(
                db_path=self.db_path,
                power_profile=PowerProfile.BALANCED,
                deployment_target=DeploymentTarget.ANDROID
            ),
            "performance": PowerEfficientDeployment(
                db_path=self.db_path,
                power_profile=PowerProfile.MAXIMUM_PERFORMANCE,
                deployment_target=DeploymentTarget.ANDROID
            ),
            "power_saver": PowerEfficientDeployment(
                db_path=self.db_path,
                power_profile=PowerProfile.POWER_SAVER,
                deployment_target=DeploymentTarget.ANDROID
            )
        }
    
    def tearDown(self):
        """Clean up after each test."""
        for deployment in self.deployments.values():
            deployment.cleanup()
    
    def test_initialization(self):
        """Test initialization of different deployment profiles."""
        # Check that deployments were created successfully
        for profile, deployment in self.deployments.items():
            self.assertIsNotNone(deployment)
            self.assertEqual(deployment.db_path, self.db_path)
            
            # Check configuration differences
            if profile == "balanced":
                self.assertEqual(deployment.power_profile, PowerProfile.BALANCED)
                self.assertFalse(deployment.config["thermal_management"]["proactive_throttling"])
            elif profile == "performance":
                self.assertEqual(deployment.power_profile, PowerProfile.MAXIMUM_PERFORMANCE)
                self.assertFalse(deployment.config["thermal_management"]["proactive_throttling"])
                self.assertFalse(deployment.config["power_management"]["sleep_between_inferences"])
            elif profile == "power_saver":
                self.assertEqual(deployment.power_profile, PowerProfile.POWER_SAVER)
                self.assertTrue(deployment.config["thermal_management"]["proactive_throttling"])
                self.assertTrue(deployment.config["power_management"]["sleep_between_inferences"])
    
    def test_model_type_inference(self):
        """Test model type inference."""
        deployment = self.deployments["balanced"]
        
        # Create test models
        vision_model = os.path.join(self.test_model_dir, "vision_resnet.onnx")
        audio_model = os.path.join(self.test_model_dir, "whisper_base.onnx")
        llm_model = os.path.join(self.test_model_dir, "llama_model.onnx")
        text_model = os.path.join(self.test_model_dir, "bert_model.onnx")
        
        for model_path in [vision_model, audio_model, llm_model, text_model]:
            with open(model_path, "w") as f:
                f.write("Mock model file for testing")
        
        # Test inference
        self.assertEqual(deployment._infer_model_type(vision_model), "vision")
        self.assertEqual(deployment._infer_model_type(audio_model), "audio")
        self.assertEqual(deployment._infer_model_type(llm_model), "llm")
        self.assertEqual(deployment._infer_model_type(text_model), "text")
    
    def test_config_update(self):
        """Test configuration update."""
        deployment = self.deployments["balanced"]
        
        # Original config
        original_batch_size = deployment.config["inference_optimization"]["optimal_batch_size"]
        original_method = deployment.config["quantization"]["preferred_method"]
        
        # Update config
        new_config = {
            "inference_optimization": {
                "optimal_batch_size": 16
            },
            "quantization": {
                "preferred_method": "int4"
            }
        }
        
        updated_config = deployment.update_config(new_config)
        
        # Check that config was updated
        self.assertEqual(updated_config["inference_optimization"]["optimal_batch_size"], 16)
        self.assertEqual(updated_config["quantization"]["preferred_method"], "int4")
        
        # Check that profile was changed to CUSTOM
        self.assertEqual(deployment.power_profile, PowerProfile.CUSTOM)
    
    def test_prepare_model(self):
        """Test model preparation."""
        deployment = self.deployments["balanced"]
        
        # Prepare model
        result = deployment.prepare_model_for_deployment(
            model_path=self.test_model_path,
            model_type="text"
        )
        
        # Check preparation result
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["model_type"], "text")
        self.assertTrue(os.path.exists(result["output_model_path"]))
        self.assertIn("optimizations_applied", result)
    
    def test_prepare_model_with_quantization(self):
        """Test model preparation with quantization."""
        # Skip if Qualcomm quantization is not available
        if not HAS_QUALCOMM_QUANTIZATION:
            self.skipTest("Qualcomm quantization not available")
        
        deployment = self.deployments["power_saver"]
        
        # Prepare model with specific quantization method
        result = deployment.prepare_model_for_deployment(
            model_path=self.test_model_path,
            model_type="text",
            quantization_method="int8"
        )
        
        # Check preparation result
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["quantization_method"], "int8")
        self.assertTrue("quantization_int8" in result["optimizations_applied"])
        self.assertTrue("power_efficiency_metrics" in result)
    
    def test_load_model(self):
        """Test model loading."""
        deployment = self.deployments["balanced"]
        
        # Prepare model
        prep_result = deployment.prepare_model_for_deployment(
            model_path=self.test_model_path,
            model_type="text"
        )
        
        # Load model
        load_result = deployment.load_model(
            model_path=prep_result["output_model_path"]
        )
        
        # Check loading result
        self.assertEqual(load_result["status"], "loaded")
        self.assertTrue("model" in load_result)
        self.assertTrue("loading_time_seconds" in load_result)
        
        # Check that model is in active models
        self.assertTrue(prep_result["output_model_path"] in deployment.active_models)
    
    def test_run_inference(self):
        """Test inference execution."""
        deployment = self.deployments["balanced"]
        
        # Prepare and load model
        prep_result = deployment.prepare_model_for_deployment(
            model_path=self.test_model_path,
            model_type="text"
        )
        
        load_result = deployment.load_model(
            model_path=prep_result["output_model_path"]
        )
        
        # Run inference
        inference_result = deployment.run_inference(
            model_path=prep_result["output_model_path"],
            inputs="Sample text for inference"
        )
        
        # Check inference result
        self.assertEqual(inference_result["status"], "success")
        self.assertTrue("outputs" in inference_result)
        self.assertTrue("inference_time_seconds" in inference_result)
        
        # Check that model stats were updated
        model_stats = deployment.model_stats[prep_result["output_model_path"]]
        self.assertEqual(model_stats["inference_count"], 1)
        self.assertTrue(model_stats["total_inference_time_seconds"] > 0)
    
    def test_run_multiple_inferences(self):
        """Test multiple inference executions."""
        deployment = self.deployments["balanced"]
        
        # Prepare and load model
        prep_result = deployment.prepare_model_for_deployment(
            model_path=self.test_model_path,
            model_type="text"
        )
        
        load_result = deployment.load_model(
            model_path=prep_result["output_model_path"]
        )
        
        # Run multiple inferences
        num_inferences = 5
        for i in range(num_inferences):
            inference_result = deployment.run_inference(
                model_path=prep_result["output_model_path"],
                inputs=f"Sample text for inference {i}"
            )
            self.assertEqual(inference_result["status"], "success")
        
        # Check that model stats were updated
        model_stats = deployment.model_stats[prep_result["output_model_path"]]
        self.assertEqual(model_stats["inference_count"], num_inferences)
        self.assertTrue(model_stats["total_inference_time_seconds"] > 0)
    
    def test_batch_inference(self):
        """Test batch inference."""
        deployment = self.deployments["balanced"]
        
        # Prepare and load model
        prep_result = deployment.prepare_model_for_deployment(
            model_path=self.test_model_path,
            model_type="text"
        )
        
        load_result = deployment.load_model(
            model_path=prep_result["output_model_path"]
        )
        
        # Run batch inference
        inference_result = deployment.run_inference(
            model_path=prep_result["output_model_path"],
            inputs="Sample text for batch inference",
            batch_size=4
        )
        
        # Check inference result
        self.assertEqual(inference_result["status"], "success")
        self.assertTrue("outputs" in inference_result)
        self.assertTrue("inference_time_seconds" in inference_result)
    
    def test_unload_model(self):
        """Test model unloading."""
        deployment = self.deployments["balanced"]
        
        # Prepare and load model
        prep_result = deployment.prepare_model_for_deployment(
            model_path=self.test_model_path,
            model_type="text"
        )
        
        load_result = deployment.load_model(
            model_path=prep_result["output_model_path"]
        )
        
        # Check that model is in active models
        self.assertTrue(prep_result["output_model_path"] in deployment.active_models)
        
        # Unload model
        unload_result = deployment.unload_model(
            model_path=prep_result["output_model_path"]
        )
        
        # Check unload result
        self.assertTrue(unload_result)
        self.assertFalse(prep_result["output_model_path"] in deployment.active_models)
        
        # Check that model stats were updated
        model_stats = deployment.model_stats[prep_result["output_model_path"]]
        self.assertEqual(model_stats["status"], "unloaded")
        self.assertTrue("unloaded_at" in model_stats)
    
    def test_get_deployment_status(self):
        """Test getting deployment status."""
        deployment = self.deployments["balanced"]
        
        # Get initial status
        initial_status = deployment.get_deployment_status()
        self.assertEqual(initial_status["deployment_target"], "ANDROID")
        self.assertEqual(initial_status["power_profile"], "BALANCED")
        self.assertEqual(initial_status["active_models_count"], 0)
        
        # Prepare and load model
        prep_result = deployment.prepare_model_for_deployment(
            model_path=self.test_model_path,
            model_type="text"
        )
        
        load_result = deployment.load_model(
            model_path=prep_result["output_model_path"]
        )
        
        # Get updated status
        updated_status = deployment.get_deployment_status()
        self.assertEqual(updated_status["active_models_count"], 1)
        self.assertEqual(updated_status["deployed_models_count"], 1)
        
        # Get status for specific model
        model_status = deployment.get_deployment_status(
            model_path=prep_result["output_model_path"]
        )
        
        self.assertTrue(model_status["active"])
        self.assertEqual(model_status["deployment_info"]["model_type"], "text")
        self.assertEqual(model_status["deployment_info"]["status"], "ready")
    
    def test_get_power_efficiency_report(self):
        """Test generating power efficiency report."""
        deployment = self.deployments["balanced"]
        
        # Prepare and load model
        prep_result = deployment.prepare_model_for_deployment(
            model_path=self.test_model_path,
            model_type="text"
        )
        
        load_result = deployment.load_model(
            model_path=prep_result["output_model_path"]
        )
        
        # Run inference
        inference_result = deployment.run_inference(
            model_path=prep_result["output_model_path"],
            inputs="Sample text for inference"
        )
        
        # Get report in different formats
        json_report = deployment.get_power_efficiency_report(
            report_format="json"
        )
        
        markdown_report = deployment.get_power_efficiency_report(
            report_format="markdown"
        )
        
        html_report = deployment.get_power_efficiency_report(
            report_format="html"
        )
        
        # Check reports
        self.assertTrue(isinstance(json_report, dict))
        self.assertTrue("models" in json_report)
        self.assertTrue(prep_result["output_model_path"] in json_report["models"])
        
        self.assertTrue(isinstance(markdown_report, str))
        self.assertTrue("# Power Efficiency Report" in markdown_report)
        
        self.assertTrue(isinstance(html_report, str))
        self.assertTrue("<!DOCTYPE html>" in html_report)
    
    @unittest.skipIf(not HAS_THERMAL_MONITORING, "Thermal monitoring not available")
    def test_thermal_monitoring_integration(self):
        """Test integration with thermal monitoring."""
        deployment = self.deployments["thermal_aware"] = PowerEfficientDeployment(
            db_path=self.db_path,
            power_profile=PowerProfile.THERMAL_AWARE,
            deployment_target=DeploymentTarget.ANDROID
        )
        
        # Check that thermal monitor was initialized
        self.assertIsNotNone(deployment.thermal_monitor)
        
        # Prepare and load model
        prep_result = deployment.prepare_model_for_deployment(
            model_path=self.test_model_path,
            model_type="text"
        )
        
        load_result = deployment.load_model(
            model_path=prep_result["output_model_path"]
        )
        
        # Check thermal status
        thermal_status = deployment._check_thermal_status()
        self.assertTrue("thermal_status" in thermal_status)
        self.assertTrue("thermal_throttling" in thermal_status)
        self.assertTrue("temperatures" in thermal_status)

    def test_performance_profile_behavior(self):
        """Test behavior of models under different power profiles."""
        # Use different profiles
        performance_deployment = self.deployments["performance"]
        power_saver_deployment = self.deployments["power_saver"]
        
        # Prepare models with each profile
        perf_result = performance_deployment.prepare_model_for_deployment(
            model_path=self.test_model_path,
            model_type="text"
        )
        
        saver_result = power_saver_deployment.prepare_model_for_deployment(
            model_path=self.test_model_path,
            model_type="text"
        )
        
        # Load models
        perf_load = performance_deployment.load_model(
            model_path=perf_result["output_model_path"]
        )
        
        saver_load = power_saver_deployment.load_model(
            model_path=saver_result["output_model_path"]
        )
        
        # Run inferences
        perf_inference = performance_deployment.run_inference(
            model_path=perf_result["output_model_path"],
            inputs="Sample text for inference"
        )
        
        saver_inference = power_saver_deployment.run_inference(
            model_path=saver_result["output_model_path"],
            inputs="Sample text for inference"
        )
        
        # Compare performance metrics (in a real test, these would differ significantly)
        # Since this is a mock environment, we mainly check that both profiles work
        self.assertEqual(perf_inference["status"], "success")
        self.assertEqual(saver_inference["status"], "success")
        
        # Check different optimization methods were applied
        self.assertNotEqual(
            perf_result["quantization_method"],
            saver_result["quantization_method"],
            "Different quantization methods should be applied based on power profile"
        )
    
    def test_full_deployment_lifecycle(self):
        """Test the full model deployment lifecycle."""
        deployment = self.deployments["balanced"]
        
        # 1. Prepare model
        logger.info("1. Preparing model...")
        prep_result = deployment.prepare_model_for_deployment(
            model_path=self.test_model_path,
            model_type="text"
        )
        
        # 2. Load model
        logger.info("2. Loading model...")
        load_result = deployment.load_model(
            model_path=prep_result["output_model_path"]
        )
        
        # 3. Run multiple inferences
        logger.info("3. Running inferences...")
        num_inferences = 3
        for i in range(num_inferences):
            inference_result = deployment.run_inference(
                model_path=prep_result["output_model_path"],
                inputs=f"Sample text for inference {i}"
            )
            self.assertEqual(inference_result["status"], "success")
        
        # 4. Check status
        logger.info("4. Checking status...")
        status = deployment.get_deployment_status(
            model_path=prep_result["output_model_path"]
        )
        self.assertTrue(status["active"])
        self.assertEqual(status["stats"]["inference_count"], num_inferences)
        
        # 5. Generate report
        logger.info("5. Generating report...")
        report = deployment.get_power_efficiency_report(
            model_path=prep_result["output_model_path"],
            report_format="json"
        )
        self.assertTrue(prep_result["output_model_path"] in report["models"])
        
        # 6. Unload model
        logger.info("6. Unloading model...")
        unload_result = deployment.unload_model(
            model_path=prep_result["output_model_path"]
        )
        self.assertTrue(unload_result)
        
        # 7. Verify unloaded
        logger.info("7. Verifying unloaded...")
        final_status = deployment.get_deployment_status(
            model_path=prep_result["output_model_path"]
        )
        self.assertFalse(final_status["active"])
        
        # 8. Clean up
        logger.info("8. Cleaning up...")
        deployment.cleanup()
        
        logger.info("Full deployment lifecycle test completed successfully")


def run_tests():
    """Run all tests."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPowerEfficientDeployment)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return result.wasSuccessful()


def main():
    """Command-line interface for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Power-Efficient Model Deployment")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run tests
    success = run_tests()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())