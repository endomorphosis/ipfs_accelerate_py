#!/usr/bin/env python3
"""
Test script for ResourcePool integration with WebNN/WebGPU Recovery System

This script tests the integration of the ResourcePool with the WebNN/WebGPU
Resource Pool Bridge Recovery system.

Usage:
    python test_resource_pool_with_recovery.py
"""

import os
import sys
import time
import json
import logging
from unittest.mock import MagicMock, patch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import ResourcePool
from resource_pool import ResourcePool, global_resource_pool, WEBNN_WEBGPU_RESOURCE_POOL_AVAILABLE

def test_resource_pool_initialization():
    """Test ResourcePool initialization with WebNN/WebGPU support."""
    # Create a new resource pool
    pool = ResourcePool()
    
    # Log initialization status
    logger.info(f"ResourcePool initialized with WebNN/WebGPU support: {pool.web_resource_pool_initialized}")
    
    # Display general stats
    stats = pool.get_stats()
    logger.info(f"ResourcePool stats: hits={stats['hits']}, misses={stats['misses']}")
    logger.info(f"Web resource pool available: {stats['web_resource_pool']['available']}")
    logger.info(f"Web resource pool initialized: {stats['web_resource_pool']['initialized']}")
    
    # Log web pool metrics if available
    if stats['web_resource_pool']['initialized']:
        if 'metrics' in stats['web_resource_pool']:
            logger.info(f"Web resource pool metrics: {json.dumps(stats['web_resource_pool']['metrics'], indent=2)}")
        else:
            logger.info("Web resource pool metrics not available")
    
    return pool

def test_get_model_with_web_preference():
    """Test getting a model with WebNN/WebGPU preference."""
    pool = global_resource_pool
    
    # Mock constructor for testing
    def mock_constructor():
        mock_model = MagicMock()
        mock_model.name = "test_model"
        return mock_model
    
    # Try to get a model with WebGPU preference
    hardware_preferences = {"priority_list": ["webgpu", "cpu"]}
    
    logger.info("Getting model with WebGPU preference")
    model = pool.get_model("text", "bert-test", constructor=mock_constructor, hardware_preferences=hardware_preferences)
    
    if model:
        logger.info(f"Model loaded successfully: {model}")
        logger.info(f"Model type: {type(model)}")
        
        # Test inference if possible
        try:
            result = model({"input_ids": [1, 2, 3]})
            logger.info(f"Inference result: {result}")
        except Exception as e:
            logger.error(f"Error running inference: {e}")
    else:
        logger.error("Failed to load model")
    
    return model

def test_concurrent_execution():
    """Test concurrent execution with WebNN/WebGPU support."""
    pool = global_resource_pool
    
    # Get multiple models
    models = []
    for model_type, model_name in [("text", "bert-test"), ("vision", "vit-test")]:
        def mock_constructor():
            mock_model = MagicMock()
            mock_model.name = f"{model_type}_{model_name}"
            mock_model.model_type = model_type
            mock_model.model_name = model_name
            mock_model.model_id = f"{model_type}:{model_name}"
            return mock_model
        
        model = pool.get_model(model_type, model_name, constructor=mock_constructor, 
                              hardware_preferences={"priority_list": ["webgpu", "cpu"]})
        if model:
            models.append(model)
    
    # Execute concurrently
    if len(models) >= 2:
        logger.info(f"Running concurrent execution with {len(models)} models")
        
        # Prepare inputs
        models_and_inputs = [(model, {"input_ids": [1, 2, 3]}) for model in models]
        
        # Execute concurrently
        results = pool.execute_concurrent(models_and_inputs)
        
        logger.info(f"Concurrent execution results: {results}")
    else:
        logger.warning("Not enough models loaded for concurrent execution")

def test_resource_pool_cleanup():
    """Test ResourcePool cleanup with WebNN/WebGPU support."""
    pool = global_resource_pool
    
    # Get stats before cleanup
    before_stats = pool.get_stats()
    logger.info(f"Before cleanup: {before_stats['cached_models']} cached models")
    
    # Perform cleanup
    pool.clear()
    
    # Get stats after cleanup
    after_stats = pool.get_stats()
    logger.info(f"After cleanup: {after_stats['cached_models']} cached models")
    logger.info(f"Web resource pool initialized: {after_stats['web_resource_pool']['initialized']}")

def main():
    """Main test function."""
    logger.info("Starting WebNN/WebGPU Resource Pool integration test")
    
    # Check if WebNN/WebGPU Resource Pool is available
    if not WEBNN_WEBGPU_RESOURCE_POOL_AVAILABLE:
        logger.warning("WebNN/WebGPU Resource Pool is not available")
        # Continue with tests to verify fallbacks work
    
    # Test ResourcePool initialization
    pool = test_resource_pool_initialization()
    
    # Test getting a model with WebNN/WebGPU preference
    model = test_get_model_with_web_preference()
    
    # Test concurrent execution
    test_concurrent_execution()
    
    # Test ResourcePool cleanup
    test_resource_pool_cleanup()
    
    logger.info("WebNN/WebGPU Resource Pool integration test completed")

if __name__ == "__main__":
    main()