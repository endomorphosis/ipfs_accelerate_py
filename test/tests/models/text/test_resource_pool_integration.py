#!/usr/bin/env python3
"""
Test script for ResourcePoolBridgeIntegration with Adaptive Scaling.

This script tests the enhanced WebGPU/WebNN resource pool integration with
adaptive scaling for efficient model execution.
"""

import os
import sys
import time
import json
import anyio
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import resource pool bridge
from test.tests.web.web_platform.resource_pool_bridge import ResourcePoolBridgeIntegration

async def test_adaptive_scaling():
    """Test adaptive scaling functionality."""
    logger.info("Starting adaptive scaling test")
    
    # Create integration with adaptive scaling enabled
    integration = ResourcePoolBridgeIntegration(
        max_connections=4,
        enable_gpu=True,
        enable_cpu=True,
        headless=True,
        adaptive_scaling=True,
        monitoring_interval=5  # Short interval for testing
    )
    
    # Initialize integration
    integration.initialize()
    
    # Get initial metrics
    initial_metrics = integration.get_metrics()
    logger.info(f"Initial metrics: {json.dumps(initial_metrics, indent=2)}")
    
    # Load models of different types to trigger browser-specific optimizations
    models = []
    model_types = [
        ('text_embedding', 'bert-base-uncased'),
        ('vision', 'vit-base-patch16-224'),
        ('audio', 'whisper-tiny'),
        ('text_generation', 'opt-125m'),
        ('multimodal', 'clip-vit-base-patch32')
    ]
    
    # Load models with varying hardware preferences
    for model_type, model_name in model_types:
        model = integration.get_model(
            model_type=model_type,
            model_name=model_name,
            hardware_preferences={'priority_list': ['webgpu', 'webnn', 'cpu']}
        )
        models.append((model, model_type, model_name))
    
    # Get metrics after loading models
    after_load_metrics = integration.get_metrics()
    logger.info(f"Metrics after loading models: {json.dumps(after_load_metrics, indent=2)}")
    
    # Run models in sequence first to establish patterns
    logger.info("Running models in sequence")
    for model, model_type, model_name in models:
        # Create appropriate input based on model type
        if model_type == 'text_embedding' or model_type == 'text_generation':
            inputs = "This is a test input for text models."
        elif model_type == 'vision':
            inputs = {"image": {"width": 224, "height": 224, "channels": 3}}
        elif model_type == 'audio':
            inputs = {"audio": {"duration": 5.0, "sample_rate": 16000}}
        elif model_type == 'multimodal':
            inputs = {
                "image": {"width": 224, "height": 224, "channels": 3},
                "text": "This is a multimodal test input."
            }
        else:
            inputs = "Default test input"
        
        # Run inference
        result = model(inputs)
        logger.info(f"Model {model_name} ({model_type}) result: {result.get('inference_time'):.2f}s using {result.get('browser')} browser")
    
    # Create inputs for concurrent execution
    model_inputs = []
    for model, model_type, model_name in models:
        # Create appropriate input based on model type
        if model_type == 'text_embedding' or model_type == 'text_generation':
            inputs = "This is a test input for concurrent execution."
        elif model_type == 'vision':
            inputs = {"image": {"width": 224, "height": 224, "channels": 3}}
        elif model_type == 'audio':
            inputs = {"audio": {"duration": 5.0, "sample_rate": 16000}}
        elif model_type == 'multimodal':
            inputs = {
                "image": {"width": 224, "height": 224, "channels": 3},
                "text": "This is a multimodal test input."
            }
        else:
            inputs = "Default test input"
        
        model_inputs.append((model, inputs))
    
    # Run concurrent execution test
    logger.info("Running concurrent execution test")
    concurrent_results = await integration.execute_concurrent(model_inputs)
    
    # Log results
    for i, result in enumerate(concurrent_results):
        model, model_type, model_name = models[i]
        logger.info(f"Concurrent model {model_name} ({model_type}) result: {result.get('inference_time'):.2f}s using {result.get('browser')} browser")
    
    # Get metrics after concurrent execution
    after_concurrent_metrics = integration.get_metrics()
    logger.info(f"Metrics after concurrent execution: {json.dumps(after_concurrent_metrics, indent=2)}")
    
    # Run stress test to trigger adaptive scaling
    logger.info("Running stress test to trigger adaptive scaling")
    for _ in range(3):  # Run 3 batches
        # Run concurrent execution
        batch_results = await integration.execute_concurrent(model_inputs)
        
        # Get metrics after batch
        batch_metrics = integration.get_metrics()
        
        # Check scaling events
        scaling_events = batch_metrics.get('adaptive_scaling', {}).get('scaling_events', [])
        if scaling_events:
            logger.info(f"Scaling events detected: {len(scaling_events)}")
            for event in scaling_events[-3:]:  # Show last 3 events
                logger.info(f"  - Event: {event}")
        
        # Short delay to allow monitoring to run
        await anyio.sleep(5)
    
    # Get final metrics
    final_metrics = integration.get_metrics()
    logger.info(f"Final metrics: {json.dumps(final_metrics, indent=2)}")
    
    # Clean up
    integration.close()
    logger.info("Test completed successfully")

def main():
    """Main function to run the test."""
    # Create and run event loop
    loop = # TODO: Remove event loop management - anyio
    loop.run_until_complete(test_adaptive_scaling())
    loop.close()

if __name__ == "__main__":
    main()