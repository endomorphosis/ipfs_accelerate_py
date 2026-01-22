#!/usr/bin/env python3
"""
Test Script for Enhanced Error Recovery with WebGPU/WebNN Resource Pool

This script demonstrates the integration of the Enhanced Error Recovery system with
the WebGPU/WebNN Resource Pool, showcasing features like:
- Performance-based recovery strategy selection
- Adaptive recovery timeouts
- Progressive recovery escalation
- Recovery performance tracking
- Hardware-aware resource selection
- Model sharding fault tolerance
- Browser connection recovery

Usage:
    python run_test_enhanced_recovery.py [--models MODEL_LIST] [--test-performance-tracking]
                                     [--test-adaptive-timeouts] [--test-progressive-recovery]
                                     [--test-hardware-aware] [--test-sharded-recovery]
                                     [--stress-test] [--duration SECONDS]
                                     [--analyze-recovery-metrics]
"""

import argparse
import asyncio
import json
import logging
import os
import random
import signal
import sys
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple

# Import the resource pool integration
from resource_pool_bridge import ResourcePoolBridgeIntegration, ModelProxy

# Import enhanced recovery system
from resource_pool_enhanced_recovery import (
    EnhancedResourcePoolRecovery, 
    FaultTolerantModelProxyEnhanced,
    ResourcePoolErrorCategory
)

# Import error recovery with performance tracking
from error_recovery_with_performance import (
    RecoveryPerformanceMetric,
    RecoveryPerformanceRecord,
    ProgressiveRecoveryLevel
)

from error_recovery_strategies import (
    ErrorCategory
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("enhanced_recovery_test.log")
    ]
)
logger = logging.getLogger(__name__)

# Sample model configurations for testing
SAMPLE_MODELS = {
    "bert": {
        "name": "bert-base-uncased",
        "type": "text_embedding",
        "input_example": "This is a sample text for embedding",
        "hardware_preferences": {"priority_list": ["webgpu", "cpu"]}
    },
    "vit": {
        "name": "vit-base-patch16-224",
        "type": "vision",
        "input_example": {"image_data": "simulated_image_data", "width": 224, "height": 224},
        "hardware_preferences": {"priority_list": ["webgpu", "cpu"]}
    },
    "whisper": {
        "name": "whisper-small",
        "type": "audio",
        "input_example": {"audio_data": "simulated_audio_data", "sample_rate": 16000},
        "hardware_preferences": {"priority_list": ["webgpu", "cpu"]}
    },
    "llama": {
        "name": "llama-7b",
        "type": "large_language_model",
        "input_example": "Write a short poem about technology",
        "hardware_preferences": {"priority_list": ["webgpu", "webnn", "cpu"]}
    }
}

class MockStateManager:
    """Mock state manager for testing."""
    
    def __init__(self):
        self.browsers = {}
        self.models = {}
        self.operations = {}
        self.sync_interval = 5
        
    async def register_browser(self, browser_id, browser_type, capabilities):
        self.browsers[browser_id] = {
            "id": browser_id,
            "type": browser_type,
            "capabilities": capabilities,
            "status": "ready",
            "created_at": datetime.now().isoformat()
        }
        return True
        
    async def register_model(self, model_id, model_name, model_type, browser_id):
        self.models[model_id] = {
            "id": model_id,
            "name": model_name,
            "type": model_type,
            "browser_id": browser_id,
            "created_at": datetime.now().isoformat()
        }
        return True
        
    def get_browser_state(self, browser_id):
        return self.browsers.get(browser_id)
        
    def get_model_state(self, model_id):
        return self.models.get(model_id)
        
    async def record_operation(self, operation_id, model_id, operation_type, start_time, status, metadata):
        self.operations[operation_id] = {
            "id": operation_id,
            "model_id": model_id,
            "operation_type": operation_type,
            "start_time": start_time,
            "status": status,
            "metadata": metadata
        }
        return True
        
    async def complete_operation(self, operation_id, status, end_time, result):
        if operation_id in self.operations:
            self.operations[operation_id].update({
                "status": status,
                "end_time": end_time,
                "result": result
            })
            return True
        return False
        
    async def update_model_browser(self, model_id, browser_id):
        if model_id in self.models:
            self.models[model_id]["browser_id"] = browser_id
            return True
        return False

class MockPerformanceTracker:
    """Mock performance tracker for testing."""
    
    def __init__(self):
        self.performance_records = []
        
    async def record_operation_performance(
        self, browser_id, model_id, model_type, operation_type, latency, success, metadata
    ):
        self.performance_records.append({
            "browser_id": browser_id,
            "model_id": model_id,
            "model_type": model_type,
            "operation_type": operation_type,
            "latency": latency,
            "success": success,
            "metadata": metadata,
            "timestamp": datetime.now().isoformat()
        })
        return True
        
    async def get_operation_history(self, model_type=None, browser_id=None, time_range=None):
        filtered_records = []
        for record in self.performance_records:
            if model_type and record["model_type"] != model_type:
                continue
                
            if browser_id and record["browser_id"] != browser_id:
                continue
                
            # Simple time range filter
            if time_range:
                # Not implementing full time range filtering for the mock
                pass
                
            filtered_records.append(record)
            
        return filtered_records

class MockShardingManager:
    """Mock sharding manager for testing."""
    
    def __init__(self):
        self.sharded_models = {}
        
    def get_sharded_model(self, model_id):
        return self.sharded_models.get(model_id)
        
    async def create_sharded_model(self, model_id, model_name, num_shards, sharding_strategy):
        # Create a sharded model with specified number of shards
        shards = {}
        for i in range(num_shards):
            shard_id = f"shard-{i}-{uuid.uuid4().hex[:4]}"
            shards[shard_id] = {
                "id": shard_id,
                "browser_id": f"browser-{i}-{uuid.uuid4().hex[:4]}",
                "index": i,
                "status": "ready"
            }
            
        self.sharded_models[model_id] = {
            "id": model_id,
            "name": model_name,
            "num_shards": num_shards,
            "sharding_strategy": sharding_strategy,
            "shards": shards,
            "created_at": datetime.now().isoformat()
        }
        return True
        
    async def update_shard_browser(self, model_id, shard_id, browser_id):
        if model_id in self.sharded_models and shard_id in self.sharded_models[model_id]["shards"]:
            self.sharded_models[model_id]["shards"][shard_id]["browser_id"] = browser_id
            return True
        return False

async def setup_resource_pool_recovery():
    """Set up the enhanced recovery manager for testing."""
    
    # Create mock components
    state_manager = MockStateManager()
    performance_tracker = MockPerformanceTracker()
    sharding_manager = MockShardingManager()
    connection_pool = {}
    
    # Create 3 sample browser connections
    for i in range(3):
        browser_id = f"browser-{i}-{uuid.uuid4().hex[:4]}"
        browser_type = ["chrome", "firefox", "edge"][i % 3]
        
        connection_pool[browser_id] = {
            'id': browser_id,
            'type': browser_type,
            'status': 'ready',
            'capabilities': {
                'webgpu': True, 
                'webnn': browser_type == 'edge',
                'compute_shaders': browser_type == 'firefox'
            },
            'created_at': datetime.now().isoformat(),
            'active_models': set(),
            'performance_metrics': {}
        }
        
    # Create enhanced recovery manager
    recovery_manager = EnhancedResourcePoolRecovery(
        state_manager=state_manager,
        performance_tracker=performance_tracker,
        sharding_manager=sharding_manager,
        connection_pool=connection_pool,
        recovery_database_path=":memory:",  # Use in-memory database for testing
        enable_performance_tracking=True,
        enable_hardware_aware_recovery=True,
        enable_progressive_recovery=True,
        adaptive_timeouts=True
    )
    
    # Initialize recovery manager
    await recovery_manager.initialize()
    
    return recovery_manager, state_manager, performance_tracker, sharding_manager, connection_pool

async def create_fault_tolerant_model(recovery_manager, model_name):
    """Create a fault-tolerant model using the enhanced recovery manager."""
    
    if model_name not in SAMPLE_MODELS:
        logger.error(f"Model {model_name} not found in sample models")
        return None
        
    model_config = SAMPLE_MODELS[model_name]
    
    # Get a browser from connection pool
    browser_id = list(recovery_manager.connection_pool.keys())[0]
    
    # Create model ID
    model_id = f"model-{model_name}-{uuid.uuid4().hex[:4]}"
    
    # Create model in state manager
    if recovery_manager.state_manager:
        await recovery_manager.state_manager.register_model(
            model_id=model_id,
            model_name=model_config["name"],
            model_type=model_config["type"],
            browser_id=browser_id
        )
    
    # Set model recovery settings
    await recovery_manager.set_model_recovery_settings(
        model_id=model_id,
        recovery_timeout=30,
        state_persistence=True,
        failover_strategy="progressive",
        priority_level="medium"
    )
    
    # Create fault tolerant model proxy
    model = FaultTolerantModelProxyEnhanced(
        model_id=model_id,
        model_name=model_config["name"],
        model_type=model_config["type"],
        browser_id=browser_id,
        recovery_manager=recovery_manager
    )
    
    # Add to active models in browser
    if recovery_manager.connection_pool and browser_id in recovery_manager.connection_pool:
        recovery_manager.connection_pool[browser_id]['active_models'].add(model_id)
    
    return model, model_config

async def test_basic_functionality(recovery_manager):
    """Test basic functionality of the enhanced recovery system."""
    logger.info("Testing basic functionality of the enhanced recovery system")
    
    # Create a fault tolerant model
    model, model_config = await create_fault_tolerant_model(recovery_manager, "bert")
    
    if not model:
        logger.error("Failed to create model")
        return False
    
    logger.info(f"Created model: {await model.get_info()}")
    
    # Run inference
    try:
        result = await model(model_config["input_example"])
        logger.info(f"Inference result: {result}")
        
        # Get model info
        info = await model.get_info()
        logger.info(f"Model info: {info}")
        
        return True
    except Exception as e:
        logger.error(f"Error in basic functionality test: {str(e)}")
        return False

async def test_performance_tracking(recovery_manager):
    """Test performance tracking functionality."""
    logger.info("Testing performance tracking functionality")
    
    # Create a fault tolerant model
    model, model_config = await create_fault_tolerant_model(recovery_manager, "bert")
    
    if not model:
        logger.error("Failed to create model")
        return False
    
    # Run multiple inferences to generate performance data
    for i in range(5):
        try:
            result = await model(model_config["input_example"])
            logger.info(f"Inference {i+1} completed")
        except Exception as e:
            logger.error(f"Error in inference {i+1}: {str(e)}")
            return False
    
    # Check performance tracker
    if recovery_manager.performance_tracker:
        records = await recovery_manager.performance_tracker.get_operation_history(
            model_type=model_config["type"]
        )
        
        if not records:
            logger.error("No performance records found")
            return False
        
        logger.info(f"Found {len(records)} performance records")
        
        # Check a few records
        for i, record in enumerate(records[:2]):
            logger.info(f"Performance record {i+1}: {record}")
        
        return True
    else:
        logger.error("Performance tracker not available")
        return False

async def test_fault_tolerance(recovery_manager):
    """Test fault tolerance functionality."""
    logger.info("Testing fault tolerance functionality")
    
    # Create a fault tolerant model
    model, model_config = await create_fault_tolerant_model(recovery_manager, "vit")
    
    if not model:
        logger.error("Failed to create model")
        return False
    
    original_browser_id = model.browser_id
    logger.info(f"Model created with browser ID: {original_browser_id}")
    
    # Run baseline inference
    try:
        result = await model(model_config["input_example"])
        logger.info(f"Baseline inference completed")
    except Exception as e:
        logger.error(f"Error in baseline inference: {str(e)}")
        return False
    
    # Simulate browser crash
    logger.info(f"Simulating browser crash by changing browser_id to 'crashed-browser'")
    model.browser_id = "crashed-browser"
    
    # Run inference again - should trigger recovery
    try:
        result = await model(model_config["input_example"])
        logger.info(f"Recovery inference completed: {result}")
        
        # Check that browser ID changed (recovery happened)
        if model.browser_id != original_browser_id and model.browser_id != "crashed-browser":
            logger.info(f"✅ Fault tolerance recovery successful, new browser_id: {model.browser_id}")
            
            # Get model info
            info = await model.get_info()
            logger.info(f"Model info after recovery: {info}")
            
            return True
        else:
            logger.error(f"❌ Fault tolerance recovery failed - browser ID did not change")
            return False
    except Exception as e:
        logger.error(f"Error in recovery inference: {str(e)}")
        return False

async def test_progressive_recovery(recovery_manager):
    """Test progressive recovery functionality."""
    logger.info("Testing progressive recovery functionality")
    
    # Create a fault tolerant model
    model, model_config = await create_fault_tolerant_model(recovery_manager, "whisper")
    
    if not model:
        logger.error("Failed to create model")
        return False
    
    original_browser_id = model.browser_id
    logger.info(f"Model created with browser ID: {original_browser_id}")
    
    # Setup mock to track recovery levels
    recovery_levels = []
    
    # Save the original recover_operation method
    original_recover_operation = recovery_manager.error_recovery.recover_operation
    
    # Create a mock that tracks recovery levels
    async def mock_recover_operation(*args, **kwargs):
        # Call the original method
        result = await original_recover_operation(*args, **kwargs)
        
        # Track the recovery level
        if hasattr(result, 'recovery_level'):
            recovery_levels.append(result.recovery_level)
            
        return result
    
    # Replace the method
    recovery_manager.error_recovery.recover_operation = mock_recover_operation
    
    # Fail multiple times to trigger progressive recovery
    for i in range(3):
        # Simulate browser crash
        logger.info(f"Simulating browser crash {i+1}")
        model.browser_id = f"crashed-browser-{i}"
        
        # Run inference - should trigger recovery with escalating levels
        try:
            result = await model(model_config["input_example"])
            logger.info(f"Recovery {i+1} inference completed")
            
            # Get new browser ID
            new_browser_id = model.browser_id
            logger.info(f"Recovery {i+1} assigned browser ID: {new_browser_id}")
            
        except Exception as e:
            logger.error(f"Error in recovery {i+1} inference: {str(e)}")
            # Continue to next iteration
    
    # Restore original method
    recovery_manager.error_recovery.recover_operation = original_recover_operation
    
    # Check that we saw multiple recovery levels
    if len(recovery_levels) < 2:
        logger.error(f"Did not see multiple recovery levels: {recovery_levels}")
        return False
    
    # Check if we saw progressively higher recovery levels
    logger.info(f"Recovery levels: {recovery_levels}")
    
    # If we tracked at least some recovery levels and they increased, test is successful
    if len(recovery_levels) >= 2 and recovery_levels[-1] > recovery_levels[0]:
        logger.info("✅ Progressive recovery successful - recovery levels escalated")
        return True
    else:
        logger.info("❌ Progressive recovery test failed - recovery levels did not escalate")
        return False

async def test_adaptive_timeouts(recovery_manager):
    """Test adaptive timeout functionality."""
    logger.info("Testing adaptive timeout functionality")
    
    # Check if adaptive timeouts are enabled
    if not recovery_manager.error_recovery.adaptive_timeouts:
        logger.error("Adaptive timeouts not enabled")
        return False
    
    # Create a fault tolerant model
    model, model_config = await create_fault_tolerant_model(recovery_manager, "bert")
    
    if not model:
        logger.error("Failed to create model")
        return False
    
    # Track original timeout
    original_timeout = None
    
    # Access the timeout directly from the component settings
    for component_id, settings in recovery_manager.error_recovery.component_settings.items():
        if hasattr(settings, 'recovery_timeout'):
            original_timeout = settings.recovery_timeout
            break
    
    if original_timeout is None:
        logger.error("Could not find original timeout")
        return False
    
    logger.info(f"Original timeout: {original_timeout} seconds")
    
    # Simulate performance data
    for i in range(10):
        await recovery_manager.error_recovery.record_recovery_performance(
            component_id=model.model_id,
            operation_type="inference",
            recovery_action="model_reinitialize",
            success=True,
            execution_time=original_timeout * 0.8 * (0.9 + 0.2 * random.random()),  # Vary around 80% of timeout
            metrics={
                RecoveryPerformanceMetric.RECOVERY_TIME: original_timeout * 0.8 * (0.9 + 0.2 * random.random()),
                RecoveryPerformanceMetric.SUCCESS_RATE: 1.0,
                RecoveryPerformanceMetric.RESOURCE_USAGE: 50.0
            }
        )
    
    # Trigger timeout adaptation
    await recovery_manager.error_recovery.adapt_timeouts()
    
    # Check if timeout changed
    new_timeout = None
    
    # Access the timeout directly from the component settings
    for component_id, settings in recovery_manager.error_recovery.component_settings.items():
        if hasattr(settings, 'recovery_timeout'):
            new_timeout = settings.recovery_timeout
            break
    
    if new_timeout is None:
        logger.error("Could not find new timeout")
        return False
    
    logger.info(f"New timeout after adaptation: {new_timeout} seconds")
    
    # Check if timeout changed appropriately
    if new_timeout != original_timeout:
        logger.info(f"✅ Adaptive timeout successful - timeout changed from {original_timeout} to {new_timeout}")
        return True
    else:
        logger.info("❌ Adaptive timeout test failed - timeout did not change")
        return False

async def test_hardware_aware_recovery(recovery_manager):
    """Test hardware-aware recovery functionality."""
    logger.info("Testing hardware-aware recovery functionality")
    
    # Check if hardware-aware recovery is enabled
    if not recovery_manager.hardware_matcher:
        logger.error("Hardware-aware recovery not enabled")
        return False
    
    # Create a sharded model
    model_id = f"sharded-model-{uuid.uuid4().hex[:4]}"
    model_name = "llama-13b"
    num_shards = 3
    
    # Create the sharded model
    await recovery_manager.sharding_manager.create_sharded_model(
        model_id=model_id,
        model_name=model_name,
        num_shards=num_shards,
        sharding_strategy="layer"
    )
    
    # Get the sharded model
    sharded_model = recovery_manager.sharding_manager.get_sharded_model(model_id)
    
    if not sharded_model:
        logger.error(f"Sharded model {model_id} not found")
        return False
    
    logger.info(f"Created sharded model with {num_shards} shards")
    
    # Get a shard ID
    shard_id = list(sharded_model["shards"].keys())[0]
    original_browser_id = sharded_model["shards"][shard_id]["browser_id"]
    
    logger.info(f"Testing recovery for shard {shard_id}, browser {original_browser_id}")
    
    # Set up recovery context
    recovery_context = {
        "model_id": model_id,
        "failed_shard_ids": [shard_id],
        "error": "Simulated shard failure",
        "strategy": "retry_failed_shards"
    }
    
    # Simulate hardware matcher integration
    # Save original method
    original_get_hardware_matches = None
    if hasattr(recovery_manager.hardware_matcher, 'get_hardware_matches'):
        original_get_hardware_matches = recovery_manager.hardware_matcher.get_hardware_matches
    
    # Mock hardware matcher
    async def mock_get_hardware_matches(model_type, operation_type, hardware_preferences=None):
        # Return appropriate hardware type based on model type
        if model_type == "text_embedding":
            return ["cpu", "webgpu"]
        elif model_type == "vision":
            return ["webgpu", "cpu"]
        elif model_type == "audio":
            return ["webgpu", "cpu"]
        elif model_type == "large_language_model":
            return ["webgpu", "webnn", "cpu"]
        else:
            return ["cpu"]
    
    # Replace the method if it exists
    if original_get_hardware_matches:
        recovery_manager.hardware_matcher.get_hardware_matches = mock_get_hardware_matches
    
    # Recover sharded model
    success = await recovery_manager.recover_sharded_model(
        model_id=model_id,
        failed_shard_ids=[shard_id],
        error="Simulated shard failure",
        strategy="reassign_shards"
    )
    
    # Restore original method if it exists
    if original_get_hardware_matches:
        recovery_manager.hardware_matcher.get_hardware_matches = original_get_hardware_matches
    
    if not success:
        logger.error("Shard recovery failed")
        return False
    
    # Get updated sharded model
    sharded_model = recovery_manager.sharding_manager.get_sharded_model(model_id)
    
    if not sharded_model:
        logger.error(f"Sharded model {model_id} not found after recovery")
        return False
    
    # Check if browser ID changed
    new_browser_id = sharded_model["shards"][shard_id]["browser_id"]
    
    if new_browser_id != original_browser_id:
        logger.info(f"✅ Hardware-aware recovery successful - browser changed from {original_browser_id} to {new_browser_id}")
        return True
    else:
        logger.info(f"❌ Hardware-aware recovery test failed - browser did not change")
        return False

async def test_sharded_recovery(recovery_manager):
    """Test recovery of sharded models."""
    logger.info("Testing recovery of sharded models")
    
    # Create a sharded model
    model_id = f"sharded-model-{uuid.uuid4().hex[:4]}"
    model_name = "llama-13b"
    num_shards = 3
    
    # Create the sharded model
    await recovery_manager.sharding_manager.create_sharded_model(
        model_id=model_id,
        model_name=model_name,
        num_shards=num_shards,
        sharding_strategy="layer"
    )
    
    # Get the sharded model
    sharded_model = recovery_manager.sharding_manager.get_sharded_model(model_id)
    
    if not sharded_model:
        logger.error(f"Sharded model {model_id} not found")
        return False
    
    logger.info(f"Created sharded model with {num_shards} shards")
    
    # Test different recovery strategies
    strategies = ["retry_failed_shards", "reassign_shards", "full_retry"]
    results = {}
    
    for strategy in strategies:
        # Get a random shard ID
        shard_id = random.choice(list(sharded_model["shards"].keys()))
        original_browser_id = sharded_model["shards"][shard_id]["browser_id"]
        
        logger.info(f"Testing {strategy} recovery for shard {shard_id}, browser {original_browser_id}")
        
        # Simulate browser failure for the shard
        recovery_manager.sharding_manager.sharded_models[model_id]["shards"][shard_id]["browser_id"] = "crashed-browser"
        
        # Recover sharded model with the strategy
        success = await recovery_manager.recover_sharded_model(
            model_id=model_id,
            failed_shard_ids=[shard_id],
            error="Simulated shard failure",
            strategy=strategy
        )
        
        # Check recovery success
        if not success:
            logger.error(f"Recovery failed for strategy {strategy}")
            results[strategy] = False
            continue
        
        # For reassign_shards, check if browser ID changed
        if strategy == "reassign_shards":
            # Get updated sharded model
            updated_model = recovery_manager.sharding_manager.get_sharded_model(model_id)
            new_browser_id = updated_model["shards"][shard_id]["browser_id"]
            
            if new_browser_id != original_browser_id and new_browser_id != "crashed-browser":
                logger.info(f"✅ Shard recovery with {strategy} successful - browser changed from {original_browser_id} to {new_browser_id}")
                results[strategy] = True
            else:
                logger.info(f"❌ Shard recovery with {strategy} failed - browser did not change")
                results[strategy] = False
        else:
            # For other strategies, trust the success flag
            logger.info(f"✅ Shard recovery with {strategy} completed successfully")
            results[strategy] = True
    
    # Return success if at least two strategies worked
    successful_strategies = sum(1 for result in results.values() if result)
    logger.info(f"Successfully tested {successful_strategies}/{len(strategies)} recovery strategies")
    
    return successful_strategies >= 2

async def analyze_recovery_metrics(recovery_manager):
    """Analyze recovery metrics and optimize strategies."""
    logger.info("Analyzing recovery metrics and optimizing strategies")
    
    # Generate some recovery performance data
    models = []
    model_types = ["text_embedding", "vision", "audio", "large_language_model"]
    error_categories = list(ResourcePoolErrorCategory)
    recovery_actions = ["model_reinitialize", "model_migrate", "browser_reconnect", "browser_restart", "shard_retry", "shard_reassign"]
    
    # Create models of different types
    for model_type in model_types:
        model_name = next((name for name, config in SAMPLE_MODELS.items() if config["type"] == model_type), None)
        if model_name:
            model, model_config = await create_fault_tolerant_model(recovery_manager, model_name)
            if model:
                models.append((model, model_config))
    
    if not models:
        logger.error("Failed to create any models for metrics generation")
        return False
    
    # Generate recovery performance data
    for i in range(50):  # Generate a good amount of data
        for model, model_config in models:
            # Generate random performance metrics
            error_category = random.choice(error_categories)
            recovery_action = random.choice(recovery_actions)
            
            # 80% success rate
            success = random.random() < 0.8
            
            # Generate realistic metrics
            execution_time = random.uniform(0.5, 3.0)
            resource_usage = random.uniform(20.0, 80.0)
            recovery_time = random.uniform(1.0, 5.0)
            success_rate = 1.0 if success else 0.0
            
            # Record recovery performance
            await recovery_manager.error_recovery.record_recovery_performance(
                component_id=model.model_id,
                operation_type="inference",
                recovery_action=recovery_action,
                success=success,
                execution_time=execution_time,
                metrics={
                    RecoveryPerformanceMetric.RECOVERY_TIME: recovery_time,
                    RecoveryPerformanceMetric.SUCCESS_RATE: success_rate,
                    RecoveryPerformanceMetric.RESOURCE_USAGE: resource_usage
                }
            )
    
    logger.info("Generated recovery performance data")
    
    # Analyze recovery performance
    for model_type in model_types:
        logger.info(f"\nAnalyzing recovery performance for {model_type} models:")
        
        analysis = await recovery_manager.analyze_recovery_performance(
            model_type=model_type,
            time_range="7d",
            metrics=["success_rate", "recovery_time", "resource_usage"]
        )
        
        if not analysis:
            logger.warning(f"No analysis data for {model_type}")
            continue
        
        # Log top-level metrics
        if "overall_metrics" in analysis:
            logger.info(f"Overall metrics: {analysis['overall_metrics']}")
        
        # Log top strategies
        if "top_strategies" in analysis:
            top_strategies = analysis["top_strategies"]
            if top_strategies:
                logger.info(f"Top strategies:")
                for i, strategy in enumerate(top_strategies[:3]):
                    logger.info(f"  {i+1}. {strategy['strategy']}: Score={strategy['score']:.2f}, Success={strategy['success_rate']:.1%}")
    
    # Optimize recovery strategies
    logger.info("\nOptimizing recovery strategies:")
    
    try:
        optimization_result = await recovery_manager.optimize_recovery_strategies()
        
        if optimization_result:
            logger.info("Strategy optimization results:")
            
            # Log optimized strategies per component
            if "component_strategies" in optimization_result:
                component_strategies = optimization_result["component_strategies"]
                for component_id, strategies in list(component_strategies.items())[:2]:  # Show first two components
                    logger.info(f"  Component {component_id}:")
                    for strategy, score in strategies.items():
                        logger.info(f"    {strategy}: {score:.2f}")
            
            # Log global strategy improvements
            if "global_improvements" in optimization_result:
                logger.info(f"Global improvements: {optimization_result['global_improvements']}")
            
            return True
        else:
            logger.warning("No optimization results")
            return False
    except Exception as e:
        logger.error(f"Error in strategy optimization: {str(e)}")
        return False

async def test_browser_recovery(recovery_manager):
    """Test browser recovery functionality."""
    logger.info("Testing browser recovery functionality")
    
    # Get an existing browser ID
    browser_ids = list(recovery_manager.connection_pool.keys())
    if not browser_ids:
        logger.error("No browsers available in connection pool")
        return False
    
    browser_id = browser_ids[0]
    browser_type = recovery_manager.connection_pool[browser_id]["type"]
    
    logger.info(f"Testing recovery for browser {browser_id} of type {browser_type}")
    
    # Recover browser
    success, new_browser_id = await recovery_manager.recover_browser(
        browser_id=browser_id,
        error="Simulated browser crash",
        browser_type=browser_type
    )
    
    if not success:
        logger.error("Browser recovery failed")
        return False
    
    logger.info(f"Browser recovery successful, new browser ID: {new_browser_id}")
    
    # Check that new browser ID is different
    if new_browser_id == browser_id:
        logger.error("New browser ID is the same as the old one")
        return False
    
    # Check that the new browser is in the connection pool
    if new_browser_id not in recovery_manager.connection_pool:
        logger.error(f"New browser {new_browser_id} not found in connection pool")
        return False
    
    # Check that the new browser has the same type
    if recovery_manager.connection_pool[new_browser_id]["type"] != browser_type:
        logger.error(f"New browser has type {recovery_manager.connection_pool[new_browser_id]['type']}, expected {browser_type}")
        return False
    
    logger.info(f"✅ Browser recovery created a valid replacement browser")
    return True

async def stress_test(recovery_manager, model_list, duration, fault_injection):
    """Run a stress test with enhanced recovery."""
    logger.info(f"Running stress test for {duration} seconds with fault injection: {fault_injection}")
    
    # Track results
    total_operations = 0
    successful_operations = 0
    failed_operations = 0
    recovery_attempts = 0
    recovery_successes = 0
    
    # Create models
    models = []
    for model_name in model_list:
        if model_name not in SAMPLE_MODELS:
            continue
            
        model, model_config = await create_fault_tolerant_model(recovery_manager, model_name)
        
        if model:
            models.append((model, model_config, model_name))
            logger.info(f"Created model {model_name} for stress test")
        else:
            logger.error(f"Failed to create model {model_name} for stress test")
    
    if not models:
        logger.error("No models were created for stress test")
        return False
    
    # Run operations for the specified duration
    start_time = time.time()
    end_time = start_time + duration
    
    while time.time() < end_time:
        # Select a random model
        model, model_config, model_name = random.choice(models)
        
        try:
            # Inject fault randomly if enabled
            if fault_injection and random.random() < 0.3:
                original_browser_id = model.browser_id
                logger.info(f"Injecting fault for model {model_name} by changing browser_id to 'crashed-browser'")
                model.browser_id = "crashed-browser"
                recovery_attempts += 1
            
            # Run inference
            result = await model(model_config["input_example"])
            
            # Check if recovery happened
            if fault_injection and 'original_browser_id' in locals():
                if model.browser_id != original_browser_id and model.browser_id != "crashed-browser":
                    logger.info(f"Fault recovery successful for model {model_name}, new browser_id: {model.browser_id}")
                    recovery_successes += 1
                elif model.browser_id == "crashed-browser":
                    logger.warning(f"Fault recovery failed for model {model_name}")
            
            # Operation completed successfully
            successful_operations += 1
            
        except Exception as e:
            logger.error(f"Operation failed for model {model_name}: {str(e)}")
            failed_operations += 1
        
        total_operations += 1
        
        # Brief pause to avoid flooding
        await asyncio.sleep(0.1)
    
    # Log results
    elapsed = time.time() - start_time
    operations_per_second = total_operations / elapsed
    
    logger.info(f"Stress test completed:")
    logger.info(f"- Duration: {elapsed:.2f} seconds")
    logger.info(f"- Total operations: {total_operations}")
    logger.info(f"- Successful operations: {successful_operations}")
    logger.info(f"- Failed operations: {failed_operations}")
    logger.info(f"- Operations per second: {operations_per_second:.2f}")
    
    if fault_injection:
        logger.info(f"- Fault injections: {recovery_attempts}")
        logger.info(f"- Successful recoveries: {recovery_successes}")
        if recovery_attempts > 0:
            logger.info(f"- Recovery success rate: {recovery_successes / recovery_attempts * 100:.2f}%")
    
    return True

async def main():
    """Main entry point for the test script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test Enhanced Error Recovery for WebGPU/WebNN Resource Pool")
    
    parser.add_argument("--models", default="bert,vit,whisper", help="Comma-separated list of models to test")
    parser.add_argument("--test-performance-tracking", action="store_true", help="Test performance tracking functionality")
    parser.add_argument("--test-adaptive-timeouts", action="store_true", help="Test adaptive timeouts")
    parser.add_argument("--test-progressive-recovery", action="store_true", help="Test progressive recovery escalation")
    parser.add_argument("--test-hardware-aware", action="store_true", help="Test hardware-aware resource selection")
    parser.add_argument("--test-sharded-recovery", action="store_true", help="Test recovery of sharded models")
    parser.add_argument("--test-browser-recovery", action="store_true", help="Test browser recovery")
    parser.add_argument("--analyze-recovery-metrics", action="store_true", help="Analyze recovery metrics and optimize strategies")
    parser.add_argument("--stress-test", action="store_true", help="Run stress test with high concurrency")
    parser.add_argument("--duration", type=int, default=30, help="Duration of stress test in seconds")
    parser.add_argument("--fault-injection", action="store_true", help="Enable fault injection in stress test")
    
    args = parser.parse_args()
    
    # Parse model list
    model_list = args.models.split(",")
    
    logger.info("Starting Enhanced Error Recovery for WebGPU/WebNN Resource Pool test")
    logger.info(f"Models: {model_list}")
    
    # Setup resource pool recovery
    recovery_manager, state_manager, performance_tracker, sharding_manager, connection_pool = await setup_resource_pool_recovery()
    
    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_event_loop()
    
    should_exit = False
    
    def shutdown_handler(signum, frame):
        nonlocal should_exit
        logger.info(f"Received signal {signum}, initiating shutdown")
        should_exit = True
    
    # Register signal handlers
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, shutdown_handler)
    
    # Run tests based on arguments
    test_results = {}
    
    try:
        # Always run basic functionality test
        test_results["basic_functionality"] = await test_basic_functionality(recovery_manager)
        
        # Run selected tests
        if args.test_performance_tracking:
            test_results["performance_tracking"] = await test_performance_tracking(recovery_manager)
        
        if args.test_adaptive_timeouts:
            test_results["adaptive_timeouts"] = await test_adaptive_timeouts(recovery_manager)
        
        if args.test_progressive_recovery:
            test_results["progressive_recovery"] = await test_progressive_recovery(recovery_manager)
        
        if args.test_hardware_aware:
            test_results["hardware_aware_recovery"] = await test_hardware_aware_recovery(recovery_manager)
        
        if args.test_sharded_recovery:
            test_results["sharded_recovery"] = await test_sharded_recovery(recovery_manager)
        
        if args.test_browser_recovery:
            test_results["browser_recovery"] = await test_browser_recovery(recovery_manager)
        
        # Always test fault tolerance
        test_results["fault_tolerance"] = await test_fault_tolerance(recovery_manager)
        
        if args.analyze_recovery_metrics:
            test_results["analyze_recovery_metrics"] = await analyze_recovery_metrics(recovery_manager)
        
        # Run stress test last
        if args.stress_test:
            test_results["stress_test"] = await stress_test(recovery_manager, model_list, args.duration, args.fault_injection)
        
    except Exception as e:
        logger.error(f"Error in test execution: {str(e)}")
        
    # Print test results
    logger.info("\n=== Test Results ===")
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    success_count = sum(1 for result in test_results.values() if result)
    total_count = len(test_results)
    
    logger.info(f"\nSummary: {success_count}/{total_count} tests passed")
    
    # Clean up
    logger.info("Tests completed, shutting down")

if __name__ == "__main__":
    asyncio.run(main())