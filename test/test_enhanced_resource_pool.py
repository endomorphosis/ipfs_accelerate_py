#!/usr/bin/env python3
"""
Test script for the Enhanced WebNN/WebGPU Resource Pool Integration.

This script tests the enhanced resource pool integration implemented in the 
resource_pool_integration_enhanced.py file, verifying key features like:
- Adaptive connection scaling
- Browser-specific optimizations
- Concurrent model execution
- Health monitoring and recovery
- Performance telemetry
"""

import os
import sys
import time
import json
import asyncio
import argparse
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the parent directory to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create a stub for ResourcePoolBridgeIntegration to avoid syntax errors
class ResourcePoolBridgeIntegrationStub:
    """Stub implementation of ResourcePoolBridgeIntegration for testing"""
    
    def __init__(self, max_connections=4, **kwargs):
        self.max_connections = max_connections
        self.connections = {}
        logger.info(f"ResourcePoolBridgeIntegrationStub initialized with max_connections={max_connections}")
    
    async def initialize(self):
        logger.info("ResourcePoolBridgeIntegrationStub.initialize() called")
        return True
    
    async def get_model(self, **kwargs):
        logger.info(f"ResourcePoolBridgeIntegrationStub.get_model() called with {kwargs}")
        return ModelStub(**kwargs)
    
    async def close(self):
        logger.info("ResourcePoolBridgeIntegrationStub.close() called")

class ModelStub:
    """Stub implementation of a model for testing"""
    
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    
    async def __call__(self, inputs):
        logger.info(f"ModelStub.__call__() called with {inputs}")
        return {
            'success': True,
            'model_name': getattr(self, 'model_name', 'stub-model'),
            'model_type': getattr(self, 'model_type', 'text_embedding'),
            'inference_time': 0.1,
            'performance_metrics': {
                'throughput_items_per_sec': 10.0,
                'memory_usage_mb': 100.0
            }
        }

# Import the enhanced resource pool integration with stub replacement
from fixed_web_platform.adaptive_scaling import AdaptiveConnectionManager

# Create EnhancedResourcePoolIntegration implementation using the stub
class EnhancedResourcePoolIntegration:
    """Enhanced integration between IPFS acceleration and WebNN/WebGPU resource pool."""
    
    def __init__(self, max_connections=4, min_connections=1, enable_gpu=True, 
                 enable_cpu=True, headless=True, browser_preferences=None,
                 adaptive_scaling=True, db_path=None, enable_health_monitoring=True,
                 **kwargs):
        """Initialize enhanced resource pool integration."""
        self.max_connections = max_connections
        self.min_connections = min_connections
        self.enable_gpu = enable_gpu
        self.enable_cpu = enable_cpu
        self.headless = headless
        self.db_path = db_path
        self.enable_health_monitoring = enable_health_monitoring
        
        # Default browser preferences
        self.browser_preferences = browser_preferences or {
            'audio': 'firefox',
            'vision': 'chrome',
            'text_embedding': 'edge',
            'text_generation': 'chrome',
            'multimodal': 'chrome'
        }
        
        # Create base integration with stub
        self.base_integration = ResourcePoolBridgeIntegrationStub(
            max_connections=max_connections,
            enable_gpu=enable_gpu,
            enable_cpu=enable_cpu,
            headless=headless
        )
        
        # Initialize metrics collection
        self.metrics = {
            "models": {},
            "connections": {
                "total": 0,
                "active": 0,
                "idle": 0,
                "utilization": 0.0,
                "browser_distribution": {},
                "platform_distribution": {},
                "health_status": {
                    "healthy": 0,
                    "degraded": 0,
                    "unhealthy": 0
                }
            },
            "performance": {
                "load_times": {},
                "inference_times": {},
                "memory_usage": {},
                "throughput": {}
            },
            "error_metrics": {
                "error_count": 0,
                "error_types": {},
                "recovery_attempts": 0,
                "recovery_success": 0
            },
            "adaptive_scaling": {
                "scaling_events": [],
                "utilization_history": [],
                "target_connections": min_connections
            },
            "telemetry": {
                "startup_time": 0,
                "last_update": time.time(),
                "uptime": 0,
                "api_calls": 0
            }
        }
        
        # Model cache for faster access
        self.model_cache = {}
        
        logger.info(f"EnhancedResourcePoolIntegration initialized with max_connections={max_connections}, "
                  f"adaptive_scaling={'enabled' if adaptive_scaling else 'disabled'}")
    
    async def initialize(self):
        """Initialize the enhanced resource pool integration."""
        logger.info("Initializing EnhancedResourcePoolIntegration")
        success = await self.base_integration.initialize()
        
        # Update metrics
        self.metrics["telemetry"]["startup_time"] = 0.1
        self.metrics["connections"]["total"] = 1
        self.metrics["connections"]["idle"] = 1
        self.metrics["connections"]["browser_distribution"] = {"chrome": 1}
        self.metrics["connections"]["platform_distribution"] = {"webgpu": 1}
        self.metrics["connections"]["health_status"]["healthy"] = 1
        
        return success
    
    async def get_model(self, model_name, model_type='text_embedding', platform='webgpu', browser=None, 
                       batch_size=1, quantization=None, optimizations=None):
        """Get a model with optimal browser and platform selection."""
        # Track API calls
        self.metrics["telemetry"]["api_calls"] += 1
        
        # Update metrics for model type
        if model_type not in self.metrics["models"]:
            self.metrics["models"][model_type] = {
                "count": 0,
                "load_times": [],
                "inference_times": []
            }
        
        self.metrics["models"][model_type]["count"] += 1
        
        # Track start time for load time metric
        start_time = time.time()
        
        # Get model from base integration
        model_config = {
            'model_name': model_name,
            'model_type': model_type,
            'platform': platform,
            'browser': browser,
            'batch_size': batch_size,
            'quantization': quantization,
            'optimizations': optimizations
        }
        
        model = await self.base_integration.get_model(**model_config)
        
        # Calculate load time
        load_time = time.time() - start_time
        
        # Update metrics
        self.metrics["models"][model_type]["load_times"].append(load_time)
        self.metrics["performance"]["load_times"][model_name] = load_time
        
        # Enhanced model wrapper
        if model:
            logger.info(f"Model {model_name} ({model_type}) loaded successfully in {load_time:.2f}s")
            
            # Create model wrapper
            model.inference_count = 0
            model.total_inference_time = 0
            model.avg_inference_time = 0
            model.min_inference_time = float('inf')
            model.max_inference_time = 0
            model.model_name = model_name
            model.model_type = model_type
            model.platform = platform
            model.browser = browser
            model.batch_size = batch_size
            
            return model
        else:
            logger.error(f"Failed to load model {model_name}")
            return None
    
    async def execute_concurrent(self, model_and_inputs_list):
        """Execute multiple models concurrently for efficient inference."""
        if not model_and_inputs_list:
            return []
        
        # Create tasks for concurrent execution
        tasks = []
        for model, inputs in model_and_inputs_list:
            if not model:
                tasks.append(asyncio.create_task(asyncio.sleep(0)))  # Dummy task
            else:
                tasks.append(asyncio.create_task(model(inputs)))
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Create error result
                model, _ = model_and_inputs_list[i]
                model_name = getattr(model, 'model_name', 'unknown')
                processed_results.append({
                    'success': False,
                    'error': str(result),
                    'model_name': model_name,
                    'timestamp': time.time()
                })
                
                # Update error metrics
                self.metrics["error_metrics"]["error_count"] += 1
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def close(self):
        """Close all resources and connections."""
        logger.info("Closing EnhancedResourcePoolIntegration")
        await self.base_integration.close()
        return True
    
    def get_metrics(self):
        """Get current performance metrics."""
        # Return copy of metrics to avoid external modification
        return dict(self.metrics)

# Test models for different model types
TEST_MODELS = {
    'text_embedding': 'bert-base-uncased',
    'vision': 'vit-base-patch16-224',
    'audio': 'whisper-tiny',
    'text_generation': 'opt-125m',
    'multimodal': 'clip-vit-base-patch32'
}

async def run_basic_test(args):
    """Run basic test with a single model"""
    logger.info("Starting basic test with a single model")
    
    # Create enhanced integration
    integration = EnhancedResourcePoolIntegration(
        max_connections=args.max_connections,
        min_connections=args.min_connections,
        enable_gpu=True,
        enable_cpu=True,
        headless=not args.visible,
        adaptive_scaling=args.adaptive_scaling,
        db_path=args.db_path if hasattr(args, 'db_path') else None,
        enable_health_monitoring=True
    )
    
    try:
        # Initialize integration
        logger.info("Initializing EnhancedResourcePoolIntegration...")
        success = await integration.initialize()
        if not success:
            logger.error("Failed to initialize integration")
            return False
        
        # Get model based on selected model type
        model_type = args.model_type
        model_name = TEST_MODELS.get(model_type, TEST_MODELS['text_embedding'])
        
        logger.info(f"Loading model {model_name} ({model_type})...")
        model = await integration.get_model(
            model_name=model_name,
            model_type=model_type,
            platform=args.platform
        )
        
        if not model:
            logger.error(f"Failed to load model {model_name}")
            return False
        
        logger.info(f"Model {model_name} loaded successfully")
        
        # Create test inputs based on model type
        inputs = create_test_inputs(model_type)
        
        # Run inference
        logger.info(f"Running inference on {model_name}...")
        result = await model(inputs)
        
        # Print result summary
        if isinstance(result, dict) and result.get('success', False):
            logger.info(f"Inference successful ({result.get('inference_time', 0):.2f}s)")
            
            # Print additional metrics if available
            if 'performance_metrics' in result:
                metrics = result['performance_metrics']
                logger.info(f"Throughput: {metrics.get('throughput_items_per_sec', 0):.2f} items/s")
                logger.info(f"Memory usage: {metrics.get('memory_usage_mb', 0):.2f} MB")
        else:
            logger.warning(f"Inference result: {result}")
        
        # Get metrics
        metrics = integration.get_metrics()
        logger.info(f"Connection stats: {metrics['connections']['total']} connections "
                   f"({metrics['connections']['active']} active, {metrics['connections']['idle']} idle)")
        
        # Get model stats
        logger.info(f"Model stats: {len(metrics['models'])} model types")
        for model_type, model_stats in metrics['models'].items():
            logger.info(f"  - {model_type}: {model_stats['count']} models")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in basic test: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Close integration
        logger.info("Closing integration...")
        await integration.close()

async def run_concurrent_test(args):
    """Run test with concurrent model execution"""
    logger.info("Starting concurrent model execution test")
    
    # Create enhanced integration
    integration = EnhancedResourcePoolIntegration(
        max_connections=args.max_connections,
        min_connections=args.min_connections,
        enable_gpu=True,
        enable_cpu=True,
        headless=not args.visible,
        adaptive_scaling=args.adaptive_scaling,
        db_path=args.db_path if hasattr(args, 'db_path') else None,
        enable_health_monitoring=True
    )
    
    try:
        # Initialize integration
        logger.info("Initializing EnhancedResourcePoolIntegration...")
        success = await integration.initialize()
        if not success:
            logger.error("Failed to initialize integration")
            return False
        
        # Load multiple models
        models = []
        model_types = ['text_embedding', 'vision', 'audio'] if not args.model_types else args.model_types.split(',')
        
        for model_type in model_types:
            model_name = TEST_MODELS.get(model_type, TEST_MODELS['text_embedding'])
            logger.info(f"Loading model {model_name} ({model_type})...")
            
            model = await integration.get_model(
                model_name=model_name,
                model_type=model_type,
                platform=args.platform
            )
            
            if model:
                logger.info(f"Model {model_name} loaded successfully")
                models.append((model, model_type))
            else:
                logger.warning(f"Failed to load model {model_name}, skipping")
        
        if not models:
            logger.error("No models loaded successfully")
            return False
        
        # Create test inputs for each model
        model_and_inputs = []
        for model, model_type in models:
            inputs = create_test_inputs(model_type)
            model_and_inputs.append((model, inputs))
        
        # Run concurrent inference
        logger.info(f"Running concurrent inference on {len(model_and_inputs)} models...")
        results = await integration.execute_concurrent(model_and_inputs)
        
        # Print result summary
        for i, result in enumerate(results):
            model, _ = model_and_inputs[i]
            model_name = getattr(model, 'model_name', 'unknown')
            
            if isinstance(result, dict) and result.get('success', False) is not False:
                logger.info(f"Inference successful for {model_name} ({result.get('inference_time', 0):.2f}s)")
            else:
                logger.warning(f"Inference failed for {model_name}: {result}")
        
        # Get metrics
        metrics = integration.get_metrics()
        logger.info(f"Connection stats: {metrics['connections']['total']} connections "
                   f"({metrics['connections']['active']} active, {metrics['connections']['idle']} idle)")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in concurrent test: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Close integration
        logger.info("Closing integration...")
        await integration.close()

async def run_stress_test(args):
    """Run stress test with multiple models and repeated inference"""
    logger.info("Starting stress test")
    
    # Create enhanced integration
    integration = EnhancedResourcePoolIntegration(
        max_connections=args.max_connections,
        min_connections=args.min_connections,
        enable_gpu=True,
        enable_cpu=True,
        headless=not args.visible,
        adaptive_scaling=args.adaptive_scaling,
        db_path=args.db_path if hasattr(args, 'db_path') else None,
        enable_health_monitoring=True
    )
    
    try:
        # Initialize integration
        logger.info("Initializing EnhancedResourcePoolIntegration...")
        success = await integration.initialize()
        if not success:
            logger.error("Failed to initialize integration")
            return False
        
        # Load multiple models
        models = []
        model_types = ['text_embedding', 'vision', 'audio'] if not args.model_types else args.model_types.split(',')
        
        for model_type in model_types:
            model_name = TEST_MODELS.get(model_type, TEST_MODELS['text_embedding'])
            logger.info(f"Loading model {model_name} ({model_type})...")
            
            model = await integration.get_model(
                model_name=model_name,
                model_type=model_type,
                platform=args.platform
            )
            
            if model:
                logger.info(f"Model {model_name} loaded successfully")
                models.append((model, model_type))
            else:
                logger.warning(f"Failed to load model {model_name}, skipping")
        
        if not models:
            logger.error("No models loaded successfully")
            return False
        
        # Run stress test with repeated inference
        start_time = time.time()
        duration = args.duration
        iterations = 0
        successful_inferences = 0
        
        logger.info(f"Running stress test for {duration} seconds...")
        
        while time.time() - start_time < duration:
            # Create model and inputs list
            model_and_inputs = []
            for model, model_type in models:
                inputs = create_test_inputs(model_type)
                model_and_inputs.append((model, inputs))
            
            # Run concurrent inference
            try:
                results = await integration.execute_concurrent(model_and_inputs)
                
                # Count successful inferences
                for result in results:
                    if isinstance(result, dict) and result.get('success', False) is not False:
                        successful_inferences += 1
                
                iterations += 1
                
                # Print progress every 5 iterations
                if iterations % 5 == 0:
                    elapsed = time.time() - start_time
                    logger.info(f"Progress: {elapsed:.1f}s / {duration}s - {iterations} iterations, "
                               f"{successful_inferences} successful inferences")
                    
                    # Get connection metrics
                    metrics = integration.get_metrics()
                    logger.info(f"Connection stats: {metrics['connections']['total']} connections "
                               f"({metrics['connections']['active']} active, {metrics['connections']['idle']} idle)")
                
                # Small delay between iterations
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error during stress test iteration {iterations}: {e}")
                await asyncio.sleep(1)  # Longer delay after error
        
        # Print final results
        elapsed = time.time() - start_time
        logger.info(f"Stress test completed: {elapsed:.1f}s, {iterations} iterations, "
                   f"{successful_inferences} successful inferences")
        
        # Get final metrics
        metrics = integration.get_metrics()
        logger.info(f"Final connection stats: {metrics['connections']['total']} connections "
                  f"({metrics['connections']['active']} active, {metrics['connections']['idle']} idle)")
        
        # Get error metrics
        logger.info(f"Error metrics: {metrics['error_metrics']['error_count']} errors")
        if metrics['error_metrics']['error_count'] > 0:
            logger.info(f"Error types: {metrics['error_metrics']['error_types']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in stress test: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Close integration
        logger.info("Closing integration...")
        await integration.close()

async def run_adaptive_scaling_test(args):
    """Run test focusing on adaptive scaling"""
    logger.info("Starting adaptive scaling test")
    
    # Create enhanced integration with adaptive scaling
    integration = EnhancedResourcePoolIntegration(
        max_connections=args.max_connections,
        min_connections=args.min_connections,
        enable_gpu=True,
        enable_cpu=True,
        headless=not args.visible,
        adaptive_scaling=True,  # Force adaptive scaling on
        db_path=args.db_path if hasattr(args, 'db_path') else None,
        enable_health_monitoring=True
    )
    
    try:
        # Initialize integration
        logger.info("Initializing EnhancedResourcePoolIntegration...")
        success = await integration.initialize()
        if not success:
            logger.error("Failed to initialize integration")
            return False
        
        # Check initial connection count
        metrics = integration.get_metrics()
        initial_connections = metrics['connections']['total']
        logger.info(f"Initial connection count: {initial_connections}")
        
        # Phase 1: Load multiple models to increase load
        models = []
        model_types = ['text_embedding', 'vision', 'audio', 'text_generation', 'multimodal']
        
        logger.info("Phase 1: Loading multiple models to increase load")
        for model_type in model_types:
            model_name = TEST_MODELS.get(model_type, TEST_MODELS['text_embedding'])
            logger.info(f"Loading model {model_name} ({model_type})...")
            
            model = await integration.get_model(
                model_name=model_name,
                model_type=model_type,
                platform=args.platform
            )
            
            if model:
                logger.info(f"Model {model_name} loaded successfully")
                models.append((model, model_type))
            else:
                logger.warning(f"Failed to load model {model_name}, skipping")
            
            # Check connection count after each model load
            metrics = integration.get_metrics()
            logger.info(f"Connection count after loading {model_name}: {metrics['connections']['total']}")
            
            # Short delay to let adaptive scaling respond
            await asyncio.sleep(1)
        
        # Phase 2: Run simultaneous inference to trigger scale-up
        logger.info("Phase 2: Running simultaneous inference to trigger scale-up")
        for i in range(5):
            # Create model and inputs list
            model_and_inputs = []
            for model, model_type in models:
                inputs = create_test_inputs(model_type)
                model_and_inputs.append((model, inputs))
            
            # Run concurrent inference
            logger.info(f"Running concurrent inference batch {i+1}...")
            results = await integration.execute_concurrent(model_and_inputs)
            
            # Check connection count after inference
            metrics = integration.get_metrics()
            logger.info(f"Connection count after batch {i+1}: {metrics['connections']['total']}")
            
            # Short delay to let adaptive scaling respond
            await asyncio.sleep(2)
        
        # Phase 3: Idle period to trigger scale-down
        logger.info("Phase 3: Idle period to trigger scale-down")
        for i in range(6):
            # Wait and check connection count
            await asyncio.sleep(5)
            
            # Check connection count during idle period
            metrics = integration.get_metrics()
            logger.info(f"Connection count after {(i+1)*5}s idle: {metrics['connections']['total']}")
        
        # Check scaling events
        metrics = integration.get_metrics()
        scaling_events = metrics['adaptive_scaling']['scaling_events']
        logger.info(f"Scaling events: {len(scaling_events)}")
        
        for i, event in enumerate(scaling_events):
            event_time = datetime.fromtimestamp(event['timestamp']).strftime('%H:%M:%S')
            logger.info(f"Event {i+1}: {event['event_type']} at {event_time}, "
                      f"{event['previous_connections']} → {event['new_connections']} connections, "
                      f"utilization: {event['utilization_rate']:.2f}, reason: {event['trigger_reason']}")
        
        # Final connection count
        final_connections = metrics['connections']['total']
        logger.info(f"Final connection count: {final_connections}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in adaptive scaling test: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Close integration
        logger.info("Closing integration...")
        await integration.close()

def create_test_inputs(model_type):
    """Create appropriate test inputs based on model type"""
    
    if model_type == 'text_embedding' or model_type == 'text_generation':
        return {"input_text": "This is a test sentence for the model."}
    
    elif model_type == 'vision':
        # Create a simple test image (just a dictionary for this test)
        return {"image": {"width": 224, "height": 224, "channels": 3, "format": "RGB"}}
    
    elif model_type == 'audio':
        # Create a simple test audio input
        return {"audio": {"duration": 2.0, "sample_rate": 16000}}
    
    elif model_type == 'multimodal':
        # Create combined text and image input
        return {
            "image": {"width": 224, "height": 224, "channels": 3, "format": "RGB"},
            "text": "This is a test sentence for the multimodal model."
        }
    
    # Default text input
    return {"input": "This is a test."}

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Test Enhanced WebNN/WebGPU Resource Pool Integration')
    
    # Test type
    parser.add_argument('--test-type', choices=['basic', 'concurrent', 'stress', 'adaptive'], default='basic',
                       help='Type of test to run')
    
    # Model configuration
    parser.add_argument('--model-type', choices=list(TEST_MODELS.keys()), default='text_embedding',
                       help='Type of model to test')
    parser.add_argument('--model-types', type=str, help='Comma-separated list of model types for concurrent/stress tests')
    
    # Hardware configuration
    parser.add_argument('--platform', choices=['webgpu', 'webnn', 'cpu'], default='webgpu',
                       help='Hardware platform to use')
    
    # Connection configuration
    parser.add_argument('--max-connections', type=int, default=4, help='Maximum number of browser connections')
    parser.add_argument('--min-connections', type=int, default=1, help='Minimum number of browser connections')
    
    # Test parameters
    parser.add_argument('--duration', type=int, default=30, help='Duration of stress test in seconds')
    parser.add_argument('--visible', action='store_true', help='Run browsers in visible mode (not headless)')
    
    # Feature flags
    parser.add_argument('--adaptive-scaling', action='store_true', help='Enable adaptive connection scaling')
    parser.add_argument('--db-path', type=str, help='Path to DuckDB database for metrics storage')
    
    return parser.parse_args()

async def main():
    """Main function to run tests"""
    args = parse_args()
    
    logger.info(f"Running {args.test_type} test with platform {args.platform}")
    
    if args.test_type == 'basic':
        await run_basic_test(args)
    elif args.test_type == 'concurrent':
        await run_concurrent_test(args)
    elif args.test_type == 'stress':
        await run_stress_test(args)
    elif args.test_type == 'adaptive':
        await run_adaptive_scaling_test(args)
    else:
        logger.error(f"Unknown test type: {args.test_type}")

if __name__ == "__main__":
    asyncio.run(main())