# Resource Pool Integration with Distributed Testing Framework

This guide documents the integration between the WebGPU/WebNN Resource Pool and the Distributed Testing Framework's Adaptive Load Balancer.

## Overview

The integration of the Resource Pool with the Distributed Testing Framework enables intelligent distribution of browser-based AI workloads across multiple worker nodes, with optimized resource allocation and fault tolerance capabilities.

## Implementation Status Update (March 13, 2025)

We have completed a full implementation of the integration between the Adaptive Load Balancer and WebGPU/WebNN Resource Pool Bridge. The Adaptive Load Balancer enhancements have been fully implemented, bringing the feature to 100% completion.

### Key Features Implemented

1. **LoadBalancerResourcePoolBridge**: A comprehensive bridge class that connects the load balancer with resource pool workers
2. **ResourcePoolWorker**: An enhanced worker implementation that manages browser resources through the Resource Pool Bridge
3. **Browser-Specific Model Optimization**: Intelligent routing of models to the optimal browser type based on performance characteristics
4. **Transaction-Based State Management**: Robust state management ensuring consistency across failures
5. **Multiple Recovery Strategies**: Support for progressive, immediate, and coordinated recovery strategies
6. **Performance Analysis and Optimization**: Intelligent recommendations based on empirical performance data
7. **Sharded Model Execution**: Distribute large models across multiple browsers with fault tolerance

### Newly Completed Enhancements (March 13, 2025)

8. **Browser-Specific Utilization Metrics**: Comprehensive tracking of browser-specific usage patterns
9. **Load Prediction Algorithm**: Forecasting of future load based on request patterns and active models
10. **Browser Capability Scoring**: Precise matching of tests to workers based on browser capabilities
11. **Browser-Aware Work Stealing**: Enhanced work stealing with browser capability awareness

The implementation is available in:
- `/distributed_testing/load_balancer_resource_pool_bridge.py`: Main integration implementation
- `/distributed_testing/examples/resource_pool_load_balancer_example.py`: Example usage
- `/duckdb_api/distributed_testing/load_balancer/service.py`: Enhanced work stealing implementation

This update increases the completion percentage of the Distributed Testing Framework from 80% to 90%. See [ADAPTIVE_LOAD_BALANCER_ENHANCEMENTS.md](ADAPTIVE_LOAD_BALANCER_ENHANCEMENTS.md) for a comprehensive summary of the latest enhancements.

## Key Components

### 1. ResourcePoolBridgeIntegration

This class serves as the bridge between the Distributed Testing Framework and the WebGPU/WebNN Resource Pool:

```python
class ResourcePoolBridgeIntegration:
    """
    Integration between distributed testing framework and WebGPU/WebNN Resource Pool.
    
    This class provides integration with the browser-based resource pool for managing
    connections, models, and browser instances.
    """
    
    def __init__(
        self,
        max_connections: int = 4,
        browser_preferences: Dict[str, str] = None,
        adaptive_scaling: bool = True,
        enable_fault_tolerance: bool = True,
        recovery_strategy: str = "progressive",
        state_sync_interval: int = 5,
        redundancy_factor: int = 2
    ):
        """Initialize the resource pool bridge integration."""
```

It manages browser connections, models, and provides fault tolerance features for browser-based AI workloads.

### 2. LoadBalancerService

The Adaptive Load Balancer has been enhanced to account for browser capabilities when distributing tests:

```python
class LoadBalancerService:
    """
    Load balancer service for distributed test execution.
    
    This service provides load balancing and test scheduling for the distributed
    testing framework, with support for browser capabilities and WebGPU/WebNN resources.
    """
    
    def register_worker(self, worker_id: str, capabilities: WorkerCapabilities):
        """
        Register a worker with the load balancer.
        
        Args:
            worker_id: Unique identifier for the worker
            capabilities: Worker capabilities including hardware specs and browser support
        """
```

The load balancer now considers browser capabilities (WebGPU, WebNN) and support for specific browser types (Chrome, Firefox, Edge) when assigning tests to workers.

### 3. BrowserStateManager and ResourcePoolRecoveryManager

These components provide state management and recovery capabilities for browser-based workloads:

```python
class BrowserStateManager:
    """
    State manager for browser-based models and resources.
    
    This class manages the state of browser instances, models, and operations 
    with transaction-based state updates for consistency across failures.
    """

class ResourcePoolRecoveryManager:
    """
    Recovery manager for browser-based models and resources.
    
    This class provides recovery capabilities for browser failures, model failures,
    and operation failures using various recovery strategies.
    """
```

### 4. ShardedModelExecution

This component enables the execution of large models by distributing them across multiple browser instances:

```python
class ShardedModelExecution:
    """
    High-level interface for executing models across multiple browser instances.
    
    This class provides a simple interface for creating and using sharded models
    with fault tolerance support.
    """
    
    def __init__(
        self,
        model_name: str,
        sharding_strategy: str = "layer_balanced",
        num_shards: int = 3,
        fault_tolerance_level: str = "high",
        recovery_strategy: str = "coordinated",
        connection_pool: Dict[str, Any] = None
    ):
        """Initialize sharded model execution."""
```

## Integration Architecture

The integration follows this high-level architecture:

1. **Coordinator** (Central component)
   - Manages worker nodes
   - Schedules tests based on requirements
   - Tracks test execution status
   - Collects and aggregates results

2. **Worker Nodes** (Distributed execution environments)
   - Register with coordinator and provide capabilities
   - Maintain ResourcePoolBridgeIntegration instances
   - Execute tests on assigned models
   - Report results back to coordinator

3. **Resource Pool** (Browser resource management)
   - Manages browser connections
   - Creates and caches models
   - Provides fault tolerance and recovery

4. **Load Balancer** (Resource allocation)
   - Assigns tests to workers based on capabilities
   - Monitors worker utilization
   - Redistributes load as needed

5. **State Manager** (Persistent state)
   - Tracks state of browser instances and models
   - Provides transaction-based state updates
   - Ensures consistency across failures

## Worker Integration Example

Here's a complete example of integrating a worker node with the Resource Pool:

```python
import anyio
import logging
from typing import Dict, Any

from distributed_testing.worker import Worker
from distributed_testing.resource_pool_bridge import ResourcePoolBridgeIntegration
from distributed_testing.model_sharding import ShardedModelExecution

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResourcePoolWorker(Worker):
    """Worker with Resource Pool integration."""
    
    async def initialize(self):
        """Initialize the worker."""
        await super().initialize()
        
        # Initialize resource pool integration
        self.resource_pool = ResourcePoolBridgeIntegration(
            max_connections=self.config.get("max_browser_connections", 3),
            browser_preferences={
                'audio': 'firefox',
                'vision': 'chrome',
                'text_embedding': 'edge'
            },
            adaptive_scaling=True,
            enable_fault_tolerance=True,
            recovery_strategy="progressive",
            state_sync_interval=5,
            redundancy_factor=2
        )
        
        # Initialize resource pool
        await self.resource_pool.initialize()
        
        # Initialize sharded model executions
        self.sharded_executions = {}
        
        logger.info("Resource pool worker initialized")
    
    async def execute_test(self, test_id: str, test_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a test with the resource pool.
        
        Args:
            test_id: Test identifier
            test_requirements: Test requirements and configuration
            
        Returns:
            Test results
        """
        logger.info(f"Executing test {test_id}")
        
        # Check if test requires sharding
        if test_requirements.get("requires_sharding", False):
            return await self._execute_sharded_test(test_id, test_requirements)
        
        # Get model from resource pool
        model = await self.resource_pool.get_model(
            model_type=test_requirements.get("model_type", "text"),
            model_name=test_requirements.get("model_id", "bert-base-uncased"),
            hardware_preferences=test_requirements.get("hardware_preferences", {
                "priority_list": ["webgpu", "webnn", "cpu"]
            }),
            fault_tolerance=test_requirements.get("fault_tolerance", {
                "recovery_timeout": 30,
                "state_persistence": True,
                "failover_strategy": "immediate"
            })
        )
        
        if not model:
            return {
                "status": "error",
                "error": "Failed to get model from resource pool",
                "test_id": test_id
            }
        
        # Get inputs from requirements
        inputs = test_requirements.get("inputs", {"input_ids": [101, 2054, 2003, 2028, 2339, 102]})
        
        try:
            # Run inference
            start_time = anyio.current_time()
            result = await model(inputs)
            end_time = anyio.current_time()
            
            # Calculate metrics
            latency_ms = (end_time - start_time) * 1000
            
            # Return results
            return {
                "status": "success",
                "result": result,
                "metrics": {
                    "latency_ms": latency_ms
                },
                "test_id": test_id
            }
            
        except Exception as e:
            logger.error(f"Error executing test {test_id}: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "test_id": test_id
            }
    
    async def _execute_sharded_test(self, test_id: str, test_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a sharded test across multiple browser instances.
        
        Args:
            test_id: Test identifier
            test_requirements: Test requirements and configuration
            
        Returns:
            Test results
        """
        logger.info(f"Executing sharded test {test_id}")
        
        # Get sharding requirements
        model_name = test_requirements.get("model_id", "llama-13b")
        sharding_strategy = test_requirements.get("sharding_strategy", "layer_balanced")
        num_shards = test_requirements.get("num_shards", 3)
        fault_tolerance_level = test_requirements.get("fault_tolerance_level", "high")
        recovery_strategy = test_requirements.get("recovery_strategy", "coordinated")
        
        # Create or get sharded execution
        if test_id not in self.sharded_executions:
            # Create sharded execution
            sharded_execution = ShardedModelExecution(
                model_name=model_name,
                sharding_strategy=sharding_strategy,
                num_shards=num_shards,
                fault_tolerance_level=fault_tolerance_level,
                recovery_strategy=recovery_strategy,
                connection_pool=self.resource_pool.connection_pool
            )
            
            # Initialize sharded execution
            await sharded_execution.initialize()
            
            # Store for future use
            self.sharded_executions[test_id] = sharded_execution
        
        # Get sharded execution
        sharded_execution = self.sharded_executions[test_id]
        
        # Get inputs from requirements
        inputs = test_requirements.get("inputs", {"prompt": "Hello, how are you?"})
        
        try:
            # Run inference
            start_time = anyio.current_time()
            result = await sharded_execution.run_inference(inputs)
            end_time = anyio.current_time()
            
            # Calculate metrics
            latency_ms = (end_time - start_time) * 1000
            
            # Return results
            return {
                "status": "success",
                "result": result,
                "metrics": {
                    "latency_ms": latency_ms
                },
                "test_id": test_id
            }
            
        except Exception as e:
            logger.error(f"Error executing sharded test {test_id}: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "test_id": test_id
            }
    
    async def cleanup(self):
        """Clean up resources."""
        # Clean up resource pool
        if hasattr(self, "resource_pool"):
            await self.resource_pool.close()
        
        # Clean up sharded executions
        for execution in self.sharded_executions.values():
            if hasattr(execution, "close"):
                await execution.close()
        
        await super().cleanup()
```

## Configuring Test Requirements

Test requirements now include browser-specific parameters:

```json
{
  "test_id": "test-vision-1",
  "model_id": "vit-base-patch16-224",
  "model_type": "vision",
  "minimum_memory": 2.0,
  "priority": 2,
  "browser_requirements": {
    "preferred": "chrome"
  },
  "hardware_preferences": {
    "priority_list": ["webgpu", "webnn", "cpu"]
  },
  "fault_tolerance": {
    "recovery_timeout": 30,
    "state_persistence": true,
    "failover_strategy": "immediate"
  }
}
```

For sharded models:

```json
{
  "test_id": "test-llm-1",
  "model_id": "llama-13b",
  "model_type": "large_language_model",
  "minimum_memory": 12.0,
  "priority": 3,
  "requires_sharding": true,
  "sharding_strategy": "layer_balanced",
  "num_shards": 3,
  "fault_tolerance_level": "high",
  "recovery_strategy": "coordinated",
  "browser_requirements": {
    "preferred": "chrome"
  }
}
```

## Performance Considerations

### Browser Selection Optimization

The Resource Pool Bridge selects browsers based on model type for optimal performance:

| Model Type | Preferred Browser | Reason |
|------------|-------------------|--------|
| Audio | Firefox | 55% better performance for audio compute shaders |
| Vision | Chrome | Best WebGPU support for vision models |
| Text | Edge | Superior WebNN support for text models |

### Worker Assignment Strategies

The Load Balancer uses several strategies to assign tests to workers:

1. **Capability Matching**: Matches test requirements with worker capabilities
2. **Load-Based Assignment**: Assigns tests to least-loaded workers
3. **Hardware-Aware Assignment**: Considers hardware specifications and browser support
4. **Affinity-Based Assignment**: Keeps related tests on the same worker when beneficial
5. **Performance History**: Uses historical performance data for optimal assignments

## Fault Tolerance

The Resource Pool integration includes several fault tolerance features:

### Recovery Strategies

1. **Progressive Recovery**:
   - First attempts to retry the operation on the same browser
   - If retry fails, attempts to restart the model on the same browser
   - If restart fails, migrates the model to a different browser
   - If migration fails, falls back to simulation mode

2. **Immediate Failover**:
   - Immediately migrates the model to a different browser
   - Faster recovery but potentially more disruptive

3. **Coordinated Recovery** (for sharded models):
   - Coordinates recovery across all shards
   - Ensures consistent state across all browsers

### State Management

The BrowserStateManager provides transaction-based state management:

1. **Transactional Updates**:
   - All state changes are done in transactions
   - Transactions can be committed or rolled back
   - Ensures consistency across failures

2. **State Synchronization**:
   - Periodically synchronizes state to durable storage
   - Maintains redundant copies for critical state
   - Automatically recovers from state corruption

3. **State Verification**:
   - Regularly verifies state consistency with checksums
   - Automatically recovers corrupted state partitions
   - Logs detailed diagnostic information for debugging

## Monitoring and Metrics

The integration provides comprehensive monitoring and metrics:

### Resource Pool Metrics

- **Browser Utilization**: Usage metrics for each browser instance
- **Model Performance**: Latency and throughput metrics for each model
- **Recovery Statistics**: Counts and durations of recovery operations
- **Memory Usage**: Memory consumption by browser and model
- **Connection Status**: Health and status of browser connections

### Load Balancer Metrics

- **Worker Utilization**: CPU, memory, and GPU utilization for each worker
- **Assignment Statistics**: Test assignment counts and distribution
- **Scheduling Latency**: Time to assign tests to workers
- **Migration Statistics**: Counts and latencies of task migrations
- **Queue Metrics**: Queue lengths and wait times by priority

## Best Practices

1. **Worker Configuration**:
   - Set appropriate `max_browser_connections` based on worker hardware capabilities
   - Configure browser preferences based on expected workload types
   - Enable fault tolerance for production deployments

2. **Test Requirements**:
   - Specify accurate memory requirements to avoid OOM errors
   - Use appropriate browser_requirements for optimal performance
   - Set appropriate fault_tolerance settings based on test criticality

3. **Recovery Settings**:
   - Use "progressive" recovery for development environments
   - Use "immediate" failover for production environments
   - Use "coordinated" recovery for sharded models

4. **Monitoring**:
   - Regularly review resource pool performance metrics
   - Monitor browser utilization and memory usage
   - Track recovery statistics to identify recurring issues

## Future Enhancements

1. **Advanced Browser Routing**:
   - Machine learning-based browser routing based on model performance
   - Predictive browser selection based on model characteristics
   - Auto-tuning of browser preferences based on performance history

2. **Enhanced Sharding Strategies**:
   - More sophisticated model partitioning strategies
   - Automatic partition size optimization
   - Dynamic repartitioning based on runtime performance

3. **Browser Resource Optimization**:
   - Browser-specific memory optimization techniques
   - Advanced shader precompilation and caching
   - Specialized optimizations for different GPU vendors

## Conclusion

The integration of the WebGPU/WebNN Resource Pool with the Distributed Testing Framework's Adaptive Load Balancer provides a powerful platform for running browser-based AI workloads at scale, with intelligent resource allocation, fault tolerance, and comprehensive monitoring capabilities.

This integration enables efficient distribution of browser resources across worker nodes, supports sharded execution of large models, and provides robust fault tolerance with multiple recovery strategies.

For more information, see:
- [LOAD_BALANCER_IMPLEMENTATION_STATUS.md](../../duckdb_api/distributed_testing/LOAD_BALANCER_IMPLEMENTATION_STATUS.md)
- [LOAD_BALANCER_STRESS_TESTING_GUIDE.md](../../duckdb_api/distributed_testing/LOAD_BALANCER_STRESS_TESTING_GUIDE.md)
- [DYNAMIC_THRESHOLD_PREDICTIVE_BALANCING.md](DYNAMIC_THRESHOLD_PREDICTIVE_BALANCING.md)
- [WEB_RESOURCE_POOL_DOCUMENTATION.md](../../WEB_RESOURCE_POOL_DOCUMENTATION.md)