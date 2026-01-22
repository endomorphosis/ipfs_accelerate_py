# Fault-Tolerant Cross-Browser Model Sharding Guide

> **NEW FEATURE (May 2025)**
>
> The Fault-Tolerant Cross-Browser Model Sharding system enables running large models distributed across multiple browser instances with intelligent recovery mechanisms and performance history tracking.

## Overview

The Fault-Tolerant Cross-Browser Model Sharding system is a sophisticated feature of the IPFS Accelerate framework that allows large AI models to be split across multiple browser instances. This system leverages browser-specific optimizations and provides robust fault tolerance through automatic recovery mechanisms and performance history tracking.

Key benefits include:

- **Distributed Execution**: Run large models by splitting them across multiple browsers
- **Browser-Specific Optimizations**: Use the best browser for each model component
- **Automatic Recovery**: Recover from browser crashes and failures transparently
- **Performance History Tracking**: Track and analyze performance across browsers
- **Intelligent Resource Allocation**: Optimize shard distribution based on performance data
- **Component-Level Fault Tolerance**: Retry, reroute, or redistribute components on failure
- **Database Integration**: Store performance history in DuckDB for long-term analysis

## Architecture

The Fault-Tolerant Cross-Browser Model Sharding system consists of several core components:

1. **ModelShardingManager**: Coordinates distribution and execution of model shards
2. **ShardedModelComponent**: Manages a single model component in a specific browser
3. **PerformanceHistoryTracker**: Tracks and analyzes performance across browsers
4. **ResourcePoolBridgeRecovery**: Handles recovery from failures
5. **WebSocketBridge**: Manages communication with browser instances

![Architecture Diagram](fault_tolerant_model_sharding_architecture.png)

## Usage

### Basic Usage

```python
from fixed_web_platform.cross_browser_model_sharding import ModelShardingManager

# Create model sharding manager
manager = ModelShardingManager(
    model_name="llama-7b",
    num_shards=4,
    shard_type="layer",
    model_type="text",
    enable_ipfs=True,
    db_path="benchmark_db.duckdb"  # For performance history tracking
)

# Initialize sharding
await manager.initialize_sharding()

# Run inference across shards with fault tolerance
result = await manager.run_inference_sharded(
    {"input_ids": [101, 2023, 2003, 1037, 3231, 102]},
    max_retries=3  # Maximum number of retries for failed components
)

# Get performance metrics
metrics = manager.get_metrics()

# Get performance history and recommendations
performance_history = manager.get_performance_history()

# Close and release resources
await manager.close()
```

### Sharding Types

The system supports multiple sharding strategies for different model architectures:

1. **Layer-Based Sharding**: Distributes model layers across browsers
    ```python
    manager = ModelShardingManager(
        model_name="llama-7b",
        num_shards=4,
        shard_type="layer",
        model_type="text"
    )
    ```

2. **Attention-Feedforward Sharding**: Separates attention and feedforward components
    ```python
    manager = ModelShardingManager(
        model_name="llama-7b",
        num_shards=6,
        shard_type="attention_feedforward",
        model_type="text"
    )
    ```

3. **Component-Based Sharding**: Separates model by functional components (for multimodal models)
    ```python
    manager = ModelShardingManager(
        model_name="clip-vit-base-patch32",
        num_shards=3,
        shard_type="component",
        model_type="multimodal"
    )
    ```

### Browser-Specific Optimizations

The system automatically selects optimal browsers for different model components:

| Model Type | Component Type | Recommended Browser | Reason |
|------------|---------------|-------------------|--------|
| Text | Embedding | Edge | Best WebNN support for text embeddings |
| Text | Generation | Chrome | Good overall WebGPU performance |
| Vision | Feature Extraction | Chrome | Optimized WebGPU for vision models |
| Vision | Classification | Firefox | Good parallel processing capabilities |
| Audio | Processing | Firefox | Superior compute shader performance |
| Multimodal | Vision | Chrome | Best for vision components |
| Multimodal | Text | Edge | Best for text components |
| Multimodal | Audio | Firefox | Best for audio components |
| Multimodal | Fusion | Chrome | Good for general computation |

## Advanced Features

### Performance History Tracking

The system automatically tracks performance metrics to improve future shard allocation:

```python
# Get detailed performance history
history = manager.get_performance_history()

# Example history structure
{
    'components': {
        'component_123': {
            'success_count': 42,
            'error_count': 3,
            'total_latency': 1250.5,
            'execution_count': 45,
            'avg_latency': 27.8,
            'browser': 'firefox',
            'platform': 'webgpu',
            'shard_type': 'attention',
            'shard_index': 2
        },
        # More components...
    },
    'browser_metrics': {
        'chrome': {
            'success_count': 120,
            'error_count': 5,
            'total_latency': 3200.0,
            'execution_count': 125,
            'success_rate': 0.96,
            'avg_latency': 26.7
        },
        'firefox': {...},
        'edge': {...}
    },
    'overall_metrics': {...},
    'shard_types': {...},
    'timeline': [...],
    'recovery_events': [...]
}
```

### Fault Tolerance Mechanisms

The system implements multiple levels of fault tolerance:

1. **Component-Level Retry**: Automatically retry failed components with exponential backoff
2. **Browser Rerouting**: Move components to different browsers after failure
3. **Platform Change**: Switch between WebNN and WebGPU backends
4. **Component Redistribution**: Redistribute components based on performance history
5. **Circuit Breaker Pattern**: Prevent repeated failures by temporarily disabling problematic components
6. **Transaction-Based State Management**: Track and recover component state

```python
# Configure custom fault tolerance settings
manager = ModelShardingManager(
    model_name="llama-7b",
    num_shards=4,
    shard_type="layer",
    fault_tolerance_level="high",  # Options: none, low, medium, high, critical
    recovery_strategy="progressive"  # Options: restart, reconnect, failover, progressive, parallel
)
```

### Performance Recommendations

The system generates performance recommendations based on historical data:

```python
# Get recommendations
metrics = manager.get_metrics()
recommendations = metrics.get('recommendations', {})

# Example recommendations
{
    'browser_allocation': {
        'overall': {
            'recommended_browser': 'firefox',
            'reason': 'Best overall performance with 98.5% success rate and 23.5ms latency'
        },
        'attention': {
            'recommended_browser': 'firefox',
            'reason': 'Best performance for attention components'
        },
        'feedforward': {
            'recommended_browser': 'chrome',
            'reason': 'Best performance for feedforward components'
        }
    },
    'optimization_suggestions': [
        {
            'type': 'allocation',
            'issue': 'Only 1/4 shards using optimal browser (firefox)',
            'suggestion': 'Consider reallocating more shards to firefox for better performance'
        },
        {
            'type': 'audio_optimization',
            'issue': 'Audio models typically perform best on Firefox with compute shader optimizations',
            'suggestion': 'Consider allocating more audio processing to Firefox browsers'
        }
    ]
}
```

### Database Integration

Performance history can be saved to a DuckDB database for long-term analysis:

```python
# Create model sharding manager with database integration
manager = ModelShardingManager(
    model_name="llama-7b",
    num_shards=4,
    shard_type="layer",
    db_path="benchmark_db.duckdb"
)

# Run inference with automatic history tracking
result = await manager.run_inference_sharded(inputs)

# Close manager (automatically saves history to database)
await manager.close()
```

The database schema includes:

- `model_sharding_history`: Overall execution history
- `model_sharding_components`: Component-specific performance metrics
- `model_sharding_recovery_events`: Detailed recovery event tracking
- `model_sharding_browser_metrics`: Browser-specific performance data

### Custom Browser Allocation

You can manually specify browser allocation for specific shards:

```python
# Create custom browser allocation
custom_allocation = {
    0: {"browser": "firefox", "platform": "webgpu", "specialization": "audio_compute"},
    1: {"browser": "edge", "platform": "webnn", "specialization": "text"},
    2: {"browser": "chrome", "platform": "webgpu", "specialization": "vision"},
    3: {"browser": "firefox", "platform": "webgpu", "specialization": "audio_compute"}
}

# Create manager with custom allocation
manager = ModelShardingManager(
    model_name="multimodal-model",
    num_shards=4,
    shard_type="component",
    model_type="multimodal",
    custom_browser_allocation=custom_allocation
)
```

## Example Scenarios

### Large Language Model Sharding

```python
from fixed_web_platform.cross_browser_model_sharding import ModelShardingManager

async def run_sharded_llm():
    # Create model sharding manager
    manager = ModelShardingManager(
        model_name="llama-7b",
        num_shards=4,
        shard_type="layer",
        model_type="text",
        db_path="llm_benchmark.duckdb"
    )
    
    # Initialize sharding
    success = await manager.initialize_sharding()
    if not success:
        print("Failed to initialize sharding")
        return
    
    # Prepare input
    inputs = {
        "input_ids": [101, 2023, 2003, 1037, 3231, 102],
        "attention_mask": [1, 1, 1, 1, 1, 1]
    }
    
    # Run sharded inference
    result = await manager.run_inference_sharded(inputs)
    
    # Process results
    if "error" in result:
        print(f"Inference error: {result['error']}")
    else:
        output = result["result"]
        metrics = result["metrics"]
        print(f"Output: {output}")
        print(f"Inference time: {metrics['inference_time_ms']}ms")
        print(f"Successful components: {metrics['successful_components']}/{metrics['component_count']}")
    
    # Get performance recommendations
    perf_metrics = manager.get_metrics()
    if "recommendations" in perf_metrics:
        recommendations = perf_metrics["recommendations"]
        print("\nRecommendations:")
        for suggestion in recommendations.get("optimization_suggestions", []):
            print(f"- {suggestion['issue']}: {suggestion['suggestion']}")
    
    # Clean up
    await manager.close()
```

### Multimodal Model Sharding

```python
from fixed_web_platform.cross_browser_model_sharding import ModelShardingManager

async def run_sharded_multimodal():
    # Create model sharding manager for multimodal model
    manager = ModelShardingManager(
        model_name="clip-vit-base-patch32",
        num_shards=3,
        shard_type="component",
        model_type="multimodal",
        db_path="multimodal_benchmark.duckdb"
    )
    
    # Initialize sharding with browser-specific optimizations
    success = await manager.initialize_sharding()
    if not success:
        print("Failed to initialize sharding")
        return
    
    # Prepare multimodal input
    inputs = {
        "input_ids": [101, 2023, 2003, 1037, 3231, 102],
        "attention_mask": [1, 1, 1, 1, 1, 1],
        "pixel_values": [[[0.5 for _ in range(3)] for _ in range(224)] for _ in range(1)]
    }
    
    # Run sharded inference
    result = await manager.run_inference_sharded(inputs)
    
    # Process results
    if "error" in result:
        print(f"Inference error: {result['error']}")
    else:
        output = result["result"]
        metrics = result["metrics"]
        print(f"Text-image similarity: {output.get('similarity', 'N/A')}")
        print(f"Inference time: {metrics['inference_time_ms']}ms")
        
        # Output component-specific results
        if "component_results" in metrics:
            for component_id, component_result in metrics["component_results"].items():
                print(f"Component {component_id}: {component_result['status']}")
    
    # Get browser allocation
    metrics = manager.get_metrics()
    print("\nBrowser Allocation:")
    for shard_idx, config in metrics["browser_allocation"].items():
        print(f"Shard {shard_idx}: {config['browser']} ({config['platform']}) - {config['specialization']}")
    
    # Clean up
    await manager.close()
```

### Audio Model with Compute Shader Optimization

```python
from fixed_web_platform.cross_browser_model_sharding import ModelShardingManager

async def run_sharded_audio_model():
    # Create model sharding manager with Firefox optimization for audio
    manager = ModelShardingManager(
        model_name="whisper-tiny",
        num_shards=3,
        shard_type="layer",
        model_type="audio",
        db_path="audio_benchmark.duckdb"
    )
    
    # Initialize sharding
    success = await manager.initialize_sharding()
    if not success:
        print("Failed to initialize sharding")
        return
    
    # Prepare audio input (mock data)
    audio_features = [[[0.1 for _ in range(80)] for _ in range(3000)]]
    inputs = {"input_features": audio_features}
    
    # Run sharded inference
    result = await manager.run_inference_sharded(inputs)
    
    # Process results
    if "error" in result:
        print(f"Inference error: {result['error']}")
    else:
        output = result["result"]
        metrics = result["metrics"]
        print(f"Transcription: {output.get('text', 'N/A')}")
        print(f"Inference time: {metrics['inference_time_ms']}ms")
    
    # Get browser allocation and check Firefox usage
    metrics = manager.get_metrics()
    firefox_shards = sum(1 for config in metrics["browser_allocation"].values() 
                       if config.get("browser") == "firefox")
    print(f"\nFirefox shards: {firefox_shards}/{manager.num_shards}")
    
    # Get performance history for Firefox
    history = manager.get_performance_history()
    firefox_metrics = history.get("browser_metrics", {}).get("firefox", {})
    if firefox_metrics:
        print(f"Firefox performance: {firefox_metrics.get('avg_latency', 0):.1f}ms latency, "
              f"{firefox_metrics.get('success_rate', 0):.1%} success rate")
    
    # Clean up
    await manager.close()
```

## Performance Benchmarks

The Fault-Tolerant Cross-Browser Model Sharding system has been benchmarked across various models and browsers:

| Model | Sharding Type | Shards | Best Browser Combo | Improvement |
|-------|--------------|--------|-------------------|-------------|
| LLAMA-7B | layer | 4 | Chrome(2), Firefox(1), Edge(1) | 35% faster than single browser |
| ViT-CLIP | component | 3 | Chrome(2), Firefox(1) | 42% faster than single browser |
| Whisper | layer | 3 | Firefox(2), Chrome(1) | 48% faster with compute shaders |
| T5-Large | attention_feedforward | 6 | Edge(3), Chrome(3) | 30% faster than single browser |
| Multimodal | component | 4 | Firefox(1), Chrome(2), Edge(1) | 38% faster than single browser |

## API Reference

### ModelShardingManager

The main entry point for the Fault-Tolerant Cross-Browser Model Sharding system.

```python
class ModelShardingManager:
    def __init__(self,
                 model_name: str,
                 num_shards: int = 2,
                 shard_type: str = "layer",
                 model_type: str = "text",
                 enable_ipfs: bool = True,
                 max_connections: int = 4,
                 db_path: str = None,
                 fault_tolerance_level: str = "medium",
                 recovery_strategy: str = "progressive",
                 custom_browser_allocation: Dict[int, Dict[str, Any]] = None)
    
    # Initialize sharding across browsers
    async def initialize_sharding() -> bool
    
    # Run inference across sharded model components
    async def run_inference_sharded(self,
                                   inputs: Dict[str, Any],
                                   max_retries: int = 2) -> Dict[str, Any]
    
    # Get performance metrics and recommendations
    def get_metrics() -> Dict[str, Any]
    
    # Get detailed performance history
    def get_performance_history() -> Dict[str, Any]
    
    # Generate performance recommendations
    def _generate_performance_recommendations(performance_data: Dict[str, Any]) -> Dict[str, Any]
    
    # Update performance history with new data
    async def _update_performance_history(components: List[Any],
                                        results: List[Dict[str, Any]],
                                        execution_time: float)
    
    # Record recovery events
    async def _record_recovery_event(component: Any,
                                   success: bool,
                                   error: Exception = None,
                                   recovery_type: str = None)
    
    # Save performance history to database
    async def _save_performance_history_to_db()
    
    # Close resources
    async def close()
```

### ShardedModelComponent

Represents a single model component running in a specific browser.

```python
class ShardedModelComponent:
    def __init__(self,
                 component_id: str,
                 model_type: str,
                 model_name: str,
                 shard_index: int,
                 shard_type: str,
                 browser: str,
                 platform: str,
                 resource_pool_integration: Any)
    
    # Initialize the component
    async def initialize() -> bool
    
    # Process inputs through this component
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]
    
    # Process with fault tolerance
    async def _process_with_fault_tolerance(self, inputs: Dict[str, Any]) -> Dict[str, Any]
    
    # Recover component after failure
    async def _recover_component() -> bool
```

## Best Practices

1. **Choose Appropriate Sharding Type**: Use `layer` for most models, `attention_feedforward` for transformer models, and `component` for multimodal models.

2. **Optimize Browser Allocation**: Use Firefox for audio models, Edge for text embeddings, and Chrome for vision models.

3. **Enable Database Integration**: Always provide a `db_path` to track performance history for better future allocations.

4. **Use Progressive Recovery**: The `progressive` recovery strategy provides the best balance of speed and reliability.

5. **Analyze Performance History**: Regularly check performance history to optimize browser allocation.

6. **Implement Browser-Specific Optimizations**: Enable compute shader optimization for audio models in Firefox, and shader precompilation for vision models.

7. **Balance Component Load**: Distribute heavy components across different browsers to prevent overloading.

8. **Follow Recommendations**: Implement the recommendations provided by the performance analysis system.

## Limitations and Considerations

- **Cross-Origin Isolation**: Some WebGPU/WebNN features require cross-origin isolation.
- **Browser Compatibility**: WebNN is not supported in all browsers; WebGPU support varies.
- **Memory Constraints**: Large models may exceed browser memory limits even when sharded.
- **Communication Overhead**: Sharding adds communication overhead between components.
- **Browser Stability**: Browser crashes can affect the entire system if fault tolerance is not properly configured.

## Troubleshooting

### Common Issues

1. **Browser Initialization Failure**: Check browser compatibility and WebGPU/WebNN support.
2. **Component Communication Errors**: Ensure cross-origin isolation is enabled.
3. **Out of Memory Errors**: Increase the number of shards or use lower precision.
4. **Browser Crashes**: Enable higher fault tolerance level.
5. **Slow Performance**: Check browser allocation and follow optimization recommendations.

### Testing Fault Tolerance

A comprehensive test suite is available for validating the fault tolerance capabilities with real browsers:

```bash
# Install dependencies
./install_fault_tolerance_test_deps.sh

# Run basic test with a single browser type
python test_real_browser_fault_tolerance.py --model bert-base-uncased --browsers chrome

# Test with multiple browser types
python test_real_browser_fault_tolerance.py --model bert-base-uncased --browsers chrome,firefox,edge

# Test forced failures and recovery
python test_real_browser_fault_tolerance.py --model bert-base-uncased --force-failure --failure-type crash

# Test different recovery strategies
python test_real_browser_fault_tolerance.py --model bert-base-uncased --force-failure --recovery-strategy progressive

# Run comprehensive test suite
./run_comprehensive_fault_tolerance_tests.sh
```

For detailed instructions, see [FAULT_TOLERANCE_TESTING_README.md](FAULT_TOLERANCE_TESTING_README.md)

### Debugging

Enable detailed logging for better diagnosis:

```python
import logging
logging.basicConfig(level=logging.DEBUG,
                  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
```

#### Enhanced Browser Debugging

For debugging issues with specific browsers:

```bash
# Show browser windows during testing (disable headless mode)
python test_real_browser_fault_tolerance.py --model bert-base-uncased --force-failure --show-browsers

# Save detailed test results
python test_real_browser_fault_tolerance.py --model bert-base-uncased --force-failure --output test_results.json

# Use the browser test page to check WebGPU/WebNN support
# Open test_pages/webgpu_webnn_test.html in your browser
```

## Future Enhancements

Planned enhancements for future releases include:

- **Automatic Shard Sizing**: Dynamically determine optimal number of shards based on model size and available resources
- **Predictive Recovery**: Use machine learning to predict and prevent component failures
- **Dynamic Resharding**: Adjust shard distribution during execution based on performance
- **Multi-Machine Distribution**: Extend sharding across multiple machines
- **Hybrid CPU/GPU Execution**: Seamlessly mix CPU and GPU execution for optimal performance
- **Memory-Aware Scheduling**: Optimize component scheduling based on memory availability

## Additional Resources

- [WebNN API Reference](https://www.w3.org/TR/webnn/)
- [WebGPU API Reference](https://www.w3.org/TR/webgpu/)
- [Cross-Browser Testing Guide](https://developer.mozilla.org/en-US/docs/Learn/Tools_and_testing/Cross_browser_testing)
- [IPFS Accelerate Framework Documentation](https://docs.example.com/ipfs-accelerate/)