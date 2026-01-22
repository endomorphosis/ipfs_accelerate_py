# Advanced Memory Optimization for Web Platform Models

This guide covers advanced techniques for optimizing memory usage when running large machine learning models in web browsers. These strategies are particularly useful for running models larger than 1B parameters.

## Table of Contents

- [Understanding Browser Memory Constraints](#understanding-browser-memory-constraints)
- [Ultra-Low Precision Techniques](#ultra-low-precision-techniques)
- [Progressive Loading Strategies](#progressive-loading-strategies)
- [Memory-Efficient KV Cache](#memory-efficient-kv-cache)
- [Model Sharding Across Browser Tabs](#model-sharding-across-browser-tabs)
- [Memory Pressure Handling](#memory-pressure-handling)
- [Browser-Specific Memory Optimizations](#browser-specific-memory-optimizations)
- [Measuring and Monitoring Memory Usage](#measuring-and-monitoring-memory-usage)
- [Troubleshooting Memory Issues](#troubleshooting-memory-issues)

## Understanding Browser Memory Constraints

Web browsers have specific memory limitations that affect ML model deployment:

| Browser | Typical Memory Limit | Notes |
|---------|---------------------|-------|
| Chrome/Edge | ~4GB per tab on 64-bit systems | Can be higher on systems with more RAM |
| Firefox | ~4GB per tab on 64-bit systems | Often more efficient memory use than Chrome |
| Safari | ~2GB per tab on desktop | More restrictive, especially on mobile |
| Mobile Browsers | ~1-2GB per tab | Highly variable based on device |

Key constraints to consider:
- **JavaScript Heap Size**: Limited by browser and system memory
- **ArrayBuffer Size**: Limited to 2GB in many browsers
- **WebGPU Buffer Size**: Varies by GPU and browser implementation
- **Garbage Collection**: Unpredictable timing can affect performance

## Ultra-Low Precision Techniques

### 4-bit, 3-bit, and 2-bit Quantization

Our framework supports ultra-low precision to dramatically reduce memory footprint:

```python
from fixed_web_platform.webgpu_ultra_low_precision import configure_precision

# Configure mixed precision with ultra-low bits
precision_config = configure_precision(
    model_name="llama-7b",
    default_bits=2,         # 2-bit for most weights
    attention_bits=8,       # 8-bit for attention (more sensitive)
    feed_forward_bits=2,    # 2-bit for feed-forward
    embedding_bits=8,       # 8-bit for embeddings
    lm_head_bits=8,         # 8-bit for language model head
    adaptive=True           # Enable adaptive precision
)

# Initialize model with ultra-low precision
model = LargeLanguageModel("llama-7b")
result = init_webgpu(
    model=model,
    precision_config=precision_config
)
```

### Memory Savings by Precision

| Precision | Memory Reduction | Accuracy Impact | Recommended For |
|-----------|-----------------|-----------------|----------------|
| 8-bit | 50% | <1% | Critical layers, Safari |
| 4-bit | 75% | 1-3% | General use, balanced |
| 3-bit | 81.25% | 3-4% | Large models (3-7B) |
| 2-bit | 87.5% | 4-6% | Very large models (>7B) |

### Layer-Specific Precision Assignment

For optimal performance, use different precision for different layers:

```python
# Detailed layer-specific precision
layer_precision = {
    "embeddings": 8,         # Higher precision for embeddings
    "attention.query": 8,    # Higher precision for attention
    "attention.key": 4,      # Medium precision for key projections
    "attention.value": 4,    # Medium precision for value projections
    "attention.output": 8,   # Higher precision for attention output
    "mlp.up": 3,             # Lower precision for MLP up projection
    "mlp.down": 3,           # Lower precision for MLP down projection
    "mlp.gate": 3,           # Lower precision for MLP gate
    "lm_head": 8             # Higher precision for final projection
}

# Apply to precision configuration
precision_config = configure_precision(
    model_name="llama-7b", 
    layer_precision=layer_precision
)
```

## Progressive Loading Strategies

Progressive loading reduces initial memory footprint by loading model components on-demand:

```python
from fixed_web_platform.progressive_model_loader import ProgressiveModelLoader

# Create loader with priority configuration
loader = ProgressiveModelLoader(
    model_path="llama-7b",
    config={
        "priority_components": [
            "embeddings",      # Load first (needed immediately)
            "layers.0",        # First layer
            "lm_head"          # Output projections
        ],
        "background_loading": True,
        "unload_unused_layers": True,
        "memory_threshold_mb": 2000  # Start unloading when memory exceeds 2GB
    }
)

# Load critical components first
await loader.load_critical_components()

# Start background loading of other components
loader.start_background_loading()

# Use the model while other components load
model = loader.get_partial_model()
first_output = model.generate("Hello", max_tokens=5)

# Later, when more components are loaded:
full_output = model.generate("Hello", max_tokens=50)
```

### Memory-Efficient Component Management

For multimodal models, load and unload components based on active modality:

```python
from fixed_web_platform.progressive_model_loader import MultimodalComponentManager

# Create component manager
component_manager = MultimodalComponentManager(
    model_name="llava",
    components=["vision_encoder", "text_encoder", "cross_attention", "decoder"]
)

# Load specific components for image processing
await component_manager.load_components(["vision_encoder", "cross_attention"])

# Process image
image_embeddings = component_manager.process_image(image)

# Unload vision components and load text components
await component_manager.swap_modality(
    unload=["vision_encoder"],
    load=["text_encoder", "decoder"]
)

# Generate text based on image embeddings
response = component_manager.generate_text(image_embeddings, prompt)
```

## Memory-Efficient KV Cache

For long context generation, implement efficient KV cache with ultra-low precision:

```python
from fixed_web_platform.webgpu_kv_cache_optimization import KVCacheManager

# Create KV cache with adaptive precision
kv_cache = KVCacheManager(
    num_layers=32,
    num_heads=32,
    head_dim=128,
    max_seq_len=8192,
    config={
        "default_bits": 2,           # 2-bit cache for most tokens
        "recent_token_bits": 8,      # 8-bit for recent tokens
        "rolling_window": 1024,      # Only keep recent tokens in higher precision
        "prune_attn_threshold": 0.05 # Prune attention scores below this threshold
    }
)

# During generation, update KV cache with new token
kv_cache.update(layer_idx, query, key, value)

# Get cached KV for a specific position
k, v = kv_cache.get(layer_idx, position)
```

### KV Cache Pruning Strategies

For long context generation, implement cache pruning to maintain efficiency:

```python
# Configure KV cache pruning
kv_cache.configure_pruning(
    strategy="threshold",      # Prune based on attention score
    threshold=0.01,           # Remove entries below this score
    min_tokens_to_keep=512,   # Always keep at least this many tokens
    max_tokens_to_drop=0.5,   # Drop up to 50% of the cache when needed
    frequency_aware=True      # Consider token frequency in pruning decisions
)

# Execute pruning when memory pressure is detected
if memory_pressure_detected():
    tokens_pruned = kv_cache.prune()
    print(f"Pruned {tokens_pruned} tokens from KV cache")
```

## Model Sharding Across Browser Tabs

For extremely large models (>7B parameters), implement cross-tab model sharding:

```python
from fixed_web_platform.model_sharding import ModelShardingManager

# Configure model sharding across multiple tabs
sharding_manager = ModelShardingManager(
    model_name="llama-13b",
    config={
        "num_shards": 4,              # Split model across 4 tabs
        "sharding_type": "layer",     # Shard by layers
        "worker_urls": ["worker.html"],
        "coordinator_port": 8765,     # Port for coordination
        "memory_per_shard_mb": 2000   # Target memory per shard
    }
)

# Initialize sharding (opens browser tabs as workers)
await sharding_manager.initialize()

# Run inference across shards
output = await sharding_manager.generate(
    prompt="Explain WebGPU",
    max_tokens=100
)
```

## Memory Pressure Handling

Implement memory pressure detection and handling:

```python
from fixed_web_platform.memory_pressure_handler import MemoryPressureHandler

# Configure memory pressure handler
memory_handler = MemoryPressureHandler(
    thresholds={
        "warning": 0.7,      # 70% of available memory
        "critical": 0.85,    # 85% of available memory
        "emergency": 0.95    # 95% of available memory
    },
    actions={
        "warning": ["gc", "compress_tensors"],
        "critical": ["unload_unused_components", "reduce_precision", "prune_kv_cache"],
        "emergency": ["reduce_batch_size", "reduce_context_window", "abort_current_operation"]
    }
)

# Register components for memory monitoring
memory_handler.register_component(model, "model")
memory_handler.register_component(kv_cache, "kv_cache")

# Start monitoring memory and handle pressure events
memory_handler.start_monitoring(interval_ms=1000)

# When running operations that might cause memory pressure:
with memory_handler.operation_context("generation"):
    output = model.generate(prompt, max_tokens=1000)
```

## Browser-Specific Memory Optimizations

### Chrome/Edge

```python
if browser == "chrome" or browser == "edge":
    # Chrome/Edge specific optimizations
    os.environ["CHROME_MEMORY_HINTS"] = "1"  # Enable memory hints API if available
    
    # Use larger workgroup sizes on Chrome/Edge
    workgroup_config = {"x": 128, "y": 1, "z": 1}
    
    # Enable shared array buffer if available (helps with memory efficiency)
    shared_memory_available = capabilities["webassembly"]["threads"]
```

### Firefox

```python
if browser == "firefox":
    # Firefox specific memory optimizations
    
    # Use Firefox-optimized workgroup sizes for better memory utilization
    workgroup_config = {"x": 256, "y": 1, "z": 1}
    
    # Enable Firefox-specific memory optimizations
    os.environ["MOZ_MEMORY_OPTIMIZED"] = "1"
    
    # Firefox handles 2-bit tensors more efficiently
    recommended_bits = 2 if model_size > "7b" else 4
```

### Safari

```python
if browser == "safari":
    # Safari has more restricted memory, use higher precision but fewer layers
    
    # Use 8-bit precision for better numerical stability in Safari
    recommended_bits = 8
    
    # Use smaller workgroup sizes to avoid memory issues
    workgroup_config = {"x": 64, "y": 1, "z": 1}
    
    # Prioritize WebAssembly fallback for better memory management
    use_wasm_fallback = True
    
    # Enable aggressive progressive loading
    progressive_loading_config = {
        "load_threshold": 0.1,  # Load components gradually in smaller chunks
        "unload_threshold": 0.7,  # Unload components more aggressively
        "prioritize_current_layer": True  # Focus memory on active layer
    }
```

## Measuring and Monitoring Memory Usage

Integrate memory monitoring to track usage:

```python
from fixed_web_platform.memory_monitor import MemoryMonitor

# Create memory monitor
monitor = MemoryMonitor()

# Start monitoring with callback
monitor.start(
    interval_ms=2000,
    callback=lambda stats: print(f"Memory: {stats['used_mb']}MB/{stats['total_mb']}MB")
)

# Get detailed memory breakdown
memory_stats = monitor.get_detailed_stats()
print(f"JS Heap: {memory_stats['js_heap_mb']}MB")
print(f"WebGPU Buffers: {memory_stats['webgpu_buffers_mb']}MB")
print(f"Model Weights: {memory_stats['model_weights_mb']}MB")
print(f"KV Cache: {memory_stats['kv_cache_mb']}MB")

# Monitor specific model components
monitor.track_component(model.embeddings, "embeddings")
monitor.track_component(model.layers[0], "first_layer")
monitor.track_component(kv_cache, "kv_cache")

# Log memory timeline for analysis
monitor.start_logging("memory_log.json")

# Generate a visualization of memory usage
monitor.generate_report("memory_report.html")
```

## Troubleshooting Memory Issues

### Common Memory Issues and Solutions

| Issue | Symptoms | Solution |
|-------|----------|----------|
| JS Heap OOM | "Out of memory" error in browser console | Reduce precision, enable progressive loading |
| WebGPU Buffer Allocation Failure | "Failed to allocate buffer" error | Split buffers into smaller chunks, use lower precision |
| Slow Garbage Collection | Periodic freezes, growing memory usage | Manually trigger GC between operations, reduce object creation |
| KV Cache Growth | Memory grows with token generation | Implement KV cache pruning, use 2-bit KV cache |
| Multiple Copies of Tensors | Unexpectedly high memory usage | Check for accidental tensor duplication, use shared memory |

### Fixing Out-of-Memory Errors

If you encounter "Out of Memory" errors:

1. **Implement Ultra-Low Precision**
   ```python
   # Lower precision from 4-bit to 2-bit
   precision_config = configure_precision(
       model_name="llama-7b",
       default_bits=2,
       adaptive=True
   )
   ```

2. **Enable Component Unloading**
   ```python
   # Unload unused components
   loader.configure({
       "unload_unused_layers": True,
       "keep_n_layers_loaded": 4  # Only keep 4 layers in memory at once
   })
   ```

3. **Implement KV Cache Pruning**
   ```python
   # Prune KV cache when it gets too large
   kv_cache.enable_auto_pruning(
       max_tokens=2048,       # Maximum tokens to keep
       pruning_interval=512   # Prune every 512 tokens
   )
   ```

4. **Use Model Sharding**
   ```python
   # Split model across tabs if single tab memory is insufficient
   sharding_manager = ModelShardingManager(
       model_name="llama-13b",
       num_shards=4
   )
   ```

5. **Implement Memory Pressure Handling**
   ```python
   # Configure emergency actions for memory pressure
   memory_handler.configure_emergency_actions([
       "gc",                      # Force garbage collection
       "unload_unused_components", # Unload components not in use
       "reduce_batch_size",       # Reduce batch size dynamically
       "truncate_kv_cache",       # Truncate KV cache to recent tokens only
       "reduce_context_window"    # Reduce context window temporarily
   ])
   ```

For more detailed information, refer to the [Web Platform Integration Guide](../web_platform_integration_guide.md) and the [Model Performance Optimization Guide](/docs/model_performance_optimization_guide.md).