# IPFS Accelerate Python Framework - Development Guide

> **ORGANIZATION UPDATE (March 10, 2025):**
>
> The codebase has been reorganized for better maintainability with the following top-level structure:
> 
> - All generator files moved to the top-level `generators/` directory (216 files) with subdirectories:
>   - `generators/benchmark_generators/`: Benchmark generation tools
>   - `generators/models/`: Model implementations and skills
>   - `generators/runners/`: Test runner scripts
>   - `generators/skill_generators/`: Skill generation tools
>   - `generators/template_generators/`: Template generation utilities
>   - `generators/templates/`: Template files for model generation
>   - `generators/test_generators/`: Test generation tools
>   - `generators/utils/`: Utility functions
>   - `generators/hardware/`: Hardware-specific generator tools
>
> - All database-related tools moved to the top-level `duckdb_api/` directory (83 files) with subdirectories:
>   - `duckdb_api/core/`: Core database functionality
>   - `duckdb_api/migration/`: Migration tools for JSON to database
>   - `duckdb_api/schema/`: Database schema definitions 
>   - `duckdb_api/utils/`: Utility functions for database operations
>   - `duckdb_api/visualization/`: Result visualization tools
>   - `duckdb_api/distributed_testing/`: Distributed testing framework components
>
> - Web platform implementations remain in `fixed_web_platform/` directory with subdirectories:
>   - `fixed_web_platform/unified_framework/`: Unified API for cross-browser WebNN/WebGPU
>   - `fixed_web_platform/wgsl_shaders/`: WebGPU Shading Language shader implementations
>
> - CI/CD workflow files moved from `test/.github/workflows/` to the standard `.github/workflows/` location
>
> **UPCOMING MIGRATION (Q2-Q3 2025):**
> 
> All WebGPU/WebNN implementations will be moved from `/fixed_web_platform/` to a dedicated `ipfs_accelerate_js` folder once all tests pass. This migration will create a clearer separation between JavaScript-based components and Python-based components.
>
> The structure and contents of the `ipfs_accelerate_js` folder will maintain isomorphism with the `ipfs_accelerate_py` structure, ensuring consistent organization across both implementations. Key directories will include:
> - `ipfs_accelerate_js/core/`: Core JavaScript functionality matching Python counterparts
> - `ipfs_accelerate_js/models/`: Model implementations with feature parity
> - `ipfs_accelerate_js/utils/`: Utility functions with equivalent capabilities
> - `ipfs_accelerate_js/webgpu/`: WebGPU-specific optimizations
> - `ipfs_accelerate_js/webnn/`: WebNN-specific implementations
> - `ipfs_accelerate_js/shaders/`: WGSL shader implementations
> - `ipfs_accelerate_js/transformers/`: Integration with transformers.js
> - `ipfs_accelerate_js/api_backends/`: Client implementations for various model serving APIs (OpenAI, Hugging Face, etc.)
> - `ipfs_accelerate_js/resource_pool/`: Browser resource management with connection pooling
>
> The implementation will follow a phased approach:
> 1. **Phase 1 (June-July 2025)**: Core browser acceleration architecture (WebNN/WebGPU)
> 2. **Phase 2 (July-August 2025)**: Hardware backend optimizations and resource pooling
> 3. **Phase 3 (August-September 2025)**: API backends and advanced feature integration

## Current Focus Areas (Q2 2025):

- 🔄 **WebGPU/WebNN Resource Pool Integration** (IN PROGRESS - 85% complete)
  - Enables concurrent execution of multiple AI models across heterogeneous browser backends
  - Creates browser-aware load balancing for model type optimization
  - Implements connection pooling for browser instance lifecycle management
  - NEW: Fault-tolerant cross-browser model sharding with recovery (May 2025)
  - NEW: Performance history tracking and trend analysis (May 2025)
  - Target completion: May 25, 2025
  
- 🔄 **Distributed Testing Framework** (IN PROGRESS - 90% complete)
  - 🔄 Advanced fault tolerance and recovery mechanisms (IN PROGRESS - 95% complete)
  - Target completion: May 15, 2025 (accelerated timeline)

- 📋 **WebGPU/WebNN Migration to ipfs_accelerate_js** (PLANNED - After all tests pass)
  - Move all WebGPU/WebNN implementations to dedicated folder structure
  - Create clearer separation between JavaScript and Python components
  - Update import paths and documentation to reflect new structure
  - Simplify future JavaScript SDK development
  - Target completion: Q3 2025

## Key Features and Components

### Cross-Model Tensor Sharing

The Cross-Model Tensor Sharing system enables efficient sharing of tensors between multiple models, 
significantly improving memory efficiency and performance for multi-model workloads:

- **Shared Tensor Memory**: Multiple models can share the same tensor memory for common components
- **Reference Counting**: Intelligent memory management ensures tensors are only freed when no longer needed
- **Zero-Copy Tensor Views**: Create views into tensors without duplicating memory
- **Browser Storage Types**: Support for different tensor storage formats (CPU, WebGPU, WebNN)
- **Automatic Memory Optimization**: Identifies and frees unused tensors to reduce memory footprint
- **Intelligent Sharing Patterns**: Automatically identifies which models can share tensors

### Performance Benefits

- **Memory Reduction**: Up to 30% memory reduction for common multi-model workflows
- **Inference Speedup**: Up to 30% faster inference when reusing cached embeddings
- **Increased Throughput**: Higher throughput when running multiple related models
- **Browser Resource Efficiency**: More efficient use of limited browser memory resources

### Tensor Sharing Types

The system automatically identifies compatible model combinations for sharing:

| Tensor Type | Compatible Models | Description |
|-------------|------------------|-------------|
| text_embedding | BERT, T5, LLAMA, BART | Text embeddings for NLP models |
| vision_embedding | ViT, CLIP, DETR | Vision embeddings for image models |
| audio_embedding | Whisper, Wav2Vec2, CLAP | Audio embeddings for speech/audio models |
| vision_text_joint | CLIP, LLaVA, BLIP | Joint embeddings for multimodal models |
| audio_text_joint | CLAP, Whisper-Text | Joint embeddings for audio-text models |

### WebNN and WebGPU Benchmarking Tools

The framework includes comprehensive tools for benchmarking real WebNN and WebGPU implementations in browsers with clear distinction between real hardware acceleration and simulation mode:

```bash
# Run WebGPU benchmarks with Chrome
python benchmark_real_webnn_webgpu.py --webgpu --chrome

# Run WebNN benchmarks with Edge (best WebNN support)
python benchmark_real_webnn_webgpu.py --webnn --edge

# Run audio model benchmarks with Firefox (best for compute shaders)
python benchmark_real_webnn_webgpu.py --audio --firefox

# Benchmark with quantization (8-bit)
python benchmark_real_webnn_webgpu.py --text --bits 8

# Benchmark with mixed precision (4-bit)
python benchmark_real_webnn_webgpu.py --text --bits 4 --mixed-precision

# Run comprehensive benchmarks across multiple models
python benchmark_real_webnn_webgpu.py --comprehensive

# Store results in database
python benchmark_real_webnn_webgpu.py --text --db-path ./benchmark_db.duckdb

# Generate HTML report
python benchmark_real_webnn_webgpu.py --text --output-format html

# Check browser capabilities for WebNN/WebGPU support
python check_browser_webnn_webgpu.py --browser firefox

# Fix WebNN/WebGPU benchmarking issues
python fix_real_webnn_webgpu_benchmarks.py --browser chrome --fix-all
```

### Browser-Specific Optimizations

Different browsers excel at different tasks:

| Browser | Best For | Features | Command Flag |
|---------|----------|----------|-------------|
| Firefox | Audio models | 20-25% better performance for Whisper, CLAP | `--browser firefox --optimize-audio` |
| Edge | WebNN models | Superior WebNN implementation | `--browser edge --platform webnn` |
| Chrome | Vision models | Solid all-around WebGPU support | `--browser chrome --platform webgpu` |

For detailed instructions, see:
- [WebNN/WebGPU Benchmark System](WEBNN_WEBGPU_BENCHMARK_README.md)
- [Real WebNN/WebGPU Implementation Update](REAL_WEBNN_WEBGPU_IMPLEMENTATION_UPDATE.md)

## Web Resource Pool Integration

The WebGPU/WebNN Resource Pool Integration enables concurrent execution of multiple AI models across heterogeneous browser backends. It dramatically improves throughput, reduces resource waste, and provides fine-grained control over browser-based hardware acceleration resources.

### Key Features

- **Concurrent Model Execution**: Run multiple models simultaneously (3.5x throughput improvement)
- **Connection Pooling**: Efficiently manage browser connections with lifecycle management
- **Browser-Aware Load Balancing**: Distribute models to optimal browsers based on model type
- **Adaptive Resource Scaling**: Dynamically adjust resource allocation based on demand
- **Real-Time Monitoring**: Track resource utilization and performance metrics

### Using the Resource Pool

```python
# Create resource pool integration
from fixed_web_platform.resource_pool_bridge import ResourcePoolBridgeIntegration

integration = ResourcePoolBridgeIntegration(
    max_connections=4,
    browser_preferences={
        'audio': 'firefox',     # Firefox for audio models
        'vision': 'chrome',     # Chrome for vision models
        'text_embedding': 'edge' # Edge for embedding models
    },
    adaptive_scaling=True
)

# Initialize the integration
integration.initialize()

# Get model from resource pool
model = integration.get_model(
    model_type='text_embedding',
    model_name='bert-base-uncased',
    hardware_preferences={'priority_list': ['webgpu', 'cpu']}
)

# Run inference
result = model(inputs)
```

### Running Tests

```bash
# Test resource pool with multiple models
python test_web_resource_pool.py --models bert,vit,whisper

# Test concurrent model execution
python test_web_resource_pool.py --concurrent-models --models bert,vit,whisper

# Run stress test with high concurrency
python test_web_resource_pool.py --stress-test --duration 120
```

For detailed documentation, see:
- [WEB_RESOURCE_POOL_INTEGRATION.md](WEB_RESOURCE_POOL_INTEGRATION.md) - Comprehensive guide
- [WEBNN_WEBGPU_DATABASE_INTEGRATION.md](WEBNN_WEBGPU_DATABASE_INTEGRATION.md) - Database integration details

## Documentation and Reference

For a complete overview of all available documentation, refer to:
- [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md) - Comprehensive index of all project documentation with categorization

Major documentation categories include:
- [PHASE16_COMPLETION_REPORT.md](PHASE16_COMPLETION_REPORT.md) - Comprehensive report on the completed Phase 16 implementation
- [HARDWARE_BENCHMARKING_GUIDE.md](HARDWARE_BENCHMARKING_GUIDE.md) - Main hardware benchmarking documentation
- [BENCHMARK_DATABASE_GUIDE.md](BENCHMARK_DATABASE_GUIDE.md) - Benchmark database architecture and usage
- [WEB_PLATFORM_INTEGRATION_GUIDE.md](WEB_PLATFORM_INTEGRATION_GUIDE.md) - Web platform integration guide
- [REAL_WEBNN_WEBGPU_IMPLEMENTATION_UPDATE.md](REAL_WEBNN_WEBGPU_IMPLEMENTATION_UPDATE.md) - Latest WebNN/WebGPU implementation
- [WEBNN_WEBGPU_BENCHMARK_README.md](WEBNN_WEBGPU_BENCHMARK_README.md) - Overview of WebNN/WebGPU benchmark system
- [WEBNN_WEBGPU_DATABASE_INTEGRATION.md](WEBNN_WEBGPU_DATABASE_INTEGRATION.md) - How WebNN/WebGPU integrates with DuckDB
- [WEB_RESOURCE_POOL_INTEGRATION.md](WEB_RESOURCE_POOL_INTEGRATION.md) - Resource pool integration with web platform
- [TEMPLATE_INHERITANCE_GUIDE.md](TEMPLATE_INHERITANCE_GUIDE.md) - Template inheritance system documentation