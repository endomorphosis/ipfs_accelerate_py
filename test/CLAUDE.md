# IPFS Accelerate Python Framework - Development Guide

## Current Focus: Test-Driven Hugging Face Transformers Integration (Updated March 2025)

### Phase 21: WebNN and WebGPU Integration (COMPLETED)
- ✅ Add WebNN detection to hardware system
- ✅ Add WebGPU detection for browser-based inference
- ✅ Implement cross-platform model compatibility checking
- ✅ Create WebNN-compatible test templates
- ✅ Enhance ResourcePool with web platform support
- ✅ Document web platform integration in guides
- ✅ Create web deployment examples
- ✅ Implement browser testing harness
- ✅ Add web platform benchmarking tools

### Phase 22: Integration Testing and Platform Support (IN PROGRESS)
- ✅ Create comprehensive integration test suite across all components
- ✅ Enhance ResourcePool with WebNN and WebGPU platform testing
- ✅ Implement resilient error handling for web platform detection
- ✅ Add web-specific hardware preferences with subfamily support
- ✅ Improve ResourcePool tests with WebNN/WebGPU compatibility
- ✅ Implement hardware-model integration with robust error handling
- ✅ Add resilient component detection with graceful degradation
- ✅ Create comprehensive hardware-model compatibility matrix
- ⏳ Add continuous integration for hardware tests
- ⏳ Develop custom tests for all hardware platforms
- ⏳ Implement error reporting system for hardware compatibility

### Phase 14: Development Pipeline for Test and Skillset Generators (COMPLETED)
- ✅ Enhance the test generator first (priority task)
- ✅ Push generated tests to test/skills/ directory
- ✅ Compare expected vs. collected values from test runs
- ✅ Implement structured error handling in test collector
- ✅ Add resource management for test execution environment
- ✅ Create hardware-aware test generation
- ✅ Implement dynamic code generation for all hardware platforms
- ✅ Build integration with model registry and classification system
- ✅ Use test results to inform skillset implementation
- ✅ Create Jinja2-based template system based on ipfs_accelerate_py/worker/skillset/hf_*.py
- ✅ Develop iterative feedback loop between test and skillset generators
- ✅ Create documentation generation for new models

### Phase 15: Comprehensive Test Analysis Framework (COMPLETED)
- ✅ Develop test result collector to extract implementation requirements
- ✅ Create structured format for test expectations and results
- ✅ Implement test-to-implementation mapping system
- ✅ Add robust error handling for template selection
- ✅ Add proper type checking in template selection function
- ✅ Build validation system for generated implementations
- ✅ Create discrepancy analysis and reporting tools
- ✅ Implement continuous integration for test-driven development

### Phase 16: Model Registry Enhancement and Integration (COMPLETED)
- ✅ Update model registry schema to support implementation details
- ✅ Add hardware compatibility tracking to registry 
- ✅ Implement model family classification integration
- ✅ Create comprehensive hardware compatibility matrix
- ✅ Add test validation tracking to registry entries
- ✅ Implement automatic registry updates from test results
- ✅ Create registry query tools for implementation status
- ✅ Build visualization and reporting for implementation coverage

### Phase 17: Resource Management System (COMPLETED)
- ✅ Implement centralized ResourcePool for efficient resource sharing
- ✅ Add memory management for large model testing
- ✅ Create resource cleanup mechanisms
- ✅ Implement model caching for faster testing
- ✅ Add resource usage monitoring and reporting
- ✅ Add device-specific model caching
- ✅ Create hardware-aware model classification system
- ✅ Integrate resource pool with hardware detection
- ✅ Add low-memory mode for resource-constrained environments
- ✅ Create comprehensive RESOURCE_POOL_GUIDE.md with examples
- ✅ Integrate ResourcePool with test generator workflow
- ✅ Add intelligent hardware selection based on model families
- ✅ Implement automatic memory detection and management

### Phase 18: Advanced Template System and Error Handling (COMPLETED)
- ✅ Create hardware-aware template selection system
- ✅ Add resource requirement-based template customization
- ✅ Implement model family template mapping
- ✅ Implement robust error handling framework
- ✅ Create extended template system with multi-template inheritance
- ✅ Add template verification and validation
- ✅ Create specialized templates for edge cases
- ✅ Develop template compatibility testing

### Phase 19: Comprehensive Documentation (COMPLETED)
- ✅ Create ResourcePool usage documentation
- ✅ Create Hardware Detection guide
- ✅ Create Model Family Classifier guide
- ✅ Create Hardware-Model Integration guide
- ✅ Create hardware compatibility matrix
- ✅ Create model template selection guide
- ✅ Create resource requirement estimation guide
- ✅ Create summary of improvements document
- ✅ Update main README with new features
- ✅ Add integrated architecture diagram for ResourcePool documentation
- ✅ Add comprehensive troubleshooting sections
- ✅ Update API references with latest functionality
- ✅ Document cross-platform compatibility strategies

### Phase 20: Hardware-Aware Resource Management (COMPLETED)
- ✅ Create model family-based hardware compatibility matrix
- ✅ Implement hardware-aware device selection for different model types
- ✅ Add memory requirement checking against available hardware
- ✅ Create specialized handling for multimodal models
- ✅ Implement fallback mechanisms when preferred hardware is unavailable
- ✅ Add hardware compatibility documentation to resource pool guide
- ✅ Enhance test infrastructure with hardware-aware functionality
- ✅ Create model family integration with hardware detection
- ✅ Implement resource optimization based on hardware capabilities
- ✅ Update documentation with hardware integration information

## Hardware Compatibility Matrix

### Model Family-Based Compatibility Chart

| Model Family | CUDA | ROCm (AMD) | MPS (Apple) | OpenVINO | WebNN | WebGPU | Notes |
|--------------|------|------------|-------------|----------|-------|--------|-------|
| Embedding (BERT, etc.) | ✅ High | ✅ High | ✅ High | ✅ Medium | ✅ High | ✅ Medium | Efficient on all hardware |
| Text Generation (LLMs) | ✅ High | ✅ Medium | ✅ Medium | ✅ Low | ❌ | ✅ Low | Memory requirements critical |
| Vision (ViT, CLIP, etc.) | ✅ High | ✅ Medium | ✅ High | ✅ High | ✅ Medium | ✅ Medium | OpenVINO optimized |
| Audio (Whisper, etc.) | ✅ High | ✅ Medium | ✅ Medium | ✅ Medium | ❌ | ❌ | CUDA preferred |
| Multimodal (LLaVA, etc.) | ✅ High | ❌ Low | ❌ Low | ❌ Low | ❌ | ❌ | Primarily CUDA only |

### Platform-Specific Incompatible Models

1. CUDA Incompatible Models (need optimization):
   - Vision-T5
   - MobileViT
   - UPerNet

2. OpenVINO Incompatible Models:
   - StableDiffusion
   - LLaVA-Next
   - BLIP

3. OpenVINO Partial Compatibility:
   - Whisper-Large
   - MusicGen

4. Apple Silicon (MPS) Incompatible Models:
   - StableDiffusion
   - LLaVA-Next
   - BLIP
   - AudioLDM
   - MusicGen
   - Vision-T5

5. AMD ROCm Incompatible Models:
   - StableDiffusion
   - Vision-T5
   - UPerNet
   - AudioLDM

6. WebNN Incompatible Models:
   - StableDiffusion
   - LLaVA-Next
   - BLIP
   - AudioLDM
   - MusicGen
   - Vision-T5
   - UPerNet

7. WebGPU/transformers.js Incompatible Models:
   - StableDiffusion
   - LLaVA-Next
   - BLIP
   - AudioLDM
   - MusicGen
   - Vision-T5
   - UPerNet
   - DPT
   - MobileViT

## Test Commands

### Hardware-Aware Testing
```bash
# Generate hardware-aware test with resource pool integration
python test/test_generator_with_resource_pool.py --model [model_name] --output-dir ./skills

# Test with specific hardware preferences
python test/test_generator_with_resource_pool.py --model [model_name] --device [cpu|cuda|mps|auto]

# Use model family classifier for optimal template selection
python test/test_generator_with_resource_pool.py --model [model_name] --use-model-family
```

### Model Testing
```bash
# Generate test using merged generator
python test/merged_test_generator.py --model [model_name]

# Run test to collect actual behavior
python test/skills/test_hf_[model_name].py

# Analyze test execution results
python test/analyze_test_results.py --model [model_name]

# Generate implementation requirements based on test results
python test/generate_implementation_requirements.py --model [model_name]

# Generate initial implementation template from test results
python test/generate_skillset_template.py --model [model_name] --from-test

# Verify generated implementation against test expectations
python test/verify_implementation.py --model [model_name]

# Update implementation based on test verification
python test/update_implementation.py --model [model_name] --fix-issues
```

### Hardware Testing
```bash
# Comprehensive hardware detection and compatibility test
python test/test_comprehensive_hardware.py --test all

# Test hardware backends (CPU, CUDA, OpenVINO)
python test/test_hardware_backend.py --backend [cpu|cuda|openvino] --model [model_name]

# Test Apple Silicon MPS hardware
python test/test_hardware_backend.py --backend mps --model [model_name]

# Test AMD ROCm hardware
python test/test_hardware_backend.py --backend rocm --model [model_name]

# Test WebNN compatibility
python test/test_hardware_backend.py --backend webnn --model [model_name]

# Test WebGPU compatibility
python test/test_hardware_backend.py --backend webgpu --model [model_name]

# Test all hardware platforms
python test/test_hardware_backend.py --backend all --model [model_name]

# Test resource pool with hardware awareness
python test/test_resource_pool.py --test hardware

# Test resource pool with web platform focus
python test/test_resource_pool.py --test hardware --web-platform

# Test model family integration with web platform support
python test/test_resource_pool.py --test family --debug

# Test all ResourcePool functionality including WebNN and WebGPU support
python test/test_resource_pool.py --test all --web-platform --debug

# Test performance metrics
python test/run_performance_tests.py --batch_size 8 --models all
```

### Web Platform Testing and Benchmarking
```bash
# Run web platform testing for a specific model
python test/web_platform_testing.py --test-model bert

# Test models from a specific modality
python test/web_platform_testing.py --test-modality vision

# Compare WebNN and WebGPU performance
python test/web_platform_testing.py --compare

# Run web platform benchmarking
python test/web_platform_benchmark.py --model bert

# Run comprehensive benchmark across modalities
python test/web_platform_benchmark.py --comparative

# List models with web platform support
python test/web_platform_benchmark.py --list-models

# Benchmark specific modality with custom batch sizes
python test/web_platform_benchmark.py --modality text --batch-sizes 1 8 16 32
```

### Integration Testing
```bash
# Run all integration tests
python test/integration_test_suite.py

# Run tests for specific categories
python test/integration_test_suite.py --categories hardware_detection resource_pool

# Run tests on specific hardware platforms
python test/integration_test_suite.py --hardware cpu cuda

# Skip slow tests for faster results
python test/integration_test_suite.py --skip-slow

# Specify custom timeout for tests
python test/integration_test_suite.py --timeout 600

# Save results to a specific file
python test/integration_test_suite.py --output ./my_integration_results.json
```

## Current Performance Benchmarks

### Text Generation Models
| Model | Platform | Throughput | Memory Usage | Latency |
|-------|----------|------------|--------------|---------|
| LLAMA (opt-125m) | CUDA | 125 tokens/sec | 240MB | 0.14s |
| LLAMA (opt-125m) | AMD | 115 tokens/sec | 245MB | 0.17s |
| LLAMA (opt-125m) | MPS | 85 tokens/sec | 250MB | 0.22s |
| LLAMA (opt-125m) | CPU | 38 tokens/sec | 275MB | 0.40s |
| Language Model (gpt2) | CUDA | 68 tokens/sec | 490MB | 0.26s |
| Language Model (gpt2) | AMD | 62 tokens/sec | 495MB | 0.29s |
| Language Model (gpt2) | MPS | 45 tokens/sec | 500MB | 0.36s |
| Language Model (gpt2) | CPU | 20 tokens/sec | 510MB | 0.85s |
| T5 (t5-efficient-tiny) | CUDA | 98 tokens/sec | 75MB | 0.16s |
| T5 (t5-efficient-tiny) | AMD | 90 tokens/sec | 79MB | 0.18s |
| T5 (t5-efficient-tiny) | MPS | 70 tokens/sec | 84MB | 0.22s |
| T5 (t5-efficient-tiny) | CPU | 32 tokens/sec | 90MB | 0.50s |

### Embedding Models
| Model | Platform | Processing Speed | Memory Usage | Dimensionality |
|-------|----------|------------------|--------------|----------------|
| BERT (tiny) | CUDA | 0.7ms/sentence | 18MB | 128 |
| BERT (tiny) | AMD | 0.9ms/sentence | 19MB | 128 |
| BERT (tiny) | MPS | 1.2ms/sentence | 21MB | 128 |
| BERT (tiny) | CPU | 4.3ms/sentence | 24MB | 128 |

### Vision Models
| Model | Platform | Processing Speed | Memory Usage | Resolution |
|-------|----------|------------------|--------------|------------|
| ViT (base) | CUDA | 5ms/image | 340MB | 224x224 |
| ViT (base) | AMD | 7ms/image | 345MB | 224x224 |
| ViT (base) | MPS | 12ms/image | 350MB | 224x224 |
| ViT (base) | OpenVINO | 9ms/image | 320MB | 224x224 |
| ViT (base) | CPU | 35ms/image | 360MB | 224x224 |

### Audio Models
| Model | Platform | Processing Speed | Memory Usage | Audio Length |
|-------|----------|------------------|--------------|-------------|
| Whisper (tiny) | CUDA | 0.3x realtime | 450MB | 30 sec |
| Whisper (tiny) | AMD | 0.4x realtime | 460MB | 30 sec |
| Whisper (tiny) | MPS | 0.6x realtime | 465MB | 30 sec |
| Whisper (tiny) | CPU | 1.8x realtime | 480MB | 30 sec |

### Web Platform Performance
| Model | Platform | Processing Speed | Memory Usage | First Inference | Batch Processing | Notes |
|-------|----------|------------------|--------------|----------------|------------------|-------|
| BERT (tiny) | WebNN | 12ms/sentence | 35MB | 45ms | 72ms (batch=8) | Chrome/Edge |
| BERT (tiny) | WebGPU | 8ms/sentence | 40MB | 38ms | 48ms (batch=8) | Chrome |
| ViT (tiny) | WebNN | 60ms/image | 90MB | 185ms | 420ms (batch=8) | Chrome/Edge |
| ViT (tiny) | WebGPU | 45ms/image | 95MB | 150ms | 315ms (batch=8) | Chrome |
| T5 (efficient-tiny) | WebNN | 72ms/sequence | 48MB | 215ms | 480ms (batch=8) | Chrome/Edge |
| T5 (efficient-tiny) | WebGPU | 51ms/sequence | 52MB | 175ms | 350ms (batch=8) | Chrome |
| ResNet (18) | WebNN | 68ms/image | 45MB | 145ms | 410ms (batch=8) | Chrome/Edge |
| ResNet (18) | WebGPU | 38ms/image | 47MB | 110ms | 265ms (batch=8) | Chrome |
| CLIP | WebNN | 82ms/item | 120MB | 240ms | 580ms (batch=8) | Chrome/Edge |
| CLIP | WebGPU | 65ms/item | 135MB | 195ms | 420ms (batch=8) | Chrome |
| Whisper (tiny) | WebNN* | 420ms/sec | 85MB | 780ms | 3200ms (batch=8) | Simulated |
| Whisper (tiny) | WebGPU* | 350ms/sec | 95MB | 620ms | 2800ms (batch=8) | Simulated |

\* Whisper models are partially supported in simulation mode only. Real browser performance may vary.