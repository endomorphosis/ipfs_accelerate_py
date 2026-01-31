# Hardware-Model Integration Guide

## Overview

The Hardware-Model Integration system combines hardware detection with model family classification to create an intelligent system that can:

1. Determine optimal hardware for different model types
2. Recommend appropriate code templates based on model family and hardware
3. Estimate resource requirements for different models on different hardware
4. Configure resource pools for optimal resource sharing
5. Generate hardware-aware test files

This guide explains how to use these capabilities in your workflow.

## Core Components

The system consists of three main components:

1. **Hardware Detection** (`scripts/generators/hardware/hardware_detection.py`): Detects available hardware and capabilities
2. **Model Family Classifier** (`model_family_classifier.py`): Classifies models into families and subfamilies
3. **Hardware-Model Integration** (`hardware_model_integration.py`): Combines the two for intelligent hardware selection

These components work together with the ResourcePool system to provide efficient resource management with hardware awareness.

## Hardware-Aware Model Classification

### Basic Classification

```python
from hardware_model_integration import get_hardware_aware_model_classification

# Classify a model with hardware awareness
result = get_hardware_aware_model_classification(
    model_name="bert-base-uncased",
    model_class="BertModel",
    tasks=["fill-mask", "feature-extraction"]
)

print(f"Model family: {result['family']}")
print(f"Recommended hardware: {result['recommended_hardware']}")
print(f"Recommended template: {result['recommended_template']}")
print(f"Resource requirements: {result['resource_requirements']}")
```

### Using the Classifier Directly

For more control, you can use the classifier class directly:

```python
from hardware_model_integration import HardwareAwareModelClassifier

# Create classifier with options
classifier = HardwareAwareModelClassifier(
    hardware_cache_path="./hardware_cache.json",
    model_db_path="./model_database.json",
    force_refresh=False
)

# Get detailed classification
classification = classifier.classify_model(
    model_name="gpt2",
    model_class="GPT2LMHeadModel",
    tasks=["text-generation"],
    methods=["generate", "forward"]
)

# Access detailed results
print(f"Family: {classification['family']}")
print(f"Subfamily: {classification.get('subfamily')}")
print(f"Confidence: {classification.get('confidence', 0):.2f}")
print(f"Hardware compatibility: {classification['hardware_profile']}")
print(f"Recommended hardware: {classification['recommended_hardware']}")
print(f"Resource requirements: {classification['resource_requirements']}")
```

### Command Line Usage

```bash
# Classify a specific model
python hardware_model_integration.py --model bert-base-uncased

# Get model recommendations for a specific task
python hardware_model_integration.py --task image-classification

# Get model recommendations with hardware constraints
python hardware_model_integration.py --task text-generation --hw cuda

# Generate resource pool configuration for multiple models
python hardware_model_integration.py --resource-config bert-base-uncased gpt2 vit-base-patch16-224
```

## Integration with ResourcePool

The hardware-aware model classifier integrates seamlessly with the ResourcePool system to provide optimal resource allocation.

### Basic Integration

```python
from hardware_model_integration import HardwareAwareModelClassifier
from resource_pool import get_global_resource_pool
import os

# Create classifier
classifier = HardwareAwareModelClassifier()

# Get optimal resource pool configuration
models = ["bert-base-uncased", "gpt2", "vit-base-patch16-224"]
config = classifier.get_optimal_resource_pool_config(models)

# Configure resource pool based on requirements
pool = get_global_resource_pool()
if config["resource_pool_config"]["low_memory_mode"]:
    os.environ["RESOURCE_POOL_LOW_MEMORY"] = "1"

# Get model classification with hardware awareness
classification = classifier.classify_model("bert-base-uncased")
recommended_hw = classification["recommended_hardware"]

# Create a hardware-aware model constructor
def create_model():
    import torch
    from transformers import AutoModel
    model = AutoModel.from_pretrained("bert-base-uncased")
    return model  # ResourcePool will handle device placement

# Get model from resource pool with hardware preferences
model = pool.get_model(
    classification["family"], 
    "bert-base-uncased", 
    constructor=create_model,
    hardware_preferences={"device": recommended_hw}
)
```

### Hardware-Aware Resource Pool Configuration

The system can generate optimal resource pool configurations:

```python
from hardware_model_integration import HardwareAwareModelClassifier
import json

classifier = HardwareAwareModelClassifier()

# Generate optimal configuration for multiple models
models = ["bert-base-uncased", "t5-small", "clip-vit-base-patch32"]
config = classifier.get_optimal_resource_pool_config(models)

# Print configuration details
print(f"Low memory mode: {config['resource_pool_config']['low_memory_mode']}")
print(f"Recommended timeout: {config['resource_pool_config']['recommended_timeout_mins']} minutes")
print(f"Maximum memory required: {config['resource_requirements']['max_memory_mb']} MB")
print(f"Preferred hardware: {config['hardware_recommendations']['preferred_hardware']}")
print(f"PyTorch device: {config['hardware_recommendations']['torch_device']}")

# Save configuration to file
with open("resource_pool_config.json", "w") as f:
    json.dump(config, f, indent=2)
```

## Model Recommendations by Task

The system can recommend models for specific tasks based on hardware constraints:

```python
from hardware_model_integration import HardwareAwareModelClassifier

classifier = HardwareAwareModelClassifier()

# Get model recommendations for text generation with hardware constraints
recommendations = classifier.recommend_model_for_task(
    task="text-generation",
    hardware_constraints=["cuda"]
)

# Print recommendations
print(f"Task: {recommendations['task']}")
print(f"Hardware constraints: {recommendations['hardware_constraints']}")
print("\nRecommended models:")

for i, model in enumerate(recommendations["recommendations"], 1):
    print(f"\n{i}. {model['model_name']}")
    print(f"   Family: {model['family']}")
    print(f"   Subfamily: {model.get('subfamily', 'N/A')}")
    print(f"   Optimal hardware: {model['recommended_hardware']}")
    print(f"   Memory required: {model['resource_requirements']['recommended_memory_mb']} MB")
    print(f"   Confidence: {model.get('confidence', 0):.2f}")
```

## Hardware Compatibility Checking

You can check which models are compatible with specific hardware:

```python
from hardware_model_integration import HardwareAwareModelClassifier

classifier = HardwareAwareModelClassifier()

# Define a list of models to check
models = [
    "bert-base-uncased", 
    "gpt2", 
    "t5-small", 
    "facebook/wav2vec2-base",
    "llava-hf/llava-1.5-7b-hf"
]

# Check compatibility with different hardware platforms
mps_compatible = classifier.get_compatible_models_for_hardware(models, "mps")
cuda_compatible = classifier.get_compatible_models_for_hardware(models, "cuda")
openvino_compatible = classifier.get_compatible_models_for_hardware(models, "openvino")

print(f"MPS compatible models: {mps_compatible}")
print(f"CUDA compatible models: {cuda_compatible}")
print(f"OpenVINO compatible models: {openvino_compatible}")
```

## Hardware-Specific Resource Requirements

The system provides detailed resource requirements for different model types on different hardware:

```python
from hardware_model_integration import HardwareAwareModelClassifier

classifier = HardwareAwareModelClassifier()

# Get resource requirements for a model with different hardware options
model_name = "gpt2"

# Get classification with resource requirements
classification = classifier.classify_model(model_name)
requirements = classification["resource_requirements"]

print(f"Resource requirements for {model_name} on {classification['recommended_hardware']}:")
print(f"  Minimum memory: {requirements['min_memory_mb']} MB")
print(f"  Recommended memory: {requirements['recommended_memory_mb']} MB")
print(f"  CPU cores: {requirements['cpu_cores']}")
print(f"  Disk space: {requirements['disk_space_mb']} MB")
print(f"  Batch size: {requirements['batch_size']}")
```

## Integration with Test Generator

The hardware-model integration system works with the test generator to create hardware-aware test files:

```python
from hardware_model_integration import get_hardware_aware_model_classification
import jinja2
import os

# Get hardware-aware classification for the model
classification = get_hardware_aware_model_classification("bert-base-uncased")

# Load Jinja2 template
template_env = jinja2.Environment(loader=jinja2.FileSystemLoader("./templates"))
template = template_env.get_template(classification["recommended_template"])

# Generate test file with hardware awareness
test_content = template.render(
    model_name=classification["model_name"],
    model_family=classification["family"],
    model_subfamily=classification.get("subfamily"),
    recommended_hardware=classification["recommended_hardware"],
    resource_requirements=classification["resource_requirements"],
    torch_device=classification.get("torch_device", "cuda:0" if classification["recommended_hardware"] == "cuda" else "cpu")
)

# Save the test file
output_dir = "./generated_tests"
os.makedirs(output_dir, exist_ok=True)
test_filename = f"test_hf_{classification['model_name'].replace('/', '_').replace('-', '_')}.py"
with open(os.path.join(output_dir, test_filename), "w") as f:
    f.write(test_content)

print(f"Generated hardware-aware test file: {test_filename}")
```

### Using the Test Generator With Resource Pool

The project includes a `test_generator_with_resource_pool.py` script that combines all these components:

```bash
# Generate a hardware-aware test for BERT
python scripts/generators/models/test_generator_with_resource_pool.py --model bert-base-uncased --output-dir ./skills

# Generate with debug logging
python scripts/generators/models/test_generator_with_resource_pool.py --model bert-base-uncased --debug

# Clear the resource cache before generating
python scripts/generators/models/test_generator_with_resource_pool.py --model bert-base-uncased --clear-cache

# Generate with a custom timeout for resource cleanup
python scripts/generators/models/test_generator_with_resource_pool.py --model bert-base-uncased --timeout 60

# Force a specific hardware device
python scripts/generators/models/test_generator_with_resource_pool.py --model bert-base-uncased --device cuda
```

## Classification and Template Output Examples

### Example 1: BERT (Embedding Model)

```
=== Hardware-Aware Classification for bert-base-uncased ===
Family: embedding
Subfamily: masked_lm
Recommended Hardware: cpu
Recommended Template: hf_embedding_template.py

Resource Requirements:
  min_memory_mb: 2000
  recommended_memory_mb: 4000
  cpu_cores: 2
  disk_space_mb: 500
  batch_size: 1

Hardware Compatibility:
  cuda: ✅ (System: ✅)
  mps: ✅ (System: ❌)
  rocm: ✅ (System: ❌)
  openvino: ✅ (System: ✅)
  webnn: ✅ (System: ❌)
  webgpu: ✅ (System: ❌)
```

### Example 2: GPT2 (Text Generation Model)

```
=== Hardware-Aware Classification for gpt2 ===
Family: text_generation
Subfamily: causal_lm
Recommended Hardware: cuda
Recommended Template: hf_text_generation_template.py

Resource Requirements:
  min_memory_mb: 8000
  recommended_memory_mb: 16000
  cpu_cores: 4
  disk_space_mb: 5000
  batch_size: 4

Hardware Compatibility:
  cuda: ✅ (System: ✅)
  mps: ✅ (System: ❌)
  rocm: ✅ (System: ❌)
  openvino: ✅ (System: ✅)
  webnn: ✅ (System: ❌)
  webgpu: ✅ (System: ❌)
```

### Example 3: ViT (Vision Model)

```
=== Hardware-Aware Classification for vit-base-patch16-224 ===
Family: vision
Subfamily: image_classifier
Recommended Hardware: openvino
Recommended Template: hf_vision_template.py

Resource Requirements:
  min_memory_mb: 4000
  recommended_memory_mb: 8000
  cpu_cores: 2
  disk_space_mb: 500
  batch_size: 4

Hardware Compatibility:
  cuda: ✅ (System: ✅)
  mps: ✅ (System: ❌)
  rocm: ✅ (System: ❌)
  openvino: ✅ (System: ✅)
  webnn: ✅ (System: ❌)
  webgpu: ✅ (System: ❌)
```

## Hardware Compatibility Matrix

Below is a hardware compatibility matrix for different model families:

| Model Family | CUDA | MPS (Apple) | ROCm | OpenVINO | WebNN | WebGPU |
|--------------|------|-------------|------|----------|-------|--------|
| Embedding (BERT, etc.) | ✅ REAL | ✅ REAL | ✅ REAL | ✅ REAL | ✅ REAL | ✅ REAL |
| Text Generation (GPT2, etc.) | ✅ REAL | ✅ REAL | ✅ REAL | ✅ REAL | ✅ REAL | ✅ REAL |
| Vision (ViT, etc.) | ✅ REAL | ✅ REAL | ✅ REAL | ✅ REAL | ✅ REAL | ✅ REAL |
| Audio (Whisper, etc.) | ✅ REAL | ✅ REAL | ✅ REAL | ✅ REAL | ⚠️ SIMULATION | ⚠️ SIMULATION |
| Multimodal (LLAVA, etc.) | ✅ REAL | ⚠️ SIMULATION | ⚠️ SIMULATION | ⚠️ SIMULATION | ⚠️ SIMULATION | ⚠️ SIMULATION |
| Video (XCLIP, etc.) | ✅ REAL | ✅ REAL | ✅ REAL | ✅ REAL | ⚠️ SIMULATION | ⚠️ SIMULATION |

Notes:
- ✅ REAL: Full implementation with actual hardware acceleration
- ⚠️ SIMULATION: Simulated implementation that mimics real behavior but may not use hardware acceleration
- Test generators will automatically create the appropriate implementation types for each model/hardware combination

## Performance Considerations

- **Caching**: Hardware detection results are cached to avoid repeated system queries
- **Database Integration**: Model classifications can be cached in a database for faster lookups
- **Hardware Optimization**: The system applies hardware-specific optimizations based on detected capabilities
- **Resource Constraints**: Automatically adjusts for resource constraints on the host system
- **Memory Pressure**: Detects system memory pressure and adjusts accordingly

## Advanced Integration

### Custom Hardware Profiles

You can provide custom hardware compatibility profiles:

```python
from hardware_model_integration import HardwareAwareModelClassifier

# Create custom hardware profile
custom_hw_profile = {
    "cuda": {"compatible": True, "memory_usage": {"peak": 2000}},
    "mps": {"compatible": False},
    "openvino": {"compatible": True},
    "webnn": {"compatible": False}
}

# Use custom profile in classification
classifier = HardwareAwareModelClassifier()
classification = classifier.classify_model(
    model_name="my-custom-model",
    hw_compat_override=custom_hw_profile
)
```

### Hardware Detection Integration

You can integrate directly with the hardware detection system:

```python
from hardware_detection import detect_available_hardware, HardwareDetector
from model_family_classifier import classify_model

# Get hardware information
hw_info = detect_available_hardware()
best_device = hw_info["torch_device"]

# Create a detector for more control
detector = HardwareDetector()
cuda_details = detector.get_cuda_details()
custom_device = detector.get_torch_device_with_priority(
    priority_list=["cuda", "mps", "cpu"],
    preferred_index=1  # Use second GPU if available
)

# Use hardware info with model classification
model_info = classify_model("bert-base-uncased")
```

## Resources

- **ResourcePool Guide**: See `RESOURCE_POOL_GUIDE.md` for details on resource pooling
- **Hardware Detection Guide**: See `HARDWARE_DETECTION_GUIDE.md` for hardware detection details
- **Model Family Classifier Guide**: See `MODEL_FAMILY_CLASSIFIER_GUIDE.md` for model classification details
- **Test Generator README**: See `TEST_GENERATOR_README.md` for test generation information