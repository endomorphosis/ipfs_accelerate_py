# Model Family Classifier Guide

## Overview

The Model Family Classifier is a key component of the IPFS Accelerate framework that analyzes model characteristics to classify them into appropriate families. This classification enables intelligent template selection, hardware optimization, and resource allocation based on the model's characteristics.

## Features (Enhanced March 2025)

- **Comprehensive Classification**: Classifies models into seven major families with extensive keyword coverage
- **Enhanced Model Detection**: Recognizes 300+ model architectures with specialized detection logic
- **Subfamily Classification**: Identifies 40+ specific model subfamilies for precise browser selection
- **Browser-Specific Optimization**: Routes models to optimal browsers based on family and subfamily
- **Multiple Analysis Methods**: Analyzes model name, class, tasks, methods, and hardware compatibility
- **Confidence Scoring**: Provides weighted confidence levels for classification decisions
- **Template Selection**: Recommends family or subfamily-specific templates for code generation
- **Hardware Integration**: Works with hardware detection for optimal device allocation
- **WebGPU/WebNN Optimization**: Specialized support for browser-based hardware acceleration
- **Persistent Classification**: Optional model database for storing classification results
- **Extensible Definition**: Flexible model family definitions that can be customized

## Model Families (Updated March 2025)

The classifier now organizes models into these seven main families with enhanced subfamilies:

1. **Embedding Models** (e.g., BERT, RoBERTa, E5, BGE, DistilBERT)
   - Text embedding and representation models
   - Subfamilies:
     - **masked_lm**: Models like BERT, RoBERTa using masked language modeling
     - **sentence_transformer**: Sentence-level embeddings (E5, BGE, SBERT, GTE)
     - **token_classifier**: NER and token-level classification models
     - **cross_encoder**: Ranking and cross-encoder models
     - **multilingual_embedding**: Cross-lingual models like XLM-RoBERTa, LaBSE
     - **domain_specific**: Specialized embeddings (SciBERT, BioBERT, FinBERT)

2. **Text Generation Models** (e.g., GPT, LLaMA, Mistral, T5, Gemma)
   - Text generation and language models
   - Subfamilies:
     - **causal_lm**: Decoder-only models like GPT, LLaMA, Mistral, Gemma
     - **seq2seq**: Encoder-decoder models like T5, BART, PaLM
     - **chat_model**: Instruction-tuned models like LLaMA-2-Chat, Vicuna
     - **code_generation**: Code-specific models like CodeLLaMA, StarCoder
     - **instructional**: Instruction-following models like Flan-T5, Alpaca
     - **creative_writing**: Story and creative text generation models

3. **Vision Models** (e.g., ViT, ResNet, YOLO, DINOv2, DETR)
   - Vision and image processing models
   - Subfamilies:
     - **image_classifier**: Image classification models like ViT, ResNet
     - **object_detector**: Object detection models like YOLO, DETR
     - **segmentation**: Segmentation models like SAM, Mask2Former
     - **depth_estimation**: 3D and depth estimation models
     - **pose_estimation**: Human pose and keypoint detection models
     - **video_understanding**: Video analysis models like ViViT, TimeSformer
     - **face_analysis**: Facial recognition and analysis models

4. **Audio Models** (e.g., Whisper, Wav2Vec2, AudioLDM, MusicGen)
   - Audio processing models
   - Subfamilies:
     - **speech_recognition**: Speech-to-text models like Whisper, Wav2Vec2
     - **audio_classifier**: Sound and audio classification models like CLAP
     - **text_to_speech**: TTS models like VALL-E, Bark, MMS-TTS
     - **music_generation**: Music generation models like MusicGen, AudioGen
     - **voice_conversion**: Voice style transfer and conversion models
     - **speaker_identification**: Speaker diarization and verification
     - **audio_enhancement**: Audio denoising and source separation models

5. **Multimodal Models** (e.g., LLaVA, BLIP, CLIP, Kosmos)
   - Models that combine multiple modalities
   - Subfamilies:
     - **vision_language**: Vision-language models like LLaVA, BLIP
     - **image_text_encoder**: Joint encoders like CLIP, SigLIP
     - **document_qa**: Document analysis models like LayoutLM, Donut
     - **audio_visual**: Audio-visual fusion models
     - **multimodal_chatbot**: Conversational multimodal models
     - **video_language**: Video-text models for captioning and QA

6. **Text-to-X Generation Models** (NEW - March 2025)
   - Models that generate non-text outputs from text inputs
   - Subfamilies:
     - **text_to_image**: Image generation models like Stable Diffusion, DALL-E
     - **text_to_audio**: Audio generation models like AudioGen, Bark
     - **text_to_video**: Video generation models like Sora, Gen-2
     - **text_to_3d**: 3D content generation models like DreamFusion
     - **controllable_generation**: ControlNet and style-guided generation

7. **Scientific & Domain-Specific Models** (NEW - March 2025)
   - Specialized scientific AI models
   - Subfamilies:
     - **protein_structure**: Protein folding models like AlphaFold, ESMFold
     - **drug_discovery**: Molecular modeling and drug design models
     - **medical_imaging**: Medical image analysis models
     - **physics_simulation**: Physics and equation-solving models
     - **robotics**: Robot control and policy models

## Basic Usage

```python
from model_family_classifier import classify_model

# Basic classification with just the model name
result = classify_model(model_name="bert-base-uncased")
print(f"Model family: {result['family']}")
print(f"Subfamily: {result['subfamily']}")
print(f"Confidence: {result['confidence']:.2f}")

# More detailed classification with additional information
detailed_result = classify_model(
    model_name="gpt2",
    model_class="GPT2LMHeadModel",
    tasks=["text-generation", "text-completion"]
)
print(f"Model family: {detailed_result['family']}")
print(f"Confidence: {detailed_result['confidence']:.2f}")

# Get the recommended template for a model
from model_family_classifier import ModelFamilyClassifier
classifier = ModelFamilyClassifier()
template = classifier.get_template_for_family(detailed_result['family'])
print(f"Recommended template: {template}")
```

## Integration with WebGPU/WebNN Optimization (March 2025)

The enhanced Model Family Classifier integrates with the WebGPU/WebNN Resource Pool to provide browser-specific optimizations:

```python
from model_family_classifier import classify_model
from fixed_web_platform.resource_pool_bridge import ResourcePoolBridgeIntegration

# Step 1: Classify the model
model_info = classify_model("bert-base-uncased")
model_family = model_info["family"]
model_subfamily = model_info["subfamily"]

# Step 2: Create resource pool integration with browser preferences
integration = ResourcePoolBridgeIntegration(
    max_connections=4,
    browser_preferences={
        'audio': 'firefox',     # Firefox has better compute shader performance for audio
        'vision': 'chrome',     # Chrome has good WebGPU support for vision models
        'text_embedding': 'edge',  # Edge has excellent WebNN support for text embeddings
        'text_generation': 'edge',  # Edge works well for text generation
        'multimodal': 'chrome',    # Chrome handles multimodal models well
        'text_to_x': 'chrome',     # Chrome for text-to-X generation models
        'scientific': 'chrome'     # Chrome for scientific models
    },
    adaptive_scaling=True
)

# Step 3: Initialize integration
integration.initialize()

# Step 4: Configure optimal execution settings based on model family and subfamily
execution_options = {}

# Apply family-specific optimizations
if model_family == "audio":
    # Audio models work best with Firefox compute shader optimizations
    execution_options["compute_shaders"] = True
    execution_options["workgroup_size"] = [256, 1, 1]  # Firefox-optimized workgroup size
    # Specific audio subfamily optimizations
    if model_subfamily == "speech_recognition":
        execution_options["streaming"] = True
    elif model_subfamily == "music_generation":
        execution_options["batch_size"] = 1  # Music generation needs more memory
    
elif model_family == "vision":
    # Vision models benefit from shader precompilation and parallel loading
    execution_options["precompile_shaders"] = True
    execution_options["parallel_loading"] = True
    # Video models need special handling
    if model_subfamily == "video_understanding":
        execution_options["frame_buffer_size"] = 16
    
elif model_family == "text_embedding":
    # Text embedding models work well with WebNN
    execution_options["platform"] = "webnn" 
    execution_options["webnn_ops_fallback"] = False  # Use pure WebNN when possible
    
elif model_family == "text_generation":
    # Text generation models need memory optimization
    execution_options["kv_cache_optimization"] = True
    execution_options["batch_size"] = 1
    # Add specialized handling for different subfamilies
    if model_subfamily == "causal_lm":
        execution_options["precision"] = 8  # Use 8-bit precision for causal LMs
    
elif model_family == "multimodal":
    # Multimodal models need parallel loading
    execution_options["parallel_loading"] = True
    execution_options["platform"] = "webgpu"
    
elif model_family == "text_to_x":
    # Text-to-X models need specialized settings
    execution_options["platform"] = "webgpu"
    execution_options["precompile_shaders"] = True
    
# Step 5: Get model with optimized browser selection
model = integration.get_model(
    model_type=model_family,
    model_name="bert-base-uncased",
    hardware_preferences={
        'priority_list': ['webgpu', 'webnn', 'cpu'],
        'model_family': model_family,
        'model_subfamily': model_subfamily,
        'execution_options': execution_options
    }
)

# Step 6: Run inference with browser-specific optimizations
result = model({"input_ids": [101, 2023, 2003, 1037, 3231, 102]})
print(f"Model executed on {result['browser']} browser with {result['platform']} platform")
print(f"Performance: {result['metrics']['latency_ms']}ms latency, {result['metrics']['throughput_items_per_second']} items/sec")
```

## Integration with Traditional Hardware Detection

The classifier also works with the traditional Hardware Detection module:

```python
from model_family_classifier import classify_model
from hardware_detection import HardwareDetector, CUDA, MPS, OPENVINO, CPU, QNN, WEBGPU, WEBNN

# Step 1: Classify the model
model_info = classify_model("bert-base-uncased")
model_family = model_info["family"]
model_subfamily = model_info["subfamily"]

# Step 2: Create hardware detector
detector = HardwareDetector()

# Step 3: Define priority list based on model family and subfamily
if model_family == "text_generation":
    if model_subfamily == "causal_lm":
        # Large causal LMs need CUDA, fallback to CPU
        priority_list = [CUDA, CPU]
    elif model_subfamily == "seq2seq":
        # Seq2seq models can use more hardware types
        priority_list = [CUDA, OPENVINO, MPS, CPU]
    elif model_subfamily == "code_generation":
        # Code models need more memory
        priority_list = [CUDA, CPU]
    else:
        # Other text generation models
        priority_list = [CUDA, MPS, CPU]
    
elif model_family == "vision":
    if model_subfamily in ["object_detector", "segmentation"]:
        # Detection and segmentation can use OpenVINO
        priority_list = [CUDA, OPENVINO, MPS, CPU]
    elif model_subfamily == "video_understanding":
        # Video models need faster hardware
        priority_list = [CUDA, ROCm, CPU]
    else:
        # General vision models
        priority_list = [CUDA, OPENVINO, MPS, QNN, WEBGPU, CPU]
    
elif model_family == "audio":
    if model_subfamily == "speech_recognition":
        # Speech recognition works well on multiple hardware
        priority_list = [CUDA, MPS, OPENVINO, CPU]
    elif model_subfamily == "music_generation":
        # Music generation needs more memory
        priority_list = [CUDA, CPU]
    else:
        # Other audio models
        priority_list = [CUDA, MPS, QNN, CPU]
    
elif model_family == "multimodal":
    # Multimodal models often need CUDA
    priority_list = [CUDA, CPU]
    
elif model_family == "text_to_x":
    # Text-to-X generation models need GPU
    priority_list = [CUDA, WEBGPU, CPU]
    
elif model_family == "scientific":
    # Scientific models often need specialized hardware
    priority_list = [CUDA, CPU]
    
else:  # embedding
    if model_subfamily in ["sentence_transformer", "cross_encoder"]:
        # These work well across hardware
        priority_list = [CUDA, MPS, OPENVINO, QNN, WEBNN, CPU]
    else:
        # Other embedding models are flexible
        priority_list = [CUDA, MPS, OPENVINO, QNN, WEBNN, WEBGPU, CPU]

# Step 4: Get the optimal hardware based on model family
best_hardware = detector.get_hardware_by_priority(priority_list)
print(f"Optimal hardware for {model_family}/{model_subfamily} model: {best_hardware}")
```

## Comprehensive Classification

The classifier can take multiple sources of information for more accurate classification:

```python
from model_family_classifier import classify_model

# Provide all available information for best results
result = classify_model(
    model_name="facebook/wav2vec2-base-960h",
    model_class="Wav2Vec2ForCTC",
    tasks=["automatic-speech-recognition", "audio-classification"],
    methods=["forward", "recognize", "transcribe"],
    hw_compatibility={
        "cuda": {"compatible": True, "memory_usage": {"peak": 1500}},
        "mps": {"compatible": True},
        "rocm": {"compatible": False}
    }
)

print(f"Model family: {result['family']}")  # audio
print(f"Subfamily: {result['subfamily']}")  # speech_recognition
print(f"Confidence: {result['confidence']:.2f}")

# Examine detailed analysis
for analysis in result["analyses"]:
    print(f"- {analysis['source']}: {analysis.get('family')} " + 
          f"(confidence: {analysis.get('confidence', 0):.2f})")
```

## Persisting Classifications with Model Database

You can persist classification results using the model database:

```python
from model_family_classifier import classify_model

# Create a model database to store classifications
model_db_path = "model_classifications.json"

# Classify model and update database
result = classify_model(
    model_name="gpt2",
    model_class="GPT2LMHeadModel",
    tasks=["text-generation"],
    model_db_path=model_db_path
)

# Next time, classification will use the database if model is found
cached_result = classify_model(
    model_name="gpt2",
    model_db_path=model_db_path
)

print(f"Cached result: {cached_result['family']}")
```

## Working with Custom Model Families

You can customize the model family definitions:

```python
from model_family_classifier import ModelFamilyClassifier

# Define custom model families
custom_families = {
    "retrieval": {
        "description": "Retrieval and search models",
        "keywords": ["retrieval", "search", "index", "rag", "retrieve"],
        "tasks": ["retrieval", "dense-passage-retrieval", "document-search"],
        "methods": ["retrieve", "search", "index", "query"]
    },
    # Add other custom families...
}

# Create classifier with custom family definitions
classifier = ModelFamilyClassifier(model_family_defs=custom_families)

# Use the classifier
result = classifier.classify_model(model_name="facebook/dpr-ctx_encoder-single-nq-base")
print(f"Model family: {result['family']}")
```

## Template Selection Based on Classification

A key use case is selecting the appropriate implementation template:

```python
from model_family_classifier import ModelFamilyClassifier

classifier = ModelFamilyClassifier()

# First, classify the model
classification = classifier.classify_model(
    model_name="gpt2",
    model_class="GPT2LMHeadModel",
    tasks=["text-generation"]
)

# Get the recommended template
template = classifier.get_template_for_family(
    classification["family"], 
    classification.get("subfamily")
)

print(f"Model: {classification['model_name']}")
print(f"Family: {classification['family']}")
print(f"Subfamily: {classification.get('subfamily')}")
print(f"Recommended template: {template}")
```

## Using Classification in Test Generation

The classification can be used to generate better tests:

```python
from model_family_classifier import classify_model

# First classify the model
model_info = classify_model("bert-base-uncased")
family = model_info["family"]
subfamily = model_info["subfamily"]

# Determine appropriate test cases based on family
if family == "embedding":
    test_cases = [
        {"text": "Hello world", "expected_dim": 768},
        {"text": "Multiple sentences. For testing.", "expected_dim": 768},
        {"pairs": [["This is a test", "This is similar"]], "expected_score_range": [0.7, 1.0]}
    ]
elif family == "text_generation":
    test_cases = [
        {"prompt": "Once upon a time", "max_length": 50, "expected_min_length": 10},
        {"prompt": "The best way to learn", "max_length": 30, "expected_min_length": 5}
    ]
elif family == "vision":
    test_cases = [
        {"image_path": "test.jpg", "expected_classes": ["person", "dog"]},
        {"image_path": "test2.jpg", "expected_shape": [1, 1000]}
    ]

# Generate appropriate imports based on family
if family == "embedding":
    imports = ["import torch", "from transformers import AutoModel, AutoTokenizer"]
elif family == "text_generation":
    imports = ["import torch", "from transformers import AutoModelForCausalLM, AutoTokenizer"]
elif family == "vision":
    imports = ["import torch", "import PIL", "from transformers import AutoImageProcessor, AutoModelForImageClassification"]

# Generate the test file
test_content = f"""
# Test file for {model_info['model_name']} ({family})
{chr(10).join(imports)}

def test_{family}_model():
    # Test implementation based on model family: {family}, subfamily: {subfamily}
    pass
"""

print(test_content)
```

## Integration with ResourcePool

The classifier works with the ResourcePool to optimize resource allocation:

```python
from model_family_classifier import classify_model
from resource_pool import get_global_resource_pool

# First, classify the model
model_info = classify_model("bert-base-uncased")
family = model_info["family"]

# Get the resource pool
pool = get_global_resource_pool()

# Create model constructor function that uses the classification
def create_model():
    if family == "embedding":
        from transformers import AutoModel
        return AutoModel.from_pretrained("bert-base-uncased")
    elif family == "text_generation":
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained("gpt2")
    # Add other model families as needed
    
# The resource pool will use optimal hardware based on model type
model = pool.get_model(family, "bert-base-uncased", constructor=create_model)
```

## Working with Data Structures

The classifier returns a detailed data structure that includes:

```python
{
    "model_name": "bert-base-uncased",
    "family": "embedding",
    "subfamily": "masked_lm",
    "confidence": 0.85,
    "subfamily_confidence": 0.75,
    "source": "combined_analysis",
    "analyses": [
        {
            "family": "embedding",
            "confidence": 0.8,
            "source": "name_analysis"
        },
        {
            "family": "embedding",
            "confidence": 0.9,
            "source": "class_analysis"
        },
        {
            "family": "embedding",
            "confidence": 0.85,
            "source": "task_analysis"
        }
    ]
}
```

## Best Practices

1. **Provide Multiple Sources**: Supply model name, class, tasks, and methods for best accuracy
2. **Check Confidence**: Check the confidence score to validate the classification
3. **Use with Hardware Detection**: Combine with hardware detection for optimal device allocation
4. **Persist Classifications**: Use a model database for frequently classified models
5. **Consider Subfamilies**: Use subfamily information for more specific optimizations
6. **Validate Results**: Verify classifications match expected model behavior
7. **Update Model Database**: Keep your model database updated with new models
8. **Use Template Selection**: Let the classifier recommend the appropriate template
9. **Integrate with ResourcePool**: Combine with ResourcePool for optimal resource usage
10. **Handle Unknown Models**: Implement fallbacks for unknown models

## Advanced Browser Selection for WebGPU/WebNN

The enhanced model type detection system specifically optimizes browser selection for WebGPU/WebNN workloads:

### Audio Models
- **Firefox**: Optimized for audio models with compute shader acceleration (20-25% better for Whisper)
  - Optimized workgroup size: 256x1x1 (vs Chrome's 128x2x1)
  - Ideal for: speech_recognition, audio_classifier, music_generation
  - Special handling: streaming mode for speech_recognition

### Vision Models
- **Chrome**: Optimized for vision models with efficient WebGPU implementation
  - Enhanced feature: shader precompilation for 30-45% faster startup
  - Parallel loading: 25-35% shorter loading time
  - Ideal for: image_classifier, object_detector, segmentation
  - Special handling: frame buffer optimizations for video models

### Text Models
- **Edge**: Superior WebNN support for text models
  - Optimized for text_embedding models using Edge's superior WebNN implementation
  - Efficient operation routing between WebNN and WebGPU backends
  - Memory-efficient KV cache implementation for text_generation
  - Ideal for: masked_lm, sentence_transformer, causal_lm, seq2seq

### Multimodal Models
- **Chrome**: Best for multimodal models requiring parallel processing
  - Parallel loading optimizations for vision-language components
  - Memory optimization for large multimodal models
  - Ideal for: vision_language, image_text_encoder, multimodal_chatbot

### Scientific Models
- **Chrome**: Most reliable for scientific computing workloads
  - Best WebGPU compute support for tensor operations
  - Higher precision arithmetic support
  - Ideal for: protein_structure, medical_imaging, physics_simulation

## Version History

### v2.0 (March 2025)
- **Major update**: Added support for two new model families (Text-to-X, Scientific)
- **Enhanced Detection**: Expanded keyword matching with 300+ model architectures
- **Browser Optimization**: Added browser-specific optimizations for WebGPU/WebNN
- **Subfamily System**: Expanded to 40+ model subfamilies with specialized routing
- **Template Selection**: Added subfamily-specific template selection
- **Hardware Integration**: Enhanced integration with WebGPU/WebNN Resource Pool
- **Task Mapping**: Added 100+ new task mappings for precise classification
- **Edge Cases**: Added specific handling for confusing model types (e.g., BERT-CLIP)
- **Performance Optimizations**: Added Firefox compute shader support for audio models

### v1.1 (February 2025)
- Added enhanced hardware compatibility analysis for better model-hardware matching
- Improved confidence scoring with weighted analysis
- Improved subfamily detection with partial matching support
- Added integration with the ResourcePool
- Enhanced template selection based on model subfamily
- Added detailed logging for classification decisions

### v1.0 (January 2025)
- Initial release with basic model family classification
- Support for embedding, text generation, vision, audio, and multimodal families
- Basic confidence scoring
- Model database for persisting classifications
- Template selection based on model family