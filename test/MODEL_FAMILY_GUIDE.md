# Model Family Classifier Guide

## Overview

The Model Family Classifier is a key component of the IPFS Accelerate framework that analyzes model characteristics to classify them into appropriate families. This classification enables intelligent template selection, hardware optimization, and resource allocation based on the model's characteristics.

## Features

- **Comprehensive Classification**: Classifies models into embedding, text generation, vision, audio, and multimodal families
- **Subfamily Classification**: Identifies specific model subfamilies for more precise classification
- **Multiple Analysis Methods**: Analyzes model name, class, tasks, methods, and hardware compatibility
- **Confidence Scoring**: Provides confidence levels for classification decisions
- **Template Selection**: Recommends appropriate template based on model family
- **Hardware Integration**: Works with hardware detection for optimal device allocation
- **Persistent Classification**: Optional model database for storing classification results
- **Extensible Definition**: Flexible model family definitions that can be customized

## Model Families

The classifier organizes models into these main families:

1. **Embedding Models** (e.g., BERT, RoBERTa, DistilBERT)
   - Text embedding and representation models
   - Subfamilies: masked_lm, sentence_transformer, token_classifier

2. **Text Generation Models** (e.g., GPT, LLaMA, T5)
   - Text generation and language models
   - Subfamilies: causal_lm, seq2seq, chat_model

3. **Vision Models** (e.g., ViT, ResNet, YOLO)
   - Vision and image processing models
   - Subfamilies: image_classifier, object_detector, segmentation, depth_estimation

4. **Audio Models** (e.g., Whisper, Wav2Vec2, HuBERT)
   - Audio processing models
   - Subfamilies: speech_recognition, audio_classifier, text_to_speech

5. **Multimodal Models** (e.g., LLaVA, BLIP, CLIP)
   - Models that combine multiple modalities
   - Subfamilies: vision_language, image_text_encoder, document_qa

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

## Integration with Hardware Detection

The Model Family Classifier works together with the Hardware Detection module to provide model-specific hardware recommendations:

```python
from model_family_classifier import classify_model
from hardware_detection import HardwareDetector, CUDA, MPS, OPENVINO, CPU

# Step 1: Classify the model
model_info = classify_model("bert-base-uncased")
model_family = model_info["family"]

# Step 2: Create hardware detector
detector = HardwareDetector()

# Step 3: Define priority list based on model family
if model_family == "text_generation":
    # LLMs need CUDA, fallback to CPU (MPS has limitations)
    priority_list = [CUDA, CPU]
    
elif model_family == "vision":
    # Vision models work well with CUDA or OpenVINO
    priority_list = [CUDA, OPENVINO, MPS, CPU]
    
elif model_family == "audio":
    # Audio models need fast processing
    priority_list = [CUDA, MPS, CPU]
    
elif model_family == "multimodal":
    # Multimodal models often need CUDA
    priority_list = [CUDA, CPU]
    
else:  # embedding
    # Embedding models are flexible
    priority_list = [CUDA, MPS, OPENVINO, CPU]

# Step 4: Get the optimal hardware based on model family
best_hardware = detector.get_hardware_by_priority(priority_list)
print(f"Optimal hardware for {model_family} model: {best_hardware}")

# Step 5: Get torch device with appropriate device index based on model family
if model_family == "text_generation":
    # Use primary GPU (index 0) for LLMs (typically need most memory)
    device = detector.get_torch_device_with_priority(priority_list, preferred_index=0)
elif model_family == "vision" or model_family == "audio":
    # Can use secondary GPU (index 1) if available
    device = detector.get_torch_device_with_priority(priority_list, preferred_index=1)
else:
    # Use any available GPU for other models
    device = detector.get_torch_device_with_priority(priority_list)

print(f"Recommended device: {device}")
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

## Version History

### v1.1 (March 2025)
- Added enhanced hardware compatibility analysis for better model-hardware matching
- Improved confidence scoring with weighted analysis
- Improved subfamily detection with partial matching support
- Added integration with the ResourcePool
- Enhanced template selection based on model subfamily
- Added detailed logging for classification decisions

### v1.0 (February 2025)
- Initial release with basic model family classification
- Support for embedding, text generation, vision, audio, and multimodal families
- Basic confidence scoring
- Model database for persisting classifications
- Template selection based on model family