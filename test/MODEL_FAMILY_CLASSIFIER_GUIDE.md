# Model Family Classifier Guide

## Overview

The Model Family Classifier provides robust classification of machine learning models into families and subfamilies based on their characteristics. It analyzes model names, tasks, class information, and hardware compatibility to make intelligent decisions about model categorization, which enables optimized resource allocation, template selection, and hardware assignment.

## Features

- **Comprehensive Model Classification**: Classifies models into core families (embedding, text generation, vision, audio, multimodal)
- **Subfamily Detection**: Identifies specialized model subfamilies for more granular classification
- **Multiple Analysis Methods**: Uses model name, class, tasks, methods, and hardware compatibility for classification
- **Confidence Scoring**: Provides confidence scores for classification decisions
- **Database Integration**: Optional model database for caching and improving classification accuracy
- **Hardware Compatibility Analysis**: Uses hardware compatibility patterns to improve classification
- **Template Selection**: Recommends appropriate templates based on model family
- **Weighted Classification**: Uses weighted scoring for more accurate classification
- **Graceful Fallbacks**: Provides sensible defaults when information is limited
- **Extensible Architecture**: Easily extensible for new model families and subfamilies
- **ResourcePool Integration**: Seamless integration with ResourcePool for optimal device selection

## Model Families

The classifier categorizes models into the following core families:

1. **Embedding Models** (e.g., BERT, RoBERTa, DistilBERT):
   - Used for sentence embeddings, token classification, text classification
   - Subfamilies: masked_lm, sentence_transformer, token_classifier
   - Example models: bert-base-uncased, roberta-base, distilbert-base-uncased

2. **Text Generation Models** (e.g., GPT, LLaMA, T5):
   - Used for text generation, translation, summarization
   - Subfamilies: causal_lm, seq2seq, chat_model
   - Example models: gpt2, llama-7b, t5-small

3. **Vision Models** (e.g., ViT, ResNet, YOLO):
   - Used for image classification, object detection, segmentation
   - Subfamilies: image_classifier, object_detector, segmentation, depth_estimation
   - Example models: vit-base-patch16-224, resnet-50, yolos-small

4. **Audio Models** (e.g., Whisper, Wav2Vec2):
   - Used for speech recognition, audio classification
   - Subfamilies: speech_recognition, audio_classifier, text_to_speech
   - Example models: whisper-small, wav2vec2-base, bark

5. **Multimodal Models** (e.g., CLIP, LLaVA, BLIP):
   - Combine multiple modalities (text+vision, etc.)
   - Subfamilies: vision_language, image_text_encoder, document_qa
   - Example models: clip-vit-base-patch32, llava-7b, blip-base

## Usage

### Basic Usage

```python
from model_family_classifier import classify_model

# Classify a model by name only
result = classify_model("bert-base-uncased")
print(f"Model family: {result['family']}")
print(f"Confidence: {result['confidence']:.2f}")
if result.get('subfamily'):
    print(f"Subfamily: {result['subfamily']}")

# Classify with more information for higher accuracy
result = classify_model(
    model_name="t5-small",
    model_class="T5ForConditionalGeneration",
    tasks=["translation", "summarization"]
)

# Print detailed classification
print(f"Model: {result['model_name']}")
print(f"Family: {result['family']}")
print(f"Subfamily: {result.get('subfamily')}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Source: {result['source']}")
```

### Using Model Database

```python
from model_family_classifier import classify_model

# Classify with model database for caching and improved accuracy
model_db_path = "model_classification_db.json"
result = classify_model(
    model_name="facebook/wav2vec2-base",
    model_class="Wav2Vec2Model",
    tasks=["automatic-speech-recognition"],
    model_db_path=model_db_path
)

# The classification will be saved to the database for future use
print(f"Model family: {result['family']}")
```

### Analyzing Hardware Compatibility

```python
from model_family_classifier import classify_model

# Hardware compatibility information can help with classification
hw_compatibility = {
    "cuda": {"compatible": True, "memory_usage": {"peak": 15000}},
    "mps": {"compatible": False},
    "openvino": {"compatible": False}
}

# Classify with hardware information
result = classify_model(
    model_name="llava-hf/llava-1.5-7b-hf",
    model_class="LlavaForConditionalGeneration",
    hw_compatibility=hw_compatibility
)

# High memory requirements and MPS incompatibility help identify multimodal LLMs
print(f"Model family: {result['family']}")  # Should identify as "multimodal"
```

### Getting Template Recommendations

```python
from model_family_classifier import ModelFamilyClassifier

# Create a classifier instance
classifier = ModelFamilyClassifier()

# Classify the model
result = classifier.classify_model("gpt2")
family = result["family"]
subfamily = result.get("subfamily")

# Get recommended template
template = classifier.get_template_for_family(family, subfamily)
print(f"Recommended template for gpt2: {template}")  # hf_text_generation_template.py
```

### Advanced Example with Multiple Analysis Methods

```python
from model_family_classifier import classify_model

# Combine multiple sources of information for best results
result = classify_model(
    model_name="vit-base-patch16-224",
    model_class="ViTForImageClassification",
    tasks=["image-classification"],
    methods=["classify", "process_image"]
)

# See which analysis methods contributed to the classification
for analysis in result["analyses"]:
    if analysis.get("family"):
        source = analysis.get("source", "unknown")
        confidence = analysis.get("confidence", 0)
        print(f"Analysis method: {source}, family: {analysis['family']}, confidence: {confidence:.2f}")
```

## ModelFamilyClassifier API

### Core Methods

The `ModelFamilyClassifier` class provides these core methods:

```python
from model_family_classifier import ModelFamilyClassifier

# Create classifier instance
classifier = ModelFamilyClassifier(model_db_path="model_db.json")

# Analyze model name
name_analysis = classifier.analyze_model_name("bert-base-uncased")

# Analyze model class
class_analysis = classifier.analyze_model_class("BertModel")

# Analyze model tasks
task_analysis = classifier.analyze_model_tasks(["fill-mask", "feature-extraction"])

# Analyze model methods
method_analysis = classifier.analyze_model_methods(["encode", "embed"])

# Analyze hardware compatibility
hw_analysis = classifier.analyze_hardware_compatibility({
    "cuda": {"compatible": True},
    "mps": {"compatible": True}
})

# Classify model using all available information
classification = classifier.classify_model(
    model_name="bert-base-uncased",
    model_class="BertModel",
    tasks=["fill-mask", "feature-extraction"],
    methods=["encode", "embed"],
    hw_compatibility={"cuda": {"compatible": True}}
)

# Get template for model family
template = classifier.get_template_for_family("embedding", "masked_lm")

# Update model database with classification results
classifier.update_model_db("bert-base-uncased", classification)
```

### Classification Options

The classification process combines multiple analysis methods with configurable weights:

```python
# Analysis method weights (can be customized)
method_weights = {
    "name_analysis": 0.7,
    "class_analysis": 0.9,
    "task_analysis": 1.0,
    "method_analysis": 0.8,
    "hardware_analysis": 0.5
}
```

## Template Selection

The classifier automatically recommends appropriate template files based on model family:

| Model Family | Template File |
|--------------|---------------|
| embedding | hf_embedding_template.py |
| text_generation | hf_text_generation_template.py |
| vision | hf_vision_template.py |
| audio | hf_audio_template.py |
| multimodal | hf_multimodal_template.py |
| default | hf_template.py |

## Subfamily Classification

For more detailed classification, the classifier identifies specialized subfamilies:

### Embedding Subfamilies

- **masked_lm**: BERT-like models with masked language modeling capability
- **sentence_transformer**: Models designed for sentence embeddings (SentenceBERT, SimCSE)
- **token_classifier**: Models specialized for token classification tasks (NER, etc.)

### Text Generation Subfamilies

- **causal_lm**: Autoregressive language models (GPT, LLaMA)
- **seq2seq**: Encoder-decoder models (T5, BART)
- **chat_model**: Models fine-tuned for dialogue (instruction-tuned models)

### Vision Subfamilies

- **image_classifier**: Models for image classification
- **object_detector**: Models for object detection (DETR, YOLO)
- **segmentation**: Models for image segmentation
- **depth_estimation**: Models for depth estimation

### Audio Subfamilies

- **speech_recognition**: Models for speech-to-text
- **audio_classifier**: Models for audio classification
- **text_to_speech**: Models for text-to-audio generation

### Multimodal Subfamilies

- **vision_language**: Models combining vision and language (LLaVA, BLIP)
- **image_text_encoder**: Models encoding both images and text (CLIP)
- **document_qa**: Models for document question answering

## Hardware Analysis-Based Classification

The classifier can use hardware compatibility patterns to improve classification:

| Hardware Pattern | Family Hint |
|------------------|-------------|
| High CUDA memory (>5GB) | text_generation (large LLM) |
| Medium CUDA memory (2-5GB) | text_generation (medium model) |
| MPS incompatible but CUDA compatible | multimodal |
| OpenVINO compatible, WebNN incompatible | vision |
| ROCm incompatible but MPS compatible | audio |
| CUDA and MPS incompatible | multimodal (complex vision-language) |

## Integration with ResourcePool

The model family classifier integrates with ResourcePool for optimal device selection:

```python
from resource_pool import get_global_resource_pool
from model_family_classifier import classify_model

# Classify model
classification = classify_model("bert-base-uncased")
model_family = classification["family"]

# Use model family for resource pool
pool = get_global_resource_pool()

# Create hardware preferences based on model family
hardware_preferences = {}
if model_family == "embedding":
    hardware_preferences = {"priority_list": ["cuda", "mps", "cpu"]}
elif model_family == "text_generation":
    hardware_preferences = {"priority_list": ["cuda", "cpu"]}
elif model_family == "vision":
    hardware_preferences = {"priority_list": ["cuda", "openvino", "mps", "cpu"]}
elif model_family == "audio":
    hardware_preferences = {"priority_list": ["cuda", "cpu"]}
elif model_family == "multimodal":
    hardware_preferences = {"priority_list": ["cuda", "cpu"]}

# Load model with optimal hardware selection
model = pool.get_model(
    model_family,  # Pass the model family for best device selection
    "bert-base-uncased",
    constructor=lambda: AutoModel.from_pretrained("bert-base-uncased"),
    hardware_preferences=hardware_preferences
)
```

## Classification Performance

The classifier accuracy depends on the information provided:

| Information Available | Expected Accuracy |
|-----------------------|------------------|
| Model name only | 70-80% |
| Model name + class | 85-90% |
| Model name + class + tasks | 90-95% |
| Model name + class + tasks + methods | 95-98% |
| All information + hardware compatibility | 98-99% |

## Fallback Strategies

The classifier uses multiple fallback strategies when information is limited:

1. **Model Name Analysis**: Uses naming conventions to identify family
2. **Class Name Patterns**: Extracts family from class names like "AutoModelFor..."
3. **Task Inference**: Infers likely tasks from model architecture
4. **Default Templates**: Provides sensible default templates when classification is uncertain

## Troubleshooting

### Model Misclassification

If a model is incorrectly classified:

1. **Provide More Information**: Add model_class, tasks, or methods for better accuracy
2. **Check Confidence Score**: Low confidence indicates uncertain classification
3. **Add to Model Database**: Add correct classification to the database for future use
4. **Check for Unusual Naming**: Some models have names that don't follow conventions
5. **Use Hardware Analysis**: Hardware compatibility can help with difficult cases

### Multiple Possible Classifications

If a model could belong to multiple families:

1. **Check Subfamily**: The subfamily may clarify the primary purpose
2. **Examine Confidence Scores**: Higher confidence typically indicates better match
3. **Consider Primary Use Case**: Choose the family that matches your use case
4. **Use Model-Specific Templates**: Create specialized templates for hybrid models

## Version History

### v2.0 (March 2025)
- Added hardware compatibility analysis for improved classification
- Enhanced subfamily detection with confidence scoring
- Added integration with WebNN/WebGPU compatibility
- Improved classification with weighted analysis methods
- Added support for model database updates and caching
- Enhanced template selection based on subfamilies
- Added comprehensive error handling and logging

### v1.5 (February 2025)
- Added subfamily classification for more detailed categorization
- Enhanced classification with method analysis
- Improved model class analysis
- Added confidence scoring for classification decisions
- Improved template selection logic
- Added support for model database

### v1.0 (January 2025)
- Initial implementation with basic classification
- Support for embedding, text_generation, vision, audio, and multimodal families
- Basic model name and task analysis
- Simple template selection
- Preliminary integration with ResourcePool