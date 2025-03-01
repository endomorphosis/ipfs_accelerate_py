# Hugging Face Pipeline Tasks

This document provides a comprehensive overview of the pipeline tasks available in the Hugging Face Transformers library. Each section describes a pipeline task, its purpose, compatible models, expected inputs and outputs, and basic usage examples.

## 1. Text Generation Tasks

### text-generation
- **Description**: Generates text based on a prompt using auto-regressive models
- **Primary Models**: GPT-2, LLaMA, Mistral, Bloom, GPT-J, Falcon, Gemma, Mixtral
- **Input**: Text prompt
- **Output**: Generated text continuation
- **Example**:
```python
from transformers import pipeline
generator = pipeline("text-generation", model="gpt2")
result = generator("The quick brown fox jumps over", max_length=25)
```
- **Implementation Classes**:
  - `AutoModelForCausalLM` - For decoder-only models 
  - Used with `pipeline("text-generation")`

### text2text-generation
- **Description**: Transforms input text into output text (e.g., translation, summarization)
- **Primary Models**: T5, BART, MT5, mBART
- **Input**: Text to transform
- **Output**: Transformed text
- **Example**:
```python
from transformers import pipeline
translator = pipeline("text2text-generation", model="t5-small")
result = translator("translate English to French: Hello, how are you?")
```
- **Implementation Classes**:
  - `AutoModelForSeq2SeqLM` - For encoder-decoder models
  - Used with `pipeline("text2text-generation")`

### summarization
- **Description**: Generates a concise summary of longer text
- **Primary Models**: BART, T5, Pegasus, LED
- **Input**: Long text
- **Output**: Summarized text
- **Example**:
```python
from transformers import pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
result = summarizer("Long text to summarize...", max_length=130, min_length=30)
```
- **Implementation Classes**:
  - `AutoModelForSeq2SeqLM`
  - Used with `pipeline("summarization")`

### translation_XX_to_YY
- **Description**: Translates text from one language to another
- **Primary Models**: T5, BART, M2M-100, MarianMT
- **Input**: Text in source language
- **Output**: Translated text in target language
- **Example**:
```python
from transformers import pipeline
translator = pipeline("translation_en_to_fr", model="t5-small")
result = translator("This is a test")
```
- **Implementation Classes**:
  - `AutoModelForSeq2SeqLM`
  - Used with `pipeline("translation_XX_to_YY")` where XX and YY are language codes

## 2. Language Understanding Tasks

### feature-extraction
- **Description**: Extracts feature vectors/embeddings from text
- **Primary Models**: BERT, RoBERTa, MPNet, DistilBERT, DeBERTa
- **Input**: Text
- **Output**: Embedding vectors (hidden states)
- **Example**:
```python
from transformers import pipeline
extractor = pipeline("feature-extraction", model="bert-base-uncased")
features = extractor("Extract features from this text")
```
- **Implementation Classes**:
  - `AutoModel` 
  - Used with `pipeline("feature-extraction")`

### fill-mask
- **Description**: Predicts masked tokens in text
- **Primary Models**: BERT, RoBERTa, ALBERT
- **Input**: Text with [MASK] tokens
- **Output**: Top predictions for masked tokens with scores
- **Example**:
```python
from transformers import pipeline
unmasker = pipeline("fill-mask", model="bert-base-uncased")
result = unmasker("The quick brown [MASK] jumps over the lazy dog")
```
- **Implementation Classes**:
  - `AutoModelForMaskedLM`
  - Used with `pipeline("fill-mask")`

### question-answering
- **Description**: Extracts answers to questions from context
- **Primary Models**: BERT, RoBERTa, DistilBERT
- **Input**: Question and context
- **Output**: Answer span with start/end positions and score
- **Example**:
```python
from transformers import pipeline
qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
result = qa(question="What is the capital of France?", context="Paris is the capital of France.")
```
- **Implementation Classes**:
  - `AutoModelForQuestionAnswering`
  - Used with `pipeline("question-answering")`

### token-classification
- **Description**: Labels each token in text (NER, POS tagging)
- **Primary Models**: BERT, RoBERTa, DistilBERT
- **Input**: Text
- **Output**: List of entities with position, label, and score
- **Example**:
```python
from transformers import pipeline
ner = pipeline("token-classification", model="dbmdz/bert-large-cased-finetuned-conll03-english")
result = ner("My name is Jean and I live in New York")
```
- **Implementation Classes**:
  - `AutoModelForTokenClassification`
  - Used with `pipeline("token-classification")` or `pipeline("ner")`

### sentiment-analysis
- **Description**: Classifies sentiment of text (positive/negative)
- **Primary Models**: BERT, RoBERTa, DistilBERT
- **Input**: Text
- **Output**: Sentiment label with score
- **Example**:
```python
from transformers import pipeline
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
result = sentiment_analyzer("I really enjoyed this movie!")
```
- **Implementation Classes**:
  - `AutoModelForSequenceClassification`
  - Used with `pipeline("sentiment-analysis")` or `pipeline("text-classification")`

### text-classification
- **Description**: General purpose text classification
- **Primary Models**: BERT, RoBERTa, DistilBERT, XLNet
- **Input**: Text
- **Output**: Classification label with score
- **Example**:
```python
from transformers import pipeline
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
result = classifier("This movie was great!")
```
- **Implementation Classes**:
  - `AutoModelForSequenceClassification`
  - Used with `pipeline("text-classification")`

### zero-shot-classification
- **Description**: Classifies text without task-specific training
- **Primary Models**: BART, T5, DeBERTa
- **Input**: Text and candidate labels
- **Output**: Scores for each candidate label
- **Example**:
```python
from transformers import pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
result = classifier("I loved this movie", candidate_labels=["positive", "negative", "neutral"])
```
- **Implementation Classes**:
  - `AutoModelForSequenceClassification` (with NLI training)
  - Used with `pipeline("zero-shot-classification")`

## 3. Computer Vision Tasks

### image-classification
- **Description**: Classifies images into categories
- **Primary Models**: ViT, ResNet, ConvNeXT, DeiT, Swin
- **Input**: Image (path, URL, or PIL Image)
- **Output**: Class labels with confidence scores
- **Example**:
```python
from transformers import pipeline
classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
result = classifier("path/to/image.jpg")
```
- **Implementation Classes**:
  - `AutoModelForImageClassification`
  - Used with `pipeline("image-classification")`

### object-detection
- **Description**: Locates and classifies objects in images
- **Primary Models**: DETR, YOLOS, Conditional DETR
- **Input**: Image (path, URL, or PIL Image)
- **Output**: Bounding boxes, class labels, and confidence scores
- **Example**:
```python
from transformers import pipeline
detector = pipeline("object-detection", model="facebook/detr-resnet-50")
result = detector("path/to/image.jpg")
```
- **Implementation Classes**:
  - `AutoModelForObjectDetection`
  - Used with `pipeline("object-detection")`

### image-segmentation
- **Description**: Generates pixel-level masks for objects in images
- **Primary Models**: SegFormer, DETR, UPerNet, Mask2Former
- **Input**: Image (path, URL, or PIL Image)
- **Output**: Segmentation masks with class labels
- **Example**:
```python
from transformers import pipeline
segmenter = pipeline("image-segmentation", model="facebook/mask2former-swin-tiny-coco")
result = segmenter("path/to/image.jpg")
```
- **Implementation Classes**:
  - `AutoModelForImageSegmentation`
  - Used with `pipeline("image-segmentation")`

### depth-estimation
- **Description**: Estimates depth from images
- **Primary Models**: DPT, Depth Anything, GLPN
- **Input**: Image (path, URL, or PIL Image)
- **Output**: Pixel-wise depth map
- **Example**:
```python
from transformers import pipeline
depth_estimator = pipeline("depth-estimation", model="Intel/dpt-large")
result = depth_estimator("path/to/image.jpg")
```
- **Implementation Classes**:
  - `AutoModelForDepthEstimation`
  - Used with `pipeline("depth-estimation")`

## 4. Audio Tasks

### automatic-speech-recognition
- **Description**: Transcribes speech to text
- **Primary Models**: Whisper, Wav2Vec2, Hubert
- **Input**: Audio file (path, URL, or array)
- **Output**: Transcribed text
- **Example**:
```python
from transformers import pipeline
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-small")
result = transcriber("path/to/audio.wav")
```
- **Implementation Classes**:
  - `AutoModelForSpeechSeq2Seq` (for Whisper)
  - `AutoModelForCTC` (for Wav2Vec2)
  - Used with `pipeline("automatic-speech-recognition")`

### audio-classification
- **Description**: Classifies audio into categories
- **Primary Models**: Wav2Vec2, HuBERT, Audio Spectrogram Transformer
- **Input**: Audio file (path, URL, or array)
- **Output**: Audio class labels with confidence scores
- **Example**:
```python
from transformers import pipeline
classifier = pipeline("audio-classification", model="superb/hubert-large-superb-er")
result = classifier("path/to/audio.wav")
```
- **Implementation Classes**:
  - `AutoModelForAudioClassification`
  - Used with `pipeline("audio-classification")`

### text-to-audio
- **Description**: Generates audio from text
- **Primary Models**: Bark, MusicGen, SpeechT5
- **Input**: Text
- **Output**: Generated audio
- **Example**:
```python
from transformers import pipeline
synthesizer = pipeline("text-to-audio", model="suno/bark-small")
result = synthesizer("Hello, how are you?")
```
- **Implementation Classes**:
  - `AutoModelForTextToWaveform` or `AutoModelForTextToSpectrogram`
  - Used with `pipeline("text-to-audio")`

## 5. Multimodal Tasks

### image-to-text
- **Description**: Generates text descriptions from images
- **Primary Models**: BLIP, ViT-GPT2, Vision Encoder Decoder
- **Input**: Image (path, URL, or PIL Image)
- **Output**: Generated text description
- **Example**:
```python
from transformers import pipeline
captioner = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
result = captioner("path/to/image.jpg")
```
- **Implementation Classes**:
  - `AutoModelForVision2Seq` or `VisionEncoderDecoderModel`
  - Used with `pipeline("image-to-text")`

### document-question-answering
- **Description**: Answers questions about document images
- **Primary Models**: LayoutLM, LayoutLMv2, LayoutLMv3, DONUT
- **Input**: Document image and question
- **Output**: Answer extracted from document
- **Example**:
```python
from transformers import pipeline
qa = pipeline("document-question-answering", model="impira/layoutlm-document-qa")
result = qa("path/to/document.jpg", "What is the invoice number?")
```
- **Implementation Classes**:
  - `AutoModelForDocumentQuestionAnswering`
  - Used with `pipeline("document-question-answering")`

### visual-question-answering
- **Description**: Answers questions about images
- **Primary Models**: ViLT, BLIP, CLIP+BERT
- **Input**: Image and question
- **Output**: Answer to the question about the image
- **Example**:
```python
from transformers import pipeline
vqa = pipeline("visual-question-answering", model="dandelin/vilt-b32-finetuned-vqa")
result = vqa("path/to/image.jpg", "What color is the car?")
```
- **Implementation Classes**:
  - `AutoModelForVisualQuestionAnswering`
  - Used with `pipeline("visual-question-answering")`

## 6. Other Specialized Tasks

### table-question-answering
- **Description**: Answers questions about tabular data
- **Primary Models**: TAPAS, TABLE-BERT
- **Input**: Table data and question
- **Output**: Answer extracted from table
- **Example**:
```python
from transformers import pipeline
table_qa = pipeline("table-question-answering", model="google/tapas-base-finetuned-wtq")
result = table_qa(table=table_data, query="Who has the highest score?")
```
- **Implementation Classes**:
  - `AutoModelForTableQuestionAnswering`
  - Used with `pipeline("table-question-answering")`

### time-series-prediction
- **Description**: Forecasts future values in time series data
- **Primary Models**: Time Series Transformer, Autoformer, Informer
- **Input**: Historical time series data
- **Output**: Predicted future values
- **Example**:
```python
from transformers import pipeline
forecaster = pipeline("time-series-prediction", model="huggingface/time-series-transformer-tourism-monthly")
result = forecaster(past_values=past_values, past_time_features=past_features, future_time_features=future_features)
```
- **Implementation Classes**:
  - `AutoModelForTimeSeriesPrediction`
  - Used with `pipeline("time-series-prediction")`

### conversational
- **Description**: Maintains dialogue state for conversations
- **Primary Models**: DialoGPT, BlenderBot
- **Input**: Conversation history and new user input
- **Output**: Model response with updated conversation state
- **Example**:
```python
from transformers import pipeline
chatbot = pipeline("conversational", model="microsoft/DialoGPT-medium")
result = chatbot({"text": "Hello, how are you?", "past_user_inputs": [], "generated_responses": []})
```
- **Implementation Classes**:
  - `AutoModelForCausalLM` or `BlenderbotForConditionalGeneration`
  - Used with `pipeline("conversational")`

### text-to-image
- **Description**: Generates images from text descriptions
- **Primary Models**: Stable Diffusion (not directly in transformers)
- **Input**: Text description
- **Output**: Generated image
- **Example**:
```python
from diffusers import DiffusionPipeline
pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
image = pipeline("A beautiful sunset over mountains").images[0]
```
- **Implementation Classes**:
  - `DiffusionPipeline` (from diffusers library)

### protein-folding
- **Description**: Predicts 3D protein structures from amino acid sequences
- **Primary Models**: ESM-2
- **Input**: Protein sequence
- **Output**: Predicted 3D structure
- **Example**:
```python
from transformers import pipeline
folder = pipeline("protein-folding", model="facebook/esmfold_v1")
result = folder("MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG")
```
- **Implementation Classes**:
  - `EsmForProteinFolding`
  - Used with `pipeline("protein-folding")`

## Usage Patterns

### Basic usage pattern

```python
from transformers import pipeline

# Using default model
pipe = pipeline(task="text-classification")
result = pipe("I love machine learning!")

# Specifying a model
pipe = pipeline(task="text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
result = pipe("I love machine learning!")

# Using with PyTorch/TensorFlow models directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
result = pipe("I love machine learning!")
```

### Device selection

```python
# Run on CPU
pipe = pipeline("text-generation", model="gpt2", device=-1)

# Run on GPU (cuda:0)
pipe = pipeline("text-generation", model="gpt2", device=0)

# Run on specific GPU 
pipe = pipeline("text-generation", model="gpt2", device="cuda:1")
```

### Batch processing

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
results = classifier(["I love this movie", "This was a terrible film", "Absolutely amazing!"], batch_size=2)
```

### Handling pipeline task-specific parameters

```python
# Text generation with parameters
generator = pipeline("text-generation", model="gpt2")
result = generator(
    "Once upon a time",
    max_length=50,
    num_return_sequences=3,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95
)

# Zero-shot classification with specific labels
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
result = classifier(
    "This is a delicious meal with pasta and sauce",
    candidate_labels=["food", "travel", "technology", "sports"],
    hypothesis_template="This text is about {}."
)
```