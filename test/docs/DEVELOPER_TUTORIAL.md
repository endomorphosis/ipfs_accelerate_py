# Developer Tutorial: Building Web-Accelerated AI Applications

**Date: March 6, 2025**  
**Version: 1.0**

This tutorial provides step-by-step guidance for developers looking to build web-accelerated AI applications using the IPFS Accelerate Python Framework. By following these examples, you'll learn how to implement efficient, cross-platform AI applications that leverage hardware acceleration in browsers.

## Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Tutorial 1: Text Embedding with BERT](#tutorial-1-text-embedding-with-bert)
4. [Tutorial 2: Image Classification with Vision Models](#tutorial-2-image-classification-with-vision-models)
5. [Tutorial 3: Streaming Text Generation](#tutorial-3-streaming-text-generation)
6. [Tutorial 4: Audio Transcription with Whisper](#tutorial-4-audio-transcription-with-whisper)
7. [Tutorial 5: Building a Multimodal Application](#tutorial-5-building-a-multimodal-application)
8. [Advanced Techniques](#advanced-techniques)
9. [Deployment Guidelines](#deployment-guidelines)
10. [Resources and References](#resources-and-references)

## Introduction

The IPFS Accelerate Python Framework allows you to create AI-powered applications that run efficiently across various hardware platforms, including web browsers. This tutorial focuses on implementing applications that leverage WebGPU and WebNN for accelerated inference directly in browsers.

### Why Web-Accelerated AI?

- **Accessibility**: No need for users to install software or have specialized hardware
- **Cross-Platform**: Works on any device with a modern browser
- **Privacy**: Processing happens on the user's device, keeping data local
- **Performance**: Hardware acceleration provides near-native speeds
- **Cost-Efficient**: Reduces server-side compute costs

## Prerequisites

To follow this tutorial, you'll need:

1. **Python 3.9+** with pip installed
2. **IPFS Accelerate Python Framework** (install using pip):
   ```bash
   pip install ipfs-accelerate-py
   ```
3. **Modern Browser** with WebGPU support:
   - Chrome 113+ or Edge 113+
   - Firefox 121+
   - Safari 17.4+ (limited support)
4. **Development Environment**:
   - Any text editor or IDE
   - Node.js 16+ (for web development components)
   - Basic knowledge of Python and JavaScript

## Tutorial 1: Text Embedding with BERT

In this tutorial, we'll create a semantic search application using BERT embeddings.

### Step 1: Set Up the Environment

Create a new directory for your project and set up a virtual environment:

```bash
mkdir semantic-search-app
cd semantic-search-app
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install ipfs-accelerate-py flask numpy
```

### Step 2: Create the Backend

Create a file named `app.py`:

```python
from flask import Flask, request, jsonify
from fixed_web_platform.unified_web_framework import UnifiedWebPlatform
import numpy as np

app = Flask(__name__)
embedding_model = None

def initialize_model():
    global embedding_model
    # Initialize the model with WebGPU as primary platform and WebNN as fallback
    embedding_model = UnifiedWebPlatform(
        model_name="bert-base-uncased",
        model_type="text_embedding",
        platform="auto",  # Auto-selects best available platform
        enable_shader_precompilation=True
    )
    # Pre-load the model
    embedding_model.load_model()
    print("Model initialized successfully")

# Create a simple in-memory document database
documents = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning models process data to make predictions",
    "Neural networks consist of layers of interconnected nodes",
    "Web browsers can now run AI models directly using WebGPU",
    "BERT is a transformer-based machine learning model for NLP tasks"
]
document_embeddings = []

@app.route('/api/embed', methods=['POST'])
def embed_text():
    data = request.json
    text = data.get('text', '')
    
    # Generate embedding
    embedding = embedding_model.run_inference({"input_text": text})
    
    # Convert to list for JSON serialization
    return jsonify({"embedding": embedding.tolist()})

@app.route('/api/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query', '')
    
    # Generate query embedding
    query_embedding = embedding_model.run_inference({"input_text": query})
    
    # Calculate similarity with all documents
    similarities = []
    for i, doc_embedding in enumerate(document_embeddings):
        similarity = np.dot(query_embedding, doc_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
        )
        similarities.append((similarity, documents[i]))
    
    # Sort by similarity (highest first)
    similarities.sort(reverse=True, key=lambda x: x[0])
    
    # Return top 3 results
    results = [
        {"text": doc, "score": float(score)}
        for score, doc in similarities[:3]
    ]
    
    return jsonify({"results": results})

@app.route('/api/index', methods=['POST'])
def index_document():
    data = request.json
    document = data.get('document', '')
    
    # Add to documents list
    documents.append(document)
    
    # Generate and store embedding
    embedding = embedding_model.run_inference({"input_text": document})
    document_embeddings.append(embedding)
    
    return jsonify({"status": "success", "document_id": len(documents) - 1})

@app.route('/')
def index():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    initialize_model()
    
    # Pre-compute embeddings for sample documents
    for doc in documents:
        embedding = embedding_model.run_inference({"input_text": doc})
        document_embeddings.append(embedding)
    
    app.run(debug=True)
```

### Step 3: Create the Frontend

Create a directory for static files:

```bash
mkdir static
```

Create `static/index.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Semantic Search with WebGPU</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        .container {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .search-box {
            display: flex;
            gap: 10px;
        }
        input {
            flex: 1;
            padding: 10px;
            font-size: 16px;
        }
        button {
            padding: 10px 20px;
            background-color: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
        }
        .result {
            border: 1px solid #ddd;
            padding: 15px;
            border-radius: 5px;
        }
        .score {
            color: #666;
            font-size: 14px;
        }
        .add-doc {
            margin-top: 30px;
            border-top: 1px solid #ddd;
            padding-top: 20px;
        }
        textarea {
            width: 100%;
            height: 100px;
            padding: 10px;
            margin-bottom: 10px;
            font-size: 16px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Semantic Search</h1>
        <p>Search documents using semantic meaning rather than keyword matching.</p>
        
        <div class="search-box">
            <input type="text" id="query" placeholder="Enter your search query...">
            <button onclick="search()">Search</button>
        </div>
        
        <div id="results"></div>
        
        <div class="add-doc">
            <h2>Add New Document</h2>
            <textarea id="document" placeholder="Enter document text..."></textarea>
            <button onclick="addDocument()">Add Document</button>
        </div>
    </div>

    <script>
        async function search() {
            const query = document.getElementById('query').value;
            if (!query) return;
            
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = '<p>Searching...</p>';
            
            try {
                const response = await fetch('/api/search', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ query })
                });
                
                const data = await response.json();
                
                if (data.results.length === 0) {
                    resultsDiv.innerHTML = '<p>No results found.</p>';
                    return;
                }
                
                let html = '<h2>Search Results</h2>';
                data.results.forEach(result => {
                    const scorePercentage = (result.score * 100).toFixed(2);
                    html += `
                        <div class="result">
                            <p>${result.text}</p>
                            <div class="score">Relevance: ${scorePercentage}%</div>
                        </div>
                    `;
                });
                
                resultsDiv.innerHTML = html;
            } catch (error) {
                resultsDiv.innerHTML = `<p>Error: ${error.message}</p>`;
            }
        }
        
        async function addDocument() {
            const documentText = document.getElementById('document').value;
            if (!documentText) return;
            
            try {
                const response = await fetch('/api/index', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ document: documentText })
                });
                
                const data = await response.json();
                if (data.status === 'success') {
                    document.getElementById('document').value = '';
                    alert('Document added successfully!');
                }
            } catch (error) {
                alert(`Error adding document: ${error.message}`);
            }
        }
    </script>
</body>
</html>
```

### Step 4: Run the Application

Run the application:

```bash
python app.py
```

Open your browser and navigate to `http://localhost:5000` to use the semantic search application.

### Key Concepts

1. **Unified Web Framework**: The `UnifiedWebPlatform` class provides a consistent API for working with models across different hardware platforms.
2. **Platform Auto-Selection**: Setting `platform="auto"` automatically chooses the best available platform (WebGPU, WebNN, or CPU).
3. **Shader Precompilation**: Enabling shader precompilation improves initial inference speed by precompiling WebGPU shaders.
4. **Embedding Generation**: Text is converted to numerical vectors that capture semantic meaning.
5. **Cosine Similarity**: Used to measure the semantic similarity between documents.

## Tutorial 2: Image Classification with Vision Models

In this tutorial, we'll create a web application for image classification using a pre-trained vision model.

### Step 1: Set Up the Environment

Create a new project directory:

```bash
mkdir image-classifier
cd image-classifier
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install ipfs-accelerate-py flask Pillow
```

### Step 2: Create the Backend

Create a file named `app.py`:

```python
from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import base64
from fixed_web_platform.unified_web_framework import UnifiedWebPlatform

app = Flask(__name__)
vision_model = None

def initialize_model():
    global vision_model
    # Initialize vision model with WebGPU
    vision_model = UnifiedWebPlatform(
        model_name="vit-base-patch16-224",
        model_type="image_classification",
        platform="webgpu",
        fallback_to_webnn=True,
        enable_shader_precompilation=True  # For faster first inference
    )
    # Pre-load the model
    vision_model.load_model()
    print("Vision model initialized successfully")

# ImageNet class names (simplified for this example)
class_names = [
    "tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark",
    "electric ray", "stingray", "rooster", "hen", "ostrich", "brambling", "goldfinch",
    "house finch", "junco", "indigo bunting", "American robin", "bulbul", "jay",
    "magpie", "chickadee", "water ouzel", "kite", "bald eagle", "vulture",
    "great grey owl", "fire salamander", "smooth newt", "newt", "spotted salamander",
    "axolotl", "American bullfrog", "tree frog", "tailed frog", "loggerhead sea turtle",
    "leatherback sea turtle", "mud turtle", "terrapin", "box turtle", "banded gecko"
] # Truncated for brevity

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/classify', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    img = Image.open(file.stream)
    
    # Resize and preprocess the image
    img = img.resize((224, 224))
    img = img.convert('RGB')
    
    # Run inference
    result = vision_model.run_inference({"image": img})
    
    # Get top 5 predictions
    top_indices = result.argsort()[-5:][::-1]
    top_predictions = [
        {"class": class_names[idx] if idx < len(class_names) else f"Class {idx}", 
         "score": float(result[idx])}
        for idx in top_indices
    ]
    
    return jsonify({"predictions": top_predictions})

if __name__ == '__main__':
    initialize_model()
    app.run(debug=True)
```

### Step 3: Create the Frontend

Create a templates directory:

```bash
mkdir templates
```

Create `templates/index.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WebGPU Image Classifier</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        .container {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .upload-section {
            display: flex;
            flex-direction: column;
            gap: 10px;
            align-items: center;
        }
        #imagePreview {
            max-width: 400px;
            max-height: 400px;
            margin-top: 20px;
            border: 1px solid #ddd;
            display: none;
        }
        .prediction {
            display: flex;
            justify-content: space-between;
            padding: 10px;
            border: 1px solid #ddd;
            margin-bottom: 5px;
            border-radius: 5px;
        }
        .prediction-bar {
            background-color: #4CAF50;
            height: 20px;
            border-radius: 5px;
        }
        .prediction-label {
            font-weight: bold;
        }
        .prediction-score {
            text-align: right;
        }
        button {
            padding: 10px 20px;
            background-color: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
        }
        input[type="file"] {
            display: none;
        }
        .file-upload {
            display: inline-block;
            padding: 10px 20px;
            background-color: #2196F3;
            color: white;
            cursor: pointer;
            border-radius: 5px;
            margin-bottom: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>WebGPU Image Classifier</h1>
        <p>Upload an image to classify it using a vision transformer model accelerated with WebGPU/WebNN.</p>
        
        <div class="upload-section">
            <label class="file-upload">
                Choose Image
                <input type="file" id="imageInput" accept="image/*" onchange="previewImage()">
            </label>
            <button id="classifyBtn" onclick="classifyImage()" disabled>Classify Image</button>
            <img id="imagePreview" alt="Preview">
        </div>
        
        <div id="results" style="display: none;">
            <h2>Classification Results</h2>
            <div id="predictions"></div>
        </div>
    </div>

    <script>
        let selectedFile = null;
        
        function previewImage() {
            const input = document.getElementById('imageInput');
            const preview = document.getElementById('imagePreview');
            const classifyBtn = document.getElementById('classifyBtn');
            
            if (input.files && input.files[0]) {
                selectedFile = input.files[0];
                const reader = new FileReader();
                
                reader.onload = function(e) {
                    preview.src = e.target.result;
                    preview.style.display = 'block';
                    classifyBtn.disabled = false;
                }
                
                reader.readAsDataURL(input.files[0]);
            }
        }
        
        async function classifyImage() {
            if (!selectedFile) return;
            
            const resultsDiv = document.getElementById('results');
            const predictionsDiv = document.getElementById('predictions');
            
            resultsDiv.style.display = 'none';
            predictionsDiv.innerHTML = '<p>Classifying image...</p>';
            
            const formData = new FormData();
            formData.append('image', selectedFile);
            
            try {
                const response = await fetch('/api/classify', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.error) {
                    predictionsDiv.innerHTML = `<p>Error: ${data.error}</p>`;
                    return;
                }
                
                let html = '';
                data.predictions.forEach(prediction => {
                    const scorePercentage = (prediction.score * 100).toFixed(2);
                    html += `
                        <div class="prediction">
                            <div class="prediction-info">
                                <div class="prediction-label">${prediction.class}</div>
                                <div class="prediction-score">${scorePercentage}%</div>
                            </div>
                            <div class="prediction-bar" style="width: ${scorePercentage}%"></div>
                        </div>
                    `;
                });
                
                predictionsDiv.innerHTML = html;
                resultsDiv.style.display = 'block';
            } catch (error) {
                predictionsDiv.innerHTML = `<p>Error: ${error.message}</p>`;
            }
        }
    </script>
</body>
</html>
```

### Step 4: Run the Application

Run the application:

```bash
python app.py
```

Open your browser and navigate to `http://localhost:5000` to use the image classification application.

### Key Concepts

1. **Vision Model with WebGPU**: We use a Vision Transformer model accelerated with WebGPU.
2. **Fallback Mechanism**: If WebGPU is not available, the system falls back to WebNN.
3. **Shader Precompilation**: Improves initial inference time by precompiling shaders.
4. **Image Preprocessing**: Resizing and conversion to the format expected by the model.
5. **Top-K Predictions**: Returning the top 5 most likely classifications.

## Tutorial 3: Streaming Text Generation

In this tutorial, we'll create a text generation application with streaming output.

### Step 1: Set Up the Environment

Create a new project directory:

```bash
mkdir text-generator
cd text-generator
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install ipfs-accelerate-py flask flask-socketio
```

### Step 2: Create the Backend

Create a file named `app.py`:

```python
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from fixed_web_platform.unified_web_framework import UnifiedWebPlatform
from fixed_web_platform.webgpu_streaming_inference import StreamingInferencePipeline
import anyio
import threading

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
generator_model = None
streaming_pipeline = None

def initialize_models():
    global generator_model, streaming_pipeline
    
    # Initialize the model
    generator_model = UnifiedWebPlatform(
        model_name="t5-small",
        model_type="text_generation",
        platform="webgpu",
        fallback_to_webnn=True,
        enable_shader_precompilation=True
    )
    generator_model.load_model()
    
    # Initialize streaming pipeline
    streaming_pipeline = StreamingInferencePipeline(
        model=generator_model,
        max_length=100,
        stream_buffer_size=2  # Smaller buffer for more responsive streaming
    )
    
    print("Generator model and streaming pipeline initialized successfully")

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('generate_text')
def handle_generate_text(data):
    prompt = data.get('prompt', '')
    if not prompt:
        emit('error', {'message': 'Prompt is required'})
        return
    
    # Start generator in a thread to not block the main thread
    threading.Thread(target=generate_streaming_text, args=(prompt,)).start()

def generate_streaming_text(prompt):
    async def _generate():
        try:
            # Start with the prefix for T5
            input_text = f"translate English to French: {prompt}" if "translate" in prompt.lower() else prompt
            
            async for token in streaming_pipeline.generate_streaming(input_text):
                # Emit each token as it's generated
                socketio.emit('token', {'token': token})
            
            # Signal completion
            socketio.emit('generation_complete')
        except Exception as e:
            socketio.emit('error', {'message': str(e)})
    
    # Run the async function
    loop = anyio.new_event_loop()
    anyio.set_event_loop(loop)
    loop.run_until_complete(_generate())

if __name__ == '__main__':
    initialize_models()
    socketio.run(app, debug=True)
```

### Step 3: Create the Frontend

Create a templates directory:

```bash
mkdir templates
```

Create `templates/index.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WebGPU Streaming Text Generator</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        .container {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        textarea {
            width: 100%;
            height: 100px;
            padding: 10px;
            font-size: 16px;
        }
        #generateBtn {
            padding: 10px 20px;
            background-color: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
        }
        #generateBtn:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
        }
        .output-container {
            border: 1px solid #ddd;
            padding: 15px;
            border-radius: 5px;
            min-height: 100px;
            max-height: 400px;
            overflow-y: auto;
            background-color: #f9f9f9;
        }
        .blinking-cursor {
            display: inline-block;
            width: 10px;
            height: 20px;
            background-color: black;
            animation: blink 1s step-end infinite;
        }
        @keyframes blink {
            from, to { opacity: 1; }
            50% { opacity: 0; }
        }
        .examples {
            margin-top: 30px;
            border-top: 1px solid #ddd;
            padding-top: 20px;
        }
        .example {
            margin-bottom: 10px;
            cursor: pointer;
            color: #2196F3;
            text-decoration: underline;
        }
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
</head>
<body>
    <div class="container">
        <h1>WebGPU Streaming Text Generator</h1>
        <p>Enter a prompt to generate text with a streaming response.</p>
        
        <textarea id="promptInput" placeholder="Enter your prompt here..."></textarea>
        <button id="generateBtn" onclick="generateText()">Generate Text</button>
        
        <div>
            <h2>Generated Text</h2>
            <div class="output-container">
                <div id="outputText"></div>
                <span id="cursor" class="blinking-cursor"></span>
            </div>
        </div>
        
        <div class="examples">
            <h3>Example Prompts:</h3>
            <div class="example" onclick="useExample('Summarize: The Industrial Revolution was a period of major innovation that started in Great Britain and spread throughout the world during the late 1700s and early 1800s.')">Summarize a historical text</div>
            <div class="example" onclick="useExample('Translate English to French: The weather is beautiful today.')">Translate to French</div>
            <div class="example" onclick="useExample('Write a short poem about artificial intelligence.')">Generate a poem</div>
            <div class="example" onclick="useExample('Explain the concept of machine learning to a 10-year-old.')">Explain a concept</div>
        </div>
    </div>

    <script>
        const socket = io();
        const generateBtn = document.getElementById('generateBtn');
        const outputText = document.getElementById('outputText');
        const cursor = document.getElementById('cursor');
        
        // Socket.io event handlers
        socket.on('connect', () => {
            console.log('Connected to server');
        });
        
        socket.on('token', (data) => {
            // Add token to output
            outputText.textContent += data.token;
            
            // Scroll to bottom
            outputText.parentElement.scrollTop = outputText.parentElement.scrollHeight;
        });
        
        socket.on('generation_complete', () => {
            // Enable generate button
            generateBtn.disabled = false;
            generateBtn.textContent = 'Generate Text';
            
            // Hide cursor when complete
            cursor.style.display = 'none';
        });
        
        socket.on('error', (data) => {
            outputText.textContent = `Error: ${data.message}`;
            generateBtn.disabled = false;
            generateBtn.textContent = 'Generate Text';
            cursor.style.display = 'none';
        });
        
        function generateText() {
            const prompt = document.getElementById('promptInput').value.trim();
            if (!prompt) return;
            
            // Clear previous output
            outputText.textContent = '';
            cursor.style.display = 'inline-block';
            
            // Disable button during generation
            generateBtn.disabled = true;
            generateBtn.textContent = 'Generating...';
            
            // Send request to server
            socket.emit('generate_text', { prompt });
        }
        
        function useExample(text) {
            document.getElementById('promptInput').value = text;
        }
    </script>
</body>
</html>
```

### Step 4: Run the Application

Run the application:

```bash
python app.py
```

Open your browser and navigate to `http://localhost:5000` to use the streaming text generation application.

### Key Concepts

1. **Streaming Inference Pipeline**: The `StreamingInferencePipeline` class enables token-by-token generation.
2. **WebSockets**: Used to stream generated tokens to the frontend in real-time.
3. **Asynchronous Generation**: `async for` loop allows processing tokens as they're generated.
4. **User Experience**: Blinking cursor and immediate token display create a responsive typing effect.

## Tutorial 4: Audio Transcription with Whisper

In this tutorial, we'll create an application for transcribing audio using Whisper, optimized for Firefox with compute shaders.

### Step 1: Set Up the Environment

Create a new project directory:

```bash
mkdir audio-transcriber
cd audio-transcriber
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install ipfs-accelerate-py flask
```

### Step 2: Create the Backend

Create a file named `app.py`:

```python
from flask import Flask, request, jsonify, render_template
import os
import tempfile
from fixed_web_platform.unified_web_framework import UnifiedWebPlatform
from fixed_web_platform.webgpu_audio_compute_shaders import AudioComputeOptimizer

app = Flask(__name__)
transcription_model = None
audio_optimizer = None

def initialize_model():
    global transcription_model, audio_optimizer
    
    # Initialize the Whisper model
    transcription_model = UnifiedWebPlatform(
        model_name="whisper-tiny.en",
        model_type="audio_transcription",
        platform="webgpu",
        fallback_to_webnn=True,
        enable_shader_precompilation=True
    )
    transcription_model.load_model()
    
    # Initialize audio compute optimizer
    audio_optimizer = AudioComputeOptimizer(
        model_type="whisper",
        browser_specific=True  # Enable Firefox-specific optimizations
    )
    
    print("Transcription model initialized successfully")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/detect-browser', methods=['GET'])
def detect_browser():
    user_agent = request.headers.get('User-Agent', '').lower()
    
    if 'firefox' in user_agent:
        browser = 'firefox'
    elif 'chrome' in user_agent or 'edg' in user_agent:
        browser = 'chrome'
    elif 'safari' in user_agent and 'chrome' not in user_agent:
        browser = 'safari'
    else:
        browser = 'unknown'
    
    return jsonify({
        'browser': browser,
        'is_optimized': browser == 'firefox',
        'performance_gain': '20%' if browser == 'firefox' else '0%'
    })

@app.route('/api/transcribe', methods=['POST'])
def transcribe_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
        file.save(temp_file.name)
        temp_path = temp_file.name
    
    try:
        # Detect browser for optimizations
        user_agent = request.headers.get('User-Agent', '').lower()
        is_firefox = 'firefox' in user_agent
        
        # Apply browser-specific optimizations if Firefox
        if is_firefox:
            audio_optimizer.optimize_for_firefox()
            processed_audio = audio_optimizer.preprocess_audio(temp_path)
            result = transcription_model.run_inference({
                "audio": processed_audio,
                "use_optimized_compute": True
            })
        else:
            # Standard processing for other browsers
            result = transcription_model.run_inference({
                "audio": temp_path,
                "use_optimized_compute": False
            })
        
        # Format result
        transcription = result.get('text', '')
        segments = result.get('segments', [])
        
        formatted_segments = []
        for segment in segments:
            formatted_segments.append({
                'text': segment.get('text', ''),
                'start': segment.get('start', 0),
                'end': segment.get('end', 0)
            })
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        return jsonify({
            'transcription': transcription,
            'segments': formatted_segments,
            'browser_optimized': is_firefox
        })
    
    except Exception as e:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    initialize_model()
    app.run(debug=True)
```

### Step 3: Create the Frontend

Create a templates directory:

```bash
mkdir templates
```

Create `templates/index.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WebGPU Audio Transcription</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        .container {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .browser-info {
            background-color: #f0f8ff;
            padding: 15px;
            border-radius: 5px;
            border-left: 5px solid #4CAF50;
            margin-bottom: 20px;
        }
        .controls {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        button {
            padding: 10px 20px;
            background-color: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
            border-radius: 5px;
        }
        button:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
        }
        .record-btn.recording {
            background-color: #f44336;
        }
        .upload-btn {
            background-color: #2196F3;
        }
        input[type="file"] {
            display: none;
        }
        .transcription {
            border: 1px solid #ddd;
            padding: 15px;
            border-radius: 5px;
            min-height: 100px;
            max-height: 300px;
            overflow-y: auto;
            background-color: #f9f9f9;
        }
        .segments {
            margin-top: 20px;
        }
        .segment {
            padding: 10px;
            border-bottom: 1px solid #eee;
            cursor: pointer;
        }
        .segment:hover {
            background-color: #f0f0f0;
        }
        .segment-time {
            color: #666;
            font-size: 14px;
        }
        .audio-player {
            width: 100%;
            margin-top: 20px;
        }
        .loading {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }
        .spinner {
            border: 4px solid rgba(0, 0, 0, 0.1);
            border-left: 4px solid #4CAF50;
            border-radius: 50%;
            width: 20px;
            height: 20px;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>WebGPU Audio Transcription</h1>
        <p>Record or upload audio to transcribe using Whisper with WebGPU acceleration.</p>
        
        <div id="browserInfo" class="browser-info">
            Detecting browser...
        </div>
        
        <div class="controls">
            <button id="recordBtn" class="record-btn">Start Recording</button>
            <span>or</span>
            <label class="upload-btn" style="display: inline-block; padding: 10px 20px;">
                Upload Audio
                <input type="file" id="audioInput" accept="audio/*" onchange="handleFileUpload()">
            </label>
        </div>
        
        <div id="audioContainer" style="display: none;">
            <h3>Recorded Audio</h3>
            <audio id="audioPlayer" controls class="audio-player"></audio>
        </div>
        
        <div id="loadingContainer" style="display: none;" class="loading">
            <div class="spinner"></div>
            <p>Transcribing audio...</p>
        </div>
        
        <div id="resultContainer" style="display: none;">
            <h3>Transcription</h3>
            <div id="transcription" class="transcription"></div>
            
            <div class="segments">
                <h3>Segments</h3>
                <div id="segments"></div>
            </div>
        </div>
    </div>

    <script>
        // Variables for recording
        let mediaRecorder;
        let audioChunks = [];
        let audioBlob;
        let isRecording = false;
        
        // DOM elements
        const recordBtn = document.getElementById('recordBtn');
        const audioPlayer = document.getElementById('audioPlayer');
        const audioContainer = document.getElementById('audioContainer');
        const loadingContainer = document.getElementById('loadingContainer');
        const resultContainer = document.getElementById('resultContainer');
        const transcriptionDiv = document.getElementById('transcription');
        const segmentsDiv = document.getElementById('segments');
        const browserInfoDiv = document.getElementById('browserInfo');
        
        // Detect browser for optimizations
        async function detectBrowser() {
            try {
                const response = await fetch('/api/detect-browser');
                const data = await response.json();
                
                let browserName = data.browser.charAt(0).toUpperCase() + data.browser.slice(1);
                let message = `Detected browser: <strong>${browserName}</strong>`;
                
                if (data.is_optimized) {
                    message += ` <span style="color: green;">✓ Optimized for Firefox with ~${data.performance_gain} better performance</span>`;
                } else if (data.browser === 'firefox') {
                    message += ` <span style="color: green;">✓ Using Firefox-optimized compute shaders</span>`;
                } else {
                    message += ` <span style="color: orange;">Standard WebGPU implementation</span>`;
                    message += `<br><small>For best performance with audio models, consider using Firefox which offers ~20% better performance.</small>`;
                }
                
                browserInfoDiv.innerHTML = message;
            } catch (error) {
                browserInfoDiv.textContent = `Error detecting browser: ${error.message}`;
            }
        }
        
        // Initialize recording functionality
        async function initializeRecording() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                
                mediaRecorder = new MediaRecorder(stream);
                
                mediaRecorder.ondataavailable = (event) => {
                    audioChunks.push(event.data);
                };
                
                mediaRecorder.onstop = () => {
                    audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    const audioUrl = URL.createObjectURL(audioBlob);
                    audioPlayer.src = audioUrl;
                    audioContainer.style.display = 'block';
                    
                    // Auto-transcribe when recording stops
                    transcribeAudio(audioBlob);
                };
                
                recordBtn.addEventListener('click', toggleRecording);
            } catch (error) {
                alert(`Error accessing microphone: ${error.message}`);
                recordBtn.disabled = true;
            }
        }
        
        function toggleRecording() {
            if (isRecording) {
                // Stop recording
                mediaRecorder.stop();
                recordBtn.textContent = 'Start Recording';
                recordBtn.classList.remove('recording');
            } else {
                // Start recording
                audioChunks = [];
                mediaRecorder.start();
                recordBtn.textContent = 'Stop Recording';
                recordBtn.classList.add('recording');
                
                // Hide previous results
                resultContainer.style.display = 'none';
            }
            
            isRecording = !isRecording;
        }
        
        async function handleFileUpload() {
            const fileInput = document.getElementById('audioInput');
            if (fileInput.files.length === 0) return;
            
            const file = fileInput.files[0];
            
            // Create audio URL
            const audioUrl = URL.createObjectURL(file);
            audioPlayer.src = audioUrl;
            audioContainer.style.display = 'block';
            
            // Hide previous results
            resultContainer.style.display = 'none';
            
            // Transcribe the uploaded file
            transcribeAudio(file);
        }
        
        async function transcribeAudio(audioFile) {
            loadingContainer.style.display = 'flex';
            resultContainer.style.display = 'none';
            
            const formData = new FormData();
            formData.append('audio', audioFile);
            
            try {
                const response = await fetch('/api/transcribe', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.error) {
                    alert(`Error: ${data.error}`);
                    return;
                }
                
                // Display transcription
                transcriptionDiv.textContent = data.transcription;
                
                // Display segments
                segmentsDiv.innerHTML = '';
                data.segments.forEach(segment => {
                    const startTime = segment.start.toFixed(2);
                    const endTime = segment.end.toFixed(2);
                    
                    const segmentDiv = document.createElement('div');
                    segmentDiv.className = 'segment';
                    segmentDiv.innerHTML = `
                        <div>${segment.text}</div>
                        <div class="segment-time">${startTime}s - ${endTime}s</div>
                    `;
                    
                    // Add click handler to jump to segment
                    segmentDiv.addEventListener('click', () => {
                        audioPlayer.currentTime = segment.start;
                        audioPlayer.play();
                    });
                    
                    segmentsDiv.appendChild(segmentDiv);
                });
                
                resultContainer.style.display = 'block';
            } catch (error) {
                alert(`Error transcribing audio: ${error.message}`);
            } finally {
                loadingContainer.style.display = 'none';
            }
        }
        
        // Initialize on page load
        window.onload = () => {
            detectBrowser();
            initializeRecording();
        };
    </script>
</body>
</html>
```

### Step 4: Run the Application

Run the application:

```bash
python app.py
```

Open your browser and navigate to `http://localhost:5000` to use the audio transcription application.

### Key Concepts

1. **WebGPU Audio Compute Optimizations**: Firefox-specific optimizations for audio processing.
2. **Browser Detection**: Automatic detection of Firefox to apply optimized compute shaders.
3. **Audio Processing Pipeline**: Handling audio input from recording or file upload.
4. **Segmented Transcription**: Breaking down audio into timestamped segments.
5. **Interactive UI**: Clickable segments that jump to the corresponding audio position.

## Tutorial 5: Building a Multimodal Application

In this tutorial, we'll create a CLIP-based image-text similarity application with parallel loading optimization.

### Step 1: Set Up the Environment

Create a new project directory:

```bash
mkdir image-text-similarity
cd image-text-similarity
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install ipfs-accelerate-py flask Pillow
```

### Step 2: Create the Backend

Create a file named `app.py`:

```python
from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import base64
import time
from fixed_web_platform.unified_web_framework import UnifiedWebPlatform
from fixed_web_platform.progressive_model_loader import ParallelMultimodalLoader

app = Flask(__name__)
clip_model = None
parallel_loader = None

def initialize_model():
    global clip_model, parallel_loader
    
    # Initialize CLIP model with parallel loading
    parallel_loader = ParallelMultimodalLoader(
        model_path="clip-vit-base-patch32",
        components=["vision_encoder", "text_encoder"]
    )
    
    # Create the model with optimizations
    clip_model = UnifiedWebPlatform(
        model_name="clip-vit-base-patch32",
        model_type="multimodal",
        platform="webgpu",
        fallback_to_webnn=True,
        enable_shader_precompilation=True,
        enable_parallel_loading=True  # Enable parallel component loading
    )
    
    # Measure loading time
    start_time = time.time()
    clip_model.load_model()
    end_time = time.time()
    
    loading_time = end_time - start_time
    print(f"CLIP model loaded in {loading_time:.2f} seconds")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/compare', methods=['POST'])
def compare_image_text():
    # Check if request has the required parts
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    if 'texts' not in request.form:
        return jsonify({'error': 'No texts provided'}), 400
    
    # Get image file
    file = request.files['image']
    img = Image.open(file.stream)
    
    # Resize and preprocess image
    img = img.resize((224, 224))
    img = img.convert('RGB')
    
    # Get texts (comma-separated)
    texts = request.form['texts'].split('|')
    if not texts:
        return jsonify({'error': 'No texts provided'}), 400
    
    # Calculate image-text similarities
    results = []
    for text in texts:
        similarity = clip_model.run_inference({
            "image": img,
            "text": text
        })
        
        results.append({
            "text": text,
            "similarity": float(similarity)
        })
    
    # Sort by similarity (highest first)
    results.sort(key=lambda x: x["similarity"], reverse=True)
    
    # Return results
    return jsonify({"results": results})

if __name__ == '__main__':
    initialize_model()
    app.run(debug=True)
```

### Step 3: Create the Frontend

Create a templates directory:

```bash
mkdir templates
```

Create `templates/index.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image-Text Similarity with CLIP</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        .container {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .panel {
            display: flex;
            gap: 20px;
        }
        .image-panel {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .text-panel {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        #imagePreview {
            max-width: 100%;
            max-height: 300px;
            border: 1px solid #ddd;
            display: none;
        }
        textarea {
            width: 100%;
            height: 150px;
            padding: 10px;
            font-size: 16px;
        }
        button {
            padding: 10px 20px;
            background-color: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
            border-radius: 5px;
        }
        button:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
        }
        .upload-btn {
            background-color: #2196F3;
            display: inline-block;
            padding: 10px 20px;
            color: white;
            cursor: pointer;
            border-radius: 5px;
        }
        input[type="file"] {
            display: none;
        }
        .result {
            display: flex;
            justify-content: space-between;
            padding: 10px;
            border: 1px solid #ddd;
            margin-bottom: 5px;
            border-radius: 5px;
        }
        .result-bar {
            background-color: #4CAF50;
            height: 20px;
            border-radius: 5px;
        }
        .loading {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }
        .spinner {
            border: 4px solid rgba(0, 0, 0, 0.1);
            border-left: 4px solid #4CAF50;
            border-radius: 50%;
            width: 20px;
            height: 20px;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .examples {
            margin-top: 20px;
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }
        .example-tag {
            background-color: #f0f0f0;
            padding: 5px 10px;
            border-radius: 15px;
            cursor: pointer;
        }
        .example-tag:hover {
            background-color: #e0e0e0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Image-Text Similarity with CLIP</h1>
        <p>Upload an image and enter text descriptions to compare using CLIP, a multimodal model that connects images and text.</p>
        
        <div class="panel">
            <div class="image-panel">
                <h2>Image</h2>
                <label class="upload-btn">
                    Upload Image
                    <input type="file" id="imageInput" accept="image/*" onchange="previewImage()">
                </label>
                <img id="imagePreview" alt="Preview">
            </div>
            
            <div class="text-panel">
                <h2>Text Descriptions</h2>
                <p>Enter multiple descriptions separated by | (pipe)</p>
                <textarea id="textInput" placeholder="Enter text descriptions separated by | (pipe)
Example:
a dog running in a field|a cat sleeping on a couch|a person hiking on a mountain"></textarea>
                
                <div class="examples">
                    <span class="example-tag" onclick="addText('a dog')">dog</span>
                    <span class="example-tag" onclick="addText('a cat')">cat</span>
                    <span class="example-tag" onclick="addText('a person')">person</span>
                    <span class="example-tag" onclick="addText('a mountain')">mountain</span>
                    <span class="example-tag" onclick="addText('a beach')">beach</span>
                    <span class="example-tag" onclick="addText('a forest')">forest</span>
                    <span class="example-tag" onclick="addText('a city')">city</span>
                    <span class="example-tag" onclick="addText('food')">food</span>
                </div>
            </div>
        </div>
        
        <button id="compareBtn" onclick="compareImageText()" disabled>Compare Image and Text</button>
        
        <div id="loadingContainer" style="display: none;" class="loading">
            <div class="spinner"></div>
            <p>Processing...</p>
        </div>
        
        <div id="resultContainer" style="display: none;">
            <h2>Similarity Results</h2>
            <div id="results"></div>
        </div>
    </div>

    <script>
        let selectedImage = null;
        const compareBtn = document.getElementById('compareBtn');
        const loadingContainer = document.getElementById('loadingContainer');
        const resultContainer = document.getElementById('resultContainer');
        const resultsDiv = document.getElementById('results');
        
        function previewImage() {
            const input = document.getElementById('imageInput');
            const preview = document.getElementById('imagePreview');
            
            if (input.files && input.files[0]) {
                selectedImage = input.files[0];
                const reader = new FileReader();
                
                reader.onload = function(e) {
                    preview.src = e.target.result;
                    preview.style.display = 'block';
                    updateCompareButton();
                }
                
                reader.readAsDataURL(input.files[0]);
            }
        }
        
        function addText(text) {
            const textInput = document.getElementById('textInput');
            
            if (textInput.value.trim() === '') {
                textInput.value = text;
            } else if (textInput.value.endsWith('|')) {
                textInput.value += text;
            } else {
                textInput.value += '|' + text;
            }
            
            updateCompareButton();
        }
        
        function updateCompareButton() {
            const textInput = document.getElementById('textInput');
            compareBtn.disabled = !selectedImage || !textInput.value.trim();
        }
        
        document.getElementById('textInput').addEventListener('input', updateCompareButton);
        
        async function compareImageText() {
            if (!selectedImage) return;
            
            const textInput = document.getElementById('textInput').value.trim();
            if (!textInput) return;
            
            loadingContainer.style.display = 'flex';
            resultContainer.style.display = 'none';
            
            const formData = new FormData();
            formData.append('image', selectedImage);
            formData.append('texts', textInput);
            
            try {
                const response = await fetch('/api/compare', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.error) {
                    alert(`Error: ${data.error}`);
                    return;
                }
                
                // Display results
                resultsDiv.innerHTML = '';
                data.results.forEach(result => {
                    const scorePercentage = (result.similarity * 100).toFixed(2);
                    
                    const resultDiv = document.createElement('div');
                    resultDiv.className = 'result';
                    resultDiv.innerHTML = `
                        <div class="result-info">
                            <div>${result.text}</div>
                            <div>${scorePercentage}%</div>
                        </div>
                        <div class="result-bar" style="width: ${scorePercentage}%"></div>
                    `;
                    
                    resultsDiv.appendChild(resultDiv);
                });
                
                resultContainer.style.display = 'block';
            } catch (error) {
                alert(`Error comparing image and text: ${error.message}`);
            } finally {
                loadingContainer.style.display = 'none';
            }
        }
    </script>
</body>
</html>
```

### Step 4: Run the Application

Run the application:

```bash
python app.py
```

Open your browser and navigate to `http://localhost:5000` to use the image-text similarity application.

### Key Concepts

1. **Parallel Multimodal Loading**: Using `ParallelMultimodalLoader` to load vision and text encoders concurrently.
2. **CLIP Model**: Connecting images and text in a shared embedding space.
3. **Similarity Calculation**: Computing similarity between image and multiple text descriptions.
4. **Interactive UI**: Adding text descriptions via tags and visualizing similarity scores.

## Advanced Techniques

### Memory Optimization for Large Models

For larger models that push the limits of browser memory:

```python
from fixed_web_platform.webgpu_memory_optimization import WebGPUMemoryOptimizer

# Initialize memory optimizer
memory_optimizer = WebGPUMemoryOptimizer(platform=platform)

# Enable aggressive memory optimization
memory_optimizer.enable_aggressive_optimization()

# Set memory limits
memory_optimizer.set_max_memory_usage_mb(1024)  # 1GB limit

# Enable KV-cache optimization for LLMs
memory_optimizer.enable_kv_cache_optimization()

# Run memory-optimized inference
result = memory_optimizer.run_memory_optimized(
    input_data={"input_text": "Sample text"},
    batch_size=1
)
```

### Quantization for Reduced Memory Footprint

Apply quantization to reduce model size and memory requirements:

```python
from fixed_web_platform.webgpu_quantization import WebGPUQuantizer

# Initialize quantizer
quantizer = WebGPUQuantizer(model_path="llama-7b")

# Quantize model to 4-bit precision
quantized_model = quantizer.quantize(precision="int4")

# Load quantized model
platform = UnifiedWebPlatform(
    model=quantized_model,
    platform="webgpu"
)

# Run inference with quantized model
result = platform.run_inference({"input_text": "Sample text"})
```

### Multi-Model Pipeline

Combine multiple models in a pipeline:

```python
from fixed_web_platform.unified_web_framework import ModelPipeline

# Create pipeline with multiple models
pipeline = ModelPipeline()

# Add models to pipeline
pipeline.add_model(
    name="whisper",
    model_type="audio_transcription",
    platform="webgpu"
)
pipeline.add_model(
    name="t5-small",
    model_type="text_translation",
    platform="webgpu"
)

# Run pipeline
result = pipeline.run({
    "audio": "audio_file.wav",
    "source_lang": "en",
    "target_lang": "fr"
})

# Access individual model results
transcription = result["whisper"]["text"]
translation = result["t5-small"]["translation"]
```

## Deployment Guidelines

### Packaging for Web Deployment

To deploy your application to production:

1. **Create a Production-Ready Flask Application**:

   ```python
   # app.py
   from flask import Flask, request, jsonify, render_template
   from waitress import serve  # Production WSGI server
   
   app = Flask(__name__)
   
   # ... your application code ...
   
   if __name__ == "__main__":
       # Development
       if app.config.get("ENV") == "development":
           app.run(debug=True)
       # Production
       else:
           serve(app, host='0.0.0.0', port=8080)
   ```

2. **Set Up Environment Variables**:

   ```bash
   # .env file
   FLASK_ENV=production
   MODEL_CACHE_DIR=/path/to/model/cache
   ```

3. **Create a Production Requirements File**:

   ```
   # requirements.txt
   ipfs-accelerate-py==1.0.0
   flask==2.0.1
   waitress==2.0.0
   python-dotenv==0.19.0
   ```

4. **Docker Deployment**:

   ```dockerfile
   # Dockerfile
   FROM python:3.9-slim
   
   WORKDIR /app
   
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   
   COPY . .
   
   ENV FLASK_ENV=production
   
   EXPOSE 8080
   
   CMD ["python", "app.py"]
   ```

5. **Build and Run Docker Container**:

   ```bash
   docker build -t web-accelerated-ai .
   docker run -p 8080:8080 web-accelerated-ai
   ```

### Browser Compatibility Considerations

For optimal cross-browser compatibility:

1. **Feature Detection**:
   ```javascript
   async function checkWebGPUSupport() {
       if (!navigator.gpu) {
           return {supported: false, message: "WebGPU not supported"};
       }
       
       try {
           const adapter = await navigator.gpu.requestAdapter();
           if (!adapter) {
               return {supported: false, message: "Couldn't request WebGPU adapter"};
           }
           
           return {supported: true, adapter: adapter};
       } catch (error) {
           return {supported: false, message: error.message};
       }
   }
   ```

2. **Graceful Degradation**:
   ```javascript
   async function initializeModel() {
       const webGPUSupport = await checkWebGPUSupport();
       
       if (webGPUSupport.supported) {
           // Use WebGPU accelerated model
           return await loadWebGPUModel();
       } else if (window.ml && window.ml.webnn) {
           // Fall back to WebNN
           return await loadWebNNModel();
       } else {
           // Fall back to WASM
           return await loadWASMModel();
       }
   }
   ```

3. **Browser-Specific Optimizations**:
   ```javascript
   function detectBrowser() {
       const userAgent = navigator.userAgent.toLowerCase();
       
       if (userAgent.includes('firefox')) {
           return 'firefox';
       } else if (userAgent.includes('chrome') || userAgent.includes('edg')) {
           return 'chrome';
       } else if (userAgent.includes('safari') && !userAgent.includes('chrome')) {
           return 'safari';
       } else {
           return 'unknown';
       }
   }
   
   function applyBrowserOptimizations(browser) {
       switch (browser) {
           case 'firefox':
               // Apply Firefox optimizations
               return {workgroupSize: '256,1,1', useComputeShaders: true};
           
           case 'chrome':
               // Apply Chrome optimizations
               return {workgroupSize: '128,2,1', useComputeShaders: true};
           
           case 'safari':
               // Apply Safari optimizations
               return {workgroupSize: '64,4,1', useComputeShaders: false};
           
           default:
               // Generic fallback
               return {workgroupSize: '128,1,1', useComputeShaders: false};
       }
   }
   ```

## Resources and References

- [IPFS Accelerate Python Framework Documentation](https://docs.ipfs-accelerate.example.com)
- [WebGPU Specification](https://www.w3.org/TR/webgpu/)
- [WebNN API Reference](https://webmachinelearning.github.io/webnn/)
- [Browser-Specific Optimizations Guide](browser_specific_optimizations.md)
- [Error Handling Guide](ERROR_HANDLING_GUIDE.md)
- [WebGPU Implementation Guide](WEBGPU_IMPLEMENTATION_GUIDE.md)
- [Model-Specific Optimization Guides](model_specific_optimizations/)
  - [Text Models](model_specific_optimizations/text_models.md)
  - [Vision Models](model_specific_optimizations/vision_models.md)
  - [Audio Models](model_specific_optimizations/audio_models.md)
  - [Multimodal Models](model_specific_optimizations/multimodal_models.md)
- [Web Platform Quick Start Guide](WEB_PLATFORM_QUICK_START.md)