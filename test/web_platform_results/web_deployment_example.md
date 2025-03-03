# Web Deployment Example: Deploying a BERT Model to Browsers

This document provides a complete worked example of deploying a BERT model to web browsers using both WebNN and WebGPU backends. Follow this step-by-step guide to create a working web demo.

## Overview

In this example, we'll:

1. Export a BERT model for sentiment analysis
2. Set up a simple web server
3. Create a web application with both WebNN and WebGPU implementations
4. Compare performance between implementations

## Prerequisites

Before starting, ensure you have:

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- ONNX Runtime 1.14+
- Node.js 14+
- A modern browser with WebNN or WebGPU support

## Step 1: Export the Model

First, we'll export a BERT model for sentiment analysis using our export script:

```bash
# Navigate to the test directory
cd /home/barberb/ipfs_accelerate_py/test

# Run the export script
python web_platform_results/export_model_for_web.py \
  --model Xenova/bert-base-uncased-sentiment \
  --output-dir web_platform_results/models \
  --precision fp16 \
  --optimize
```

This will create:
- `web_platform_results/models/webnn/bert-base-uncased-sentiment/` - Contains the ONNX model
- `web_platform_results/models/webgpu/bert-base-uncased-sentiment/` - Contains files for transformers.js

## Step 2: Create a Web Server

Create a simple server to host our application:

```bash
# Install http-server if not already installed
npm install -g http-server

# Start a server in the test directory
cd /home/barberb/ipfs_accelerate_py/test
http-server . -p 8080
```

## Step 3: Create HTML Implementation

Create a file named `web_demo.html` with the following content:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BERT Sentiment Analysis Demo</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .container {
            display: flex;
            gap: 20px;
        }
        .implementation {
            flex: 1;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 20px;
            background-color: #f9f9f9;
        }
        textarea {
            width: 100%;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 4px;
            margin-bottom: 10px;
        }
        button {
            background-color: #1a73e8;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
        }
        .results {
            margin-top: 20px;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 4px;
            background-color: white;
        }
        .perf-info {
            font-size: 0.9em;
            color: #666;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <h1>BERT Sentiment Analysis Demo</h1>
    <p>Compare WebNN and WebGPU implementations for sentiment analysis</p>
    
    <div class="container">
        <!-- WebNN Implementation -->
        <div class="implementation">
            <h2>WebNN Implementation</h2>
            
            <textarea id="webnn-input" rows="4">This movie is amazing! I really enjoyed it.</textarea>
            <button id="webnn-run">Analyze Sentiment</button>
            <div id="webnn-status" class="status">Loading model...</div>
            
            <div class="results">
                <h3>Results:</h3>
                <pre id="webnn-output">Results will appear here</pre>
                <div id="webnn-perf" class="perf-info"></div>
            </div>
        </div>
        
        <!-- WebGPU Implementation -->
        <div class="implementation">
            <h2>WebGPU Implementation</h2>
            
            <textarea id="webgpu-input" rows="4">This movie is amazing! I really enjoyed it.</textarea>
            <button id="webgpu-run">Analyze Sentiment</button>
            <div id="webgpu-status" class="status">Loading model...</div>
            
            <div class="results">
                <h3>Results:</h3>
                <pre id="webgpu-output">Results will appear here</pre>
                <div id="webgpu-perf" class="perf-info"></div>
            </div>
        </div>
    </div>
    
    <!-- Load ONNX Runtime for WebNN -->
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/ort.min.js"></script>
    
    <!-- Load transformers.js for WebGPU -->
    <script src="https://cdn.jsdelivr.net/npm/@xenova/transformers@2.6.0"></script>
    
    <script>
        // WebNN Implementation
        async function initWebNN() {
            try {
                // Load model
                const session = await ort.InferenceSession.create(
                    'web_platform_results/models/webnn/bert-base-uncased-sentiment/model.onnx', 
                    { executionProviders: ['wasm'] }
                );
                
                // Load tokenizer info
                const tokenizer = await fetch('web_platform_results/models/webnn/bert-base-uncased-sentiment/tokenizer.json')
                    .then(res => res.json());
                
                document.getElementById('webnn-status').textContent = 'Model loaded';
                return { session, tokenizer };
            } catch (error) {
                console.error('Error loading WebNN model:', error);
                document.getElementById('webnn-status').textContent = 'Error loading model';
                throw error;
            }
        }
        
        async function runWebNN(text) {
            try {
                const startTime = performance.now();
                
                // Get model and tokenizer
                if (!window.webnnModel) {
                    window.webnnModel = await initWebNN();
                }
                const { session, tokenizer } = window.webnnModel;
                
                // Tokenize input
                const encoded = tokenizeForONNX(text, tokenizer);
                
                // Run inference
                const results = await session.run({
                    input_ids: new ort.Tensor('int64', encoded.input_ids, [1, encoded.input_ids.length]),
                    attention_mask: new ort.Tensor('int64', encoded.attention_mask, [1, encoded.attention_mask.length]),
                    token_type_ids: new ort.Tensor('int64', encoded.token_type_ids, [1, encoded.token_type_ids.length])
                });
                
                // Process results
                const logits = results.logits.data;
                const scores = Array.from(logits).map(x => Math.exp(x));
                const sum = scores.reduce((a, b) => a + b, 0);
                const probs = scores.map(x => x / sum);
                
                // Format results - 1 to 5 stars
                const labels = ['1 star', '2 stars', '3 stars', '4 stars', '5 stars'];
                const sentiments = labels.map((label, i) => ({
                    label,
                    score: probs[i]
                })).sort((a, b) => b.score - a.score);
                
                const endTime = performance.now();
                
                return { 
                    results: sentiments,
                    performance: { time: endTime - startTime }
                };
            } catch (error) {
                console.error('WebNN inference error:', error);
                throw error;
            }
        }
        
        // Helper function to tokenize text for ONNX model
        function tokenizeForONNX(text, tokenizer) {
            const tokens = ['[CLS]', ...text.toLowerCase().split(/\s+/).slice(0, 126), '[SEP]'];
            const max_length = 128;
            
            // Convert tokens to ids using vocabulary
            const input_ids = tokens.map(token => tokenizer.vocab[token] || tokenizer.vocab['[UNK]']);
            
            // Pad to max_length
            while (input_ids.length < max_length) input_ids.push(0);
            
            // Create masks
            const attention_mask = input_ids.map(id => id > 0 ? 1 : 0);
            const token_type_ids = new Array(max_length).fill(0);
            
            return {
                input_ids: input_ids,
                attention_mask: attention_mask,
                token_type_ids: token_type_ids
            };
        }
        
        // WebGPU/transformers.js Implementation
        const { pipeline, env } = window.transformers;
        env.useBrowserCache = true;
        env.useWebGPU = true;
        
        async function initWebGPU() {
            try {
                const classifier = await pipeline('sentiment-analysis', 'Xenova/bert-base-uncased-sentiment');
                document.getElementById('webgpu-status').textContent = 'Model loaded';
                return classifier;
            } catch (error) {
                console.error('Error loading WebGPU model:', error);
                document.getElementById('webgpu-status').textContent = 'Error loading model';
                throw error;
            }
        }
        
        async function runWebGPU(text) {
            try {
                const startTime = performance.now();
                
                // Get model
                if (!window.webgpuModel) {
                    window.webgpuModel = await initWebGPU();
                }
                
                // Run inference
                const result = await window.webgpuModel(text);
                
                const endTime = performance.now();
                
                return { 
                    results: result,
                    performance: { time: endTime - startTime } 
                };
            } catch (error) {
                console.error('WebGPU inference error:', error);
                throw error;
            }
        }
        
        // Initialize both implementations
        initWebNN();
        initWebGPU();
        
        // Set up event listeners
        document.getElementById('webnn-run').addEventListener('click', async () => {
            try {
                const text = document.getElementById('webnn-input').value;
                document.getElementById('webnn-output').textContent = 'Running...';
                
                const result = await runWebNN(text);
                
                document.getElementById('webnn-output').textContent = JSON.stringify(result.results, null, 2);
                document.getElementById('webnn-perf').textContent = 
                    `Inference time: ${result.performance.time.toFixed(2)}ms`;
            } catch (error) {
                document.getElementById('webnn-output').textContent = `Error: ${error.message}`;
            }
        });
        
        document.getElementById('webgpu-run').addEventListener('click', async () => {
            try {
                const text = document.getElementById('webgpu-input').value;
                document.getElementById('webgpu-output').textContent = 'Running...';
                
                const result = await runWebGPU(text);
                
                document.getElementById('webgpu-output').textContent = JSON.stringify(result.results, null, 2);
                document.getElementById('webgpu-perf').textContent = 
                    `Inference time: ${result.performance.time.toFixed(2)}ms`;
            } catch (error) {
                document.getElementById('webgpu-output').textContent = `Error: ${error.message}`;
            }
        });
    </script>
</body>
</html>
```

Save this file to `/home/barberb/ipfs_accelerate_py/test/web_platform_results/web_demo.html`.

## Step 4: Test the Implementation

1. Start the web server if not already running:
   ```bash
   cd /home/barberb/ipfs_accelerate_py/test
   http-server . -p 8080
   ```

2. Open a browser and navigate to:
   ```
   http://localhost:8080/web_platform_results/web_demo.html
   ```

3. Try entering different text examples to see sentiment analysis in action.

## Step 5: Optimize the Implementation

For better performance, you can apply these optimizations:

### WebNN Optimizations

1. **Use Quantization**: Export with int8 precision
   ```bash
   python web_platform_results/export_model_for_web.py \
     --model Xenova/bert-base-uncased-sentiment \
     --format webnn \
     --precision int8 \
     --optimize
   ```

2. **Use Smaller Model**: Try DistilBERT instead of BERT
   ```bash
   python web_platform_results/export_model_for_web.py \
     --model distilbert-base-uncased-finetuned-sst-2-english \
     --format webnn \
     --optimize
   ```

### WebGPU Optimizations

1. **Enable Quantization**:
   ```bash
   python web_platform_results/export_model_for_web.py \
     --model Xenova/bert-base-uncased-sentiment \
     --format webgpu \
     --quantized
   ```

2. **Use Batching** for multiple inputs:
   ```javascript
   const results = await classifier([
     'This movie is great!',
     'I did not like the acting',
     'The special effects were amazing'
   ]);
   ```

## Advanced: Using Web Workers

For better UI responsiveness, move model inference to a Web Worker:

1. Create a file named `worker.js`:

```javascript
// Import ONNX Runtime
importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/ort.min.js');

// Store model sessions
let onnxSession = null;
let tokenizer = null;

// Handle messages from main thread
self.addEventListener('message', async function(e) {
    const { type, data } = e.data;
    
    if (type === 'load') {
        try {
            // Load ONNX model
            onnxSession = await ort.InferenceSession.create(
                data.modelPath, 
                { executionProviders: ['wasm'] }
            );
            
            // Load tokenizer
            const response = await fetch(data.tokenizerPath);
            tokenizer = await response.json();
            
            self.postMessage({ type: 'loaded' });
        } catch (error) {
            self.postMessage({ type: 'error', error: error.message });
        }
    }
    else if (type === 'run') {
        try {
            // Ensure model is loaded
            if (!onnxSession || !tokenizer) {
                throw new Error('Model not loaded');
            }
            
            // Tokenize
            const encoded = tokenizeInput(data.text, tokenizer);
            
            // Run inference
            const results = await onnxSession.run({
                input_ids: new ort.Tensor('int64', encoded.input_ids, [1, encoded.input_ids.length]),
                attention_mask: new ort.Tensor('int64', encoded.attention_mask, [1, encoded.attention_mask.length]),
                token_type_ids: new ort.Tensor('int64', encoded.token_type_ids, [1, encoded.token_type_ids.length])
            });
            
            // Process results
            const logits = results.logits.data;
            const scores = Array.from(logits).map(x => Math.exp(x));
            const sum = scores.reduce((a, b) => a + b, 0);
            const probs = scores.map(x => x / sum);
            
            // Format results - 1 to 5 stars
            const labels = ['1 star', '2 stars', '3 stars', '4 stars', '5 stars'];
            const sentiments = labels.map((label, i) => ({
                label,
                score: probs[i]
            })).sort((a, b) => b.score - a.score);
            
            self.postMessage({ 
                type: 'result', 
                results: sentiments 
            });
        } catch (error) {
            self.postMessage({ type: 'error', error: error.message });
        }
    }
});

// Helper function to tokenize input
function tokenizeInput(text, tokenizer) {
    const tokens = ['[CLS]', ...text.toLowerCase().split(/\s+/).slice(0, 126), '[SEP]'];
    const max_length = 128;
    
    // Convert tokens to ids
    const input_ids = tokens.map(token => tokenizer.vocab[token] || tokenizer.vocab['[UNK]']);
    
    // Pad to max_length
    while (input_ids.length < max_length) input_ids.push(0);
    
    // Create masks
    const attention_mask = input_ids.map(id => id > 0 ? 1 : 0);
    const token_type_ids = new Array(max_length).fill(0);
    
    return {
        input_ids: input_ids,
        attention_mask: attention_mask,
        token_type_ids: token_type_ids
    };
}
```

2. Update your HTML to use the worker:

```javascript
// Initialize worker
const worker = new Worker('worker.js');

// Set up worker message handler
worker.addEventListener('message', function(e) {
    const { type, results, error } = e.data;
    
    if (type === 'loaded') {
        document.getElementById('webnn-status').textContent = 'Model loaded';
    }
    else if (type === 'result') {
        document.getElementById('webnn-output').textContent = JSON.stringify(results, null, 2);
    }
    else if (type === 'error') {
        document.getElementById('webnn-output').textContent = `Error: ${error}`;
    }
});

// Load the model
worker.postMessage({
    type: 'load',
    data: {
        modelPath: 'web_platform_results/models/webnn/bert-base-uncased-sentiment/model.onnx',
        tokenizerPath: 'web_platform_results/models/webnn/bert-base-uncased-sentiment/tokenizer.json'
    }
});

// Set up run button
document.getElementById('webnn-run').addEventListener('click', function() {
    const text = document.getElementById('webnn-input').value;
    document.getElementById('webnn-output').textContent = 'Running...';
    
    // Send text to worker
    worker.postMessage({
        type: 'run',
        data: { text }
    });
});
```

## Performance Analysis

To compare performance between WebNN and WebGPU implementations, add a benchmark button to your HTML:

```html
<div style="margin-top: 30px; text-align: center;">
    <button id="run-benchmark">Run Benchmark (10 iterations)</button>
    <div class="results" style="margin-top: 20px;">
        <h3>Benchmark Results:</h3>
        <pre id="benchmark-results">Run the benchmark to see results</pre>
    </div>
</div>

<script>
    // Add this to the script section
    document.getElementById('run-benchmark').addEventListener('click', async () => {
        const iterations = 10;
        const testText = "This movie was fantastic. The acting was superb and the special effects were amazing!";
        
        document.getElementById('benchmark-results').textContent = 'Running benchmark...';
        
        // Results storage
        const webnnTimes = [];
        const webgpuTimes = [];
        
        // Run tests
        for (let i = 0; i < iterations; i++) {
            // WebNN inference
            try {
                const webnnResult = await runWebNN(testText);
                webnnTimes.push(webnnResult.performance.time);
            } catch (e) {
                console.error('WebNN benchmark error:', e);
            }
            
            // WebGPU inference
            try {
                const webgpuResult = await runWebGPU(testText);
                webgpuTimes.push(webgpuResult.performance.time);
            } catch (e) {
                console.error('WebGPU benchmark error:', e);
            }
        }
        
        // Calculate statistics
        const webnnAvg = webnnTimes.reduce((a, b) => a + b, 0) / webnnTimes.length;
        const webgpuAvg = webgpuTimes.reduce((a, b) => a + b, 0) / webgpuTimes.length;
        const speedup = webnnAvg / webgpuAvg;
        
        // Format results
        const results = {
            iterations: iterations,
            webnn: {
                avg_time_ms: webnnAvg.toFixed(2),
                min_time_ms: Math.min(...webnnTimes).toFixed(2),
                max_time_ms: Math.max(...webnnTimes).toFixed(2)
            },
            webgpu: {
                avg_time_ms: webgpuAvg.toFixed(2),
                min_time_ms: Math.min(...webgpuTimes).toFixed(2),
                max_time_ms: Math.max(...webgpuTimes).toFixed(2)
            },
            comparison: {
                speedup: speedup.toFixed(2),
                faster_implementation: speedup > 1 ? "WebGPU" : "WebNN"
            }
        };
        
        // Display results
        document.getElementById('benchmark-results').textContent = 
            JSON.stringify(results, null, 2);
    });
</script>
```

## Next Steps

After completing this example, you can:

1. **Try Different Models**: Export and test different model types (Vision, Audio, etc.)
2. **Implement Progressive Loading**: Add loading indicators and progressive model loading
3. **Deploy to a Production Server**: Set up proper caching headers for optimal performance
4. **Add Service Worker**: Implement offline capabilities with service worker caching
5. **Create an End-to-End Application**: Build a complete web application with backend API integration

## Troubleshooting

### Common Issues

1. **CORS Errors**: If you see CORS errors, make sure your server is sending the proper headers.
2. **Out of Memory**: Try a smaller model or enable quantization.
3. **WebGPU Not Available**: Check browser compatibility and make sure WebGPU is enabled.
4. **Slow First Inference**: The first inference is often slower due to compilation time.

### Browser Support

For optimal experience, use:
- Chrome 113+ for both WebNN and WebGPU
- Edge 113+ for both WebNN and WebGPU
- Safari 16.4+ for WebGPU only (WebNN via polyfill)
- Firefox 115+ for WebGPU with flag enabled (WebNN via polyfill)