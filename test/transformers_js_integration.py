#!/usr/bin/env python3
"""
Transformers.js Integration for WebNN/WebGPU

This script demonstrates how to integrate with transformers.js to provide
real model inference capabilities, even when the browser doesn't support
WebNN or WebGPU natively.

It creates a browser-based environment where transformers.js can run inference
and communicates with Python via a WebSocket bridge.
"""

import os
import sys
import json
import time
import asyncio
import logging
import tempfile
import argparse
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Try to import websockets
try:
    import websockets
except ImportError:
    print("websockets package is required. Install with: pip install websockets")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# HTML template for browser integration
TRANSFORMERS_JS_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Transformers.js Integration</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            line-height: 1.6;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .status {
            padding: 10px;
            margin: 10px 0;
            border: 1px solid #ccc;
            background-color: #f8f8f8;
        }
        .log {
            height: 300px;
            overflow-y: auto;
            border: 1px solid #ccc;
            padding: 10px;
            margin-top: 10px;
            font-family: monospace;
        }
        .log-entry {
            margin-bottom: 5px;
        }
        .error {
            color: red;
        }
        .warning {
            color: orange;
        }
        .success {
            color: green;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Transformers.js Integration</h1>
        
        <div class="status" id="status">
            <h2>Status: Initializing...</h2>
        </div>
        
        <div class="status">
            <h2>Feature Detection</h2>
            <div id="features">
                <p>WebGPU: <span id="webgpu-status">Checking...</span></p>
                <p>WebNN: <span id="webnn-status">Checking...</span></p>
                <p>WebAssembly: <span id="wasm-status">Checking...</span></p>
            </div>
        </div>
        
        <div class="status">
            <h2>Inference</h2>
            <div id="inference-status">Waiting for inference request...</div>
        </div>
        
        <div class="log" id="log">
            <!-- Log entries will be added here -->
        </div>
    </div>
    
    <script type="module">
        // Main state
        const state = {
            features: {
                webgpu: false,
                webnn: false,
                wasm: false
            },
            models: {},
            pipeline: null,
            transformers: null
        };
        
        // Logging function
        function log(message, level = 'info') {
            const logElement = document.getElementById('log');
            const logEntry = document.createElement('div');
            logEntry.className = `log-entry ${level}`;
            logEntry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
            logElement.appendChild(logEntry);
            logElement.scrollTop = logElement.scrollHeight;
            console.log(`[${level}] ${message}`);
        }
        
        // Update status
        function updateStatus(message) {
            document.getElementById('status').innerHTML = `<h2>Status: ${message}</h2>`;
        }
        
        // Feature detection
        async function detectFeatures() {
            // WebGPU detection
            const webgpuStatus = document.getElementById('webgpu-status');
            if ('gpu' in navigator) {
                try {
                    const adapter = await navigator.gpu.requestAdapter();
                    if (adapter) {
                        const device = await adapter.requestDevice();
                        if (device) {
                            webgpuStatus.textContent = 'Available';
                            webgpuStatus.className = 'success';
                            state.features.webgpu = true;
                            log('WebGPU is available', 'success');
                        } else {
                            webgpuStatus.textContent = 'Device not available';
                            webgpuStatus.className = 'warning';
                            log('WebGPU device not available', 'warning');
                        }
                    } else {
                        webgpuStatus.textContent = 'Adapter not available';
                        webgpuStatus.className = 'warning';
                        log('WebGPU adapter not available', 'warning');
                    }
                } catch (error) {
                    webgpuStatus.textContent = 'Error: ' + error.message;
                    webgpuStatus.className = 'error';
                    log('WebGPU error: ' + error.message, 'error');
                }
            } else {
                webgpuStatus.textContent = 'Not supported';
                webgpuStatus.className = 'error';
                log('WebGPU is not supported in this browser', 'warning');
            }
            
            // WebNN detection
            const webnnStatus = document.getElementById('webnn-status');
            if ('ml' in navigator) {
                try {
                    // Check for specific backends
                    const backends = [];
                    
                    // Try CPU backend
                    try {
                        const cpuContext = await navigator.ml.createContext({ devicePreference: 'cpu' });
                        if (cpuContext) {
                            backends.push('cpu');
                        }
                    } catch (e) {
                        // CPU backend not available
                    }
                    
                    // Try GPU backend
                    try {
                        const gpuContext = await navigator.ml.createContext({ devicePreference: 'gpu' });
                        if (gpuContext) {
                            backends.push('gpu');
                        }
                    } catch (e) {
                        // GPU backend not available
                    }
                    
                    if (backends.length > 0) {
                        webnnStatus.textContent = 'Available (' + backends.join(', ') + ')';
                        webnnStatus.className = 'success';
                        state.features.webnn = true;
                        log('WebNN is available with backends: ' + backends.join(', '), 'success');
                    } else {
                        webnnStatus.textContent = 'No backends available';
                        webnnStatus.className = 'warning';
                        log('WebNN has no available backends', 'warning');
                    }
                } catch (error) {
                    webnnStatus.textContent = 'Error: ' + error.message;
                    webnnStatus.className = 'error';
                    log('WebNN error: ' + error.message, 'error');
                }
            } else {
                webnnStatus.textContent = 'Not supported';
                webnnStatus.className = 'error';
                log('WebNN is not supported in this browser', 'warning');
            }
            
            // WebAssembly detection
            const wasmStatus = document.getElementById('wasm-status');
            if (typeof WebAssembly === 'object') {
                wasmStatus.textContent = 'Available';
                wasmStatus.className = 'success';
                state.features.wasm = true;
                log('WebAssembly is available', 'success');
            } else {
                wasmStatus.textContent = 'Not supported';
                wasmStatus.className = 'error';
                log('WebAssembly is not supported', 'error');
            }
            
            return state.features;
        }
        
        // Initialize transformers.js
        async function initTransformers() {
            try {
                updateStatus('Loading transformers.js...');
                log('Loading transformers.js...');
                
                // Import transformers.js
                const { pipeline, env } = await import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.0');
                
                // Configure pipeline based on available features
                if (state.features.webgpu) {
                    log('Using WebGPU backend for transformers.js');
                    env.backends.onnx.wasm.numThreads = 1;
                    env.backends.onnx.webgl.numThreads = 1;
                    env.backends.onnx.webgpu.numThreads = 4;
                    env.backends.onnx.useWebGPU = true;
                } else if (state.features.webnn) {
                    log('Using WebNN backend for transformers.js');
                    env.backends.onnx.wasm.numThreads = 1;
                    env.backends.onnx.webnn.numThreads = 4;
                } else {
                    log('Using CPU backend for transformers.js');
                    env.backends.onnx.wasm.numThreads = 4;
                }
                
                // Store in state
                state.transformers = { pipeline, env };
                
                log('Transformers.js loaded successfully', 'success');
                updateStatus('Transformers.js loaded');
                
                return true;
            } catch (error) {
                log('Error loading transformers.js: ' + error.message, 'error');
                updateStatus('Error loading transformers.js');
                return false;
            }
        }
        
        // Get the task type for a model type
        function getTaskForModelType(modelType) {
            switch (modelType) {
                case 'text':
                    return 'feature-extraction';
                case 'vision':
                    return 'image-classification';
                case 'audio':
                    return 'audio-classification';
                case 'multimodal':
                    return 'image-to-text';
                default:
                    return 'feature-extraction';
            }
        }
        
        // Initialize a model
        async function initModel(modelName, modelType = 'text') {
            try {
                if (!state.transformers) {
                    log('Transformers.js not initialized', 'error');
                    return false;
                }
                
                log(`Initializing model: ${modelName} (${modelType})`);
                updateStatus(`Initializing model: ${modelName}`);
                
                // Get the task
                const task = getTaskForModelType(modelType);
                log(`Using task: ${task} for model type: ${modelType}`);
                
                // Initialize the pipeline
                const pipe = await state.transformers.pipeline(task, modelName);
                
                // Store in state
                state.models[modelName] = {
                    pipeline: pipe,
                    modelType: modelType,
                    initialized: true,
                    initTime: new Date()
                };
                
                log(`Model ${modelName} initialized successfully`, 'success');
                updateStatus(`Model ${modelName} ready`);
                
                return true;
            } catch (error) {
                log(`Error initializing model ${modelName}: ${error.message}`, 'error');
                updateStatus(`Error initializing model ${modelName}`);
                return false;
            }
        }
        
        // Run inference
        async function runInference(modelName, input, options = {}) {
            try {
                const model = state.models[modelName];
                
                if (!model) {
                    log(`Model ${modelName} not initialized`, 'error');
                    return { error: `Model ${modelName} not initialized` };
                }
                
                log(`Running inference with model: ${modelName}`);
                updateStatus(`Running inference with model: ${modelName}`);
                document.getElementById('inference-status').textContent = 'Running inference...';
                
                // Process input based on model type
                let processedInput = input;
                if (model.modelType === 'vision' && typeof input === 'object' && input.image) {
                    // For vision models, we might need to convert image paths to DOM elements
                    processedInput = input.image;
                } else if (model.modelType === 'audio' && typeof input === 'object' && input.audio) {
                    // For audio models, we might need to handle audio paths
                    processedInput = input.audio;
                } else if (model.modelType === 'multimodal' && typeof input === 'object') {
                    // For multimodal models, use the appropriate input format
                    processedInput = input;
                }
                
                // Start timer
                const startTime = performance.now();
                
                // Run inference
                const output = await model.pipeline(processedInput, options);
                
                // End timer
                const endTime = performance.now();
                const inferenceTime = endTime - startTime;
                
                // Update UI
                log(`Inference completed in ${inferenceTime.toFixed(2)}ms`, 'success');
                document.getElementById('inference-status').textContent = `Inference completed in ${inferenceTime.toFixed(2)}ms`;
                
                // Return result with metrics
                return {
                    output,
                    metrics: {
                        inference_time_ms: inferenceTime,
                        timestamp: new Date().toISOString()
                    }
                };
            } catch (error) {
                log(`Error running inference: ${error.message}`, 'error');
                document.getElementById('inference-status').textContent = `Inference error: ${error.message}`;
                return { error: error.message };
            }
        }
        
        // WebSocket connection
        let socket = null;
        
        // Initialize WebSocket
        function initWebSocket(port) {
            const url = `ws://localhost:${port}`;
            log(`Connecting to WebSocket at ${url}...`);
            
            socket = new WebSocket(url);
            
            socket.onopen = () => {
                log('WebSocket connection established', 'success');
                
                // Send features
                socket.send(JSON.stringify({
                    type: 'features',
                    data: state.features
                }));
            };
            
            socket.onclose = () => {
                log('WebSocket connection closed', 'warning');
            };
            
            socket.onerror = (error) => {
                log(`WebSocket error: ${error}`, 'error');
            };
            
            socket.onmessage = async (event) => {
                try {
                    const message = JSON.parse(event.data);
                    log(`Received message: ${message.type}`);
                    
                    switch (message.type) {
                        case 'init_model':
                            const initResult = await initModel(message.model_name, message.model_type);
                            socket.send(JSON.stringify({
                                type: 'init_model_response',
                                success: initResult,
                                model_name: message.model_name,
                                model_type: message.model_type,
                                timestamp: new Date().toISOString()
                            }));
                            break;
                            
                        case 'run_inference':
                            const inferenceResult = await runInference(
                                message.model_name,
                                message.input,
                                message.options
                            );
                            socket.send(JSON.stringify({
                                type: 'inference_response',
                                model_name: message.model_name,
                                result: inferenceResult,
                                timestamp: new Date().toISOString()
                            }));
                            break;
                            
                        case 'ping':
                            socket.send(JSON.stringify({
                                type: 'pong',
                                timestamp: new Date().toISOString()
                            }));
                            break;
                            
                        default:
                            log(`Unknown message type: ${message.type}`, 'warning');
                    }
                } catch (error) {
                    log(`Error processing message: ${error.message}`, 'error');
                    
                    // Send error response
                    socket.send(JSON.stringify({
                        type: 'error',
                        error: error.message,
                        timestamp: new Date().toISOString()
                    }));
                }
            };
        }
        
        // Main initialization function
        async function initialize() {
            try {
                // Detect features
                await detectFeatures();
                
                // Initialize transformers.js
                const transformersInitialized = await initTransformers();
                
                if (!transformersInitialized) {
                    log('Failed to initialize transformers.js', 'error');
                    updateStatus('Failed to initialize');
                    return;
                }
                
                // Get the WebSocket port from URL parameter
                const urlParams = new URLSearchParams(window.location.search);
                const port = urlParams.get('port') || 8765;
                
                // Initialize WebSocket
                initWebSocket(port);
                
                // Success
                updateStatus('Ready');
            } catch (error) {
                log(`Initialization error: ${error.message}`, 'error');
                updateStatus('Initialization error');
            }
        }
        
        // Initialize when page loads
        window.addEventListener('load', initialize);
    </script>
</body>
</html>
"""

class TransformersJSBridge:
    """Bridge between Python and transformers.js in the browser."""
    
    def __init__(self, browser_name="chrome", headless=False, port=8765):
        """Initialize the bridge.
        
        Args:
            browser_name: Browser to use (chrome, firefox, edge, safari)
            headless: Whether to run in headless mode
            port: Port for WebSocket communication
        """
        self.browser_name = browser_name
        self.headless = headless
        self.port = port
        self.driver = None
        self.html_file = None
        self.server = None
        self.features = None
        self.initialized_models = {}
        self.connection = None
        self.connected = False
        self.ready = False
    
    async def start(self):
        """Start the bridge.
        
        Returns:
            True if started successfully, False otherwise
        """
        try:
            # Create HTML file
            self.html_file = self._create_html_file()
            logger.info(f"Created HTML file: {self.html_file}")
            
            # Start WebSocket server
            await self._start_websocket_server()
            
            # Start browser
            success = self._start_browser()
            if not success:
                logger.error("Failed to start browser")
                await self.stop()
                return False
            
            # Wait for connection
            timeout = 10  # seconds
            start_time = time.time()
            while not self.connected and time.time() - start_time < timeout:
                await asyncio.sleep(0.1)
            
            if not self.connected:
                logger.error("Timeout waiting for WebSocket connection")
                await self.stop()
                return False
            
            # Wait for features
            timeout = 10  # seconds
            start_time = time.time()
            while not self.features and time.time() - start_time < timeout:
                await asyncio.sleep(0.1)
            
            if not self.features:
                logger.error("Timeout waiting for features")
                await self.stop()
                return False
            
            logger.info(f"Features: {json.dumps(self.features, indent=2)}")
            self.ready = True
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting transformers.js bridge: {e}")
            await self.stop()
            return False
    
    def _create_html_file(self):
        """Create HTML file for browser.
        
        Returns:
            Path to HTML file
        """
        fd, path = tempfile.mkstemp(suffix=".html")
        with os.fdopen(fd, "w") as f:
            f.write(TRANSFORMERS_JS_HTML)
        
        return path
    
    async def _start_websocket_server(self):
        """Start WebSocket server.
        
        Returns:
            True if started successfully, False otherwise
        """
        try:
            self.server = await websockets.serve(self._handle_websocket, "localhost", self.port)
            logger.info(f"WebSocket server started on port {self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
            return False
    
    async def _handle_websocket(self, websocket):
        """Handle WebSocket connection.
        
        Args:
            websocket: WebSocket connection
        """
        logger.info(f"WebSocket connection established")
        self.connection = websocket
        self.connected = True
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._process_message(data)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON: {message}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
        except websockets.ConnectionClosed:
            logger.info("WebSocket connection closed")
        finally:
            self.connected = False
            self.connection = None
    
    async def _process_message(self, data):
        """Process incoming message.
        
        Args:
            data: Message data
        """
        message_type = data.get("type")
        logger.info(f"Received message: {message_type}")
        
        if message_type == "features":
            # Store features
            self.features = data.get("data")
            logger.info(f"Features received: {json.dumps(self.features, indent=2)}")
            
        elif message_type == "init_model_response":
            # Handle model initialization response
            model_name = data.get("model_name")
            success = data.get("success", False)
            
            if success:
                logger.info(f"Model {model_name} initialized successfully")
                self.initialized_models[model_name] = {
                    "model_type": data.get("model_type"),
                    "initialized": True,
                    "timestamp": data.get("timestamp")
                }
            else:
                logger.error(f"Failed to initialize model {model_name}")
            
        elif message_type == "inference_response":
            # Handle inference response
            logger.info(f"Inference response received for model {data.get('model_name')}")
            
        elif message_type == "pong":
            # Handle pong
            logger.info("Pong received")
            
        elif message_type == "error":
            # Handle error
            logger.error(f"Error from browser: {data.get('error')}")
    
    def _start_browser(self):
        """Start browser.
        
        Returns:
            True if started successfully, False otherwise
        """
        try:
            # Set up browser options
            if self.browser_name == "chrome":
                options = ChromeOptions()
                if self.headless:
                    options.add_argument("--headless=new")
                options.add_argument("--no-sandbox")
                options.add_argument("--disable-dev-shm-usage")
                
                # Enable WebGPU and WebNN
                options.add_argument("--enable-features=WebGPU,WebNN")
                
                # Set up service
                service = ChromeService(ChromeDriverManager().install())
                
                # Start browser
                self.driver = webdriver.Chrome(service=service, options=options)
            else:
                # Support for other browsers can be added here
                logger.error(f"Unsupported browser: {self.browser_name}")
                return False
            
            # Load HTML file
            file_url = f"file://{self.html_file}?port={self.port}"
            logger.info(f"Loading HTML file: {file_url}")
            self.driver.get(file_url)
            
            # Wait for page to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "status"))
            )
            
            logger.info("Browser started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting browser: {e}")
            return False
    
    async def initialize_model(self, model_name, model_type="text"):
        """Initialize a model.
        
        Args:
            model_name: Name of the model
            model_type: Type of model (text, vision, audio, multimodal)
            
        Returns:
            True if initialized successfully, False otherwise
        """
        if not self.ready:
            logger.error("Bridge not ready")
            return False
        
        # Check if model is already initialized
        if model_name in self.initialized_models:
            logger.info(f"Model {model_name} already initialized")
            return True
        
        try:
            # Send initialization request
            await self._send_message({
                "type": "init_model",
                "model_name": model_name,
                "model_type": model_type
            })
            
            # Wait for initialization
            timeout = 60  # seconds
            start_time = time.time()
            while (model_name not in self.initialized_models
                  and time.time() - start_time < timeout):
                await asyncio.sleep(0.1)
            
            if model_name not in self.initialized_models:
                logger.error(f"Timeout initializing model {model_name}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing model {model_name}: {e}")
            return False
    
    async def run_inference(self, model_name, input_data, options=None):
        """Run inference with a model.
        
        Args:
            model_name: Name of the model
            input_data: Input data for inference
            options: Inference options
            
        Returns:
            Inference result or None if failed
        """
        if not self.ready:
            logger.error("Bridge not ready")
            return None
        
        # Check if model is initialized
        if model_name not in self.initialized_models:
            logger.warning(f"Model {model_name} not initialized, initializing now")
            success = await self.initialize_model(model_name)
            if not success:
                logger.error(f"Failed to initialize model {model_name}")
                return None
        
        try:
            # Create a future for the response
            inference_future = asyncio.Future()
            
            # Define response handler
            async def response_handler(data):
                if (data.get("type") == "inference_response" 
                        and data.get("model_name") == model_name):
                    inference_future.set_result(data.get("result"))
            
            # Store current handler
            old_process_message = self._process_message
            
            # Wrap process message to capture response
            async def wrapped_process_message(data):
                await old_process_message(data)
                await response_handler(data)
            
            # Set wrapped handler
            self._process_message = wrapped_process_message
            
            # Send inference request
            await self._send_message({
                "type": "run_inference",
                "model_name": model_name,
                "input": input_data,
                "options": options or {}
            })
            
            # Wait for response with timeout
            try:
                result = await asyncio.wait_for(inference_future, 60)
            except asyncio.TimeoutError:
                logger.error(f"Timeout waiting for inference response for model {model_name}")
                result = None
            
            # Restore original handler
            self._process_message = old_process_message
            
            return result
            
        except Exception as e:
            logger.error(f"Error running inference with model {model_name}: {e}")
            return None
    
    async def _send_message(self, message):
        """Send message to browser.
        
        Args:
            message: Message to send
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.connected or not self.connection:
            logger.error("WebSocket not connected")
            return False
        
        try:
            await self.connection.send(json.dumps(message))
            return True
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False
    
    async def stop(self):
        """Stop the bridge."""
        # Stop browser
        if self.driver:
            self.driver.quit()
            self.driver = None
            logger.info("Browser stopped")
        
        # Stop WebSocket server
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logger.info("WebSocket server stopped")
        
        # Delete HTML file
        if self.html_file and os.path.exists(self.html_file):
            os.unlink(self.html_file)
            logger.info("HTML file deleted")
            self.html_file = None
        
        self.ready = False
        self.connected = False
        self.connection = None
        self.features = None
        self.initialized_models = {}

async def test_transformers_js_bridge():
    """Test the transformers.js bridge."""
    # Create bridge
    bridge = TransformersJSBridge(browser_name="chrome", headless=False)
    
    try:
        # Start bridge
        logger.info("Starting transformers.js bridge")
        success = await bridge.start()
        if not success:
            logger.error("Failed to start transformers.js bridge")
            return 1
        
        # Initialize model
        logger.info("Initializing BERT model")
        success = await bridge.initialize_model("bert-base-uncased", model_type="text")
        if not success:
            logger.error("Failed to initialize BERT model")
            await bridge.stop()
            return 1
        
        # Run inference
        logger.info("Running inference with BERT model")
        result = await bridge.run_inference("bert-base-uncased", "This is a test of transformers.js integration.")
        if not result:
            logger.error("Failed to run inference with BERT model")
            await bridge.stop()
            return 1
        
        logger.info(f"Inference result: {json.dumps(result, indent=2)}")
        
        # Stop bridge
        await bridge.stop()
        logger.info("Test completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error testing transformers.js bridge: {e}")
        await bridge.stop()
        return 1

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Transformers.js Integration")
    parser.add_argument("--browser", choices=["chrome", "firefox", "edge", "safari"], default="chrome",
                      help="Browser to use")
    parser.add_argument("--headless", action="store_true",
                      help="Run in headless mode")
    parser.add_argument("--model", default="bert-base-uncased",
                      help="Model to test")
    parser.add_argument("--input", default="This is a test of transformers.js integration.",
                      help="Input text for inference")
    parser.add_argument("--test", action="store_true",
                      help="Run test")
    parser.add_argument("--port", type=int, default=8765,
                      help="Port for WebSocket communication")
    
    args = parser.parse_args()
    
    # Run test
    if args.test:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(test_transformers_js_bridge())
    
    # Create bridge
    bridge = TransformersJSBridge(browser_name=args.browser, headless=args.headless, port=args.port)
    
    # Run
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Start bridge
        logger.info("Starting transformers.js bridge")
        success = loop.run_until_complete(bridge.start())
        if not success:
            logger.error("Failed to start transformers.js bridge")
            return 1
        
        # Initialize model
        logger.info(f"Initializing model {args.model}")
        success = loop.run_until_complete(bridge.initialize_model(args.model))
        if not success:
            logger.error(f"Failed to initialize model {args.model}")
            loop.run_until_complete(bridge.stop())
            return 1
        
        # Run inference
        logger.info(f"Running inference with model {args.model}")
        result = loop.run_until_complete(bridge.run_inference(args.model, args.input))
        if not result:
            logger.error(f"Failed to run inference with model {args.model}")
            loop.run_until_complete(bridge.stop())
            return 1
        
        logger.info(f"Inference result: {json.dumps(result, indent=2)}")
        
        # Interactive mode
        logger.info("Press Enter to exit")
        input()
        
        # Stop bridge
        loop.run_until_complete(bridge.stop())
        logger.info("Bridge stopped")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Interrupted")
        loop.run_until_complete(bridge.stop())
        return 0
    except Exception as e:
        logger.error(f"Error: {e}")
        loop.run_until_complete(bridge.stop())
        return 1

if __name__ == "__main__":
    sys.exit(main())