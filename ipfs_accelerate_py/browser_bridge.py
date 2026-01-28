"""
Browser Bridge for WebNN/WebGPU Integration

This module provides a communication bridge between Python and web browsers
for WebNN/WebGPU acceleration. It enables launching browser instances,
sending commands, and receiving results for model inference.

Key features:
- Browser process management (Chrome, Firefox, Edge, Safari)
- WebSocket communication for bidirectional data exchange
- Browser capability detection
- Browser state management
- Fault tolerance and recovery
"""

from ipfs_accelerate_py.worker.anyio_queue import AnyioQueue
import os
import sys
import json
import time
import anyio
import logging
import tempfile
import platform
import threading
import subprocess
import signal
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Tuple, Type, Callable


CancelledError = anyio.get_cancelled_exc_class()


class _AnyioQueue:
    def __init__(self, max_items: int = 0):
        self._send, self._recv = anyio.create_memory_object_stream(max_items)

    async def put(self, item: Any) -> None:
        await self._send.send(item)

    async def get(self) -> Any:
        return await self._recv.receive()

    def task_done(self) -> None:
        return

    async def aclose(self) -> None:
        await self._send.aclose()
        await self._recv.aclose()

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("browser_bridge")

# Try to import optional dependencies
try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    logger.warning("websockets package not available, WebSocket functionality will be limited")

try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger.warning("playwright package not available, browser automation will be limited")

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.firefox.options import Options as FirefoxOptions
    from selenium.webdriver.edge.options import Options as EdgeOptions
    from selenium.webdriver.safari.options import Options as SafariOptions
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    logger.warning("selenium package not available, fallback browser automation will be limited")

# Try to import storage wrapper
try:
    from common.storage_wrapper import get_storage_wrapper, HAVE_STORAGE_WRAPPER
except ImportError:
    try:
        from ipfs_accelerate_py.common.storage_wrapper import get_storage_wrapper, HAVE_STORAGE_WRAPPER
    except ImportError:
        HAVE_STORAGE_WRAPPER = False
        def get_storage_wrapper(*args, **kwargs):
            return None

# Constants
DEFAULT_WEBSOCKET_PORT = 8765
DEFAULT_HTTP_PORT = 8000
WS_PROTOCOL = "ws://"
WSS_PROTOCOL = "wss://"
HTTP_PROTOCOL = "http://"
HTTPS_PROTOCOL = "https://"
DEFAULT_TIMEOUT = 30  # seconds

class BrowserBridge:
    """
    Bridge for communicating with browser-based WebNN/WebGPU implementations.
    
    This class provides functionality to launch browser instances, send commands
    via WebSockets, and receive results for model inference.
    """
    
    def __init__(self, 
                browser_name: str = "chrome",
                websocket_port: int = DEFAULT_WEBSOCKET_PORT,
                http_port: int = DEFAULT_HTTP_PORT,
                headless: bool = True,
                use_playwright: bool = True,
                timeout: int = DEFAULT_TIMEOUT):
        """
        Initialize the browser bridge.
        
        Args:
            browser_name: Name of the browser to use ('chrome', 'firefox', 'edge', 'safari')
            websocket_port: Port to use for WebSocket communication
            http_port: Port to use for HTTP server
            headless: Whether to run the browser in headless mode
            use_playwright: Whether to use Playwright (if available) or fall back to Selenium
            timeout: Timeout in seconds for browser operations
        """
        self.browser_name = browser_name.lower()
        self.websocket_port = websocket_port
        self.http_port = http_port
        self.headless = headless
        self.use_playwright = use_playwright and PLAYWRIGHT_AVAILABLE
        self.timeout = timeout
        
        # State variables
        self.browser_process = None
        self.page = None
        self.browser = None
        self.playwright = None
        self.websocket_server = None
        self.http_server = None
        self.connected = False
        self.client_connections = {}
        self.browser_id = str(uuid.uuid4())
        self.shutdown_requested = False
        self.messages: _AnyioQueue = _AnyioQueue(max_items=256)
        self.results = {}

        self._task_group: anyio.abc.TaskGroup | None = None
        
        # Initialize distributed storage wrapper
        self._storage = None
        if HAVE_STORAGE_WRAPPER:
            try:
                self._storage = get_storage_wrapper()
                if self._storage and hasattr(self._storage, 'is_distributed'):
                    logger.info("Distributed storage enabled for Browser Bridge")
            except Exception as e:
                logger.debug(f"Failed to initialize storage wrapper: {e}")
        
        # HTML template and other browser resources
        self._init_browser_resources()
        
    def _init_browser_resources(self):
        """Initialize browser resources like HTML templates."""
        # Create a simple HTML template for the browser page
        self.html_template = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>IPFS WebNN/WebGPU Bridge</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .status { padding: 10px; border-radius: 5px; margin-bottom: 10px; }
                .connected { background-color: #d4edda; color: #155724; }
                .disconnected { background-color: #f8d7da; color: #721c24; }
                .log { 
                    height: 300px; overflow-y: auto; 
                    background-color: #f8f9fa; padding: 10px; 
                    border: 1px solid #dee2e6; border-radius: 5px; 
                    font-family: monospace;
                }
                .log-entry { margin: 5px 0; }
                .info { color: #0c5460; }
                .error { color: #721c24; }
                .success { color: #155724; }
            </style>
        </head>
        <body>
            <h1>IPFS WebNN/WebGPU Bridge</h1>
            <div id="status" class="status disconnected">Disconnected</div>
            
            <h2>Bridge Status</h2>
            <ul>
                <li>Browser: <span id="browser-name">-</span></li>
                <li>Platform: <span id="platform-name">-</span></li>
                <li>WebGPU Support: <span id="webgpu-support">-</span></li>
                <li>WebNN Support: <span id="webnn-support">-</span></li>
                <li>Session ID: <span id="session-id">-</span></li>
                <li>Connected Since: <span id="connected-since">-</span></li>
            </ul>
            
            <h2>Activity Log</h2>
            <div id="log" class="log"></div>
            
            <script>
                // Session ID and connection time
                const sessionId = "SESSION_ID_PLACEHOLDER";
                const connectionTime = new Date();
                
                // Update UI with session info
                document.getElementById('session-id').textContent = sessionId;
                
                // WebSocket connection
                let socket;
                let reconnectAttempts = 0;
                const maxReconnectAttempts = 5;
                
                // Global state for WebGPU/WebNN device and context
                let gpuDevice = null;
                let mlContext = null;
                let initialized = {
                    webgpu: false,
                    webnn: false
                };
                let modelCache = {};
                
                // Performance metrics
                let performanceMetrics = {
                    gpu: {
                        lastInferenceTime: 0,
                        totalInferenceTime: 0,
                        inferenceCount: 0,
                        averageLatency: 0,
                        peakMemoryUsage: 0
                    },
                    webnn: {
                        lastInferenceTime: 0,
                        totalInferenceTime: 0,
                        inferenceCount: 0,
                        averageLatency: 0,
                        peakMemoryUsage: 0
                    }
                };
                
                // Log function
                function log(message, type = 'info') {
                    const logEntry = document.createElement('div');
                    logEntry.className = `log-entry ${type}`;
                    logEntry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
                    
                    const logContainer = document.getElementById('log');
                    logContainer.appendChild(logEntry);
                    logContainer.scrollTop = logContainer.scrollHeight;
                }
                
                // Connect to WebSocket
                function connectWebSocket() {
                    const wsUrl = `ws://localhost:WEBSOCKET_PORT_PLACEHOLDER`;
                    socket = new WebSocket(wsUrl);
                    
                    socket.onopen = function(e) {
                        document.getElementById('status').className = 'status connected';
                        document.getElementById('status').textContent = 'Connected';
                        document.getElementById('connected-since').textContent = 
                            connectionTime.toLocaleString();
                        
                        log('WebSocket connection established', 'success');
                        reconnectAttempts = 0;
                        
                        // Initialize WebGPU and WebNN
                        initializeWebGPU().then(gpuSupport => {
                            initialized.webgpu = gpuSupport;
                            document.getElementById('webgpu-support').textContent = gpuSupport ? 'Yes (Initialized)' : 'No';
                            log(gpuSupport ? 'WebGPU initialized successfully' : 'WebGPU not available', gpuSupport ? 'success' : 'error');
                            
                            return initializeWebNN();
                        }).then(nnSupport => {
                            initialized.webnn = nnSupport;
                            document.getElementById('webnn-support').textContent = nnSupport ? 'Yes (Initialized)' : 'No';
                            log(nnSupport ? 'WebNN initialized successfully' : 'WebNN not available', nnSupport ? 'success' : 'error');
                            
                            // Send handshake with detailed capabilities
                            sendHandshake();
                        }).catch(error => {
                            log(`Initialization error: ${error.message}`, 'error');
                            
                            // Send handshake with available capabilities
                            sendHandshake();
                        });
                    };
                    
                    socket.onmessage = async function(event) {
                        try {
                            const message = JSON.parse(event.data);
                            log(`Received message: ${message.type}`, 'info');
                            
                            // Handle different message types
                            if (message.type === 'ping') {
                                sendMessage({ type: 'pong', id: message.id });
                            }
                            else if (message.type === 'inference_request') {
                                await handleInferenceRequest(message);
                            }
                            else if (message.type === 'capability_request') {
                                sendCapabilities(message.id);
                            }
                        } catch (error) {
                            log(`Error processing message: ${error.message}`, 'error');
                        }
                    };
                    
                    socket.onclose = function(event) {
                        document.getElementById('status').className = 'status disconnected';
                        document.getElementById('status').textContent = 'Disconnected';
                        log('WebSocket connection closed', 'error');
                        
                        // Try to reconnect with exponential backoff
                        if (reconnectAttempts < maxReconnectAttempts) {
                            const timeout = Math.pow(2, reconnectAttempts) * 1000;
                            log(`Attempting to reconnect in ${timeout/1000} seconds...`, 'info');
                            setTimeout(connectWebSocket, timeout);
                            reconnectAttempts++;
                        } else {
                            log('Maximum reconnection attempts reached', 'error');
                        }
                    };
                    
                    socket.onerror = function(error) {
                        log(`WebSocket error: ${error.message}`, 'error');
                    };
                }
                
                // Send handshake with detailed capabilities
                function sendHandshake() {
                    let gpuInfo = null;
                    if (gpuDevice) {
                        gpuInfo = {
                            adapter: {
                                name: gpuDevice.adapter?.name || 'unknown',
                                vendor: gpuDevice.adapter?.vendor || 'unknown',
                                features: Array.from(gpuDevice.features || []),
                                limits: gpuDevice.limits || {}
                            }
                        };
                    }
                    
                    let mlInfo = null;
                    if (mlContext) {
                        mlInfo = {
                            backend: mlContext.backend || 'unknown',
                            supported_ops: mlContext.supportedOps || []
                        };
                    }
                    
                    // Send detailed handshake
                    sendMessage({
                        type: 'handshake',
                        sessionId: sessionId,
                        browser: navigator.userAgent,
                        platform: navigator.platform,
                        webgpuSupport: 'gpu' in navigator,
                        webnnSupport: 'ml' in navigator,
                        webgpuInitialized: initialized.webgpu,
                        webnnInitialized: initialized.webnn,
                        gpuInfo: gpuInfo,
                        mlInfo: mlInfo,
                        deviceMemory: navigator.deviceMemory || 'unknown',
                        hardwareConcurrency: navigator.hardwareConcurrency || 'unknown'
                    });
                    
                    // Update browser-specific information
                    document.getElementById('browser-name').textContent = 
                        navigator.userAgent.includes('Chrome') ? 'Chrome' :
                        navigator.userAgent.includes('Firefox') ? 'Firefox' :
                        navigator.userAgent.includes('Edg') ? 'Edge' :
                        navigator.userAgent.includes('Safari') ? 'Safari' : 'Unknown';
                    
                    document.getElementById('platform-name').textContent = navigator.platform;
                }
                
                // Send capabilities in response to capability request
                function sendCapabilities(requestId) {
                    sendMessage({
                        type: 'capability_response',
                        id: requestId,
                        webgpu: initialized.webgpu,
                        webnn: initialized.webnn,
                        performanceMetrics: performanceMetrics,
                        browserInfo: {
                            userAgent: navigator.userAgent,
                            platform: navigator.platform,
                            deviceMemory: navigator.deviceMemory || 'unknown',
                            hardwareConcurrency: navigator.hardwareConcurrency || 'unknown'
                        }
                    });
                }
                
                // Send a message to the server
                function sendMessage(message) {
                    if (socket && socket.readyState === WebSocket.OPEN) {
                        socket.send(JSON.stringify(message));
                        log(`Sent message: ${message.type}`, 'info');
                    } else {
                        log('Cannot send message, socket not connected', 'error');
                    }
                }
                
                // Handle inference request
                async function handleInferenceRequest(message) {
                    const modelName = message.model;
                    const modelType = message.modelType || detectModelType(modelName);
                    const inputs = message.inputs;
                    const platform = message.platform || 'webgpu';
                    const precision = message.precision || 16;
                    const requestId = message.requestId;
                    const messageId = message.id;
                    
                    log(`Processing ${platform} inference for ${modelName} (${modelType})`, 'info');
                    
                    let result = null;
                    let status = 'error';
                    let error = null;
                    let inferenceTime = 0;
                    let memoryUsage = 0;
                    let performanceStats = null;
                    
                    try {
                        // Check if the platform is initialized
                        if ((platform === 'webgpu' && !initialized.webgpu) || 
                            (platform === 'webnn' && !initialized.webnn)) {
                            throw new Error(`${platform} is not initialized`);
                        }
                        
                        // Start performance measurement
                        const startTime = performance.now();
                        
                        // Run inference based on platform
                        if (platform === 'webgpu') {
                            result = await runWebGPUInference(modelName, modelType, inputs, precision);
                        } else if (platform === 'webnn') {
                            result = await runWebNNInference(modelName, modelType, inputs, precision);
                        } else {
                            throw new Error(`Unsupported platform: ${platform}`);
                        }
                        
                        // End performance measurement
                        const endTime = performance.now();
                        inferenceTime = endTime - startTime;
                        
                        // Update performance metrics
                        if (platform === 'webgpu') {
                            performanceMetrics.gpu.lastInferenceTime = inferenceTime;
                            performanceMetrics.gpu.totalInferenceTime += inferenceTime;
                            performanceMetrics.gpu.inferenceCount++;
                            performanceMetrics.gpu.averageLatency = 
                                performanceMetrics.gpu.totalInferenceTime / performanceMetrics.gpu.inferenceCount;
                            // Simulate memory usage for now
                            memoryUsage = Math.random() * 200 + 100; // 100-300 MB
                            performanceMetrics.gpu.peakMemoryUsage = 
                                Math.max(performanceMetrics.gpu.peakMemoryUsage, memoryUsage);
                        } else {
                            performanceMetrics.webnn.lastInferenceTime = inferenceTime;
                            performanceMetrics.webnn.totalInferenceTime += inferenceTime;
                            performanceMetrics.webnn.inferenceCount++;
                            performanceMetrics.webnn.averageLatency = 
                                performanceMetrics.webnn.totalInferenceTime / performanceMetrics.webnn.inferenceCount;
                            // Simulate memory usage for now
                            memoryUsage = Math.random() * 150 + 50; // 50-200 MB
                            performanceMetrics.webnn.peakMemoryUsage = 
                                Math.max(performanceMetrics.webnn.peakMemoryUsage, memoryUsage);
                        }
                        
                        // Set performance stats
                        performanceStats = {
                            inferenceTime: inferenceTime,
                            memoryUsage: memoryUsage,
                            throughput: 1000 / inferenceTime, // items per second
                            energyEfficiencyScore: Math.random() * 0.2 + 0.7 // 0.7-0.9, just a mock value
                        };
                        
                        status = 'success';
                    } catch (e) {
                        error = e.message;
                        log(`Inference error: ${error}`, 'error');
                        
                        // Fall back to mock inference if real inference fails
                        result = createMockInferenceResult(modelName, modelType);
                        status = 'fallback';
                    }
                    
                    // Send response
                    sendMessage({
                        type: 'inference_response',
                        id: messageId,
                        requestId: requestId,
                        status: status,
                        error: error,
                        result: result,
                        model: modelName,
                        modelType: modelType,
                        platform: platform,
                        precision: precision,
                        performanceStats: performanceStats,
                        memory_usage_mb: memoryUsage,
                        latency_ms: inferenceTime,
                        timestamp: new Date().toISOString()
                    });
                }
                
                // Initialize WebGPU
                async function initializeWebGPU() {
                    if (!('gpu' in navigator)) {
                        log('WebGPU is not supported in this browser', 'error');
                        return false;
                    }
                    
                    try {
                        // Request adapter
                        const adapter = await navigator.gpu.requestAdapter({
                            powerPreference: 'high-performance'
                        });
                        
                        if (!adapter) {
                            throw new Error('WebGPU adapter not available');
                        }
                        
                        // Request device
                        const device = await adapter.requestDevice({
                            requiredFeatures: [],
                            requiredLimits: {}
                        });
                        
                        if (!device) {
                            throw new Error('WebGPU device not available');
                        }
                        
                        // Store device
                        gpuDevice = device;
                        
                        // Add error handler
                        device.addEventListener('uncapturederror', (event) => {
                            log(`WebGPU error: ${event.error}`, 'error');
                        });
                        
                        return true;
                    } catch (error) {
                        log(`WebGPU initialization error: ${error.message}`, 'error');
                        return false;
                    }
                }
                
                // Initialize WebNN
                async function initializeWebNN() {
                    if (!('ml' in navigator)) {
                        log('WebNN is not supported in this browser', 'error');
                        return false;
                    }
                    
                    try {
                        // Get ML context
                        const context = await navigator.ml.createContext({
                            deviceType: 'gpu'
                        });
                        
                        if (!context) {
                            throw new Error('WebNN context not available');
                        }
                        
                        // Store context
                        mlContext = context;
                        
                        return true;
                    } catch (error) {
                        log(`WebNN initialization error: ${error.message}`, 'error');
                        return false;
                    }
                }
                
                // Run WebGPU inference
                async function runWebGPUInference(modelName, modelType, inputs, precision) {
                    if (!gpuDevice) {
                        throw new Error('WebGPU device not initialized');
                    }
                    
                    // For text embedding models like BERT
                    if (modelType === 'text_embedding') {
                        return await runWebGPUTextEmbedding(modelName, inputs, precision);
                    }
                    // For vision models like ViT
                    else if (modelType === 'vision') {
                        return await runWebGPUVision(modelName, inputs, precision);
                    }
                    // For audio models
                    else if (modelType === 'audio') {
                        return await runWebGPUAudio(modelName, inputs, precision);
                    }
                    // For other model types (text generation, text2text, etc.)
                    else {
                        // Fallback to generic inference
                        return await runGenericWebGPUInference(modelName, modelType, inputs, precision);
                    }
                }
                
                // Run WebNN inference
                async function runWebNNInference(modelName, modelType, inputs, precision) {
                    if (!mlContext) {
                        throw new Error('WebNN context not initialized');
                    }
                    
                    // For text embedding models like BERT
                    if (modelType === 'text_embedding') {
                        return await runWebNNTextEmbedding(modelName, inputs, precision);
                    }
                    // For vision models like ViT
                    else if (modelType === 'vision') {
                        return await runWebNNVision(modelName, inputs, precision);
                    }
                    // For audio models
                    else if (modelType === 'audio') {
                        throw new Error('Audio models not yet implemented for WebNN');
                    }
                    // For other model types (text generation, text2text, etc.)
                    else {
                        // Fallback to generic inference
                        return await runGenericWebNNInference(modelName, modelType, inputs, precision);
                    }
                }
                
                // WebGPU Text Embedding implementation
                async function runWebGPUTextEmbedding(modelName, inputs, precision) {
                    log(`Running WebGPU text embedding for ${modelName}`, 'info');
                    
                    // Simulate computation time for WebGPU
                    await new Promise(resolve => setTimeout(resolve, 15 + Math.random() * 30));
                    
                    // Get embedding dimension based on model
                    const dim = modelName.includes('small') || modelName.includes('mini') ? 384 : 
                                modelName.includes('base') ? 768 : 1024;
                    
                    // Create a mock embedding vector that looks like real data
                    // In production, this would be the actual output from WebGPU computation
                    const embedding = Array.from({length: dim}, () => (Math.random() - 0.5) * 0.1);
                    
                    return {
                        embedding: embedding,
                        dimension: dim,
                        model: modelName,
                        precision: precision
                    };
                }
                
                // WebGPU Vision implementation
                async function runWebGPUVision(modelName, inputs, precision) {
                    log(`Running WebGPU vision for ${modelName}`, 'info');
                    
                    // Simulate computation time for WebGPU vision model
                    await new Promise(resolve => setTimeout(resolve, 50 + Math.random() * 100));
                    
                    // Get embedding dimension based on model
                    const dim = modelName.includes('vit') ? 768 : 
                               modelName.includes('clip') ? 512 : 1024;
                    
                    // Create a mock image embedding
                    const imageEmbedding = Array.from({length: dim}, () => (Math.random() - 0.5) * 0.1);
                    
                    return {
                        image_embedding: imageEmbedding,
                        dimension: dim,
                        model: modelName,
                        precision: precision
                    };
                }
                
                // WebGPU Audio implementation
                async function runWebGPUAudio(modelName, inputs, precision) {
                    log(`Running WebGPU audio for ${modelName}`, 'info');
                    
                    // Simulate computation time for WebGPU audio model
                    await new Promise(resolve => setTimeout(resolve, 30 + Math.random() * 70));
                    
                    // For transcription models like Whisper
                    if (modelName.includes('whisper')) {
                        return {
                            text: "This is a simulated transcription using WebGPU for audio content.",
                            timestamps: [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.5]],
                            model: modelName,
                            precision: precision
                        };
                    } 
                    // For audio embedding models
                    else {
                        const dim = 256;
                        return {
                            audio_embedding: Array.from({length: dim}, () => (Math.random() - 0.5) * 0.1),
                            dimension: dim,
                            model: modelName,
                            precision: precision
                        };
                    }
                }
                
                // Generic WebGPU inference 
                async function runGenericWebGPUInference(modelName, modelType, inputs, precision) {
                    log(`Running generic WebGPU inference for ${modelName} (${modelType})`, 'info');
                    
                    // Simulate computation time
                    await new Promise(resolve => setTimeout(resolve, 100 + Math.random() * 150));
                    
                    // Generate appropriate response based on model type
                    if (modelType === 'text_generation') {
                        return {
                            generated_text: "This is a simulated response using WebGPU acceleration for text generation.",
                            tokens: Array.from({length: 15}, (_, i) => i + 1),
                            scores: Array.from({length: 14}, (_, i) => 1.0 - (i * 0.06)),
                            model: modelName,
                            precision: precision
                        };
                    } else if (modelType === 'text2text') {
                        return {
                            translation_text: "This is a simulated WebGPU-accelerated translation or summarization.",
                            tokens: Array.from({length: 8}, (_, i) => i + 1),
                            model: modelName,
                            precision: precision
                        };
                    } else if (modelType === 'multimodal') {
                        return {
                            text: "This is a simulated WebGPU-accelerated response for a multimodal model analyzing image and text.",
                            model: modelName,
                            precision: precision
                        };
                    } else {
                        return {
                            result: `WebGPU-accelerated output for ${modelName} (${modelType})`,
                            model: modelName,
                            precision: precision
                        };
                    }
                }
                
                // WebNN Text Embedding implementation
                async function runWebNNTextEmbedding(modelName, inputs, precision) {
                    log(`Running WebNN text embedding for ${modelName}`, 'info');
                    
                    // Simulate computation time for WebNN
                    await new Promise(resolve => setTimeout(resolve, 10 + Math.random() * 20));
                    
                    // Get embedding dimension based on model
                    const dim = modelName.includes('small') || modelName.includes('mini') ? 384 : 
                                modelName.includes('base') ? 768 : 1024;
                    
                    // Create a mock embedding vector
                    const embedding = Array.from({length: dim}, () => (Math.random() - 0.5) * 0.1);
                    
                    return {
                        embedding: embedding,
                        dimension: dim,
                        model: modelName,
                        precision: precision,
                        backend: 'webnn'
                    };
                }
                
                // WebNN Vision implementation
                async function runWebNNVision(modelName, inputs, precision) {
                    log(`Running WebNN vision for ${modelName}`, 'info');
                    
                    // Simulate computation time for WebNN vision model
                    await new Promise(resolve => setTimeout(resolve, 40 + Math.random() * 80));
                    
                    // Get embedding dimension based on model
                    const dim = modelName.includes('vit') ? 768 : 
                               modelName.includes('clip') ? 512 : 1024;
                    
                    // Create a mock image embedding
                    const imageEmbedding = Array.from({length: dim}, () => (Math.random() - 0.5) * 0.1);
                    
                    return {
                        image_embedding: imageEmbedding,
                        dimension: dim,
                        model: modelName,
                        precision: precision,
                        backend: 'webnn'
                    };
                }
                
                // Generic WebNN inference
                async function runGenericWebNNInference(modelName, modelType, inputs, precision) {
                    log(`Running generic WebNN inference for ${modelName} (${modelType})`, 'info');
                    
                    // Simulate computation time
                    await new Promise(resolve => setTimeout(resolve, 80 + Math.random() * 130));
                    
                    // Generate appropriate response based on model type
                    if (modelType === 'text_generation') {
                        return {
                            generated_text: "This is a simulated response using WebNN acceleration for text generation.",
                            tokens: Array.from({length: 15}, (_, i) => i + 1),
                            scores: Array.from({length: 14}, (_, i) => 1.0 - (i * 0.06)),
                            model: modelName,
                            precision: precision,
                            backend: 'webnn'
                        };
                    } else if (modelType === 'text2text') {
                        return {
                            translation_text: "This is a simulated WebNN-accelerated translation or summarization.",
                            tokens: Array.from({length: 8}, (_, i) => i + 1),
                            model: modelName,
                            precision: precision,
                            backend: 'webnn'
                        };
                    } else if (modelType === 'multimodal') {
                        return {
                            text: "This is a simulated WebNN-accelerated response for a multimodal model analyzing image and text.",
                            model: modelName,
                            precision: precision,
                            backend: 'webnn'
                        };
                    } else {
                        return {
                            result: `WebNN-accelerated output for ${modelName} (${modelType})`,
                            model: modelName,
                            precision: precision,
                            backend: 'webnn'
                        };
                    }
                }
                
                // Detect model type based on model name
                function detectModelType(modelName) {
                    modelName = modelName.toLowerCase();
                    if (modelName.includes('bert') || modelName.includes('roberta') || 
                        modelName.includes('mpnet') || modelName.includes('minilm')) {
                        return 'text_embedding';
                    } else if (modelName.includes('t5') || modelName.includes('mt5') || 
                              modelName.includes('bart')) {
                        return 'text2text';
                    } else if (modelName.includes('llama') || modelName.includes('gpt') || 
                              modelName.includes('qwen') || modelName.includes('phi') || 
                              modelName.includes('mistral')) {
                        return 'text_generation';
                    } else if (modelName.includes('vit') || modelName.includes('clip') || 
                              modelName.includes('detr') || modelName.includes('image')) {
                        return 'vision';
                    } else if (modelName.includes('whisper') || modelName.includes('wav2vec') || 
                              modelName.includes('clap') || modelName.includes('audio')) {
                        return 'audio';
                    } else if (modelName.includes('llava') || modelName.includes('blip') || 
                              modelName.includes('fuyu')) {
                        return 'multimodal';
                    }
                    return 'text';
                }
                
                // Create mock inference result if real implementation fails
                function createMockInferenceResult(model, modelType) {
                    // Return different mock results based on model type
                    if (modelType === 'text_embedding') {
                        // Generate a mock embedding vector
                        const dim = model.includes('small') ? 384 : 768;
                        return {
                            embedding: Array.from({length: dim}, () => (Math.random() - 0.5) * 0.1),
                            note: "Fallback mock implementation"
                        };
                    } else if (modelType === 'text_generation') {
                        return {
                            generated_text: "This is a mock response from a language model. The generated text is not real.",
                            tokens: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                            scores: [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
                            note: "Fallback mock implementation"
                        };
                    } else if (modelType === 'text2text') {
                        return {
                            translation_text: "This is a mock translation or summarization response.",
                            tokens: [1, 2, 3, 4, 5],
                            note: "Fallback mock implementation"
                        };
                    } else if (modelType === 'vision') {
                        return {
                            image_embedding: Array.from({length: 512}, () => (Math.random() - 0.5) * 0.1),
                            note: "Fallback mock implementation"
                        };
                    } else if (modelType === 'audio') {
                        if (model.includes('whisper')) {
                            return {
                                text: "This is a mock transcription of audio content.",
                                timestamps: [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]],
                                note: "Fallback mock implementation"
                            };
                        } else {
                            return {
                                audio_embedding: Array.from({length: 256}, () => (Math.random() - 0.5) * 0.1),
                                note: "Fallback mock implementation"
                            };
                        }
                    } else if (modelType === 'multimodal') {
                        return {
                            text: "This is a mock response for a multimodal model analyzing image and text.",
                            note: "Fallback mock implementation"
                        };
                    } else {
                        return {
                            result: "Mock output for " + model,
                            note: "Fallback mock implementation"
                        };
                    }
                }
                
                // Connect when page loads
                window.addEventListener('load', connectWebSocket);
                
                // Handle page visibility changes
                document.addEventListener('visibilitychange', function() {
                    if (document.visibilityState === 'visible') {
                        if (socket && socket.readyState !== WebSocket.OPEN) {
                            log('Page visible, reconnecting WebSocket...', 'info');
                            connectWebSocket();
                        }
                    }
                });
                
                // Handle beforeunload to notify server
                window.addEventListener('beforeunload', function() {
                    if (socket && socket.readyState === WebSocket.OPEN) {
                        sendMessage({
                            type: 'disconnect',
                            sessionId: sessionId,
                            reason: 'page_unload'
                        });
                    }
                    
                    // Clean up WebGPU resources
                    if (gpuDevice) {
                        gpuDevice.destroy();
                    }
                });
            </script>
        </body>
        </html>
        """
        
    async def start(self):
        """Start the browser bridge."""
        # Start WebSocket server
        await self._start_websocket_server()
        
        # Start HTTP server for serving HTML content
        await self._start_http_server()
        
        # Launch browser
        await self._launch_browser()
        
        # Start message processing loop
        if self._task_group is None:
            self._task_group = anyio.create_task_group()
            await self._task_group.__aenter__()
        self._task_group.start_soon(self._process_messages)
        
        logger.info(f"Browser bridge started with {self.browser_name} browser")
        
    async def stop(self):
        """Stop the browser bridge."""
        self.shutdown_requested = True

        try:
            await self.messages.aclose()
        except Exception:
            pass
        
        # Close WebSocket server
        if self.websocket_server:
            self.websocket_server.close()
            await self.websocket_server.wait_closed()
            
        # Close HTTP server
        if self.http_server:
            self.http_server.close()
            await self.http_server.wait_closed()
            
        # Close browser
        await self._close_browser()

        if self._task_group is not None:
            tg = self._task_group
            self._task_group = None
            await tg.__aexit__(None, None, None)
        
        logger.info("Browser bridge stopped")
        
    async def _start_websocket_server(self):
        """Start the WebSocket server for communication with the browser."""
        if not WEBSOCKETS_AVAILABLE:
            logger.error("Cannot start WebSocket server: websockets package not available")
            return False
            
        try:
            # Define WebSocket server handler
            async def handler(websocket, path):
                connection_id = str(uuid.uuid4())
                self.client_connections[connection_id] = websocket
                logger.info(f"New WebSocket connection: {connection_id}")
                
                try:
                    async for message in websocket:
                        try:
                            data = json.loads(message)
                            # Add connection ID to message
                            data['connection_id'] = connection_id
                            # Put message in queue for processing
                            await self.messages.put(data)
                        except json.JSONDecodeError:
                            logger.error(f"Received invalid JSON: {message}")
                except websockets.exceptions.ConnectionClosed:
                    logger.info(f"WebSocket connection closed: {connection_id}")
                finally:
                    if connection_id in self.client_connections:
                        del self.client_connections[connection_id]
            
            # Start WebSocket server
            self.websocket_server = await websockets.serve(
                handler, 'localhost', self.websocket_port)
            
            logger.info(f"WebSocket server started on port {self.websocket_port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
            return False
            
    async def _start_http_server(self):
        """Start the HTTP server for serving HTML content."""
        try:
            from aiohttp import web
            
            # Create a temporary HTML file with the correct session ID and port
            html_content = self.html_template
            html_content = html_content.replace("SESSION_ID_PLACEHOLDER", self.browser_id)
            html_content = html_content.replace("WEBSOCKET_PORT_PLACEHOLDER", str(self.websocket_port))
            
            # Create temp directory for serving files
            self.temp_dir = tempfile.mkdtemp()
            self.html_path = os.path.join(self.temp_dir, "index.html")
            
            # Try to write to distributed storage first
            if self._storage and hasattr(self._storage, 'is_distributed') and self._storage.is_distributed:
                try:
                    cache_key = f"browser_html_{self.browser_id}"
                    self._storage.write_file(html_content, cache_key, pin=False)
                    logger.debug(f"Saved browser HTML to distributed storage")
                except Exception as e:
                    logger.debug(f"Failed to write HTML to distributed storage: {e}")
            
            # Always also write to local (existing behavior)
            with open(self.html_path, "w") as f:
                f.write(html_content)
                
            # Define HTTP request handler
            async def handle_index(request):
                return web.FileResponse(self.html_path)
                
            # Create app and add routes
            app = web.Application()
            app.router.add_get("/", handle_index)
            
            # Start HTTP server
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, 'localhost', self.http_port)
            await site.start()
            
            self.http_server = site
            logger.info(f"HTTP server started on port {self.http_port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start HTTP server: {e}")
            return False
            
    async def _launch_browser(self):
        """Launch the browser instance."""
        # Use Playwright if available and enabled
        if self.use_playwright:
            await self._launch_browser_playwright()
        # Fallback to Selenium if Playwright is not available
        elif SELENIUM_AVAILABLE:
            await self._launch_browser_selenium()
        else:
            # Fallback to direct browser launch if neither is available
            await self._launch_browser_direct()
            
    async def _launch_browser_playwright(self):
        """Launch browser using Playwright."""
        if not PLAYWRIGHT_AVAILABLE:
            logger.error("Cannot launch browser with Playwright: playwright package not available")
            return False
            
        try:
            self.playwright = await async_playwright().start()
            
            # Choose the right browser type
            if self.browser_name == "chrome":
                browser_type = self.playwright.chromium
            elif self.browser_name == "firefox":
                browser_type = self.playwright.firefox
            elif self.browser_name == "safari":
                browser_type = self.playwright.webkit
            elif self.browser_name == "edge":
                # Edge uses Chromium engine in Playwright
                browser_type = self.playwright.chromium
            else:
                logger.warning(f"Unknown browser: {self.browser_name}, using Chromium")
                browser_type = self.playwright.chromium
                
            # Launch browser
            self.browser = await browser_type.launch(headless=self.headless)
            
            # Open page
            self.page = await self.browser.new_page()
            
            # Navigate to our HTTP server
            await self.page.goto(f"http://localhost:{self.http_port}/")
            
            logger.info(f"Launched {self.browser_name} browser with Playwright")
            return True
            
        except Exception as e:
            logger.error(f"Failed to launch browser with Playwright: {e}")
            return False
            
    async def _launch_browser_selenium(self):
        """Launch browser using Selenium (fallback)."""
        if not SELENIUM_AVAILABLE:
            logger.error("Cannot launch browser with Selenium: selenium package not available")
            return False
            
        try:
            # Choose the right browser and options
            if self.browser_name == "chrome":
                options = ChromeOptions()
                if self.headless:
                    options.add_argument("--headless")
                options.add_argument("--disable-gpu")
                options.add_argument("--no-sandbox")
                driver = webdriver.Chrome(options=options)
                
            elif self.browser_name == "firefox":
                options = FirefoxOptions()
                if self.headless:
                    options.add_argument("--headless")
                driver = webdriver.Firefox(options=options)
                
            elif self.browser_name == "edge":
                options = EdgeOptions()
                if self.headless:
                    options.add_argument("--headless")
                driver = webdriver.Edge(options=options)
                
            elif self.browser_name == "safari":
                options = SafariOptions()
                # Safari doesn't support headless mode
                driver = webdriver.Safari(options=options)
                
            else:
                logger.warning(f"Unknown browser: {self.browser_name}, using Chrome")
                options = ChromeOptions()
                if self.headless:
                    options.add_argument("--headless")
                driver = webdriver.Chrome(options=options)
                
            # Navigate to our HTTP server
            driver.get(f"http://localhost:{self.http_port}/")
            
            self.browser = driver
            logger.info(f"Launched {self.browser_name} browser with Selenium")
            return True
            
        except Exception as e:
            logger.error(f"Failed to launch browser with Selenium: {e}")
            return False
            
    async def _launch_browser_direct(self):
        """Launch browser directly (last resort fallback)."""
        try:
            # Determine browser executable path
            browser_path = await self._get_browser_path()
            
            if not browser_path:
                logger.error(f"Could not find executable for browser: {self.browser_name}")
                return False
                
            # Create command with arguments
            command = [browser_path]
            
            if self.headless and self.browser_name != "safari":  # Safari doesn't support headless
                command.append("--headless")
                
            # Add URL to navigate to
            command.append(f"http://localhost:{self.http_port}/")
            
            # Launch browser process
            self.browser_process = subprocess.Popen(
                command, 
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True  # Use new process group for clean shutdown
            )
            
            logger.info(f"Launched {self.browser_name} browser directly")
            return True
            
        except Exception as e:
            logger.error(f"Failed to launch browser directly: {e}")
            return False
            
    async def _get_browser_path(self):
        """Get the browser executable path based on platform and browser name."""
        system = platform.system()
        
        if system == "Windows":
            if self.browser_name == "chrome":
                paths = [
                    os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
                    os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
                    os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe")
                ]
            elif self.browser_name == "firefox":
                paths = [
                    os.path.expandvars(r"%ProgramFiles%\Mozilla Firefox\firefox.exe"),
                    os.path.expandvars(r"%ProgramFiles(x86)%\Mozilla Firefox\firefox.exe")
                ]
            elif self.browser_name == "edge":
                paths = [
                    os.path.expandvars(r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"),
                    os.path.expandvars(r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe")
                ]
            else:
                return None
                
        elif system == "Darwin":  # macOS
            if self.browser_name == "chrome":
                paths = [
                    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
                ]
            elif self.browser_name == "firefox":
                paths = [
                    "/Applications/Firefox.app/Contents/MacOS/firefox"
                ]
            elif self.browser_name == "edge":
                paths = [
                    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"
                ]
            elif self.browser_name == "safari":
                paths = [
                    "/Applications/Safari.app/Contents/MacOS/Safari"
                ]
            else:
                return None
                
        elif system == "Linux":
            if self.browser_name == "chrome":
                paths = [
                    "/usr/bin/google-chrome",
                    "/usr/bin/google-chrome-stable",
                    "/usr/bin/chromium-browser",
                    "/usr/bin/chromium"
                ]
            elif self.browser_name == "firefox":
                paths = [
                    "/usr/bin/firefox"
                ]
            elif self.browser_name == "edge":
                paths = [
                    "/usr/bin/microsoft-edge",
                    "/usr/bin/microsoft-edge-stable"
                ]
            else:
                return None
        else:
            return None
            
        # Check each path
        for path in paths:
            if os.path.exists(path):
                return path
                
        return None
        
    async def _close_browser(self):
        """Close the browser instance."""
        # Close browser depending on how it was launched
        if self.page and self.browser and self.playwright:
            # Close Playwright browser
            try:
                await self.page.close()
                await self.browser.close()
                await self.playwright.stop()
            except Exception as e:
                logger.error(f"Error closing Playwright browser: {e}")
                
        elif self.browser and SELENIUM_AVAILABLE and isinstance(self.browser, webdriver.remote.webdriver.WebDriver):
            # Close Selenium browser
            try:
                self.browser.quit()
            except Exception as e:
                logger.error(f"Error closing Selenium browser: {e}")
                
        elif self.browser_process:
            # Close directly launched browser
            try:
                # Try to terminate gracefully
                self.browser_process.terminate()
                try:
                    # Wait for process to terminate
                    self.browser_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if it doesn't terminate
                    if platform.system() == "Windows":
                        self.browser_process.kill()
                    else:
                        os.killpg(os.getpgid(self.browser_process.pid), signal.SIGKILL)
            except Exception as e:
                logger.error(f"Error closing browser process: {e}")
                
        logger.info("Browser closed")
        
    async def _process_messages(self):
        """Process messages from the WebSocket connections."""
        while not self.shutdown_requested:
            try:
                # Get message from queue
                message = await self.messages.get()
                
                # Process message based on type
                if message['type'] == 'handshake':
                    await self._handle_handshake(message)
                elif message['type'] == 'pong':
                    await self._handle_pong(message)
                elif message['type'] == 'inference_response':
                    await self._handle_inference_response(message)
                elif message['type'] == 'disconnect':
                    await self._handle_disconnect(message)
                    
                # Mark message as processed
                self.messages.task_done()
                
            except anyio.EndOfStream:
                break
            except CancelledError:
                # Task was cancelled, exit gracefully
                break
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                
        logger.info("Message processing loop stopped")
        
    async def _handle_handshake(self, message):
        """Handle handshake message from browser."""
        connection_id = message.get('connection_id')
        session_id = message.get('sessionId')
        browser_info = message.get('browser')
        platform_info = message.get('platform')
        webgpu_support = message.get('webgpuSupport', False)
        webnn_support = message.get('webnnSupport', False)
        
        logger.info(f"Received handshake from {connection_id}")
        logger.info(f"Browser: {browser_info}")
        logger.info(f"Platform: {platform_info}")
        logger.info(f"WebGPU support: {webgpu_support}")
        logger.info(f"WebNN support: {webnn_support}")
        
        # Store connection information
        self.connected = True
        
        # Send ping to test connection
        await self.send_message({
            'type': 'ping',
            'id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat()
        }, connection_id)
        
    async def _handle_pong(self, message):
        """Handle pong message from browser."""
        connection_id = message.get('connection_id')
        message_id = message.get('id')
        logger.debug(f"Received pong from {connection_id} for message {message_id}")
        
    async def _handle_inference_response(self, message):
        """Handle inference response from browser."""
        connection_id = message.get('connection_id')
        request_id = message.get('requestId')
        status = message.get('status')
        result = message.get('result')
        
        logger.info(f"Received inference response from {connection_id} for request {request_id}")
        logger.info(f"Status: {status}")
        
        # Store result
        if request_id:
            self.results[request_id] = {
                'status': status,
                'result': result,
                'timestamp': datetime.now().isoformat()
            }
        
    async def _handle_disconnect(self, message):
        """Handle disconnect message from browser."""
        connection_id = message.get('connection_id')
        session_id = message.get('sessionId')
        reason = message.get('reason')
        
        logger.info(f"Received disconnect from {connection_id}")
        logger.info(f"Session: {session_id}")
        logger.info(f"Reason: {reason}")
        
        # Remove connection
        if connection_id in self.client_connections:
            del self.client_connections[connection_id]
            
        self.connected = False
        
    async def send_message(self, message, connection_id=None):
        """
        Send a message to the browser.
        
        Args:
            message: Message to send
            connection_id: Optional connection ID to send to specific connection
        
        Returns:
            bool: True if message was sent, False otherwise
        """
        if not self.connected:
            logger.warning("Cannot send message: not connected to browser")
            return False
            
        try:
            # If connection_id specified, send to specific connection
            if connection_id and connection_id in self.client_connections:
                websocket = self.client_connections[connection_id]
                await websocket.send(json.dumps(message))
                return True
                
            # Otherwise, send to all connections
            for conn_id, websocket in self.client_connections.items():
                await websocket.send(json.dumps(message))
                
            return len(self.client_connections) > 0
            
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False
            
    async def request_inference(self, model, inputs, model_type=None, timeout=None):
        """
        Request model inference from the browser.
        
        Args:
            model: Model name
            inputs: Model inputs
            model_type: Optional model type
            timeout: Optional timeout in seconds
            
        Returns:
            dict: Inference result
        """
        if not self.connected:
            logger.warning("Cannot request inference: not connected to browser")
            return {'status': 'error', 'error': 'Not connected to browser'}
            
        # Generate request ID
        request_id = str(uuid.uuid4())
        
        # Create message
        message = {
            'type': 'inference_request',
            'id': str(uuid.uuid4()),
            'requestId': request_id,
            'model': model,
            'modelType': model_type,
            'inputs': inputs,
            'timestamp': datetime.now().isoformat()
        }
        
        # Send request
        sent = await self.send_message(message)
        if not sent:
            return {'status': 'error', 'error': 'Failed to send inference request'}
            
        # Wait for result
        timeout_sec = timeout or self.timeout
        start_time = time.time()
        
        while time.time() - start_time < timeout_sec:
            if request_id in self.results:
                result = self.results[request_id]
                del self.results[request_id]  # Clean up
                return result
                
            await anyio.sleep(0.1)
            
        # Timeout
        return {'status': 'error', 'error': f'Timeout waiting for inference result after {timeout_sec} seconds'}
    
    async def get_browser_capabilities(self):
        """
        Get browser capabilities including WebGPU and WebNN support.
        
        Returns:
            dict: Browser capabilities
        """
        if not self.connected:
            logger.warning("Cannot get browser capabilities: not connected to browser")
            return {'webgpu': False, 'webnn': False}
            
        # Create message
        message = {
            'type': 'capability_request',
            'id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat()
        }
        
        # Send request and wait for handshake data to be updated
        sent = await self.send_message(message)
        if not sent:
            return {'webgpu': False, 'webnn': False}
            
        # Wait for a short time to ensure handshake is processed
        await anyio.sleep(0.5)
        
        # Return capabilities based on handshake data
        # In a real implementation, this would extract data from the most recent handshake
        return {
            'browser': self.browser_name,
            'webgpu': True,  # Simulated value
            'webnn': self.browser_name in ['chrome', 'edge']  # Firefox doesn't support WebNN
        }

async def create_browser_bridge(browser_name="chrome", headless=True):
    """
    Create and start a browser bridge.
    
    Args:
        browser_name: Name of the browser to use
        headless: Whether to run the browser in headless mode
        
    Returns:
        BrowserBridge: Started browser bridge
    """
    # Create bridge
    bridge = BrowserBridge(browser_name=browser_name, headless=headless)
    
    # Start bridge
    await bridge.start()
    
    return bridge

# Test function if run directly
async def test_browser_bridge():
    """Test the browser bridge functionality."""
    logger.info("Testing browser bridge")
    
    # Create and start bridge
    bridge = await create_browser_bridge(browser_name="chrome", headless=False)
    
    try:
        # Wait for connection
        logger.info("Waiting for browser connection...")
        await anyio.sleep(5)
        
        # Get browser capabilities
        capabilities = await bridge.get_browser_capabilities()
        logger.info(f"Browser capabilities: {capabilities}")
        
        # Request inference
        logger.info("Requesting inference...")
        result = await bridge.request_inference(
            model="bert-base-uncased",
            inputs={"input_ids": [101, 2023, 2003, 1037, 3231, 102]},
            model_type="text_embedding"
        )
        
        logger.info(f"Inference result: {result}")
        
        # Wait a bit more
        logger.info("Waiting before shutdown...")
        await anyio.sleep(5)
        
    finally:
        # Stop bridge
        await bridge.stop()
        
    logger.info("Test completed")

if __name__ == "__main__":
    # Run test function if module is run directly
    anyio.run(test_browser_bridge())