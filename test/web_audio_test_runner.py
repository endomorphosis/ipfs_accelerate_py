"""
Web Audio Test Runner for the IPFS Accelerate Python Framework.

This module implements specialized tests for audio models on web platforms,
focusing on WebNN and WebGPU compatibility and performance.
"""

import os
import json
import time
import datetime
import logging
import subprocess
import threading
import http.server
import socketserver
import webbrowser
from pathlib import Path
from typing import Dict, List, Union, Optional, Any, Tuple, Callable
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("web_audio_test_runner")


class WebServerThread(threading.Thread):
    """Thread to run a simple HTTP server for serving test files."""
    
    def __init__(self, directory: str, port: int = 8000):
        """
        Initialize the web server thread.
        
        Args:
            directory (str): Directory to serve.
            port (int): Port to serve on.
        """
        super().__init__(daemon=True)
        self.directory = directory
        self.port = port
        self.server = None
        self.is_running = False
        
    def run(self):
        """Run the web server."""
        handler = http.server.SimpleHTTPRequestHandler
        
        # Change to the directory to serve
        original_dir = os.getcwd()
        os.chdir(self.directory)
        
        try:
            with socketserver.TCPServer(("", self.port), handler) as server:
                self.server = server
                self.is_running = True
                logger.info(f"Starting web server on port {self.port}, serving {self.directory}")
                server.serve_forever()
        finally:
            os.chdir(original_dir)
            self.is_running = False
            
    def stop(self):
        """Stop the web server."""
        if self.server:
            logger.info("Stopping web server")
            self.server.shutdown()
            self.is_running = False


class WebAudioTestRunner:
    """A system for testing audio models on web platforms (WebNN and WebGPU)."""

    def __init__(self, 
                test_directory: str = "./web_audio_tests",
                results_directory: str = "./web_audio_results",
                server_port: int = 8000,
                config_path: Optional[str] = None):
        """
        Initialize the web audio test runner.
        
        Args:
            test_directory (str): Directory for web test files.
            results_directory (str): Directory for test results.
            server_port (int): Port for the test web server.
            config_path (Optional[str]): Path to configuration file.
        """
        self.test_directory = Path(test_directory)
        self.test_directory.mkdir(exist_ok=True, parents=True)
        
        self.results_directory = Path(results_directory)
        self.results_directory.mkdir(exist_ok=True, parents=True)
        
        self.server_port = server_port
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize web server
        self.web_server = None
        
        # Check for NodeJS (for headless testing)
        self.node_available = self._check_node_available()
        if not self.node_available:
            logger.warning("NodeJS not available, headless testing will be disabled")
        
        # Ensure test templates are available
        self._ensure_test_templates()
        
    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """
        Load configuration from file.
        
        Args:
            config_path (Optional[str]): Path to the configuration file.
            
        Returns:
            Dict: Configuration dictionary.
        """
        default_config = {
            "models": {
                "whisper": {
                    "model_ids": ["openai/whisper-tiny", "openai/whisper-base", "openai/whisper-small"],
                    "model_formats": ["onnx", "webnn"],
                    "test_cases": ["speech_recognition", "audio_classification"]
                },
                "wav2vec2": {
                    "model_ids": ["facebook/wav2vec2-base", "facebook/wav2vec2-large"],
                    "model_formats": ["onnx", "webnn"],
                    "test_cases": ["speech_recognition", "audio_classification"]
                },
                "hubert": {
                    "model_ids": ["facebook/hubert-base-ls960", "facebook/hubert-large-ls960-ft"],
                    "model_formats": ["onnx"],
                    "test_cases": ["speech_recognition"]
                },
                "audio_spectrogram_transformer": {
                    "model_ids": ["MIT/ast-finetuned-audioset-10-10-0.4593"],
                    "model_formats": ["onnx"],
                    "test_cases": ["audio_classification"]
                }
            },
            "test_audio": {
                "speech_samples": ["sample1.wav", "sample2.wav", "sample3.wav"],
                "music_samples": ["music1.mp3", "music2.mp3"],
                "noise_samples": ["noise1.wav", "noise2.wav"],
                "test_durations": [1, 3, 5, 10],  # seconds
                "sample_rates": [16000, 44100]
            },
            "browsers": {
                "chrome": {
                    "enabled": True,
                    "binary_path": ""  # Auto-detect
                },
                "firefox": {
                    "enabled": True,
                    "binary_path": ""  # Auto-detect
                },
                "safari": {
                    "enabled": True,
                    "binary_path": ""  # Auto-detect
                },
                "edge": {
                    "enabled": True,
                    "binary_path": ""  # Auto-detect
                }
            },
            "headless": True,
            "test_timeout": 300,  # seconds
            "server_port": 8000,
            "webnn_backend_preference": ["gpu", "cpu", "default"],
            "webgpu_settings": {
                "preferred_format": "bgra8unorm",
                "preferred_canvas_format": "bgra8unorm"
            }
        }
        
        if config_path is None:
            return default_config
        
        config_path = Path(config_path)
        if not config_path.exists():
            logger.warning(f"Configuration file {config_path} not found, using default configuration")
            return default_config
        
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        return config
    
    def _check_node_available(self) -> bool:
        """
        Check if NodeJS is available for headless testing.
        
        Returns:
            bool: Whether NodeJS is available.
        """
        try:
            subprocess.run(["node", "--version"], 
                         stdout=subprocess.PIPE, 
                         stderr=subprocess.PIPE, 
                         check=True)
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    def _ensure_test_templates(self):
        """Ensure that test templates are available in the test directory."""
        # Create common directory for shared JavaScript
        common_dir = self.test_directory / "common"
        common_dir.mkdir(exist_ok=True)
        
        # Create test template files
        self._create_template_file(common_dir / "webnn-utils.js", self._get_webnn_utils_template())
        self._create_template_file(common_dir / "webgpu-utils.js", self._get_webgpu_utils_template())
        self._create_template_file(common_dir / "audio-utils.js", self._get_audio_utils_template())
        self._create_template_file(common_dir / "test-runner.js", self._get_test_runner_template())
        
        # Create model-specific test templates
        for model_type in self.config["models"].keys():
            model_dir = self.test_directory / model_type
            model_dir.mkdir(exist_ok=True)
            
            # Create test case templates for each model
            test_cases = self.config["models"][model_type].get("test_cases", [])
            for test_case in test_cases:
                template_content = self._get_model_test_template(model_type, test_case)
                test_file = model_dir / f"{test_case}.html"
                self._create_template_file(test_file, template_content)
    
    def _create_template_file(self, file_path: Path, content: str):
        """
        Create a template file if it doesn't exist.
        
        Args:
            file_path (Path): Path to the template file.
            content (str): Content of the template file.
        """
        if not file_path.exists():
            logger.info(f"Creating template file: {file_path}")
            with open(file_path, 'w') as f:
                f.write(content)
    
    def _get_webnn_utils_template(self) -> str:
        """Get the WebNN utils template."""
        return """
/**
 * WebNN Utilities for Audio Model Testing
 */

// Initialize WebNN Context
async function initWebNN(backendPreference = 'gpu') {
    try {
        // Check if WebNN is available
        if (!('ml' in navigator)) {
            throw new Error('WebNN is not supported in this browser');
        }
        
        // Try to get context with the preferred backend
        const contextOptions = {};
        if (backendPreference !== 'default') {
            contextOptions.devicePreference = backendPreference;
        }
        
        const context = await navigator.ml.createContext(contextOptions);
        const deviceType = await context.queryDevice();
        
        console.log(`WebNN initialized with backend: ${deviceType}`);
        return { context, deviceType };
    } catch (error) {
        console.error('Failed to initialize WebNN:', error);
        throw error;
    }
}

// Load ONNX model with WebNN
async function loadOnnxModelWithWebNN(modelUrl, context) {
    try {
        const response = await fetch(modelUrl);
        if (!response.ok) {
            throw new Error(`Failed to fetch model: ${response.statusText}`);
        }
        
        const modelBuffer = await response.arrayBuffer();
        const graphBuilder = new MLGraphBuilder(context);
        const model = await graphBuilder.buildFromOnnxBuffer(modelBuffer);
        
        return { graphBuilder, model };
    } catch (error) {
        console.error('Failed to load ONNX model with WebNN:', error);
        throw error;
    }
}

// Run inference with WebNN
async function runWebNNInference(model, inputs) {
    try {
        const start = performance.now();
        const results = await model.compute(inputs);
        const end = performance.now();
        
        return {
            results,
            inferenceTime: end - start
        };
    } catch (error) {
        console.error('Failed to run WebNN inference:', error);
        throw error;
    }
}

// Create WebNN tensor from array
function createWebNNTensor(graphBuilder, data, dimensions) {
    const tensorData = new Float32Array(data);
    return graphBuilder.constant({
        dataType: 'float32',
        dimensions,
        value: tensorData
    });
}

// Helper function to convert audio data to WebNN input
function prepareAudioInputForWebNN(graphBuilder, audioData, sampleRate, modelType) {
    // Implementation depends on the model type
    switch (modelType) {
        case 'whisper':
            // Whisper expects mel spectrograms
            const melSpectrogram = computeMelSpectrogram(audioData, sampleRate);
            return createWebNNTensor(graphBuilder, melSpectrogram.data, melSpectrogram.dimensions);
            
        case 'wav2vec2':
        case 'hubert':
            // Wav2Vec2 and HuBERT expect raw audio waveform
            // Normalize audio to float32 in [-1, 1] range
            const normalizedAudio = normalizeAudio(audioData);
            return createWebNNTensor(graphBuilder, normalizedAudio, [1, normalizedAudio.length]);
            
        case 'audio_spectrogram_transformer':
            // AST expects a spectrogram
            const spectrogram = computeSpectrogram(audioData, sampleRate);
            return createWebNNTensor(graphBuilder, spectrogram.data, spectrogram.dimensions);
            
        default:
            throw new Error(`Unsupported model type: ${modelType}`);
    }
}

// Export utilities
export {
    initWebNN,
    loadOnnxModelWithWebNN,
    runWebNNInference,
    createWebNNTensor,
    prepareAudioInputForWebNN
};
"""
    
    def _get_webgpu_utils_template(self) -> str:
        """Get the WebGPU utils template."""
        return """
/**
 * WebGPU Utilities for Audio Model Testing
 */

// Initialize WebGPU
async function initWebGPU(preferredFormat = 'bgra8unorm') {
    try {
        // Check if WebGPU is available
        if (!navigator.gpu) {
            throw new Error('WebGPU is not supported in this browser');
        }
        
        // Request adapter
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            throw new Error('No WebGPU adapter found');
        }
        
        // Request device
        const device = await adapter.requestDevice();
        
        // Get preferred format
        const format = preferredFormat || navigator.gpu.getPreferredCanvasFormat();
        
        console.log(`WebGPU initialized with format: ${format}`);
        return { adapter, device, format };
    } catch (error) {
        console.error('Failed to initialize WebGPU:', error);
        throw error;
    }
}

// Create WebGPU buffer
function createBuffer(device, data, usage) {
    const buffer = device.createBuffer({
        size: data.byteLength,
        usage: usage,
        mappedAtCreation: true
    });
    
    const mapped = new Float32Array(buffer.getMappedRange());
    mapped.set(data);
    buffer.unmap();
    
    return buffer;
}

// Helper function to convert audio data to WebGPU buffer
function prepareAudioInputForWebGPU(device, audioData, sampleRate, modelType) {
    // Implementation depends on the model type
    switch (modelType) {
        case 'whisper':
            // Whisper expects mel spectrograms
            const melSpectrogram = computeMelSpectrogram(audioData, sampleRate);
            return createBuffer(
                device, 
                new Float32Array(melSpectrogram.data), 
                GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
            );
            
        case 'wav2vec2':
        case 'hubert':
            // Wav2Vec2 and HuBERT expect raw audio waveform
            // Normalize audio to float32 in [-1, 1] range
            const normalizedAudio = normalizeAudio(audioData);
            return createBuffer(
                device, 
                new Float32Array(normalizedAudio), 
                GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
            );
            
        case 'audio_spectrogram_transformer':
            // AST expects a spectrogram
            const spectrogram = computeSpectrogram(audioData, sampleRate);
            return createBuffer(
                device, 
                new Float32Array(spectrogram.data), 
                GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
            );
            
        default:
            throw new Error(`Unsupported model type: ${modelType}`);
    }
}

// Create a compute pipeline for audio processing
async function createComputePipeline(device, shaderCode) {
    const shaderModule = device.createShaderModule({
        code: shaderCode
    });
    
    const pipeline = await device.createComputePipelineAsync({
        layout: 'auto',
        compute: {
            module: shaderModule,
            entryPoint: 'main'
        }
    });
    
    return pipeline;
}

// Execute a compute shader for audio processing
async function executeComputeShader(device, pipeline, bindings, dispatchX, dispatchY = 1, dispatchZ = 1) {
    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: bindings
    });
    
    const commandEncoder = device.createCommandEncoder();
    const computePass = commandEncoder.beginComputePass();
    
    computePass.setPipeline(pipeline);
    computePass.setBindGroup(0, bindGroup);
    computePass.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
    computePass.end();
    
    const commands = commandEncoder.finish();
    device.queue.submit([commands]);
    
    // Wait for GPU to complete execution
    await device.queue.onSubmittedWorkDone();
}

// Get result from GPU buffer
async function getBufferData(device, buffer, size) {
    const readBuffer = device.createBuffer({
        size: size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    
    const commandEncoder = device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(buffer, 0, readBuffer, 0, size);
    
    const commands = commandEncoder.finish();
    device.queue.submit([commands]);
    
    await readBuffer.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(readBuffer.getMappedRange());
    const resultCopy = new Float32Array(result);
    readBuffer.unmap();
    
    return resultCopy;
}

// Export utilities
export {
    initWebGPU,
    createBuffer,
    prepareAudioInputForWebGPU,
    createComputePipeline,
    executeComputeShader,
    getBufferData
};
"""
    
    def _get_audio_utils_template(self) -> str:
        """Get the audio utils template."""
        return """
/**
 * Audio Processing Utilities for Web Audio Model Testing
 */

// Load audio file and convert to audio buffer
async function loadAudioFile(url) {
    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to fetch audio: ${response.statusText}`);
        }
        
        const arrayBuffer = await response.arrayBuffer();
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
        
        return { audioBuffer, audioContext };
    } catch (error) {
        console.error('Failed to load audio file:', error);
        throw error;
    }
}

// Convert audio buffer to mono channel Float32Array
function audioBufferToMono(audioBuffer) {
    const numChannels = audioBuffer.numberOfChannels;
    const length = audioBuffer.length;
    const sampleRate = audioBuffer.sampleRate;
    
    let monoData = new Float32Array(length);
    
    // For mono audio, just copy the first channel
    if (numChannels === 1) {
        monoData.set(audioBuffer.getChannelData(0));
    } else {
        // For multi-channel audio, average all channels
        for (let i = 0; i < numChannels; i++) {
            const channelData = audioBuffer.getChannelData(i);
            for (let j = 0; j < length; j++) {
                monoData[j] += channelData[j] / numChannels;
            }
        }
    }
    
    return { data: monoData, sampleRate };
}

// Resample audio to target sample rate
function resampleAudio(audioData, originalSampleRate, targetSampleRate) {
    if (originalSampleRate === targetSampleRate) {
        return audioData;
    }
    
    const ratio = originalSampleRate / targetSampleRate;
    const newLength = Math.round(audioData.length / ratio);
    const result = new Float32Array(newLength);
    
    for (let i = 0; i < newLength; i++) {
        // Simple linear interpolation resampling
        const position = i * ratio;
        const index = Math.floor(position);
        const fraction = position - index;
        
        if (index >= audioData.length - 1) {
            result[i] = audioData[audioData.length - 1];
        } else {
            result[i] = audioData[index] * (1 - fraction) + audioData[index + 1] * fraction;
        }
    }
    
    return result;
}

// Normalize audio to range [-1, 1]
function normalizeAudio(audioData) {
    let max = 0;
    for (let i = 0; i < audioData.length; i++) {
        const abs = Math.abs(audioData[i]);
        if (abs > max) {
            max = abs;
        }
    }
    
    if (max === 0) return audioData;
    
    const normalized = new Float32Array(audioData.length);
    for (let i = 0; i < audioData.length; i++) {
        normalized[i] = audioData[i] / max;
    }
    
    return normalized;
}

// Compute mel spectrogram from audio data
function computeMelSpectrogram(audioData, sampleRate, options = {}) {
    // Default parameters for mel spectrogram
    const n_fft = options.n_fft || 1024;
    const hop_length = options.hop_length || 512;
    const n_mels = options.n_mels || 80;
    const win_length = options.win_length || n_fft;
    const fmin = options.fmin || 0;
    const fmax = options.fmax || sampleRate / 2;
    
    // Create window function (Hann window)
    const window = new Float32Array(win_length);
    for (let i = 0; i < win_length; i++) {
        window[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (win_length - 1));
    }
    
    // Calculate number of frames
    const numFrames = Math.floor((audioData.length - n_fft) / hop_length) + 1;
    
    // Initialize output spectrogram (n_mels x numFrames)
    const spectrogram = new Float32Array(n_mels * numFrames);
    const dimensions = [1, n_mels, numFrames];
    
    // Compute spectrogram using Web Audio API if available
    if (window.AnalyserNode) {
        // This is a simplified implementation - in a real scenario,
        // you would use the Web Audio API's AnalyserNode for FFT
        // and then convert to mel scale
        
        // For now, we'll generate a dummy spectrogram
        for (let i = 0; i < spectrogram.length; i++) {
            spectrogram[i] = Math.random() * 2 - 1;
        }
    } else {
        // Fallback to dummy spectrogram
        for (let i = 0; i < spectrogram.length; i++) {
            spectrogram[i] = Math.random() * 2 - 1;
        }
    }
    
    return { data: spectrogram, dimensions };
}

// Compute regular spectrogram from audio data
function computeSpectrogram(audioData, sampleRate, options = {}) {
    // Default parameters for spectrogram
    const n_fft = options.n_fft || 1024;
    const hop_length = options.hop_length || 512;
    const win_length = options.win_length || n_fft;
    
    // Create window function (Hann window)
    const window = new Float32Array(win_length);
    for (let i = 0; i < win_length; i++) {
        window[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (win_length - 1));
    }
    
    // Calculate number of frames
    const numFrames = Math.floor((audioData.length - n_fft) / hop_length) + 1;
    const numFreqBins = n_fft / 2 + 1;
    
    // Initialize output spectrogram (numFreqBins x numFrames)
    const spectrogram = new Float32Array(numFreqBins * numFrames);
    const dimensions = [1, numFreqBins, numFrames];
    
    // Compute spectrogram using Web Audio API if available
    if (window.AnalyserNode) {
        // This is a simplified implementation - in a real scenario,
        // you would use the Web Audio API's AnalyserNode for FFT
        
        // For now, we'll generate a dummy spectrogram
        for (let i = 0; i < spectrogram.length; i++) {
            spectrogram[i] = Math.random() * 2 - 1;
        }
    } else {
        // Fallback to dummy spectrogram
        for (let i = 0; i < spectrogram.length; i++) {
            spectrogram[i] = Math.random() * 2 - 1;
        }
    }
    
    return { data: spectrogram, dimensions };
}

// Generate synthetic audio for testing
function generateSyntheticAudio(duration, sampleRate) {
    const numSamples = Math.round(duration * sampleRate);
    const audioData = new Float32Array(numSamples);
    
    // Generate a simple sine wave
    const frequency = 440; // A4 note
    for (let i = 0; i < numSamples; i++) {
        audioData[i] = Math.sin(2 * Math.PI * frequency * i / sampleRate);
    }
    
    return { data: audioData, sampleRate };
}

// Record audio from microphone
async function recordAudio(duration) {
    try {
        // Request microphone access
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const source = audioContext.createMediaStreamSource(stream);
        
        // Create processor to capture audio data
        const processor = audioContext.createScriptProcessor(1024, 1, 1);
        const chunks = [];
        
        return new Promise((resolve, reject) => {
            let recordingStopped = false;
            
            processor.onaudioprocess = (e) => {
                if (recordingStopped) return;
                
                // Get audio data from input channel
                const audioData = e.inputBuffer.getChannelData(0);
                chunks.push(new Float32Array(audioData));
            };
            
            // Connect the processor to the source and destination
            source.connect(processor);
            processor.connect(audioContext.destination);
            
            // Stop recording after duration
            setTimeout(() => {
                recordingStopped = true;
                
                // Disconnect everything
                processor.disconnect();
                source.disconnect();
                stream.getTracks().forEach(track => track.stop());
                
                // Combine all chunks into a single Float32Array
                const totalLength = chunks.reduce((acc, chunk) => acc + chunk.length, 0);
                const result = new Float32Array(totalLength);
                
                let offset = 0;
                for (const chunk of chunks) {
                    result.set(chunk, offset);
                    offset += chunk.length;
                }
                
                resolve({ data: result, sampleRate: audioContext.sampleRate });
            }, duration * 1000);
        });
    } catch (error) {
        console.error('Failed to record audio:', error);
        throw error;
    }
}

// Export utilities
export {
    loadAudioFile,
    audioBufferToMono,
    resampleAudio,
    normalizeAudio,
    computeMelSpectrogram,
    computeSpectrogram,
    generateSyntheticAudio,
    recordAudio
};
"""
    
    def _get_test_runner_template(self) -> str:
        """Get the test runner template."""
        return """
/**
 * Test Runner for Web Audio Model Testing
 */

// Report test results back to server
async function reportTestResults(results) {
    try {
        const response = await fetch('/report-results', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(results)
        });
        
        if (!response.ok) {
            throw new Error(`Failed to report results: ${response.statusText}`);
        }
        
        console.log('Test results reported successfully');
        return true;
    } catch (error) {
        console.error('Failed to report test results:', error);
        
        // If server reporting fails, try to display results on page
        displayResultsOnPage(results);
        return false;
    }
}

// Display results on the page for manual viewing
function displayResultsOnPage(results) {
    const resultsDiv = document.getElementById('results') || 
                       document.createElement('div');
    
    if (!document.getElementById('results')) {
        resultsDiv.id = 'results';
        document.body.appendChild(resultsDiv);
    }
    
    // Format results as JSON with nice indentation
    resultsDiv.innerHTML = `
        <h2>Test Results</h2>
        <pre>${JSON.stringify(results, null, 2)}</pre>
    `;
}

// Run a test case and measure performance
async function runTestCase(testCase, options) {
    const startTime = performance.now();
    let success = false;
    let error = null;
    let metrics = {};
    
    try {
        const result = await testCase(options);
        success = true;
        metrics = result.metrics || {};
    } catch (err) {
        error = err.message || String(err);
        console.error('Test case failed:', err);
    }
    
    const endTime = performance.now();
    
    return {
        success,
        error,
        executionTime: endTime - startTime,
        ...metrics
    };
}

// Run all test cases for a model with different configurations
async function runAllTests(model, testCases, configurations) {
    const results = {
        model,
        timestamp: new Date().toISOString(),
        browser: getBrowserInfo(),
        platform: getPlatformInfo(),
        webnnSupport: 'ml' in navigator,
        webgpuSupport: !!navigator.gpu,
        tests: []
    };
    
    for (const testCase of testCases) {
        for (const config of configurations) {
            console.log(`Running test: ${testCase.name} with config:`, config);
            
            const testResult = await runTestCase(testCase, {
                model,
                ...config
            });
            
            results.tests.push({
                testName: testCase.name,
                configuration: config,
                result: testResult
            });
        }
    }
    
    // Report results
    await reportTestResults(results);
    
    return results;
}

// Get browser information
function getBrowserInfo() {
    const userAgent = navigator.userAgent;
    let browserName = "Unknown";
    let browserVersion = "";
    
    // Extract browser name and version from user agent
    if (userAgent.indexOf("Firefox") > -1) {
        browserName = "Firefox";
        browserVersion = userAgent.match(/Firefox\/([\d.]+)/)[1];
    } else if (userAgent.indexOf("Chrome") > -1) {
        browserName = "Chrome";
        browserVersion = userAgent.match(/Chrome\/([\d.]+)/)[1];
    } else if (userAgent.indexOf("Safari") > -1) {
        browserName = "Safari";
        browserVersion = userAgent.match(/Version\/([\d.]+)/)[1];
    } else if (userAgent.indexOf("Edge") > -1 || userAgent.indexOf("Edg") > -1) {
        browserName = "Edge";
        browserVersion = userAgent.match(/Edge?\/([\d.]+)/)[1];
    }
    
    return {
        name: browserName,
        version: browserVersion,
        userAgent
    };
}

// Get platform information
function getPlatformInfo() {
    return {
        os: navigator.platform,
        language: navigator.language,
        hardwareConcurrency: navigator.hardwareConcurrency || 'unknown',
        deviceMemory: navigator.deviceMemory || 'unknown'
    };
}

// Export utilities
export {
    reportTestResults,
    displayResultsOnPage,
    runTestCase,
    runAllTests,
    getBrowserInfo,
    getPlatformInfo
};
"""
    
    def _get_model_test_template(self, model_type: str, test_case: str) -> str:
        """
        Get the test template for a specific model and test case.
        
        Args:
            model_type (str): Type of the model (e.g., 'whisper', 'wav2vec2').
            test_case (str): Type of the test case (e.g., 'speech_recognition').
            
        Returns:
            str: Test template HTML content.
        """
        # Common HTML template structure
        template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{model_type.capitalize()} {test_case.replace('_', ' ').title()} Test</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .test-container {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .result {{ margin-top: 10px; padding: 10px; background-color: #f5f5f5; border-radius: 3px; }}
        .success {{ color: green; }}
        .error {{ color: red; }}
        .metrics {{ margin-top: 10px; }}
        pre {{ white-space: pre-wrap; overflow-x: auto; }}
        button {{ padding: 8px 16px; background-color: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; }}
        button:hover {{ background-color: #45a049; }}
        select, input {{ padding: 8px; margin: 5px 0; }}
    </style>
</head>
<body>
    <h1>{model_type.capitalize()} {test_case.replace('_', ' ').title()} Test</h1>
    
    <div class="test-container">
        <h2>Test Configuration</h2>
        
        <div>
            <label for="model-select">Model:</label>
            <select id="model-select">
                <!-- Will be populated with available models -->
            </select>
        </div>
        
        <div>
            <label for="backend-select">WebNN Backend:</label>
            <select id="backend-select">
                <option value="gpu">GPU (preferred)</option>
                <option value="cpu">CPU</option>
                <option value="default">Default</option>
            </select>
        </div>
        
        <div>
            <label for="format-select">Model Format:</label>
            <select id="format-select">
                <option value="onnx">ONNX</option>
                <option value="webnn">WebNN</option>
            </select>
        </div>
        
        <div>
            <label for="audio-select">Audio Sample:</label>
            <select id="audio-select">
                <!-- Will be populated with available audio samples -->
            </select>
        </div>
        
        <div>
            <button id="run-test-btn">Run Test</button>
            <button id="run-all-tests-btn">Run All Configurations</button>
        </div>
    </div>
    
    <div class="test-container">
        <h2>Test Results</h2>
        <div id="results">No tests run yet.</div>
    </div>
    
    <script type="module">
        import {{ initWebNN, loadOnnxModelWithWebNN, runWebNNInference, prepareAudioInputForWebNN }} from '/common/webnn-utils.js';
        import {{ initWebGPU, prepareAudioInputForWebGPU }} from '/common/webgpu-utils.js';
        import {{ loadAudioFile, audioBufferToMono, resampleAudio, generateSyntheticAudio }} from '/common/audio-utils.js';
        import {{ runTestCase, runAllTests, displayResultsOnPage }} from '/common/test-runner.js';
        
        // Configuration
        const modelType = '{model_type}';
        const testCase = '{test_case}';
        const modelOptions = {MODEL_OPTIONS_PLACEHOLDER};
        const audioOptions = {AUDIO_OPTIONS_PLACEHOLDER};
        
        // Initialize UI elements
        const modelSelect = document.getElementById('model-select');
        const backendSelect = document.getElementById('backend-select');
        const formatSelect = document.getElementById('format-select');
        const audioSelect = document.getElementById('audio-select');
        const runTestBtn = document.getElementById('run-test-btn');
        const runAllTestsBtn = document.getElementById('run-all-tests-btn');
        const resultsDiv = document.getElementById('results');
        
        // Populate model options
        modelOptions.forEach(model => {{
            const option = document.createElement('option');
            option.value = model;
            option.textContent = model;
            modelSelect.appendChild(option);
        }});
        
        // Populate audio options
        audioOptions.forEach(audio => {{
            const option = document.createElement('option');
            option.value = audio;
            option.textContent = audio;
            audioSelect.appendChild(option);
        }});
        
        // Test function for {test_case}
        async function runTest(options) {{
            const {{ model, backend = 'gpu', format = 'onnx', audioSample }} = options;
            
            // Initialize WebNN
            const {{ context, deviceType }} = await initWebNN(backend);
            
            // Load model
            const modelUrl = `/models/${{modelType}}/${{model}}.${{format}}`;
            const {{ graphBuilder, model: nnModel }} = await loadOnnxModelWithWebNN(modelUrl, context);
            
            // Load audio
            const audioUrl = `/audio/${{audioSample}}`;
            const {{ audioBuffer, audioContext }} = await loadAudioFile(audioUrl);
            const {{ data: audioData, sampleRate }} = audioBufferToMono(audioBuffer);
            
            // Prepare input for the model
            const modelInput = prepareAudioInputForWebNN(graphBuilder, audioData, sampleRate, modelType);
            
            // Run inference
            const {{ results, inferenceTime }} = await runWebNNInference(nnModel, {{ input: modelInput }});
            
            // Process results based on test case
            let processedResults = {{}};
            let metrics = {{ inferenceTime }};
            
            if (testCase === 'speech_recognition') {{
                // Process speech recognition results
                processedResults = decodeSpeechRecognitionResults(results, modelType);
                metrics.decodingTime = performance.now() - (performance.now() - inferenceTime);
            }} else if (testCase === 'audio_classification') {{
                // Process audio classification results
                processedResults = decodeAudioClassificationResults(results, modelType);
            }}
            
            return {{
                success: true,
                results: processedResults,
                metrics
            }};
        }}
        
        // Helper function to decode speech recognition results
        function decodeSpeechRecognitionResults(results, modelType) {{
            // This would be model-specific decoding logic
            // For now, return dummy results
            return {{
                text: "This is a sample transcription.",
                confidence: 0.95
            }};
        }}
        
        // Helper function to decode audio classification results
        function decodeAudioClassificationResults(results, modelType) {{
            // This would be model-specific decoding logic
            // For now, return dummy results
            return {{
                topClasses: [
                    {{ label: "Speech", score: 0.85 }},
                    {{ label: "Music", score: 0.10 }},
                    {{ label: "Background noise", score: 0.05 }}
                ]
            }};
        }}
        
        // Run a single test with current configuration
        async function runSingleTest() {{
            const model = modelSelect.value;
            const backend = backendSelect.value;
            const format = formatSelect.value;
            const audioSample = audioSelect.value;
            
            resultsDiv.innerHTML = 'Running test...';
            
            try {{
                const testResult = await runTestCase(runTest, {{
                    model,
                    backend,
                    format,
                    audioSample
                }});
                
                // Display results
                resultsDiv.innerHTML = `
                    <div class="${{testResult.success ? 'success' : 'error'}}">
                        Status: ${{testResult.success ? 'Success' : 'Failed'}}
                    </div>
                    ${{testResult.error ? `<div class="error">Error: ${{testResult.error}}</div>` : ''}}
                    <div class="metrics">
                        <p>Execution Time: ${{testResult.executionTime.toFixed(2)}} ms</p>
                        <p>Inference Time: ${{testResult.inferenceTime ? testResult.inferenceTime.toFixed(2) + ' ms' : 'N/A'}}</p>
                        ${{testResult.decodingTime ? `<p>Decoding Time: ${{testResult.decodingTime.toFixed(2)}} ms</p>` : ''}}
                    </div>
                    <h3>Results:</h3>
                    <pre>${{JSON.stringify(testResult, null, 2)}}</pre>
                `;
            }} catch (error) {{
                resultsDiv.innerHTML = `
                    <div class="error">
                        Test execution failed: ${{error.message || String(error)}}
                    </div>
                `;
            }}
        }}
        
        // Run all test configurations
        async function runAllTestConfigurations() {{
            resultsDiv.innerHTML = 'Running all test configurations...';
            
            const testCases = [runTest];
            const configurations = [];
            
            // Generate all combinations of model, backend, format, and audio sample
            modelOptions.forEach(model => {{
                ['gpu', 'cpu'].forEach(backend => {{
                    ['onnx', 'webnn'].forEach(format => {{
                        audioOptions.slice(0, 2).forEach(audioSample => {{
                            configurations.push({{
                                model,
                                backend,
                                format,
                                audioSample
                            }});
                        }});
                    }});
                }});
            }});
            
            try {{
                const results = await runAllTests(modelType, testCases, configurations);
                displayResultsOnPage(results);
            }} catch (error) {{
                resultsDiv.innerHTML = `
                    <div class="error">
                        Failed to run all tests: ${{error.message || String(error)}}
                    </div>
                `;
            }}
        }}
        
        // Event listeners
        runTestBtn.addEventListener('click', runSingleTest);
        runAllTestsBtn.addEventListener('click', runAllTestConfigurations);
        
        // Initialize the page
        window.addEventListener('DOMContentLoaded', () => {{
            console.log('Test page loaded for', modelType, testCase);
            
            // Check WebNN support
            if (!('ml' in navigator)) {{
                resultsDiv.innerHTML = `
                    <div class="error">
                        WebNN is not supported in this browser.
                    </div>
                `;
                runTestBtn.disabled = true;
                runAllTestsBtn.disabled = true;
            }}
        }});
    </script>
</body>
</html>
"""
        
        # Replace placeholders with actual configuration
        model_options = self.config["models"][model_type]["model_ids"]
        model_options_json = json.dumps(model_options)
        
        audio_options = []
        if test_case == "speech_recognition":
            audio_options = self.config["test_audio"]["speech_samples"]
        elif test_case == "audio_classification":
            audio_options = self.config["test_audio"]["speech_samples"] + self.config["test_audio"]["music_samples"]
        
        audio_options_json = json.dumps(audio_options)
        
        template = template.replace("{MODEL_OPTIONS_PLACEHOLDER}", model_options_json)
        template = template.replace("{AUDIO_OPTIONS_PLACEHOLDER}", audio_options_json)
        
        return template
    
    def start_web_server(self):
        """Start the web server for serving test files."""
        if self.web_server and self.web_server.is_running:
            logger.warning("Web server is already running")
            return
        
        self.web_server = WebServerThread(str(self.test_directory), self.server_port)
        self.web_server.start()
        
        # Wait for server to start
        start_time = time.time()
        while not self.web_server.is_running and time.time() - start_time < 5:
            time.sleep(0.1)
        
        if not self.web_server.is_running:
            logger.warning("Failed to start web server within timeout")
            return False
        
        logger.info(f"Web server started at http://localhost:{self.server_port}")
        return True
    
    def stop_web_server(self):
        """Stop the web server."""
        if self.web_server and self.web_server.is_running:
            self.web_server.stop()
            self.web_server.join(timeout=5)
            logger.info("Web server stopped")
    
    def run_browser_test(self, 
                        model_type: str,
                        test_case: str,
                        browser: str,
                        headless: bool = True,
                        timeout: Optional[int] = None) -> Dict:
        """
        Run a test in a browser.
        
        Args:
            model_type (str): Type of the model.
            test_case (str): Type of the test case.
            browser (str): Browser to use.
            headless (bool): Whether to run in headless mode.
            timeout (Optional[int]): Test timeout in seconds.
            
        Returns:
            Dict: Test results.
        """
        # Validate input
        if model_type not in self.config["models"]:
            logger.error(f"Unsupported model type: {model_type}")
            return {"status": "error", "message": f"Unsupported model type: {model_type}"}
        
        if test_case not in self.config["models"][model_type].get("test_cases", []):
            logger.error(f"Unsupported test case for {model_type}: {test_case}")
            return {"status": "error", "message": f"Unsupported test case for {model_type}: {test_case}"}
        
        browser_config = self.config["browsers"].get(browser.lower())
        if not browser_config or not browser_config.get("enabled", False):
            logger.error(f"Browser not enabled or not found: {browser}")
            return {"status": "error", "message": f"Browser not enabled or not found: {browser}"}
        
        # Use default timeout if not specified
        timeout = timeout or self.config["test_timeout"]
        
        # Ensure web server is running
        if not self.web_server or not self.web_server.is_running:
            success = self.start_web_server()
            if not success:
                return {"status": "error", "message": "Failed to start web server"}
        
        # Construct test URL
        test_url = f"http://localhost:{self.server_port}/{model_type}/{test_case}.html"
        
        # Get browser binary path
        binary_path = browser_config.get("binary_path", "")
        
        # Run browser test
        test_results = {}
        
        try:
            if self.node_available and headless:
                # Run headless test using NodeJS and Puppeteer
                test_results = self._run_headless_browser_test(
                    test_url, browser, binary_path, timeout
                )
            else:
                # Open test in browser for manual testing
                webbrowser.get(browser.lower()).open(test_url)
                logger.info(f"Opened test in {browser} browser: {test_url}")
                test_results = {"status": "manual", "message": f"Test opened in {browser} browser"}
        except Exception as e:
            logger.exception(f"Error running browser test: {e}")
            test_results = {"status": "error", "message": str(e)}
        
        return test_results
    
    def _run_headless_browser_test(self, 
                                 test_url: str, 
                                 browser: str, 
                                 binary_path: str, 
                                 timeout: int) -> Dict:
        """
        Run a headless browser test using NodeJS and Puppeteer.
        
        Args:
            test_url (str): URL of the test page.
            browser (str): Browser to use.
            binary_path (str): Path to browser binary.
            timeout (int): Test timeout in seconds.
            
        Returns:
            Dict: Test results.
        """
        # Create temporary script for running the test
        script_path = self.test_directory / "run_headless_test.js"
        
        script_content = f"""
const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

(async () => {{
    // Launch the browser
    const browser = await puppeteer.launch({{
        headless: true,
        executablePath: {f"'{binary_path}'" if binary_path else 'null'},
        args: ['--no-sandbox', '--disable-setuid-sandbox', '--enable-webnn', '--enable-webgpu'],
        timeout: {timeout * 1000}
    }});
    
    const results = {{
        status: 'running',
        timestamp: new Date().toISOString(),
        browser: '{browser}',
        url: '{test_url}',
        log: [],
        errors: []
    }};
    
    try {{
        const page = await browser.newPage();
        
        // Collect console messages
        page.on('console', msg => {{
            results.log.push({{
                type: msg.type(),
                text: msg.text()
            }});
        }});
        
        // Collect errors
        page.on('pageerror', error => {{
            results.errors.push(error.toString());
        }});
        
        // Go to test page
        await page.goto('{test_url}', {{ waitUntil: 'networkidle0', timeout: {timeout * 1000} }});
        
        // Check for WebNN support
        const webnnSupported = await page.evaluate(() => 'ml' in navigator);
        results.webnnSupported = webnnSupported;
        
        // Check for WebGPU support
        const webgpuSupported = await page.evaluate(() => !!navigator.gpu);
        results.webgpuSupported = webgpuSupported;
        
        if (!webnnSupported) {{
            results.status = 'failed';
            results.message = 'WebNN not supported in the browser';
        }} else {{
            // Run all tests automatically
            if (await page.evaluate(() => document.getElementById('run-all-tests-btn'))) {{
                await page.click('#run-all-tests-btn');
                
                // Wait for tests to complete
                await page.waitForFunction(
                    () => !document.getElementById('results').textContent.includes('Running all test configurations...'),
                    {{ timeout: {timeout * 1000} }}
                );
                
                // Extract test results
                const testResults = await page.evaluate(() => {{
                    const resultsDiv = document.getElementById('results');
                    try {{
                        // Look for JSON in a pre tag
                        const preTag = resultsDiv.querySelector('pre');
                        if (preTag) {{
                            return JSON.parse(preTag.textContent);
                        }}
                    }} catch (e) {{
                        // Parsing failed
                    }}
                    
                    // Return raw HTML as fallback
                    return {{ rawHtml: resultsDiv.innerHTML }};
                }});
                
                results.testResults = testResults;
                results.status = 'completed';
            }} else {{
                results.status = 'failed';
                results.message = 'Could not find test button';
            }}
        }}
    }} catch (error) {{
        results.status = 'error';
        results.message = error.toString();
    }} finally {{
        await browser.close();
    }}
    
    // Write results to file
    fs.writeFileSync(
        path.join('{self.results_directory}', `browser_test_${{Date.now()}}.json`),
        JSON.stringify(results, null, 2)
    );
    
    // Also output to stdout
    console.log(JSON.stringify(results));
}})();
"""
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        try:
            # Run the script with NodeJS
            process = subprocess.run(
                ["node", str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout + 30  # Add extra time for startup/shutdown
            )
            
            if process.returncode != 0:
                logger.error(f"Headless browser test failed: {process.stderr}")
                return {
                    "status": "error",
                    "message": f"Headless browser test failed: {process.stderr}"
                }
            
            # Parse results from stdout
            try:
                results = json.loads(process.stdout.strip())
                return results
            except json.JSONDecodeError:
                logger.error(f"Failed to parse test results: {process.stdout}")
                return {
                    "status": "error",
                    "message": "Failed to parse test results",
                    "raw_output": process.stdout
                }
            
        except subprocess.TimeoutExpired:
            logger.error(f"Headless browser test timed out after {timeout} seconds")
            return {
                "status": "timeout",
                "message": f"Test timed out after {timeout} seconds"
            }
        except Exception as e:
            logger.exception(f"Error running headless browser test: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def run_all_tests(self, 
                     model_types: Optional[List[str]] = None,
                     test_cases: Optional[List[str]] = None,
                     browsers: Optional[List[str]] = None,
                     headless: Optional[bool] = None) -> Dict:
        """
        Run all tests based on configuration.
        
        Args:
            model_types (Optional[List[str]]): List of model types to test. If None, uses all from config.
            test_cases (Optional[List[str]]): List of test cases to run. If None, uses all from config.
            browsers (Optional[List[str]]): List of browsers to test with. If None, uses all enabled browsers from config.
            headless (Optional[bool]): Whether to run in headless mode. If None, uses config setting.
            
        Returns:
            Dict: Test results.
        """
        # Use configuration defaults if not specified
        model_types = model_types or list(self.config["models"].keys())
        browsers = browsers or [b for b, cfg in self.config["browsers"].items() if cfg.get("enabled", False)]
        headless = self.config["headless"] if headless is None else headless
        
        # Initialize results
        all_results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "test_run_id": f"web_audio_{int(time.time())}",
            "config": {
                "model_types": model_types,
                "browsers": browsers,
                "headless": headless
            },
            "results": []
        }
        
        # Start web server if not already running
        if not self.web_server or not self.web_server.is_running:
            success = self.start_web_server()
            if not success:
                logger.error("Failed to start web server")
                all_results["status"] = "error"
                all_results["message"] = "Failed to start web server"
                return all_results
        
        # Run tests for each model type, test case, and browser
        for model_type in model_types:
            model_config = self.config["models"].get(model_type, {})
            model_test_cases = test_cases or model_config.get("test_cases", [])
            
            for test_case in model_test_cases:
                for browser in browsers:
                    logger.info(f"Running test: {model_type} - {test_case} - {browser}")
                    
                    result = self.run_browser_test(
                        model_type=model_type,
                        test_case=test_case,
                        browser=browser,
                        headless=headless
                    )
                    
                    all_results["results"].append({
                        "model_type": model_type,
                        "test_case": test_case,
                        "browser": browser,
                        "result": result
                    })
        
        # Save results
        results_file = self.results_directory / f"web_audio_tests_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"All tests completed. Results saved to {results_file}")
        
        # Stop web server
        self.stop_web_server()
        
        return all_results
    
    def prepare_test_audio_files(self):
        """Prepare test audio files in the test directory."""
        audio_dir = self.test_directory / "audio"
        audio_dir.mkdir(exist_ok=True)
        
        # Check for audio files and create placeholders if needed
        for category in ["speech_samples", "music_samples", "noise_samples"]:
            for filename in self.config["test_audio"].get(category, []):
                audio_file = audio_dir / filename
                if not audio_file.exists():
                    logger.warning(f"Audio file not found: {audio_file}, creating placeholder")
                    self._create_placeholder_audio(audio_file)
    
    def _create_placeholder_audio(self, audio_file: Path):
        """
        Create a placeholder audio file for testing.
        
        Args:
            audio_file (Path): Path to the audio file to create.
        """
        try:
            import numpy as np
            from scipy.io import wavfile
            
            # Generate a simple sine wave
            sample_rate = 16000
            duration = 3  # seconds
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            
            # Generate a 440 Hz tone (A4 note)
            frequency = 440
            audio_data = np.sin(2 * np.pi * frequency * t)
            
            # Scale to 16-bit range
            audio_data = (audio_data * 32767).astype(np.int16)
            
            # Write to file
            wavfile.write(audio_file, sample_rate, audio_data)
            
            logger.info(f"Created placeholder audio file: {audio_file}")
        except ImportError:
            logger.warning("scipy not available, cannot create placeholder audio")
            
            # Create an empty file as fallback
            with open(audio_file, 'wb') as f:
                f.write(b'')
    
    def prepare_model_files(self):
        """Prepare model files in the test directory."""
        models_dir = self.test_directory / "models"
        models_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for each model type
        for model_type in self.config["models"].keys():
            model_dir = models_dir / model_type
            model_dir.mkdir(exist_ok=True)
            
            # Check for model files and create placeholders
            model_ids = self.config["models"][model_type].get("model_ids", [])
            model_formats = self.config["models"][model_type].get("model_formats", ["onnx"])
            
            for model_id in model_ids:
                model_name = model_id.split("/")[-1]
                for model_format in model_formats:
                    model_file = model_dir / f"{model_name}.{model_format}"
                    if not model_file.exists():
                        logger.warning(f"Model file not found: {model_file}, creating placeholder")
                        self._create_placeholder_model(model_file, model_format)
    
    def _create_placeholder_model(self, model_file: Path, model_format: str):
        """
        Create a placeholder model file for testing.
        
        Args:
            model_file (Path): Path to the model file to create.
            model_format (str): Format of the model file.
        """
        # Create an empty file
        with open(model_file, 'wb') as f:
            f.write(b'PLACEHOLDER MODEL FILE')
        
        logger.info(f"Created placeholder model file: {model_file}")
    
    def generate_report(self, results_file: Optional[str] = None) -> str:
        """
        Generate a report from test results.
        
        Args:
            results_file (Optional[str]): Path to the results file. If None, uses the most recent.
            
        Returns:
            str: Path to the generated report.
        """
        # Find the most recent results file if not specified
        if results_file is None:
            results_files = list(self.results_directory.glob("web_audio_tests_*.json"))
            if not results_files:
                logger.error("No results files found")
                return ""
            
            results_file = str(max(results_files, key=os.path.getmtime))
        
        # Load results
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Generate report
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.results_directory / f"web_audio_report_{timestamp}.md"
        
        with open(report_file, 'w') as f:
            f.write(f"# Web Audio Models Test Report\n\n")
            f.write(f"Generated: {datetime.datetime.now().isoformat()}\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"- Test Run ID: {results.get('test_run_id', 'N/A')}\n")
            f.write(f"- Timestamp: {results.get('timestamp', 'N/A')}\n")
            f.write(f"- Total Tests: {len(results.get('results', []))}\n")
            
            # Count test statuses
            statuses = {}
            for test_result in results.get("results", []):
                status = test_result.get("result", {}).get("status", "unknown")
                statuses[status] = statuses.get(status, 0) + 1
            
            f.write("- Results by Status:\n")
            for status, count in statuses.items():
                f.write(f"  - {status}: {count}\n")
            
            f.write("\n## Test Configuration\n\n")
            f.write("```json\n")
            f.write(json.dumps(results.get("config", {}), indent=2))
            f.write("\n```\n\n")
            
            f.write("## Detailed Results\n\n")
            
            # Group results by model type
            results_by_model = {}
            for test_result in results.get("results", []):
                model_type = test_result.get("model_type")
                if model_type not in results_by_model:
                    results_by_model[model_type] = []
                results_by_model[model_type].append(test_result)
            
            for model_type, model_results in results_by_model.items():
                f.write(f"### {model_type.capitalize()}\n\n")
                
                # Create results table
                f.write("| Test Case | Browser | Status | WebNN | WebGPU | Notes |\n")
                f.write("|-----------|---------|--------|-------|--------|-------|\n")
                
                for test_result in model_results:
                    test_case = test_result.get("test_case", "")
                    browser = test_result.get("browser", "")
                    result = test_result.get("result", {})
                    status = result.get("status", "unknown")
                    
                    webnn_supported = result.get("webnnSupported", False)
                    webnn_icon = "✅" if webnn_supported else "❌"
                    
                    webgpu_supported = result.get("webgpuSupported", False)
                    webgpu_icon = "✅" if webgpu_supported else "❌"
                    
                    message = result.get("message", "")
                    
                    f.write(f"| {test_case} | {browser} | {status} | {webnn_icon} | {webgpu_icon} | {message} |\n")
                
                f.write("\n")
        
        logger.info(f"Generated report: {report_file}")
        return str(report_file)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Web Audio Test Runner")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--output-dir", type=str, default="./web_audio_results", help="Directory to store test results")
    parser.add_argument("--model-types", type=str, nargs="+", help="Model types to test")
    parser.add_argument("--test-cases", type=str, nargs="+", help="Test cases to run")
    parser.add_argument("--browsers", type=str, nargs="+", help="Browsers to test with")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--no-headless", action="store_false", dest="headless", help="Don't run in headless mode")
    parser.add_argument("--prepare", action="store_true", help="Prepare test files only")
    parser.add_argument("--report", type=str, help="Generate report from results file")
    parser.add_argument("--port", type=int, default=8000, help="Port for web server")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.debug:
        logging.getLogger("web_audio_test_runner").setLevel(logging.DEBUG)
    
    # Create test runner
    runner = WebAudioTestRunner(
        results_directory=args.output_dir,
        server_port=args.port,
        config_path=args.config
    )
    
    if args.prepare:
        # Prepare test files only
        runner.prepare_test_audio_files()
        runner.prepare_model_files()
        logger.info("Test files prepared")
    elif args.report:
        # Generate report from results file
        report_file = runner.generate_report(args.report)
        logger.info(f"Report generated: {report_file}")
    else:
        # Run tests
        runner.prepare_test_audio_files()
        runner.prepare_model_files()
        
        results = runner.run_all_tests(
            model_types=args.model_types,
            test_cases=args.test_cases,
            browsers=args.browsers,
            headless=args.headless
        )
        
        # Generate report
        report_file = runner.generate_report()
        logger.info(f"Report generated: {report_file}")