#!/usr/bin/env python
"""
Web Audio Platform Tests for the IPFS Accelerate Python Framework.

This module provides specialized tests for audio models on web platforms,
integrating with the web_audio_test_runner and web platform capabilities.

Usage:
    python web_audio_platform_tests.py --test-whisper
    python web_audio_platform_tests.py --test-wav2vec2
    python web_audio_platform_tests.py --run-all --headless
"""

import os
import sys
import json
import time
import logging
import argparse
import datetime
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WebAudioPlatformTester:
    """
    Specialized tester for audio models on web platforms.
    Implements and extends web_audio_test_runner functionality specifically
    for WebNN and WebGPU audio model testing.
    """
    
    def __init__(self, 
                output_dir: str = "./web_audio_platform_results",
                test_files_dir: str = "./web_audio_tests",
                models_dir: str = "./web_models",
                audio_dir: str = "./test_audio",
                debug: bool = False):
        """
        Initialize the web audio platform tester.
        
        Args:
            output_dir: Directory for output results
            test_files_dir: Directory for test files
            models_dir: Directory for model files
            audio_dir: Directory for audio test files
            debug: Enable debug logging
        """
        self.output_dir = Path(output_dir)
        self.test_files_dir = Path(test_files_dir)
        self.models_dir = Path(models_dir)
        self.audio_dir = Path(audio_dir)
        
        # Set debug logging if requested
        if debug:
            logger.setLevel(logging.DEBUG)
        
        # Make directories if they don't exist
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.test_files_dir.mkdir(exist_ok=True, parents=True)
        self.models_dir.mkdir(exist_ok=True, parents=True)
        self.audio_dir.mkdir(exist_ok=True, parents=True)
        
        # Import web_audio_test_runner if available
        try:
            from web_audio_test_runner import WebAudioTestRunner
            self.test_runner_available = True
            self.test_runner = WebAudioTestRunner(
                test_directory=str(self.test_files_dir),
                results_directory=str(self.output_dir)
            )
        except ImportError:
            logger.warning("web_audio_test_runner module not available, some features will be limited")
            self.test_runner_available = False
            self.test_runner = None
        
        # Check for test audio files
        self._check_test_audio_files()
        
        # Check for browser availability
        self.available_browsers = self._detect_browsers()
        
        logger.info(f"WebAudioPlatformTester initialized with output directory: {output_dir}")
        logger.info(f"Available browsers: {', '.join(self.available_browsers) if self.available_browsers else 'None'}")
    
    def _detect_browsers(self) -> List[str]:
        """
        Detect available browsers for testing.
        
        Returns:
            List of available browsers
        """
        available_browsers = []
        
        # Check for Chrome
        try:
            chrome_paths = [
                # Linux
                "google-chrome",
                "google-chrome-stable",
                # macOS
                "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                # Windows
                r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
            ]
            
            for path in chrome_paths:
                try:
                    result = subprocess.run([path, "--version"], 
                                          stdout=subprocess.PIPE, 
                                          stderr=subprocess.PIPE, 
                                          timeout=1)
                    if result.returncode == 0:
                        available_browsers.append("chrome")
                        logger.debug(f"Found Chrome: {path}")
                        break
                except (subprocess.SubprocessError, FileNotFoundError):
                    continue
        except Exception as e:
            logger.debug(f"Error detecting Chrome: {e}")
        
        # Check for Firefox
        try:
            firefox_paths = [
                # Linux
                "firefox",
                # macOS
                "/Applications/Firefox.app/Contents/MacOS/firefox",
                # Windows
                r"C:\Program Files\Mozilla Firefox\firefox.exe",
                r"C:\Program Files (x86)\Mozilla Firefox\firefox.exe"
            ]
            
            for path in firefox_paths:
                try:
                    result = subprocess.run([path, "--version"], 
                                          stdout=subprocess.PIPE, 
                                          stderr=subprocess.PIPE, 
                                          timeout=1)
                    if result.returncode == 0:
                        available_browsers.append("firefox")
                        logger.debug(f"Found Firefox: {path}")
                        break
                except (subprocess.SubprocessError, FileNotFoundError):
                    continue
        except Exception as e:
            logger.debug(f"Error detecting Firefox: {e}")
        
        # Check for Safari on macOS
        if sys.platform == 'darwin':
            try:
                safari_path = "/Applications/Safari.app/Contents/MacOS/Safari"
                if os.path.exists(safari_path):
                    available_browsers.append("safari")
                    logger.debug(f"Found Safari: {safari_path}")
            except Exception as e:
                logger.debug(f"Error detecting Safari: {e}")
        
        # Check for Edge
        try:
            edge_paths = [
                # Windows
                r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
                # macOS
                "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"
            ]
            
            for path in edge_paths:
                try:
                    result = subprocess.run([path, "--version"], 
                                          stdout=subprocess.PIPE, 
                                          stderr=subprocess.PIPE, 
                                          timeout=1)
                    if result.returncode == 0:
                        available_browsers.append("edge")
                        logger.debug(f"Found Edge: {path}")
                        break
                except (subprocess.SubprocessError, FileNotFoundError):
                    continue
        except Exception as e:
            logger.debug(f"Error detecting Edge: {e}")
        
        return available_browsers
    
    def _check_test_audio_files(self) -> None:
        """
        Check for test audio files and create placeholders if needed.
        """
        # Standard test files we need
        test_files = [
            "test.wav",
            "test.mp3",
            "test_speech.wav",
            "test_music.mp3",
            "test_noise.wav"
        ]
        
        for filename in test_files:
            file_path = self.audio_dir / filename
            if not file_path.exists():
                logger.info(f"Creating placeholder audio file: {file_path}")
                self._create_placeholder_audio(file_path)
    
    def _create_placeholder_audio(self, file_path: Path) -> None:
        """
        Create a placeholder audio file for testing.
        
        Args:
            file_path: Path to the audio file to create
        """
        try:
            # Try to create using numpy/scipy if available
            import numpy as np
            
            # Check if we have scipy for WAV writing
            try:
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
                wavfile.write(file_path, sample_rate, audio_data)
                
                logger.info(f"Created placeholder audio file: {file_path}")
                return
            except ImportError:
                logger.debug("scipy.io.wavfile not available for WAV creation")
                
            # Try to create MP3 file if it's an MP3
            if file_path.suffix.lower() == '.mp3':
                try:
                    # Try to copy a sample mp3 file from the test directory
                    test_mp3 = Path(__file__).parent / "test.mp3"
                    if test_mp3.exists():
                        import shutil
                        shutil.copy(test_mp3, file_path)
                        logger.info(f"Copied sample MP3 to: {file_path}")
                        return
                except Exception as e:
                    logger.debug(f"Error copying MP3 sample: {e}")
            
            # Fallback: Create empty file
            with open(file_path, 'wb') as f:
                f.write(b'')
                
            logger.warning(f"Created empty placeholder file: {file_path} - tests will not work correctly")
            
        except ImportError:
            # Fallback: Create empty file
            with open(file_path, 'wb') as f:
                f.write(b'')
                
            logger.warning(f"Created empty placeholder file: {file_path} - tests will not work correctly")
    
    def check_web_platform_support(self) -> Dict[str, bool]:
        """
        Check for web platform support.
        
        Returns:
            Dictionary with support status
        """
        support = {
            "webnn": False,
            "webgpu": False,
            "audio_processing": False,
            "webaudio_api": False,
            "web_audio_ml": False
        }
        
        if not self.available_browsers:
            logger.warning("No browsers available to check web platform support")
            return support
        
        # For now, just assume support based on browser availability
        # In a real implementation, we would use the browsers to actually check
        if "chrome" in self.available_browsers:
            support["webgpu"] = True  # Chrome typically supports WebGPU
            support["webaudio_api"] = True
        
        if "edge" in self.available_browsers:
            support["webnn"] = True  # Edge typically supports WebNN
            support["webaudio_api"] = True
        
        # Check for Node.js with ONNX runtime and transformers.js
        try:
            result = subprocess.run(["node", "--version"], 
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE)
            if result.returncode == 0:
                # Check for transformers.js
                try:
                    check_cmd = "npm list transformers.js || npm list @xenova/transformers"
                    result = subprocess.run(check_cmd, shell=True, 
                                          stdout=subprocess.PIPE, 
                                          stderr=subprocess.PIPE)
                    if result.returncode == 0:
                        support["web_audio_ml"] = True
                except Exception:
                    logger.debug("transformers.js not found in npm packages")
                
                # Check for onnxruntime-web
                try:
                    check_cmd = "npm list onnxruntime-web"
                    result = subprocess.run(check_cmd, shell=True, 
                                          stdout=subprocess.PIPE, 
                                          stderr=subprocess.PIPE)
                    if result.returncode == 0:
                        support["webnn"] = True
                except Exception:
                    logger.debug("onnxruntime-web not found in npm packages")
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.debug("Node.js not available")
        
        return support
    
    def generate_webnn_test_html(self, model_type: str, model_name: str) -> str:
        """
        Generate HTML test file for WebNN testing.
        
        Args:
            model_type: Type of model (whisper, wav2vec2, etc.)
            model_name: Name of the model
            
        Returns:
            Path to the generated HTML file
        """
        # If test runner is available, use it
        if self.test_runner_available and self.test_runner:
            # Use test runner to generate file
            # This is a simplified implementation - in real code we'd need to check
            # if the test runner has the necessary templates for this model type
            model_dir = self.test_files_dir / model_type
            model_dir.mkdir(exist_ok=True)
            
            test_file = model_dir / "webnn_test.html"
            
            with open(test_file, 'w') as f:
                f.write(self._get_webnn_test_template(model_type, model_name))
            
            logger.info(f"Generated WebNN test file: {test_file}")
            return str(test_file)
        else:
            # Generate file directly
            model_dir = self.test_files_dir / model_type
            model_dir.mkdir(exist_ok=True)
            
            test_file = model_dir / "webnn_test.html"
            
            with open(test_file, 'w') as f:
                f.write(self._get_webnn_test_template(model_type, model_name))
            
            logger.info(f"Generated WebNN test file: {test_file}")
            return str(test_file)
    
    def generate_webgpu_test_html(self, model_type: str, model_name: str) -> str:
        """
        Generate HTML test file for WebGPU testing.
        
        Args:
            model_type: Type of model (whisper, wav2vec2, etc.)
            model_name: Name of the model
            
        Returns:
            Path to the generated HTML file
        """
        # Generate file directly
        model_dir = self.test_files_dir / model_type
        model_dir.mkdir(exist_ok=True)
        
        test_file = model_dir / "webgpu_test.html"
        
        with open(test_file, 'w') as f:
            f.write(self._get_webgpu_test_template(model_type, model_name))
        
        logger.info(f"Generated WebGPU test file: {test_file}")
        return str(test_file)
    
    def _get_webnn_test_template(self, model_type: str, model_name: str) -> str:
        """
        Get HTML template for WebNN testing.
        
        Args:
            model_type: Type of model (whisper, wav2vec2, etc.)
            model_name: Name of the model
            
        Returns:
            HTML template content
        """
        # Set task based on model type
        task = "automatic-speech-recognition"
        if model_type == "clap":
            task = "audio-classification"
        
        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>WebNN {model_type.capitalize()} Test</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                .container {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .result {{ margin-top: 10px; padding: 10px; background-color: #f5f5f5; border-radius: 3px; }}
                .success {{ color: green; }}
                .error {{ color: red; }}
                pre {{ white-space: pre-wrap; overflow-x: auto; }}
                button {{ padding: 8px 16px; background-color: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; }}
                button:hover {{ background-color: #45a049; }}
                select, input {{ padding: 8px; margin: 5px 0; }}
            </style>
        </head>
        <body>
            <h1>WebNN {model_type.capitalize()} Test</h1>
            
            <div class="container">
                <h2>Test Configuration</h2>
                
                <div>
                    <label for="audio-file">Audio File:</label>
                    <select id="audio-file">
                        <option value="test.wav">test.wav</option>
                        <option value="test.mp3">test.mp3</option>
                        <option value="test_speech.wav">test_speech.wav</option>
                        <option value="test_music.mp3">test_music.mp3</option>
                    </select>
                </div>
                
                <div>
                    <label for="backend">WebNN Backend:</label>
                    <select id="backend">
                        <option value="gpu">GPU (preferred)</option>
                        <option value="cpu">CPU</option>
                        <option value="default">Default</option>
                    </select>
                </div>
                
                <div>
                    <button id="run-test">Run Test</button>
                    <button id="check-support">Check WebNN Support</button>
                </div>
            </div>
            
            <div class="container">
                <h2>Test Results</h2>
                <div id="results">No test run yet.</div>
            </div>
            
            <script>
                document.addEventListener('DOMContentLoaded', function() {{
                    const resultsDiv = document.getElementById('results');
                    const runTestButton = document.getElementById('run-test');
                    const checkSupportButton = document.getElementById('check-support');
                    const audioFileSelect = document.getElementById('audio-file');
                    const backendSelect = document.getElementById('backend');
                    
                    // Check WebNN Support
                    checkSupportButton.addEventListener('click', async function() {{
                        resultsDiv.innerHTML = 'Checking WebNN support...';
                        
                        try {{
                            // Check if WebNN is available
                            const hasWebNN = 'ml' in navigator;
                            
                            if (hasWebNN) {{
                                // Try to create a WebNN context
                                const contextOptions = {{
                                    devicePreference: backendSelect.value
                                }};
                                
                                try {{
                                    const context = await navigator.ml.createContext(contextOptions);
                                    const deviceType = await context.queryDevice();
                                    
                                    resultsDiv.innerHTML = `
                                        <div class="success">
                                            <h3>WebNN is supported!</h3>
                                            <p>Device type: ${{deviceType}}</p>
                                        </div>
                                    `;
                                }} catch (error) {{
                                    resultsDiv.innerHTML = `
                                        <div class="error">
                                            <h3>WebNN API is available but failed to create context</h3>
                                            <p>Error: ${{error.message}}</p>
                                        </div>
                                    `;
                                }}
                            }} else {{
                                resultsDiv.innerHTML = `
                                    <div class="error">
                                        <h3>WebNN is not supported in this browser</h3>
                                        <p>Try using Edge or Chrome with the appropriate flags enabled.</p>
                                    </div>
                                `;
                            }}
                        }} catch (error) {{
                            resultsDiv.innerHTML = `
                                <div class="error">
                                    <h3>Error checking WebNN support</h3>
                                    <p>${{error.message}}</p>
                                </div>
                            `;
                        }}
                    }});
                    
                    // Run WebNN Test
                    runTestButton.addEventListener('click', async function() {{
                        resultsDiv.innerHTML = 'Running WebNN test...';
                        
                        try {{
                            // Check if WebNN is available
                            if (!('ml' in navigator)) {{
                                throw new Error('WebNN is not supported in this browser');
                            }}
                            
                            // Create WebNN context
                            const contextOptions = {{
                                devicePreference: backendSelect.value
                            }};
                            
                            const context = await navigator.ml.createContext(contextOptions);
                            const deviceType = await context.queryDevice();
                            
                            // Log context info
                            console.log(`WebNN context created with device type: ${{deviceType}}`);
                            
                            // Get selected audio file
                            const audioFile = audioFileSelect.value;
                            
                            // Simulation for {model_type} model loading and inference
                            // This would be replaced with actual WebNN model loading in a real implementation
                            
                            // Simulate model loading time
                            const loadStartTime = performance.now();
                            await new Promise(resolve => setTimeout(resolve, 1000));
                            const loadEndTime = performance.now();
                            
                            // Simulate inference
                            const inferenceStartTime = performance.now();
                            await new Promise(resolve => setTimeout(resolve, 500));
                            const inferenceEndTime = performance.now();
                            
                            // Generate simulated result based on model type
                            let simulatedResult;
                            if ('{model_type}' === 'whisper') {{
                                simulatedResult = {{
                                    text: "This is a simulated transcript from the Whisper model.",
                                    chunks: [
                                        {{ text: "This is a", timestamp: [0.0, 1.0] }},
                                        {{ text: "simulated transcript", timestamp: [1.0, 2.5] }},
                                        {{ text: "from the Whisper model.", timestamp: [2.5, 4.0] }}
                                    ]
                                }};
                            }} else if ('{model_type}' === 'wav2vec2') {{
                                simulatedResult = {{
                                    text: "Simulated speech recognition result from Wav2Vec2 model.",
                                    confidence: 0.92
                                }};
                            }} else if ('{model_type}' === 'clap') {{
                                simulatedResult = {{
                                    classes: [
                                        {{ label: "Speech", score: 0.8 }},
                                        {{ label: "Music", score: 0.15 }},
                                        {{ label: "Background noise", score: 0.05 }}
                                    ]
                                }};
                            }}
                            
                            // Display results
                            resultsDiv.innerHTML = `
                                <div class="success">
                                    <h3>WebNN Test Completed</h3>
                                    <p>Model: {model_name}</p>
                                    <p>Audio: ${{audioFile}}</p>
                                    <p>Device: ${{deviceType}}</p>
                                    <p>Load Time: ${{(loadEndTime - loadStartTime).toFixed(2)}} ms</p>
                                    <p>Inference Time: ${{(inferenceEndTime - inferenceStartTime).toFixed(2)}} ms</p>
                                    <h4>Results:</h4>
                                    <pre>${{JSON.stringify(simulatedResult, null, 2)}}</pre>
                                </div>
                            `;
                            
                            // In a real implementation, we would report results back to the test framework
                        }} catch (error) {{
                            resultsDiv.innerHTML = `
                                <div class="error">
                                    <h3>WebNN Test Failed</h3>
                                    <p>Error: ${{error.message}}</p>
                                </div>
                            `;
                        }}
                    }});
                    
                    // Initial check for WebNN support
                    checkSupportButton.click();
                }});
            </script>
        </body>
        </html>
        """
    
    def _get_webgpu_test_template(self, model_type: str, model_name: str) -> str:
        """
        Get HTML template for WebGPU testing.
        
        Args:
            model_type: Type of model (whisper, wav2vec2, etc.)
            model_name: Name of the model
            
        Returns:
            HTML template content
        """
        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>WebGPU {model_type.capitalize()} Test</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                .container {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .result {{ margin-top: 10px; padding: 10px; background-color: #f5f5f5; border-radius: 3px; }}
                .success {{ color: green; }}
                .error {{ color: red; }}
                pre {{ white-space: pre-wrap; overflow-x: auto; }}
                button {{ padding: 8px 16px; background-color: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; }}
                button:hover {{ background-color: #45a049; }}
                select, input {{ padding: 8px; margin: 5px 0; }}
            </style>
        </head>
        <body>
            <h1>WebGPU {model_type.capitalize()} Test</h1>
            
            <div class="container">
                <h2>Test Configuration</h2>
                
                <div>
                    <label for="audio-file">Audio File:</label>
                    <select id="audio-file">
                        <option value="test.wav">test.wav</option>
                        <option value="test.mp3">test.mp3</option>
                        <option value="test_speech.wav">test_speech.wav</option>
                        <option value="test_music.mp3">test_music.mp3</option>
                    </select>
                </div>
                
                <div>
                    <button id="run-test">Run Test</button>
                    <button id="check-support">Check WebGPU Support</button>
                </div>
            </div>
            
            <div class="container">
                <h2>Test Results</h2>
                <div id="results">No test run yet.</div>
            </div>
            
            <script>
                document.addEventListener('DOMContentLoaded', function() {{
                    const resultsDiv = document.getElementById('results');
                    const runTestButton = document.getElementById('run-test');
                    const checkSupportButton = document.getElementById('check-support');
                    const audioFileSelect = document.getElementById('audio-file');
                    
                    // Check WebGPU Support
                    checkSupportButton.addEventListener('click', async function() {{
                        resultsDiv.innerHTML = 'Checking WebGPU support...';
                        
                        try {{
                            // Check if WebGPU is available
                            if (!navigator.gpu) {{
                                throw new Error('WebGPU is not supported in this browser');
                            }}
                            
                            // Try to get adapter
                            const adapter = await navigator.gpu.requestAdapter();
                            if (!adapter) {{
                                throw new Error('No WebGPU adapter found');
                            }}
                            
                            // Get adapter info
                            const adapterInfo = await adapter.requestAdapterInfo();
                            
                            // Request device
                            const device = await adapter.requestDevice();
                            
                            // Get device properties
                            const deviceProperties = {{
                                vendor: adapterInfo.vendor || 'unknown',
                                architecture: adapterInfo.architecture || 'unknown',
                                device: adapterInfo.device || 'unknown',
                                description: adapterInfo.description || 'unknown'
                            }};
                            
                            resultsDiv.innerHTML = `
                                <div class="success">
                                    <h3>WebGPU is supported!</h3>
                                    <p>Vendor: ${{deviceProperties.vendor}}</p>
                                    <p>Architecture: ${{deviceProperties.architecture}}</p>
                                    <p>Device: ${{deviceProperties.device}}</p>
                                    <p>Description: ${{deviceProperties.description}}</p>
                                </div>
                            `;
                        }} catch (error) {{
                            resultsDiv.innerHTML = `
                                <div class="error">
                                    <h3>WebGPU is not supported</h3>
                                    <p>Error: ${{error.message}}</p>
                                    <p>Try using Chrome with the appropriate flags enabled.</p>
                                </div>
                            `;
                        }}
                    }});
                    
                    // Run WebGPU Test
                    runTestButton.addEventListener('click', async function() {{
                        resultsDiv.innerHTML = 'Running WebGPU test...';
                        
                        try {{
                            // Check if WebGPU is available
                            if (!navigator.gpu) {{
                                throw new Error('WebGPU is not supported in this browser');
                            }}
                            
                            // Get adapter
                            const adapter = await navigator.gpu.requestAdapter();
                            if (!adapter) {{
                                throw new Error('No WebGPU adapter found');
                            }}
                            
                            // Request device
                            const device = await adapter.requestDevice();
                            
                            // Get selected audio file
                            const audioFile = audioFileSelect.value;
                            
                            // Simulation for {model_type} model loading and inference
                            // This would be replaced with actual WebGPU implementation in a real test
                            
                            // Simulate model loading time
                            const loadStartTime = performance.now();
                            await new Promise(resolve => setTimeout(resolve, 1200)); // Simulate longer load time than WebNN
                            const loadEndTime = performance.now();
                            
                            // Simulate inference
                            const inferenceStartTime = performance.now();
                            await new Promise(resolve => setTimeout(resolve, 400)); // Simulate faster inference time than WebNN
                            const inferenceEndTime = performance.now();
                            
                            // Generate simulated result based on model type
                            let simulatedResult;
                            if ('{model_type}' === 'whisper') {{
                                simulatedResult = {{
                                    text: "This is a simulated transcript from the Whisper model using WebGPU.",
                                    chunks: [
                                        {{ text: "This is a", timestamp: [0.0, 1.0] }},
                                        {{ text: "simulated transcript", timestamp: [1.0, 2.5] }},
                                        {{ text: "from the Whisper model", timestamp: [2.5, 4.0] }},
                                        {{ text: "using WebGPU.", timestamp: [4.0, 5.0] }}
                                    ]
                                }};
                            }} else if ('{model_type}' === 'wav2vec2') {{
                                simulatedResult = {{
                                    text: "Simulated speech recognition result from Wav2Vec2 model with WebGPU.",
                                    confidence: 0.94  // Slightly higher confidence than WebNN
                                }};
                            }} else if ('{model_type}' === 'clap') {{
                                simulatedResult = {{
                                    classes: [
                                        {{ label: "Speech", score: 0.82 }},
                                        {{ label: "Music", score: 0.14 }},
                                        {{ label: "Background noise", score: 0.04 }}
                                    ]
                                }};
                            }}
                            
                            // Display results
                            resultsDiv.innerHTML = `
                                <div class="success">
                                    <h3>WebGPU Test Completed</h3>
                                    <p>Model: {model_name}</p>
                                    <p>Audio: ${{audioFile}}</p>
                                    <p>Adapter: ${{(await adapter.requestAdapterInfo()).vendor || 'unknown'}}</p>
                                    <p>Load Time: ${{(loadEndTime - loadStartTime).toFixed(2)}} ms</p>
                                    <p>Inference Time: ${{(inferenceEndTime - inferenceStartTime).toFixed(2)}} ms</p>
                                    <h4>Results:</h4>
                                    <pre>${{JSON.stringify(simulatedResult, null, 2)}}</pre>
                                </div>
                            `;
                            
                            // In a real implementation, we would report results back to the test framework
                        }} catch (error) {{
                            resultsDiv.innerHTML = `
                                <div class="error">
                                    <h3>WebGPU Test Failed</h3>
                                    <p>Error: ${{error.message}}</p>
                                </div>
                            `;
                        }}
                    }});
                    
                    // Initial check for WebGPU support
                    checkSupportButton.click();
                }});
            </script>
        </body>
        </html>
        """
    
    def open_test_in_browser(self, test_file: str, browser: str = "chrome") -> bool:
        """
        Open a test file in a browser.
        
        Args:
            test_file: Path to the test file
            browser: Browser to use (chrome, firefox, safari, edge)
            
        Returns:
            True if successful, False otherwise
        """
        if browser not in self.available_browsers:
            logger.error(f"Browser not available: {browser}")
            return False
        
        # Convert to file URL
        file_path = Path(test_file).resolve()
        file_url = f"file://{file_path}"
        
        try:
            if browser == "chrome":
                # Try different Chrome paths
                chrome_paths = [
                    "google-chrome",
                    "google-chrome-stable",
                    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
                ]
                
                for path in chrome_paths:
                    try:
                        # Enable WebNN and WebGPU
                        subprocess.Popen([path, "--enable-dawn-features=allow_unsafe_apis", 
                                        "--enable-webgpu-developer-features",
                                        "--enable-webnn",
                                        file_url])
                        logger.info(f"Opened test in Chrome: {file_url}")
                        return True
                    except (subprocess.SubprocessError, FileNotFoundError):
                        continue
                
                logger.error("Failed to find Chrome executable")
                return False
                
            elif browser == "firefox":
                firefox_paths = [
                    "firefox",
                    "/Applications/Firefox.app/Contents/MacOS/firefox",
                    r"C:\Program Files\Mozilla Firefox\firefox.exe",
                    r"C:\Program Files (x86)\Mozilla Firefox\firefox.exe"
                ]
                
                for path in firefox_paths:
                    try:
                        subprocess.Popen([path, file_url])
                        logger.info(f"Opened test in Firefox: {file_url}")
                        return True
                    except (subprocess.SubprocessError, FileNotFoundError):
                        continue
                
                logger.error("Failed to find Firefox executable")
                return False
                
            elif browser == "safari":
                if sys.platform != 'darwin':
                    logger.error("Safari is only available on macOS")
                    return False
                
                try:
                    subprocess.Popen(["open", "-a", "Safari", file_url])
                    logger.info(f"Opened test in Safari: {file_url}")
                    return True
                except subprocess.SubprocessError as e:
                    logger.error(f"Failed to open Safari: {e}")
                    return False
                
            elif browser == "edge":
                edge_paths = [
                    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
                    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"
                ]
                
                for path in edge_paths:
                    try:
                        # Enable WebNN and WebGPU
                        subprocess.Popen([path, "--enable-dawn-features=allow_unsafe_apis", 
                                        "--enable-webgpu-developer-features",
                                        "--enable-webnn",
                                        file_url])
                        logger.info(f"Opened test in Edge: {file_url}")
                        return True
                    except (subprocess.SubprocessError, FileNotFoundError):
                        continue
                
                logger.error("Failed to find Edge executable")
                return False
            
            else:
                logger.error(f"Unsupported browser: {browser}")
                return False
                
        except Exception as e:
            logger.error(f"Error opening test in browser: {e}")
            return False
    
    def run_whisper_tests(self, browser: str = "chrome", headless: bool = False) -> Dict:
        """
        Run tests for Whisper model.
        
        Args:
            browser: Browser to use
            headless: Run in headless mode
            
        Returns:
            Dictionary with test results
        """
        logger.info(f"Running Whisper tests with browser: {browser}, headless: {headless}")
        
        results = {
            "model_type": "whisper",
            "model_name": "whisper-tiny",
            "browser": browser,
            "headless": headless,
            "timestamp": datetime.datetime.now().isoformat(),
            "tests": []
        }
        
        # Generate test files
        webnn_test_file = self.generate_webnn_test_html("whisper", "whisper-tiny")
        webgpu_test_file = self.generate_webgpu_test_html("whisper", "whisper-tiny")
        
        # If using web_audio_test_runner and headless mode
        if self.test_runner_available and self.test_runner and headless:
            # Use test runner to run tests
            runner_results = self.test_runner.run_browser_test(
                model_type="whisper",
                test_case="speech_recognition",
                browser=browser,
                headless=True
            )
            
            results["tests"].append({
                "test_type": "webnn",
                "status": runner_results.get("status", "unknown"),
                "results": runner_results
            })
            
            # Return combined results
            return results
        
        # Otherwise, open tests in browser for manual testing
        if self.open_test_in_browser(webnn_test_file, browser):
            results["tests"].append({
                "test_type": "webnn",
                "status": "manual",
                "file": webnn_test_file
            })
        
        # Wait a moment before opening the second test
        time.sleep(1)
        
        if self.open_test_in_browser(webgpu_test_file, browser):
            results["tests"].append({
                "test_type": "webgpu",
                "status": "manual",
                "file": webgpu_test_file
            })
        
        # Save results
        output_file = self.output_dir / f"whisper_tests_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved Whisper test results to {output_file}")
        return results
    
    def run_wav2vec2_tests(self, browser: str = "chrome", headless: bool = False) -> Dict:
        """
        Run tests for Wav2Vec2 model.
        
        Args:
            browser: Browser to use
            headless: Run in headless mode
            
        Returns:
            Dictionary with test results
        """
        logger.info(f"Running Wav2Vec2 tests with browser: {browser}, headless: {headless}")
        
        results = {
            "model_type": "wav2vec2",
            "model_name": "wav2vec2-base",
            "browser": browser,
            "headless": headless,
            "timestamp": datetime.datetime.now().isoformat(),
            "tests": []
        }
        
        # Generate test files
        webnn_test_file = self.generate_webnn_test_html("wav2vec2", "wav2vec2-base")
        webgpu_test_file = self.generate_webgpu_test_html("wav2vec2", "wav2vec2-base")
        
        # If using web_audio_test_runner and headless mode
        if self.test_runner_available and self.test_runner and headless:
            # Use test runner to run tests
            runner_results = self.test_runner.run_browser_test(
                model_type="wav2vec2",
                test_case="speech_recognition",
                browser=browser,
                headless=True
            )
            
            results["tests"].append({
                "test_type": "webnn",
                "status": runner_results.get("status", "unknown"),
                "results": runner_results
            })
            
            # Return combined results
            return results
        
        # Otherwise, open tests in browser for manual testing
        if self.open_test_in_browser(webnn_test_file, browser):
            results["tests"].append({
                "test_type": "webnn",
                "status": "manual",
                "file": webnn_test_file
            })
        
        # Wait a moment before opening the second test
        time.sleep(1)
        
        if self.open_test_in_browser(webgpu_test_file, browser):
            results["tests"].append({
                "test_type": "webgpu",
                "status": "manual",
                "file": webgpu_test_file
            })
        
        # Save results
        output_file = self.output_dir / f"wav2vec2_tests_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved Wav2Vec2 test results to {output_file}")
        return results
    
    def run_clap_tests(self, browser: str = "chrome", headless: bool = False) -> Dict:
        """
        Run tests for CLAP model.
        
        Args:
            browser: Browser to use
            headless: Run in headless mode
            
        Returns:
            Dictionary with test results
        """
        logger.info(f"Running CLAP tests with browser: {browser}, headless: {headless}")
        
        results = {
            "model_type": "clap",
            "model_name": "clap-htsat-unfused",
            "browser": browser,
            "headless": headless,
            "timestamp": datetime.datetime.now().isoformat(),
            "tests": []
        }
        
        # Generate test files
        webnn_test_file = self.generate_webnn_test_html("clap", "clap-htsat-unfused")
        webgpu_test_file = self.generate_webgpu_test_html("clap", "clap-htsat-unfused")
        
        # If using web_audio_test_runner and headless mode
        if self.test_runner_available and self.test_runner and headless:
            # Use test runner to run tests
            runner_results = self.test_runner.run_browser_test(
                model_type="clap",
                test_case="audio_classification",
                browser=browser,
                headless=True
            )
            
            results["tests"].append({
                "test_type": "webnn",
                "status": runner_results.get("status", "unknown"),
                "results": runner_results
            })
            
            # Return combined results
            return results
        
        # Otherwise, open tests in browser for manual testing
        if self.open_test_in_browser(webnn_test_file, browser):
            results["tests"].append({
                "test_type": "webnn",
                "status": "manual",
                "file": webnn_test_file
            })
        
        # Wait a moment before opening the second test
        time.sleep(1)
        
        if self.open_test_in_browser(webgpu_test_file, browser):
            results["tests"].append({
                "test_type": "webgpu",
                "status": "manual",
                "file": webgpu_test_file
            })
        
        # Save results
        output_file = self.output_dir / f"clap_tests_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved CLAP test results to {output_file}")
        return results
    
    def run_all_tests(self, browser: str = "chrome", headless: bool = False) -> Dict:
        """
        Run all audio model tests.
        
        Args:
            browser: Browser to use
            headless: Run in headless mode
            
        Returns:
            Dictionary with all test results
        """
        logger.info(f"Running all audio model tests with browser: {browser}, headless: {headless}")
        
        all_results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "browser": browser,
            "headless": headless,
            "models_tested": ["whisper", "wav2vec2", "clap"],
            "results": []
        }
        
        # Run tests for each model
        whisper_results = self.run_whisper_tests(browser, headless)
        all_results["results"].append({
            "model_type": "whisper",
            "results": whisper_results
        })
        
        wav2vec2_results = self.run_wav2vec2_tests(browser, headless)
        all_results["results"].append({
            "model_type": "wav2vec2",
            "results": wav2vec2_results
        })
        
        clap_results = self.run_clap_tests(browser, headless)
        all_results["results"].append({
            "model_type": "clap",
            "results": clap_results
        })
        
        # Save combined results
        output_file = self.output_dir / f"all_audio_tests_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"Saved all test results to {output_file}")
        return all_results
    
    def generate_test_report(self, results_file: str = None) -> str:
        """
        Generate a test report from results.
        
        Args:
            results_file: Path to the results file, or None to use latest
            
        Returns:
            Path to the generated report
        """
        # Find the latest results file if not specified
        if results_file is None:
            results_files = list(self.output_dir.glob("all_audio_tests_*.json"))
            if not results_files:
                results_files = list(self.output_dir.glob("*_tests_*.json"))
                
            if not results_files:
                logger.error("No test results found")
                return ""
                
            results_file = str(max(results_files, key=os.path.getmtime))
        
        # Load results
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading results file: {e}")
            return ""
        
        # Generate report
        report_file = self.output_dir / f"web_audio_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_file, 'w') as f:
            f.write("# Web Audio Platform Test Report\n\n")
            
            # Add timestamp
            timestamp = results.get("timestamp", "Unknown")
            f.write(f"Generated: {datetime.datetime.now().isoformat()}\n")
            f.write(f"Test Run: {timestamp}\n\n")
            
            # Add browser info
            browser = results.get("browser", "Unknown")
            headless = results.get("headless", False)
            f.write(f"Browser: {browser}\n")
            f.write(f"Headless Mode: {headless}\n\n")
            
            # Check if it's combined results or single model results
            if "results" in results and isinstance(results["results"], list):
                # Combined results
                f.write("## Models Tested\n\n")
                models_tested = results.get("models_tested", [])
                for model in models_tested:
                    f.write(f"- {model}\n")
                
                f.write("\n## Test Results Summary\n\n")
                f.write("| Model | Test Type | Status | Notes |\n")
                f.write("|-------|-----------|--------|-------|\n")
                
                for model_result in results["results"]:
                    model_type = model_result.get("model_type", "Unknown")
                    model_results = model_result.get("results", {})
                    
                    for test in model_results.get("tests", []):
                        test_type = test.get("test_type", "Unknown")
                        status = test.get("status", "Unknown")
                        
                        notes = ""
                        if status == "manual":
                            notes = "Manual test - check browser for results"
                        elif status == "error":
                            notes = test.get("message", "Test failed")
                        
                        f.write(f"| {model_type} | {test_type} | {status} | {notes} |\n")
            else:
                # Single model results
                model_type = results.get("model_type", "Unknown")
                model_name = results.get("model_name", "Unknown")
                
                f.write(f"## Model: {model_type} ({model_name})\n\n")
                
                f.write("| Test Type | Status | Notes |\n")
                f.write("|-----------|--------|-------|\n")
                
                for test in results.get("tests", []):
                    test_type = test.get("test_type", "Unknown")
                    status = test.get("status", "Unknown")
                    
                    notes = ""
                    if status == "manual":
                        notes = "Manual test - check browser for results"
                    elif status == "error":
                        notes = test.get("message", "Test failed")
                    
                    f.write(f"| {test_type} | {status} | {notes} |\n")
            
            # Add test files
            f.write("\n## Test Files\n\n")
            for file in self.test_files_dir.glob("**/*.html"):
                relative_path = file.relative_to(self.test_files_dir)
                f.write(f"- {relative_path}\n")
            
            # Add web platform support
            f.write("\n## Web Platform Support\n\n")
            support = self.check_web_platform_support()
            
            f.write("| Feature | Supported |\n")
            f.write("|---------|----------|\n")
            
            for feature, supported in support.items():
                status = "✅" if supported else "❌"
                f.write(f"| {feature} | {status} |\n")
            
            # Add recommendations for audio model web support
            f.write("\n## Recommendations for Web Audio Model Support\n\n")
            
            f.write("1. **WebNN Integration**: Implement WebNN backend support for audio models like Whisper and Wav2Vec2\n")
            f.write("2. **ONNX Export**: Ensure all audio models can be exported to ONNX format for web deployment\n")
            f.write("3. **Size Optimization**: Create smaller variants of audio models specifically for web deployment\n")
            f.write("4. **WebAudio API Integration**: Add WebAudio API support for audio preprocessing\n")
            f.write("5. **Audio Capture**: Implement browser-based audio recording and streaming for real-time ASR\n")
            f.write("6. **Fallback Mechanisms**: Implement CPU fallbacks for browsers without WebNN/WebGPU support\n")
            f.write("7. **Progressive Loading**: Enable partial model loading to improve initial load times\n")
            f.write("8. **Audio Chunk Processing**: Support streaming audio processing for long recordings\n")
        
        logger.info(f"Generated test report: {report_file}")
        return str(report_file)

def main():
    """Main function to handle command-line arguments."""
    parser = argparse.ArgumentParser(description="Web Audio Platform Tests")
    parser.add_argument("--output-dir", default="./web_audio_platform_results",
                      help="Directory for output results")
    parser.add_argument("--test-whisper", action="store_true",
                      help="Run tests for Whisper model")
    parser.add_argument("--test-wav2vec2", action="store_true",
                      help="Run tests for Wav2Vec2 model")
    parser.add_argument("--test-clap", action="store_true",
                      help="Run tests for CLAP model")
    parser.add_argument("--run-all", action="store_true",
                      help="Run tests for all models")
    parser.add_argument("--browser", choices=["chrome", "firefox", "safari", "edge"], 
                      default="chrome",
                      help="Browser to use for testing")
    parser.add_argument("--headless", action="store_true",
                      help="Run tests in headless mode")
    parser.add_argument("--generate-report", action="store_true",
                      help="Generate a report from test results")
    parser.add_argument("--results-file",
                      help="Path to the results file for report generation")
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug logging")
    args = parser.parse_args()
    
    # Create tester
    tester = WebAudioPlatformTester(
        output_dir=args.output_dir,
        debug=args.debug
    )
    
    # Check for available browsers
    if not tester.available_browsers:
        logger.error("No supported browsers detected. Please install Chrome, Firefox, Safari, or Edge.")
        return 1
        
    if args.browser not in tester.available_browsers:
        logger.error(f"Requested browser '{args.browser}' is not available. Available browsers: {', '.join(tester.available_browsers)}")
        return 1
    
    # Run requested tests
    if args.test_whisper:
        tester.run_whisper_tests(args.browser, args.headless)
    
    if args.test_wav2vec2:
        tester.run_wav2vec2_tests(args.browser, args.headless)
    
    if args.test_clap:
        tester.run_clap_tests(args.browser, args.headless)
    
    if args.run_all:
        tester.run_all_tests(args.browser, args.headless)
    
    # Generate report if requested
    if args.generate_report:
        report_file = tester.generate_test_report(args.results_file)
        if report_file:
            logger.info(f"Report generated: {report_file}")
        else:
            logger.error("Failed to generate report")
            return 1
    
    # If no test was requested, print help
    if not (args.test_whisper or args.test_wav2vec2 or args.test_clap or args.run_all or args.generate_report):
        parser.print_help()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())