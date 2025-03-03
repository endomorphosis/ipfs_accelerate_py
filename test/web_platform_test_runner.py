#!/usr/bin/env python
"""
Web Platform Test Runner for the IPFS Accelerate Python Framework.

This module provides a comprehensive testing framework for running HuggingFace models
on web platforms (WebNN and WebGPU), supporting text, vision, and multimodal models.

Usage:
    python web_platform_test_runner.py --model bert-base-uncased --hardware webnn
    python web_platform_test_runner.py --model vit-base-patch16-224 --hardware webgpu
    python web_platform_test_runner.py --model all-key-models --generate-report
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

# Define the high priority models (same as in benchmark_all_key_models.py)
HIGH_PRIORITY_MODELS = {
    "bert": {"name": "bert-base-uncased", "family": "embedding", "modality": "text"},
    "clap": {"name": "laion/clap-htsat-unfused", "family": "audio", "modality": "audio"},
    "clip": {"name": "openai/clip-vit-base-patch32", "family": "multimodal", "modality": "multimodal"},
    "detr": {"name": "facebook/detr-resnet-50", "family": "vision", "modality": "vision"},
    "llama": {"name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "family": "text_generation", "modality": "text"},
    "llava": {"name": "llava-hf/llava-1.5-7b-hf", "family": "multimodal", "modality": "multimodal"},
    "llava_next": {"name": "llava-hf/llava-v1.6-mistral-7b", "family": "multimodal", "modality": "multimodal"},
    "qwen2": {"name": "Qwen/Qwen2-0.5B-Instruct", "family": "text_generation", "modality": "text"},
    "t5": {"name": "t5-small", "family": "text_generation", "modality": "text"},
    "vit": {"name": "google/vit-base-patch16-224", "family": "vision", "modality": "vision"},
    "wav2vec2": {"name": "facebook/wav2vec2-base", "family": "audio", "modality": "audio"},
    "whisper": {"name": "openai/whisper-tiny", "family": "audio", "modality": "audio"},
    "xclip": {"name": "microsoft/xclip-base-patch32", "family": "multimodal", "modality": "multimodal"}
}

# Smaller versions for testing
SMALL_VERSIONS = {
    "bert": "prajjwal1/bert-tiny",
    "t5": "google/t5-efficient-tiny",
    "vit": "facebook/deit-tiny-patch16-224",
    "whisper": "openai/whisper-tiny",
    "llama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "qwen2": "Qwen/Qwen2-0.5B-Instruct"
}

class WebPlatformTestRunner:
    """
    Framework for testing HuggingFace models on web platforms (WebNN and WebGPU).
    """
    
    def __init__(self, 
                 output_dir: str = "./web_platform_results",
                 test_files_dir: str = "./web_platform_tests",
                 models_dir: str = "./web_models",
                 sample_data_dir: str = "./sample_data",
                 use_small_models: bool = True,
                 debug: bool = False):
        """
        Initialize the web platform test runner.
        
        Args:
            output_dir: Directory for output results
            test_files_dir: Directory for test files
            models_dir: Directory for model files
            sample_data_dir: Directory for sample data files
            use_small_models: Use smaller model variants when available
            debug: Enable debug logging
        """
        self.output_dir = Path(output_dir)
        self.test_files_dir = Path(test_files_dir)
        self.models_dir = Path(models_dir)
        self.sample_data_dir = Path(sample_data_dir)
        self.use_small_models = use_small_models
        
        # Set debug logging if requested
        if debug:
            logger.setLevel(logging.DEBUG)
        
        # Make directories if they don't exist
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.test_files_dir.mkdir(exist_ok=True, parents=True)
        self.models_dir.mkdir(exist_ok=True, parents=True)
        self.sample_data_dir.mkdir(exist_ok=True, parents=True)
        
        # Check for sample data files
        self._check_sample_data_files()
        
        # Check for browser availability
        self.available_browsers = self._detect_browsers()
        
        # Get models to test
        self.models = self._get_models()
        
        logger.info(f"WebPlatformTestRunner initialized with output directory: {output_dir}")
        logger.info(f"Available browsers: {', '.join(self.available_browsers) if self.available_browsers else 'None'}")
    
    def _get_models(self) -> Dict[str, Dict[str, str]]:
        """
        Get the models to test, using small variants if requested.
        
        Returns:
            Dictionary of models to test
        """
        models = {}
        
        for key, model_info in HIGH_PRIORITY_MODELS.items():
            model_data = model_info.copy()
            
            # Use small version if available and requested
            if self.use_small_models and key in SMALL_VERSIONS:
                model_data["name"] = SMALL_VERSIONS[key]
                model_data["size"] = "small"
            else:
                model_data["size"] = "base"
                
            models[key] = model_data
            
        return models
    
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
    
    def _check_sample_data_files(self) -> None:
        """
        Check for sample data files and create placeholders if needed.
        """
        # Sample data for different modalities
        sample_files = {
            "text": ["sample.txt", "sample_paragraph.txt"],
            "image": ["sample.jpg", "sample_image.png"],
            "audio": ["sample.wav", "sample.mp3"],
            "video": ["sample.mp4"]
        }
        
        for modality, files in sample_files.items():
            modality_dir = self.sample_data_dir / modality
            modality_dir.mkdir(exist_ok=True, parents=True)
            
            for filename in files:
                file_path = modality_dir / filename
                if not file_path.exists():
                    logger.info(f"Creating placeholder {modality} file: {file_path}")
                    self._create_placeholder_file(file_path, modality)
    
    def _create_placeholder_file(self, file_path: Path, modality: str) -> None:
        """
        Create a placeholder file for testing.
        
        Args:
            file_path: Path to the file to create
            modality: Type of file (text, image, audio, video)
        """
        try:
            # Check if we can copy from test directory
            test_file = Path(__file__).parent / "test" / file_path.name
            if test_file.exists():
                import shutil
                shutil.copy(test_file, file_path)
                logger.info(f"Copied sample file to: {file_path}")
                return
                
            # Otherwise create placeholder
            if modality == "text":
                with open(file_path, 'w') as f:
                    f.write("This is a sample text file for testing natural language processing models.\n")
                    f.write("It contains multiple sentences that can be used for inference tasks.\n")
                    f.write("The quick brown fox jumps over the lazy dog.\n")
            elif modality == "image":
                # Create a small blank image
                try:
                    from PIL import Image
                    img = Image.new('RGB', (224, 224), color='white')
                    img.save(file_path)
                except ImportError:
                    # If PIL not available, create empty file
                    with open(file_path, 'wb') as f:
                        f.write(b'')
                    logger.warning(f"PIL not available, created empty image file: {file_path}")
            else:
                # For audio and video, just create empty file
                with open(file_path, 'wb') as f:
                    f.write(b'')
                logger.warning(f"Created empty {modality} file: {file_path}")
                
        except Exception as e:
            logger.error(f"Error creating placeholder file {file_path}: {e}")
            # Create empty file as fallback
            with open(file_path, 'wb') as f:
                f.write(b'')
    
    def check_web_platform_support(self, platform: str = "webnn") -> Dict[str, bool]:
        """
        Check for web platform support.
        
        Args:
            platform: Web platform to check (webnn or webgpu)
            
        Returns:
            Dictionary with support status
        """
        support = {
            "available": False,
            "transformers_js": False,
            "onnx_runtime": False,
            "web_browser": False
        }
        
        if not self.available_browsers:
            logger.warning("No browsers available to check web platform support")
            return support
        
        # Check browser support
        if platform == "webnn" and "edge" in self.available_browsers:
            support["web_browser"] = True
        elif platform == "webgpu" and "chrome" in self.available_browsers:
            support["web_browser"] = True
        
        # Check for Node.js with transformers.js
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
                        support["transformers_js"] = True
                except Exception:
                    logger.debug("transformers.js not found in npm packages")
                
                # Check for onnxruntime-web
                try:
                    check_cmd = "npm list onnxruntime-web"
                    result = subprocess.run(check_cmd, shell=True, 
                                          stdout=subprocess.PIPE, 
                                          stderr=subprocess.PIPE)
                    if result.returncode == 0:
                        support["onnx_runtime"] = True
                except Exception:
                    logger.debug("onnxruntime-web not found in npm packages")
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.debug("Node.js not available")
        
        # Mark as available if we have browser and either transformers.js or onnxruntime
        support["available"] = support["web_browser"] and (support["transformers_js"] or support["onnx_runtime"])
        
        return support
    
    def generate_test_html(self, model_key: str, platform: str) -> str:
        """
        Generate HTML test file for web platform testing.
        
        Args:
            model_key: Key of the model to test (bert, vit, etc.)
            platform: Web platform to test (webnn or webgpu)
            
        Returns:
            Path to the generated HTML file
        """
        model_info = self.models.get(model_key)
        if not model_info:
            logger.error(f"Model key not found: {model_key}")
            return ""
        
        model_name = model_info["name"]
        model_family = model_info["family"]
        modality = model_info["modality"]
        
        # Create directory for this model's tests
        model_dir = self.test_files_dir / model_key
        model_dir.mkdir(exist_ok=True)
        
        # Create HTML file
        test_file = model_dir / f"{platform}_test.html"
        
        if platform == "webnn":
            template = self._get_webnn_test_template(model_key, model_name, modality)
        else:  # webgpu
            template = self._get_webgpu_test_template(model_key, model_name, modality)
        
        with open(test_file, 'w') as f:
            f.write(template)
        
        logger.info(f"Generated {platform} test file for {model_key}: {test_file}")
        return str(test_file)
    
    def _get_webnn_test_template(self, model_key: str, model_name: str, modality: str) -> str:
        """
        Get HTML template for WebNN testing.
        
        Args:
            model_key: Key of the model (bert, vit, etc.)
            model_name: Full name of the model
            modality: Modality of the model (text, image, audio, multimodal)
            
        Returns:
            HTML template content
        """
        # Set input type based on modality
        input_selector = ""
        if modality == "text":
            input_selector = """
            <div>
                <label for="text-input">Text Input:</label>
                <select id="text-input">
                    <option value="sample.txt">sample.txt</option>
                    <option value="sample_paragraph.txt">sample_paragraph.txt</option>
                    <option value="custom">Custom Text</option>
                </select>
                <textarea id="custom-text" style="display: none; width: 100%; height: 100px;">The quick brown fox jumps over the lazy dog.</textarea>
            </div>
            """
        elif modality == "image" or modality == "vision":
            input_selector = """
            <div>
                <label for="image-input">Image Input:</label>
                <select id="image-input">
                    <option value="sample.jpg">sample.jpg</option>
                    <option value="sample_image.png">sample_image.png</option>
                    <option value="upload">Upload Image</option>
                </select>
                <input type="file" id="image-upload" style="display: none;" accept="image/*">
            </div>
            """
        elif modality == "audio":
            input_selector = """
            <div>
                <label for="audio-input">Audio Input:</label>
                <select id="audio-input">
                    <option value="sample.wav">sample.wav</option>
                    <option value="sample.mp3">sample.mp3</option>
                    <option value="upload">Upload Audio</option>
                </select>
                <input type="file" id="audio-upload" style="display: none;" accept="audio/*">
            </div>
            """
        elif modality == "multimodal":
            input_selector = """
            <div>
                <label for="text-input">Text Input:</label>
                <select id="text-input">
                    <option value="sample.txt">sample.txt</option>
                    <option value="custom">Custom Text</option>
                </select>
                <textarea id="custom-text" style="display: none; width: 100%; height: 100px;">Describe this image in detail.</textarea>
            </div>
            <div>
                <label for="image-input">Image Input:</label>
                <select id="image-input">
                    <option value="sample.jpg">sample.jpg</option>
                    <option value="sample_image.png">sample_image.png</option>
                    <option value="upload">Upload Image</option>
                </select>
                <input type="file" id="image-upload" style="display: none;" accept="image/*">
            </div>
            """
        
        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>WebNN {model_key.capitalize()} Test</title>
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
                select, input, textarea {{ padding: 8px; margin: 5px 0; }}
            </style>
        </head>
        <body>
            <h1>WebNN {model_key.capitalize()} Test</h1>
            
            <div class="container">
                <h2>Test Configuration</h2>
                
                {input_selector}
                
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
                    const backendSelect = document.getElementById('backend');
                    
                    // Handle input selectors
                    const setupInputHandlers = () => {{
                        // Text input handling
                        const textInputSelect = document.getElementById('text-input');
                        const customTextArea = document.getElementById('custom-text');
                        
                        if (textInputSelect) {{
                            textInputSelect.addEventListener('change', function() {{
                                if (this.value === 'custom') {{
                                    customTextArea.style.display = 'block';
                                }} else {{
                                    customTextArea.style.display = 'none';
                                }}
                            }});
                        }}
                        
                        // Image input handling
                        const imageInputSelect = document.getElementById('image-input');
                        const imageUpload = document.getElementById('image-upload');
                        
                        if (imageInputSelect) {{
                            imageInputSelect.addEventListener('change', function() {{
                                if (this.value === 'upload') {{
                                    imageUpload.style.display = 'block';
                                    imageUpload.click();
                                }} else {{
                                    imageUpload.style.display = 'none';
                                }}
                            }});
                        }}
                        
                        // Audio input handling
                        const audioInputSelect = document.getElementById('audio-input');
                        const audioUpload = document.getElementById('audio-upload');
                        
                        if (audioInputSelect) {{
                            audioInputSelect.addEventListener('change', function() {{
                                if (this.value === 'upload') {{
                                    audioUpload.style.display = 'block';
                                    audioUpload.click();
                                }} else {{
                                    audioUpload.style.display = 'none';
                                }}
                            }});
                        }}
                    }};
                    
                    setupInputHandlers();
                    
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
                            
                            // Get input data based on modality
                            let inputData = 'No input data';
                            let inputType = '{modality}';
                            
                            // Simulation for {model_key} model loading and inference
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
                            if ('{model_key}' === 'bert') {{
                                simulatedResult = {{
                                    logits: [-0.2, 0.5, 1.2, -0.8, 0.3],
                                    embeddings: "[ array of 768 embedding values ]",
                                    tokens: ["[CLS]", "the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "[SEP]"]
                                }};
                            }} else if ('{model_key}' === 't5') {{
                                simulatedResult = {{
                                    generated_text: "This is a simulated T5 model output. The generated text contains information that would have been created based on the input."
                                }};
                            }} else if ('{model_key}' === 'vit') {{
                                simulatedResult = {{
                                    logits: [0.1, 0.2, 0.15, 0.55],
                                    predicted_class: "dog",
                                    confidence: 0.85
                                }};
                            }} else if ('{model_key}' === 'clip') {{
                                simulatedResult = {{
                                    image_features: "[ array of image embedding values ]",
                                    text_features: "[ array of text embedding values ]",
                                    similarity_score: 0.78
                                }};
                            }} else {{
                                simulatedResult = {{
                                    result: "Simulated output for {model_key} model",
                                    confidence: 0.92
                                }};
                            }}
                            
                            // Display results
                            resultsDiv.innerHTML = `
                                <div class="success">
                                    <h3>WebNN Test Completed</h3>
                                    <p>Model: {model_name}</p>
                                    <p>Input Type: ${{inputType}}</p>
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
    
    def _get_webgpu_test_template(self, model_key: str, model_name: str, modality: str) -> str:
        """
        Get HTML template for WebGPU testing.
        
        Args:
            model_key: Key of the model (bert, vit, etc.)
            model_name: Full name of the model
            modality: Modality of the model (text, image, audio, multimodal)
            
        Returns:
            HTML template content
        """
        # Set input type based on modality
        input_selector = ""
        if modality == "text":
            input_selector = """
            <div>
                <label for="text-input">Text Input:</label>
                <select id="text-input">
                    <option value="sample.txt">sample.txt</option>
                    <option value="sample_paragraph.txt">sample_paragraph.txt</option>
                    <option value="custom">Custom Text</option>
                </select>
                <textarea id="custom-text" style="display: none; width: 100%; height: 100px;">The quick brown fox jumps over the lazy dog.</textarea>
            </div>
            """
        elif modality == "image" or modality == "vision":
            input_selector = """
            <div>
                <label for="image-input">Image Input:</label>
                <select id="image-input">
                    <option value="sample.jpg">sample.jpg</option>
                    <option value="sample_image.png">sample_image.png</option>
                    <option value="upload">Upload Image</option>
                </select>
                <input type="file" id="image-upload" style="display: none;" accept="image/*">
            </div>
            """
        elif modality == "audio":
            input_selector = """
            <div>
                <label for="audio-input">Audio Input:</label>
                <select id="audio-input">
                    <option value="sample.wav">sample.wav</option>
                    <option value="sample.mp3">sample.mp3</option>
                    <option value="upload">Upload Audio</option>
                </select>
                <input type="file" id="audio-upload" style="display: none;" accept="audio/*">
            </div>
            """
        elif modality == "multimodal":
            input_selector = """
            <div>
                <label for="text-input">Text Input:</label>
                <select id="text-input">
                    <option value="sample.txt">sample.txt</option>
                    <option value="custom">Custom Text</option>
                </select>
                <textarea id="custom-text" style="display: none; width: 100%; height: 100px;">Describe this image in detail.</textarea>
            </div>
            <div>
                <label for="image-input">Image Input:</label>
                <select id="image-input">
                    <option value="sample.jpg">sample.jpg</option>
                    <option value="sample_image.png">sample_image.png</option>
                    <option value="upload">Upload Image</option>
                </select>
                <input type="file" id="image-upload" style="display: none;" accept="image/*">
            </div>
            """
        
        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>WebGPU {model_key.capitalize()} Test</title>
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
                select, input, textarea {{ padding: 8px; margin: 5px 0; }}
            </style>
        </head>
        <body>
            <h1>WebGPU {model_key.capitalize()} Test</h1>
            
            <div class="container">
                <h2>Test Configuration</h2>
                
                {input_selector}
                
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
                    
                    // Handle input selectors
                    const setupInputHandlers = () => {{
                        // Text input handling
                        const textInputSelect = document.getElementById('text-input');
                        const customTextArea = document.getElementById('custom-text');
                        
                        if (textInputSelect) {{
                            textInputSelect.addEventListener('change', function() {{
                                if (this.value === 'custom') {{
                                    customTextArea.style.display = 'block';
                                }} else {{
                                    customTextArea.style.display = 'none';
                                }}
                            }});
                        }}
                        
                        // Image input handling
                        const imageInputSelect = document.getElementById('image-input');
                        const imageUpload = document.getElementById('image-upload');
                        
                        if (imageInputSelect) {{
                            imageInputSelect.addEventListener('change', function() {{
                                if (this.value === 'upload') {{
                                    imageUpload.style.display = 'block';
                                    imageUpload.click();
                                }} else {{
                                    imageUpload.style.display = 'none';
                                }}
                            }});
                        }}
                        
                        // Audio input handling
                        const audioInputSelect = document.getElementById('audio-input');
                        const audioUpload = document.getElementById('audio-upload');
                        
                        if (audioInputSelect) {{
                            audioInputSelect.addEventListener('change', function() {{
                                if (this.value === 'upload') {{
                                    audioUpload.style.display = 'block';
                                    audioUpload.click();
                                }} else {{
                                    audioUpload.style.display = 'none';
                                }}
                            }});
                        }}
                    }};
                    
                    setupInputHandlers();
                    
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
                            
                            // Get input data based on modality
                            let inputData = 'No input data';
                            let inputType = '{modality}';
                            
                            // Simulation for {model_key} model loading and inference
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
                            if ('{model_key}' === 'bert') {{
                                simulatedResult = {{
                                    logits: [-0.15, 0.6, 1.3, -0.7, 0.35],
                                    embeddings: "[ array of 768 embedding values ]",
                                    tokens: ["[CLS]", "the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "[SEP]"]
                                }};
                            }} else if ('{model_key}' === 't5') {{
                                simulatedResult = {{
                                    generated_text: "This is a simulated T5 model output using WebGPU. The generated text would be based on the input provided."
                                }};
                            }} else if ('{model_key}' === 'vit') {{
                                simulatedResult = {{
                                    logits: [0.12, 0.22, 0.13, 0.53],
                                    predicted_class: "dog",
                                    confidence: 0.87
                                }};
                            }} else if ('{model_key}' === 'clip') {{
                                simulatedResult = {{
                                    image_features: "[ array of image embedding values ]",
                                    text_features: "[ array of text embedding values ]",
                                    similarity_score: 0.81
                                }};
                            }} else {{
                                simulatedResult = {{
                                    result: "Simulated output for {model_key} model using WebGPU",
                                    confidence: 0.94
                                }};
                            }}
                            
                            // Display results
                            resultsDiv.innerHTML = `
                                <div class="success">
                                    <h3>WebGPU Test Completed</h3>
                                    <p>Model: {model_name}</p>
                                    <p>Input Type: ${{inputType}}</p>
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
    
    def open_test_in_browser(self, test_file: str, platform: str = "webnn", headless: bool = False) -> bool:
        """
        Open a test file in a browser.
        
        Args:
            test_file: Path to the test file
            platform: Web platform to test (webnn or webgpu)
            headless: Run in headless mode
            
        Returns:
            True if successful, False otherwise
        """
        if platform == "webnn" and "edge" not in self.available_browsers:
            logger.error("Edge browser not available for WebNN tests")
            return False
        
        if platform == "webgpu" and "chrome" not in self.available_browsers:
            logger.error("Chrome browser not available for WebGPU tests")
            return False
        
        # Convert to file URL
        file_path = Path(test_file).resolve()
        file_url = f"file://{file_path}"
        
        try:
            if platform == "webnn":
                # Use Edge for WebNN
                edge_paths = [
                    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
                    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"
                ]
                
                for path in edge_paths:
                    try:
                        # Enable WebNN 
                        cmd = [path, "--enable-dawn-features=allow_unsafe_apis", 
                             "--enable-webgpu-developer-features",
                             "--enable-webnn"]
                        
                        if headless:
                            cmd.append("--headless=new")
                        
                        cmd.append(file_url)
                        
                        subprocess.Popen(cmd)
                        logger.info(f"Opened WebNN test in Edge: {file_url}")
                        return True
                    except (subprocess.SubprocessError, FileNotFoundError):
                        continue
                
                logger.error("Failed to find Edge executable")
                return False
                
            else:  # webgpu
                # Use Chrome for WebGPU
                chrome_paths = [
                    "google-chrome",
                    "google-chrome-stable",
                    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
                ]
                
                for path in chrome_paths:
                    try:
                        # Enable WebGPU
                        cmd = [path, "--enable-dawn-features=allow_unsafe_apis", 
                             "--enable-webgpu-developer-features"]
                        
                        if headless:
                            cmd.append("--headless=new")
                            
                        cmd.append(file_url)
                        
                        subprocess.Popen(cmd)
                        logger.info(f"Opened WebGPU test in Chrome: {file_url}")
                        return True
                    except (subprocess.SubprocessError, FileNotFoundError):
                        continue
                
                logger.error("Failed to find Chrome executable")
                return False
                
        except Exception as e:
            logger.error(f"Error opening test in browser: {e}")
            return False
    
    def run_model_test(self, model_key: str, platform: str = "webnn", headless: bool = False) -> Dict:
        """
        Run a test for a specific model on a web platform.
        
        Args:
            model_key: Key of the model to test
            platform: Web platform to test (webnn or webgpu)
            headless: Run in headless mode
            
        Returns:
            Dictionary with test results
        """
        if model_key not in self.models:
            logger.error(f"Model key not found: {model_key}")
            return {"status": "error", "message": f"Model key not found: {model_key}"}
        
        model_info = self.models[model_key]
        
        logger.info(f"Running {platform} test for {model_key} ({model_info['name']})")
        
        # Check platform support
        support = self.check_web_platform_support(platform)
        if not support["available"]:
            logger.warning(f"{platform} is not supported in the current environment")
            return {
                "model_key": model_key,
                "model_name": model_info["name"],
                "platform": platform,
                "status": "skipped",
                "reason": f"{platform} is not supported in the current environment",
                "timestamp": datetime.datetime.now().isoformat()
            }
        
        # Create results directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_result_dir = self.output_dir / f"{model_key}_{platform}_{timestamp}"
        model_result_dir.mkdir(exist_ok=True, parents=True)
        
        # Generate test HTML
        test_file = self.generate_test_html(model_key, platform)
        
        # Open in browser if not headless
        browser_opened = False
        if not headless:
            browser_opened = self.open_test_in_browser(test_file, platform, headless)
        
        # Create result
        result = {
            "model_key": model_key,
            "model_name": model_info["name"],
            "platform": platform,
            "status": "manual" if browser_opened else "automated",
            "test_file": test_file,
            "browser_opened": browser_opened,
            "headless": headless,
            "timestamp": datetime.datetime.now().isoformat(),
            "platform_support": support
        }
        
        # Save result
        result_file = model_result_dir / "result.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Test result saved to {result_file}")
        
        return result
    
    def run_all_models(self, platform: str = "webnn", headless: bool = False) -> Dict:
        """
        Run tests for all models on a web platform.
        
        Args:
            platform: Web platform to test (webnn or webgpu)
            headless: Run in headless mode
            
        Returns:
            Dictionary with all test results
        """
        logger.info(f"Running {platform} tests for all models")
        
        results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "platform": platform,
            "headless": headless,
            "models_tested": [],
            "results": []
        }
        
        # Run tests for key model categories
        for model_key, model_info in self.models.items():
            # Skip audio models, they're handled by web_audio_platform_tests.py
            if model_info["modality"] == "audio":
                logger.info(f"Skipping audio model {model_key}, handled by web_audio_platform_tests.py")
                continue
                
            model_result = self.run_model_test(model_key, platform, headless)
            results["models_tested"].append(model_key)
            results["results"].append(model_result)
            
            # Small delay between tests to avoid browser issues
            time.sleep(1)
        
        # Save combined results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"all_models_{platform}_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"All model test results saved to {results_file}")
        
        return results
    
    def generate_test_report(self, results_file: Optional[str] = None, platform: Optional[str] = None) -> str:
        """
        Generate a test report from results.
        
        Args:
            results_file: Path to the results file, or None to use latest
            platform: Filter report to specific platform
            
        Returns:
            Path to the generated report
        """
        # Find the latest results file if not specified
        if results_file is None:
            result_pattern = f"all_models_{'*' if platform is None else platform}_*.json"
            results_files = list(self.output_dir.glob(result_pattern))
            
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
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"web_platform_report_{timestamp}.md"
        
        with open(report_file, 'w') as f:
            f.write("# Web Platform Test Report\n\n")
            
            # Add timestamp
            test_timestamp = results.get("timestamp", "Unknown")
            f.write(f"Generated: {datetime.datetime.now().isoformat()}\n")
            f.write(f"Test Run: {test_timestamp}\n\n")
            
            # Add platform info
            test_platform = results.get("platform", "Unknown")
            headless = results.get("headless", False)
            f.write(f"Platform: {test_platform}\n")
            f.write(f"Headless Mode: {headless}\n\n")
            
            # Models tested
            f.write("## Models Tested\n\n")
            models_tested = results.get("models_tested", [])
            for model in models_tested:
                f.write(f"- {model}\n")
            
            f.write("\n## Test Results Summary\n\n")
            f.write("| Model | Modality | Status | Support |\n")
            f.write("|-------|----------|--------|----------|\n")
            
            model_results = results.get("results", [])
            for result in model_results:
                model_key = result.get("model_key", "Unknown")
                model_modality = ""
                
                # Look up modality from model key
                if model_key in self.models:
                    model_modality = self.models[model_key].get("modality", "")
                
                status = result.get("status", "Unknown")
                
                # Get support status
                support = "🟢 Supported"  # Default
                platform_support = result.get("platform_support", {})
                if platform_support:
                    if not platform_support.get("available", False):
                        support = "🔴 Not supported"
                    elif not platform_support.get("web_browser", False):
                        support = "🔸 Browser support missing"
                    elif not platform_support.get("transformers_js", False) and not platform_support.get("onnx_runtime", False):
                        support = "🔸 Runtime support missing"
                
                f.write(f"| {model_key} | {model_modality} | {status} | {support} |\n")
            
            # Add web platform support
            f.write("\n## Web Platform Support\n\n")
            
            # Collect support info from results
            platform_support = {}
            browser_support = {}
            
            for result in model_results:
                if "platform_support" in result:
                    for key, value in result["platform_support"].items():
                        platform_support[key] = platform_support.get(key, 0) + (1 if value else 0)
            
            # Calculate percentages
            total_models = len(model_results)
            if total_models > 0:
                f.write("| Feature | Support Rate |\n")
                f.write("|---------|-------------|\n")
                
                for key, count in platform_support.items():
                    percentage = (count / total_models) * 100
                    f.write(f"| {key} | {percentage:.1f}% ({count}/{total_models}) |\n")
            
            # Add recommendations
            f.write("\n## Recommendations\n\n")
            
            f.write("1. **Model Support Improvements**:\n")
            
            # Identify models with issues
            models_with_issues = []
            for result in model_results:
                if result.get("status") != "success":
                    models_with_issues.append(result.get("model_key", "Unknown"))
            
            if models_with_issues:
                f.write("   - Focus on improving support for these models: " + ", ".join(models_with_issues) + "\n")
            
            f.write("2. **Platform Integration Recommendations**:\n")
            
            if test_platform == "webnn":
                f.write("   - Improve WebNN support for transformers.js integration\n")
                f.write("   - Optimize model quantization for WebNN deployment\n")
                f.write("   - Provide pre-converted ONNX models for key model types\n")
            else:  # webgpu
                f.write("   - Extend WebGPU shader implementation for model inference\n")
                f.write("   - Implement tensor operation kernels specific to model families\n")
                f.write("   - Investigate WebGPU-specific optimizations for model weights\n")
                
            f.write("3. **General Web Platform Recommendations**:\n")
            f.write("   - Create API-compatible wrappers across WebNN and WebGPU for model inference\n")
            f.write("   - Implement automatic hardware selection based on available features\n")
            f.write("   - Develop model splitting techniques for larger models that exceed browser memory limits\n")
        
        logger.info(f"Generated test report: {report_file}")
        return str(report_file)

def main():
    """Main function to handle command-line arguments."""
    parser = argparse.ArgumentParser(description="Web Platform Model Tests")
    parser.add_argument("--output-dir", default="./web_platform_results",
                      help="Directory for output results")
    parser.add_argument("--model", required=False,
                      help="Model to test (bert, vit, clip, etc. or 'all-key-models')")
    parser.add_argument("--platform", choices=["webnn", "webgpu"], default="webnn",
                      help="Web platform to test")
    parser.add_argument("--browser", choices=["edge", "chrome"], 
                      help="Browser to use (defaults based on platform)")
    parser.add_argument("--headless", action="store_true",
                      help="Run tests in headless mode")
    parser.add_argument("--small-models", action="store_true",
                      help="Use smaller model variants when available")
    parser.add_argument("--generate-report", action="store_true",
                      help="Generate a report from test results")
    parser.add_argument("--results-file",
                      help="Path to the results file for report generation")
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug logging")
    args = parser.parse_args()
    
    # Create test runner
    tester = WebPlatformTestRunner(
        output_dir=args.output_dir,
        use_small_models=args.small_models,
        debug=args.debug
    )
    
    # Check for available browsers
    if not tester.available_browsers:
        logger.error("No supported browsers detected. Please install Chrome or Edge.")
        return 1
    
    # Validate browser selection against platform
    if args.browser:
        if args.platform == "webnn" and args.browser != "edge":
            logger.error("WebNN tests require Edge browser")
            return 1
        elif args.platform == "webgpu" and args.browser != "chrome":
            logger.error("WebGPU tests require Chrome browser")
            return 1
    
    # Run tests if model specified
    if args.model:
        if args.model == "all-key-models":
            # Run all models
            tester.run_all_models(args.platform, args.headless)
        else:
            # Run single model
            tester.run_model_test(args.model, args.platform, args.headless)
    
    # Generate report if requested
    if args.generate_report:
        report_file = tester.generate_test_report(args.results_file, args.platform)
        if report_file:
            logger.info(f"Report generated: {report_file}")
        else:
            logger.error("Failed to generate report")
            return 1
    
    # If no model or report was requested, print help
    if not args.model and not args.generate_report:
        parser.print_help()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())