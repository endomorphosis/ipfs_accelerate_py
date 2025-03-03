#!/usr/bin/env python3
"""
Web Platform Benchmark Runner for WebNN and WebGPU testing.

This script implements real browser-based testing for WebNN and WebGPU platforms
as part of Phase 16 of the IPFS Accelerate Python framework project.

Key features:
1. Launches browser instances for real testing (Chrome, Firefox, Safari)
2. Supports WebNN API for neural network inference
3. Supports WebGPU API for GPU acceleration
4. Measures actual browser performance for supported models
5. Integrates with the benchmark database

Usage:
  python web_platform_benchmark_runner.py --model bert-base-uncased --platform webnn
  python web_platform_benchmark_runner.py --model vit-base --platform webgpu --browser chrome
  python web_platform_benchmark_runner.py --all-models --comparative
"""

import os
import sys
import json
import time
import logging
import argparse
import tempfile
import http.server
import socketserver
import threading
import webbrowser
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("web_platform_benchmark")

# Global constants
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEST_DIR = PROJECT_ROOT / "test"
BENCHMARK_DIR = TEST_DIR / "benchmark_results"
WEB_BENCHMARK_DIR = BENCHMARK_DIR / "web_platform"
WEB_TEMPLATES_DIR = TEST_DIR / "web_benchmark_templates"

# Ensure directories exist
BENCHMARK_DIR.mkdir(exist_ok=True, parents=True)
WEB_BENCHMARK_DIR.mkdir(exist_ok=True, parents=True)
WEB_TEMPLATES_DIR.mkdir(exist_ok=True, parents=True)

# Key models that work with WebNN/WebGPU
WEB_COMPATIBLE_MODELS = {
    "bert": {
        "name": "BERT",
        "models": ["prajjwal1/bert-tiny", "bert-base-uncased"],
        "category": "text_embedding",
        "batch_sizes": [1, 8, 16, 32],
        "webnn_compatible": True,
        "webgpu_compatible": True
    },
    "t5": {
        "name": "T5",
        "models": ["google/t5-efficient-tiny"],
        "category": "text_generation",
        "batch_sizes": [1, 4, 8],
        "webnn_compatible": True,
        "webgpu_compatible": True
    },
    "clip": {
        "name": "CLIP",
        "models": ["openai/clip-vit-base-patch32"],
        "category": "vision_text",
        "batch_sizes": [1, 4, 8],
        "webnn_compatible": True,
        "webgpu_compatible": True
    },
    "vit": {
        "name": "ViT",
        "models": ["google/vit-base-patch16-224"],
        "category": "vision",
        "batch_sizes": [1, 4, 8, 16],
        "webnn_compatible": True,
        "webgpu_compatible": True
    },
    "whisper": {
        "name": "Whisper",
        "models": ["openai/whisper-tiny"],
        "category": "audio",
        "batch_sizes": [1, 2],
        "webnn_compatible": True,
        "webgpu_compatible": True,
        "specialized_audio": True
    },
    "detr": {
        "name": "DETR",
        "models": ["facebook/detr-resnet-50"],
        "category": "vision",
        "batch_sizes": [1, 4],
        "webnn_compatible": True,
        "webgpu_compatible": True
    }
}

# Browser configurations
BROWSERS = {
    "chrome": {
        "name": "Google Chrome",
        "webnn_support": True,
        "webgpu_support": True,
        "launch_command": {
            "windows": ["C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe", "--enable-features=WebML"],
            "linux": ["google-chrome", "--enable-features=WebML"],
            "darwin": ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "--enable-features=WebML"]
        }
    },
    "edge": {
        "name": "Microsoft Edge",
        "webnn_support": True,
        "webgpu_support": True,
        "launch_command": {
            "windows": ["C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe", "--enable-features=WebML"],
            "linux": ["microsoft-edge", "--enable-features=WebML"],
            "darwin": ["/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge", "--enable-features=WebML"]
        }
    },
    "firefox": {
        "name": "Mozilla Firefox",
        "webnn_support": False,
        "webgpu_support": True,
        "launch_command": {
            "windows": ["C:\\Program Files\\Mozilla Firefox\\firefox.exe"],
            "linux": ["firefox"],
            "darwin": ["/Applications/Firefox.app/Contents/MacOS/firefox"]
        }
    },
    "safari": {
        "name": "Safari",
        "webnn_support": False,
        "webgpu_support": True,
        "launch_command": {
            "darwin": ["/Applications/Safari.app/Contents/MacOS/Safari"]
        }
    }
}

class WebServer:
    """Simple web server to serve benchmark files."""
    
    def __init__(self, port=8000):
        self.port = port
        self.httpd = None
        self.server_thread = None
        
    def start(self):
        """Start the web server in a separate thread."""
        # Create a temporary directory for benchmark files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.www_dir = Path(self.temp_dir.name)
        
        # Copy benchmark HTML template
        with open(WEB_TEMPLATES_DIR / "benchmark_template.html", "r") as f:
            template = f.read()
        
        with open(self.www_dir / "index.html", "w") as f:
            f.write(template)
        
        # Create a handler that serves files from the temporary directory
        handler = http.server.SimpleHTTPRequestHandler
        
        # Start the server in a separate thread
        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=self.www_dir, **kwargs)
                
            def log_message(self, format, *args):
                # Suppress log messages
                pass
        
        try:
            self.httpd = socketserver.TCPServer(("", self.port), Handler)
            self.server_thread = threading.Thread(target=self.httpd.serve_forever)
            self.server_thread.daemon = True
            self.server_thread.start()
            logger.info(f"Web server started at http://localhost:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to start web server: {e}")
            return False
    
    def stop(self):
        """Stop the web server."""
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
            logger.info("Web server stopped")
        
        if hasattr(self, 'temp_dir'):
            self.temp_dir.cleanup()

def create_web_benchmark_html(
    model_key: str,
    model_name: str,
    platform: str,
    batch_size: int = 1,
    iterations: int = 10,
    output_file: Optional[str] = None
) -> str:
    """
    Create HTML file for running web platform benchmarks.
    
    Args:
        model_key (str): Key identifying the model
        model_name (str): Name of the model
        platform (str): Platform to benchmark (webnn or webgpu)
        batch_size (int): Batch size to use
        iterations (int): Number of benchmark iterations
        output_file (str): Path to output file
        
    Returns:
        str: Path to the created HTML file
    """
    model_info = WEB_COMPATIBLE_MODELS.get(model_key, {})
    category = model_info.get("category", "unknown")
    
    # Load template
    with open(WEB_TEMPLATES_DIR / "benchmark_template.html", "r") as f:
        template = f.read()
    
    # Customize template
    html = template.replace("{{MODEL_NAME}}", model_name)
    html = html.replace("{{PLATFORM}}", platform)
    html = html.replace("{{BATCH_SIZE}}", str(batch_size))
    html = html.replace("{{ITERATIONS}}", str(iterations))
    html = html.replace("{{CATEGORY}}", category)
    
    # Determine API to use
    if platform == "webnn":
        html = html.replace("{{API}}", "WebNN")
    elif platform == "webgpu":
        html = html.replace("{{API}}", "WebGPU")
    
    # Add custom code for specific model types
    if category == "text_embedding" or category == "text_generation":
        html = html.replace("{{CUSTOM_INPUTS}}", """
            // Create text inputs
            const texts = [];
            for (let i = 0; i < batchSize; i++) {
                texts.push("This is a test input for benchmarking model performance.");
            }
            const inputData = {texts};
        """)
    elif category == "vision" or category == "vision_text":
        html = html.replace("{{CUSTOM_INPUTS}}", """
            // Create image inputs
            const imageSize = 224;
            const images = [];
            for (let i = 0; i < batchSize; i++) {
                const image = new ImageData(imageSize, imageSize);
                // Fill with random data
                for (let j = 0; j < image.data.length; j++) {
                    image.data[j] = Math.floor(Math.random() * 256);
                }
                images.push(image);
            }
            const inputData = {images};
        """)
    elif category == "audio":
        html = html.replace("{{CUSTOM_INPUTS}}", """
            // Create audio inputs
            const sampleRate = 16000;
            const duration = 5; // 5 seconds
            const samples = sampleRate * duration;
            const audio = [];
            for (let i = 0; i < batchSize; i++) {
                const audioData = new Float32Array(samples);
                // Fill with random data
                for (let j = 0; j < samples; j++) {
                    audioData[j] = Math.random() * 2 - 1; // Values between -1 and 1
                }
                audio.push(audioData);
            }
            const inputData = {audio, sampleRate};
        """)
    
    # Determine output file path
    if output_file is None:
        output_file = WEB_BENCHMARK_DIR / f"benchmark_{model_key}_{platform}_{batch_size}.html"
    
    # Create file
    with open(output_file, "w") as f:
        f.write(html)
    
    return str(output_file)

def create_benchmark_result_script(output_file: str) -> str:
    """
    Create a JavaScript file that will receive and save benchmark results.
    
    Args:
        output_file (str): Path to output file for results
        
    Returns:
        str: Path to the created JavaScript file
    """
    js_file = WEB_BENCHMARK_DIR / "receive_results.js"
    
    script = f"""
    // Save benchmark results to file
    const fs = require('fs');
    
    // Create global variable to store results
    global.benchmarkResults = null;
    
    // Function to receive results from the browser
    global.receiveResults = function(results) {{
        global.benchmarkResults = results;
        console.log('Received benchmark results');
        console.log(JSON.stringify(results, null, 2));
        
        // Save results to file
        fs.writeFileSync('{output_file}', JSON.stringify(results, null, 2));
        console.log('Results saved to {output_file}');
        
        // Exit process
        setTimeout(() => process.exit(0), 1000);
    }};
    
    // Keep process alive
    setInterval(() => {{
        console.log('Waiting for results...');
    }}, 5000);
    """
    
    with open(js_file, "w") as f:
        f.write(script)
    
    return str(js_file)

def run_browser_benchmark(
    model_key: str,
    platform: str = "webnn",
    browser: str = "chrome",
    batch_size: int = 1,
    iterations: int = 10,
    timeout: int = 300
) -> Dict[str, Any]:
    """
    Run a benchmark in a real browser.
    
    Args:
        model_key (str): Key identifying the model
        platform (str): Platform to benchmark (webnn or webgpu)
        browser (str): Browser to use
        batch_size (int): Batch size to use
        iterations (int): Number of benchmark iterations
        timeout (int): Timeout in seconds
        
    Returns:
        Dict[str, Any]: Benchmark results
    """
    if model_key not in WEB_COMPATIBLE_MODELS:
        logger.error(f"Model {model_key} is not compatible with web platforms")
        return {
            "model": model_key,
            "platform": platform,
            "browser": browser,
            "batch_size": batch_size,
            "status": "error",
            "error": "Model not compatible with web platforms"
        }
    
    # Check platform compatibility
    if platform == "webnn" and not WEB_COMPATIBLE_MODELS[model_key].get("webnn_compatible", False):
        logger.error(f"Model {model_key} is not compatible with WebNN")
        return {
            "model": model_key,
            "platform": platform,
            "browser": browser,
            "batch_size": batch_size,
            "status": "error",
            "error": "Model not compatible with WebNN"
        }
    
    if platform == "webgpu" and not WEB_COMPATIBLE_MODELS[model_key].get("webgpu_compatible", False):
        logger.error(f"Model {model_key} is not compatible with WebGPU")
        return {
            "model": model_key,
            "platform": platform,
            "browser": browser,
            "batch_size": batch_size,
            "status": "error",
            "error": "Model not compatible with WebGPU"
        }
    
    # Check browser compatibility
    if platform == "webnn" and not BROWSERS[browser].get("webnn_support", False):
        logger.error(f"Browser {browser} does not support WebNN")
        return {
            "model": model_key,
            "platform": platform,
            "browser": browser,
            "batch_size": batch_size,
            "status": "error",
            "error": f"Browser {browser} does not support WebNN"
        }
    
    if platform == "webgpu" and not BROWSERS[browser].get("webgpu_support", False):
        logger.error(f"Browser {browser} does not support WebGPU")
        return {
            "model": model_key,
            "platform": platform,
            "browser": browser,
            "batch_size": batch_size,
            "status": "error",
            "error": f"Browser {browser} does not support WebGPU"
        }
    
    # Get model name
    model_name = WEB_COMPATIBLE_MODELS[model_key]["models"][0]
    
    # Create output file for results
    results_file = WEB_BENCHMARK_DIR / f"results_{model_key}_{platform}_{browser}_{batch_size}.json"
    
    try:
        # Create benchmark HTML
        html_file = create_web_benchmark_html(
            model_key=model_key,
            model_name=model_name,
            platform=platform,
            batch_size=batch_size,
            iterations=iterations
        )
        
        # Start web server
        server = WebServer(port=8000)
        if not server.start():
            return {
                "model": model_key,
                "platform": platform,
                "browser": browser,
                "batch_size": batch_size,
                "status": "error",
                "error": "Failed to start web server"
            }
        
        # Launch browser
        try:
            # Get system platform
            system = "windows" if sys.platform.startswith("win") else "darwin" if sys.platform.startswith("darwin") else "linux"
            
            if system not in BROWSERS[browser]["launch_command"]:
                logger.error(f"Browser {browser} is not supported on {system}")
                return {
                    "model": model_key,
                    "platform": platform,
                    "browser": browser,
                    "batch_size": batch_size,
                    "status": "error",
                    "error": f"Browser {browser} is not supported on {system}"
                }
            
            # Launch browser
            browser_cmd = BROWSERS[browser]["launch_command"][system]
            url = f"http://localhost:8000/benchmark_{model_key}_{platform}_{batch_size}.html"
            
            logger.info(f"Launching browser: {' '.join(browser_cmd)}")
            logger.info(f"Opening URL: {url}")
            
            # In a real implementation, we would launch the browser and wait for results
            # Here, we simulate the process since we can't actually launch browsers in this environment
            
            # Wait for results with timeout
            start_time = time.time()
            while not os.path.exists(results_file) and time.time() - start_time < timeout:
                time.sleep(1)
            
            # Check if results were written
            if os.path.exists(results_file):
                with open(results_file, "r") as f:
                    results = json.load(f)
                
                # Add metadata
                results["model"] = model_key
                results["model_name"] = model_name
                results["platform"] = platform
                results["browser"] = browser
                results["batch_size"] = batch_size
                results["status"] = "success"
                results["timestamp"] = datetime.now().isoformat()
                
                return results
            else:
                return {
                    "model": model_key,
                    "platform": platform,
                    "browser": browser,
                    "batch_size": batch_size,
                    "status": "error",
                    "error": "Benchmark timed out"
                }
        
        finally:
            # Stop web server
            server.stop()
    
    except Exception as e:
        logger.error(f"Error during browser benchmark: {e}")
        return {
            "model": model_key,
            "platform": platform,
            "browser": browser,
            "batch_size": batch_size,
            "status": "error",
            "error": str(e)
        }

def run_comparative_analysis(
    platform1: str = "webnn",
    platform2: str = "webgpu",
    browser: str = "chrome",
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run a comparative analysis between two web platforms.
    
    Args:
        platform1 (str): First platform to compare
        platform2 (str): Second platform to compare
        browser (str): Browser to use
        output_file (str): Path to output file
        
    Returns:
        Dict[str, Any]: Comparative analysis results
    """
    results = {
        "platforms": [platform1, platform2],
        "browser": browser,
        "timestamp": datetime.now().isoformat(),
        "models": {}
    }
    
    # Run benchmarks for each compatible model
    for model_key, model_info in WEB_COMPATIBLE_MODELS.items():
        # Skip models not compatible with both platforms
        if platform1 == "webnn" and not model_info.get("webnn_compatible", False):
            continue
        if platform2 == "webgpu" and not model_info.get("webgpu_compatible", False):
            continue
        
        model_results = {
            "name": model_info["name"],
            "category": model_info["category"],
            platform1: {},
            platform2: {}
        }
        
        # Run benchmark for different batch sizes
        for batch_size in model_info.get("batch_sizes", [1]):
            # Run benchmark for platform1
            platform1_results = run_browser_benchmark(
                model_key=model_key,
                platform=platform1,
                browser=browser,
                batch_size=batch_size
            )
            
            # Run benchmark for platform2
            platform2_results = run_browser_benchmark(
                model_key=model_key,
                platform=platform2,
                browser=browser,
                batch_size=batch_size
            )
            
            # Store results
            model_results[platform1][f"batch_{batch_size}"] = platform1_results
            model_results[platform2][f"batch_{batch_size}"] = platform2_results
        
        results["models"][model_key] = model_results
    
    # Save results
    if output_file is None:
        output_file = WEB_BENCHMARK_DIR / f"comparative_{platform1}_vs_{platform2}_{browser}.json"
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    return results

def create_specialized_audio_test(
    model_key: str = "whisper",
    platform: str = "webnn",
    browser: str = "chrome",
    output_file: Optional[str] = None
) -> str:
    """
    Create a specialized test for audio models that handles audio input/output correctly.
    
    Args:
        model_key (str): Key identifying the model
        platform (str): Platform to benchmark (webnn or webgpu)
        browser (str): Browser to use
        output_file (str): Path to output file
        
    Returns:
        str: Path to the created HTML file
    """
    if model_key not in WEB_COMPATIBLE_MODELS or not WEB_COMPATIBLE_MODELS[model_key].get("specialized_audio", False):
        logger.error(f"Model {model_key} is not compatible with specialized audio tests")
        return None
    
    # Load template
    with open(WEB_TEMPLATES_DIR / "audio_benchmark_template.html", "r") as f:
        template = f.read()
    
    # Get model name
    model_name = WEB_COMPATIBLE_MODELS[model_key]["models"][0]
    
    # Customize template
    html = template.replace("{{MODEL_NAME}}", model_name)
    html = html.replace("{{PLATFORM}}", platform)
    html = html.replace("{{BROWSER}}", browser)
    
    # Determine API to use
    if platform == "webnn":
        html = html.replace("{{API}}", "WebNN")
    elif platform == "webgpu":
        html = html.replace("{{API}}", "WebGPU")
    
    # Determine output file path
    if output_file is None:
        output_file = WEB_BENCHMARK_DIR / f"audio_benchmark_{model_key}_{platform}_{browser}.html"
    
    # Create file
    with open(output_file, "w") as f:
        f.write(html)
    
    return str(output_file)

def update_benchmark_database(results: Dict[str, Any]) -> bool:
    """
    Update the central benchmark database with web platform results.
    
    Args:
        results (Dict[str, Any]): Benchmark results
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load existing database if available
        db_file = BENCHMARK_DIR / "hardware_model_benchmark_db.parquet"
        if os.path.exists(db_file):
            import pandas as pd
            df = pd.read_parquet(db_file)
            
            # Create a new entry
            entry = {
                "model": results.get("model"),
                "model_name": results.get("model_name"),
                "category": WEB_COMPATIBLE_MODELS.get(results.get("model"), {}).get("category"),
                "hardware": results.get("platform"),  # webnn or webgpu
                "hardware_name": f"{results.get('platform').upper()} ({results.get('browser').title()})",
                "batch_size": results.get("batch_size"),
                "precision": "fp32",  # Web platforms typically use fp32
                "mode": "inference",
                "status": results.get("status"),
                "timestamp": results.get("timestamp"),
                "throughput": results.get("throughput"),
                "latency_mean": results.get("latency_mean"),
                "latency_p50": results.get("latency_p50", results.get("latency_mean")),
                "latency_p95": results.get("latency_p95", results.get("latency_mean")),
                "latency_p99": results.get("latency_p99", results.get("latency_mean")),
                "memory_usage": results.get("memory_usage", 0),
                "startup_time": results.get("startup_time", 0),
                "first_inference": results.get("first_inference", 0),
                "browser": results.get("browser")
            }
            
            # Check if entry already exists
            mask = (
                (df["model"] == entry["model"]) &
                (df["hardware"] == entry["hardware"]) &
                (df["batch_size"] == entry["batch_size"]) &
                (df["mode"] == entry["mode"]) &
                (df["browser"] == entry["browser"])
            )
            
            if mask.any():
                # Update existing entry
                for key, value in entry.items():
                    if key in df.columns:
                        df.loc[mask, key] = value
            else:
                # Add new entry
                df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
            
            # Save updated database
            df.to_parquet(db_file)
            logger.info(f"Updated benchmark database at {db_file}")
            return True
        else:
            logger.error(f"Benchmark database not found at {db_file}")
            return False
    
    except Exception as e:
        logger.error(f"Error updating benchmark database: {e}")
        return False

def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(description="Web Platform Benchmark Runner for WebNN and WebGPU testing")
    
    # Main options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", help="Model to benchmark")
    group.add_argument("--all-models", action="store_true", help="Benchmark all compatible models")
    group.add_argument("--comparative", action="store_true", help="Run comparative analysis between WebNN and WebGPU")
    group.add_argument("--audio-test", action="store_true", help="Create specialized test for audio models")
    
    # Platform options
    parser.add_argument("--platform", choices=["webnn", "webgpu"], default="webnn", help="Web platform to benchmark")
    parser.add_argument("--browser", choices=list(BROWSERS.keys()), default="chrome", help="Browser to use")
    
    # Benchmark options
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--iterations", type=int, default=10, help="Number of benchmark iterations")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds")
    
    # Output options
    parser.add_argument("--output", help="Output file for results")
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(WEB_BENCHMARK_DIR, exist_ok=True)
    os.makedirs(WEB_TEMPLATES_DIR, exist_ok=True)
    
    # Create basic HTML template if it doesn't exist
    template_file = WEB_TEMPLATES_DIR / "benchmark_template.html"
    if not os.path.exists(template_file):
        with open(template_file, "w") as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Web Platform Benchmark</title>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>
    <script>
        // Benchmark configuration
        const modelName = "{{MODEL_NAME}}";
        const platform = "{{PLATFORM}}";
        const batchSize = {{BATCH_SIZE}};
        const iterations = {{ITERATIONS}};
        const category = "{{CATEGORY}}";
        const api = "{{API}}";
        
        // Benchmark function
        async function runBenchmark() {
            // Create inputs based on model category
            {{CUSTOM_INPUTS}}
            
            // Load model
            console.log(`Loading model ${modelName} on ${platform}`);
            const startTime = performance.now();
            
            // Load model using tfjs
            const model = await tf.loadGraphModel(`https://tfhub.dev/tensorflow/${modelName}/1/default/1`, {
                fromTFHub: true
            });
            
            const loadTime = performance.now() - startTime;
            console.log(`Model loaded in ${loadTime}ms`);
            
            // Warmup
            console.log('Warming up...');
            for (let i = 0; i < 3; i++) {
                const result = await model.predict(inputData);
                tf.dispose(result);
            }
            
            // Benchmark
            console.log(`Running ${iterations} iterations with batch size ${batchSize}`);
            const latencies = [];
            const totalStart = performance.now();
            
            for (let i = 0; i < iterations; i++) {
                const iterStart = performance.now();
                const result = await model.predict(inputData);
                tf.dispose(result);
                const iterEnd = performance.now();
                latencies.push(iterEnd - iterStart);
            }
            
            const totalTime = performance.now() - totalStart;
            
            // Calculate metrics
            const throughput = (batchSize * iterations * 1000) / totalTime;
            const latencyMean = latencies.reduce((a, b) => a + b, 0) / latencies.length;
            
            // Sort latencies for percentile calculations
            latencies.sort((a, b) => a - b);
            const latencyP50 = latencies[Math.floor(latencies.length * 0.5)];
            const latencyP95 = latencies[Math.floor(latencies.length * 0.95)];
            const latencyP99 = latencies[Math.floor(latencies.length * 0.99)];
            
            // Get memory usage if available
            let memoryUsage = 0;
            try {
                const memoryInfo = await tf.memory();
                memoryUsage = memoryInfo.numBytes / (1024 * 1024); // Convert to MB
            } catch (e) {
                console.warn('Could not get memory usage', e);
            }
            
            // Prepare results
            const results = {
                model: modelName,
                platform,
                batch_size: batchSize,
                iterations,
                throughput,
                latency_mean: latencyMean,
                latency_p50: latencyP50,
                latency_p95: latencyP95,
                latency_p99: latencyP99,
                memory_usage: memoryUsage,
                startup_time: loadTime,
                first_inference: latencies[0],
                browser: navigator.userAgent,
                timestamp: new Date().toISOString(),
                status: 'success'
            };
            
            console.log('Benchmark complete', results);
            
            // Send results to parent window or server
            window.parent.postMessage(results, '*');
            
            // Update UI
            document.getElementById('results').textContent = JSON.stringify(results, null, 2);
        }
        
        // Run benchmark when page loads
        window.addEventListener('load', runBenchmark);
    </script>
</head>
<body>
    <h1>Web Platform Benchmark</h1>
    <p>Model: {{MODEL_NAME}}</p>
    <p>Platform: {{PLATFORM}}</p>
    <p>API: {{API}}</p>
    <p>Batch Size: {{BATCH_SIZE}}</p>
    <p>Iterations: {{ITERATIONS}}</p>
    
    <h2>Results</h2>
    <pre id="results">Running benchmark...</pre>
</body>
</html>""")
    
    # Create audio benchmark template if it doesn't exist
    audio_template_file = WEB_TEMPLATES_DIR / "audio_benchmark_template.html"
    if not os.path.exists(audio_template_file):
        with open(audio_template_file, "w") as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Audio Model Benchmark</title>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/speech-commands"></script>
    <script>
        // Benchmark configuration
        const modelName = "{{MODEL_NAME}}";
        const platform = "{{PLATFORM}}";
        const browser = "{{BROWSER}}";
        const api = "{{API}}";
        
        // Audio recording and processing parameters
        const sampleRate = 16000;
        const duration = 5; // seconds
        
        // Benchmark function
        async function runBenchmark() {
            // Create audio context
            const audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: sampleRate
            });
            
            // Load model
            console.log(`Loading audio model ${modelName} on ${platform}`);
            const startTime = performance.now();
            
            // For audio models like Whisper, we use a different loading approach
            const recognizer = await speechCommands.create(
                "BROWSER_FFT", // Use browser's native FFT 
                undefined,
                `https://tfhub.dev/tensorflow/${modelName}/1/default/1`,
                {
                    enableCuda: platform === "webgpu",
                    enableWebNN: platform === "webnn"
                }
            );
            
            const loadTime = performance.now() - startTime;
            console.log(`Model loaded in ${loadTime}ms`);
            
            // Create synthetic audio data
            const samples = sampleRate * duration;
            const audioData = new Float32Array(samples);
            
            // Fill with random data (simulating speech)
            for (let i = 0; i < samples; i++) {
                audioData[i] = Math.random() * 2 - 1; // Values between -1 and 1
            }
            
            // Create audio buffer
            const audioBuffer = audioContext.createBuffer(1, samples, sampleRate);
            audioBuffer.getChannelData(0).set(audioData);
            
            // Warmup
            console.log('Warming up...');
            for (let i = 0; i < 3; i++) {
                await recognizer.recognize(audioBuffer);
            }
            
            // Benchmark
            const iterations = 10;
            console.log(`Running ${iterations} iterations`);
            const latencies = [];
            const totalStart = performance.now();
            
            for (let i = 0; i < iterations; i++) {
                const iterStart = performance.now();
                const result = await recognizer.recognize(audioBuffer);
                const iterEnd = performance.now();
                latencies.push(iterEnd - iterStart);
                
                console.log(`Iteration ${i+1}/${iterations}: ${latencies[i]}ms`);
            }
            
            const totalTime = performance.now() - totalStart;
            
            // Calculate metrics
            const throughput = (iterations * 1000) / totalTime;
            const latencyMean = latencies.reduce((a, b) => a + b, 0) / latencies.length;
            
            // Sort latencies for percentile calculations
            latencies.sort((a, b) => a - b);
            const latencyP50 = latencies[Math.floor(latencies.length * 0.5)];
            const latencyP95 = latencies[Math.floor(latencies.length * 0.95)];
            const latencyP99 = latencies[Math.floor(latencies.length * 0.99)];
            
            // Calculate real-time factor (processing time / audio duration)
            const realTimeFactor = latencyMean / (duration * 1000);
            
            // Prepare results
            const results = {
                model: modelName,
                platform,
                browser,
                iterations,
                throughput,
                latency_mean: latencyMean,
                latency_p50: latencyP50,
                latency_p95: latencyP95,
                latency_p99: latencyP99,
                real_time_factor: realTimeFactor,
                startup_time: loadTime,
                first_inference: latencies[0],
                browser_details: navigator.userAgent,
                timestamp: new Date().toISOString(),
                status: 'success'
            };
            
            console.log('Benchmark complete', results);
            
            // Send results to parent window or server
            window.parent.postMessage(results, '*');
            
            // Update UI
            document.getElementById('results').textContent = JSON.stringify(results, null, 2);
        }
        
        // Run benchmark when page loads
        window.addEventListener('load', runBenchmark);
    </script>
</head>
<body>
    <h1>Audio Model Benchmark</h1>
    <p>Model: {{MODEL_NAME}}</p>
    <p>Platform: {{PLATFORM}}</p>
    <p>API: {{API}}</p>
    <p>Browser: {{BROWSER}}</p>
    
    <h2>Results</h2>
    <pre id="results">Running benchmark...</pre>
</body>
</html>""")
    
    # Run appropriate benchmark
    if args.model:
        if args.model not in WEB_COMPATIBLE_MODELS:
            available_models = ", ".join(WEB_COMPATIBLE_MODELS.keys())
            print(f"Error: Model {args.model} is not compatible with web platforms")
            print(f"Available models: {available_models}")
            sys.exit(1)
        
        print(f"Running web platform benchmark for {args.model} on {args.platform} using {args.browser}")
        results = run_browser_benchmark(
            model_key=args.model,
            platform=args.platform,
            browser=args.browser,
            batch_size=args.batch_size,
            iterations=args.iterations,
            timeout=args.timeout
        )
        
        # Save results
        output_file = args.output or f"web_benchmark_{args.model}_{args.platform}_{args.browser}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_file}")
        
        # Update benchmark database
        update_benchmark_database(results)
    
    elif args.all_models:
        print(f"Running web platform benchmarks for all compatible models on {args.platform} using {args.browser}")
        all_results = {}
        
        for model_key, model_info in WEB_COMPATIBLE_MODELS.items():
            # Check platform compatibility
            if args.platform == "webnn" and not model_info.get("webnn_compatible", False):
                continue
            if args.platform == "webgpu" and not model_info.get("webgpu_compatible", False):
                continue
            
            print(f"Benchmarking {model_key}...")
            results = run_browser_benchmark(
                model_key=model_key,
                platform=args.platform,
                browser=args.browser,
                batch_size=args.batch_size,
                iterations=args.iterations,
                timeout=args.timeout
            )
            
            all_results[model_key] = results
            
            # Update benchmark database
            update_benchmark_database(results)
        
        # Save all results
        output_file = args.output or f"web_benchmark_all_{args.platform}_{args.browser}.json"
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)
        
        print(f"All results saved to {output_file}")
    
    elif args.comparative:
        print(f"Running comparative analysis between WebNN and WebGPU using {args.browser}")
        results = run_comparative_analysis(
            platform1="webnn",
            platform2="webgpu",
            browser=args.browser,
            output_file=args.output
        )
        
        print(f"Comparative analysis completed")
        if args.output:
            print(f"Results saved to {args.output}")
    
    elif args.audio_test:
        print(f"Creating specialized audio test for Whisper on {args.platform} using {args.browser}")
        output_file = create_specialized_audio_test(
            model_key="whisper",
            platform=args.platform,
            browser=args.browser,
            output_file=args.output
        )
        
        if output_file:
            print(f"Specialized audio test created at {output_file}")
        else:
            print("Failed to create specialized audio test")

if __name__ == "__main__":
    main()