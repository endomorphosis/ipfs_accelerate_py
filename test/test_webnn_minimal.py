#!/usr/bin/env python3
"""
Minimal WebNN and WebGPU test for checking browser support and basic performance.

This script provides a focused, minimal test of WebNN and WebGPU support in browsers,
implementing the core functionality needed to verify hardware acceleration capabilities.

Usage:
    python test_webnn_minimal.py --browser chrome
    python test_webnn_minimal.py --browser edge --model prajjwal1/bert-tiny
"""

import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Minimal WebNN and WebGPU test")
    parser.add_argument("--browser", type=str, default="edge",
                        choices=["chrome", "edge", "firefox", "safari"],
                        help="Browser to test")
    parser.add_argument("--model", type=str, default="prajjwal1/bert-tiny",
                        help="Model to test")
    parser.add_argument("--output", type=str, default="webnn_minimal_results.json",
                        help="Output file for results")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Timeout in seconds")
    parser.add_argument("--webgpu-only", action="store_true",
                        help="Only test WebGPU (skip WebNN)")
    return parser.parse_args()

def check_browser_capabilities(browser: str, timeout: int = 300) -> Dict:
    """Check browser WebNN and WebGPU capabilities.
    
    Args:
        browser: Browser to test.
        timeout: Timeout in seconds.
        
    Returns:
        Dictionary with browser capabilities.
    """
    print(f"Checking {browser} capabilities...")
    
    # Set browser-specific environment variables to enable WebNN/WebGPU
    browser_flags = ""
    if browser == "chrome":
        browser_flags = "--enable-features=WebML,WebNN,WebNNDMLCompute --disable-web-security --enable-dawn-features=allow_unsafe_apis --enable-webgpu-developer-features --ignore-gpu-blocklist"
    elif browser == "edge":
        browser_flags = "--enable-features=WebML,WebNN,WebNNDMLCompute --disable-web-security --enable-dawn-features=allow_unsafe_apis --enable-webgpu-developer-features --ignore-gpu-blocklist"
    elif browser == "firefox":
        browser_flags = "--MOZ_WEBGPU_FEATURES=dawn --MOZ_ENABLE_WEBGPU=1"
    
    # Create a minimal HTML file to check capabilities
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>WebNN and WebGPU Capability Check</title>
    </head>
    <body>
        <h1>WebNN and WebGPU Capability Check</h1>
        <div id="results"></div>
        
        <script>
            async function checkCapabilities() {
                const results = {
                    webnn: {
                        supported: 'ml' in navigator,
                        backends: [],
                        error: null
                    },
                    webgpu: {
                        supported: 'gpu' in navigator,
                        adapter: null,
                        error: null
                    }
                };
                
                // Check WebNN
                if (results.webnn.supported) {
                    try {
                        // Try to create both CPU and GPU contexts
                        const cpuContext = await navigator.ml.createContext({devicePreference: 'cpu'});
                        results.webnn.backends.push('cpu');
                        
                        try {
                            const gpuContext = await navigator.ml.createContext({devicePreference: 'gpu'});
                            results.webnn.backends.push('gpu');
                        } catch (e) {
                            console.log('GPU WebNN not available:', e);
                        }
                    } catch (e) {
                        results.webnn.error = e.toString();
                    }
                }
                
                // Check WebGPU
                if (results.webgpu.supported) {
                    try {
                        const adapter = await navigator.gpu.requestAdapter();
                        if (adapter) {
                            const info = await adapter.requestAdapterInfo();
                            results.webgpu.adapter = {
                                vendor: info.vendor,
                                architecture: info.architecture,
                                device: info.device,
                                description: info.description
                            };
                        } else {
                            results.webgpu.error = 'No adapter available';
                        }
                    } catch (e) {
                        results.webgpu.error = e.toString();
                    }
                }
                
                // Update page with results
                document.getElementById('results').textContent = JSON.stringify(results, null, 2);
                
                // Send results back to Python
                console.log('CAPABILITY_RESULTS:' + JSON.stringify(results));
            }
            
            // Run the check
            checkCapabilities();
        </script>
    </body>
    </html>
    """
    
    # Write HTML to temporary file
    html_path = "webnn_capability_check.html"
    with open(html_path, "w") as f:
        f.write(html_content)
    
    # Prepare command to run browser
    if browser == "chrome":
        cmd = ["google-chrome", browser_flags, "--headless", "--dump-dom", html_path]
    elif browser == "edge":
        cmd = ["msedge", browser_flags, "--headless", "--dump-dom", html_path]
    elif browser == "firefox":
        cmd = ["firefox", browser_flags, "--headless", "--dump-dom", html_path]
    else:
        return {"error": f"Unsupported browser: {browser}"}
    
    try:
        # Run the command and capture output
        process = subprocess.Popen(
            " ".join(cmd),
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for process to complete with timeout
        stdout, stderr = process.communicate(timeout=timeout)
        
        # Parse capability results from console output
        capabilities = {"browser": browser, "webnn_available": False, "webgpu_available": False}
        
        # Look for capability results in the output
        for line in stdout.split('\n'):
            if "CAPABILITY_RESULTS:" in line:
                results_str = line.split("CAPABILITY_RESULTS:")[1].strip()
                try:
                    results = json.loads(results_str)
                    capabilities["webnn_available"] = results["webnn"]["supported"] and not results["webnn"]["error"]
                    capabilities["webgpu_available"] = results["webgpu"]["supported"] and not results["webgpu"]["error"]
                    
                    if results["webnn"]["backends"]:
                        capabilities["webnn_backends"] = results["webnn"]["backends"]
                    
                    if results["webgpu"]["adapter"]:
                        capabilities["webgpu_adapter"] = results["webgpu"]["adapter"]
                        
                except json.JSONDecodeError:
                    capabilities["error"] = "Failed to parse capability results"
        
        return capabilities
    
    except subprocess.TimeoutExpired:
        return {"browser": browser, "error": f"Timeout after {timeout}s"}
    except Exception as e:
        return {"browser": browser, "error": str(e)}
    finally:
        # Clean up temporary file
        if os.path.exists(html_path):
            os.remove(html_path)

def run_webnn_benchmark(browser: str, model: str, timeout: int = 300) -> Dict:
    """Run a WebNN benchmark.
    
    Args:
        browser: Browser to test.
        model: Model to test.
        timeout: Timeout in seconds.
        
    Returns:
        Dictionary with benchmark results.
    """
    print(f"Running WebNN benchmark for {model} on {browser}...")
    
    # Set browser-specific environment variables to enable WebNN
    browser_flags = ""
    if browser == "chrome":
        browser_flags = "--enable-features=WebML,WebNN,WebNNDMLCompute --disable-web-security --enable-dawn-features=allow_unsafe_apis --enable-webgpu-developer-features --ignore-gpu-blocklist"
    elif browser == "edge":
        browser_flags = "--enable-features=WebML,WebNN,WebNNDMLCompute --disable-web-security --enable-dawn-features=allow_unsafe_apis --enable-webgpu-developer-features --ignore-gpu-blocklist"
    elif browser == "firefox":
        browser_flags = "--MOZ_WEBGPU_FEATURES=dawn --MOZ_ENABLE_WEBGPU=1"
    
    # Basic WebNN benchmark
    cmd = f"python3 test_webnn_benchmark.py --browser {browser} --model {model} --iterations 5 --batch-size 1 --output webnn_benchmark_{browser}_{model.replace('/', '_')}.json"
    
    try:
        # Run the command
        process = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        # Check for success
        if process.returncode == 0:
            # Try to load the output file
            output_file = f"webnn_benchmark_{browser}_{model.replace('/', '_')}.json"
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    results = json.load(f)
                return results
            else:
                return {"browser": browser, "model": model, "error": "Output file not found", "stdout": process.stdout, "stderr": process.stderr}
        else:
            return {"browser": browser, "model": model, "error": "Benchmark failed", "stdout": process.stdout, "stderr": process.stderr}
    
    except subprocess.TimeoutExpired:
        return {"browser": browser, "model": model, "error": f"Timeout after {timeout}s"}
    except Exception as e:
        return {"browser": browser, "model": model, "error": str(e)}

def run_webgpu_benchmark(browser: str, model: str, timeout: int = 300) -> Dict:
    """Run a WebGPU benchmark.
    
    Args:
        browser: Browser to test.
        model: Model to test.
        timeout: Timeout in seconds.
        
    Returns:
        Dictionary with benchmark results.
    """
    print(f"Running WebGPU benchmark for {model} on {browser}...")
    
    # For WebGPU, we use the web platform test runner
    cmd = f"python3 test_web_platform_optimizations.py --model {model} --browser {browser} --platform webgpu --output webgpu_benchmark_{browser}_{model.replace('/', '_')}.json"
    
    try:
        # Run the command
        process = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        # Parse results from output
        results = {"browser": browser, "model": model}
        
        # Look for performance metrics in stdout
        for line in process.stdout.split('\n'):
            if "Inference Time:" in line:
                try:
                    results["inference_time_ms"] = float(line.split("Inference Time:")[1].strip().split()[0])
                except (ValueError, IndexError):
                    pass
            
            if "Loading Time:" in line:
                try:
                    results["loading_time_ms"] = float(line.split("Loading Time:")[1].strip().split()[0])
                except (ValueError, IndexError):
                    pass
            
            if "Memory Usage:" in line:
                try:
                    results["memory_usage_mb"] = float(line.split("Memory Usage:")[1].strip().split()[0])
                except (ValueError, IndexError):
                    pass
        
        # Check if we got any results
        if "inference_time_ms" not in results:
            results["error"] = "No performance metrics found in output"
            results["stdout"] = process.stdout
            results["stderr"] = process.stderr
        
        return results
    
    except subprocess.TimeoutExpired:
        return {"browser": browser, "model": model, "error": f"Timeout after {timeout}s"}
    except Exception as e:
        return {"browser": browser, "model": model, "error": str(e)}

def main():
    """Main function."""
    args = parse_args()
    
    # Results dictionary
    results = {
        "timestamp": time.time(),
        "browser": args.browser,
        "model": args.model,
        "capabilities": None,
        "webnn_benchmark": None,
        "webgpu_benchmark": None
    }
    
    # Check browser capabilities
    print(f"Testing {args.browser} WebNN and WebGPU support...")
    results["capabilities"] = check_browser_capabilities(args.browser, args.timeout)
    
    # Run WebNN benchmark if supported
    if not args.webgpu_only and results["capabilities"].get("webnn_available", False):
        print(f"Running WebNN benchmark for {args.model}...")
        results["webnn_benchmark"] = run_webnn_benchmark(args.browser, args.model, args.timeout)
    
    # Run WebGPU benchmark if supported
    if results["capabilities"].get("webgpu_available", False):
        print(f"Running WebGPU benchmark for {args.model}...")
        results["webgpu_benchmark"] = run_webgpu_benchmark(args.browser, args.model, args.timeout)
    
    # Save results to file
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {args.output}")
    
    # Print summary
    print("\nSummary:")
    print(f"Browser: {args.browser}")
    print(f"WebNN Available: {results['capabilities'].get('webnn_available', False)}")
    print(f"WebGPU Available: {results['capabilities'].get('webgpu_available', False)}")
    
    if results.get("webnn_benchmark") and "error" not in results["webnn_benchmark"]:
        print(f"WebNN Benchmark: Success")
        if "cpu_time_ms" in results["webnn_benchmark"] and "webnn_time_ms" in results["webnn_benchmark"]:
            speedup = results["webnn_benchmark"]["cpu_time_ms"] / results["webnn_benchmark"]["webnn_time_ms"]
            print(f"WebNN Speedup: {speedup:.2f}x")
    elif results.get("webnn_benchmark"):
        print(f"WebNN Benchmark: Failed - {results['webnn_benchmark'].get('error', 'Unknown error')}")
    
    if results.get("webgpu_benchmark") and "error" not in results["webgpu_benchmark"]:
        print(f"WebGPU Benchmark: Success")
        if "inference_time_ms" in results["webgpu_benchmark"]:
            print(f"WebGPU Inference Time: {results['webgpu_benchmark']['inference_time_ms']:.2f} ms")
    elif results.get("webgpu_benchmark"):
        print(f"WebGPU Benchmark: Failed - {results['webgpu_benchmark'].get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()