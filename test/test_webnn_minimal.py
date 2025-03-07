#!/usr/bin/env python3
"""
Minimal WebNN Test - Implements a simple WebNN based inference test in Chrome with Selenium

This file contains a minimal test implementation for real WebNN inference
with HuggingFace models in Chrome via Transformers.js. It uses Selenium
to automate the browser and doesn't require complex WebSocket bridges.

This is a simplified version that focuses only on WebNN functionality.

Usage:
    python test_webnn_minimal.py --model bert-base-uncased --browser chrome
    python test_webnn_minimal.py --model bert-base-uncased --browser edge --bits 8
    python test_webnn_minimal.py --model bert-base-uncased --browser chrome --bits 4 --mixed-precision
"""

import os
import sys
import json
import time
import tempfile
import logging
import argparse
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service as ChromeService
    from selenium.webdriver.edge.service import Service as EdgeService
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.edge.options import Options as EdgeOptions
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    HAS_SELENIUM = True
except ImportError:
    logger.error("selenium package is required. Install with: pip install selenium")
    HAS_SELENIUM = False

# Minimal HTML for WebNN testing
WEBNN_TEST_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WebNN Test</title>
    <style>
        body { font-family: Arial, sans-serif; padding: 20px; }
        #log { height: 300px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px; margin-bottom: 20px; }
        .success { color: green; font-weight: bold; }
        .error { color: red; font-weight: bold; }
        .warning { color: orange; font-weight: bold; }
        pre { background-color: #f5f5f5; padding: 10px; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>WebNN Inference Test</h1>
    <div id="status">
        <p>WebNN Support: <span id="webnn-support">Checking...</span></p>
        <p>Backend: <span id="webnn-backend">Unknown</span></p>
        <p>Precision: <span id="precision-level">Default</span></p>
    </div>
    <div id="log"></div>
    <div>
        <h2>Results:</h2>
        <pre id="results">Running test...</pre>
    </div>
    
    <script type="module">
        // Log function
        function log(message, className) {
            const logElem = document.getElementById('log');
            const entry = document.createElement('div');
            if (className) entry.className = className;
            entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
            logElem.appendChild(entry);
            logElem.scrollTop = logElem.scrollHeight;
            console.log(message);
        }
        
        // Check WebNN support
        async function checkWebNN() {
            const supportElem = document.getElementById('webnn-support');
            const backendElem = document.getElementById('webnn-backend');
            
            if (!('ml' in navigator)) {
                supportElem.textContent = 'Not Supported';
                supportElem.className = 'error';
                log('WebNN is not supported in this browser', 'error');
                return false;
            }
            
            try {
                // Try GPU backend first
                let gpuContext = null;
                try {
                    gpuContext = await navigator.ml.createContext({ devicePreference: 'gpu' });
                    if (gpuContext) {
                        log('WebNN GPU backend is available', 'success');
                    }
                } catch (e) {
                    log(`Failed to create WebNN GPU context: ${e.message}`);
                }
                
                // Fall back to CPU if GPU isn't available
                let cpuContext = null;
                if (!gpuContext) {
                    try {
                        cpuContext = await navigator.ml.createContext({ devicePreference: 'cpu' });
                        if (cpuContext) {
                            log('WebNN CPU backend is available', 'success');
                        }
                    } catch (e) {
                        log(`Failed to create WebNN CPU context: ${e.message}`);
                    }
                }
                
                const context = gpuContext || cpuContext;
                if (!context) {
                    supportElem.textContent = 'Failed to create context';
                    supportElem.className = 'error';
                    log('Failed to create any WebNN context', 'error');
                    return false;
                }
                
                supportElem.textContent = 'Supported';
                supportElem.className = 'success';
                backendElem.textContent = context.deviceType || (gpuContext ? 'GPU' : 'CPU');
                log(`WebNN context created with backend: ${context.deviceType || (gpuContext ? 'GPU' : 'CPU')}`, 'success');
                
                // Store context for later use
                window.webnnContext = context;
                return true;
            } catch (e) {
                supportElem.textContent = `Error: ${e.message}`;
                supportElem.className = 'error';
                log(`WebNN error: ${e.message}`, 'error');
                return false;
            }
        }
        
        // Run test with the given model
        async function runTest(modelId, bitPrecision, mixedPrecision) {
            const resultsElem = document.getElementById('results');
            const precisionElem = document.getElementById('precision-level');
            
            // Update precision display
            let precisionText = `${bitPrecision}-bit`;
            if (mixedPrecision) {
                precisionText += " mixed precision";
            }
            precisionElem.textContent = precisionText;
            
            resultsElem.textContent = `Loading model: ${modelId} with ${precisionText}...`;
            
            try {
                // Check WebNN support first
                const webnnSupported = await checkWebNN();
                if (!webnnSupported) {
                    resultsElem.textContent = 'WebNN is not supported in this browser';
                    return;
                }
                
                // Import transformers.js
                log('Loading transformers.js library...');
                const { pipeline, env } = await import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.0');
                log('transformers.js library loaded successfully', 'success');
                
                // Configure precision if specified
                if (bitPrecision !== 16) {
                    log(`Setting quantization to ${bitPrecision}-bit${mixedPrecision ? ' mixed precision' : ''}`, 'warning');
                    
                    // Set quantization configuration
                    env.USE_INT8 = bitPrecision <= 8;
                    env.USE_INT4 = bitPrecision <= 4;
                    env.USE_INT2 = bitPrecision <= 2;
                    
                    // Mixed precision settings
                    if (mixedPrecision) {
                        env.MIXED_PRECISION = true;
                        log('Mixed precision enabled for attention layers', 'warning');
                    }
                    
                    // Log current settings
                    log(`Quantization config: INT8=${env.USE_INT8}, INT4=${env.USE_INT4}, INT2=${env.USE_INT2}, MIXED=${env.MIXED_PRECISION}`, 'warning');
                    
                    // Estimate memory savings
                    const memoryReduction = {
                        8: '50%', // 16-bit → 8-bit
                        4: '75%', // 16-bit → 4-bit
                        2: '87.5%' // 16-bit → 2-bit
                    };
                    log(`Estimated memory reduction: ${memoryReduction[bitPrecision]}`, 'success');
                }
                
                // Initialize the model
                log(`Loading model: ${modelId}...`);
                const startTime = performance.now();
                
                // For text models, use feature-extraction
                const pipe = await pipeline('feature-extraction', modelId, {
                    backend: 'webnn',
                    quantized: bitPrecision < 16,
                    revision: 'default'
                });
                
                const loadTime = performance.now() - startTime;
                log(`Model loaded in ${loadTime.toFixed(2)}ms`, 'success');
                
                // Run inference
                log('Running inference...');
                const inferenceStart = performance.now();
                
                // Run multiple inferences to get better timing data
                const numRuns = 5;
                let totalTime = 0;
                let result;
                
                for (let i = 0; i < numRuns; i++) {
                    const runStart = performance.now();
                    result = await pipe('This is a test input for WebNN inference with quantization testing.');
                    const runTime = performance.now() - runStart;
                    totalTime += runTime;
                    log(`Run ${i+1}: ${runTime.toFixed(2)}ms`);
                }
                
                const averageInferenceTime = totalTime / numRuns;
                const inferenceTime = performance.now() - inferenceStart;
                
                log(`All inference runs completed in ${inferenceTime.toFixed(2)}ms (avg: ${averageInferenceTime.toFixed(2)}ms)`, 'success');
                
                // Get memory usage if possible
                let memoryUsage = null;
                try {
                    if (performance.memory) {
                        memoryUsage = {
                            totalJSHeapSize: performance.memory.totalJSHeapSize / (1024 * 1024),
                            usedJSHeapSize: performance.memory.usedJSHeapSize / (1024 * 1024),
                            jsHeapSizeLimit: performance.memory.jsHeapSizeLimit / (1024 * 1024)
                        };
                    }
                } catch (e) {
                    log(`Unable to get memory usage: ${e.message}`, 'warning');
                }
                
                // Prepare results
                const resultSummary = {
                    model: modelId,
                    webnn_supported: webnnSupported,
                    webnn_backend: document.getElementById('webnn-backend').textContent,
                    bit_precision: bitPrecision,
                    mixed_precision: mixedPrecision,
                    load_time_ms: loadTime,
                    inference_time_ms: inferenceTime,
                    average_inference_time_ms: averageInferenceTime,
                    output_shape: [result.data.length],
                    output_sample: Array.from(result.data.slice(0, 5)),
                    memory_usage: memoryUsage,
                    estimated_model_memory_mb: (result.data.length * 4 * (16 / bitPrecision)) / (1024 * 1024) // Rough estimate
                };
                
                // Display results
                resultsElem.textContent = JSON.stringify(resultSummary, null, 2);
                
                // Expose for selenium to retrieve
                window.testResults = resultSummary;
                
                log('Test completed successfully', 'success');
            } catch (e) {
                log(`Error: ${e.message}`, 'error');
                resultsElem.textContent = `Error: ${e.message}\n\n${e.stack}`;
                
                // Expose error for selenium
                window.testError = e.message;
            }
        }
        
        // Get the model ID from the URL
        const urlParams = new URLSearchParams(window.location.search);
        const modelId = urlParams.get('model') || 'bert-base-uncased';
        const bitPrecision = parseInt(urlParams.get('bits') || '16', 10);
        const mixedPrecision = urlParams.get('mixed') === 'true';
        
        // Run the test when the page loads
        window.addEventListener('DOMContentLoaded', () => {
            log(`Starting WebNN test with model: ${modelId} at ${bitPrecision}-bit${mixedPrecision ? ' mixed' : ''} precision`);
            runTest(modelId, bitPrecision, mixedPrecision);
        });
    </script>
</body>
</html>
"""

class WebNNSeleniumTester:
    """Class to run WebNN tests using Selenium."""
    
    def __init__(self, model="bert-base-uncased", browser="chrome", headless=True, 
                 bits=16, mixed_precision=False):
        """Initialize the tester.
        
        Args:
            model: Model ID to test
            browser: Browser to use (chrome, edge)
            headless: Whether to run in headless mode
            bits: Bit precision for quantization (16, 8, 4, 2)
            mixed_precision: Whether to use mixed precision
        """
        self.model = model
        self.browser = browser.lower()
        self.headless = headless
        self.bits = bits
        self.mixed_precision = mixed_precision
        self.driver = None
        self.html_path = None
    
    def setup(self):
        """Set up the test environment."""
        if not HAS_SELENIUM:
            logger.error("Selenium is required for this test")
            return False
        
        # Create HTML file
        try:
            fd, self.html_path = tempfile.mkstemp(suffix=".html")
            with os.fdopen(fd, 'w') as f:
                f.write(WEBNN_TEST_HTML)
        except Exception as e:
            logger.error(f"Failed to create HTML file: {e}")
            return False
        
        logger.info(f"Created test HTML at: {self.html_path}")
        
        # Initialize browser
        try:
            if self.browser == "chrome":
                options = ChromeOptions()
                if self.headless:
                    options.add_argument("--headless=new")
                options.add_argument("--disable-gpu")
                options.add_argument("--no-sandbox")
                options.add_argument("--disable-dev-shm-usage")
                options.add_argument("--enable-features=WebNN")
                self.driver = webdriver.Chrome(options=options)
            elif self.browser == "edge":
                options = EdgeOptions()
                if self.headless:
                    options.add_argument("--headless=new")
                options.add_argument("--disable-gpu")
                options.add_argument("--no-sandbox")
                options.add_argument("--disable-dev-shm-usage")
                options.add_argument("--enable-features=WebNN")
                self.driver = webdriver.Edge(options=options)
            else:
                logger.error(f"Unsupported browser: {self.browser}")
                return False
            
            logger.info(f"Browser initialized: {self.browser}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize browser: {e}")
            return False
    
    def run_test(self):
        """Run the WebNN test."""
        if not self.driver or not self.html_path:
            logger.error("Setup not completed")
            return None
        
        try:
            # Load the test page with model parameter
            url = f"file://{self.html_path}?model={self.model}&bits={self.bits}&mixed={str(self.mixed_precision).lower()}"
            logger.info(f"Loading test page: {url}")
            self.driver.get(url)
            
            # Wait for the test to complete (results element to be populated)
            try:
                WebDriverWait(self.driver, 180).until(
                    lambda d: d.execute_script("return window.testResults || window.testError")
                )
            except Exception as e:
                logger.error(f"Timeout waiting for test to complete: {e}")
                return None
            
            # Get test results
            test_results = self.driver.execute_script("return window.testResults")
            test_error = self.driver.execute_script("return window.testError")
            
            if test_error:
                logger.error(f"Test failed: {test_error}")
                return {"error": test_error}
            
            if test_results:
                logger.info("Test completed successfully")
                return test_results
            
            logger.error("No test results found")
            return None
        except Exception as e:
            logger.error(f"Error running test: {e}")
            return None
        finally:
            # Take screenshot if not headless
            if not self.headless:
                try:
                    screenshot_path = f"webnn_test_{self.browser}_{self.bits}bit_{time.time()}.png"
                    self.driver.save_screenshot(screenshot_path)
                    logger.info(f"Screenshot saved to: {screenshot_path}")
                except Exception as e:
                    logger.error(f"Failed to save screenshot: {e}")
    
    def cleanup(self):
        """Clean up resources."""
        if self.driver:
            try:
                self.driver.quit()
                logger.info("Browser closed")
            except Exception as e:
                logger.error(f"Error closing browser: {e}")
        
        if self.html_path and os.path.exists(self.html_path):
            try:
                os.unlink(self.html_path)
                logger.info("Test HTML file removed")
            except Exception as e:
                logger.error(f"Error removing HTML file: {e}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run WebNN inference test with Selenium")
    parser.add_argument("--model", default="bert-base-uncased", help="Model ID to test")
    parser.add_argument("--browser", default="chrome", choices=["chrome", "edge"], help="Browser to use")
    parser.add_argument("--no-headless", action="store_true", help="Run in non-headless mode")
    parser.add_argument("--bits", type=int, default=16, choices=[16, 8, 4, 2], 
                        help="Bit precision for quantization")
    parser.add_argument("--mixed-precision", action="store_true", 
                        help="Use mixed precision (higher precision for attention layers)")
    parser.add_argument("--output-json", help="Path to save results as JSON")
    args = parser.parse_args()
    
    logger.info(f"Starting WebNN test with model: {args.model} on {args.browser} at {args.bits}-bit precision"
                f"{' with mixed precision' if args.mixed_precision else ''}")
    
    # Create and run tester
    tester = WebNNSeleniumTester(
        model=args.model,
        browser=args.browser,
        headless=not args.no_headless,
        bits=args.bits,
        mixed_precision=args.mixed_precision
    )
    
    try:
        if not tester.setup():
            logger.error("Failed to set up test environment")
            return 1
        
        results = tester.run_test()
        if not results:
            logger.error("Test failed to complete")
            return 1
        
        if "error" in results:
            logger.error(f"Test failed: {results['error']}")
            return 1
        
        # Save results to JSON if requested
        if args.output_json:
            try:
                with open(args.output_json, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Results saved to: {args.output_json}")
            except Exception as e:
                logger.error(f"Failed to save results to JSON: {e}")
        
        # Print results summary
        print("\n======= WebNN Test Results Summary =======")
        print(f"Model: {results['model']}")
        print(f"Browser: {args.browser}")
        print(f"WebNN Backend: {results['webnn_backend']}")
        print(f"Precision: {results['bit_precision']}-bit{' mixed' if results['mixed_precision'] else ''}")
        print(f"Load Time: {results['load_time_ms']:.2f}ms")
        print(f"Average Inference Time: {results.get('average_inference_time_ms', results['inference_time_ms']):.2f}ms")
        print(f"Output Shape: {results['output_shape']}")
        
        # Print memory usage if available
        if results.get('memory_usage'):
            print("\nMemory Usage:")
            print(f"  Used JS Heap: {results['memory_usage']['usedJSHeapSize']:.2f}MB")
            print(f"  Total JS Heap: {results['memory_usage']['totalJSHeapSize']:.2f}MB")
            print(f"  JS Heap Limit: {results['memory_usage']['jsHeapSizeLimit']:.2f}MB")
        
        print(f"Estimated Model Memory: {results.get('estimated_model_memory_mb', 'N/A')}")
        print("=========================================")
        
        return 0
    finally:
        tester.cleanup()

if __name__ == "__main__":
    sys.exit(main())