"""
Test for cross-browser model sharding in IPFS Accelerate.

This test verifies that large models can be sharded across multiple browsers,
with different browsers handling different components based on their capabilities.
"""

import os
import pytest
import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple

# Import common utilities
from common.hardware_detection import detect_hardware

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.firefox.options import Options as FirefoxOptions
    from selenium.webdriver.edge.options import Options as EdgeOptions
    from selenium.webdriver.common.by import By
except ImportError:
    pass

# Integration test fixtures
@pytest.fixture
def browsers():
    """Set up multiple browsers for cross-browser testing."""
    browser_types = os.environ.get("TEST_BROWSERS", "chrome,firefox").split(",")
    
    # For each browser type, decide if we need to skip the test
    available_browsers = []
    for browser in browser_types:
        if browser not in ["chrome", "firefox", "edge"]:
            logging.warning(f"Unsupported browser type: {browser}")
            continue
        
        # Check if browser is installed
        try:
            if browser == "chrome":
                options = ChromeOptions()
                options.add_argument('--headless')
                options.add_argument('--no-sandbox')
                options.add_argument('--disable-dev-shm-usage')
                options.add_argument('--enable-features=Vulkan,WebGPU')
                driver = webdriver.Chrome(options=options)
                available_browsers.append(("chrome", driver))
            elif browser == "firefox":
                options = FirefoxOptions()
                options.add_argument('-headless')
                driver = webdriver.Firefox(options=options)
                available_browsers.append(("firefox", driver))
            elif browser == "edge":
                options = EdgeOptions()
                options.add_argument('--headless')
                driver = webdriver.Edge(options=options)
                available_browsers.append(("edge", driver))
        except Exception as e:
            logging.warning(f"Failed to initialize {browser}: {e}")
    
    # Skip test if we don't have at least 2 browsers
    if len(available_browsers) < 2:
        pytest.skip(f"At least 2 browsers are needed. Available: {len(available_browsers)}")
    
    yield available_browsers
    
    # Close all browsers
    for _, driver in available_browsers:
        driver.quit()

@pytest.fixture
def model_sharding_page(temp_dir):
    """Create a test page for cross-browser model sharding."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Cross-Browser Model Sharding</title>
        <script>
            // Browser-specific configuration will be passed via browserType
            const browserType = new URLSearchParams(window.location.search).get('browser') || 'unknown';
            
            // Mock model sharding - would be more complex in real implementation
            async function initializeModelShard() {
                const resultElement = document.getElementById('result');
                const statusElement = document.getElementById('status');
                
                statusElement.textContent = `Initializing ${browserType} for model sharding...`;
                
                try {
                    // Detect browser capabilities
                    const capabilities = {
                        webgpu: !!navigator.gpu,
                        webnn: !!(navigator.ml || window.ml),
                        webgl: !!window.WebGLRenderingContext && !!document.createElement('canvas').getContext('webgl')
                    };
                    
                    // Display detected capabilities
                    document.getElementById('capabilities').textContent = 
                        `Capabilities: WebGPU=${capabilities.webgpu}, WebNN=${capabilities.webnn}, WebGL=${capabilities.webgl}`;
                    
                    // Simulate shard initialization based on browser type
                    let shardConfig;
                    if (browserType === 'chrome') {
                        shardConfig = {
                            shardId: 'attention_heads',
                            shardType: 'attention',
                            layerRange: [0, 6],  // First half of layers
                            preferredBackend: capabilities.webgpu ? 'webgpu' : 'webgl'
                        };
                    } else if (browserType === 'firefox') {
                        shardConfig = {
                            shardId: 'feed_forward',
                            shardType: 'feedforward',
                            layerRange: [0, 6],  // First half of layers
                            preferredBackend: 'webgl'
                        };
                    } else if (browserType === 'edge') {
                        shardConfig = {
                            shardId: 'attention_heads',
                            shardType: 'attention',
                            layerRange: [6, 12],  // Second half of layers
                            preferredBackend: capabilities.webnn ? 'webnn' : 'webgl'
                        };
                    } else {
                        shardConfig = {
                            shardId: 'feed_forward',
                            shardType: 'feedforward',
                            layerRange: [6, 12],  // Second half of layers
                            preferredBackend: 'webgl'
                        };
                    }
                    
                    // Display shard configuration
                    document.getElementById('shard-config').textContent = 
                        `Shard Config: ${JSON.stringify(shardConfig)}`;
                    
                    // Simulate model initialization time
                    await new Promise(resolve => setTimeout(resolve, 1000));
                    
                    // Simulate loading the model shard
                    statusElement.textContent = `Loading model shard...`;
                    await new Promise(resolve => setTimeout(resolve, 1500));
                    
                    // Simulate inference
                    statusElement.textContent = `Running inference on shard...`;
                    await new Promise(resolve => setTimeout(resolve, 500));
                    
                    // Report success
                    resultElement.textContent = 'Shard Ready';
                    statusElement.textContent = `${browserType} shard initialized successfully`;
                    
                    // Store shard info in a way that can be accessed from outside
                    window.shardInfo = {
                        browserType,
                        capabilities,
                        shardConfig,
                        status: 'ready',
                        timestamp: Date.now()
                    };
                    
                    // Return shard info as JSON for easier retrieval
                    document.getElementById('shard-info').textContent = JSON.stringify(window.shardInfo);
                    
                } catch (error) {
                    resultElement.textContent = 'Shard Error';
                    statusElement.textContent = `Error: ${error.message}`;
                    
                    window.shardInfo = {
                        browserType,
                        status: 'error',
                        error: error.message,
                        timestamp: Date.now()
                    };
                    
                    document.getElementById('shard-info').textContent = JSON.stringify(window.shardInfo);
                }
            }
            
            window.onload = initializeModelShard;
        </script>
    </head>
    <body>
        <h1>Cross-Browser Model Sharding</h1>
        <div id="result">Initializing...</div>
        <div id="status">Please wait...</div>
        <div id="capabilities">Detecting capabilities...</div>
        <div id="shard-config">Loading configuration...</div>
        <pre id="shard-info"></pre>
    </body>
    </html>
    """
    
    file_path = os.path.join(temp_dir, 'model_sharding_test.html')
    with open(file_path, 'w') as f:
        f.write(html_content)
    
    return file_path

@pytest.mark.integration
@pytest.mark.browser
class TestCrossBrowserModelSharding:
    """
    Tests for cross-browser model sharding.
    
    These tests verify that models can be sharded across multiple
    browsers, with each browser handling different components.
    """
    
    def test_browser_availability(self, browsers):
        """Test that multiple browsers are available."""
        assert len(browsers) >= 2, f"At least 2 browsers are required, found {len(browsers)}"
        browser_types = [t for t, _ in browsers]
        logging.info(f"Available browsers: {browser_types}")
    
    def test_browser_capabilities(self, browsers):
        """Test browser capabilities detection."""
        for browser_type, driver in browsers:
            # Navigate to about:blank
            driver.get("about:blank")
            
            # Inject and execute JavaScript to detect capabilities
            script = """
            return {
                browser: navigator.userAgent,
                webgpu: !!navigator.gpu,
                webnn: !!(navigator.ml || window.ml),
                webgl: !!window.WebGLRenderingContext
            }
            """
            capabilities = driver.execute_script(script)
            logging.info(f"Browser {browser_type} capabilities: {capabilities}")
            
            # At least one acceleration technology should be available
            assert capabilities["webgl"] or capabilities["webgpu"] or capabilities["webnn"], \
                f"Browser {browser_type} does not support any acceleration technology"
    
    def test_basic_model_sharding(self, browsers, model_sharding_page):
        """Test basic model sharding across browsers."""
        shard_results = []
        
        # Load test page in each browser with appropriate query parameter
        for browser_type, driver in browsers:
            url = f"file://{model_sharding_page}?browser={browser_type}"
            driver.get(url)
            
            # Wait for sharding to complete
            time.sleep(3)
            
            # Check result status
            result_element = driver.find_element(By.ID, 'result')
            status_element = driver.find_element(By.ID, 'status')
            
            assert result_element.text == 'Shard Ready', f"Shard not ready in {browser_type}: {result_element.text}"
            assert f"{browser_type} shard initialized successfully" in status_element.text, \
                f"Unexpected status in {browser_type}: {status_element.text}"
            
            # Get shard information
            shard_info_element = driver.find_element(By.ID, 'shard-info')
            try:
                shard_info = json.loads(shard_info_element.text)
                shard_info['actual_browser_type'] = browser_type
                shard_results.append(shard_info)
            except json.JSONDecodeError:
                pytest.fail(f"Invalid shard info in {browser_type}: {shard_info_element.text}")
        
        # Verify we have enough shards
        assert len(shard_results) >= 2, f"At least 2 shards are required, found {len(shard_results)}"
        
        # Verify shard assignments are complementary
        shard_types = [s['shardConfig']['shardType'] for s in shard_results]
        assert 'attention' in shard_types, "No attention shard found"
        assert 'feedforward' in shard_types, "No feedforward shard found"
        
        # Verify layer coverage - we need full coverage of layers 0-12
        covered_layers = set()
        for shard in shard_results:
            layer_range = shard['shardConfig']['layerRange']
            for layer in range(layer_range[0], layer_range[1]):
                covered_layers.add(layer)
        
        assert len(covered_layers) == 12, f"Incomplete layer coverage: {sorted(covered_layers)}"
        
        # Log shard allocation for debugging
        for shard in shard_results:
            browser = shard['actual_browser_type']
            shard_type = shard['shardConfig']['shardType']
            layers = shard['shardConfig']['layerRange']
            backend = shard['shardConfig']['preferredBackend']
            logging.info(f"Browser {browser} assigned {shard_type} shard for layers {layers} using {backend}")
    
    @pytest.mark.skip(reason="Advanced test requiring real model inference")
    def test_integrated_model_inference(self, browsers, model_sharding_page):
        """
        Test integrated model inference across browsers.
        
        This test simulates full model inference by coordinating multiple browser shards.
        Skipped by default as it requires real model inference.
        """
        # This would be implemented in a real-world scenario
        # It would coordinate multiple browser instances to perform
        # inference on different parts of a model and combine the results
        pass