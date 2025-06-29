#!/usr/bin/env python3
"""
Simple WebGPU Test

A simplified test for WebGPU implementation using Selenium and browser automation.
"""

import os
import sys
import time
import json
import asyncio
import logging
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Simple HTML to test WebGPU support
WEBGPU_TEST_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>WebGPU Test</title>
</head>
<body>
    <h1>WebGPU Test</h1>
    <div id="status">Checking WebGPU support...</div>
    
    <script>
        async function checkWebGPU() {
            const statusElement = document.getElementById('status');
            
            try {
                if (!navigator.gpu) {
                    statusElement.textContent = 'WebGPU is not supported in this browser';
                    return false;
                }
                
                const adapter = await navigator.gpu.requestAdapter();
                if (!adapter) {
                    statusElement.textContent = 'WebGPU adapter not available';
                    return false;
                }
                
                // Get adapter info
                let adapterInfo;
                try {
                    adapterInfo = await adapter.requestAdapterInfo();
                } catch (e) {
                    adapterInfo = { description: 'Info not available' };
                }
                
                // Request device
                const device = await adapter.requestDevice();
                if (!device) {
                    statusElement.textContent = 'WebGPU device not available';
                    return false;
                }
                
                // Success!
                statusElement.textContent = `WebGPU is available! Adapter: ${adapterInfo.description || adapterInfo.vendor || 'Unknown'}`;
                
                // Store the result as a global variable so we can access it from Selenium
                window.webgpuTestResult = {
                    supported: true,
                    adapter: adapterInfo,
                    timestamp: new Date().toISOString()
                };
                
                return true;
            } catch (error) {
                statusElement.textContent = `WebGPU error: ${error.message}`;
                
                // Store the error as a global variable
                window.webgpuTestResult = {
                    supported: false,
                    error: error.message,
                    timestamp: new Date().toISOString()
                };
                
                return false;
            }
        }
        
        // Run the test
        checkWebGPU().then(result => {
            console.log('WebGPU test result:', result);
        });
    </script>
</body>
</html>
"""

async def test_webgpu():
    """Test WebGPU support in Chrome."""
    # Create a temporary HTML file
    html_path = os.path.join(os.getcwd(), "webgpu_test.html")
    with open(html_path, "w") as f:
        f.write(WEBGPU_TEST_HTML)
    
    logger.info(f"Created test HTML file: {html_path}")
    
    # Set up Chrome
    try:
        logger.info("Setting up Chrome with WebGPU support")
        
        # Set up Chrome options
        options = ChromeOptions()
        options.add_argument("--enable-features=WebGPU")
        options.add_argument("--disable-gpu-sandbox")
        options.add_argument("--no-sandbox")
        
        # Set up Chrome service
        service = ChromeService(ChromeDriverManager().install())
        
        # Start Chrome
        driver = webdriver.Chrome(service=service, options=options)
        
        # Load the test page
        file_url = f"file://{html_path}"
        logger.info(f"Loading test page: {file_url}")
        driver.get(file_url)
        
        # Wait for the test to complete
        time.sleep(3)
        
        # Get the status element
        status_element = driver.find_element(By.ID, "status")
        status_text = status_element.text
        
        logger.info(f"Status: {status_text}")
        
        # Get the test result from JavaScript
        js_result = driver.execute_script("return window.webgpuTestResult")
        
        if js_result:
            logger.info(f"WebGPU test result: {json.dumps(js_result, indent=2)}")
            
            if js_result.get("supported", False):
                logger.info("WebGPU is supported!")
            else:
                logger.warning("WebGPU is not supported")
        else:
            logger.warning("No WebGPU test result found")
        
        # Close the browser
        driver.quit()
        
        # Clean up
        os.remove(html_path)
        
        # Return success based on test result
        return js_result and js_result.get("supported", False)
        
    except Exception as e:
        logger.error(f"Error testing WebGPU: {e}")
        
        # Clean up
        if os.path.exists(html_path):
            os.remove(html_path)
            
        return False

def main():
    """Main function."""
    logger.info("Testing WebGPU support in Chrome")
    
    # Run the test
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(test_webgpu())
    
    if result:
        logger.info("WebGPU test passed")
        return 0
    else:
        logger.error("WebGPU test failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())